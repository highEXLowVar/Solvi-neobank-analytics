"""The ``calculate_uplift_or_stats`` tool: vetted estimators over the marts.

Some numbers can't honestly be produced by one SQL query (a changepoint scan,
a geometric churn fit, LTV extrapolated past the observation window), and an
LLM improvising them in SQL is where made-up figures come from. So the agent
gets the same estimators the analysis layer uses, computed live from the
warehouse it's querying. They mirror analysis/kyc_incident.py,
analysis/retention_economics.py and analysis/experiment_eval.py, and
evals/test_stats.py checks they reproduce analysis/out/*.json exactly.

Nothing here touches simulator config or data/ground_truth/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

from .warehouse import Warehouse

LLR_THRESHOLD = 25.0
MIN_SIDE_DAYS = 21
MIN_COHORT = 200
MAX_K = 24
PROJECT_TO = 36
BONUS_EUR = 10.0
SEGMENTS = ("channel", "country", "device", "age_band")


def two_prop(k1, n1, k0, n0) -> dict:
    p1, p0 = k1 / n1, k0 / n0
    se_un = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    pp = (k1 + k0) / (n1 + n0)
    se_pool = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
    z = (p1 - p0) / se_pool if se_pool > 0 else 0.0
    return {
        "rate_a": round(p1, 4), "rate_b": round(p0, 4),
        "delta_pp": round((p1 - p0) * 100, 2),
        "ci95_pp": [round((p1 - p0 - 1.96 * se_un) * 100, 2), round((p1 - p0 + 1.96 * se_un) * 100, 2)],
        "z": round(float(z), 2),
        "p_value": float(f"{2 * st.norm.sf(abs(z)):.3g}"),
    }


def _llr_scan(k: np.ndarray, n: np.ndarray) -> tuple[int, float]:
    def ll(k_, n_):
        if n_ == 0 or k_ == 0 or k_ == n_:
            return 0.0
        p = k_ / n_
        return k_ * np.log(p) + (n_ - k_) * np.log(1 - p)

    K, N = k.cumsum(), n.cumsum()
    best_tau, best_llr = -1, 0.0
    ll0 = ll(K[-1], N[-1])
    for tau in range(MIN_SIDE_DAYS, len(k) - MIN_SIDE_DAYS):
        llr = ll(K[tau - 1], N[tau - 1]) + ll(K[-1] - K[tau - 1], N[-1] - N[tau - 1]) - ll0
        if llr > best_llr:
            best_tau, best_llr = tau, llr
    return best_tau, best_llr


def changepoint(wh: Warehouse, platform: str | None = None, doc_type: str | None = None) -> dict:
    """binomial LLR changepoint scan on daily KYC decisions for one slice.
    a break only counts when LLR > 25 (about z > 7); then the post-break
    series is re-scanned once for a recovery."""
    where = []
    if platform:
        where.append(f"platform = '{platform}'")
    if doc_type:
        where.append(f"doc_type = '{doc_type}'")
    clause = ("where " + " and ".join(where)) if where else ""
    g = wh.con.sql(f"""
        select decision_date, sum(n_decisions) n, sum(n_approved) k
        from agg_kyc_daily {clause} group by 1 order by 1
    """).df()
    if g.empty:
        return {"error": "no KYC decisions for that slice"}
    g["decision_date"] = pd.to_datetime(g["decision_date"])
    g = g.set_index("decision_date").asfreq("D", fill_value=0).reset_index()
    k, n = g["k"].to_numpy(float), g["n"].to_numpy(float)
    out = {"slice": {"platform": platform or "all", "doc_type": doc_type or "all"},
           "days_scanned": len(g), "overall_rate": round(k.sum() / n.sum(), 4),
           "llr_threshold": LLR_THRESHOLD}
    tau, llr = _llr_scan(k, n)
    out["max_llr"] = round(llr, 1)
    if llr <= LLR_THRESHOLD:
        out["break_detected"] = False
        return out
    out["break_detected"] = True
    out["break_date"] = str(g["decision_date"].iloc[tau].date())
    out["level_before"] = round(k[:tau].sum() / n[:tau].sum(), 3)
    k2, n2 = k[tau:], n[tau:]
    tau2, llr2 = _llr_scan(k2, n2)
    if llr2 > LLR_THRESHOLD:
        out["recovery_date"] = str(g["decision_date"].iloc[tau + tau2].date())
        out["level_during"] = round(k2[:tau2].sum() / n2[:tau2].sum(), 3)
        out["level_after"] = round(k2[tau2:].sum() / n2[tau2:].sum(), 3)
    else:
        out["level_during"] = round(k2.sum() / n2.sum(), 3)
    if platform:
        rel = wh.con.sql(f"""
            select version, release_date from stg_app_releases
            where platform = '{platform}' and release_date <= date '{out['break_date']}'
            order by release_date desc limit 1
        """).fetchone()
        if rel:
            out["closest_release_before_break"] = {"version": f"{rel[0]:.2f}".rstrip("0").rstrip("."),
                                                   "release_date": str(rel[1])}
    return out


def channel_unit_economics(wh: Warehouse, channel: str | None = None) -> dict:
    """CAC per activated user, 36-month LTV (observed margin + geometric tail),
    payback month and implied monthly churn, per acquisition channel."""
    ret = wh.con.sql(f"""
        select channel, months_since_activation as k,
               sum(n_active) as active, sum(cohort_size) as total
        from agg_retention_cohorts
        where cohort_size >= {MIN_COHORT} and months_since_activation <= {MAX_K}
        group by 1, 2 order by 1, 2
    """).df()
    act = dict(wh.con.sql("select channel, avg(is_activated::int) from dim_users group by 1").fetchall())
    econ = wh.con.sql(f"""
        select channel, months_since_activation as k, avg_margin_eur, cac_eur
        from agg_unit_economics where months_since_activation <= {MAX_K}
        order by channel, k
    """).df()
    out = {}
    for ch, g in econ.groupby("channel"):
        if channel and ch != channel:
            continue
        g = g.sort_values("k")
        m, kk = g["avg_margin_eur"].to_numpy(), g["k"].to_numpy()
        cac_signup = float(g["cac_eur"].iloc[0])
        cac = cac_signup / act[ch]
        cum = m.cumsum()
        mask = (kk >= 6) & (m > 0)
        slope, intercept = np.polyfit(kk[mask], np.log(m[mask]), 1)
        proj_k = np.arange(kk.max() + 1, PROJECT_TO + 1)
        full = np.concatenate([cum, cum[-1] + (np.exp(intercept) * np.exp(slope) ** proj_k).cumsum()])
        r = ret[ret["channel"] == ch]
        rk, rr = r["k"].to_numpy(), (r["active"] / r["total"]).round(4).to_numpy()
        rm = (rk >= 3) & (rr > 0)
        s = float(np.exp(np.polyfit(rk[rm], np.log(rr[rm]), 1)[0]))
        out[ch] = {
            "cac_per_signup_eur": cac_signup,
            "activation_rate": round(act[ch], 3),
            "cac_per_activated_eur": round(cac, 2),
            "ltv_36m_eur": round(float(full[PROJECT_TO]), 2),
            "ltv_minus_cac_eur": round(float(full[PROJECT_TO]) - cac, 2),
            "payback_month": next((int(i) for i, v in enumerate(full) if v >= cac), None),
            "implied_monthly_churn": round(1 - s, 4),
            "observed_margin_months": int(kk.max()) + 1,
        }
    if channel and not out:
        return {"error": f"unknown channel {channel!r}"}
    return {"by_channel": out,
            "method": "LTV = observed avg margin per activated user by tenure, geometric tail fit "
                      "from month 6 projected to month 36; churn = 1 - geometric decay of "
                      "cohort retention, months 3-24, cohorts >= 200 users"}


def ab_test(wh: Warehouse, segment_by: str | None = None) -> dict:
    """TOPUP10 read-out: conversion lift with a two-proportion z-test, then the
    money: bonus cost per treated user, cost per *incremental* conversion, and
    the 60-day margin difference net of the bonus."""
    if segment_by and segment_by not in SEGMENTS:
        return {"error": f"segment_by must be one of {SEGMENTS}"}
    df = wh.con.sql("select * from fct_experiment_users").df()
    if df.empty:
        return {"error": "no experiment subjects in range"}
    ltv = {}
    if segment_by == "channel":
        ltv = {ch: e["ltv_36m_eur"] for ch, e in channel_unit_economics(wh)["by_channel"].items()}

    def readout(d: pd.DataFrame) -> dict:
        t, c = d[d["variant"] == "treatment"], d[d["variant"] == "control"]
        res = two_prop(t["conv_14d"].sum(), len(t), c["conv_14d"].sum(), len(c))
        res = {"n_treatment": len(t), "n_control": len(c),
               "conv_treatment": res.pop("rate_a"), "conv_control": res.pop("rate_b"), **res}
        redemption = t["redeemed"].mean()
        cost_treated = BONUS_EUR * redemption
        d_conv = res["delta_pp"] / 100
        res.update({
            "bonus_redemption_rate": round(redemption, 3),
            "bonus_cost_per_treated_user_eur": round(cost_treated, 2),
            "bonus_cost_per_incremental_conversion_eur": round(cost_treated / d_conv, 1) if d_conv > 0 else None,
            "margin_60d_treatment_eur": round(t["margin_60d_eur"].mean(), 3),
            "margin_60d_control_eur": round(c["margin_60d_eur"].mean(), 3),
            "margin_60d_delta_net_of_bonus_eur": round(t["margin_60d_eur"].mean() - c["margin_60d_eur"].mean(), 3),
            "margin_60d_delta_excl_bonus_eur": round(
                (t["margin_60d_eur"] + t["bonus_cost_eur"]).mean() - c["margin_60d_eur"].mean(), 3),
        })
        return res

    out = {"experiment_id": df["experiment_id"].iloc[0], "overall": readout(df)}
    if segment_by:
        segs = {}
        for v, g in df.groupby(segment_by):
            segs[str(v)] = readout(g)
            if v in ltv:
                segs[str(v)]["channel_avg_ltv_36m_eur"] = ltv[v]
        out["by_" + segment_by] = segs
    return out


def calculate_uplift_or_stats(wh: Warehouse, metric_slice: dict) -> dict:
    kind = metric_slice.get("analysis")
    try:
        if kind == "kyc_changepoint":
            return changepoint(wh, metric_slice.get("platform"), metric_slice.get("doc_type"))
        if kind == "channel_unit_economics":
            return channel_unit_economics(wh, metric_slice.get("channel"))
        if kind == "ab_test_readout":
            return ab_test(wh, metric_slice.get("segment_by"))
        if kind == "two_proportion_test":
            r = two_prop(*(float(metric_slice[x]) for x in ("k_a", "n_a", "k_b", "n_b")))
            return r
    except (KeyError, ValueError, ZeroDivisionError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"error": "analysis must be one of kyc_changepoint, channel_unit_economics, "
                     "ab_test_readout, two_proportion_test"}
