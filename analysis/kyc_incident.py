"""KYC funnel health: changepoint detection and incident sizing.

Method (bear with me)
----------------------
For every (platform x doc_type) segment we scan all candidate change dates and
compute the binomial log-likelihood-ratio of a one-break model vs no break
(binary segmentation, nothing fancier). A break gets declared when the LLR
clears a conservative threshold, then we re-scan the post-break series for a
second break (recovery). Detected dates get cross-referenced against the app
release calendar, and the cumulative damage is sized against the segment's
own pre-break baseline - not some global average, that would be daft.

Run:  python -m analysis.kyc_incident
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import connect, save_json, wilson_ci

LLR_THRESHOLD = 25.0  # ~ z > 7. we ONLY want unmistakable breaks
MIN_SIDE_DAYS = 21


def _llr_scan(k: np.ndarray, n: np.ndarray) -> tuple[int, float]:
    """best single changepoint by binomial log-likelihood ratio.

    returns (index of first day of the new regime, LLR). its vectorised over
    all candidate splits so it wont be dog slow on the full date range.
    """

    def ll(k_, n_):
        k_, n_ = float(k_), float(n_)
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


def main() -> None:
    con = connect()
    daily = con.sql(
        """
        select decision_date, platform, doc_type, n_decisions, n_approved
        from agg_kyc_daily order by decision_date
        """
    ).df()
    releases = con.sql("select * from stg_app_releases order by release_date").df()

    segments, incident = [], None
    for (platform, doc), g in daily.groupby(["platform", "doc_type"]):
        g = (
            g.set_index("decision_date")[["n_decisions", "n_approved"]]
            .asfreq("D", fill_value=0)
            .reset_index()
        )
        k, n = g["n_approved"].to_numpy(), g["n_decisions"].to_numpy()
        tau, llr = _llr_scan(k, n)
        seg = {"platform": platform, "doc_type": doc, "llr": round(llr, 1)}
        if llr > LLR_THRESHOLD:
            break_date = g["decision_date"].iloc[tau]
            # rescan after the break for a recovery point
            k2, n2 = k[tau:], n[tau:]
            tau2, llr2 = _llr_scan(k2, n2)
            seg["break_date"] = str(break_date.date())
            pre_k, pre_n = k[:tau].sum(), n[:tau].sum()
            seg["level_before"] = round(pre_k / pre_n, 3)
            if llr2 > LLR_THRESHOLD:
                rec_date = g["decision_date"].iloc[tau + tau2]
                seg["recovery_date"] = str(rec_date.date())
                seg["level_during"] = round(k[tau : tau + tau2].sum() / n[tau : tau + tau2].sum(), 3)
                seg["level_after"] = round(k2[tau2:].sum() / n2[tau2:].sum(), 3)
            else:
                seg["level_during"] = round(k2.sum() / n2.sum(), 3)
            incident = {**seg, "g": g, "tau": tau, "tau2": tau2 if llr2 > LLR_THRESHOLD else None}
        segments.append(seg)

    assert incident is not None, "no segment cleared the changepoint threshold"
    g, tau = incident.pop("g"), incident.pop("tau")
    tau2 = incident.pop("tau2")

    # !!!! damage: expected approvals lost vs the segment's own baseline 
    # (comparing to the GLOBAL average here would be wrong, learned that one the hard way)
    p0 = incident["level_before"]
    n_arr, k_arr = g["n_decisions"].to_numpy(), g["n_approved"].to_numpy()
    lost_users = float(np.maximum(p0 * n_arr[tau:] - k_arr[tau:], 0).sum())

    # downstream conversion of approved users (segment-blended, from the warehouse)
    downstream = con.sql(
        """
        select avg(has_topup::int) * avg(case when has_topup then is_activated::int end)
        from dim_users
        where is_kyc_approved and device = 'android' and doc_type = 'national_id'
        """
    ).fetchone()[0]
    # 12month margin per activated user: per tenure averages summed, blended
    #across channels (each tenure month uses its own observable denminator)
    margin12 = con.sql(
        """
        select sum(margin_k) from (
            select months_since_activation,
                   sum(margin_sum_eur) / sum(n_observable) as margin_k
            from agg_unit_economics
            where months_since_activation between 0 and 11
            group by 1
        )
        """
    ).fetchone()[0]
    lost_activations = lost_users * downstream
    lost_margin = lost_activations * margin12

    #release attribution 
    android = releases[releases["platform"] == "android"].copy()
    android["gap_days"] = (
        pd.to_datetime(incident["break_date"]) - pd.to_datetime(android["release_date"])
    ).dt.days
    closest = android[android["gap_days"] >= 0].sort_values("gap_days").iloc[0]
    recovery_release = None
    if incident.get("recovery_date"):
        android["gap2"] = (
            pd.to_datetime(incident["recovery_date"]) - pd.to_datetime(android["release_date"])
        ).dt.days
        recovery_release = android[android["gap2"] >= 0].sort_values("gap2").iloc[0]

    #rejection-reason fingerprint
    #finish this
    reasons = con.sql(
        f"""
        select case when month < date '{incident['break_date']}' then 'before' else 'after' end as period,
               reject_reason, sum(n_rejections) as n
        from agg_kyc_reject_reasons_monthly
        where platform = 'android' and doc_type = 'national_id'
        group by 1, 2 order by 1, 3 desc
        """
    ).df()
    reason_mix = {
        period: {r.reject_reason: int(r.n) for r in grp.itertuples()}
        for period, grp in reasons.groupby("period")
    }

    #weekly series for  dashboard 
    weekly = con.sql(
        """
        select date_trunc('week', decision_date) as week,
               case when platform = 'android' and doc_type = 'national_id'
                    then 'android_national_id' else 'all_other' end as segment,
               sum(n_decisions) as n, sum(n_approved) as k
        from agg_kyc_daily
        -- decisions trail past the extract date for late submitters, so we drop
        -- the incomplete trailing week here - otherwise the chart just tails off
        -- into noise and someone asks why the line goes wobbly at the end
        where decision_date <= date '2026-05-31'
        group by 1, 2 order by 1
        """
    ).df()
    series = {}
    for seg_name, grp in weekly.groupby("segment"):
        lo_hi = [wilson_ci(int(r.k), int(r.n)) for r in grp.itertuples()]
        series[seg_name] = {
            "week": [str(w.date()) for w in grp["week"]],
            "rate": (grp["k"] / grp["n"]).round(4).tolist(),
            "lo": [round(x[0], 4) for x in lo_hi],
            "hi": [round(x[1], 4) for x in lo_hi],
            "n": grp["n"].astype(int).tolist(),
        }

    payload = {
        "scan": sorted(segments, key=lambda s: -s["llr"]),
        "incident": incident,
        "attribution": {
            "suspect_release": {
                "version": closest["version"],
                "release_date": str(pd.to_datetime(closest["release_date"]).date()),
                "days_before_break": int(closest["gap_days"]),
            },
            "recovery_release": None
            if recovery_release is None
            else {
                "version": recovery_release["version"],
                "release_date": str(pd.to_datetime(recovery_release["release_date"]).date()),
            },
        },
        "impact": {
            "lost_approvals": round(lost_users),
            "downstream_activation_rate": round(downstream, 3),
            "lost_activations": round(lost_activations),
            "margin_12m_per_activated_eur": round(margin12, 2),
            "lost_margin_12m_eur": round(lost_margin),
        },
        "reason_mix": reason_mix,
        "weekly_series": series,
    }
    save_json("kyc_incident.json", payload)

    print(f"\nbreak: {incident['break_date']}  {incident['level_before']:.1%} -> "
          f"{incident['level_during']:.1%}  (LLR {incident['llr']})")
    if incident.get("recovery_date"):
        print(f"partial recovery: {incident['recovery_date']} -> {incident['level_after']:.1%}")
    print(f"suspect release: {closest['version']} ({closest['release_date']}), "
          f"{closest['gap_days']}d before break")
    print(f"impact: ~{lost_users:.0f} lost approvals, ~{lost_activations:.0f} lost activations, "
          f"~EUR {lost_margin:,.0f} 12m margin")


if __name__ == "__main__":
    main()
