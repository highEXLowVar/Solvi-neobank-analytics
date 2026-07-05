"""Cohort retention, channel unit economics, CAC payback.

Retention is the classic triangle thing (share of an activation cohort still
transacting k months later). Channel economics combine the per-tenure margin
with the assumed CAC. The tail beyond the observation window is extrapolated
with a fitted geometric decay - documented properly in DESIGN.md and checked
against ground truth so it's not just vibes.

Run:  python -m analysis.retention_economics
"""

from __future__ import annotations

import numpy as np

from .common import connect, save_json

MAX_K = 24 #months of tenure shown
PROJECT_TO = 36 #months for projected LTV
MIN_COHORT = 200


def main() -> None:
    con = connect()

    ##retention curves by channel (cohort-size weighted)
    ret = con.sql(
        f"""
        select channel, months_since_activation as k,
               sum(n_active) as active, sum(cohort_size) as total
        from agg_retention_cohorts
        where cohort_size >= {MIN_COHORT} and months_since_activation <= {MAX_K}
        group by 1, 2 order by 1, 2
        """
    ).df()
    curves = {}
    for ch, g in ret.groupby("channel"):
        curves[ch] = {
            "k": g["k"].astype(int).tolist(),
            "retention": (g["active"] / g["total"]).round(4).tolist(),
        }

    # geometric tail fit on k>= 3 (loglinear): retention_k *TIDDLE* A * s^k
    churn_fit = {}
    for ch, c in curves.items():
        k = np.array(c["k"]) if c["k"] else np.array([])
        r = np.array(c["retention"])
        mask = (k >= 3) & (k <= MAX_K) & (r > 0)
        slope, _ = np.polyfit(k[mask], np.log(r[mask]), 1)
        churn_fit[ch] = {"monthly_survival": round(float(np.exp(slope)), 4),
                         "implied_monthly_churn": round(1 - float(np.exp(slope)), 4)}

    # overal cohort heatmap (for dashboard)
    heat = con.sql(
        f"""
        select activation_month, months_since_activation as k,
               sum(n_active)::double / sum(cohort_size) as retention,
               sum(cohort_size) as cohort_size
        from agg_retention_cohorts
        where months_since_activation <= {MAX_K}
        group by 1, 2
        having sum(cohort_size) >= {MIN_COHORT}
        order by 1, 2
        """
    ).df()
    heatmap = [
        {
            "cohort": str(m.date())[:7],
            "size": int(g["cohort_size"].iloc[0]),
            "retention": g.sort_values("k")["retention"].round(3).tolist(),
        }
        for m, g in heat.groupby("activation_month")
    ]

    # unit economics: cumulative margin vs CAC
    # CAC is quoted per sign-up start but margins only accrue per *activated*
    # user... so we convert CAC to cost-per-activated using the channel's own
    # actvation rate. skip this step and paid channels look about 3x healthier
    # than they actually are, which is a fun way to make a bad recommendation.
    act_rates = dict(
        con.sql(
            "select channel, avg(is_activated::int) from dim_users group by 1"
        ).fetchall()
    )
    econ = con.sql(
        f"""
        select channel, months_since_activation as k, avg_margin_eur, cac_eur, n_observable
        from agg_unit_economics
        where months_since_activation <= {MAX_K}
        order by channel, k
        """
    ).df()
    economics = {}
    for ch, g in econ.groupby("channel"):
        g = g.sort_values("k")
        m = g["avg_margin_eur"].to_numpy()
        cac_signup = float(g["cac_eur"].iloc[0])
        cac = cac_signup / act_rates[ch]
        cum = m.cumsum()

        #geometric extrapolation of monthly margin beyond the window (past month 24ish)
        #fix  damn numpy errors
        k = g["k"].to_numpy()
        mask = (k >= 6) & (m > 0)
        slope, intercept = np.polyfit(k[mask], np.log(m[mask]), 1)
        s = float(np.exp(slope))
        proj_k = np.arange(k.max() + 1, PROJECT_TO + 1)
        proj_m = np.exp(intercept) * s**proj_k
        full_cum = np.concatenate([cum, cum[-1] + proj_m.cumsum()])

        payback = next((int(i) for i, v in enumerate(full_cum) if v >= cac), None)
        economics[ch] = {
            "cac_per_signup_eur": cac_signup,
            "activation_rate": round(act_rates[ch], 3),
            "cac_eur": round(cac, 2),
            "k": k.astype(int).tolist(),
            "cum_margin": cum.round(3).tolist(),
            "projected_cum_margin": full_cum[len(cum):].round(3).tolist(),
            "ltv_36m_eur": round(float(full_cum[PROJECT_TO]), 2),
            "payback_month": payback,
            "monthly_margin_steady_eur": round(float(m[mask].mean()), 3),
        }

    #blended ARPU for context
    arpu = con.sql(
        "select avg(revenue_eur) from fct_activity_monthly where is_active"
    ).fetchone()[0]

    payload = {
        "retention_by_channel": curves,
        "churn_fit": churn_fit,
        "cohort_heatmap": heatmap,
        "economics": economics,
        "blended_arpu_eur": round(arpu, 3),
    }
    save_json("retention_econ.json", payload)

    print("\nchannel        churn/mo   LTV36    CAC/activated   payback")
    for ch, e in economics.items():
        pb = f"{e['payback_month']}mo" if e["payback_month"] is not None else f">{PROJECT_TO}mo"
        print(f"{ch:<13} {churn_fit[ch]['implied_monthly_churn']:>7.1%}  "
              f"EUR{e['ltv_36m_eur']:>6.1f}  EUR{e['cac_eur']:>7.1f}      {pb}")


if __name__ == "__main__":
    main()
