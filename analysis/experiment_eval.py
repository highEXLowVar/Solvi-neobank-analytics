"""TOPUP10 experiment evaluation. this is the file the whole project was for really.

Sections (roughly in the order a reviewer would ask for them)
---------------------------------------------------------------
1. Sample-ratio-mismatch check (chi-square) - always check this FIRST or
   everything downstream is potentially rubbish.
2. Primary outcome: first top-up within 14 days. Two-proportion z-test,
   unpooled 95% CI.
3. Covariate adjustment: linear probability model with HC1 robust errors on
   pre-randomisation covariates (channel, country, device, age band). New
   users have no pre-period, so this is the closest thing to CUPED-style
   variance reduction available here, and it doubles as a randomisation
   sanity check (coefficients shouldn't move much, if they do something's off).
4. Heterogeneity: per-channel effects + interaction Wald test.
5. Guardrail: 60-day contribution margin net of bonus cost (Welch + bootstrap).
6. Unit economics of the incentive: cost per incremental conversion vs an
   upper bound on incremental user value (channel-average 36m LTV).
7. Sequential monitoring demo: naive daily peeking vs a simulation-calibrated
   O'Brien-Fleming-shaped boundary, on A/A resamples of this experiment's own
   enrollment pattern. mostly here because peeking bugs me.

Run:  python -m analysis.experiment_eval   (after retention_economics)
"""
#go back to lecture notes before uploading





from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .common import OUT, connect, save_json

RNG = np.random.default_rng(7)
N_BOOT = 10_000
N_AA_SIMS = 4_000


def two_prop(k1, n1, k0, n0):
    """two-proportion z-test. returns (delta, unpooled 95% CI, z, p) - standard stuff"""
    p1, p0 = k1 / n1, k0 / n0
    se_un = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    p_pool = (k1 + k0) / (n1 + n0)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
    z = (p1 - p0) / se_pool
    return p1 - p0, (p1 - p0 - 1.96 * se_un, p1 - p0 + 1.96 * se_un), z, 2 * stats.norm.sf(abs(z))


def main() -> None:
    con = connect()
    df = con.sql("select * from fct_experiment_users").df()
    df["treat"] = (df["variant"] == "treatment").astype(int)
    df["conv"] = df["conv_14d"].astype(int)
    t, c = df[df["treat"] == 1], df[df["treat"] == 0]

    # 1.SRM
    chi2, srm_p = stats.chisquare([len(t), len(c)])[:2]

    # 2. primary 
    delta, ci, z, p = two_prop(t["conv"].sum(), len(t), c["conv"].sum(), len(c))
    primary = {
        "n_treatment": len(t), "n_control": len(c),
        "conv_treatment": round(t["conv"].mean(), 4), "conv_control": round(c["conv"].mean(), 4),
        "delta_pp": round(delta * 100, 2),
        "ci_pp": [round(ci[0] * 100, 2), round(ci[1] * 100, 2)],
        "relative_lift": round(delta / c["conv"].mean(), 4),
        "z": round(z, 2), "p_value": float(f"{p:.2e}"),
    }

    #3. covariate adjustment (LPM,HC1)
    unadj = smf.ols("conv ~ treat", df).fit(cov_type="HC1")
    adj = smf.ols("conv ~ treat + C(channel) + C(country) + C(device) + C(age_band)", df).fit(cov_type="HC1")
    adjustment = {
        "delta_unadjusted_pp": round(unadj.params["treat"] * 100, 2),
        "se_unadjusted_pp": round(unadj.bse["treat"] * 100, 3),
        "delta_adjusted_pp": round(adj.params["treat"] * 100, 2),
        "se_adjusted_pp": round(adj.bse["treat"] * 100, 3),
        "variance_reduction": round(1 - (adj.bse["treat"] / unadj.bse["treat"]) ** 2, 3),
        "note": "no pre-period exists for new users, so adjustment uses "
                "pre-randomisation covariates (CUPED-style regression adjustment)",
    }

    # 4. heterogeneity. FIXXX!!!!
    inter = smf.ols("conv ~ treat * C(channel) + C(country) + C(device) + C(age_band)", df).fit(cov_type="HC1")
    inter_terms = [x for x in inter.params.index if x.startswith("treat:")]
    wald = inter.wald_test(inter_terms, scalar=True)
    by_channel = {}
    for ch, g in df.groupby("channel"):
        gt, gc = g[g["treat"] == 1], g[g["treat"] == 0]
        d, dci, _, dp = two_prop(gt["conv"].sum(), len(gt), gc["conv"].sum(), len(gc))
        by_channel[ch] = {
            "n_per_arm": [len(gt), len(gc)],
            "conv_control": round(gc["conv"].mean(), 4),
            "delta_pp": round(d * 100, 2),
            "ci_pp": [round(dci[0] * 100, 2), round(dci[1] * 100, 2)],
            "p_value": round(dp, 4),
        }
    heterogeneity = {"by_channel": by_channel,
                     "interaction_wald_p": round(float(wald.pvalue), 4)}

    #5. margin guardrail (60d, net of bonus)
    mt, mc = t["margin_60d_eur"].to_numpy(), c["margin_60d_eur"].to_numpy()
    welch = stats.ttest_ind(mt, mc, equal_var=False)
    boots = np.array([
        RNG.choice(mt, len(mt)).mean() - RNG.choice(mc, len(mc)).mean() for _ in range(N_BOOT)
    ])
    margin = {
        "mean_treatment_eur": round(mt.mean(), 3), "mean_control_eur": round(mc.mean(), 3),
        "delta_eur": round(mt.mean() - mc.mean(), 3),
        "boot_ci_eur": [round(float(np.percentile(boots, 2.5)), 3),
                        round(float(np.percentile(boots, 97.5)), 3)],
        "welch_p": round(welch.pvalue, 4),
        "delta_excl_bonus_eur": round(
            (mt + t["bonus_cost_eur"].to_numpy()).mean() - mc.mean(), 3
        ),
    }

    #6. incentive economics
    ltv = json.loads((OUT / "retention_econ.json").read_text())["economics"]
    economics = {}
    for ch, h in by_channel.items():
        gt = df[(df["channel"] == ch) & (df["treat"] == 1)]
        redemption = gt["redeemed"].mean()
        cost_per_treated = 10.0 * redemption
        d = h["delta_pp"] / 100
        cost_per_incr = cost_per_treated / d if d > 0 else None
        ltv36 = ltv[ch]["ltv_36m_eur"]
        economics[ch] = {
            "redemption_rate": round(redemption, 3),
            "cost_per_treated_eur": round(cost_per_treated, 2),
            "cost_per_incremental_eur": None if cost_per_incr is None else round(cost_per_incr, 0),
            "avg_user_ltv36_eur": ltv36,
            "pays_back_even_at_avg_ltv": bool(cost_per_incr is not None and cost_per_incr < ltv36),
        }
    verdict = (
        #rewrite when not tired lol
        "My recommendation is not to ship it. The lift is real (p well under 0.0001), but "
        "the money doesn't work. Even if I generously assume the extra users are worth the "
        "channel-average 36-month value (the test's own margin data says they are worth "
        "less), the cost of each extra conversion is higher than that value in every "
        "channel. And the lift is statistically indistinguishable across channels (the "
        "interaction test finds nothing), so there is no 'responsive' segment to rescue "
        "it by aiming better. I would redesign it: "
        "pay the reward on the first card transaction instead of the first top-up, or give "
        "something that barely costs Solvi anything, like a month of fee-free FX."
    )

    #7. sequential monitoring (peeking) demo 
    daily = con.sql(
        """
        select assigned_date, variant, sum(n_assigned) n, sum(n_converted_14d) k
        from agg_experiment_daily group by 1, 2 order by 1
        """
    ).df()
    piv_n = daily.pivot(index="assigned_date", columns="variant", values="n").fillna(0)
    piv_k = daily.pivot(index="assigned_date", columns="variant", values="k").fillna(0)
    n1d, n0d = piv_n["treatment"].to_numpy(), piv_n["control"].to_numpy()
    D = len(n1d)
    p_c = c["conv"].mean()

    def z_path(k1c, n1c, k0c, n0c):
        pp = (k1c + k0c) / (n1c + n0c)
        se = np.sqrt(np.maximum(pp * (1 - pp), 1e-12) * (1 / n1c + 1 / n0c))
        return (k1c / n1c - k0c / n0c) / se

    # A/A world:both arms draw from controls' conversion rate
    k1 = RNG.binomial(n1d.astype(int), p_c, (N_AA_SIMS, D)).cumsum(axis=1)
    k0 = RNG.binomial(n0d.astype(int), p_c, (N_AA_SIMS, D)).cumsum(axis=1)
    Z = z_path(k1, n1d.cumsum(), k0, n0d.cumsum())
    naive_hit = np.maximum.accumulate(np.abs(Z) >= 1.96, axis=1)
    naive_fpr = naive_hit.mean(axis=0)

    info = (n1d + n0d).cumsum() / (n1d + n0d).sum()

    def fpr_at(Cb):
        return np.maximum.accumulate(np.abs(Z) >= Cb / np.sqrt(info), axis=1)[:, -1].mean()

    lo, hi = 1.5, 5.0
    for _ in range(40):  #bisecting our way to a 5% family-wise rate, no closed form for this
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fpr_at(mid) > 0.05 else (lo, mid)
    C_obf = (lo + hi) / 2 #sorted
    obf_fpr = np.maximum.accumulate(np.abs(Z) >= C_obf / np.sqrt(info), axis=1).mean(axis=0)

    real_z = z_path(
        piv_k["treatment"].to_numpy().cumsum(), n1d.cumsum(),
        piv_k["control"].to_numpy().cumsum(), n0d.cumsum(),
    )
    boundary = C_obf / np.sqrt(info)
    peeking = {
        "note": "A/A simulations replay this experiment's real daily enrollment "
                "with both arms at the control conversion rate",
        "n_sims": N_AA_SIMS,
        "days": [str(d.date()) for d in piv_n.index],
        "naive_fpr": naive_fpr.round(4).tolist(),
        "obf_constant": round(C_obf, 3),
        "obf_fpr_final": round(float(obf_fpr[-1]), 4),
        "boundary": boundary.round(3).tolist(),
        "real_z_path": real_z.round(3).tolist(),
        "first_naive_cross_day": int(np.argmax(np.abs(real_z) >= 1.96)),
        "first_obf_cross_day": int(np.argmax(np.abs(real_z) >= boundary)),
    }

    payload = {
        "experiment_id": df["experiment_id"].iloc[0],
        "srm": {"chi2": round(chi2, 3), "p": round(srm_p, 4)},
        "primary": primary,
        "adjustment": adjustment,
        "heterogeneity": heterogeneity,
        "margin_60d": margin,
        "economics": economics,
        "verdict": verdict,
        "peeking": peeking,
    }
    save_json("experiment.json", payload)
    #ugly
    print(f"\nprimary: {primary['conv_control']:.1%} -> {primary['conv_treatment']:.1%}  "
          f"(+{primary['delta_pp']}pp, z={primary['z']}, p={primary['p_value']:.1e})")
    print(f"margin 60d net of bonus: {margin['delta_eur']:+.2f} EUR/user "
          f"(CI {margin['boot_ci_eur']}, excl. bonus {margin['delta_excl_bonus_eur']:+.2f})")
    print("\nchannel        uplift     cost/incr   LTV36   pays back?")
    for ch, e in economics.items():
        print(f"{ch:<13} {by_channel[ch]['delta_pp']:>5.1f}pp   "
              f"EUR{e['cost_per_incremental_eur']:>6.0f}   EUR{e['avg_user_ltv36_eur']:>5.1f}   "
              f"{'YES' if e['pays_back_even_at_avg_ltv'] else 'no'}")
    print(f"\npeeking: naive daily FPR {naive_fpr[-1]:.1%} vs calibrated OBF {obf_fpr[-1]:.1%} "
          f"(C={C_obf:.2f}); real z crosses naive day {peeking['first_naive_cross_day']}, "
          f"OBF day {peeking['first_obf_cross_day']}")


if __name__ == "__main__":
    main()
