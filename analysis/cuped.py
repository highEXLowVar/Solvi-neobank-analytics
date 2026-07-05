"""CUPED on TOPUP10, done properly this time.

Follow-up to section 3 of experiment_eval. There I noted that new users have no
pre-period, so I settled for regression adjustment on pre-randomisation
covariates and called it "CUPED-style". After reading the actual paper (Deng,
Xu, Kohavi & Walker 2013, "Improving the Sensitivity of Online Controlled
Experiments by Utilizing Pre-Experiment Data") I wanted the real estimator,
not the style. The trick that makes it possible without a pre-period is CUPAC
(DoorDash's extension): the CUPED covariate doesn't have to be the metric's
own pre-period value, it can be ANY variable independent of treatment. So I
train a model on users who passed KYC *before* the experiment ever started,
predicting the same 14-day top-up outcome from the same pre-randomisation
covariates, and use its prediction as the control variate.

Why bother when section 3 already adjusts for the same covariates?
- it's the estimator experimentation platforms actually run (one theta, no
  refitting a regression per metric per experiment),
- the covariate model is fitted on ~100k out-of-experiment users, so it can
  soak up interactions/nonlinearities without spending experiment degrees of
  freedom or inviting "adjusted until significant" accusations,
- and honestly, I wanted to see how close variance reduction gets to rho^2
  like the paper promises.

Sections
--------
1. Fit the covariate models on pre-experiment users only (a gradient-boosted
   classifier for conversion, a regressor for 60-day margin), report held-out
   AUC / R2 so the covariate quality is on the record.
2. CUPED on the primary outcome (conv_14d): theta = cov(Y,X)/var(X),
   Y_adj = Y - theta*(X - mean(X)), same delta, smaller SE.
3. Same machinery on the noisy margin guardrail, where variance reduction
   actually matters most in practice.
4. Comparison table: unadjusted vs section-3 regression adjustment vs CUPED.

The one thing to be careful about: the covariate must not touch anything
post-randomisation. Everything the models see (channel, country, device,
age band) is fixed before the coin flip, and the training users finished
their outcome window before the experiment began.

Run:  python -m analysis.cuped   (after experiment_eval)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .common import OUT, connect, save_json

FEATURES = ["channel", "country", "device", "age_band"]
Z95 = 1.96


def as_categories(df: pd.DataFrame) -> pd.DataFrame:
    """HistGradientBoosting handles categoricals natively if the dtype says so."""
    out = df[FEATURES].copy()
    for col in FEATURES:
        out[col] = out[col].astype("category")
    return out


def cuped_adjust(y: np.ndarray, x: np.ndarray, treat: np.ndarray) -> dict:
    """The whole method is four lines. theta is fitted pooled across arms,
    which is fine because x is independent of assignment by construction."""
    theta = np.cov(y, x)[0, 1] / np.var(x)
    y_adj = y - theta * (x - x.mean())

    def arm_stats(vals, mask):
        v = vals[mask]
        return v.mean(), v.var(ddof=1) / len(v)

    m1, v1 = arm_stats(y_adj, treat == 1)
    m0, v0 = arm_stats(y_adj, treat == 0)
    m1_raw, v1_raw = arm_stats(y, treat == 1)
    m0_raw, v0_raw = arm_stats(y, treat == 0)
    delta, se = m1 - m0, np.sqrt(v1 + v0)
    delta_raw, se_raw = m1_raw - m0_raw, np.sqrt(v1_raw + v0_raw)
    return {
        "theta": round(float(theta), 4),
        "rho_y_x": round(float(np.corrcoef(y, x)[0, 1]), 4),
        "delta_unadjusted": float(delta_raw),
        "se_unadjusted": float(se_raw),
        "delta_cuped": float(delta),
        "se_cuped": float(se),
        "ci_cuped": [float(delta - Z95 * se), float(delta + Z95 * se)],
        "variance_reduction": round(1 - (se / se_raw) ** 2, 4),
    }


def main() -> None:
    con = connect()
    # order by user_id so the numbers don't depend on warehouse row order
    exp = con.sql("select * from fct_experiment_users order by user_id").df()
    treat = (exp["variant"] == "treatment").to_numpy().astype(int)

    # 1. training cohort: KYC-approved users whose whole story predates the
    # experiment. their "conv_14d" is rebuilt with the same definition the
    # mart uses (first top-up within 14 days of approval), and the margin
    # proxy reuses the monthly mart the same way fct_experiment_users windows
    # subscriptions. proxy != exact 60-day margin, but a covariate only needs
    # to correlate with the outcome, it doesn't need to BE the outcome.
    pre = con.sql(
        """
        with pre_users as (
            select
                user_id, channel, country, device, age_band, kyc_approved_ts,
                coalesce(first_topup_ts < kyc_approved_ts + interval 14 day, false) as conv_14d
            from dim_users
            where is_kyc_approved
              and experiment_id is null
              and kyc_approved_ts < (select min(assigned_ts) from fct_experiment_users)
        )
        select p.*, coalesce(m.margin_60d_proxy, 0) as margin_60d_proxy
        from pre_users p
        left join (
            select p.user_id, sum(a.contribution_margin_eur) as margin_60d_proxy
            from pre_users p
            join fct_activity_monthly a
              on a.user_id = p.user_id
             and a.month >= date_trunc('month', p.kyc_approved_ts)
             and a.month < p.kyc_approved_ts + interval 60 day
            group by 1
        ) m using (user_id)
        order by p.user_id
        """
    ).df()

    X_pre = as_categories(pre)
    y_pre_conv = pre["conv_14d"].astype(int).to_numpy()
    y_pre_marg = pre["margin_60d_proxy"].to_numpy()

    # held-out quality check first, so the AUC/R2 below isn't a training score
    idx_tr, idx_te = train_test_split(np.arange(len(pre)), test_size=0.25, random_state=7)
    clf = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=7)
    clf.fit(X_pre.iloc[idx_tr], y_pre_conv[idx_tr])
    auc = roc_auc_score(y_pre_conv[idx_te], clf.predict_proba(X_pre.iloc[idx_te])[:, 1])
    reg = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=7)
    reg.fit(X_pre.iloc[idx_tr], y_pre_marg[idx_tr])
    r2 = r2_score(y_pre_marg[idx_te], reg.predict(X_pre.iloc[idx_te]))

    # refit on all pre-experiment users for the covariates we actually use
    clf.fit(X_pre, y_pre_conv)
    reg.fit(X_pre, y_pre_marg)
    X_exp = as_categories(exp)
    x_conv = clf.predict_proba(X_exp)[:, 1]
    x_marg = reg.predict(X_exp)

    # 2. primary outcome
    y_conv = exp["conv_14d"].astype(int).to_numpy()
    conv = cuped_adjust(y_conv, x_conv, treat)

    # 3. margin guardrail
    y_marg = exp["margin_60d_eur"].to_numpy()
    marg = cuped_adjust(y_marg, x_marg, treat)

    # 4. line up against experiment_eval's section 3
    prior = json.loads((OUT / "experiment.json").read_text())["adjustment"]
    comparison = {
        "unadjusted": {"delta_pp": prior["delta_unadjusted_pp"], "se_pp": prior["se_unadjusted_pp"]},
        "regression_adjustment": {
            "delta_pp": prior["delta_adjusted_pp"],
            "se_pp": prior["se_adjusted_pp"],
            "variance_reduction": prior["variance_reduction"],
        },
        "cuped_cupac": {
            "delta_pp": round(conv["delta_cuped"] * 100, 2),
            "se_pp": round(conv["se_cuped"] * 100, 3),
            "variance_reduction": conv["variance_reduction"],
        },
        "note": "same information set by design, so CUPED shouldn't beat the "
                "regression by much here. with only four categorical covariates "
                "the model can't invent signal that isn't there; the win on real "
                "data comes from rich pre-period behaviour, which new users "
                "don't have. the machinery is the point.",
    }

    payload = {
        "covariate_models": {
            "n_pre_experiment_users": len(pre),
            "features": FEATURES,
            "conv_model_auc_heldout": round(float(auc), 4),
            "margin_model_r2_heldout": round(float(r2), 4),
        },
        "primary_conv_14d": conv,
        "margin_60d": marg,
        "comparison": comparison,
    }
    save_json("cuped.json", payload)

    print(f"\ncovariate models: {len(pre):,} pre-experiment users, "
          f"AUC {auc:.3f} (conv), R2 {r2:.3f} (margin)")
    print("\nmethod                delta        se      var.reduction")
    print(f"unadjusted          {prior['delta_unadjusted_pp']:>6.2f}pp   {prior['se_unadjusted_pp']:.3f}pp        -")
    print(f"regression (sec.3)  {prior['delta_adjusted_pp']:>6.2f}pp   {prior['se_adjusted_pp']:.3f}pp     {prior['variance_reduction']:.1%}")
    print(f"CUPED/CUPAC         {conv['delta_cuped']*100:>6.2f}pp   {conv['se_cuped']*100:.3f}pp     {conv['variance_reduction']:.1%}"
          f"   (rho^2 = {conv['rho_y_x']**2:.3f})")
    print(f"\nmargin guardrail: {marg['delta_unadjusted']:+.2f} EUR "
          f"se {marg['se_unadjusted']:.3f} -> {marg['se_cuped']:.3f} "
          f"({marg['variance_reduction']:.1%} variance reduction)")


if __name__ == "__main__":
    main()
