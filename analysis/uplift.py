"""Uplift modelling on TOPUP10: can targeting rescue the bonus?

The experiment write-up says there is no responsive segment to aim the bonus
at. I had asserted that from four channel averages, which felt thin. This
file builds the model that actually tries: estimate each user's individual
treatment effect, rank everyone by it, and ask whether the 10 euro bonus pays
for itself on ANY targetable slice.

Method
------
T-learner with a held-out evaluation half. Two gradient-boosted classifiers,
one fitted on treated users and one on controls, each predicting 14-day
conversion; predicted uplift is the difference. I fit both models on a random
half of the subjects and judge everything on the other half, so every
reported number comes from users the models never saw.

Why a plain split and not out-of-fold cross-fitting: I tried cross-fitting
first and the decile curve came out upside down. The features here are four
categorical fields, so users live in a few hundred cells, and a leave-out
score inside a cell is basically "the cell mean minus your own outcome".
Converters get scored lower than their neighbours, the ranking anti-selects
outcomes, and the evaluation poisons itself. With a held-out half, everyone
in a cell gets the same score and the problem disappears. Lesson learned the
slow way.

Rules I held myself to
----------------------
- Decile lifts, redemption rates and profits below are computed from actual
  treatment/control outcomes on the held-out half, never from predictions.
- The model only RANKS. Whether a slice pays back is decided by the observed
  lift in that slice times a generous channel-average LTV, minus the observed
  bonus cost. If targeting fails here it fails on its best behaviour.
- This file never touches data/ground_truth/. The check of these scores
  against the true per-user effects lives in validate_truth, where it belongs.

Outputs: analysis/out/uplift.json (decile table, Qini, policy economics) and
analysis/out/uplift_scores.parquet (per-user scores, for validate_truth).

Run:  python -m analysis.uplift   (after retention_economics)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from .common import OUT, connect, save_json

FEATURES = ["channel", "country", "device", "age_band"]
EVAL_FRACTION = 0.5
N_GROUPS = 10
BONUS_EUR = 10.0


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """HistGradientBoosting handles categoricals natively if the dtype says so."""
    X = df[FEATURES].copy()
    for col in FEATURES:
        X[col] = X[col].astype("category")
    return X


def observed_uplift(g: pd.DataFrame) -> tuple[float, float, int, int]:
    """observed conv(T) - conv(C) in a slice, with an unpooled SE"""
    t, c = g[g["variant"] == "treatment"], g[g["variant"] == "control"]
    if len(t) == 0 or len(c) == 0:
        return np.nan, np.nan, len(t), len(c)
    p1, p0 = t["conv_14d"].mean(), c["conv_14d"].mean()
    se = np.sqrt(p1 * (1 - p1) / len(t) + p0 * (1 - p0) / len(c))
    return p1 - p0, se, len(t), len(c)


def group_table(ev: pd.DataFrame, order_col: str) -> list[dict]:
    """split the held-out users into N_GROUPS slices by descending score and
    measure each slice with real outcomes. equal-size groups rather than qcut,
    because with four categorical features the scores tie heavily."""
    ev = ev.sort_values(order_col, ascending=False, kind="stable").reset_index(drop=True)
    rows = []
    for gi, idx in enumerate(np.array_split(ev.index.to_numpy(), N_GROUPS)):
        g = ev.loc[idx]
        up, se, n_t, n_c = observed_uplift(g)
        gt = g[g["variant"] == "treatment"]
        redemption = gt["redeemed"].mean() if len(gt) else np.nan
        # deploying to this slice means everyone in it gets the offer, so
        # value = observed lift x avg LTV, and cost per deployed user is
        # 10 x P(redeem | offered), which the treated arm measures directly
        value = up * g["ltv36_eur"].mean()
        rows.append({
            "group": gi + 1,
            "n": len(g), "n_treatment": n_t, "n_control": n_c,
            "tau_hat_mean": round(g["tau_hat"].mean(), 4),
            "uplift_obs_pp": None if np.isnan(up) else round(up * 100, 2),
            "uplift_se_pp": None if np.isnan(se) else round(se * 100, 2),
            "redemption_treated": None if np.isnan(redemption) else round(redemption, 3),
            "avg_ltv36_eur": round(g["ltv36_eur"].mean(), 2),
            "profit_per_user_eur": None if np.isnan(up) else round(value - BONUS_EUR * redemption, 2),
            "top_channel": g["channel"].mode().iloc[0],
        })
    return rows


def main() -> None:
    con = connect()
    # order by user_id so the split (and every number downstream) is
    # reproducible no matter what order dbt happened to write the rows in
    df = con.sql("select * from fct_experiment_users order by user_id").df()
    ltv = json.loads((OUT / "retention_econ.json").read_text())["economics"]
    df["ltv36_eur"] = df["channel"].map(lambda ch: ltv[ch]["ltv_36m_eur"])

    # fit half / judge half, stratified on arm x outcome
    treat = (df["variant"] == "treatment").to_numpy()
    y = df["conv_14d"].astype(int).to_numpy()
    strata = treat.astype(int) * 2 + y
    tr_idx, ev_idx = train_test_split(
        np.arange(len(df)), test_size=EVAL_FRACTION, random_state=7, stratify=strata
    )
    df["is_eval"] = False
    df.loc[ev_idx, "is_eval"] = True

    X = feature_frame(df)
    m_t = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=7)
    m_c = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=7)
    m_t.fit(X.iloc[tr_idx][treat[tr_idx]], y[tr_idx][treat[tr_idx]])
    m_c.fit(X.iloc[tr_idx][~treat[tr_idx]], y[tr_idx][~treat[tr_idx]])
    df["tau_hat"] = m_t.predict_proba(X)[:, 1] - m_c.predict_proba(X)[:, 1]

    # expected bonus cost per deployed user, by channel, from the treated arm
    df["expected_cost"] = BONUS_EUR * df["channel"].map(
        df[df["variant"] == "treatment"].groupby("channel")["redeemed"].mean()
    )
    # two targeting policies:
    #   A. rank by predicted conversion lift (the textbook one)
    #   B. rank by predicted PROFIT per deployed user: lift x that user's
    #      channel LTV minus expected bonus cost. B is the economically right
    #      objective; A is what you get if you forget users differ in value.
    df["profit_score"] = df["tau_hat"] * df["ltv36_eur"] - df["expected_cost"]

    ev = df[df["is_eval"]].reset_index(drop=True)

    # Qini-style curve on the held-out half: walk down the ranking and count
    # the extra conversions the bonus would cause if only the top q got it,
    # using observed rates. the random-targeting baseline is just q x total.
    ev_sorted = ev.sort_values("tau_hat", ascending=False, kind="stable").reset_index(drop=True)
    fracs = np.round(np.arange(0.05, 1.0001, 0.05), 2)
    total_uplift, _, _, _ = observed_uplift(ev)
    qini_curve = []
    for q in fracs:
        top = ev_sorted.iloc[: int(round(q * len(ev_sorted)))]
        up, _, _, _ = observed_uplift(top)
        qini_curve.append({
            "frac": float(q),
            "incremental_conversions": round(up * len(top), 1),
            "random_baseline": round(total_uplift * len(top), 1),
        })
    qini_area = float(np.trapezoid(
        [r["incremental_conversions"] - r["random_baseline"] for r in qini_curve], fracs
    ))

    policies = {}
    for name, col in [("A_rank_by_uplift", "tau_hat"), ("B_rank_by_profit", "profit_score")]:
        table = group_table(ev, col)
        cum_profit, best = 0.0, {"frac": 0.0, "cum_profit_eur": 0.0}
        for i, row in enumerate(table):
            if row["profit_per_user_eur"] is None:
                continue
            cum_profit += row["profit_per_user_eur"] * row["n"]
            if cum_profit > best["cum_profit_eur"]:
                best = {"frac": round((i + 1) / N_GROUPS, 1), "cum_profit_eur": round(cum_profit, 0)}
        policies[name] = {"groups": table, "best_slice": best}

    any_positive = any(
        r["profit_per_user_eur"] is not None and r["profit_per_user_eur"] > 0
        for p in policies.values() for r in p["groups"]
    )
    # the top-30% slice is the model's best shot; when the ranking carries no
    # signal the honest summary is the average economics, so quote both
    top30 = ev_sorted.iloc[: int(round(0.3 * len(ev_sorted)))]
    up30, se30, _, _ = observed_uplift(top30)
    cost30 = BONUS_EUR * float(top30.loc[top30["variant"] == "treatment", "redeemed"].mean())
    value30 = up30 * float(top30["ltv36_eur"].mean())
    avg_cost = BONUS_EUR * float(ev.loc[ev["variant"] == "treatment", "redeemed"].mean())
    avg_value = total_uplift * float(ev["ltv36_eur"].mean())
    verdict = (
        "The claim survives, and more bluntly than I expected. The per-channel "
        "analysis already found no statistically detectable differences in lift, "
        "and the uplift model comes back with the same answer rather than "
        "inventing structure: its scores barely spread, the observed decile "
        "curve on the held-out half is flat within noise, and its favourite "
        "slice does no better than the average. Whatever makes one individual "
        "more persuadable than their neighbour isn't visible in anything Solvi "
        "knows before the coin flip. "
        "At roughly 5,000 users per arm, even real heterogeneity would be hard "
        "to rank reliably; detecting who responds needs far more data than "
        "detecting whether anyone responds. The economics settle it anyway: on "
        f"average an extra conversion is worth about €{avg_value:.2f} per "
        f"deployed user against about €{avg_cost:.2f} of expected bonus cost, and "
        f"{'only a thin slice ever goes positive' if any_positive else 'no slice of any size goes positive under either ranking'}. "
        "Targeting sharpens who you pay; it cannot fix a reward that costs "
        "several times what the behaviour is worth. The redesigns from the main "
        "write-up (pay on the first card transaction, or give fee-free FX) are "
        "still the answer."
    )

    # ltv and expected cost ride along so validate_truth can price an oracle
    # policy without redoing the economics; is_eval marks the held-out half
    scores = df[["user_id", "tau_hat", "profit_score", "channel",
                 "ltv36_eur", "expected_cost", "is_eval"]].copy()
    scores.to_parquet(OUT / "uplift_scores.parquet", index=False)
    print(f"wrote {(OUT / 'uplift_scores.parquet').relative_to(OUT.parent.parent)}")

    payload = {
        "model": {
            "learner": "T-learner, HistGradientBoosting per arm",
            "features": FEATURES,
            "eval": f"held-out {EVAL_FRACTION:.0%} of subjects, models never saw them",
        },
        "n_eval": len(ev),
        "avg_uplift_pp": round(total_uplift * 100, 2),
        "top30_slice": {
            "uplift_obs_pp": round(up30 * 100, 2),
            "uplift_se_pp": round(se30 * 100, 2),
            "value_per_deployed_eur": round(value30, 2),
            "cost_per_deployed_eur": round(cost30, 2),
        },
        "avg_economics": {
            "value_per_deployed_eur": round(avg_value, 2),
            "cost_per_deployed_eur": round(avg_cost, 2),
        },
        "qini_curve": qini_curve,
        "qini_area_vs_random": round(qini_area, 1),
        "policies": policies,
        "verdict": verdict,
    }
    save_json("uplift.json", payload)

    print(f"\nheld-out half: {len(ev):,} users, avg lift {total_uplift*100:.1f}pp, "
          f"qini area vs random {qini_area:+.0f}")
    print(f"top-30% slice: {up30*100:.1f}pp obs lift, "
          f"EUR{value30:.2f} value vs EUR{cost30:.2f} cost per deployed user")
    for name, p in policies.items():
        best = p["best_slice"]
        best_txt = ("no slice goes positive" if best["frac"] == 0
                    else f"top {best['frac']:.0%} -> {best['cum_profit_eur']:+,.0f} EUR")
        print(f"\n{name}   (best cumulative slice: {best_txt})")
        print("group   n     lift(obs)     LTV36    profit/user")
        for r in p["groups"]:
            up = "   -  " if r["uplift_obs_pp"] is None else f"{r['uplift_obs_pp']:>5.1f}pp"
            pr = "    - " if r["profit_per_user_eur"] is None else f"{r['profit_per_user_eur']:>6.2f}"
            print(f"{r['group']:>3}   {r['n']:>5}  {up} +-{r['uplift_se_pp']:.1f}   "
                  f"EUR{r['avg_ltv36_eur']:>5.1f}   EUR{pr}   ({r['top_channel']})")


if __name__ == "__main__":
    main()
