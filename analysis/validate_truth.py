"""Close the loop: compare what the analytics recovered against the planted truth.

This is the ONLY analysis file allowed anywhere near simulator config or
data/ground_truth/ , every other script is only allowed to see the warehouse,
same as a real analyst would. If you catch yourself importing simulator.config
into one of the other analysis/*.py files, you've broken the whole point of
this repo, stop and undo it.

Run:  python -m analysis.validate_truth   (after the other analyses)
"""

from __future__ import annotations

import json

import duckdb

from simulator import config as C

from .common import OUT, ROOT

TOL_DATE_DAYS = 4
TOL_LEVEL = 0.03
TOL_UPLIFT_PP = 3.5


def check(name, truth, recovered, ok):
    print(f"  {'PASS' if ok else 'FAIL':<6}{name:<46}truth={truth}  recovered={recovered}")
    return {"name": name, "truth": str(truth), "recovered": str(recovered), "ok": bool(ok)}


def main() -> None:
    kyc = json.loads((OUT / "kyc_incident.json").read_text())
    econ = json.loads((OUT / "retention_econ.json").read_text())
    exp = json.loads((OUT / "experiment.json").read_text())
    con = duckdb.connect()
    results = []

    print("\n-- KYC incident --")
    inc = kyc["incident"]
    results.append(check(
        "affected segment", "android x national_id",
        f"{inc['platform']} x {inc['doc_type']}",
        inc["platform"] == "android" and inc["doc_type"] == "national_id",
    ))
    import datetime as dt
    truth_break = C.INCIDENT_RELEASE_DATE.astype(dt.date)
    rec_break = dt.date.fromisoformat(inc["break_date"])
    results.append(check(
        "break date", truth_break, rec_break,
        abs((rec_break - truth_break).days) <= TOL_DATE_DAYS,
    ))
    truth_fix = C.HOTFIX_RELEASE_DATE.astype(dt.date)
    rec_fix = dt.date.fromisoformat(inc["recovery_date"])
    results.append(check(
        "recovery date", truth_fix, rec_fix,
        abs((rec_fix - truth_fix).days) <= TOL_DATE_DAYS,
    ))
    # observed during_incident level sits slightly above the configured 0.67 -
    # THis is expected, version laggards are still on pre-bug releases so they
    # drag the segment average up a bit. not a bug tho, checked  twice.
    results.append(check(
        "level during incident", C.INCIDENT_APPROVAL, inc["level_during"],
        abs(inc["level_during"] - C.INCIDENT_APPROVAL) <= TOL_LEVEL,
    ))
    results.append(check(
        "level after hotfix", C.POST_HOTFIX_APPROVAL, inc["level_after"],
        abs(inc["level_after"] - C.POST_HOTFIX_APPROVAL) <= TOL_LEVEL,
    ))

    print("\n-- experiment: per-channel uplift vs true SATE --")
    cf = con.sql(f"""
        select u.channel, avg(cf.p_treatment - cf.p_control) * 100 as sate_pp
        from '{ROOT / "data/ground_truth/experiment_counterfactuals.parquet"}' cf
        join '{ROOT / "data/raw/users.parquet"}' u using(user_id)
        group by 1
    """).df().set_index("channel")["sate_pp"]
    for ch, h in exp["heterogeneity"]["by_channel"].items():
        truth_pp = cf[ch]
        est = h["delta_pp"]
        in_ci = h["ci_pp"][0] <= truth_pp <= h["ci_pp"][1]
        results.append(check(
            f"uplift {ch}", f"{truth_pp:.1f}pp", f"{est:.1f}pp (CI {h['ci_pp']})",
            abs(est - truth_pp) <= TOL_UPLIFT_PP and in_ci,
        ))

    print("\n-- channel churn (geometric-tail fit vs configured base hazard) --")
    for ch, fit in econ["churn_fit"].items():
        truth_h = C.CHURN_BASE[ch]
        rec_h = fit["implied_monthly_churn"]
        results.append(check(
            f"monthly churn {ch}", f"{truth_h:.1%}", f"{rec_h:.1%}",
            abs(rec_h - truth_h) <= 0.02,
        ))

    print("\n-- incremental users are low-intent (ground-truth engagement) --")
    # lambda_effective carries the planted low-intent penalty. only exists for
    # activated users, which is handy because that's exactly where the penalty
    # actually shows up. everywhere else it would just be noise
    lat = con.sql(f"""
        select avg(case when cf.incremental_potential and cf.variant = 'treatment'
                        then l.lambda_effective end) as incr,
               avg(case when not cf.incremental_potential
                        then l.lambda_effective end) as rest
        from '{ROOT / "data/ground_truth/experiment_counterfactuals.parquet"}' cf
        join '{ROOT / "data/ground_truth/user_latents.parquet"}' l using(user_id)
        where l.lambda_effective is not null
    """).fetchone()
    results.append(check(
        "incremental users are low-intent", "materially lower",
        f"{lat[0]:.1f} vs {lat[1]:.1f} txns/mo",
        lat[0] < 0.85 * lat[1],
    ))

    print("\n-- uplift model vs true individual effects --")
    # the uplift scores were produced blind (analysis/uplift.py, judged on a
    # held-out half), so the simulator's true per-user effects can grade them
    # here with no sampling noise at all. this file is the one place allowed.
    up = con.sql(f"""
        select s.tau_hat, s.ltv36_eur, s.expected_cost, s.is_eval,
               cf.p_treatment - cf.p_control as tau_true
        from '{OUT / "uplift_scores.parquet"}' s
        join '{ROOT / "data/ground_truth/experiment_counterfactuals.parquet"}' cf using (user_id)
    """).df()
    # calibration first, on the held-out half only: the model's average
    # predicted effect should sit on the true average effect for those exact
    # subjects. a ranking metric can look fine with the sign flipped or the
    # scale off; this can't.
    ev = up[up["is_eval"]]
    mean_hat = ev["tau_hat"].mean() * 100
    mean_true = ev["tau_true"].mean() * 100
    results.append(check(
        "uplift model calibrated on average (held-out)", f"{mean_true:.1f}pp",
        f"{mean_hat:.1f}pp", abs(mean_hat - mean_true) <= 1.5,
    ))
    # the model's business conclusion was "no slice pays back". price the
    # oracle policy (rank by TRUE effect x LTV minus expected cost) and
    # confirm the truth agrees with the model rather than hiding a segment
    # the model missed.
    uplift_out = json.loads((OUT / "uplift.json").read_text())
    model_none = all(
        p["best_slice"]["cum_profit_eur"] <= 0 for p in uplift_out["policies"].values()
    )
    profit_true = up["tau_true"] * up["ltv36_eur"] - up["expected_cost"]
    best_oracle = profit_true.sort_values(ascending=False).cumsum().max()
    results.append(check(
        "model finds no profitable slice; truth agrees", "none on either side",
        f"model: none, oracle best slice {best_oracle:.0f} EUR" if model_none
        else f"model found one?! oracle best {best_oracle:.0f} EUR",
        model_none and best_oracle <= 0,
    ))
    # the punchline of the whole uplift exercise: even an oracle that KNOWS
    # each user's true effect cannot make TOPUP10 pay, because no user's
    # effect x LTV covers the expected bonus cost. targeting was never going
    # to save this incentive, and now that's proven rather than argued.
    results.append(check(
        "TOPUP10 unprofitable even for an oracle", "best slice <= 0 EUR",
        f"best oracle slice {best_oracle:.0f} EUR, "
        f"{(profit_true > 0).mean():.0%} of users individually profitable",
        best_oracle <= 0,
    ))
    # export the oracle curve so the README figure can be drawn without any
    # other script touching ground truth. a percentile grid is plenty.
    curve = profit_true.sort_values(ascending=False).cumsum().reset_index(drop=True)
    grid = [max(0, int(round(q / 100 * len(curve))) - 1) for q in range(1, 101)]
    (OUT / "oracle_curve.json").write_text(json.dumps({
        "frac": [0.0] + [q / 100 for q in range(1, 101)],
        "cum_profit_eur": [0.0] + [round(float(curve.iloc[i]), 1) for i in grid],
        "best_eur": round(float(best_oracle), 1),
    }))
    #PICK UP FROM HERE
    n_pass = sum(r["ok"] for r in results)
    print(f"\n{n_pass}/{len(results)} checks passed")
    (OUT / "validation.json").write_text(json.dumps(
        {"passed": n_pass, "total": len(results), "checks": results}, indent=1))
    if n_pass != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
