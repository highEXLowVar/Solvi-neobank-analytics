"""Generate the full Solvi dataset.

Usage:  python -m simulator.run [--out data]

Writes:
  data/raw/    what the warehouse is allowed to see
  data/ground_truth/  latents and counterfactuals, for validation only (dont leak this)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import activity, config as C, funnel, users


def main(out_dir: str = "data") -> None:
    t0 = time.time()
    rng = np.random.default_rng(C.SEED)
    raw = Path(out_dir) / "raw"
    truth = Path(out_dir) / "ground_truth"
    raw.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)

    print("1/4 users ...", flush=True)
    users_df, latents = users.generate_users(rng)
    print(f"      {len(users_df):,} sign-up starts")

    print("2/4 onboarding funnel + experiment ...", flush=True)
    state, releases_cal, truth_exp = funnel.run_funnel(rng, users_df, latents)
    events = funnel.build_funnel_events(rng, state)
    print(
        f"      approved {state['approved'].sum():,} | "
        f"topup {state['has_topup'].sum():,} | activated {state['activated'].sum():,} | "
        f"experiment {len(truth_exp):,}"
    )

    print("3/4 activity, subscriptions, payouts ...", flush=True)
    txns, topups, subs, payouts, truth_users = activity.simulate_activity(rng, state, latents)
    print(f"      {len(txns):,} card txns | {len(topups):,} topups | {len(subs):,} sub-months")

    print("4/4 writing parquet ...", flush=True)
    users_df.to_parquet(raw / "users.parquet", index=False)
    events.to_parquet(raw / "funnel_events.parquet", index=False)
    txns.to_parquet(raw / "card_transactions.parquet", index=False)
    topups.to_parquet(raw / "topups.parquet", index=False)
    subs.to_parquet(raw / "subscriptions.parquet", index=False)
    payouts.to_parquet(raw / "incentive_payouts.parquet", index=False)

    # FIX THIS!!!!!!!!!

    assigns = state.loc[state["variant"].notna(), ["user_id", "variant"]].copy()
    assigns["experiment_id"] = C.EXP_ID
    assigns["assigned_ts"] = state.loc[state["variant"].notna(), "ts_decision"].astype("datetime64[us]")
    assigns.to_parquet(raw / "experiment_assignments.parquet", index=False)

    releases_cal.to_csv(raw / "app_releases.csv", index=False)
    pd.DataFrame(
        {"channel": list(C.CAC), "cac_eur": list(C.CAC.values())}
    ).to_csv(raw / "channels.csv", index=False)

    latents.merge(truth_users, on="user_id", how="left").to_parquet(
        truth / "user_latents.parquet", index=False
    )
    truth_exp.to_parquet(truth / "experiment_counterfactuals.parquet", index=False)

    print(f"done in {time.time() - t0:.1f}s -> {raw.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    main(p.parse_args().out)
