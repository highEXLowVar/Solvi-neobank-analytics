"""Post-activation life: monthly engagement, churn, transactions, top-ups, subscriptions.

survival and intensity get modelled per user-month as one big vectorised matrix,
then expanded out to event-level transactions so the warehouse gets realistic
looking raw data rather than monthly aggregates. bit dense, sorry in advance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _month_index(ts: np.ndarray) -> np.ndarray:
    """0-based month index within the simulation window."""
    m = ts.astype("datetime64[M]")
    return (m - C.MONTHS[0]).astype(int)


def _season(months: np.ndarray) -> np.ndarray:
    mnum = (months.astype("datetime64[M]").astype(int) % 12) + 1
    s = np.ones(len(months))
    s[mnum == 12] = C.SEASON_DECEMBER
    s[np.isin(mnum, [6, 7, 8])] = C.SEASON_SUMMER
    return s


def simulate_activity(rng: np.random.Generator, state: pd.DataFrame, latents: pd.DataFrame):
    act = state[state["activated"]].reset_index(drop=True)
    lam_base = latents.set_index("user_id").loc[act["user_id"], "engagement_lambda"].to_numpy()

    # planted finding #3: bonus-induced (incremental) users are low-intent
    penaltyFactor2 = np.where(
        act["incremental"].to_numpy(),
        np.vectorize(C.EXP_INCREMENTAL_ENGAGEMENT.get)(act["channel"].to_numpy()),
        1.0,
    )
    lam = lam_base * penaltyFactor2

    n_users, n_months = len(act), len(C.MONTHS)
    act_ts = act["ts_card"].to_numpy().astype("datetime64[s]")
    m0 = _month_index(act_ts)
    col = np.arange(n_months)[None, :]
    tenure = col - m0[:, None]  # months since activation (negative = pre-activation)

    # ---- churn hazard and survival ------------------------------------------
    churn_base = np.vectorize(C.CHURN_BASE.get)(act["channel"].to_numpy())
    shape = np.select(
        [tenure <= 0, tenure == 1, tenure == 2, tenure >= 12],
        [C.CHURN_TENURE_SHAPE[0], C.CHURN_TENURE_SHAPE[1], C.CHURN_TENURE_SHAPE[2],
         C.CHURN_TENURE_SHAPE[3] * C.CHURN_LATE_FACTOR],
        default=C.CHURN_TENURE_SHAPE[3],
    )
    chMean2 = np.vectorize(C.ENGAGEMENT_MEAN.get)(act["channel"].to_numpy())
    eng_factor = np.clip((chMean2 / np.maximum(lam, 0.3)) ** C.CHURN_ENGAGEMENT_EXP, 0.5, 2.0)
    hazard = np.clip(churn_base[:, None] * shape * eng_factor[:, None], 0, 0.5)

    survive = np.where(tenure >= 0, rng.random((n_users, n_months)) >= hazard, True)
    cum = np.cumprod(survive, axis=1)
    prev = np.concatenate([np.ones((n_users, 1)), cum[:, :-1]], axis=1)
    alive = (tenure == 0) | ((tenure > 0) & (prev > 0))

    # ---- monthly transaction intensity --------------------------------------
    ramp = np.select(
        [tenure == 0, tenure == 1],
        [C.TENURE_RAMP[0], C.TENURE_RAMP[1]],
        default=C.TENURE_DECAY ** np.maximum(tenure - 2, 0),
    )
    season = _season(C.MONTHS)[None, :]
    boost = np.ones((n_users, n_months))
    redeem = act["redeemed"].to_numpy()
    for k, b in enumerate(C.EXP_ACTIVITY_BOOST):
        boost[redeem] = np.where(tenure[redeem] == k, b, boost[redeem])

    # month-0 exposure: only the fraction of the month after activation
    month_starts = C.MONTHS.astype("datetime64[s]")
    month_ends = (C.MONTHS + 1).astype("datetime64[s]")
    days_in_month = ((C.MONTHS + 1).astype("datetime64[D]") - C.MONTHS.astype("datetime64[D]")).astype(int)
    frac0 = (month_ends[m0] - act_ts).astype("timedelta64[s]").astype(float) / (
        days_in_month[m0] * 86400.0
    )
    exposure = np.where(tenure == 0, frac0[:, None], 1.0)

    mu = lam[:, None] * ramp * season * boost * exposure * alive
    counts = rng.poisson(mu)

    # ---- expand to transaction level ----------------------------------------
    u_idx, m_idx = np.nonzero(counts)
    reps = counts[u_idx, m_idx]
    t_user = np.repeat(u_idx, reps)
    t_month = np.repeat(m_idx, reps)
    lo = np.maximum(month_starts[t_month].astype(np.int64), np.repeat(act_ts[u_idx], reps).astype(np.int64))
    hi = np.minimum(month_ends[t_month].astype(np.int64), (C.END_DATE + 1).astype("datetime64[s]").astype(np.int64))
    txn_ts = (lo + rng.random(len(lo)) * np.maximum(hi - lo, 1)).astype("datetime64[s]")

    amounts = np.exp(rng.normal(C.TXN_AMOUNT_LOG_MEDIAN, C.TXN_AMOUNT_SIGMA, len(t_user))).round(2)
    summer = np.isin((txn_ts.astype("datetime64[M]").astype(int) % 12) + 1, [6, 7, 8])
    is_fx = rng.random(len(t_user)) < (C.FX_SHARE_BASE + C.FX_SHARE_SUMMER_EXTRA * summer)
    mcc = rng.choice(C.MCC_CATEGORIES, len(t_user), p=C.MCC_MIX)

    txns = pd.DataFrame(
        {
            "user_id": act["user_id"].to_numpy()[t_user],
            "txn_ts": txn_ts.astype("datetime64[us]"),
            "amount_eur": amounts,
            "mcc_category": pd.Categorical(mcc, categories=C.MCC_CATEGORIES),
            "is_fx": is_fx,
        }
    ).sort_values("txn_ts", kind="stable").reset_index(drop=True)
    txns.insert(0, "txn_id", np.arange(1, len(txns) + 1))

    # ---- top-ups: first top-up (incl. non-activated) + recurring -------------
    first = state[state["has_topup"]]
    first_rows = pd.DataFrame(
        {
            "user_id": first["user_id"].to_numpy(),
            "topup_ts": first["ts_topup"].to_numpy().astype("datetime64[us]"),
            "amount_eur": first["first_topup_amount"].to_numpy().round(2),
            "method": rng.choice(C.TOPUP_METHODS, len(first), p=C.TOPUP_METHOD_MIX),
        }
    )
    tu_scale = np.clip(lam / 12.0, 0.4, 2.5)
    tu_counts = rng.poisson(C.TOPUPS_PER_MONTH * tu_scale[:, None] * exposure * alive)
    tu_counts[np.arange(n_users), m0] = np.maximum(tu_counts[np.arange(n_users), m0] - 1, 0)
    u_idx, m_idx = np.nonzero(tu_counts)
    reps = tu_counts[u_idx, m_idx]
    t_user, t_month = np.repeat(u_idx, reps), np.repeat(m_idx, reps)
    lo = np.maximum(month_starts[t_month].astype(np.int64), np.repeat(act_ts[u_idx], reps).astype(np.int64))
    hi = np.minimum(month_ends[t_month].astype(np.int64), (C.END_DATE + 1).astype("datetime64[s]").astype(np.int64))
    tu_ts = (lo + rng.random(len(lo)) * np.maximum(hi - lo, 1)).astype("datetime64[s]")
    rec_rows = pd.DataFrame(
        {
            "user_id": act["user_id"].to_numpy()[t_user],
            "topup_ts": tu_ts.astype("datetime64[us]"),
            "amount_eur": np.exp(rng.normal(C.TOPUP_AMOUNT_LOG_MEDIAN, C.TOPUP_AMOUNT_SIGMA, len(t_user))).round(2),
            "method": rng.choice(C.TOPUP_METHODS, len(t_user), p=C.TOPUP_METHOD_MIX),
        }
    )
    topups = pd.concat([first_rows, rec_rows], ignore_index=True).sort_values("topup_ts").reset_index(drop=True)
    topups.insert(0, "topup_id", np.arange(1, len(topups) + 1))

    # ---- subscriptions (monthly state machine) 
    #### this loop is slow-ish for what it does, vectorise properly if  row count grows
    age_f = np.vectorize(C.SUB_UPGRADE_AGE.get)(act["age_band"].to_numpy())
    eng_f = np.clip((lam / chMean2) ** 0.6, 0.4, 2.2)
    up_hazard = C.SUB_UPGRADE_BASE * age_f * eng_f
    plan = np.zeros(n_users, dtype=np.int8)  # 0 free, 1 plus, 2 premium
    sub_rows = []
    for m in range(n_months):
        live = alive[:, m]
        plan[~live] = 0
        lapse = (plan > 0) & live & (rng.random(n_users) < C.SUB_LAPSE)
        plan[lapse] = 0
        upgrade = (plan == 0) & live & (rng.random(n_users) < up_hazard)
        plan[upgrade] = np.where(rng.random(int(upgrade.sum())) < C.SUB_PLUS_SHARE, 1, 2)
        subbed = plan > 0
        if subbed.any():
            names = np.where(plan[subbed] == 1, "plus", "premium")
            sub_rows.append(
                pd.DataFrame(
                    {
                        "user_id": act["user_id"].to_numpy()[subbed],
                        "month": np.full(int(subbed.sum()), C.MONTHS[m].astype("datetime64[D]")),
                        "plan": names,
                        "mrr_eur": np.where(plan[subbed] == 1, C.SUB_PLANS["plus"], C.SUB_PLANS["premium"]),
                    }
                )
            )
    subs = pd.concat(sub_rows, ignore_index=True)

    # ---- incentive payouts ( cost side of TOPUP10) ------------------------
    red = state[state["redeemed"]]
    payouts = pd.DataFrame(
        {
            "user_id": red["user_id"].to_numpy(),
            "payout_ts": (red["ts_topup"].to_numpy() + np.timedelta64(2, "h")).astype("datetime64[us]"),
            "amount_eur": C.EXP_BONUS_EUR,
            "campaign": C.EXP_ID,
        }
    )

    truth_users = pd.DataFrame(
        {
            "user_id": act["user_id"],
            "lambda_base": lam_base,
            "lambda_effective": lam,
            "churn_base": churn_base,
        }
    )
    return txns, topups, subs, payouts, truth_users
