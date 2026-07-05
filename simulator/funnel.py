"""Onboarding funnel: email verification -> KYC -> first top-up -> first card txn.

Also runs the TOPUP10 experiment via per-user counterfactual draws, so true
incrementality is known and saved as ground truth. this file got long, sorry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C
from . import releases


def _lognormal_delay(rng, n, log_median, sigma, unit="h", max_value=None):
    """Lognormal delays as timedelta64[s]; optionally truncated (inverse-CDF, no pile-up)."""
    if max_value is not None:
        u_max = norm.cdf((np.log(max_value) - log_median) / sigma)
        u = rng.uniform(0, u_max, n)
        x2 = np.exp(log_median + sigma * norm.ppf(u))
    else:
        x2 = rng.lognormal(log_median, sigma, n)
    seconds = {"m": 60, "h": 3600, "d": 86400}[unit]  #faffy ik but avoids chain of if/elifs
    return (x2 * seconds).astype("timedelta64[s]")


def _lookup(mapping: dict, keys: np.ndarray) -> np.ndarray:
    return np.vectorize(mapping.get)(keys)


def run_funnel(rng: np.random.Generator, users: pd.DataFrame, latents: pd.DataFrame):
    n = len(users)
    ts_signup = users["signup_ts"].to_numpy()
    channel = users["channel"].to_numpy()
    device = users["device"].to_numpy()
    age = users["age_band"].to_numpy()
    doc = latents["doc_type"].to_numpy()

    # ---- stage passage -----------------------------
    p_email = _lookup(C.EMAIL_VERIFIED_BASE, channel)
    email_ok = rng.random(n) < p_email
    ts_email = ts_signup + _lognormal_delay(rng, n, np.log(4.0), 1.0, "m")

    p_kyc_start = _lookup(C.KYC_START_BASE, channel)
    kyc_started = email_ok & (rng.random(n) < p_kyc_start)
    ts_kyc_start = ts_email + _lognormal_delay(rng, n, np.log(2.0), 1.4, "h")

    p_submit = C.DOC_SUBMIT_BASE - np.where(device == "android", C.DOC_SUBMIT_ANDROID_PENALTY, 0.0)
    submitted = kyc_started & (rng.random(n) < p_submit)
    ts_submit = ts_kyc_start + _lognormal_delay(rng, n, np.log(9.0), 0.8, "m")

    # ---- KYC approval (incident is fully version-mediated) 
    cal = releases.release_calendar()
    inc_versions = releases.incident_versions(cal)
    android_post_hotfix = (
        cal[(cal["platform"] == "android")
            & (cal["release_date"] >= pd.Timestamp(C.HOTFIX_RELEASE_DATE))]["version"].tolist()
    )
    app_version = releases.version_at(rng, cal, device, ts_submit)

    p_approve = _lookup(C.KYC_APPROVAL_BY_DOC, doc) - np.where(device == "android", C.KYC_ANDROID_PENALTY, 0.0)
    seg = (device == "android") & (doc == "national_id")
    on_buggy = seg & np.isin(app_version, inc_versions)
    on_hotfix = seg & np.isin(app_version, android_post_hotfix)
    p_approve = np.where(on_buggy, C.INCIDENT_APPROVAL, p_approve)
    p_approve = np.where(on_hotfix, C.POST_HOTFIX_APPROVAL, p_approve)

    approved = submitted & (rng.random(n) < p_approve)
    rejected = submitted & ~approved

    n_attempts = np.ones(n, dtype=int)
    n_attempts[approved] = rng.choice([1, 2, 3], approved.sum(), p=C.ATTEMPTS_IF_APPROVED)
    n_attempts[rejected] = rng.choice([1, 2, 3], rejected.sum(), p=C.ATTEMPTS_IF_REJECTED)

    # rejection reasons: the incident's excess rejections are all doc_unreadable
    # FIX this whole mixture thing is a bit much, revisit when theres time (there wont be)
    reason = np.full(n, None, dtype=object)
    base_reject = 1.0 - (_lookup(C.KYC_APPROVAL_BY_DOC, doc) - np.where(device == "android", C.KYC_ANDROID_PENALTY, 0.0))
    for mask_inc, p_inc in ((on_buggy, 1 - C.INCIDENT_APPROVAL), (on_hotfix, 1 - C.POST_HOTFIX_APPROVAL)):
        m = rejected & mask_inc
        if m.sum():
            excess = np.maximum(p_inc - base_reject[m], 0)
            mixTemp3 = np.outer(base_reject[m], C.REJECT_REASON_MIX)
            mixTemp3[:, 0] += excess
            mixTemp3 /= mixTemp3.sum(axis=1, keepdims=True)
            cum = mixTemp3.cumsum(axis=1)
            reason[m] = np.array(C.REJECT_REASONS, dtype=object)[
                (rng.random(m.sum())[:, None] < cum).argmax(axis=1)
            ]
    m = rejected & ~on_buggy & ~on_hotfix
    reason[m] = rng.choice(C.REJECT_REASONS, m.sum(), p=C.REJECT_REASON_MIX)

    #decision latency after the last attempt: mostly automated, some manual review
    manualReview2 = rng.random(n) < C.MANUAL_REVIEW_SHARE
    decision_gap = np.where(
        manualReview2,
        _lognormal_delay(rng, n, np.log(30.0), 0.6, "h").astype("timedelta64[s]").astype(np.int64),
        _lognormal_delay(rng, n, np.log(30.0), 0.7, "m").astype("timedelta64[s]").astype(np.int64),
    ).astype("timedelta64[s]")
    #xtra attempts add hours
    extra_gap = np.zeros(n, dtype="timedelta64[s]")
    multi = n_attempts > 1
    extra_gap[multi] = _lognormal_delay(rng, int(multi.sum()), np.log(6.0), 1.3, "h") * (
        n_attempts[multi] - 1
    )
    ts_decision = ts_submit + extra_gap + decision_gap

    # experiment assignment at KYC approval
    in_window = (
        approved
        & (ts_decision >= C.EXP_START.astype("datetime64[s]"))
        & (ts_decision < C.EXP_END.astype("datetime64[s]"))
    )
    variant = np.full(n, None, dtype=object)
    variant[in_window] = np.where(rng.random(int(in_window.sum())) < 0.5, "treatment", "control")


    #first top-up: per-user counterfactual draw -------------------------
    p_control = np.clip(_lookup(C.TOPUP_14D_BASE, channel) + _lookup(C.TOPUP_AGE_ADJ, age), 0.01, 0.99)
    delta = np.where(in_window, _lookup(C.EXP_UPLIFT, channel), 0.0)
    u = rng.random(n)
    converts_control = u < p_control
    converts_treat = u < p_control + delta
    incremental_potential = ~converts_control & converts_treat
    is_treated = variant == "treatment"
    converts14 = approved & np.where(is_treated, converts_treat, converts_control)

    late = approved & ~converts14 & (rng.random(n) < C.LATE_TOPUP_RATE)
    topup_delay = np.zeros(n, dtype="timedelta64[s]")
    topup_delay[converts14] = _lognormal_delay(
        rng, int(converts14.sum()), np.log(1.2), 1.1, "d", max_value=13.9
    )
    topup_delay[late] = (rng.uniform(15, 45, int(late.sum())) * 86400).astype("timedelta64[s]")
    has_topup = converts14 | late
    ts_topup = ts_decision + topup_delay

    amount = np.exp(rng.normal(C.FIRST_TOPUP_LOG_MEDIAN, C.FIRST_TOPUP_SIGMA, n))
    aware = is_treated & converts14 & (amount < C.EXP_MIN_TOPUP) & (
        rng.random(n) < C.EXP_TREATED_THRESHOLD_AWARE
    )
    amount[aware] = rng.uniform(20, 30, int(aware.sum()))
    redeemed = is_treated & converts14 & (amount >= C.EXP_MIN_TOPUP)

    #first card transaction = activation
    activated = has_topup & (rng.random(n) < C.CARD_TXN_GIVEN_TOPUP)
    ts_card = ts_topup + _lognormal_delay(rng, n, np.log(2.0), 1.0, "d")
    endS_FINAL = (C.END_DATE + 1).astype("datetime64[s]")
    has_topup &= ts_topup < endS_FINAL
    activated &= has_topup & (ts_card < endS_FINAL)

    # materialised incrementality... converted *only because* treated
    incremental = is_treated & incremental_potential & converts14

    state = pd.DataFrame(
        {
            "user_id": users["user_id"],
            "channel": channel, "country": users["country"], "device": device,
            "age_band": age, "doc_type": doc,
            "email_ok": email_ok, "kyc_started": kyc_started, "submitted": submitted,
            "approved": approved, "n_attempts": n_attempts, "app_version": app_version,
            "reject_reason": reason,
            "ts_signup": ts_signup, "ts_email": ts_email, "ts_kyc_start": ts_kyc_start,
            "ts_submit": ts_submit, "ts_decision": ts_decision,
            "has_topup": has_topup, "converts14": converts14 & has_topup,
            "ts_topup": ts_topup, "first_topup_amount": amount,
            "activated": activated, "ts_card": ts_card,
            "variant": variant, "redeemed": redeemed & has_topup, "incremental": incremental,
        }
    )

    truth_exp = pd.DataFrame(
        {
            "user_id": users["user_id"][in_window],
            "variant": variant[in_window],
            "u_draw": u[in_window],
            "p_control": p_control[in_window],
            "p_treatment": (p_control + delta)[in_window],
            "incremental_potential": incremental_potential[in_window],
        }
    )
    return state, cal, truth_exp


def build_funnel_events(rng: np.random.Generator, state: pd.DataFrame) -> pd.DataFrame:
    """long event table, one row per funnel event. kyc submissions get one row per attempt btw"""

    def block(mask, event, ts, **props):
        df = pd.DataFrame({"user_id": state.loc[mask, "user_id"], "event_type": event, "event_ts": ts[mask]})
        for k, v in props.items():
            df[k] = v[mask] if isinstance(v, (np.ndarray, pd.Series)) else v
        return df

    ts = {k: state[k].to_numpy() for k in ["ts_signup", "ts_email", "ts_kyc_start", "ts_submit", "ts_decision"]}
    all_true = np.ones(len(state), dtype=bool)
    blocks = [
        block(all_true, "signup_start", ts["ts_signup"]),
        block(state["email_ok"].to_numpy(), "email_verified", ts["ts_email"]),
        block(state["kyc_started"].to_numpy(), "kyc_start", ts["ts_kyc_start"]),
    ]

    #1 kyc_doc_submitted row per attempt
    sub = state[state["submitted"]]
    repsCount2 = sub["n_attempts"].to_numpy()
    attempt_gap = _lognormal_delay(rng, int(repsCount2.sum()), np.log(6.0), 1.3, "h")
    sub_ids = np.repeat(sub["user_id"].to_numpy(), repsCount2)
    attempt_no = np.concatenate([np.arange(1, r + 1) for r in repsCount2])
    base_ts = np.repeat(sub["ts_submit"].to_numpy(), repsCount2)
    cum_gap = np.where(attempt_no == 1, np.timedelta64(0, "s"), attempt_gap) * (attempt_no - 1)
    blocks.append(
        pd.DataFrame(
            {
                "user_id": sub_ids, "event_type": "kyc_doc_submitted",
                "event_ts": base_ts + cum_gap, "attempt": attempt_no,
                "doc_type": np.repeat(sub["doc_type"].to_numpy(), repsCount2),
                "app_version": np.repeat(sub["app_version"].to_numpy(), repsCount2),
            }
        )
    )

    appr = (state["submitted"] & state["approved"]).to_numpy()
    rej = (state["submitted"] & ~state["approved"]).to_numpy()
    blocks.append(block(appr, "kyc_approved", ts["ts_decision"],
                        doc_type=state["doc_type"].to_numpy(),
                        app_version=state["app_version"].to_numpy()))
    blocks.append(block(rej, "kyc_rejected", ts["ts_decision"],
                        doc_type=state["doc_type"].to_numpy(),
                        app_version=state["app_version"].to_numpy(),
                        reject_reason=state["reject_reason"].to_numpy()))

    events = pd.concat(blocks, ignore_index=True)
    events["event_ts"] = events["event_ts"].astype("datetime64[us]")
    events = events.sort_values(["event_ts", "user_id"], kind="stable").reset_index(drop=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    if "attempt" in events:
        events["attempt"] = events["attempt"].astype("Int64")
    return events
