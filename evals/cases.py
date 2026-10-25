"""Eval cases: 17 retrieval targets, one per check in analysis/validate_truth.py,
spread over 6 investigation prompts, plus 3 noise traps.

Grading targets are the values the warehouse itself supports (what the
pipeline recovered, analysis/out/*.json), because the agent only ever sees
the warehouse. 14 of the 17 validation checks have a direct warehouse
number. The other three compare the uplift model against data/ground_truth/,
which is kept out of the warehouse on purpose, so their target here is the
warehouse-side number that carries the same conclusion:

  incremental users are low-intent       -> 60-day margin delta excl. bonus (+0.127 EUR/user)
  uplift model calibrated on average     -> overall TOPUP10 lift (5.16pp)
  model finds no profitable slice        -> 60-day margin delta net of bonus (-6.741 EUR/user)
  TOPUP10 unprofitable even for oracle   -> cheapest cost per incremental conversion, any
                                            channel (EUR 102, paid_social, vs LTV EUR 11)

That is 4 numbers for 3 checks plus the per-channel breakdown; the mapping is
listed per target below so the count of 17 is checkable.

The noise traps were picked by looking: each is a slice that moves enough
to look worrying, where the changepoint scan finds no break (LLR well under
25) or a two-proportion test is unremarkable, and where nothing was planted
in the simulator (simulator/config.py).
"""

from __future__ import annotations

TOL_REL = 0.01  # the +-1% the brief asks for

CASES = [
    {
        "id": "kyc_drop",
        "question": "Why did KYC conversion drop in recent weeks?",
        "as_of": "2025-12-15",
        "expected_action": "ROLLBACK_RELEASE",
        "expected_incident": True,
        "targets": [
            {"check": "affected segment", "label": "Affected segment", "kind": "category",
             "all_of": ["android", "national"], "value": "android x national_id"},
            {"check": "break date", "label": "Break date", "kind": "date", "value": "2025-11-10"},
        ],
        "site": True,
    },
    {
        "id": "kyc_recovery",
        "question": "Android KYC approvals for national ID documents fell late last year. What "
                    "level did they fall to, when did they recover, and to what level?",
        "as_of": None,
        "expected_action": None,  # informational: graded on retrieval only
        "expected_incident": True,
        "targets": [
            {"check": "recovery date", "label": "Recovery date", "kind": "date", "value": "2026-01-08"},
            {"check": "level during incident", "label": "Approval rate during incident",
             "kind": "number", "value": 0.686},
            {"check": "level after hotfix", "label": "Approval rate after hotfix",
             "kind": "number", "value": 0.794},
        ],
    },
    {
        "id": "topup10_rollout",
        "question": "Should we roll out the €10 activation bonus from the A/B test?",
        "as_of": None,
        "expected_action": "DO_NOT_SHIP",
        "expected_incident": None,
        "targets": [
            {"check": "uplift model calibrated on average (held-out)",
             "label": "Conversion lift (pp)", "kind": "number", "value": 5.16},
            {"check": "model finds no profitable slice; truth agrees",
             "label": "60-day margin delta net of bonus (EUR/user)", "kind": "number", "value": -6.741},
            {"check": "incremental users are low-intent",
             "label": "60-day margin delta excl. bonus (EUR/user)", "kind": "number", "value": 0.127,
             "abs_tol": 0.005},
        ],
        "site": True,
    },
    {
        "id": "topup10_targeting",
        "question": "Is there an acquisition channel where the €10 bonus is worth rolling out on "
                    "its own, even if it doesn't pay overall?",
        "as_of": None,
        "expected_action": "DO_NOT_SHIP",
        "expected_incident": None,
        "targets": [
            {"check": "uplift influencer", "label": "Lift, influencer (pp)", "kind": "number", "value": 4.08},
            {"check": "uplift organic", "label": "Lift, organic (pp)", "kind": "number", "value": 5.65},
            {"check": "uplift paid_social", "label": "Lift, paid social (pp)", "kind": "number", "value": 5.74},
            {"check": "uplift referral", "label": "Lift, referral (pp)", "kind": "number", "value": 5.41},
            {"check": "TOPUP10 unprofitable even for an oracle",
             "label": "Cheapest cost per incremental conversion (EUR)", "kind": "number", "value": 102.2},
        ],
    },
    {
        "id": "paid_social",
        "question": "Is the paid social acquisition channel profitable?",
        "as_of": None,
        "expected_action": "PAUSE_ACQUISITION_CHANNEL",
        "expected_incident": None,
        # decision evidence, reported separately from the 17
        "extra_targets": [
            {"label": "Paid social CAC per activated user (EUR)", "kind": "number", "value": 97.96},
            {"label": "Paid social 36-month LTV (EUR)", "kind": "number", "value": 11.2},
        ],
        "targets": [],
        "site": True,
    },
    {
        "id": "channel_churn",
        "question": "What is the steady-state monthly churn for each acquisition channel, and does "
                    "any channel need its spend cut?",
        "as_of": None,
        "expected_action": "PAUSE_ACQUISITION_CHANNEL",
        "expected_incident": None,
        "targets": [
            {"check": "monthly churn influencer", "label": "Monthly churn, influencer", "kind": "number", "value": 0.0676},
            {"check": "monthly churn organic", "label": "Monthly churn, organic", "kind": "number", "value": 0.0412},
            {"check": "monthly churn paid_social", "label": "Monthly churn, paid social", "kind": "number", "value": 0.0766},
            {"check": "monthly churn referral", "label": "Monthly churn, referral", "kind": "number", "value": 0.0326},
        ],
    },
    # ---- noise traps: the right answer is NO_ACTION_NOISE -------------------
    {
        "id": "noise_ios_licence_week",
        "question": "iOS KYC approvals for driving licences dropped to about 80% in the week of "
                    "13 April 2026, from about 89% the week before. Is the iOS app broken?",
        "as_of": None,
        "expected_action": "NO_ACTION_NOISE",
        "expected_incident": False,
        "noise": True,
        "targets": [],
        # 70/87 vs 70/79: z = -1.44, p = 0.15; full-series changepoint LLR 3.1
    },
    {
        "id": "noise_organic_activation",
        "question": "Organic sign-up activation fell from 42.1% in March 2026 to 39.5% in April. "
                    "Did something break in onboarding?",
        "as_of": None,
        "expected_action": "NO_ACTION_NOISE",
        "expected_incident": False,
        "noise": True,
        "targets": [],
        # p = 0.04 in isolation, but January and February sat at 39.2% and 40.0%:
        # April is back inside the normal range, and nothing was planted there
    },
    {
        "id": "noise_ro_card_spend",
        "question": "Card spend per active user in Romania fell about 2% between March and May "
                    "2026. Is Romanian card usage declining?",
        "as_of": None,
        "expected_action": "NO_ACTION_NOISE",
        "expected_incident": False,
        "noise": True,
        "targets": [],
        # EUR 319.55 -> 312.60, standard errors about 3.6 each: z = 1.4
    },
]

N_CHECKS = sum(len(c["targets"]) for c in CASES)
assert N_CHECKS == 17, N_CHECKS
