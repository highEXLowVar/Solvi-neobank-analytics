"""All the knobs for the Solvi generative model - basically the answer key.

Single source of truth for the sim. The analytics layer (dbt + analysis/) must
NEVER import from here, it has to recover these numbers from the data itself
or the whole point of this project falls apart. DESIGN.md is the readable
version of this if a constant below doesn't make sense at a glance.
"""

from __future__ import annotations

import numpy as np

SEED = 797

#  time window
START_DATE = np.datetime64("2024-01-01")
END_DATE = np.datetime64("2026-05-31")  # inclusive
N_DAYS = int((END_DATE - START_DATE).astype(int)) + 1  # 882
MONTHS = np.arange(np.datetime64("2024-01"), np.datetime64("2026-06"))  # 29 periods

#acquisition
BASE_SIGNUPS_DAY0 = 95.0
GROWTH_FACTOR_TOTAL = 3.4  #multiplicative growth over the whole window
DOW_FACTORS = {0: 1.05, 1: 1.06, 2: 1.05, 3: 1.04, 4: 1.00, 5: 0.85, 6: 0.88}
JANUARY_BOOST = 1.15
ARRIVAL_NOISE_SHAPE = 16.7  #gamma shape for NegBin-style overdispersion

CHANNELS = ["organic", "referral", "paid_social", "influencer"]
CHANNEL_SHARE_START = {"organic": 0.42, "referral": 0.16, "paid_social": 0.26, "influencer": 0.16}
CHANNEL_SHARE_END = {"organic": 0.34, "referral": 0.24, "paid_social": 0.30, "influencer": 0.12}
#campaign bursts: (min_gap, max_gap, min_len, max_len, min_factor, max_factor) in days
CAMPAIGNS = {
    "paid_social": (20, 45, 10, 20, 1.5, 2.1),
    "influencer": (12, 30, 1, 3, 2.0, 3.2),
}

#assumed acquisition cost per signup-start, EUR(finance gave us these, dont ask how)
CAC = {"organic": 2.0, "referral": 12.0, "paid_social": 26.0, "influencer": 19.0}

# ---------------------------------------------------------------- demographics
COUNTRIES = ["FR", "ES", "PT", "PL", "RO"]
COUNTRY_MIX = [0.26, 0.22, 0.12, 0.24, 0.16]
IOS_SHARE = {"FR": 0.38, "ES": 0.32, "PT": 0.30, "PL": 0.22, "RO": 0.18}

AGE_BANDS = ["18-24", "25-34", "35-44", "45+"]
AGE_MIX_BY_CHANNEL = {
    "organic": [0.24, 0.38, 0.24, 0.14],
    "referral": [0.30, 0.40, 0.20, 0.10],
    "paid_social": [0.28, 0.37, 0.23, 0.12],
    "influencer": [0.46, 0.36, 0.13, 0.05],
}

DOC_TYPES = ["national_id", "passport", "driving_licence", "residence_permit"]
DOC_MIX_BY_COUNTRY = {
    "FR": [0.42, 0.30, 0.20, 0.08],
    "ES": [0.50, 0.26, 0.17, 0.07],
    "PT": [0.48, 0.27, 0.17, 0.08],
    "PL": [0.55, 0.24, 0.14, 0.07],
    "RO": [0.55, 0.25, 0.12, 0.08],
}

# ---------------------------------------------------------------- funnel rates
EMAIL_VERIFIED_BASE = {"organic": 0.93, "referral": 0.95, "paid_social": 0.87, "influencer": 0.90}
KYC_START_BASE = {"organic": 0.86, "referral": 0.90, "paid_social": 0.80, "influencer": 0.85}
DOC_SUBMIT_BASE = 0.92
DOC_SUBMIT_ANDROID_PENALTY = 0.01

# final KYC approval probability (after up to 3 attempts), by document type
KYC_APPROVAL_BY_DOC = {
    "national_id": 0.90,
    "passport": 0.93,
    "driving_licence": 0.87,
    "residence_permit": 0.82,
}
KYC_ANDROID_PENALTY = 0.01

# attempts distribution (cosmetic, conditioned on final outcome)
ATTEMPTS_IF_APPROVED = [0.78, 0.17, 0.05]
ATTEMPTS_IF_REJECTED = [0.55, 0.30, 0.15]
REJECT_REASONS = ["doc_unreadable", "data_mismatch", "expired_doc", "suspected_fraud"]
REJECT_REASON_MIX = [0.38, 0.30, 0.20, 0.12]
MANUAL_REVIEW_SHARE = 0.12

# ---- planted finding #1: Android v5.21 breaks national_id image compression ----
# yes it's intentional, no i'm not "fixing" it, that's the entire point of this repo
INCIDENT_RELEASE_DATE = np.datetime64("2025-11-10")   # android v-INC ships the bug
HOTFIX_RELEASE_DATE = np.datetime64("2026-01-08")     # partial client+server mitigation
INCIDENT_APPROVAL = 0.67   # android x national_id on affected versions, pre-hotfix
POST_HOTFIX_APPROVAL = 0.80  # residual regression, still open at end of window

# first top-up within 14 days of approval, by channel (control condition)
TOPUP_14D_BASE = {"organic": 0.70, "referral": 0.76, "paid_social": 0.54, "influencer": 0.64}
TOPUP_AGE_ADJ = {"18-24": -0.02, "25-34": 0.01, "35-44": 0.01, "45+": -0.03}
LATE_TOPUP_RATE = 0.06  # of 14d non-converters, convert on day 15-45 instead
CARD_TXN_GIVEN_TOPUP = 0.90

# ---------------------------------------------------------------- activity
ENGAGEMENT_GAMMA_SHAPE = 2.2
ENGAGEMENT_MEAN = {"organic": 12.5, "referral": 14.0, "paid_social": 9.5, "influencer": 11.0}

CHURN_BASE = {"organic": 0.045, "referral": 0.032, "paid_social": 0.085, "influencer": 0.065}
CHURN_TENURE_SHAPE = [1.6, 1.3, 1.1, 1.0]  # months 0,1,2,3+ ; *0.85 after month 12
CHURN_LATE_FACTOR = 0.85
CHURN_ENGAGEMENT_EXP = 0.35

TENURE_RAMP = [0.7, 0.9] # months 0,1 ; then 1.0
TENURE_DECAY = 0.997  # per month after ramp
SEASON_DECEMBER = 1.18
SEASON_SUMMER = 1.07 # Jun-Aug
FX_SHARE_BASE = 0.10
FX_SHARE_SUMMER_EXTRA = 0.04

TXN_AMOUNT_LOG_MEDIAN = np.log(17.0)
TXN_AMOUNT_SIGMA = 0.95
MCC_CATEGORIES = ["groceries", "restaurants", "transport", "shopping",
                  "travel", "entertainment", "utilities", "other"]
MCC_MIX = [0.24, 0.18, 0.13, 0.16, 0.07, 0.09, 0.06, 0.07]

TOPUPS_PER_MONTH = 1.4
TOPUP_AMOUNT_LOG_MEDIAN = np.log(90.0)
TOPUP_AMOUNT_SIGMA = 0.80
FIRST_TOPUP_LOG_MEDIAN = np.log(50.0)
FIRST_TOPUP_SIGMA = 0.90
TOPUP_METHODS = ["card", "bank_transfer", "apple_pay", "google_pay"]
TOPUP_METHOD_MIX = [0.46, 0.30, 0.12, 0.12]

# ------------------------------------------------------------- subsciptions
SUB_PLANS = {"plus": 3.99, "premium": 7.99}
SUB_UPGRADE_BASE = 0.009# monthly hazard, free -> paid
SUB_UPGRADE_AGE = {"18-24": 1.0, "25-34": 1.4, "35-44": 1.1, "45+": 0.7}
SUB_PLUS_SHARE = 0.70
SUB_LAPSE = 0.035  # monthly hazard, paid -> free

# -------------------------------------------------------------- experiment
EXP_ID = "topup10_2026q1"
EXP_START = np.datetime64("2026-02-02")# assignment window (KYC approval ts)
EXP_END = np.datetime64("2026-03-29")  # exclusive
EXP_UPLIFT = {"organic": 0.050, "referral": 0.035, "paid_social": 0.090, "influencer": 0.060}
# engagement multiplier applied to *ncremental converters (marginal users are low-intent)
EXP_INCREMENTAL_ENGAGEMENT = {"organic": 0.85, "referral": 0.85, "paid_social": 0.50, "influencer": 0.70}
EXP_BONUS_EUR = 10.0
EXP_MIN_TOPUP = 20.0
EXP_TREATED_THRESHOLD_AWARE = 0.85 # treated converters below €20 who bump to qualify
EXP_ACTIVITY_BOOST = [1.25, 1.12] ## months 0,1 after activation, redeemers only

# ------------------------------------------------------- app releases
ANDROID_FIRST_RELEASE = np.datetime64("2024-01-03")
IOS_FIRST_RELEASE = np.datetime64("2024-01-10")
RELEASE_GAPS = [24, 27, 23, 28, 25, 26]  # cycled, days
VERSION_ADOPTION = [0.88, 0.10, 0.02]    # latest / one behind / two behind

# economics (dbt vars too)
INTERCHANGE_RATE = 0.002
FX_FEE_RATE = 0.007
OTHER_FEES_PER_ACTIVE_MONTH = 0.35
VARIABLE_COST_PER_ACTIVE_MONTH = 0.45
