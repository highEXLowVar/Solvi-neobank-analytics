# Solvi: how the simulator works

This is the design doc for the data generator. I'm writing it out in full because the
whole project depends on it: the analysis code is not allowed to read any of these numbers,
it has to recover them from the data on its own, and `analysis/validate_truth.py` checks
how close it gets. So this file is the answer key.

Simulation window: 2024-01-01 to 2026-05-31 (882 days). Random seed: 42. All money in EUR.

## 1. Acquisition

New sign-ups per day follow a smooth growth curve with a few realistic wrinkles: a
weekly pattern (slower on weekends), a January "new year, new bank account" bump,
campaign bursts on the paid channels, and some random noise so the daily counts aren't
suspiciously clean.

```
base(t)  = 95 * exp(ln(3.4) * t / 881)     # grows from ~95/day to ~320/day
day(t)   = base(t) * weekday(t) * january(t)
per channel: negative-binomial noise around day(t) * channel_share(t) * campaign(t)
```

| channel | share (start to end) | behaviour | assumed cost per signup |
|---|---|---|---|
| organic | 42% to 34% | smooth | €2 |
| referral | 16% to 24% | grows as the active base grows (word of mouth) | €12 |
| paid_social | 26% to 30% | bursty, 10 to 20 day campaigns | €26 |
| influencer | 16% to 12% | spiky, 1 to 3 day bumps | €19 |

The acquisition costs are assumptions I set in config, not something the simulator
produces. They stand in for the numbers a finance team would hand you, and they feed the
unit-economics analysis later.

User attributes:

- country: FR 26%, ES 22%, PT 12%, PL 24%, RO 16%
- device: iOS share varies by country (FR 38%, ES 32%, PT 30%, PL 22%, RO 18%), so about
  70% of users are on Android overall
- age band: 18-24, 25-34, 35-44, 45+; influencer and referral skew younger

## 2. Onboarding funnel

A user moves through these stages, each with a timestamp:

```
signup_start -> email_verified -> kyc_start -> kyc_doc_submitted (1 to 3 attempts)
  -> kyc_approved or kyc_rejected -> first top-up -> first card transaction
```

Each stage has a base pass-through rate, nudged up or down by channel, device, and country
(all in `simulator/config.py`):

| stage | base rate | biggest driver |
|---|---|---|
| email verified, given signup | ~92% | channel quality (referral .95, paid .87) |
| kyc started, given verified | ~86% | channel (referral .90, paid_social .80) |
| doc submitted, given kyc start | ~92% | device (1pp lower on Android) |
| approved, given submitted | ~89% | document type, platform, and the incident below |
| first top-up within 14 days, given approved | ~66% | channel (referral .76 down to paid_social .54) |
| first card transaction, given top-up | ~90% | none |

Document types: national_id 47%, passport 28%, driving_licence 17%, residence_permit 8%.
Rejection reasons: doc_unreadable, data_mismatch, expired_doc, suspected_fraud. Approved
users take 1, 2, or 3 attempts (78% / 17% / 5%). Most decisions are automated and quick;
12% get routed to manual review and take a day or two.

### Planted problem 1: the KYC incident

The Android release that ships on 2025-11-10 (the generator numbers it 5.27) has a bug
that breaks image compression, but only for national-ID photos taken on Android. For that
one slice of users:

- approval drops from 89% to 67%
- the extra rejections are all `doc_unreadable` (a broken upload, not real fraud)
- a hotfix ships on 2026-01-08 (version 5.29) and brings approval back to 80%, not the full
  89%, so a 9-point gap is still open at the end of the window

Every other slice is untouched. The bug only bites users who happen to be on the buggy
versions, so users still on older releases are fine, which makes the drop look gradual
rather than instant. The analysis has to find the affected slice, find the date, tie it to
the release calendar (`app_releases`), and add up the damage.

## 3. Life after activation

Once a user activates, I give them a hidden engagement level: a per-user rate drawn from a
Gamma distribution, equal to their average card transactions per month, fixed for life.

| channel | mean transactions/mo | base monthly churn |
|---|---|---|
| organic | 12.5 | 4.5% |
| referral | 14.0 | 3.2% |
| paid_social | 9.5 | 8.5% |
| influencer | 11.0 | 6.5% |

- Monthly transaction counts are Poisson around that rate, ramped up over the first two
  months and gently decaying after, with a December spike and a summer travel bump (and
  more FX spending in summer).
- Churn is a monthly coin flip: the base rate above, higher in the first couple of months,
  lower after a year, and lower for more engaged users. Once a user churns they're gone for
  good, and the warehouse never sees a "churned" flag, it just sees the activity stop.
- Card transaction amounts are lognormal (median €17). About 10% are FX, more in summer.
- Top-ups happen a bit more than once a month while active (median €90).

### Planted problem 2: channel quality

paid_social users have both the lowest engagement (9.5 transactions/mo) and the highest
churn (8.5%/mo), so at €26 to acquire they never pay back. referral is the opposite (€12,
3.2% churn) and pays back well within a year. You can only see this by combining retention,
margin, and acquisition cost, which is the point of finding 2.

## 4. Revenue and margin (per user, per month)

```
interchange   = 0.20% of card spend        (the EU cap for consumer cards)
fx_fee        = 0.70% of FX spend
other_fees    = €0.35 per active month      (ATM, card bits, blended)
subscription  = €3.99 (Plus) or €7.99 (Premium) per subscribed month
variable_cost = €0.45 per active month      (infra, support, processing; an assumption)
margin        = interchange + fx + other + subscription - variable_cost
```

Subscriptions are a simple monthly state machine: free users upgrade at about 0.9% a month
(higher for 25-34s and more engaged users, 70% pick Plus), and subscribers lapse at 3.5% a
month. In steady state about 8 to 9% of active users are paying, which puts blended revenue
per active user around €1.7 to €1.9 a month. That's deliberately low, like a real free-tier
neobank.

## 5. The TOPUP10 experiment

Design: every user approved between 2026-02-02 and 2026-03-29 is randomly put in control or
treatment, 50/50, at the moment of approval. Treatment is "€10 credit if your first top-up
is at least €20 within 14 days."

To make the truth knowable, I draw one random number per user and use it for both arms. So
each user has a defined outcome whether or not they got the bonus, which means I know
exactly who the bonus actually moved (the "incremental" users):

| channel | control conversion | lift from bonus | engagement of the incremental users |
|---|---|---|---|
| organic | 70% | +5.0pp | 0.85x of the channel mean |
| referral | 76% | +3.5pp | 0.85x |
| influencer | 64% | +6.0pp | 0.70x |
| paid_social | 54% | +9.0pp | 0.50x |

The asymmetry is the whole trap. The bonus moves the needle most in the channels where the
extra users are worth the least. And everyone who would have converted anyway (the
always-takers) still pockets the €10, which is the cost that a naive read of the test
forgets. Treated converters also spend a bit more for two months (they're spending the free
money), which flatters any short-window revenue metric, so the analysis has to use margin
after subtracting the bonus and look past that window.

### Planted problem 3: the incentive doesn't pay back

I tuned this so it reads as an obvious statistical win that is actually an economic loss:

- Cost per extra conversion is dominated by paying the always-takers: roughly €150 in
  organic (high baseline, small lift) versus about €66 in paid_social (low baseline, big
  lift).
- Value per extra user goes the other way: an organic incremental user is worth about
  €0.95/mo for ~22 months (~€21), a paid_social one about €0.35/mo for ~9 months (~€3). So
  the users who respond most are worth least, and you can't fix it by targeting the
  "responsive" channel.
- Net: it never pays back in any channel at these margins. The right call is to not ship it
  even though the p-value is tiny, and to redesign the reward (trigger on the first card
  transaction instead, or give something that costs Solvi very little, like fee-free FX).

## 6. What comes out

`data/raw/` (the only thing the warehouse is allowed to read):

| file | one row per | approx rows |
|---|---|---|
| users.parquet | sign-up start | 181k |
| funnel_events.parquet | funnel event (KYC submits: one per attempt) | 782k |
| card_transactions.parquet | card transaction | 6.1M |
| topups.parquet | top-up | 766k |
| subscriptions.parquet | subscribed user-month | 30k |
| experiment_assignments.parquet | experiment subject | 10.2k |
| incentive_payouts.parquet | bonus paid out | 3.5k |
| app_releases.csv | app release | ~70 |
| channels.csv | channel (with its assumed cost) | 4 |

`data/ground_truth/` (validation only, never read by dbt or the analyses): each user's
hidden engagement and churn draws, the per-user experiment counterfactuals, and the
incident settings.

## 7. Things I left out on purpose

- Money isn't strictly conserved (top-ups don't have to exactly fund spending), and I don't
  model account balances.
- Churn is permanent. Real users come back; mine don't.
- There's only one experiment, and it randomises perfectly. No sample-ratio problems, no
  contamination between arms.
- Acquisition cost and variable cost are fixed assumptions, not simulated.
- No fraud, chargebacks, interchange that varies by merchant type, or country-specific
  regulation beyond the document mix.

Every one of these is a deliberate simplification. They keep the generator small enough to
fully understand while leaving in the analytical problems the project is actually about.
