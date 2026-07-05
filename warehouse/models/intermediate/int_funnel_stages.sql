-- one row per user with the timestamp of each onboarding stage reached.
-- first top-up / first card txn come from the money tables, NOT the funnel
-- events - activation only counts when actual money moves, not before

with events as (
    select * from {{ ref('stg_funnel_events') }}
),

stages as (
    select
        user_id,
        min(event_ts) filter (where event_type = 'signup_start')      as signup_ts,
        min(event_ts) filter (where event_type = 'email_verified')    as email_verified_ts,
        min(event_ts) filter (where event_type = 'kyc_start')         as kyc_start_ts,
        min(event_ts) filter (where event_type = 'kyc_doc_submitted') as first_doc_submitted_ts,
        count(*) filter (where event_type = 'kyc_doc_submitted')      as n_kyc_attempts,
        max(doc_type)                                                 as doc_type,
        arg_max(app_version, event_ts)
            filter (where event_type = 'kyc_doc_submitted')           as app_version,
        min(event_ts) filter (where event_type = 'kyc_approved')      as kyc_approved_ts,
        min(event_ts) filter (where event_type = 'kyc_rejected')      as kyc_rejected_ts,
        max(reject_reason)                                            as reject_reason
    from events
    group by user_id
),

first_topup as (
    select
        user_id,
        min(topup_ts)                 as first_topup_ts,
        arg_min(amount_eur, topup_ts) as first_topup_amount
    from {{ ref('stg_topups') }}
    group by user_id
),

first_card as (
    select user_id, min(txn_ts) as first_card_txn_ts
    from {{ ref('stg_card_transactions') }}
    group by user_id
)

select
    s.*,
    coalesce(s.kyc_approved_ts is not null, false) as is_kyc_approved,
    ft.first_topup_ts,
    ft.first_topup_amount,
    fc.first_card_txn_ts
from stages s
left join first_topup ft using (user_id)
left join first_card fc using (user_id)
