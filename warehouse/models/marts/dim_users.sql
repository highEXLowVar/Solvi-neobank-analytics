-- one row per sign-up start: acquisition context, funnel progress, activation,
-- and experiment membership. basically the workhorse table, everything joins to this

with users as (
    select * from {{ ref('stg_users') }}
),

funnel as (
    select * from {{ ref('int_funnel_stages') }}
),

assignments as (
    select * from {{ ref('stg_experiment_assignments') }}
),

payouts as (
    select user_id, sum(amount_eur) as bonus_paid_eur
    from {{ ref('stg_incentive_payouts') }}
    group by user_id
)

select
    u.user_id,
    u.signup_ts,
    u.signup_date,
    u.signup_month,
    u.channel,
    u.country,
    u.device,
    u.age_band,
    c.cac_eur,

    f.email_verified_ts is not null                       as is_email_verified,
    f.kyc_start_ts is not null                            as is_kyc_started,
    f.first_doc_submitted_ts is not null                  as is_doc_submitted,
    f.is_kyc_approved,
    f.kyc_approved_ts,
    f.kyc_rejected_ts,
    f.n_kyc_attempts,
    f.doc_type,
    f.app_version                                         as kyc_app_version,
    f.reject_reason,

    f.first_topup_ts,
    f.first_topup_amount,
    f.first_topup_ts is not null                          as has_topup,
    f.first_card_txn_ts                                   as activation_ts,
    f.first_card_txn_ts is not null                       as is_activated,
    date_trunc('month', f.first_card_txn_ts)              as activation_month,

    a.experiment_id,
    a.variant,
    a.assigned_ts,
    coalesce(p.bonus_paid_eur, 0)                         as bonus_paid_eur

from users u
left join funnel f using (user_id)
left join assignments a using (user_id)
left join payouts p using (user_id)
left join {{ ref('stg_channels') }} c using (channel)
