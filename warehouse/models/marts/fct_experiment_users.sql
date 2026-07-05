-- one row per experiment subject with primary/secondary outcomes.
--
-- outcome windows are anchored on assignment (= KYC approval) so both arms
-- get measured identically, dont anchor on signup or the comparison is unfair:
--   conv_14d      first top-up within 14 days (primary)
--   margin_60d    contribution margin in the 60 days after assignment, net of
--                 the EUR10 bonus (secondary / guardrail)

with assignments as (
    select * from {{ ref('stg_experiment_assignments') }}
),

users as (
    select * from {{ ref('dim_users') }}
),

txn_60d as (
    select
        a.user_id,
        count(*)                             as n_txns_60d,
        sum(t.amount_eur)                    as spend_60d,
        sum(t.amount_eur * t.is_fx::int)     as fx_spend_60d,
        count(distinct t.txn_month)          as active_months_60d
    from assignments a
    join {{ ref('stg_card_transactions') }} t
      on t.user_id = a.user_id
     and t.txn_ts >= a.assigned_ts
     and t.txn_ts < a.assigned_ts + interval 60 day
    group by 1
),

subs_60d as (
    select a.user_id, sum(s.mrr_eur) as sub_rev_60d
    from assignments a
    join {{ ref('stg_subscriptions') }} s
      on s.user_id = a.user_id
     and s.month >= date_trunc('month', a.assigned_ts)
     and s.month < a.assigned_ts + interval 60 day
    group by 1
),

bonus_60d as (
    select a.user_id, sum(p.amount_eur) as bonus_60d
    from assignments a
    join {{ ref('stg_incentive_payouts') }} p
      on p.user_id = a.user_id
     and p.payout_ts < a.assigned_ts + interval 60 day
    group by 1
)

select
    a.user_id,
    a.experiment_id,
    a.variant,
    a.assigned_ts,
    a.assigned_date,
    u.channel,
    u.country,
    u.device,
    u.age_band,
    u.signup_month,

    -- primary outcome
    coalesce(u.first_topup_ts < a.assigned_ts + interval 14 day, false) as conv_14d,
    u.first_topup_amount,
    u.bonus_paid_eur > 0                                                as redeemed,

    -- secondary outcomes (60-day window)
    coalesce(t.n_txns_60d, 0)        as n_txns_60d,
    coalesce(t.spend_60d, 0)         as spend_60d,
    coalesce(t.spend_60d, 0) * {{ var('interchange_rate') }}
      + coalesce(t.fx_spend_60d, 0) * {{ var('fx_fee_rate') }}
      + coalesce(t.active_months_60d, 0) * ({{ var('other_fees_per_active_month') }} - {{ var('variable_cost_per_active_month') }})
      + coalesce(s.sub_rev_60d, 0)
      - coalesce(b.bonus_60d, 0)     as margin_60d_eur,
    coalesce(b.bonus_60d, 0)         as bonus_cost_eur

from assignments a
join users u using (user_id)
left join txn_60d t using (user_id)
left join subs_60d s using (user_id)
left join bonus_60d b using (user_id)
