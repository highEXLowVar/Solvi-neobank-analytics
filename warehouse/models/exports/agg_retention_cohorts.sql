-- classic retention triangle - of users activated in cohort month M (by channel),
-- how many made at least one card transaction k months later.
-- only fully-observed (cohort, k) cells get emitted, half-observed ones would
-- just drag the average down and make retention look worse than it is

with cohorts as (
    select activation_month, channel, count(*) as cohort_size
    from {{ ref('dim_users') }}
    where is_activated
    group by 1, 2
),

active as (
    select
        d.activation_month,
        d.channel,
        f.months_since_activation,
        count(distinct f.user_id) as n_active
    from {{ ref('fct_activity_monthly') }} f
    join {{ ref('dim_users') }} d using (user_id)
    where f.n_card_txns > 0
      and d.is_activated
      and f.months_since_activation >= 0
    group by 1, 2, 3
)

select
    c.activation_month,
    c.channel,
    a.months_since_activation,
    c.cohort_size,
    a.n_active,
    a.n_active::double / c.cohort_size as retention
from cohorts c
join active a using (activation_month, channel)
where c.activation_month + to_months(a.months_since_activation::int)
      <= date '{{ var("warehouse_end_month") }}'
