-- average contribution margin per activated user at each tenure month, by
-- channel. denominator is EVERY user observable at that tenure (churned users
-- count as zero, not dropped) so cumsum(avg_margin_eur) is directly comparable
-- to CAC. get this denominator wrong and the whole payback calc lies to you.

with observable as (
    select
        channel,
        k.months_since_activation,
        count(*) as n_observable
    from {{ ref('dim_users') }} d
    cross join (
        select unnest(range(0, 30)) as months_since_activation
    ) k
    where d.is_activated
      and d.activation_month + to_months(k.months_since_activation::int)
          <= date '{{ var("warehouse_end_month") }}'
    group by 1, 2
),

margin as (
    select
        channel,
        months_since_activation,
        sum(contribution_margin_eur) as margin_sum_eur
    from {{ ref('fct_activity_monthly') }}
    where months_since_activation >= 0
    group by 1, 2
)

select
    o.channel,
    o.months_since_activation,
    o.n_observable,
    coalesce(m.margin_sum_eur, 0)                  as margin_sum_eur,
    coalesce(m.margin_sum_eur, 0) / o.n_observable as avg_margin_eur,
    c.cac_eur
from observable o
left join margin m using (channel, months_since_activation)
left join {{ ref('stg_channels') }} c using (channel)
where o.n_observable > 0
