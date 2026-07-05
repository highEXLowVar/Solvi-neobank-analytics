-- user-month activity with the full revenue/cost decomposition.
-- grain: one row per (user, month) with any money movement, subscription or payout.
-- economics constants all come from dbt vars (see dbt_project.yml) so the
-- assumptions live in one place instead of copy-pasted across five models

with txn as (
    select
        user_id,
        txn_month as month,
        count(*)                                  as n_card_txns,
        sum(amount_eur)                           as card_spend_eur,
        sum(amount_eur) filter (where is_fx)      as fx_spend_eur
    from {{ ref('stg_card_transactions') }}
    group by 1, 2
),

topup as (
    select
        user_id,
        topup_month as month,
        count(*)        as n_topups,
        sum(amount_eur) as topup_volume_eur
    from {{ ref('stg_topups') }}
    group by 1, 2
),

subs as (
    select user_id, month, plan, mrr_eur
    from {{ ref('stg_subscriptions') }}
),

bonus as (
    select user_id, payout_month as month, sum(amount_eur) as bonus_cost_eur
    from {{ ref('stg_incentive_payouts') }}
    group by 1, 2
),

joined as (
    select
        coalesce(t.user_id, tu.user_id, s.user_id, b.user_id) as user_id,
        coalesce(t.month, tu.month, s.month, b.month)         as month,
        coalesce(t.n_card_txns, 0)     as n_card_txns,
        coalesce(t.card_spend_eur, 0)  as card_spend_eur,
        coalesce(t.fx_spend_eur, 0)    as fx_spend_eur,
        coalesce(tu.n_topups, 0)       as n_topups,
        coalesce(tu.topup_volume_eur, 0) as topup_volume_eur,
        s.plan,
        coalesce(s.mrr_eur, 0)         as subscription_eur,
        coalesce(b.bonus_cost_eur, 0)  as bonus_cost_eur
    from txn t
    full outer join topup tu on t.user_id = tu.user_id and t.month = tu.month
    full outer join subs s
        on coalesce(t.user_id, tu.user_id) = s.user_id
       and coalesce(t.month, tu.month) = s.month
    full outer join bonus b
        on coalesce(t.user_id, tu.user_id, s.user_id) = b.user_id
       and coalesce(t.month, tu.month, s.month) = b.month
),

enriched as (
    select
        j.*,
        (j.n_card_txns > 0 or j.n_topups > 0) as is_active,
        d.channel,
        d.activation_month,
        datediff('month', d.activation_month, j.month) as months_since_activation
    from joined j
    left join {{ ref('dim_users') }} d using (user_id)
)

select
    *,
    card_spend_eur * {{ var('interchange_rate') }}                     as interchange_rev_eur,
    fx_spend_eur * {{ var('fx_fee_rate') }}                            as fx_rev_eur,
    case when is_active then {{ var('other_fees_per_active_month') }} else 0 end
                                                                       as other_fees_eur,
    card_spend_eur * {{ var('interchange_rate') }}
      + fx_spend_eur * {{ var('fx_fee_rate') }}
      + case when is_active then {{ var('other_fees_per_active_month') }} else 0 end
      + subscription_eur                                               as revenue_eur,
    case when is_active then {{ var('variable_cost_per_active_month') }} else 0 end
                                                                       as variable_cost_eur,
    card_spend_eur * {{ var('interchange_rate') }}
      + fx_spend_eur * {{ var('fx_fee_rate') }}
      + case when is_active then {{ var('other_fees_per_active_month') }} else 0 end
      + subscription_eur
      - case when is_active then {{ var('variable_cost_per_active_month') }} else 0 end
      - bonus_cost_eur                                                 as contribution_margin_eur
from enriched
