-- The €10 bonus can only be paid to treatment-arm users. A payout to anyone
-- else means the campaign tooling leaked outside the experiment.

select p.user_id, a.variant
from {{ ref('stg_incentive_payouts') }} p
left join {{ ref('stg_experiment_assignments') }} a using (user_id)
where coalesce(a.variant, 'none') != 'treatment'
