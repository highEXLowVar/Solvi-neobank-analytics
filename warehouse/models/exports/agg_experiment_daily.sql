-- Daily assignment counts and matured primary-outcome conversions per arm.
-- Used for the sequential-monitoring ("peeking") visual.

select
    assigned_date,
    variant,
    count(*)                        as n_assigned,
    count(*) filter (where conv_14d) as n_converted_14d
from {{ ref('fct_experiment_users') }}
group by 1, 2
