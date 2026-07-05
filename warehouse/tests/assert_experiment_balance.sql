-- Sample-ratio-mismatch guard: under 50/50 randomisation the arm imbalance
-- z-statistic should be small. |z| > 4 would flag a broken randomiser
-- (p < 6e-5), which invalidates the whole experiment readout.

with counts as (
    select
        count(*) filter (where variant = 'treatment')::double as n_t,
        count(*) filter (where variant = 'control')::double   as n_c
    from {{ ref('fct_experiment_users') }}
)

select *
from counts
where abs(n_t - n_c) / sqrt((n_t + n_c) * 0.25) > 4
