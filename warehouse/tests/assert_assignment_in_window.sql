-- All TOPUP10 assignments must fall inside the registered experiment window
-- (2026-02-02 .. 2026-03-29). Out-of-window assignments would mean the feature
-- flag outlived the experiment.

select *
from {{ ref('stg_experiment_assignments') }}
where assigned_ts < timestamp '2026-02-02'
   or assigned_ts >= timestamp '2026-03-29'
