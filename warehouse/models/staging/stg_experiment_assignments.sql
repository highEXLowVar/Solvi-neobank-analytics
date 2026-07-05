select
    user_id,
    experiment_id,
    variant,
    assigned_ts,
    cast(assigned_ts as date) as assigned_date
from {{ source('raw', 'experiment_assignments') }}
