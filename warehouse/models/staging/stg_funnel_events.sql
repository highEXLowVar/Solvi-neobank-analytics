select
    event_id,
    user_id,
    event_type,
    event_ts,
    cast(event_ts as date) as event_date,
    attempt,
    doc_type,
    app_version,
    reject_reason
from {{ source('raw', 'funnel_events') }}
