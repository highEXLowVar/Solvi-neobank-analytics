-- one row per KYC document submission attempt, with user context and the
-- user's final KYC outcome bolted on. this is what the incident analysis runs on

select
    e.event_id,
    e.user_id,
    e.event_ts,
    e.event_date,
    e.attempt,
    e.doc_type,
    e.app_version,
    u.device                as platform,
    u.channel,
    u.country,
    f.is_kyc_approved,
    f.reject_reason,
    coalesce(f.kyc_approved_ts, f.kyc_rejected_ts) as decision_ts
from {{ ref('stg_funnel_events') }} e
join {{ ref('stg_users') }} u using (user_id)
join {{ ref('int_funnel_stages') }} f using (user_id)
where e.event_type = 'kyc_doc_submitted'
