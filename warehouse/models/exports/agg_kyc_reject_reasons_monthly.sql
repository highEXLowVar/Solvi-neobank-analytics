-- Monthly rejection reason mix per platform x doc type (incident fingerprint:
-- the excess is all doc_unreadable).

select
    date_trunc('month', kyc_rejected_ts) as month,
    device                               as platform,
    doc_type,
    reject_reason,
    count(*)                             as n_rejections
from {{ ref('dim_users') }}
where kyc_rejected_ts is not null
group by 1, 2, 3, 4
