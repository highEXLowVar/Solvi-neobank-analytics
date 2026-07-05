-- daily KYC decisions split by platform x document type - this is the table
-- where the v5.21 incident just jumps out at you, no stats needed to see it

select
    cast(coalesce(kyc_approved_ts, kyc_rejected_ts) as date) as decision_date,
    device                                                    as platform,
    doc_type,
    count(*)                                                  as n_decisions,
    count(*) filter (where is_kyc_approved)                   as n_approved
from {{ ref('dim_users') }}
where is_doc_submitted
  and coalesce(kyc_approved_ts, kyc_rejected_ts) is not null
group by 1, 2, 3
