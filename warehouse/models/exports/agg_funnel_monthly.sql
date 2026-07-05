-- Onboarding funnel counts per signup month x channel x device.
-- Counts are users *reaching* each stage, so each column <= the previous one.

select
    signup_month,
    channel,
    device,
    count(*)                                  as n_signups,
    count(*) filter (where is_email_verified) as n_email_verified,
    count(*) filter (where is_kyc_started)    as n_kyc_started,
    count(*) filter (where is_doc_submitted)  as n_doc_submitted,
    count(*) filter (where is_kyc_approved)   as n_kyc_approved,
    count(*) filter (where has_topup)         as n_first_topup,
    count(*) filter (where is_activated)      as n_activated
from {{ ref('dim_users') }}
group by 1, 2, 3
