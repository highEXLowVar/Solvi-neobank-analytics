-- A user cannot reach stage k+1 without stage k: within every monthly segment
-- the funnel counts must be non-increasing. Any violation is a modelling bug.

select *
from {{ ref('agg_funnel_monthly') }}
where n_email_verified > n_signups
   or n_kyc_started    > n_email_verified
   or n_doc_submitted  > n_kyc_started
   or n_kyc_approved   > n_doc_submitted
   or n_first_topup    > n_kyc_approved
   or n_activated      > n_first_topup
