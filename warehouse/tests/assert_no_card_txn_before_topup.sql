-- Product invariant: you cannot spend before money is on the account, so the
-- first card transaction must come after the first top-up.

select user_id, first_topup_ts, first_card_txn_ts
from {{ ref('int_funnel_stages') }}
where first_card_txn_ts is not null
  and (first_topup_ts is null or first_card_txn_ts < first_topup_ts)
