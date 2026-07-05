select
    user_id,
    payout_ts,
    date_trunc('month', payout_ts) as payout_month,
    amount_eur,
    campaign
from {{ source('raw', 'incentive_payouts') }}
