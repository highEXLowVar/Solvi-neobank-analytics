select
    topup_id,
    user_id,
    topup_ts,
    date_trunc('month', topup_ts) as topup_month,
    amount_eur,
    method
from {{ source('raw', 'topups') }}
