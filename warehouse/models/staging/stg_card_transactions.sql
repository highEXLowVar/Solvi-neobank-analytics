select
    txn_id,
    user_id,
    txn_ts,
    date_trunc('month', txn_ts) as txn_month,
    amount_eur,
    mcc_category,
    is_fx
from {{ source('raw', 'card_transactions') }}
