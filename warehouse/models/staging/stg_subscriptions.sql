select
    user_id,
    date_trunc('month', month) as month,
    plan,
    mrr_eur
from {{ source('raw', 'subscriptions') }}
