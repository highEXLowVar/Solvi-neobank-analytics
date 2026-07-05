select
    user_id,
    signup_ts,
    cast(signup_ts as date)            as signup_date,
    date_trunc('month', signup_ts)     as signup_month,
    channel,
    country,
    device,
    age_band
from {{ source('raw', 'users') }}
