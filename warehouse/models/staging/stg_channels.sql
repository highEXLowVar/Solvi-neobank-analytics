-- materialised for the same reason as stg_app_releases
{{ config(materialized='table') }}

select
    channel,
    cac_eur
from {{ source('raw', 'channels') }}
