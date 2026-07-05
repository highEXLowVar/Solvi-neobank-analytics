-- materialised: CSV-backed views would resolve their relative path against the
-- caller's cwd, breaking ad-hoc queries from outside warehouse/
{{ config(materialized='table') }}

select
    platform,
    version,
    cast(release_date as date) as release_date
from {{ source('raw', 'app_releases') }}
