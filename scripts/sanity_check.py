"""quick raw-data sanity checks - did the planted effects actually show up or not"""

import duckdb

con = duckdb.connect()  # just reads parquet directly, no need for the warehouse here

print("=== KYC approval, android x national_id, by month ===")
print(con.sql("""
    with sub as (
        select user_id, max(event_ts) ts, arg_max(doc_type, event_ts) doc_type
        from 'data/raw/funnel_events.parquet'
        where event_type = 'kyc_doc_submitted' group by user_id
    ),
    outcome as (
        select user_id, max(event_type = 'kyc_approved') approved
        from 'data/raw/funnel_events.parquet'
        where event_type in ('kyc_approved','kyc_rejected') group by user_id
    )
    select date_trunc('month', sub.ts) m,
           round(avg(case when u.device='android' and sub.doc_type='national_id' then approved::int end),3) android_nid,
           round(avg(case when not (u.device='android' and sub.doc_type='national_id') then approved::int end),3) rest
    from sub join outcome using(user_id) join 'data/raw/users.parquet' u using(user_id)
    where sub.ts >= '2025-08-01'
    group by 1 order by 1
""").df().to_string(index=False))

print("\n=== reject reasons android x national_id, Nov 10 - Jan 8 ===")
print(con.sql("""
    select reject_reason, count(*) n
    from 'data/raw/funnel_events.parquet' e join 'data/raw/users.parquet' u using(user_id)
    where event_type='kyc_rejected' and u.device='android' and e.doc_type='national_id'
      and event_ts between '2025-11-10' and '2026-01-08'
    group by 1 order by 2 desc
""").df().to_string(index=False))

print("\n=== experiment: balance and conversion by arm ===")
print(con.sql("""
    with first_topup as (select user_id, min(topup_ts) ts from 'data/raw/topups.parquet' group by 1)
    select a.variant, count(*) n,
           round(avg(case when ft.ts <= a.assigned_ts + interval 14 day then 1 else 0 end), 4) conv14
    from 'data/raw/experiment_assignments.parquet' a
    left join first_topup ft using(user_id)
    group by 1 order by 1
""").df().to_string(index=False))

print("\n=== experiment uplift by channel ===")
print(con.sql("""
    with first_topup as (select user_id, min(topup_ts) ts from 'data/raw/topups.parquet' group by 1)
    select u.channel,
           count(*) filter (variant='control') n_c,
           round(avg(case when ft.ts <= a.assigned_ts + interval 14 day then 1.0 else 0 end) filter (variant='control'), 3) conv_c,
           round(avg(case when ft.ts <= a.assigned_ts + interval 14 day then 1.0 else 0 end) filter (variant='treatment'), 3) conv_t
    from 'data/raw/experiment_assignments.parquet' a
    join 'data/raw/users.parquet' u using(user_id)
    left join first_topup ft using(user_id)
    group by 1 order by 1
""").df().to_string(index=False))

print("\n=== month-3 retention by channel (activated H1-2025 cohorts) ===")
print(con.sql("""
    with act as (
        select t.user_id, u.channel, min(txn_ts) a_ts
        from 'data/raw/card_transactions.parquet' t join 'data/raw/users.parquet' u using(user_id)
        group by 1, 2
    )
    select channel,
           count(*) n,
           round(avg(exists(select 1 from 'data/raw/card_transactions.parquet' t
                     where t.user_id = act.user_id
                       and t.txn_ts between act.a_ts + interval 3 month and act.a_ts + interval 4 month)::int), 3) ret_m3,
           round(avg(exists(select 1 from 'data/raw/card_transactions.parquet' t
                     where t.user_id = act.user_id
                       and t.txn_ts between act.a_ts + interval 9 month and act.a_ts + interval 10 month)::int), 3) ret_m9
    from act
    where a_ts between '2025-01-01' and '2025-07-01'
    group by 1 order by 1
""").df().to_string(index=False))

print("\n=== monthly revenue per active user (back-of-envelope ARPU) ===")
print(con.sql("""
    with m as (
        select date_trunc('month', txn_ts) mo, user_id,
               sum(amount_eur)*0.002 + sum(amount_eur * is_fx::int)*0.007 + 0.35 as rev
        from 'data/raw/card_transactions.parquet' group by 1,2
    ),
    s as (select month mo, user_id, mrr_eur from 'data/raw/subscriptions.parquet')
    select round(avg(coalesce(m.rev,0) + coalesce(s.mrr_eur,0)), 3) arpu
    from m full join s using(mo, user_id)
""").df().to_string(index=False))
