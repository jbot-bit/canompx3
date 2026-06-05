"""AUDIT 04 - DST INTEGRITY: do stored orb_outcomes windows match the canonical resolver?
Read-only. Checks the entry_ts of the FIRST possible entry lands inside orb_utc_window's [start, ...)
and that the ORB start moved correctly across DST boundaries (proves canonical resolver was used)."""
import os, duckdb
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from pipeline.dst import orb_utc_window, is_us_dst

con = duckdb.connect(os.environ["DUCKDB_PATH"], read_only=True)
BRIS = ZoneInfo("Australia/Brisbane")

# For a DST-shifting session (CME_REOPEN) check that the EARLIEST entry_ts per day
# tracks the resolver. CME_REOPEN: winter 09:00 Bris (23:00 UTC prev), summer 08:00 Bris (22:00 UTC prev).
print("="*100)
print("AUDIT 04 - DST INTEGRITY (stored data vs canonical orb_utc_window)")
print("="*100)

# Sample MNQ CME_REOPEN: a winter day and a summer day, 5-min ORB.
# We can't see the ORB window directly in orb_outcomes (it stores ENTRY not the ORB box),
# but entry_ts must be >= ORB window end (entry only after breakout confirmed).
# More robust: compare the per-day MIN(entry_ts) hour-of-day in Brisbane between winter & summer.
print("\n[A] MNQ CME_REOPEN: min entry_ts Brisbane HH:MM, winter vs summer days (E2, 5m, rr2.0)")
print("    Canonical: ORB starts 09:00 Bris winter / 08:00 Bris summer -> entries cluster AFTER that.")
rows = con.execute("""
    SELECT trading_day,
           MIN(entry_ts) AS first_entry_utc
    FROM orb_outcomes
    WHERE symbol='MNQ' AND orb_label='CME_REOPEN' AND orb_minutes=5
      AND entry_model='E2' AND rr_target=2.0 AND entry_ts IS NOT NULL
      AND trading_day IN (DATE '2025-01-15', DATE '2025-07-15',
                          DATE '2024-01-17', DATE '2024-07-17',
                          DATE '2023-01-18', DATE '2023-07-19')
    GROUP BY trading_day ORDER BY trading_day
""").fetchall()
for td, fe in rows:
    if fe is None: 
        print(f"    {td}  (no entry)"); continue
    fe_utc = fe if fe.tzinfo else fe.replace(tzinfo=timezone.utc)
    fb = fe_utc.astimezone(BRIS)
    dst = is_us_dst(td)
    # canonical ORB start for that day
    us, ue = orb_utc_window(td, 'CME_REOPEN', 5)
    ub = us.astimezone(BRIS)
    season = "summer(EDT)" if dst else "winter(EST)"
    ok = "OK" if fb >= ub else "!! BEFORE ORB START"
    print(f"    {td} {season:<12} canon_ORB_start_Bris={ub.strftime('%H:%M')}  first_entry_Bris={fb.strftime('%H:%M')}  {ok}")

print("\n[B] FIXED-Brisbane session TOKYO_OPEN should NOT shift across DST (always 10:00 Bris).")
rows = con.execute("""
    SELECT trading_day, MIN(entry_ts) AS fe
    FROM orb_outcomes
    WHERE symbol='MNQ' AND orb_label='TOKYO_OPEN' AND orb_minutes=5
      AND entry_model='E2' AND rr_target=2.0 AND entry_ts IS NOT NULL
      AND trading_day IN (DATE '2025-01-15', DATE '2025-07-15')
    GROUP BY trading_day ORDER BY trading_day
""").fetchall()
for td, fe in rows:
    fe_utc = fe if fe.tzinfo else fe.replace(tzinfo=timezone.utc)
    fb = fe_utc.astimezone(BRIS)
    us,ue = orb_utc_window(td,'TOKYO_OPEN',5); ub=us.astimezone(BRIS)
    print(f"    {td}  canon_ORB_start_Bris={ub.strftime('%H:%M')}  first_entry_Bris={fb.strftime('%H:%M')}")

print("\n[C] Duplicate-event check: any two catalog sessions resolving to SAME Bris time on BOTH a winter & summer day?")
from pipeline.dst import SESSION_CATALOG, DYNAMIC_ORB_RESOLVERS
for d in (date(2025,1,15), date(2025,7,15)):
    seen = {}
    for label in SESSION_CATALOG:
        t = DYNAMIC_ORB_RESOLVERS[label](d)
        seen.setdefault(t, []).append(label)
    dups = {t:ls for t,ls in seen.items() if len(ls)>1}
    print(f"    {d}: collisions={dups if dups else 'none'}")

print("\n[D] entry_ts timezone storage check (are they tz-aware UTC?)")
r = con.execute("SELECT entry_ts, ts_outcome FROM orb_outcomes WHERE entry_ts IS NOT NULL LIMIT 3").fetchall()
for row in r:
    print(f"    entry_ts={row[0]!r} tzinfo={getattr(row[0],'tzinfo',None)}")
