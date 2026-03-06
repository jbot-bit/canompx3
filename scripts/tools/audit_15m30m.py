#!/usr/bin/env python3
"""Investigate what happened to 15m/30m validated strategies."""

import sys

sys.stdout.reconfigure(encoding="utf-8")
import duckdb

con = duckdb.connect("gold.db", read_only=True)

print("=" * 70)
print("15m/30m INVESTIGATION — WHAT HAPPENED?")
print("=" * 70)

# 1. How many experimental strategies exist at each aperture?
for om in [5, 15, 30]:
    r = con.execute(f"""
        SELECT instrument, COUNT(*) as total,
               SUM(CASE WHEN sample_size >= 50 THEN 1 ELSE 0 END) as n_ge50,
               SUM(CASE WHEN expectancy_r > 0 THEN 1 ELSE 0 END) as n_positive,
               SUM(CASE WHEN sharpe_ratio > 0.15 THEN 1 ELSE 0 END) as n_sharpe_ok
        FROM experimental_strategies
        WHERE orb_minutes = {om} AND instrument IN ('MGC','MNQ','MES','M2K')
        GROUP BY instrument ORDER BY instrument
    """).fetchall()
    print(f"\n--- {om}m experimental strategies ---")
    print(f"  Inst | Total | N>=50 | ExpR>0 | Sharpe>0.15")
    for row in r:
        print(f"  {row[0]}  | {row[1]:5,} | {row[2]:5} | {row[3]:6} | {row[4]:11}")

# 2. Best 15m strategies
print("\n\n--- Best 15m experimental (N>=50, by Sharpe) ---")
r = con.execute("""
    SELECT strategy_id, instrument, orb_label, entry_model, sample_size,
           ROUND(win_rate, 3) as wr, ROUND(expectancy_r, 3) as avgr,
           ROUND(sharpe_ratio, 3) as sharpe
    FROM experimental_strategies
    WHERE orb_minutes = 15 AND instrument IN ('MGC','MNQ','MES','M2K')
          AND sample_size >= 50
    ORDER BY sharpe_ratio DESC
    LIMIT 15
""").fetchdf()
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
print(r.to_string(index=False))

# 3. Best 30m strategies
print("\n--- Best 30m experimental (N>=50, by Sharpe) ---")
r = con.execute("""
    SELECT strategy_id, instrument, orb_label, entry_model, sample_size,
           ROUND(win_rate, 3) as wr, ROUND(expectancy_r, 3) as avgr,
           ROUND(sharpe_ratio, 3) as sharpe
    FROM experimental_strategies
    WHERE orb_minutes = 30 AND instrument IN ('MGC','MNQ','MES','M2K')
          AND sample_size >= 50
    ORDER BY sharpe_ratio DESC
    LIMIT 15
""").fetchdf()
print(r.to_string(index=False))

# 4. Were there EVER 15/30m validated strategies? Check if they were purged
print("\n\n--- CHECK: Are there ANY 15m/30m in validated_setups (any status)? ---")
r = con.execute("""
    SELECT orb_minutes, status, COUNT(*) FROM validated_setups
    WHERE orb_minutes IN (15, 30)
    GROUP BY orb_minutes, status
""").fetchall()
if r:
    for row in r:
        print(f"  {row[0]}m status={row[1]}: {row[2]}")
else:
    print("  ZERO 15m/30m rows in validated_setups (any status)")

# 5. What does the validator require? Check the strategy IDs that WOULD pass
print("\n--- 15m candidates that meet basic thresholds (N>=50, ExpR>0, Sharpe>0.15) ---")
for om in [15, 30]:
    r = con.execute(f"""
        SELECT COUNT(*) FROM experimental_strategies
        WHERE orb_minutes = {om}
          AND instrument IN ('MGC','MNQ','MES','M2K')
          AND sample_size >= 50
          AND expectancy_r > 0
          AND sharpe_ratio > 0.15
    """).fetchone()[0]
    print(f"  {om}m: {r} candidates meeting basic thresholds")

# 6. MNQ 15m daily_features anomaly - only 584 rows vs 1,462 for 5m/30m
print("\n--- MNQ DAILY_FEATURES ANOMALY ---")
r = con.execute("""
    SELECT orb_minutes, COUNT(*), MIN(trading_day), MAX(trading_day)
    FROM daily_features
    WHERE symbol = 'MNQ'
    GROUP BY orb_minutes ORDER BY orb_minutes
""").fetchall()
for row in r:
    print(f"  MNQ {row[0]}m: {row[1]:,} rows ({row[2]} to {row[3]})")

print("\n" + "=" * 70)
print("INVESTIGATION COMPLETE")
print("=" * 70)
con.close()
