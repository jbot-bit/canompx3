#!/usr/bin/env python3
"""
Honest analysis: MNQ 1800 session trades, last 18 months.

MNQ cost model: $2/point, $2.74 RT friction.
No cherry-picking. Reports ALL combos, sorted by ExpR.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH as DB_PATH

# --- Config ---
INSTRUMENT = "MNQ"
ORB_LABEL = "1800"
# Last 18 months from data end (2026-02-03)
END_DATE = date(2026, 2, 3)
START_DATE = END_DATE - timedelta(days=18 * 30)  # ~Aug 2024
# MNQ cost model
POINT_VALUE = 2.0      # $/point
RT_FRICTION = 2.74     # $ round-trip (commission + slippage)

print(f"=== MNQ 1800 Honest Analysis ===")
print(f"Period: {START_DATE} to {END_DATE} (~18 months)")
print(f"Cost model: ${POINT_VALUE}/pt, ${RT_FRICTION} RT friction")
print(f"DB: {DB_PATH}")
print()

con = duckdb.connect(str(DB_PATH), read_only=True)

# --- Step 1: How many 1800 break days exist? ---
break_days = con.execute("""
    SELECT COUNT(DISTINCT trading_day) as n_days
    FROM orb_outcomes
    WHERE symbol = ? AND orb_label = ?
      AND trading_day >= ? AND trading_day <= ?
""", [INSTRUMENT, ORB_LABEL, START_DATE, END_DATE]).fetchone()[0]

print(f"1800 break days in window: {break_days}")

# --- Step 2: ORB size distribution ---
print("\n--- ORB Size Distribution (from daily_features) ---")
orb_sizes = con.execute("""
    SELECT
        COUNT(*) as n,
        ROUND(AVG(orb_1800_size), 2) as avg_size,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY orb_1800_size), 2) as p25,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY orb_1800_size), 2) as p50,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY orb_1800_size), 2) as p75,
        ROUND(MIN(orb_1800_size), 2) as min_size,
        ROUND(MAX(orb_1800_size), 2) as max_size
    FROM daily_features
    WHERE symbol = ? AND orb_minutes = 5
      AND trading_day >= ? AND trading_day <= ?
      AND orb_1800_size IS NOT NULL
      AND orb_1800_break_dir IS NOT NULL
""", [INSTRUMENT, START_DATE, END_DATE]).fetchone()
print(f"  Days with 1800 break: {orb_sizes[0]}")
print(f"  ORB size: avg={orb_sizes[1]}, p25={orb_sizes[2]}, p50={orb_sizes[3]}, p75={orb_sizes[4]}")
print(f"  Range: {orb_sizes[5]} - {orb_sizes[6]}")

# --- Step 3: Pull all outcomes ---
df = con.execute("""
    SELECT
        trading_day, rr_target, confirm_bars, entry_model,
        entry_price, stop_price, pnl_r, outcome
    FROM orb_outcomes
    WHERE symbol = ? AND orb_label = ?
      AND trading_day >= ? AND trading_day <= ?
    ORDER BY trading_day, rr_target, confirm_bars, entry_model
""", [INSTRUMENT, ORB_LABEL, START_DATE, END_DATE]).fetchdf()

print(f"\nTotal outcome rows: {len(df)}")

# --- Step 4: Apply friction in R-terms ---
# friction_r = RT_FRICTION / (risk_points * POINT_VALUE)
# risk_points = |entry_price - stop_price|
df["risk_points"] = (df["entry_price"] - df["stop_price"]).abs()
df["friction_r"] = RT_FRICTION / (df["risk_points"] * POINT_VALUE)
df["pnl_r_net"] = df["pnl_r"] - df["friction_r"]

# --- Step 5: Aggregate by (entry_model, rr_target, confirm_bars) -- NO FILTER (raw) ---
print("\n" + "=" * 90)
print("RAW RESULTS (NO ORB SIZE FILTER) -- ALL 1800 break days")
print("=" * 90)
print(f"{'EM':<4} {'RR':<5} {'CB':<4} {'N':<6} {'WR%':<7} {'ExpR_raw':<10} {'ExpR_net':<10} {'Sharpe':<8} {'MaxDD_R':<9}")
print("-" * 90)

results = []
for (em, rr, cb), g in df.groupby(["entry_model", "rr_target", "confirm_bars"]):
    n = len(g)
    if n < 20:
        continue
    wr = (g["outcome"] == "win").mean()
    expr_raw = g["pnl_r"].mean()
    expr_net = g["pnl_r_net"].mean()
    std = g["pnl_r_net"].std()
    sharpe = expr_net / std if std > 0 else 0
    # Max drawdown in R
    cumsum = g["pnl_r_net"].cumsum()
    running_max = cumsum.cummax()
    dd = running_max - cumsum
    max_dd = dd.max()
    results.append((em, rr, cb, n, wr, expr_raw, expr_net, sharpe, max_dd))

results.sort(key=lambda x: -x[6])  # sort by net ExpR

for em, rr, cb, n, wr, expr_raw, expr_net, sharpe, max_dd in results:
    flag = " **" if expr_net > 0.05 and n >= 50 else ""
    print(f"{em:<4} {rr:<5} {int(cb):<4} {n:<6} {wr*100:>5.1f}%  {expr_raw:>+8.4f}  {expr_net:>+8.4f}  {sharpe:>+6.4f}  {max_dd:>7.2f}{flag}")

# --- Step 6: With G6+ filter ---
print("\n" + "=" * 90)
print("WITH ORB_G6 FILTER (orb_1800_size >= 6)")
print("=" * 90)

# Get eligible days
g6_days = con.execute("""
    SELECT trading_day
    FROM daily_features
    WHERE symbol = ? AND orb_minutes = 5
      AND trading_day >= ? AND trading_day <= ?
      AND orb_1800_size >= 6
      AND orb_1800_break_dir IS NOT NULL
""", [INSTRUMENT, START_DATE, END_DATE]).fetchdf()["trading_day"].tolist()

df_g6 = df[df["trading_day"].isin(g6_days)]
print(f"G6-eligible days: {len(g6_days)} / {break_days} ({100*len(g6_days)/max(break_days,1):.0f}%)")
print(f"Outcome rows after G6 filter: {len(df_g6)}")
print(f"\n{'EM':<4} {'RR':<5} {'CB':<4} {'N':<6} {'WR%':<7} {'ExpR_raw':<10} {'ExpR_net':<10} {'Sharpe':<8} {'MaxDD_R':<9}")
print("-" * 90)

results_g6 = []
for (em, rr, cb), g in df_g6.groupby(["entry_model", "rr_target", "confirm_bars"]):
    n = len(g)
    if n < 15:
        continue
    wr = (g["outcome"] == "win").mean()
    expr_raw = g["pnl_r"].mean()
    expr_net = g["pnl_r_net"].mean()
    std = g["pnl_r_net"].std()
    sharpe = expr_net / std if std > 0 else 0
    cumsum = g["pnl_r_net"].cumsum()
    running_max = cumsum.cummax()
    dd = running_max - cumsum
    max_dd = dd.max()
    results_g6.append((em, rr, cb, n, wr, expr_raw, expr_net, sharpe, max_dd))

results_g6.sort(key=lambda x: -x[6])

for em, rr, cb, n, wr, expr_raw, expr_net, sharpe, max_dd in results_g6:
    flag = " **" if expr_net > 0.05 and n >= 30 else ""
    print(f"{em:<4} {rr:<5} {int(cb):<4} {n:<6} {wr*100:>5.1f}%  {expr_raw:>+8.4f}  {expr_net:>+8.4f}  {sharpe:>+6.4f}  {max_dd:>7.2f}{flag}")

# --- Step 7: With G8+ filter ---
print("\n" + "=" * 90)
print("WITH ORB_G8 FILTER (orb_1800_size >= 8)")
print("=" * 90)

g8_days = con.execute("""
    SELECT trading_day
    FROM daily_features
    WHERE symbol = ? AND orb_minutes = 5
      AND trading_day >= ? AND trading_day <= ?
      AND orb_1800_size >= 8
      AND orb_1800_break_dir IS NOT NULL
""", [INSTRUMENT, START_DATE, END_DATE]).fetchdf()["trading_day"].tolist()

df_g8 = df[df["trading_day"].isin(g8_days)]
print(f"G8-eligible days: {len(g8_days)} / {break_days} ({100*len(g8_days)/max(break_days,1):.0f}%)")
print(f"Outcome rows after G8 filter: {len(df_g8)}")
print(f"\n{'EM':<4} {'RR':<5} {'CB':<4} {'N':<6} {'WR%':<7} {'ExpR_raw':<10} {'ExpR_net':<10} {'Sharpe':<8} {'MaxDD_R':<9}")
print("-" * 90)

results_g8 = []
for (em, rr, cb), g in df_g8.groupby(["entry_model", "rr_target", "confirm_bars"]):
    n = len(g)
    if n < 10:
        continue
    wr = (g["outcome"] == "win").mean()
    expr_raw = g["pnl_r"].mean()
    expr_net = g["pnl_r_net"].mean()
    std = g["pnl_r_net"].std()
    sharpe = expr_net / std if std > 0 else 0
    cumsum = g["pnl_r_net"].cumsum()
    running_max = cumsum.cummax()
    dd = running_max - cumsum
    max_dd = dd.max()
    results_g8.append((em, rr, cb, n, wr, expr_raw, expr_net, sharpe, max_dd))

results_g8.sort(key=lambda x: -x[6])

for em, rr, cb, n, wr, expr_raw, expr_net, sharpe, max_dd in results_g8:
    flag = " **" if expr_net > 0.05 and n >= 20 else ""
    print(f"{em:<4} {rr:<5} {int(cb):<4} {n:<6} {wr*100:>5.1f}%  {expr_raw:>+8.4f}  {expr_net:>+8.4f}  {sharpe:>+6.4f}  {max_dd:>7.2f}{flag}")

# --- Step 8: INSIDE_DAY filter ---
print("\n" + "=" * 90)
print("WITH INSIDE_DAY FILTER")
print("=" * 90)

id_days = con.execute("""
    SELECT d1.trading_day
    FROM daily_features d1
    JOIN daily_features d2
      ON d1.symbol = d2.symbol
      AND d1.orb_minutes = d2.orb_minutes
      AND d2.trading_day = (
          SELECT MAX(trading_day) FROM daily_features
          WHERE symbol = d1.symbol AND orb_minutes = d1.orb_minutes
            AND trading_day < d1.trading_day
      )
    WHERE d1.symbol = ? AND d1.orb_minutes = 5
      AND d1.trading_day >= ? AND d1.trading_day <= ?
      AND d1.daily_high <= d2.daily_high
      AND d1.daily_low >= d2.daily_low
""", [INSTRUMENT, START_DATE, END_DATE]).fetchdf()

# Actually, inside day = prior day's range contains current day
# But we need to check: is the prior day the inside day, or current?
# Standard: inside day = today's H < yesterday's H AND today's L > yesterday's L
# The trade happens on the NEXT day after inside day.
# Let me just check how many match vs orb_outcomes.
# Actually for simplicity, let's just report the INSIDE_DAY results from experimental_strategies
# since this filter is pre-computed. But we want 18-month window...

# Let me use a simpler approach: just flag inside days properly
id_days2 = con.execute("""
    WITH lagged AS (
        SELECT
            trading_day,
            daily_high,
            daily_low,
            LAG(daily_high) OVER (ORDER BY trading_day) as prev_high,
            LAG(daily_low) OVER (ORDER BY trading_day) as prev_low
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
          AND trading_day >= (? - INTERVAL '5 days') AND trading_day <= ?
    )
    SELECT trading_day
    FROM lagged
    WHERE daily_high < prev_high AND daily_low > prev_low
      AND trading_day >= ?
""", [INSTRUMENT, START_DATE, END_DATE, START_DATE]).fetchdf()["trading_day"].tolist()

df_id = df[df["trading_day"].isin(id_days2)]
print(f"Inside days: {len(id_days2)} / {break_days} ({100*len(id_days2)/max(break_days,1):.0f}%)")
print(f"Outcome rows after inside_day filter: {len(df_id)}")

if len(df_id) > 0:
    print(f"\n{'EM':<4} {'RR':<5} {'CB':<4} {'N':<6} {'WR%':<7} {'ExpR_raw':<10} {'ExpR_net':<10} {'Sharpe':<8} {'MaxDD_R':<9}")
    print("-" * 90)

    results_id = []
    for (em, rr, cb), g in df_id.groupby(["entry_model", "rr_target", "confirm_bars"]):
        n = len(g)
        if n < 10:
            continue
        wr = (g["outcome"] == "win").mean()
        expr_raw = g["pnl_r"].mean()
        expr_net = g["pnl_r_net"].mean()
        std = g["pnl_r_net"].std()
        sharpe = expr_net / std if std > 0 else 0
        cumsum = g["pnl_r_net"].cumsum()
        running_max = cumsum.cummax()
        dd = running_max - cumsum
        max_dd = dd.max()
        results_id.append((em, rr, cb, n, wr, expr_raw, expr_net, sharpe, max_dd))

    results_id.sort(key=lambda x: -x[6])
    for em, rr, cb, n, wr, expr_raw, expr_net, sharpe, max_dd in results_id:
        flag = " **" if expr_net > 0.10 and n >= 15 else ""
        print(f"{em:<4} {rr:<5} {int(cb):<4} {n:<6} {wr*100:>5.1f}%  {expr_raw:>+8.4f}  {expr_net:>+8.4f}  {sharpe:>+6.4f}  {max_dd:>7.2f}{flag}")

# --- Step 9: Double-break rate ---
print("\n" + "=" * 90)
print("DOUBLE-BREAK ANALYSIS")
print("=" * 90)

db_stats = con.execute("""
    SELECT
        COUNT(*) as total,
        SUM(CASE WHEN orb_1800_double_break THEN 1 ELSE 0 END) as double_breaks,
        ROUND(100.0 * SUM(CASE WHEN orb_1800_double_break THEN 1 ELSE 0 END) / COUNT(*), 1) as db_pct
    FROM daily_features
    WHERE symbol = ? AND orb_minutes = 5
      AND trading_day >= ? AND trading_day <= ?
      AND orb_1800_break_dir IS NOT NULL
""", [INSTRUMENT, START_DATE, END_DATE]).fetchone()

print(f"Break days: {db_stats[0]}, Double-breaks: {db_stats[1]} ({db_stats[2]}%)")

# --- Step 10: Honest summary ---
print("\n" + "=" * 90)
print("HONEST SUMMARY")
print("=" * 90)

# Best raw combo
if results:
    best_raw = results[0]
    print(f"Best raw (no filter): {best_raw[0]} RR{best_raw[1]} CB{best_raw[2]} | N={best_raw[3]} WR={best_raw[4]*100:.1f}% ExpR_net={best_raw[6]:+.4f}")

if results_g6:
    best_g6 = results_g6[0]
    print(f"Best G6:              {best_g6[0]} RR{best_g6[1]} CB{best_g6[2]} | N={best_g6[3]} WR={best_g6[4]*100:.1f}% ExpR_net={best_g6[6]:+.4f}")

if results_g8:
    best_g8 = results_g8[0]
    print(f"Best G8:              {best_g8[0]} RR{best_g8[1]} CB{best_g8[2]} | N={best_g8[3]} WR={best_g8[4]*100:.1f}% ExpR_net={best_g8[6]:+.4f}")

# Count how many combos are net positive
pos_raw = sum(1 for r in results if r[6] > 0)
pos_g6 = sum(1 for r in results_g6 if r[6] > 0)
pos_g8 = sum(1 for r in results_g8 if r[6] > 0)
print(f"\nNet-positive combos: raw={pos_raw}/{len(results)}, G6={pos_g6}/{len(results_g6)}, G8={pos_g8}/{len(results_g8)}")

# Annualized Sharpe check (trades_per_year * sqrt for annualization)
if results:
    best = results[0]
    tpy = best[3] / 1.5  # trades in 18 months -> per year
    sha = best[7] * (tpy ** 0.5)
    print(f"\nBest raw annualized Sharpe estimate: {sha:.2f} (trades/yr ~ {tpy:.0f})")
    if sha < 0.5:
        print("  WARNING: Below 0.5 ShANN minimum bar")

con.close()
print("\nDone.")
