#!/usr/bin/env python3
"""
Honest 2300 session analysis -- MGC + MNQ, 12m and 18m windows.

Cost models: MGC $10/pt $8.40 RT | MNQ $2/pt $2.74 RT
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH

DB = GOLD_DB_PATH
con = duckdb.connect(str(DB), read_only=True)

END = date(2026, 2, 4)
WINDOWS = {"18m": END - timedelta(days=18 * 30), "12m": END - timedelta(days=365)}

COST = {
    "MGC": {"pv": 10.0, "friction": 8.40},
    "MNQ": {"pv": 2.0, "friction": 2.74},
}


def analyze(instrument, start, end, window_name):
    cost = COST[instrument]
    orb = "2300"

    print(f"\n{'=' * 90}")
    print(f"{instrument} 2300 -- {window_name} ({start} to {end})")
    print(f"Cost: ${cost['pv']}/pt, ${cost['friction']} RT friction")
    print(f"{'=' * 90}")

    # Break days
    break_days = con.execute("""
        SELECT COUNT(DISTINCT trading_day) FROM orb_outcomes
        WHERE symbol = ? AND orb_label = ?
          AND trading_day >= ? AND trading_day <= ?
    """, [instrument, orb, start, end]).fetchone()[0]

    # ORB size stats
    orb_stats = con.execute("""
        SELECT COUNT(*) as n,
               ROUND(AVG(orb_2300_size), 2) as avg,
               ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY orb_2300_size), 2) as p50,
               ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY orb_2300_size), 2) as p90
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
          AND trading_day >= ? AND trading_day <= ?
          AND orb_2300_break_dir IS NOT NULL
    """, [instrument, start, end]).fetchone()

    print(f"Break days: {break_days} | ORB size: avg={orb_stats[1]} p50={orb_stats[2]} p90={orb_stats[3]}")

    # Double-break rate
    db = con.execute("""
        SELECT ROUND(100.0 * SUM(CASE WHEN orb_2300_double_break THEN 1 ELSE 0 END) / COUNT(*), 1)
        FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
          AND trading_day >= ? AND trading_day <= ?
          AND orb_2300_break_dir IS NOT NULL
    """, [instrument, start, end]).fetchone()[0]
    print(f"Double-break rate: {db}%")

    # Pull all outcomes
    df = con.execute("""
        SELECT trading_day, rr_target, confirm_bars, entry_model,
               entry_price, stop_price, pnl_r, outcome
        FROM orb_outcomes
        WHERE symbol = ? AND orb_label = ?
          AND trading_day >= ? AND trading_day <= ?
        ORDER BY trading_day
    """, [instrument, orb, start, end]).fetchdf()

    if len(df) == 0:
        print("  No data.")
        return

    df["risk_points"] = (df["entry_price"] - df["stop_price"]).abs()
    df["friction_r"] = cost["friction"] / (df["risk_points"] * cost["pv"])
    df["pnl_r_net"] = df["pnl_r"] - df["friction_r"]

    # Inside day filter
    id_days = con.execute("""
        WITH lagged AS (
            SELECT trading_day, daily_high, daily_low,
                   LAG(daily_high) OVER (ORDER BY trading_day) as prev_high,
                   LAG(daily_low) OVER (ORDER BY trading_day) as prev_low
            FROM daily_features
            WHERE symbol = ? AND orb_minutes = 5
              AND trading_day >= (? - INTERVAL '5 days') AND trading_day <= ?
        )
        SELECT trading_day FROM lagged
        WHERE daily_high < prev_high AND daily_low > prev_low
          AND trading_day >= ?
    """, [instrument, start, end, start]).fetchdf()["trading_day"].tolist()

    # Filter sets
    g6_days = con.execute("""
        SELECT trading_day FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
          AND trading_day >= ? AND trading_day <= ?
          AND orb_2300_size >= 6 AND orb_2300_break_dir IS NOT NULL
    """, [instrument, start, end]).fetchdf()["trading_day"].tolist()

    g8_days = con.execute("""
        SELECT trading_day FROM daily_features
        WHERE symbol = ? AND orb_minutes = 5
          AND trading_day >= ? AND trading_day <= ?
          AND orb_2300_size >= 8 AND orb_2300_break_dir IS NOT NULL
    """, [instrument, start, end]).fetchdf()["trading_day"].tolist()

    filters = [
        ("NO_FILTER", df),
        ("G6", df[df["trading_day"].isin(g6_days)]),
        ("G8", df[df["trading_day"].isin(g8_days)]),
        ("INSIDE_DAY", df[df["trading_day"].isin(id_days)]),
    ]

    for filt_name, df_f in filters:
        if len(df_f) == 0:
            continue

        n_days = df_f["trading_day"].nunique()
        print(f"\n  --- {filt_name} ({n_days} days, {len(df_f)} outcomes) ---")
        print(f"  {'EM':<4} {'RR':<5} {'CB':<4} {'N':<5} {'WR%':<7} {'ExpR_raw':<10} {'ExpR_net':<10} {'Sharpe':<8} {'MaxDD':<7}")
        print(f"  {'-' * 75}")

        results = []
        for (em, rr, cb), g in df_f.groupby(["entry_model", "rr_target", "confirm_bars"]):
            n = len(g)
            if n < 10:
                continue
            wr = (g["outcome"] == "win").mean()
            expr_raw = g["pnl_r"].mean()
            expr_net = g["pnl_r_net"].mean()
            std = g["pnl_r_net"].std()
            sharpe = expr_net / std if std > 0 else 0
            cumsum = g["pnl_r_net"].cumsum()
            dd = (cumsum.cummax() - cumsum).max()
            results.append((em, rr, cb, n, wr, expr_raw, expr_net, sharpe, dd))

        results.sort(key=lambda x: -x[6])

        shown = 0
        for em, rr, cb, n, wr, expr_raw, expr_net, sharpe, dd in results:
            if expr_net < -0.15 and shown >= 10:
                continue  # skip deep losers after showing top 10
            flag = " **" if expr_net > 0.05 and n >= 15 else ""
            print(f"  {em:<4} {rr:<5} {int(cb):<4} {n:<5} {wr*100:>5.1f}%  {expr_raw:>+8.4f}  {expr_net:>+8.4f}  {sharpe:>+6.4f}  {dd:>6.2f}{flag}")
            shown += 1

        # Count positives
        pos = sum(1 for r in results if r[6] > 0)
        print(f"  Net-positive: {pos}/{len(results)}")


for instrument in ["MGC", "MNQ"]:
    for wname, start in WINDOWS.items():
        # Check data exists
        ct = con.execute(
            "SELECT COUNT(*) FROM orb_outcomes WHERE symbol = ? AND orb_label = '2300' AND trading_day >= ?",
            [instrument, start]
        ).fetchone()[0]
        if ct > 0:
            analyze(instrument, start, END, wname)

con.close()
print("\nDone.")
