#!/usr/bin/env python3
"""
Single-break analysis: follow-through distance, MAE/MFE vs ORB size.
For all instruments and sessions in the database.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np

from pipeline.paths import GOLD_DB_PATH
from pipeline.asset_configs import get_enabled_sessions

DB = GOLD_DB_PATH
con = duckdb.connect(str(DB), read_only=True)

END = date(2026, 2, 4)
START_18M = END - timedelta(days=18 * 30)
START_12M = END - timedelta(days=365)

for instrument in ["MGC", "MNQ"]:
    ct = con.execute("SELECT COUNT(*) FROM daily_features WHERE symbol = ?", [instrument]).fetchone()[0]
    if ct == 0:
        continue

    print(f"\n{'#' * 90}")
    print(f"  {instrument} -- SINGLE vs DOUBLE BREAK ANALYSIS")
    print(f"{'#' * 90}")

    for window_name, start in [("18m", START_18M), ("12m", START_12M)]:
        print(f"\n  === {window_name} window ({start} to {END}) ===")
        print(f"  {'Session':>8s} {'Break':>6s} {'Single':>7s} {'Dbl':>5s} {'Sgl%':>6s} {'Dbl%':>6s} | {'Sgl MFE':>8s} {'Dbl MFE':>8s} {'Sgl WR@2R':>10s} {'Dbl WR@2R':>10s}")
        print(f"  {'-' * 90}")

        for orb in sorted(get_enabled_sessions(instrument)):
            row = con.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN orb_{orb}_double_break = false THEN 1 ELSE 0 END) as single,
                    SUM(CASE WHEN orb_{orb}_double_break = true THEN 1 ELSE 0 END) as dbl
                FROM daily_features
                WHERE symbol = ? AND orb_minutes = 5
                  AND trading_day >= ? AND trading_day <= ?
                  AND orb_{orb}_break_dir IS NOT NULL
            """, [instrument, start, END]).fetchone()

            total, single, dbl = row
            if total == 0:
                continue
            sgl_pct = 100 * single / total
            dbl_pct = 100 * dbl / total

            # MFE for single vs double break days (using RR4.0 E1 CB1 for max hold window)
            # Single-break days
            sgl_mfe = con.execute(f"""
                SELECT AVG(o.mfe_r)
                FROM orb_outcomes o
                JOIN daily_features d
                    ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
                WHERE o.symbol = ? AND o.orb_label = ?
                  AND o.entry_model = 'E1' AND o.confirm_bars = 1 AND o.rr_target = 4.0
                  AND o.mfe_r IS NOT NULL AND o.entry_ts IS NOT NULL
                  AND o.trading_day >= ? AND o.trading_day <= ?
                  AND d.orb_{orb}_double_break = false
            """, [instrument, orb, start, END]).fetchone()[0]

            # Double-break days
            dbl_mfe = con.execute(f"""
                SELECT AVG(o.mfe_r)
                FROM orb_outcomes o
                JOIN daily_features d
                    ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
                WHERE o.symbol = ? AND o.orb_label = ?
                  AND o.entry_model = 'E1' AND o.confirm_bars = 1 AND o.rr_target = 4.0
                  AND o.mfe_r IS NOT NULL AND o.entry_ts IS NOT NULL
                  AND o.trading_day >= ? AND o.trading_day <= ?
                  AND d.orb_{orb}_double_break = true
            """, [instrument, orb, start, END]).fetchone()[0]

            # Win rate at RR2.0 for single vs double
            sgl_wr = con.execute(f"""
                SELECT AVG(CASE WHEN o.outcome = 'win' THEN 1.0 ELSE 0.0 END)
                FROM orb_outcomes o
                JOIN daily_features d
                    ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
                WHERE o.symbol = ? AND o.orb_label = ?
                  AND o.entry_model = 'E1' AND o.confirm_bars = 2 AND o.rr_target = 2.0
                  AND o.entry_ts IS NOT NULL
                  AND o.trading_day >= ? AND o.trading_day <= ?
                  AND d.orb_{orb}_double_break = false
            """, [instrument, orb, start, END]).fetchone()[0]

            dbl_wr = con.execute(f"""
                SELECT AVG(CASE WHEN o.outcome = 'win' THEN 1.0 ELSE 0.0 END)
                FROM orb_outcomes o
                JOIN daily_features d
                    ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
                WHERE o.symbol = ? AND o.orb_label = ?
                  AND o.entry_model = 'E1' AND o.confirm_bars = 2 AND o.rr_target = 2.0
                  AND o.entry_ts IS NOT NULL
                  AND o.trading_day >= ? AND o.trading_day <= ?
                  AND d.orb_{orb}_double_break = true
            """, [instrument, orb, start, END]).fetchone()[0]

            sgl_mfe_s = f"{sgl_mfe:.2f}R" if sgl_mfe else "N/A"
            dbl_mfe_s = f"{dbl_mfe:.2f}R" if dbl_mfe else "N/A"
            sgl_wr_s = f"{sgl_wr*100:.1f}%" if sgl_wr else "N/A"
            dbl_wr_s = f"{dbl_wr*100:.1f}%" if dbl_wr else "N/A"

            print(f"  {orb:>8s} {total:>6d} {single:>7d} {dbl:>5d} {sgl_pct:>5.1f}% {dbl_pct:>5.1f}% | {sgl_mfe_s:>8s} {dbl_mfe_s:>8s} {sgl_wr_s:>10s} {dbl_wr_s:>10s}")

    # --- MAE/MFE vs ORB Size buckets ---
    print(f"\n{'=' * 90}")
    print(f"  {instrument} -- MAE/MFE vs ORB SIZE (last 18m, E1 CB1 RR4.0)")
    print(f"{'=' * 90}")

    for orb in SESSIONS:
        df = con.execute(f"""
            SELECT o.mfe_r, o.mae_r, o.pnl_r,
                   d.orb_{orb}_size as orb_size,
                   d.orb_{orb}_double_break as is_dbl
            FROM orb_outcomes o
            JOIN daily_features d
                ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
            WHERE o.symbol = ? AND o.orb_label = ?
              AND o.entry_model = 'E1' AND o.confirm_bars = 1 AND o.rr_target = 4.0
              AND o.mfe_r IS NOT NULL AND o.mae_r IS NOT NULL AND o.entry_ts IS NOT NULL
              AND o.trading_day >= ? AND o.trading_day <= ?
              AND d.orb_{orb}_size IS NOT NULL
        """, [instrument, orb, START_18M, END]).fetchdf()

        if len(df) < 20:
            continue

        # Define ORB size buckets based on instrument
        if instrument == "MGC":
            buckets = [(0, 3, "<3"), (3, 5, "3-5"), (5, 8, "5-8"), (8, 12, "8-12"), (12, 999, "12+")]
        else:  # MNQ
            buckets = [(0, 10, "<10"), (10, 20, "10-20"), (20, 35, "20-35"), (35, 50, "35-50"), (50, 999, "50+")]

        print(f"\n  --- {orb} (N={len(df)}) ---")
        print(f"  {'Bucket':>8s} {'N':>5s} {'Sgl%':>6s} {'MFE':>7s} {'MAE':>7s} {'MFE/MAE':>8s} {'P90 MFE':>8s} {'>2R%':>6s} {'>3R%':>6s}")
        print(f"  {'-' * 70}")

        for lo, hi, label in buckets:
            mask = (df["orb_size"] >= lo) & (df["orb_size"] < hi)
            sub = df[mask]
            if len(sub) < 5:
                continue
            mfe = sub["mfe_r"].values
            mae = sub["mae_r"].values
            sgl_pct = 100 * (~sub["is_dbl"]).sum() / len(sub)
            avg_mfe = mfe.mean()
            avg_mae = mae.mean()
            ratio = avg_mfe / abs(avg_mae) if abs(avg_mae) > 0.01 else float("inf")
            p90_mfe = np.percentile(mfe, 90)
            gt2 = 100 * (mfe >= 2.0).sum() / len(mfe)
            gt3 = 100 * (mfe >= 3.0).sum() / len(mfe)

            print(f"  {label:>8s} {len(sub):>5d} {sgl_pct:>5.1f}% {avg_mfe:>6.2f}R {avg_mae:>6.2f}R {ratio:>7.2f}x {p90_mfe:>7.2f}R {gt2:>5.1f}% {gt3:>5.1f}%")

con.close()
print("\nDone.")
