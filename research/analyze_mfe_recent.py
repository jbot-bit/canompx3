#!/usr/bin/env python3
"""
MFE distribution -- RECENT ONLY (last 18m and 12m).

The full 10yr dataset is dominated by low-vol years where gold was $1800-2000.
Current regime: gold $4000-5000, massive ORBs. Different game.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np

from pipeline.paths import GOLD_DB_PATH

DB = GOLD_DB_PATH
con = duckdb.connect(str(DB), read_only=True)

END = date(2026, 2, 4)
WINDOWS = {
    "18m": END - timedelta(days=18 * 30),
    "12m": END - timedelta(days=12 * 30),
}

for window_name, start in WINDOWS.items():
    print(f"\n{'#' * 90}")
    print(f"  MFE DISTRIBUTION -- {window_name} window ({start} to {END})")
    print(f"{'#' * 90}")

    for instrument in ["MGC"]:
        for orb in ["0900", "1000", "1800", "2300", "0030"]:
            # RR4.0 gives longest hold window = truest MFE
            df = con.execute(f"""
                SELECT o.mfe_r, o.pnl_r, o.entry_price, o.stop_price,
                       o.trading_day,
                       d.orb_{orb}_size as orb_size
                FROM orb_outcomes o
                LEFT JOIN daily_features d
                    ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
                WHERE o.symbol = ? AND o.orb_label = ?
                  AND o.entry_model = 'E1' AND o.confirm_bars = 1
                  AND o.rr_target = 4.0
                  AND o.mfe_r IS NOT NULL AND o.entry_ts IS NOT NULL
                  AND o.trading_day >= ? AND o.trading_day <= ?
            """, [instrument, orb, start, END]).fetchdf()

            if len(df) < 10:
                continue

            mfe = df["mfe_r"].values
            orb_sizes = df["orb_size"].dropna().values

            print(f"\n  --- {orb} (N={len(df)}, avg ORB size={np.mean(orb_sizes):.1f}pt) ---")

            pcts = [25, 50, 75, 90, 95, 99]
            vals = np.percentile(mfe, pcts)
            pct_str = "  ".join(f"P{p}={v:.2f}R" for p, v in zip(pcts, vals))
            print(f"    {pct_str}")
            print(f"    Mean={mfe.mean():.2f}R  Max={mfe.max():.2f}R")

            thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
            print(f"    {'Thr':>5s} {'N':>5s} {'%':>6s}  |  G5+N  G5+%   |  G6+N  G6+%")
            for t in thresholds:
                ct = (mfe >= t).sum()
                pct = 100 * ct / len(mfe)

                # G5 filter
                df_g5 = df[df["orb_size"] >= 5]
                mfe_g5 = df_g5["mfe_r"].values
                ct_g5 = (mfe_g5 >= t).sum() if len(mfe_g5) > 0 else 0
                pct_g5 = 100 * ct_g5 / len(mfe_g5) if len(mfe_g5) > 0 else 0

                # G6 filter
                df_g6 = df[df["orb_size"] >= 6]
                mfe_g6 = df_g6["mfe_r"].values
                ct_g6 = (mfe_g6 >= t).sum() if len(mfe_g6) > 0 else 0
                pct_g6 = 100 * ct_g6 / len(mfe_g6) if len(mfe_g6) > 0 else 0

                marker = ""
                if t == 4.0:
                    marker = " <-- current max"
                elif t > 4.0:
                    marker = " ** gap **"
                print(f"    {t:>4.1f}R {ct:>5d} {pct:>5.1f}%  |  {ct_g5:>4d}  {pct_g5:>5.1f}%  |  {ct_g6:>4d}  {pct_g6:>5.1f}%{marker}")

            # E3 for 1800/2300
            if orb in ["1800", "2300"]:
                df_e3 = con.execute("""
                    SELECT mfe_r FROM orb_outcomes
                    WHERE symbol = ? AND orb_label = ?
                      AND entry_model = 'E3' AND confirm_bars = 1
                      AND rr_target = 4.0
                      AND mfe_r IS NOT NULL AND entry_ts IS NOT NULL
                      AND trading_day >= ? AND trading_day <= ?
                """, [instrument, orb, start, END]).fetchdf()
                if len(df_e3) >= 10:
                    mfe_e3 = df_e3["mfe_r"].values
                    print(f"    E3 (N={len(df_e3)}): Mean={mfe_e3.mean():.2f}R P90={np.percentile(mfe_e3,90):.2f}R P99={np.percentile(mfe_e3,99):.2f}R")
                    for t in [4.0, 5.0, 6.0, 8.0]:
                        ct = (mfe_e3 >= t).sum()
                        print(f"      {t:.1f}R: {ct} ({100*ct/len(mfe_e3):.1f}%)")

    # Summary table
    print(f"\n  === {window_name} SUMMARY: Trades reaching 5R+ ===")
    print(f"  {'Session':>8s} {'NoFilt':>8s} {'G5':>8s} {'G6':>8s}")
    for orb in ["0900", "1000", "1800", "2300", "0030"]:
        for filt_name, filt_val in [("NoFilt", 0), ("G5", 5), ("G6", 6)]:
            filt_clause = f"AND d.orb_{orb}_size >= {filt_val}" if filt_val > 0 else ""
            row = con.execute(f"""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN o.mfe_r >= 5.0 THEN 1 ELSE 0 END) as reach5,
                       SUM(CASE WHEN o.mfe_r >= 6.0 THEN 1 ELSE 0 END) as reach6
                FROM orb_outcomes o
                LEFT JOIN daily_features d
                    ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
                WHERE o.symbol = 'MGC' AND o.orb_label = ?
                  AND o.entry_model = 'E1' AND o.confirm_bars = 1
                  AND o.rr_target = 4.0
                  AND o.mfe_r IS NOT NULL AND o.entry_ts IS NOT NULL
                  AND o.trading_day >= ? AND o.trading_day <= ?
                  {filt_clause}
            """, [orb, start, END]).fetchone()
            if filt_name == "NoFilt":
                nf_str = f"{row[1]}/{row[0]}" if row[0] else "0/0"
            elif filt_name == "G5":
                g5_str = f"{row[1]}/{row[0]}" if row[0] else "0/0"
            else:
                g6_str = f"{row[1]}/{row[0]}" if row[0] else "0/0"
        print(f"  {orb:>8s} {nf_str:>8s} {g5_str:>8s} {g6_str:>8s}")

con.close()
print("\nDone.")
