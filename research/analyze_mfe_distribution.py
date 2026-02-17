#!/usr/bin/env python3
"""
MFE (Maximum Favorable Excursion) distribution analysis.

Question: How far do ORB breakout trades ACTUALLY run before reversing?
If MFE regularly exceeds 4R, we're leaving money on the table with RR_TARGETS maxing at 4.0.

Read-only. No DB writes. Uses existing orb_outcomes MFE_R column.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np

from pipeline.paths import GOLD_DB_PATH
from pipeline.asset_configs import get_enabled_sessions

DB = GOLD_DB_PATH
con = duckdb.connect(str(DB), read_only=True)

print("=" * 90)
print("MFE DISTRIBUTION ANALYSIS -- How far do trades actually run?")
print(f"DB: {DB}")
print("=" * 90)

# -----------------------------------------------------------------------
# 1. Pull MFE_R for all sessions, both instruments
# -----------------------------------------------------------------------
for instrument in ["MGC", "MNQ"]:
    row_check = con.execute(
        "SELECT COUNT(*) FROM orb_outcomes WHERE symbol = ?", [instrument]
    ).fetchone()[0]
    if row_check == 0:
        continue

    print(f"\n{'#' * 90}")
    print(f"  {instrument}")
    print(f"{'#' * 90}")

    # Use CB1 E1 RR2.0 as reference (most trades, standard setup)
    # MFE_R is recorded regardless of RR target -- it's the max favorable move
    # But MFE_R in orb_outcomes is capped by what happens during the trade lifetime
    # Higher RR = longer hold = higher potential MFE
    # So use RR4.0 to get the longest hold window and truest MFE

    for orb in sorted(get_enabled_sessions(instrument)):
        df = con.execute("""
            SELECT mfe_r, pnl_r, entry_price, stop_price,
                   o.trading_day,
                   d.atr_20
            FROM orb_outcomes o
            LEFT JOIN daily_features d
                ON o.symbol = d.symbol AND o.trading_day = d.trading_day AND d.orb_minutes = 5
            WHERE o.symbol = ? AND o.orb_label = ?
              AND o.entry_model = 'E1' AND o.confirm_bars = 1
              AND o.rr_target = 4.0
              AND o.mfe_r IS NOT NULL
              AND o.entry_ts IS NOT NULL
        """, [instrument, orb]).fetchdf()

        if len(df) < 20:
            continue

        mfe = df["mfe_r"].values

        print(f"\n--- {orb} SESSION (N={len(df)}, E1 CB1 RR4.0 window) ---")

        # Percentiles
        pcts = [25, 50, 75, 90, 95, 99]
        vals = np.percentile(mfe, pcts)
        print(f"  MFE percentiles:")
        for p, v in zip(pcts, vals):
            print(f"    P{p:02d}: {v:>6.2f}R")

        print(f"  Mean MFE: {mfe.mean():.2f}R  |  Std: {mfe.std():.2f}R")

        # How many reach various R thresholds?
        thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        print(f"\n  Trades reaching threshold:")
        print(f"  {'Threshold':>10s} {'Count':>6s} {'%':>7s} {'If won at this R':>18s}")
        for t in thresholds:
            count = (mfe >= t).sum()
            pct = count / len(mfe) * 100
            # Hypothetical ExpR if you won t*R on these and lost -1R on the rest
            hyp_expr = (count * t + (len(mfe) - count) * (-1.0)) / len(mfe)
            marker = " <-- current max" if t == 4.0 else ""
            marker = " ** UNTESTED **" if t > 4.0 else marker
            print(f"  {t:>8.1f}R  {count:>6d}  {pct:>6.1f}%  ExpR={hyp_expr:>+6.3f}R{marker}")

        # With G5+ filter
        if instrument == "MGC":
            for filt_name, filt_size in [("G5", 5), ("G6", 6)]:
                df_filt = df[df["trading_day"].isin(
                    con.execute(f"""
                        SELECT trading_day FROM daily_features
                        WHERE symbol = ? AND orb_minutes = 5
                          AND orb_{orb}_size >= ?
                          AND orb_{orb}_break_dir IS NOT NULL
                    """, [instrument, filt_size]).fetchdf()["trading_day"].tolist()
                )]
                if len(df_filt) < 15:
                    continue
                mfe_f = df_filt["mfe_r"].values
                print(f"\n  With {filt_name} filter (N={len(df_filt)}):")
                for t in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
                    count = (mfe_f >= t).sum()
                    pct = count / len(mfe_f) * 100
                    print(f"    {t:.1f}R: {count:>5d} ({pct:>5.1f}%)")

        # E3 comparison for 1800
        if orb in ["1800", "2300"]:
            df_e3 = con.execute("""
                SELECT mfe_r FROM orb_outcomes
                WHERE symbol = ? AND orb_label = ?
                  AND entry_model = 'E3' AND confirm_bars = 1
                  AND rr_target = 4.0
                  AND mfe_r IS NOT NULL AND entry_ts IS NOT NULL
            """, [instrument, orb]).fetchdf()
            if len(df_e3) >= 20:
                mfe_e3 = df_e3["mfe_r"].values
                print(f"\n  E3 comparison (N={len(df_e3)}):")
                print(f"    Mean MFE: {mfe_e3.mean():.2f}R  P50={np.median(mfe_e3):.2f}R  P90={np.percentile(mfe_e3, 90):.2f}R")
                for t in [2.0, 3.0, 4.0, 5.0, 6.0]:
                    count = (mfe_e3 >= t).sum()
                    pct = count / len(mfe_e3) * 100
                    print(f"    {t:.1f}R: {count:>5d} ({pct:>5.1f}%)")

# -----------------------------------------------------------------------
# 2. THE KEY QUESTION: Is >4R reachable often enough to matter?
# -----------------------------------------------------------------------
print(f"\n{'=' * 90}")
print("VERDICT: Should we add RR 5.0, 6.0, 8.0 targets?")
print("=" * 90)

# Check 0900 and 1000 specifically (our best sessions)
for orb in ["0900", "1000"]:
    df = con.execute("""
        SELECT mfe_r FROM orb_outcomes
        WHERE symbol = 'MGC' AND orb_label = ?
          AND entry_model = 'E1' AND confirm_bars = 1
          AND rr_target = 4.0
          AND mfe_r IS NOT NULL AND entry_ts IS NOT NULL
    """, [orb]).fetchdf()
    if len(df) == 0:
        continue
    mfe = df["mfe_r"].values
    n = len(mfe)
    reach_5 = (mfe >= 5.0).sum()
    reach_6 = (mfe >= 6.0).sum()
    reach_8 = (mfe >= 8.0).sum()
    reach_10 = (mfe >= 10.0).sum()

    print(f"\n  {orb}: {n} trades")
    print(f"    >=5R: {reach_5} ({100*reach_5/n:.1f}%)  >=6R: {reach_6} ({100*reach_6/n:.1f}%)  "
          f">=8R: {reach_8} ({100*reach_8/n:.1f}%)  >=10R: {reach_10} ({100*reach_10/n:.1f}%)")

    if reach_5 / n > 0.15:
        print(f"    --> YES: {reach_5/n*100:.0f}% reach 5R. Worth testing RR5.0+")
    elif reach_5 / n > 0.08:
        print(f"    --> MAYBE: {reach_5/n*100:.0f}% reach 5R. Marginal, test carefully")
    else:
        print(f"    --> NO: Only {reach_5/n*100:.0f}% reach 5R. Not enough follow-through")

con.close()
print("\nDone.")
