#!/usr/bin/env python3
"""
1100 session conditional analysis: can we find profitable 1100 trades
by conditioning on what happened at 0900 and 1000?

Conditions tested:
  - 0900/1000 break direction vs 1100 break direction (alignment)
  - 0900/1000 outcome (did earlier session win or lose?)
  - Both 0900+1000 aligned with 1100 (triple alignment)
  - Day of week
  - ORB size tiers
  - 1100 break dir after 0900 double-break (reversal of reversal)

Read-only research script. No writes to gold.db.

Usage:
    python scripts/analyze_1100_conditional.py --db-path C:/db/gold.db
"""

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cost_model import get_cost_spec, to_r_multiple

COST_SPEC = get_cost_spec("MGC")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(db_path: Path, start: date, end: date) -> pd.DataFrame:
    """Load 1100 outcomes joined with 0900/1000 context from daily_features."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT
                o.trading_day,
                o.entry_model,
                o.rr_target,
                o.confirm_bars,
                o.outcome,
                o.pnl_r,
                o.entry_price,
                o.stop_price,

                -- 1100 ORB info
                d.orb_1100_size AS orb_1100_size,
                d.orb_1100_break_dir AS break_1100,
                d.orb_1100_double_break AS double_break_1100,

                -- 0900 context
                d.orb_0900_size AS orb_0900_size,
                d.orb_0900_break_dir AS break_0900,
                d.orb_0900_double_break AS double_break_0900,

                -- 1000 context
                d.orb_1000_size AS orb_1000_size,
                d.orb_1000_break_dir AS break_1000,
                d.orb_1000_double_break AS double_break_1000,

                -- Day of week (1=Mon, 7=Sun)
                EXTRACT(ISODOW FROM o.trading_day) AS dow

            FROM orb_outcomes o
            JOIN daily_features d
                ON o.symbol = d.symbol
                AND o.trading_day = d.trading_day
                AND d.orb_minutes = 5
            WHERE o.symbol = 'MGC'
                AND o.orb_minutes = 5
                AND o.orb_label = '1100'
                AND o.entry_ts IS NOT NULL
                AND o.outcome IS NOT NULL
                AND o.pnl_r IS NOT NULL
                AND o.trading_day BETWEEN ? AND ?
            ORDER BY o.trading_day
        """, [start, end]).fetchdf()
    finally:
        con.close()

    # Deduplicate E3 CB>1
    df = df[~((df["entry_model"] == "E3") & (df["confirm_bars"] > 1))].copy()

    print(f"Loaded {len(df)} 1100 outcomes ({df['trading_day'].nunique()} days)")
    return df


def load_earlier_outcomes(db_path: Path, start: date, end: date,
                          session: str) -> pd.DataFrame:
    """Load outcomes for an earlier session to check if they won/lost."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT
                trading_day,
                entry_model,
                rr_target,
                confirm_bars,
                outcome AS earlier_outcome,
                pnl_r AS earlier_pnl_r
            FROM orb_outcomes
            WHERE symbol = 'MGC'
                AND orb_minutes = 5
                AND orb_label = ?
                AND entry_ts IS NOT NULL
                AND outcome IS NOT NULL
                AND pnl_r IS NOT NULL
                AND trading_day BETWEEN ? AND ?
        """, [session, start, end]).fetchdf()
    finally:
        con.close()

    # Deduplicate E3 CB>1
    df = df[~((df["entry_model"] == "E3") & (df["confirm_bars"] > 1))].copy()
    return df


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def metrics(pnls: np.ndarray) -> dict:
    n = len(pnls)
    if n == 0:
        return {"n": 0, "wr": 0, "expr": 0, "sharpe": 0, "maxdd": 0, "total": 0}
    wr = float((pnls > 0).sum() / n)
    expr = float(pnls.mean())
    std = float(pnls.std())
    sharpe = expr / std if std > 0 else 0.0
    cumul = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumul)
    maxdd = float((cumul - peak).min())
    return {"n": n, "wr": wr, "expr": expr, "sharpe": sharpe,
            "maxdd": maxdd, "total": float(pnls.sum())}


def fmt(m: dict) -> str:
    if m["n"] == 0:
        return "N=0"
    return (f"N={m['n']:<4d}  WR={m['wr']*100:5.1f}%  ExpR={m['expr']:+.3f}  "
            f"Sharpe={m['sharpe']:.3f}  MaxDD={m['maxdd']:.1f}R  Total={m['total']:+.1f}R")


def analyze_group(df: pd.DataFrame, label: str, condition_col: str = None):
    """Analyze a dataframe, optionally grouped by a condition column."""
    if condition_col:
        groups = df.groupby(condition_col)
        for val, group in sorted(groups, key=lambda x: str(x[0])):
            pnls = group["pnl_r"].values
            m = metrics(pnls)
            print(f"  {label} = {val:20s}  {fmt(m)}")
    else:
        pnls = df["pnl_r"].values
        m = metrics(pnls)
        print(f"  {label:30s}  {fmt(m)}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(db_path: Path, start: date, end: date):
    df = load_data(db_path, start, end)
    if df.empty:
        print("No data. Exiting.")
        return

    # Load 0900 and 1000 outcomes for win/loss context
    outcomes_0900 = load_earlier_outcomes(db_path, start, end, "0900")
    outcomes_1000 = load_earlier_outcomes(db_path, start, end, "1000")

    # We'll focus on commonly validated params: E1 CB2 and E3 CB1, RR 2.0
    for em in ["E1", "E3"]:
        for rr in [1.5, 2.0, 2.5]:
            if em == "E1":
                subset = df[(df["entry_model"] == em) & (df["rr_target"] == rr)
                            & (df["confirm_bars"] == 2)]
            else:
                subset = df[(df["entry_model"] == em) & (df["rr_target"] == rr)
                            & (df["confirm_bars"] == 1)]

            if len(subset) < 20:
                continue

            print(f"\n{'='*90}")
            print(f"1100 {em} RR{rr} — {len(subset)} trades")
            print(f"{'='*90}")

            # Baseline
            print(f"\nBASELINE:")
            analyze_group(subset, "All 1100 trades")

            # --- ORB SIZE TIERS ---
            print(f"\nORB SIZE TIERS:")
            subset = subset.copy()
            subset["size_tier"] = pd.cut(
                subset["orb_1100_size"],
                bins=[0, 2, 4, 6, 10, 999],
                labels=["<2pt", "2-4pt", "4-6pt", "6-10pt", "10+pt"],
            )
            analyze_group(subset, "ORB size", "size_tier")

            # --- DAY OF WEEK ---
            print(f"\nDAY OF WEEK:")
            dow_names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri"}
            subset = subset.copy()
            subset["dow_name"] = subset["dow"].map(dow_names)
            analyze_group(subset, "Day", "dow_name")

            # --- ALIGNMENT WITH 0900 ---
            print(f"\n0900 ALIGNMENT (does 1100 break same direction as 0900?):")
            subset = subset.copy()
            has_both = subset["break_0900"].notna() & subset["break_1100"].notna()
            sub_both = subset[has_both].copy()
            if len(sub_both) > 0:
                sub_both["align_0900"] = np.where(
                    sub_both["break_0900"] == sub_both["break_1100"],
                    "SAME direction",
                    "OPPOSITE direction"
                )
                analyze_group(sub_both, "vs 0900", "align_0900")

                # No 0900 break
                no_0900 = subset[subset["break_0900"].isna()]
                if len(no_0900) > 0:
                    analyze_group(no_0900, "No 0900 break")

            # --- ALIGNMENT WITH 1000 ---
            print(f"\n1000 ALIGNMENT (does 1100 break same direction as 1000?):")
            has_both = subset["break_1000"].notna() & subset["break_1100"].notna()
            sub_both = subset[has_both].copy()
            if len(sub_both) > 0:
                sub_both["align_1000"] = np.where(
                    sub_both["break_1000"] == sub_both["break_1100"],
                    "SAME direction",
                    "OPPOSITE direction"
                )
                analyze_group(sub_both, "vs 1000", "align_1000")

            # --- TRIPLE ALIGNMENT (0900 + 1000 + 1100 all same direction) ---
            print(f"\nTRIPLE ALIGNMENT (0900 + 1000 + 1100 all break same direction):")
            has_all = (subset["break_0900"].notna()
                       & subset["break_1000"].notna()
                       & subset["break_1100"].notna())
            sub_all = subset[has_all].copy()
            if len(sub_all) > 0:
                sub_all["triple"] = np.where(
                    (sub_all["break_0900"] == sub_all["break_1100"])
                    & (sub_all["break_1000"] == sub_all["break_1100"]),
                    "ALL ALIGNED",
                    np.where(
                        (sub_all["break_0900"] != sub_all["break_1100"])
                        & (sub_all["break_1000"] != sub_all["break_1100"]),
                        "BOTH OPPOSITE",
                        "MIXED"
                    )
                )
                analyze_group(sub_all, "Triple", "triple")

            # --- 0900 DOUBLE BREAK + 1100 ---
            print(f"\n0900 DOUBLE BREAK (did 0900 reverse through ORB?):")
            has_db = subset["double_break_0900"].notna()
            sub_db = subset[has_db].copy()
            if len(sub_db) > 0:
                sub_db["db_0900"] = np.where(
                    sub_db["double_break_0900"] == True,
                    "0900 DOUBLE BROKE",
                    "0900 held direction"
                )
                analyze_group(sub_db, "0900 DB", "db_0900")

            # --- 1000 DOUBLE BREAK + 1100 ---
            print(f"\n1000 DOUBLE BREAK (did 1000 reverse through ORB?):")
            has_db = subset["double_break_1000"].notna()
            sub_db = subset[has_db].copy()
            if len(sub_db) > 0:
                sub_db["db_1000"] = np.where(
                    sub_db["double_break_1000"] == True,
                    "1000 DOUBLE BROKE",
                    "1000 held direction"
                )
                analyze_group(sub_db, "1000 DB", "db_1000")

            # --- 0900 OUTCOME (did the 0900 trade win or lose?) ---
            print(f"\n0900 TRADE OUTCOME (did 0900 same-params trade win?):")
            earlier = outcomes_0900[
                (outcomes_0900["entry_model"] == em)
                & (outcomes_0900["rr_target"] == rr)
            ][["trading_day", "earlier_outcome", "earlier_pnl_r"]].drop_duplicates("trading_day")

            merged = subset.merge(earlier, on="trading_day", how="left")
            has_earlier = merged["earlier_outcome"].notna()
            sub_merged = merged[has_earlier].copy()
            if len(sub_merged) > 0:
                analyze_group(sub_merged, "0900 outcome", "earlier_outcome")

            # --- 1000 OUTCOME ---
            print(f"\n1000 TRADE OUTCOME (did 1000 same-params trade win?):")
            earlier = outcomes_1000[
                (outcomes_1000["entry_model"] == em)
                & (outcomes_1000["rr_target"] == rr)
            ][["trading_day", "earlier_outcome", "earlier_pnl_r"]].drop_duplicates("trading_day")

            merged = subset.merge(earlier, on="trading_day", how="left")
            has_earlier = merged["earlier_outcome"].notna()
            sub_merged = merged[has_earlier].copy()
            if len(sub_merged) > 0:
                analyze_group(sub_merged, "1000 outcome", "earlier_outcome")

            # --- COMBO: 0900+1000 both won + 1100 aligned ---
            print(f"\nCOMBO: Earlier sessions won AND 1100 aligned:")
            e0900 = outcomes_0900[
                (outcomes_0900["entry_model"] == em)
                & (outcomes_0900["rr_target"] == rr)
            ][["trading_day", "earlier_outcome"]].drop_duplicates("trading_day")
            e0900 = e0900.rename(columns={"earlier_outcome": "o0900"})

            e1000 = outcomes_1000[
                (outcomes_1000["entry_model"] == em)
                & (outcomes_1000["rr_target"] == rr)
            ][["trading_day", "earlier_outcome"]].drop_duplicates("trading_day")
            e1000 = e1000.rename(columns={"earlier_outcome": "o1000"})

            combo = subset.merge(e0900, on="trading_day", how="left")
            combo = combo.merge(e1000, on="trading_day", how="left")

            # Both earlier sessions won
            both_won = combo[
                (combo["o0900"] == "win") & (combo["o1000"] == "win")
            ]
            if len(both_won) > 0:
                analyze_group(both_won, "0900+1000 both WON")

            # Both lost
            both_lost = combo[
                (combo["o0900"] == "loss") & (combo["o1000"] == "loss")
            ]
            if len(both_lost) > 0:
                analyze_group(both_lost, "0900+1000 both LOST")

            # 0900 won + aligned
            won_aligned = combo[
                (combo["o0900"] == "win")
                & (combo["break_0900"] == combo["break_1100"])
            ]
            if len(won_aligned) > 0:
                analyze_group(won_aligned, "0900 WON + aligned")

            # 0900 lost + 1100 opposite
            lost_opposite = combo[
                (combo["o0900"] == "loss")
                & (combo["break_0900"] != combo["break_1100"])
                & combo["break_0900"].notna()
            ]
            if len(lost_opposite) > 0:
                analyze_group(lost_opposite, "0900 LOST + 1100 opposite")


def main():
    parser = argparse.ArgumentParser(description="1100 conditional analysis")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    args = parser.parse_args()

    run_analysis(args.db_path, args.start, args.end)


if __name__ == "__main__":
    main()
