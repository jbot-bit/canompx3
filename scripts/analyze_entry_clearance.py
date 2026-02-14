#!/usr/bin/env python3
"""
Entry Price Clearance Filter: does the entry clear prior session ORB levels?

Hypothesis: 1800 has 81% double-break rate because it's an "Inside Session"
trapped within the 0900/1000 range. If entry is physically inside the prior
range, chop is structurally guaranteed. Requiring entry to CLEAR the prior
range should filter out false breakouts.

For each trade:
  prior_high = MAX(orb_high for all earlier sessions that day)
  prior_low  = MIN(orb_low  for all earlier sessions that day)

  CLEAR: entry > prior_high (long) or entry < prior_low (short)
  CHOP:  entry inside [prior_low, prior_high]

Prior levels are CUMULATIVE -- never reset. Each session uses ALL earlier
sessions, not just the immediately preceding one.

Read-only research script. No writes to gold.db.

Usage:
    python scripts/analyze_entry_clearance.py --db-path C:/db/gold.db
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._alt_strategy_utils import compute_strategy_metrics


# Session ordering within a trading day (09:00 Brisbane boundary)
SESSION_ORDER = ["0900", "1000", "1100", "1800", "2300", "0030"]

# For each session, which earlier sessions form the prior levels
PRIOR_SESSIONS = {
    "1000": ["0900"],
    "1100": ["0900", "1000"],
    "1800": ["0900", "1000", "1100"],
    "2300": ["0900", "1000", "1100", "1800"],
    "0030": ["0900", "1000", "1100", "1800", "2300"],
}

# ORB size filter thresholds (from config.py)
SIZE_TIERS = {"G2": 2.0, "G4": 4.0, "G6": 6.0, "G8": 8.0}

DOW_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(db_path: Path, session: str, start: date, end: date) -> pd.DataFrame:
    """Load outcomes for a session joined with daily_features for prior levels."""
    prior = PRIOR_SESSIONS.get(session)
    if not prior:
        raise ValueError(f"No prior sessions defined for {session}")

    # Build SELECT columns for all prior session ORB highs/lows
    prior_cols = []
    for s in prior:
        prior_cols.append(f"d.orb_{s}_high AS prior_{s}_high")
        prior_cols.append(f"d.orb_{s}_low AS prior_{s}_low")
    prior_cols_sql = ",\n                ".join(prior_cols)

    # Also grab this session's ORB info
    sql = f"""
        SELECT
            o.trading_day,
            o.entry_model,
            o.rr_target,
            o.confirm_bars,
            o.outcome,
            o.pnl_r,
            o.entry_price,
            o.stop_price,
            o.target_price,

            -- This session's ORB info
            d.orb_{session}_size AS orb_size,
            d.orb_{session}_break_dir AS break_dir,
            d.orb_{session}_double_break AS double_break,
            d.orb_{session}_high AS orb_high,
            d.orb_{session}_low AS orb_low,

            -- Prior session levels
            {prior_cols_sql},

            -- Day of week
            EXTRACT(ISODOW FROM o.trading_day) AS dow

        FROM orb_outcomes o
        JOIN daily_features d
            ON o.symbol = d.symbol
            AND o.trading_day = d.trading_day
            AND d.orb_minutes = 5
        WHERE o.symbol = 'MGC'
            AND o.orb_minutes = 5
            AND o.orb_label = ?
            AND o.entry_ts IS NOT NULL
            AND o.outcome IS NOT NULL
            AND o.pnl_r IS NOT NULL
            AND o.trading_day BETWEEN ? AND ?
        ORDER BY o.trading_day
    """

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(sql, [session, start, end]).fetchdf()
    finally:
        con.close()

    # Deduplicate E3 CB>1 (CB1-CB5 on E3 are ~identical trades)
    df = df[~((df["entry_model"] == "E3") & (df["confirm_bars"] > 1))].copy()

    # Compute composite prior_high and prior_low
    high_cols = [f"prior_{s}_high" for s in prior]
    low_cols = [f"prior_{s}_low" for s in prior]

    # Use nanmax/nanmin so missing sessions don't poison the composite
    df["prior_high"] = df[high_cols].max(axis=1)
    df["prior_low"] = df[low_cols].min(axis=1)

    # Classify: CLEAR vs CHOP
    # CLEAR = entry escaped the prior structure
    # CHOP = entry trapped inside prior range
    has_prior = df["prior_high"].notna() & df["prior_low"].notna()
    df["clearance"] = np.where(
        ~has_prior,
        "NO_PRIOR",
        np.where(
            ((df["break_dir"] == "long") & (df["entry_price"] > df["prior_high"]))
            | ((df["break_dir"] == "short") & (df["entry_price"] < df["prior_low"])),
            "CLEAR",
            "CHOP",
        ),
    )

    # Margin: how far entry is from nearest prior level (in points)
    df["clearance_margin"] = np.where(
        df["break_dir"] == "long",
        df["entry_price"] - df["prior_high"],
        df["prior_low"] - df["entry_price"],
    )

    print(f"Loaded {len(df)} {session} outcomes ({df['trading_day'].nunique()} days)")
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt(m: dict, db_rate: float = None) -> str:
    """Format metrics dict as a one-line summary."""
    if m is None or m["n"] == 0:
        return "N=0"
    s = (f"N={m['n']:<5d}  WR={m['wr']*100:5.1f}%  ExpR={m['expr']:+.3f}  "
         f"Sharpe={m['sharpe']:.3f}  MaxDD={m['maxdd']:.1f}R  Total={m['total']:+.1f}R")
    if db_rate is not None:
        s += f"  DB={db_rate*100:.0f}%"
    return s


def db_rate(df: pd.DataFrame) -> float:
    """Compute double-break rate for a subset."""
    valid = df["double_break"].notna()
    if valid.sum() == 0:
        return 0.0
    return float(df.loc[valid, "double_break"].astype(bool).mean())


def report_subset(df: pd.DataFrame, label: str):
    """Print one-line metrics for a subset."""
    pnls = df["pnl_r"].values
    m = compute_strategy_metrics(pnls)
    dbr = db_rate(df)
    print(f"  {label:30s}  {fmt(m, dbr)}")


def report_session(df: pd.DataFrame, session: str):
    """Full report for one session."""
    prior = PRIOR_SESSIONS[session]
    prior_label = " + ".join(prior)

    # Group by (entry_model, rr_target)
    for em in ["E1", "E3"]:
        for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
            if em == "E3":
                subset = df[(df["entry_model"] == em) & (df["rr_target"] == rr)
                            & (df["confirm_bars"] == 1)]
            else:  # E1
                subset = df[(df["entry_model"] == em) & (df["rr_target"] == rr)
                            & (df["confirm_bars"] == 2)]

            if len(subset) < 30:
                continue

            print(f"\n{'='*100}")
            print(f"{session} {em} RR{rr}  |  Prior levels: MAX({prior_label})")
            print(f"{'='*100}")

            # --- Baseline ---
            report_subset(subset, "Baseline (all)")

            # --- CLEAR vs CHOP ---
            for tag in ["CLEAR", "CHOP", "NO_PRIOR"]:
                sub = subset[subset["clearance"] == tag]
                if len(sub) > 0:
                    report_subset(sub, tag)

            # --- CLEAR by ORB size tier ---
            clear = subset[subset["clearance"] == "CLEAR"]
            if len(clear) >= 10:
                print(f"\n  CLEAR by ORB size tier:")
                for tier_name, threshold in SIZE_TIERS.items():
                    tier_sub = clear[clear["orb_size"] >= threshold]
                    if len(tier_sub) >= 5:
                        pnls = tier_sub["pnl_r"].values
                        m = compute_strategy_metrics(pnls)
                        dbr = db_rate(tier_sub)
                        print(f"    {tier_name}+ (>={threshold}pt):  {fmt(m, dbr)}")

            # --- CLEAR by day of week ---
            if len(clear) >= 10:
                print(f"\n  CLEAR by day of week:")
                for dow_num in sorted(DOW_NAMES.keys()):
                    dow_sub = clear[clear["dow"] == dow_num]
                    if len(dow_sub) >= 3:
                        pnls = dow_sub["pnl_r"].values
                        m = compute_strategy_metrics(pnls)
                        if m:
                            print(f"    {DOW_NAMES[dow_num]}:  {fmt(m)}")

            # --- Clearance margin distribution for CLEAR trades ---
            if len(clear) >= 10:
                margins = clear["clearance_margin"].values
                print(f"\n  CLEAR margin stats: "
                      f"mean={np.mean(margins):.1f}pt  "
                      f"median={np.median(margins):.1f}pt  "
                      f"min={np.min(margins):.1f}pt  "
                      f"max={np.max(margins):.1f}pt")

                # Margin buckets
                print(f"  CLEAR by margin bucket:")
                for lo, hi, label in [(0, 2, "0-2pt"), (2, 5, "2-5pt"),
                                       (5, 10, "5-10pt"), (10, 999, "10+pt")]:
                    bucket = clear[(clear["clearance_margin"] >= lo)
                                   & (clear["clearance_margin"] < hi)]
                    if len(bucket) >= 3:
                        pnls = bucket["pnl_r"].values
                        m = compute_strategy_metrics(pnls)
                        dbr = db_rate(bucket)
                        print(f"    {label:10s}  {fmt(m, dbr)}")

            # --- CHOP breakdown: how deep inside prior range? ---
            chop = subset[subset["clearance"] == "CHOP"]
            if len(chop) >= 10:
                chop_margins = chop["clearance_margin"].values  # negative = inside
                print(f"\n  CHOP margin stats: "
                      f"mean={np.mean(chop_margins):.1f}pt  "
                      f"median={np.median(chop_margins):.1f}pt")

                # CHOP: near edge vs deep inside
                near_edge = chop[chop["clearance_margin"] >= -2]
                deep = chop[chop["clearance_margin"] < -2]
                if len(near_edge) >= 3:
                    report_subset(near_edge, "CHOP near edge (>-2pt)")
                if len(deep) >= 3:
                    report_subset(deep, "CHOP deep inside (<-2pt)")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results: dict):
    """Print a compact summary table across all sessions."""
    print(f"\n\n{'#'*100}")
    print(f"SUMMARY: CLEAR vs CHOP across all sessions")
    print(f"{'#'*100}")
    print(f"\n{'Session':<8} {'EM':<4} {'RR':<5} {'Tag':<8} "
          f"{'N':>5} {'WR':>6} {'ExpR':>7} {'Sharpe':>7} {'MaxDD':>7} {'DB%':>5}")
    print("-" * 80)

    for key in sorted(all_results.keys()):
        session, em, rr = key
        data = all_results[key]
        for tag in ["Baseline", "CLEAR", "CHOP"]:
            if tag in data:
                m, dbr = data[tag]
                if m and m["n"] >= 10:
                    print(f"{session:<8} {em:<4} {rr:<5.1f} {tag:<8} "
                          f"{m['n']:>5d} {m['wr']*100:>5.1f}% {m['expr']:>+6.3f} "
                          f"{m['sharpe']:>7.3f} {m['maxdd']:>6.1f}R {dbr*100:>4.0f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(db_path: Path, start: date, end: date):
    """Run entry clearance analysis for all applicable sessions."""
    all_results = {}

    for session in ["1000", "1100", "1800", "2300", "0030"]:
        print(f"\n\n{'#'*100}")
        print(f"SESSION: {session}  |  Prior sessions: {', '.join(PRIOR_SESSIONS[session])}")
        print(f"{'#'*100}")

        df = load_data(db_path, session, start, end)
        if df.empty:
            print(f"  No data for {session}")
            continue

        report_session(df, session)

        # Collect summary data
        for em in ["E1", "E3"]:
            for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
                if em == "E3":
                    subset = df[(df["entry_model"] == em) & (df["rr_target"] == rr)
                                & (df["confirm_bars"] == 1)]
                else:
                    subset = df[(df["entry_model"] == em) & (df["rr_target"] == rr)
                                & (df["confirm_bars"] == 2)]

                if len(subset) < 30:
                    continue

                key = (session, em, rr)
                all_results[key] = {}

                pnls = subset["pnl_r"].values
                all_results[key]["Baseline"] = (
                    compute_strategy_metrics(pnls), db_rate(subset))

                for tag in ["CLEAR", "CHOP"]:
                    sub = subset[subset["clearance"] == tag]
                    if len(sub) > 0:
                        all_results[key][tag] = (
                            compute_strategy_metrics(sub["pnl_r"].values),
                            db_rate(sub))

    print_summary(all_results)


def main():
    parser = argparse.ArgumentParser(
        description="Entry Price Clearance Filter analysis")
    parser.add_argument("--db-path", type=Path, default=Path("C:/db/gold.db"))
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 4))
    args = parser.parse_args()

    print("Entry Price Clearance Filter Analysis")
    print(f"Database: {args.db_path}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Sessions: {', '.join(PRIOR_SESSIONS.keys())}")
    print(f"Prior levels are CUMULATIVE (MAX/MIN of ALL earlier sessions)")

    run_analysis(args.db_path, args.start, args.end)


if __name__ == "__main__":
    main()
