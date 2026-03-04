#!/usr/bin/env python3
"""
P4: E3 x Direction at 1000 — does filtering to LONG improve E3 expectancy?

Hypothesis: 1000 session has documented LONG bias at E1. E3 (retrace entry) on
LONG days = buying the dip into an upward-biased session. Should have higher WR
and ExpR than directionally-naive E3.

break_dir is known at entry time (the break has already occurred when E3 waits
for retrace), so this is NOT look-ahead.

Usage:
    python research/research_e3_direction.py
    python research/research_e3_direction.py --db-path C:/db/gold.db
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import false_discovery_control

from pipeline.paths import GOLD_DB_PATH
from pipeline.log import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_CSV = OUTPUT_DIR / "e3_direction_results.csv"
OUTPUT_MD  = OUTPUT_DIR / "e3_direction_summary.md"

# Minimum N to report a group
MIN_N = 30

# BH FDR threshold
BH_Q = 0.10

# Sessions to test — 1000 is primary hypothesis; others as comparison
TARGET_SESSIONS = ["1000", "0900", "1800"]

# Instruments
INSTRUMENTS = ["MGC", "MES", "MNQ"]


def load_e3_outcomes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Load E3 outcomes joined to break_dir from daily_features.

    CRITICAL: join includes orb_minutes to prevent 3x row inflation.
    break_dir is known at entry time (break already occurred before E3 retrace wait).
    """
    return con.execute("""
        SELECT
            oo.symbol,
            oo.orb_label        AS session,
            oo.rr_target,
            oo.confirm_bars,
            oo.outcome,
            oo.pnl_r,
            oo.trading_day,
            EXTRACT(YEAR FROM oo.trading_day) AS year,
            df.us_dst,
            CASE oo.orb_label
                WHEN '0900' THEN df.orb_0900_break_dir
                WHEN '1000' THEN df.orb_1000_break_dir
                WHEN '1800' THEN df.orb_1800_break_dir
                ELSE NULL
            END AS break_dir
        FROM orb_outcomes oo
        JOIN daily_features df
          ON  oo.symbol      = df.symbol
          AND oo.trading_day = df.trading_day
          AND oo.orb_minutes = df.orb_minutes          -- prevents 3x row inflation
        WHERE oo.entry_model  = 'E3'
          AND oo.orb_minutes  = 5
          AND oo.orb_label    IN ('1000', '0900', '1800')
          AND oo.outcome      IS NOT NULL
          AND oo.pnl_r        IS NOT NULL
    """).fetchdf()


def compute_stats(series: pd.Series, label: str) -> dict:
    """Compute avg_r, win_rate, N and t-test for a pnl_r series."""
    n = len(series)
    if n < MIN_N:
        return None
    avg_r = series.mean()
    win_rate = (series > 0).mean()
    t_stat, p_val = scipy_stats.ttest_1samp(series.dropna(), 0.0)
    return {
        "label": label,
        "n": n,
        "avg_r": round(avg_r, 4),
        "win_rate": round(win_rate, 4),
        "t_stat": round(t_stat, 4),
        "p_val": round(p_val, 6),
    }


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (symbol, session, rr_target, confirm_bars):
      Compare ALL vs LONG vs SHORT.
      Also check LONG vs SHORT year-by-year for consistency.
    """
    rows = []

    groups = df.groupby(["symbol", "session", "rr_target", "confirm_bars"])

    for (sym, sess, rr, cb), grp in groups:
        all_stats  = compute_stats(grp["pnl_r"], "ALL")
        if all_stats is None:
            continue

        long_grp  = grp[grp["break_dir"] == "long"]
        short_grp = grp[grp["break_dir"] == "short"]

        long_stats  = compute_stats(long_grp["pnl_r"],  "LONG")
        short_stats = compute_stats(short_grp["pnl_r"], "SHORT")

        # Compute direction split ratio (sanity check — should be ~50/50, not 100/0)
        n_long  = len(long_grp)
        n_short = len(short_grp)
        n_total = len(grp)
        long_pct = n_long / n_total if n_total > 0 else 0

        for direction, stats in [("ALL", all_stats), ("LONG", long_stats), ("SHORT", short_stats)]:
            if stats is None:
                continue
            rows.append({
                "symbol":       sym,
                "session":      sess,
                "rr_target":    rr,
                "confirm_bars": int(cb),
                "direction":    direction,
                "n":            stats["n"],
                "avg_r":        stats["avg_r"],
                "win_rate":     stats["win_rate"],
                "t_stat":       stats["t_stat"],
                "p_val":        stats["p_val"],
                "long_pct":     round(long_pct, 3),  # sanity: ~0.5 expected
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # BH FDR across all tested rows
    p_vals = result["p_val"].values
    result["p_bh"] = false_discovery_control(p_vals, method="bh").round(6)

    return result.sort_values(["symbol", "session", "direction", "rr_target", "confirm_bars"])


def year_by_year(df: pd.DataFrame, sym: str, sess: str) -> pd.DataFrame:
    """Year-by-year breakdown of E3 LONG vs ALL for a given symbol/session."""
    sub = df[
        (df["symbol"]  == sym) &
        (df["session"] == sess) &
        (df["break_dir"].isin(["long", None, "short"]))
    ].copy()

    rows = []
    for year, ygrp in sub.groupby("year"):
        for direction, dgrp in [("ALL", ygrp), ("LONG", ygrp[ygrp["break_dir"] == "long"])]:
            n = len(dgrp)
            if n < 10:
                continue
            rows.append({
                "year":      int(year),
                "direction": direction,
                "n":         n,
                "avg_r":     round(dgrp["pnl_r"].mean(), 4),
                "win_rate":  round((dgrp["pnl_r"] > 0).mean(), 4),
            })

    return pd.DataFrame(rows)


def write_summary(result: pd.DataFrame, df_raw: pd.DataFrame) -> None:
    """Write markdown summary."""
    long_only = result[result["direction"] == "LONG"]
    survivors = long_only[
        (long_only["p_bh"] <= BH_Q) & (long_only["avg_r"] > 0)
    ]
    # Also find where LONG > ALL meaningfully
    long_vs_all = (
        result[result["direction"].isin(["LONG", "ALL"])]
        .pivot_table(index=["symbol","session","rr_target","confirm_bars"],
                     columns="direction", values="avg_r")
        .dropna()
        .assign(lift=lambda x: x["LONG"] - x["ALL"])
        .sort_values("lift", ascending=False)
    )

    lines = [
        "# P4: E3 x Direction at 1000",
        "",
        "## Hypothesis",
        "1000 session has documented LONG bias at E1. E3 (retrace entry) on LONG days",
        "= buying the dip into an upward-biased session.",
        "break_dir is known at entry time — NOT look-ahead.",
        "",
        "## Method",
        "- E3 outcomes joined to daily_features.orb_{label}_break_dir",
        "- JOIN includes orb_minutes (prevents 3x row inflation)",
        "- Sessions tested: 1000 (primary), 0900, 1800",
        f"- BH FDR at q={BH_Q} across {len(result)} rows",
        f"- Min N = {MIN_N}",
        "",
        f"## Summary: {len(result)} groups tested",
        f"- LONG groups with BH-sig positive avg_r: {len(survivors)}",
        "",
    ]

    if not survivors.empty:
        lines += [
            "## SURVIVED — LONG direction filter improves E3",
            "",
            "| Symbol | Session | RR | CB | N_long | avg_r_long | WR_long | p_bh |",
            "|--------|---------|----|----|--------|------------|---------|------|",
        ]
        for _, r in survivors.iterrows():
            lines.append(
                f"| {r.symbol} | {r.session} | {r.rr_target} | {r.confirm_bars} | "
                f"{r.n} | {r.avg_r:+.4f} | {r.win_rate:.1%} | {r.p_bh:.4f} |"
            )
        lines += [""]
    else:
        lines += [
            "## DID NOT SURVIVE",
            "",
            "No LONG-filtered E3 groups passed BH FDR with avg_r > 0.",
            "E3 x LONG direction filter is NOT supported as a standalone improvement.",
            "",
        ]

    # Top lifts (LONG - ALL)
    if not long_vs_all.empty:
        top_lifts = long_vs_all.head(10).reset_index()
        lines += [
            "## Top ExpR Lifts: LONG vs ALL",
            "",
            "| Symbol | Session | RR | CB | avg_r_ALL | avg_r_LONG | lift |",
            "|--------|---------|----|----|-----------|------------|------|",
        ]
        for _, r in top_lifts.iterrows():
            lines.append(
                f"| {r.symbol} | {r.session} | {r.rr_target} | {int(r.confirm_bars)} | "
                f"{r.ALL:+.4f} | {r.LONG:+.4f} | {r.lift:+.4f} |"
            )
        lines += [""]

    # Year-by-year for MGC 1000 and MNQ 1000
    for sym, sess in [("MGC", "1000"), ("MNQ", "1000"), ("MES", "1000")]:
        yby = year_by_year(df_raw, sym, sess)
        if yby.empty:
            continue
        lines += [
            f"## {sym} {sess} — Year-by-Year (ALL vs LONG)",
            "",
            "| Year | Direction | N | avg_r | WR |",
            "|------|-----------|---|-------|----|",
        ]
        for _, r in yby.iterrows():
            lines.append(
                f"| {int(r.year)} | {r.direction} | {r.n} | {r.avg_r:+.4f} | {r.win_rate:.1%} |"
            )
        lines += [""]

    lines += [
        "## CAVEATS",
        "- E3 has lower fill rate than E1 — sample is smaller",
        "- break_dir is NOT look-ahead: the break has already occurred when E3 waits for retrace",
        "- DST not split for 0900/1800 (1000 is clean — no DST contamination)",
        "- ORB_G4+ filter not applied — validated strategy quality days may differ",
        "",
        "## NEXT STEPS",
        "- If SURVIVED: add direction_filter='long' to E3 LiveStrategySpec entries",
        "- Sensitivity: does lift hold at RR2.0 and RR2.5? If only RR1.0 → curve-fit risk",
        "- Check: is LONG pct ~50%? If not, investigate data issue",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Summary written to {OUTPUT_MD}")


def main():
    parser = argparse.ArgumentParser(description="P4: E3 x direction analysis")
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else Path(
        os.environ.get("DUCKDB_PATH", str(GOLD_DB_PATH))
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("P4: E3 x DIRECTION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"DB: {db_path}")

    with duckdb.connect(str(db_path), read_only=True) as con:
        df = load_e3_outcomes(con)

    logger.info(f"E3 outcomes loaded: {len(df):,}")
    logger.info(f"  by session: {df.groupby('session').size().to_dict()}")
    logger.info(f"  by symbol:  {df.groupby('symbol').size().to_dict()}")
    logger.info(f"  break_dir:  {df['break_dir'].value_counts().to_dict()}")

    # Sanity: long_pct should be ~50%
    long_pct_overall = (df["break_dir"] == "long").mean()
    logger.info(f"  long_pct:   {long_pct_overall:.1%} (expect ~50%)")
    if long_pct_overall > 0.75 or long_pct_overall < 0.25:
        logger.warning("WARN: long_pct far from 50% — possible data issue")

    result = analyze(df)

    if result.empty:
        logger.info("No groups with N >= 30. Nothing to report.")
        sys.exit(0)

    result.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Results: {len(result)} rows -> {OUTPUT_CSV}")

    write_summary(result, df)

    # Print key findings
    long_only = result[result["direction"] == "LONG"]
    survivors = long_only[(long_only["p_bh"] <= BH_Q) & (long_only["avg_r"] > 0)]

    logger.info("-" * 60)
    logger.info(f"LONG groups BH-sig positive: {len(survivors)}/{len(long_only)}")

    # Show 1000 session comparison specifically
    for sym in INSTRUMENTS:
        sub = result[
            (result["symbol"] == sym) &
            (result["session"] == "1000") &
            (result["confirm_bars"] == 1)
        ].pivot_table(
            index="rr_target", columns="direction", values=["avg_r", "n"]
        )
        if not sub.empty:
            logger.info(f"\n{sym} 1000 CB1 (ALL vs LONG vs SHORT):")
            logger.info(f"\n{sub.to_string()}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
