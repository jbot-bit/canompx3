#!/usr/bin/env python3
"""
P5b: Time-exit analysis — do trades held past T80 have negative expectancy?

T80 = time by which 80% of winning trades have already exited (from winner_speed_summary.csv).
Hypothesis: trades surviving past T80 are predominantly losers. If avg_r < 0 past T80,
a time-stop at T80 improves Sharpe for free with no change to entry logic.

Usage:
    python research/research_time_exit.py
    python research/research_time_exit.py --db-path C:/db/gold.db
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
OUTPUT_CSV = OUTPUT_DIR / "time_exit_results.csv"
OUTPUT_MD  = OUTPUT_DIR / "time_exit_summary.md"

# Minimum N required in the past-T80 group to report a finding
MIN_N_AFTER = 30

# p-value threshold after BH correction
BH_Q = 0.10


def load_t80(csv_path: Path) -> pd.DataFrame:
    """Load T80 values from winner_speed_summary.csv."""
    df = pd.read_csv(csv_path)
    # Keep only CORE/PRELIMINARY sample classes (enough data)
    df = df[df["sample_class"].isin(["CORE", "PRELIMINARY", "HIGH-CONFIDENCE"])]
    # Return key columns
    return df[["symbol", "session", "rr_target", "confirm_bars", "t80", "n_winners"]].copy()


def load_outcomes(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Load all non-null outcomes with time-to-exit from orb_outcomes."""
    return con.execute("""
        SELECT
            symbol,
            orb_label AS session,
            rr_target,
            confirm_bars,
            outcome,
            pnl_r,
            date_diff('second', entry_ts, exit_ts) / 60.0 AS minutes_to_exit
        FROM orb_outcomes
        WHERE orb_minutes = 5
          AND outcome IN ('win', 'loss', 'scratch')
          AND entry_ts IS NOT NULL
          AND exit_ts IS NOT NULL
          AND exit_ts > entry_ts
    """).fetchdf()


def analyze_time_exit(outcomes: pd.DataFrame, t80_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (symbol, session, rr_target, confirm_bars) group:
      - Split trades into before_t80 and after_t80
      - Compute avg_r, win_rate, N for each half
      - t-test: H0: avg_r_after == 0
    """
    rows = []

    for _, t80_row in t80_df.iterrows():
        sym     = t80_row["symbol"]
        sess    = t80_row["session"]
        rr      = t80_row["rr_target"]
        cb      = t80_row["confirm_bars"]
        t80_val = t80_row["t80"]

        if pd.isna(t80_val) or t80_val <= 0:
            continue

        grp = outcomes[
            (outcomes["symbol"]       == sym)  &
            (outcomes["session"]      == sess) &
            (outcomes["rr_target"]    == rr)   &
            (outcomes["confirm_bars"] == cb)
        ]

        if grp.empty:
            continue

        before = grp[grp["minutes_to_exit"] <= t80_val]
        after  = grp[grp["minutes_to_exit"] >  t80_val]

        n_before = len(before)
        n_after  = len(after)

        if n_after < MIN_N_AFTER:
            continue

        avg_r_before = before["pnl_r"].mean()
        avg_r_after  = after["pnl_r"].mean()
        wr_before    = (before["outcome"] == "win").mean() if n_before > 0 else float("nan")
        wr_after     = (after["outcome"]  == "win").mean()

        # One-sample t-test: is avg_r_after significantly different from 0?
        t_stat, p_val = scipy_stats.ttest_1samp(after["pnl_r"].dropna(), 0.0)

        rows.append({
            "symbol":        sym,
            "session":       sess,
            "rr_target":     rr,
            "confirm_bars":  cb,
            "t80_minutes":   t80_val,
            "n_before":      n_before,
            "n_after":       n_after,
            "avg_r_before":  round(avg_r_before, 4),
            "avg_r_after":   round(avg_r_after,  4),
            "wr_before":     round(wr_before,     4),
            "wr_after":      round(wr_after,       4),
            "t_stat":        round(t_stat, 4),
            "p_val":         round(p_val, 6),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR correction across all groups
    p_vals = result["p_val"].values
    rejected = false_discovery_control(p_vals, method="bh") <= BH_Q
    result["p_bh"] = false_discovery_control(p_vals, method="bh")
    result["p_bh"] = result["p_bh"].round(6)

    # Verdict: SURVIVES if BH-significant AND avg_r_after < 0 AND N_after >= MIN_N_AFTER
    def verdict(row):
        if row["p_bh"] <= BH_Q and row["avg_r_after"] < 0:
            return "SURVIVES — time-stop recommended"
        elif row["p_val"] <= 0.05 and row["avg_r_after"] < 0:
            return "NOTABLE (p<0.05, not BH-sig)"
        elif row["avg_r_after"] < 0:
            return "DIRECTIONAL (not significant)"
        else:
            return "NO-BENEFIT (avg_r_after >= 0)"
    result["verdict"] = result.apply(verdict, axis=1)

    return result.sort_values(["symbol", "session", "rr_target", "confirm_bars"])


def write_summary(result: pd.DataFrame, t80_df: pd.DataFrame) -> None:
    """Write markdown summary."""
    survivors   = result[result["verdict"].str.startswith("SURVIVES")]
    notable     = result[result["verdict"].str.startswith("NOTABLE")]
    directional = result[result["verdict"].str.startswith("DIRECTIONAL")]

    lines = [
        "# P5b: Time-Exit Analysis",
        "",
        "## Hypothesis",
        "Trades surviving past T80 (80th percentile of winner exit times) are predominantly",
        "losers. If avg_r < 0 past T80, a time-stop at T80 improves Sharpe with no entry change.",
        "",
        f"## Method",
        f"- T80 values from: `research/output/winner_speed_summary.csv`",
        f"- Outcomes from: `orb_outcomes` (outcome IN win/loss/scratch, orb_minutes=5)",
        f"- BH FDR correction at q={BH_Q} across {len(result)} tested groups",
        f"- Minimum N_after = {MIN_N_AFTER} to report",
        "",
        f"## Results: {len(result)} groups tested",
        f"- SURVIVES (BH-sig, avg_r < 0): {len(survivors)}",
        f"- NOTABLE (p<0.05, not BH-sig): {len(notable)}",
        f"- DIRECTIONAL (negative, not sig): {len(directional)}",
        f"- NO-BENEFIT (avg_r_after >= 0): {len(result) - len(survivors) - len(notable) - len(directional)}",
        "",
    ]

    if not survivors.empty:
        lines += [
            "## SURVIVED — Time-Stop Recommended",
            "",
            "| Symbol | Session | RR | CB | T80 | N_after | avg_r_after | WR_after | p_bh |",
            "|--------|---------|----|----|-----|---------|-------------|----------|------|",
        ]
        for _, r in survivors.iterrows():
            lines.append(
                f"| {r.symbol} | {r.session} | {r.rr_target} | {int(r.confirm_bars)} | "
                f"{r.t80_minutes:.0f}m | {r.n_after} | {r.avg_r_after:+.4f} | "
                f"{r.wr_after:.1%} | {r.p_bh:.4f} |"
            )
        lines += [
            "",
            "**Mechanism:** Past T80, the market has moved on. Remaining open positions are",
            "dead exposure — the breakout momentum has dissipated and mean-reversion dominates.",
            "A time-stop at T80 exits these positions before they turn negative.",
            "",
            "**What could kill it:** If momentum sessions (e.g. MGC 0900 trending day) continue",
            "past T80, a hard time-stop would prematurely exit winners. Recommend per-session",
            "T80 values rather than a global cutoff.",
            "",
        ]
    else:
        lines += [
            "## DID NOT SURVIVE",
            "",
            "No groups passed BH FDR correction with avg_r_after < 0.",
            "Time-stop at T80 is NOT supported by this data.",
            "",
        ]

    if not notable.empty:
        lines += [
            "## Notable (p<0.05, not BH-sig)",
            "",
            "| Symbol | Session | RR | CB | T80 | N_after | avg_r_after | p_val |",
            "|--------|---------|----|----|-----|---------|-------------|-------|",
        ]
        for _, r in notable.iterrows():
            lines.append(
                f"| {r.symbol} | {r.session} | {r.rr_target} | {int(r.confirm_bars)} | "
                f"{r.t80_minutes:.0f}m | {r.n_after} | {r.avg_r_after:+.4f} | {r.p_val:.4f} |"
            )
        lines.append("")

    lines += [
        "## CAVEATS",
        "- T80 from winner_speed_summary.csv uses ALL winners regardless of filter",
        "- Filter (ORB_G4/G5/etc.) not applied — T80 may differ for validated strategies only",
        "- Small N_after for high-RR groups reduces statistical power",
        "- DST regime not split (1000 session is clean; 0900/1800 may differ by DST half)",
        "",
        "## NEXT STEPS",
        "- If any group SURVIVES: implement time_stop_minutes per session in paper_trader.py",
        "- Run separate analysis splitting by DST regime for 0900/1800",
        "- Compare T80 for G4+ only vs all outcomes (filter may concentrate earlier winners)",
    ]

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Summary written to {OUTPUT_MD}")


def main():
    parser = argparse.ArgumentParser(description="P5b: Time-exit analysis past T80")
    parser.add_argument("--db-path", type=str, default=None, help="Path to gold.db")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else Path(
        os.environ.get("DUCKDB_PATH", str(GOLD_DB_PATH))
    )

    winner_speed_csv = OUTPUT_DIR / "winner_speed_summary.csv"
    if not winner_speed_csv.exists():
        logger.error(f"FATAL: {winner_speed_csv} not found. Run research_winner_speed.py first.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("P5b: TIME-EXIT ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"DB: {db_path}")

    # Load T80 reference values
    t80_df = load_t80(winner_speed_csv)
    logger.info(f"T80 reference: {len(t80_df)} groups (CORE/PRELIMINARY only)")

    # Load outcomes
    with duckdb.connect(str(db_path), read_only=True) as con:
        outcomes = load_outcomes(con)

    logger.info(f"Outcomes loaded: {len(outcomes):,} trades")
    logger.info(f"  win:     {(outcomes.outcome == 'win').sum():,}")
    logger.info(f"  loss:    {(outcomes.outcome == 'loss').sum():,}")
    logger.info(f"  scratch: {(outcomes.outcome == 'scratch').sum():,}")

    # Analyse
    result = analyze_time_exit(outcomes, t80_df)

    if result.empty:
        logger.info("No groups had N_after >= 30. Nothing to report.")
        sys.exit(0)

    # Output
    result.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Results: {len(result)} groups → {OUTPUT_CSV}")

    write_summary(result, t80_df)

    # Print key findings
    survivors = result[result["verdict"].str.startswith("SURVIVES")]
    logger.info("-" * 60)
    logger.info(f"SURVIVED (BH-sig, avg_r < 0): {len(survivors)}/{len(result)} groups")

    if not survivors.empty:
        logger.info("Groups where time-stop is recommended:")
        for _, r in survivors.iterrows():
            logger.info(
                f"  {r.symbol} {r.session} RR{r.rr_target} CB{int(r.confirm_bars)}: "
                f"T80={r.t80_minutes:.0f}m, avg_r_after={r.avg_r_after:+.4f}, "
                f"N_after={r.n_after}, p_bh={r.p_bh:.4f}"
            )
    else:
        logger.info("DID NOT SURVIVE — time-stop not supported by data")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
