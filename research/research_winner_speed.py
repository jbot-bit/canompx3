#!/usr/bin/env python3
"""
Winner Speed Profiling — Time-to-target analysis for ORB winning trades.

For each winning trade, computes how long (in minutes) from entry to target hit.
Reports cumulative % of winners that have hit by each checkpoint, and
T50/T80/T90 percentiles per (symbol, session, rr_target, confirm_bars).

Key question: if 80% of winners hit within 2h, holding positions open for 7h
is dead exposure — remaining open positions are overwhelmingly losers.

Read-only: no writes to gold.db.

Output:
  research/output/winner_speed_summary.csv     -- T50/T80/T90 per group
  research/output/winner_speed_cumulative.csv  -- cumulative % by checkpoint

Usage:
  python research/research_winner_speed.py
  python research/research_winner_speed.py --db-path C:/db/gold.db
  python research/research_winner_speed.py --instruments MGC --sessions 0900 1000
"""

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import pandas as pd

from pipeline.paths import GOLD_DB_PATH

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minutes after entry to measure cumulative hit %
CHECKPOINTS = [5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 240, 300, 360, 420, 480]

# Approximate session max windows (minutes from entry to when we close all positions).
# Used to compute dead_exposure = session_max - T80. These are estimates —
# actual session close is determined by trading_day end logic in outcome_builder.
SESSION_MAX_MINUTES: dict[str, int] = {
    "0900":           480,
    "1000":           480,
    "1100":           420,
    "1130":           360,
    "1800":           420,
    "2300":           360,
    "0030":           300,
    "CME_OPEN":       480,
    "LONDON_OPEN":    420,
    "US_EQUITY_OPEN": 300,
    "US_DATA_OPEN":   240,
    "US_POST_EQUITY": 180,
    "CME_CLOSE":      180,
}
DEFAULT_SESSION_MAX = 480


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_sample(n: int) -> str:
    """Per RESEARCH_RULES.md thresholds."""
    if n < 30:
        return "INVALID"
    if n < 100:
        return "REGIME"
    if n < 200:
        return "PRELIMINARY"
    return "CORE"


def find_percentile(sorted_vals: list[float], pct: float) -> float | None:
    """Value at given percentile (0-100) from a pre-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return None
    idx = int(math.ceil(pct / 100.0 * n)) - 1
    return sorted_vals[max(0, min(idx, n - 1))]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_winners(con, instruments: list[str], sessions: list[str] | None) -> pd.DataFrame:
    """
    Load all winning ORB trades with minutes_to_target from orb_outcomes.

    IMPORTANT: always filters orb_minutes = 5 to avoid triple-row inflation
    from the (5, 15, 30) minuteperiod split in daily_features.
    """
    inst_list = ", ".join(f"'{i}'" for i in instruments)
    sess_filter = ""
    if sessions:
        sess_list = ", ".join(f"'{s}'" for s in sessions)
        sess_filter = f"AND o.orb_label IN ({sess_list})"

    sql = f"""
        SELECT
            o.trading_day,
            o.symbol,
            o.orb_label,
            o.rr_target,
            o.confirm_bars,
            o.entry_model,
            o.pnl_r,
            date_diff('second', o.entry_ts, o.exit_ts) / 60.0 AS minutes_to_target
        FROM orb_outcomes o
        WHERE o.outcome = 'win'
          AND o.orb_minutes = 5
          AND o.symbol IN ({inst_list})
          AND o.entry_ts IS NOT NULL
          AND o.exit_ts IS NOT NULL
          AND o.exit_ts > o.entry_ts
          {sess_filter}
        ORDER BY o.symbol, o.orb_label, o.rr_target, o.trading_day
    """
    return con.execute(sql).fetchdf()


# ---------------------------------------------------------------------------
# Per-group statistics
# ---------------------------------------------------------------------------

def analyze_group(minutes: list[float], symbol: str, session: str,
                  rr_target: float, confirm_bars: int) -> dict:
    """Compute T-percentiles and dead exposure for one group."""
    n = len(minutes)
    session_max = SESSION_MAX_MINUTES.get(session, DEFAULT_SESSION_MAX)

    base = {
        "symbol": symbol, "session": session,
        "rr_target": rr_target, "confirm_bars": confirm_bars,
        "n_winners": n, "sample_class": classify_sample(n),
        "session_max_min": session_max,
    }

    if n == 0:
        return {**base, "t50": None, "t80": None, "t90": None, "t95": None,
                "t_max_obs": None, "dead_exposure_min": None}

    srt = sorted(minutes)
    t80 = find_percentile(srt, 80)

    return {
        **base,
        "t50":              round(find_percentile(srt, 50), 1),
        "t80":              round(t80, 1),
        "t90":              round(find_percentile(srt, 90), 1),
        "t95":              round(find_percentile(srt, 95), 1),
        "t_max_obs":        round(srt[-1], 1),
        "dead_exposure_min": round(session_max - t80, 1),
    }


def build_cumulative(minutes: list[float], symbol: str, session: str,
                     rr_target: float, confirm_bars: int) -> list[dict]:
    """Cumulative % of winners hit by each checkpoint."""
    n = len(minutes)
    if n == 0:
        return []
    rows = []
    for cp in CHECKPOINTS:
        n_hit = sum(1 for m in minutes if m <= cp)
        rows.append({
            "symbol": symbol, "session": session,
            "rr_target": rr_target, "confirm_bars": confirm_bars,
            "checkpoint_min": cp,
            "n_hit": n_hit,
            "n_total": n,
            "pct_hit": round(100.0 * n_hit / n, 1),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Winner speed profiling — time-to-target for ORB wins"
    )
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--instruments", nargs="+", default=["MGC", "MES", "MNQ"])
    parser.add_argument("--sessions", nargs="*", default=None,
                        help="Sessions to include (default: all)")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH

    print(f"\n{'=' * 80}")
    print(f"  WINNER SPEED PROFILING — Time-to-Target Analysis")
    print(f"  Database:    {db_path}")
    print(f"  Instruments: {args.instruments}")
    print(f"  Sessions:    {args.sessions or 'all'}")
    print(f"{'=' * 80}\n")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        print("  Loading winning trades from orb_outcomes...")
        df = load_winners(con, args.instruments, args.sessions)
    finally:
        con.close()

    if df.empty:
        print("  No winning trades found. Check instrument/session arguments.")
        return

    print(f"  {len(df):,} winning trade rows loaded")

    # Validation: flag and drop impossible timings
    bad = df[df["minutes_to_target"] <= 0]
    if not bad.empty:
        print(f"  WARNING: {len(bad)} rows with minutes_to_target <= 0 (data issue) — excluded")
        df = df[df["minutes_to_target"] > 0]

    # Row counts for sanity check
    print("\n  Winners per session:")
    for (sym, sess), grp in df.groupby(["symbol", "orb_label"]):
        print(f"    {sym:>4}  {sess:<14}  {len(grp):>5} rows")

    # Aggregate
    summary_rows: list[dict] = []
    cumulative_rows: list[dict] = []

    for (symbol, session, rr_target, confirm_bars), grp in df.groupby(
        ["symbol", "orb_label", "rr_target", "confirm_bars"], sort=True
    ):
        minutes = grp["minutes_to_target"].dropna().tolist()
        stats = analyze_group(minutes, symbol, session, rr_target, int(confirm_bars))
        summary_rows.append(stats)
        cumulative_rows.extend(build_cumulative(
            minutes, symbol, session, rr_target, int(confirm_bars)
        ))

    summary_df = pd.DataFrame(summary_rows)
    cumulative_df = pd.DataFrame(cumulative_rows)

    # Save
    output_dir = PROJECT_ROOT / "research" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "winner_speed_summary.csv"
    cumulative_path = output_dir / "winner_speed_cumulative.csv"

    summary_df.to_csv(summary_path, index=False, float_format="%.1f")
    cumulative_df.to_csv(cumulative_path, index=False, float_format="%.1f")

    print(f"\n  Saved: {summary_path} ({len(summary_rows)} rows)")
    print(f"  Saved: {cumulative_path} ({len(cumulative_rows)} rows)")

    # -----------------------------------------------------------------------
    # Console: ranked table by dead exposure (worst offenders first)
    # -----------------------------------------------------------------------
    valid = summary_df[
        (summary_df["sample_class"] != "INVALID")
        & summary_df["dead_exposure_min"].notna()
    ].sort_values("dead_exposure_min", ascending=False)

    print(f"\n{'=' * 90}")
    print(f"  WINNER SPEED TABLE  (REGIME+ only, ranked by dead exposure)")
    print(f"  dead_exposure = session_max - T80  |  >120m = consider tighter close")
    print(f"{'=' * 90}")
    print(f"  {'Sym':>4}  {'Session':>14}  {'RR':>4}  {'CB':>3}  {'N':>5}  "
          f"{'Class':>11}  {'T50':>6}  {'T80':>6}  {'T90':>6}  {'SessMax':>8}  {'DeadExp':>8}")
    print(f"  {'-' * 82}")

    for _, r in valid.iterrows():
        def _fmt(v, suffix="m"):
            return f"{v:.0f}{suffix}" if pd.notna(v) else "N/A"
        print(
            f"  {r['symbol']:>4}  {r['session']:>14}  {r['rr_target']:>4.1f}  "
            f"{int(r['confirm_bars']):>3}  {int(r['n_winners']):>5}  "
            f"{r['sample_class']:>11}  {_fmt(r['t50']):>6}  {_fmt(r['t80']):>6}  "
            f"{_fmt(r['t90']):>6}  {int(r['session_max_min']):>7}m  {_fmt(r['dead_exposure_min']):>8}"
        )

    # Flag worst offenders (deduplicate first to avoid index mismatch)
    deduped = valid.drop_duplicates(["symbol", "session"])
    worst = deduped[deduped["dead_exposure_min"] > 120]
    if not worst.empty:
        print(f"\n  HIGH DEAD EXPOSURE (>2h) — consider tighter session close:")
        for _, r in worst.iterrows():
            print(f"    {r['symbol']} {r['session']}: "
                  f"T80={r['t80']:.0f}m, session_max={r['session_max_min']}m, "
                  f"dead_exposure={r['dead_exposure_min']:.0f}m")

    # -----------------------------------------------------------------------
    # Console: cumulative hit % table (one representative per symbol/session)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  CUMULATIVE HIT % BY CHECKPOINT")
    print(f"  (Best REGIME+ combo per symbol/session)")
    print(f"{'=' * 60}")

    seen: set[tuple] = set()
    for _, r in valid.iterrows():
        key = (r["symbol"], r["session"])
        if key in seen:
            continue
        seen.add(key)

        cum = cumulative_df[
            (cumulative_df["symbol"] == r["symbol"])
            & (cumulative_df["session"] == r["session"])
            & (cumulative_df["rr_target"] == r["rr_target"])
            & (cumulative_df["confirm_bars"] == r["confirm_bars"])
        ].sort_values("checkpoint_min")

        if cum.empty:
            continue

        print(f"\n  {r['symbol']} {r['session']}  RR{r['rr_target']:.1f}  "
              f"CB{int(r['confirm_bars'])}  "
              f"(N={int(r['n_winners'])}, {r['sample_class']})")
        print(f"  {'Min':>5}   {'% Hit':>7}   {'Count':>12}")
        print(f"  {'-' * 35}")
        for _, row in cum.iterrows():
            # Mark T80 threshold visually
            marker = " <-- T80" if abs(row["pct_hit"] - 80.0) < 5 else ""
            print(f"  {int(row['checkpoint_min']):>5}m   {row['pct_hit']:>6.1f}%   "
                  f"{int(row['n_hit']):>5}/{int(row['n_total'])}{marker}")

    print(f"\n  Outputs: {output_dir}\n")


if __name__ == "__main__":
    main()
