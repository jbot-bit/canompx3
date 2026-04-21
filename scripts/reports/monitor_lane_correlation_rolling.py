#!/usr/bin/env python3
"""Rolling-correlation tripwire for deployed lanes.

Why this monitor exists:
    Audit `docs/audit/results/2026-04-20-6lane-correlation-concentration-audit.md`
    refuted Carver Ch 11's crisis-correlation-jump warning (p170) for the current
    6-lane MNQ book on backtest data (2019-2026). Forward-looking verification
    requires a live tripwire. If 30-day rolling pairwise correlation on any pair
    exceeds 0.30 for ≥ 10 consecutive days, the book is entering a regime where
    the backtest-derived independence assumption is breaking — investigate
    before it becomes a drawdown event.

Data source:
    paper_trades table — live + shadow trades with pnl_r. Covers the
    deployed lanes from 2026-01-02 onward (shadow mode pre-full-allocator;
    live + shadow post 2026-04-13 allocator wiring).

Scope (read-only):
    - Loads deployed lane list from docs/runtime/lane_allocation.json
      (authoritative source per memory/allocator_wiring_apr13.md).
    - Builds per-lane daily P&L series (pnl_r summed if multiple same-day
      trades per lane — rare but possible).
    - Computes rolling 30-day pairwise Pearson correlation.
    - ALARM if any pair > 0.30 for ≥ 10 consecutive trading days.
    - Writes JSON summary to docs/runtime/lane_correlation_monitor.json
      (runtime artefact; gitignored workflow but file-safe).
    - Never writes to gold.db.

Scheduling:
    Run daily after the nightly backfill completes. Not yet cron-wired;
    see docs/runtime/README_monitors.md for operational notes (to add
    on first wire-up).

Usage:
    DUCKDB_PATH=<gold.db> PYTHONPATH=. python scripts/reports/monitor_lane_correlation_rolling.py
    (or supply --db-path / --json-out for overrides)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH  # noqa: E402

LANE_ALLOC_PATH = PROJECT_ROOT / "docs" / "runtime" / "lane_allocation.json"
DEFAULT_JSON_OUT = PROJECT_ROOT / "docs" / "runtime" / "lane_correlation_monitor.json"

# Thresholds derived from Carver Ch 11 p170 discussion on crisis correlation
# jumps. 0.30 is the "flag" threshold used in the full-history audit; 10 days
# of breach = regime change (short enough to catch, long enough to not noise-trip).
ROLLING_WINDOW_DAYS = 30
ALARM_CORR_THRESHOLD = 0.30
ALARM_CONSECUTIVE_DAYS = 10
HISTORY_DAYS = 180  # enough for 30-day rolling + consecutive-day runs


def load_deployed_lane_ids() -> list[str]:
    with open(LANE_ALLOC_PATH) as f:
        data = json.load(f)
    raw = data.get("deployed_lanes") or data.get("lanes") or []
    return [ln["strategy_id"] for ln in raw]


def load_daily_pnl_series(
    con: duckdb.DuckDBPyConnection,
    strategy_ids: list[str],
    days_back: int,
) -> pd.DataFrame:
    """Return wide DataFrame indexed by trading_day, one column per strategy_id,
    cells = summed pnl_r on days where any live/shadow trade was placed for
    that strategy. NaN on no-trade days → treated as 0.0 for correlation
    (portfolio accounting: no-trade day = 0 subsystem return)."""
    placeholders = ", ".join(["?"] * len(strategy_ids))
    rows = con.execute(
        f"""
        SELECT strategy_id, trading_day, SUM(pnl_r) AS pnl_r
          FROM paper_trades
         WHERE pnl_r IS NOT NULL
           AND strategy_id IN ({placeholders})
           AND trading_day >= CURRENT_DATE - INTERVAL '{days_back} DAY'
         GROUP BY 1, 2
         ORDER BY 2, 1
        """,
        strategy_ids,
    ).fetchdf()
    if rows.empty:
        return pd.DataFrame()
    wide = rows.pivot(index="trading_day", columns="strategy_id", values="pnl_r")
    wide.index = pd.to_datetime(wide.index)
    return wide


def rolling_pairwise_corr(returns_wide: pd.DataFrame, window: int) -> pd.DataFrame:
    """Long-form DataFrame of rolling pairwise correlations.

    Columns: [trading_day, pair_a, pair_b, corr]
    Missing windows (not enough non-NaN overlapping data) yield NaN corr.
    """
    filled = returns_wide.fillna(0.0).sort_index()
    cols = list(filled.columns)
    records: list[dict] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            rolling = filled[a].rolling(window=window, min_periods=window).corr(filled[b])
            for td, corr_val in rolling.items():
                if pd.notna(corr_val):
                    records.append({"trading_day": td, "pair_a": a, "pair_b": b, "corr": float(corr_val)})
    if not records:
        return pd.DataFrame(columns=["trading_day", "pair_a", "pair_b", "corr"])
    return pd.DataFrame(records)


def consecutive_breach_runs(series: pd.Series, threshold: float, min_run: int) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Find runs of consecutive days where series > threshold.
    Returns list of (run_start_date, run_end_date, run_length) for runs >= min_run."""
    above = (series > threshold).astype(int).values
    dates = series.index.to_list()
    runs: list[tuple] = []
    run_start_idx = None
    for k, v in enumerate(above):
        if v == 1 and run_start_idx is None:
            run_start_idx = k
        elif v == 0 and run_start_idx is not None:
            run_len = k - run_start_idx
            if run_len >= min_run:
                runs.append((dates[run_start_idx], dates[k - 1], run_len))
            run_start_idx = None
    if run_start_idx is not None:
        run_len = len(above) - run_start_idx
        if run_len >= min_run:
            runs.append((dates[run_start_idx], dates[-1], run_len))
    return runs


def compute_alarms(rolling_df: pd.DataFrame) -> list[dict]:
    """Scan rolling pair correlations and emit alarms for every pair whose
    rolling correlation exceeds ALARM_CORR_THRESHOLD for >= ALARM_CONSECUTIVE_DAYS."""
    alarms: list[dict] = []
    if rolling_df.empty:
        return alarms
    for (a, b), group in rolling_df.groupby(["pair_a", "pair_b"]):
        ts = group.set_index("trading_day")["corr"].sort_index()
        runs = consecutive_breach_runs(ts, ALARM_CORR_THRESHOLD, ALARM_CONSECUTIVE_DAYS)
        for run_start, run_end, run_len in runs:
            peak_corr = float(ts.loc[run_start:run_end].max())
            alarms.append(
                {
                    "pair_a": a,
                    "pair_b": b,
                    "run_start": str(pd.Timestamp(run_start).date()),
                    "run_end": str(pd.Timestamp(run_end).date()),
                    "run_length_days": run_len,
                    "peak_corr": round(peak_corr, 4),
                    "threshold": ALARM_CORR_THRESHOLD,
                    "min_run": ALARM_CONSECUTIVE_DAYS,
                }
            )
    return alarms


def summarize_current_window(rolling_df: pd.DataFrame) -> dict:
    """Most-recent-window summary: mean pairwise corr, max pair corr."""
    if rolling_df.empty:
        return {
            "as_of": None,
            "mean_pairwise_corr": None,
            "max_pair_corr": None,
            "max_pair": None,
            "pairs_reported": 0,
        }
    last_day = rolling_df["trading_day"].max()
    tail = rolling_df[rolling_df["trading_day"] == last_day]
    if tail.empty:
        return {
            "as_of": str(pd.Timestamp(last_day).date()),
            "mean_pairwise_corr": None,
            "max_pair_corr": None,
            "max_pair": None,
            "pairs_reported": 0,
        }
    max_row = tail.loc[tail["corr"].idxmax()]
    return {
        "as_of": str(pd.Timestamp(last_day).date()),
        "mean_pairwise_corr": round(float(tail["corr"].mean()), 4),
        "max_pair_corr": round(float(max_row["corr"]), 4),
        "max_pair": [max_row["pair_a"], max_row["pair_b"]],
        "pairs_reported": len(tail),
    }


def build_report(
    deployed_ids: list[str],
    returns_wide: pd.DataFrame,
    rolling_df: pd.DataFrame,
    alarms: list[dict],
) -> dict:
    current = summarize_current_window(rolling_df)
    per_lane_coverage = {
        sid: {
            "n_days_with_trade": int((returns_wide[sid].notna() & (returns_wide[sid] != 0)).sum()) if sid in returns_wide.columns else 0,
            "first_day": str(returns_wide[sid].dropna().index.min().date()) if sid in returns_wide.columns and returns_wide[sid].notna().any() else None,
            "last_day": str(returns_wide[sid].dropna().index.max().date()) if sid in returns_wide.columns and returns_wide[sid].notna().any() else None,
        }
        for sid in deployed_ids
    }
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "deployed_lane_ids": deployed_ids,
        "rolling_window_days": ROLLING_WINDOW_DAYS,
        "alarm_corr_threshold": ALARM_CORR_THRESHOLD,
        "alarm_consecutive_days": ALARM_CONSECUTIVE_DAYS,
        "history_days_loaded": HISTORY_DAYS,
        "current_window": current,
        "per_lane_coverage": per_lane_coverage,
        "alarms": alarms,
        "alarm_count": len(alarms),
        "monitor_status": "ALARM" if alarms else "CLEAR",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deployed-lane rolling-correlation tripwire")
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH), help="DuckDB path")
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT), help="JSON artefact path")
    parser.add_argument("--days-back", type=int, default=HISTORY_DAYS, help="History window size")
    parser.add_argument("--verbose", action="store_true", help="Print current-window rolling detail")
    args = parser.parse_args()

    deployed_ids = load_deployed_lane_ids()
    if not deployed_ids:
        print("ERROR: no deployed lanes in manifest")
        return 2

    print(f"Loaded {len(deployed_ids)} deployed lanes from {LANE_ALLOC_PATH.name}")

    con = duckdb.connect(args.db_path, read_only=True)
    try:
        returns_wide = load_daily_pnl_series(con, deployed_ids, args.days_back)
    finally:
        con.close()

    if returns_wide.empty:
        print(f"WARN: no paper_trades rows in last {args.days_back} days for deployed lanes")
        report = build_report(deployed_ids, returns_wide, pd.DataFrame(), [])
    else:
        print(f"Loaded returns window: {returns_wide.index.min().date()} -> {returns_wide.index.max().date()} ({len(returns_wide)} days)")
        rolling_df = rolling_pairwise_corr(returns_wide, ROLLING_WINDOW_DAYS)
        alarms = compute_alarms(rolling_df)
        report = build_report(deployed_ids, returns_wide, rolling_df, alarms)

    # Write JSON artefact
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Console summary
    cw = report["current_window"]
    print()
    print("=" * 72)
    print(f"  MONITOR STATUS : {report['monitor_status']}")
    print(f"  As-of day      : {cw.get('as_of', 'N/A')}")
    print(f"  Pairs reported : {cw.get('pairs_reported', 0)}")
    if cw.get("mean_pairwise_corr") is not None:
        print(f"  Mean pair corr : {cw['mean_pairwise_corr']:+.4f}")
        print(f"  Max pair corr  : {cw['max_pair_corr']:+.4f}  ({cw['max_pair']})")
    print(f"  Alarms         : {report['alarm_count']}")
    for a in report["alarms"]:
        print(f"    [ALARM] {a['pair_a']} <-> {a['pair_b']}")
        print(f"            {a['run_start']} -> {a['run_end']}  ({a['run_length_days']}d)  peak={a['peak_corr']:+.4f}")
    print("=" * 72)
    print(f"  Artefact written: {out_path}")

    if args.verbose and not returns_wide.empty:
        print()
        print("-" * 72)
        print("Per-lane data coverage:")
        for sid, cov in report["per_lane_coverage"].items():
            print(f"  {sid:55s}  n={cov['n_days_with_trade']:3d}  [{cov['first_day']} .. {cov['last_day']}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
