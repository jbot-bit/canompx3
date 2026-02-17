#!/usr/bin/env python3
"""
Rolling window evaluation of ORB breakout strategies.

For each train_months setting, slides month-by-month through the data,
running regime discovery + validation on each training window. The test
month is the month immediately following the training window.

Uses existing regime/ module infrastructure (regime_strategies +
regime_validated tables) with run_label='rolling_{train}m_{YYYY}_{MM}'.

Also computes double-break frequency per window/session and auto-degrades
strategies whose session has >67% double-break rate.

Usage:
    python scripts/rolling_eval.py --train-months 12 --test-start 2024-07 --test-end 2026-01
    python scripts/rolling_eval.py --train-months 6 12 18 --test-start 2024-07 --test-end 2026-01 --dry-run
"""

import sys
import json
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.stdout.reconfigure(line_buffering=True)

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.init_db import ORB_LABELS
from trading_app.regime.discovery import run_regime_discovery
from trading_app.regime.validator import run_regime_validation
from trading_app.regime.schema import init_regime_schema

# Double-break threshold: if >= this fraction of break-days had double
# breaks, single-direction breakout strategies are auto-DEGRADED.
DOUBLE_BREAK_THRESHOLD = 0.67

def generate_rolling_windows(
    train_months: int,
    test_start: date,
    test_end: date,
) -> list[dict]:
    """Generate rolling windows with train/test date ranges.

    Each window:
      - train: [test_month - train_months, test_month - 1 day]
      - test:  [test_month first day, test_month last day]

    Args:
        train_months: Number of months in training window.
        test_start: First test month (1st of month).
        test_end: Last test month (1st of month).

    Returns list of dicts with keys:
        run_label, train_start, train_end, test_start, test_end
    """
    windows = []
    current = date(test_start.year, test_start.month, 1)
    end = date(test_end.year, test_end.month, 1)

    while current <= end:
        train_start = current - relativedelta(months=train_months)
        train_end = current - relativedelta(days=1)
        test_end_date = (current + relativedelta(months=1)) - relativedelta(days=1)

        run_label = f"rolling_{train_months}m_{current.year}_{current.month:02d}"

        windows.append({
            "run_label": run_label,
            "train_months": train_months,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": current,
            "test_end": test_end_date,
        })

        current = current + relativedelta(months=1)

    return windows

def compute_double_break_pct(
    db_path: Path,
    train_start: date,
    train_end: date,
    orb_minutes: int = 5,
) -> dict[str, float]:
    """Compute double-break percentage per ORB session in a date range.

    Returns dict: {orb_label: fraction of break-days with double_break}.
    Only counts days where break_dir IS NOT NULL (a break occurred).
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        result = {}
        for label in ORB_LABELS:
            row = con.execute(f"""
                SELECT
                    COUNT(*) FILTER (WHERE orb_{label}_double_break = TRUE) as double_ct,
                    COUNT(*) as break_ct
                FROM daily_features
                WHERE orb_minutes = ?
                  AND trading_day >= ?
                  AND trading_day <= ?
                  AND orb_{label}_break_dir IS NOT NULL
            """, [orb_minutes, train_start, train_end]).fetchone()

            double_ct, break_ct = row
            if break_ct > 0:
                result[label] = double_ct / break_ct
            else:
                result[label] = 0.0

        return result
    finally:
        con.close()

def mark_degraded_by_double_break(
    db_path: Path,
    run_label: str,
    degraded_sessions: set[str],
) -> int:
    """Mark regime_strategies as REJECTED for sessions with high double-break.

    Returns count of strategies marked.
    """
    if not degraded_sessions:
        return 0

    con = duckdb.connect(str(db_path))
    try:
        total = 0
        for session in degraded_sessions:
            count = con.execute("""
                UPDATE regime_strategies
                SET validation_status = 'REJECTED',
                    validation_notes = 'Auto-degraded: double-break >67% in training window'
                WHERE run_label = ?
                  AND orb_label = ?
                  AND (validation_status IS NULL OR validation_status = '')
            """, [run_label, session]).fetchone()

            affected = con.execute("""
                SELECT COUNT(*) FROM regime_strategies
                WHERE run_label = ? AND orb_label = ?
                  AND validation_notes LIKE 'Auto-degraded%'
            """, [run_label, session]).fetchone()[0]
            total += affected

        con.commit()
        return total
    finally:
        con.close()

def run_rolling_evaluation(
    db_path: Path | None = None,
    instrument: str = "MGC",
    train_months_list: list[int] | None = None,
    test_start: str = "2024-07",
    test_end: str = "2026-01",
    orb_minutes: int = 5,
    min_sample: int = 20,
    stress_multiplier: float = 1.5,
    min_years_positive_pct: float = 0.0,
    dry_run: bool = False,
) -> dict:
    """Run rolling window evaluation for all train_months settings.

    Returns summary dict with per-window results.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    if train_months_list is None:
        train_months_list = [12, 18]

    test_start_date = date.fromisoformat(test_start + "-01")
    test_end_date = date.fromisoformat(test_end + "-01")

    # Ensure regime schema exists
    if not dry_run:
        init_regime_schema(db_path=db_path)

    all_results = {}

    for train_months in train_months_list:
        windows = generate_rolling_windows(
            train_months, test_start_date, test_end_date
        )

        print(f"\n{'='*60}")
        print(f"ROLLING EVALUATION: {train_months}m training window")
        print(f"  Test range: {test_start} to {test_end}")
        print(f"  Windows: {len(windows)}")
        print(f"{'='*60}\n")

        window_results = []

        for i, w in enumerate(windows):
            run_label = w["run_label"]
            print(f"\n--- Window {i+1}/{len(windows)}: {run_label} ---")
            print(f"  Train: {w['train_start']} to {w['train_end']}")
            print(f"  Test:  {w['test_start']} to {w['test_end']}")

            # Step 1: Compute double-break frequency
            db_pct = compute_double_break_pct(
                db_path, w["train_start"], w["train_end"], orb_minutes
            )
            degraded_sessions = {
                label for label, pct in db_pct.items()
                if pct >= DOUBLE_BREAK_THRESHOLD
            }

            if degraded_sessions:
                print(f"  Double-break degraded sessions: {degraded_sessions}")
                for label in degraded_sessions:
                    print(f"    {label}: {db_pct[label]:.1%}")

            # Step 2: Run discovery
            strategies_found = run_regime_discovery(
                db_path=db_path,
                instrument=instrument,
                start_date=w["train_start"],
                end_date=w["train_end"],
                run_label=run_label,
                orb_minutes=orb_minutes,
                dry_run=dry_run,
            )

            # Step 3: Mark double-break degraded BEFORE validation
            degraded_count = 0
            if not dry_run and degraded_sessions:
                degraded_count = mark_degraded_by_double_break(
                    db_path, run_label, degraded_sessions
                )
                print(f"  Auto-degraded {degraded_count} strategies (double-break)")

            # Step 4: Validate remaining (relaxed params for short windows)
            passed, rejected = 0, 0
            if not dry_run:
                passed, rejected = run_regime_validation(
                    db_path=db_path,
                    instrument=instrument,
                    run_label=run_label,
                    min_sample=min_sample,
                    stress_multiplier=stress_multiplier,
                    min_years_positive_pct=min_years_positive_pct,
                    dry_run=dry_run,
                )

            result = {
                "run_label": run_label,
                "train_months": train_months,
                "train_start": str(w["train_start"]),
                "train_end": str(w["train_end"]),
                "test_start": str(w["test_start"]),
                "test_end": str(w["test_end"]),
                "strategies_discovered": strategies_found,
                "strategies_passed": passed,
                "strategies_rejected": rejected,
                "strategies_degraded_double_break": degraded_count,
                "double_break_pct": {k: round(v, 3) for k, v in db_pct.items()},
            }
            window_results.append(result)

        all_results[f"{train_months}m"] = window_results

    return all_results

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rolling window evaluation of ORB strategies"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--train-months", type=int, nargs="+", default=[12, 18],
                        help="Training window sizes in months (e.g. 6 12 18)")
    parser.add_argument("--test-start", default="2024-07",
                        help="First test month YYYY-MM")
    parser.add_argument("--test-end", default="2026-01",
                        help="Last test month YYYY-MM")
    parser.add_argument("--orb-minutes", type=int, default=5)
    parser.add_argument("--min-sample", type=int, default=20,
                        help="Min sample size per window (relaxed for short windows)")
    parser.add_argument("--stress-multiplier", type=float, default=1.5)
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to gold.db (default: project gold.db)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Write results JSON to this path")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else None

    results = run_rolling_evaluation(
        db_path=db_path,
        instrument=args.instrument,
        train_months_list=args.train_months,
        test_start=args.test_start,
        test_end=args.test_end,
        orb_minutes=args.orb_minutes,
        min_sample=args.min_sample,
        stress_multiplier=args.stress_multiplier,
        dry_run=args.dry_run,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("ROLLING EVALUATION SUMMARY")
    print(f"{'='*60}")

    for key, windows in results.items():
        total_passed = sum(w["strategies_passed"] for w in windows)
        total_degraded = sum(w["strategies_degraded_double_break"] for w in windows)
        print(f"\n{key}:")
        print(f"  Windows: {len(windows)}")
        print(f"  Total strategies passed: {total_passed}")
        print(f"  Total auto-degraded (double-break): {total_degraded}")

        for w in windows:
            p = w["strategies_passed"]
            d = w["strategies_discovered"]
            db = w["strategies_degraded_double_break"]
            print(f"    {w['run_label']}: {p} passed / {d} discovered"
                  f"{f' ({db} degraded)' if db else ''}")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {output_path}")

if __name__ == "__main__":
    main()
