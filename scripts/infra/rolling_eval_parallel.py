#!/usr/bin/env python3
"""
Parallel rolling window evaluation — uses all CPU cores.

Each worker gets its own copy of gold.db to avoid DuckDB write locks.
Results are merged into the main DB at the end.

Usage:
    python scripts/rolling_eval_parallel.py --db-path C:/db/gold.db --workers 16
"""

import sys
import json
import shutil
import tempfile
from pathlib import Path
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(line_buffering=True)

from scripts.infra.rolling_eval import (
    generate_rolling_windows,
    compute_double_break_pct,
    run_rolling_evaluation,
    DOUBLE_BREAK_THRESHOLD,
)
from pipeline.paths import GOLD_DB_PATH
from trading_app.regime.schema import init_regime_schema
from trading_app.regime.discovery import run_regime_discovery
from trading_app.regime.validator import run_regime_validation


def _process_window(args: dict) -> dict:
    """Process a single rolling window in an isolated DB copy."""
    src_db = args["src_db"]
    window = args["window"]
    instrument = args["instrument"]
    orb_minutes = args["orb_minutes"]
    min_sample = args["min_sample"]
    stress_multiplier = args["stress_multiplier"]
    worker_id = args["worker_id"]

    # Each worker gets its own temp DB copy
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"rolling_w{worker_id}_"))
    tmp_db = tmp_dir / "gold.db"
    shutil.copy2(src_db, tmp_db)

    run_label = window["run_label"]

    try:
        init_regime_schema(db_path=tmp_db)

        # Step 1: Double-break
        db_pct = compute_double_break_pct(
            tmp_db, window["train_start"], window["train_end"], orb_minutes
        )
        degraded_sessions = {
            label for label, pct in db_pct.items()
            if pct >= DOUBLE_BREAK_THRESHOLD
        }

        # Step 2: Discovery
        strategies_found = run_regime_discovery(
            db_path=tmp_db,
            instrument=instrument,
            start_date=window["train_start"],
            end_date=window["train_end"],
            run_label=run_label,
            orb_minutes=orb_minutes,
        )

        # Step 3: Mark degraded
        degraded_count = 0
        if degraded_sessions:
            from scripts.infra.rolling_eval import mark_degraded_by_double_break
            degraded_count = mark_degraded_by_double_break(
                tmp_db, run_label, degraded_sessions
            )

        # Step 4: Validate
        passed, rejected = run_regime_validation(
            db_path=tmp_db,
            instrument=instrument,
            run_label=run_label,
            min_sample=min_sample,
            stress_multiplier=stress_multiplier,
            min_years_positive_pct=0.0,
        )

        result = {
            "run_label": run_label,
            "train_months": window["train_months"],
            "train_start": str(window["train_start"]),
            "train_end": str(window["train_end"]),
            "test_start": str(window["test_start"]),
            "test_end": str(window["test_end"]),
            "strategies_discovered": strategies_found,
            "strategies_passed": passed,
            "strategies_rejected": rejected,
            "strategies_degraded_double_break": degraded_count,
            "double_break_pct": {k: round(v, 3) for k, v in db_pct.items()},
            "tmp_db": str(tmp_db),
        }
        return result

    except Exception as e:
        # Clean up on error
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {
            "run_label": run_label,
            "error": str(e),
            "tmp_db": None,
        }


def merge_results_to_main(main_db: Path, tmp_dbs: list[Path], run_labels: list[str]):
    """Merge regime_strategies and regime_validated from temp DBs into main."""
    import duckdb

    con = duckdb.connect(str(main_db))
    init_regime_schema(db_path=main_db)

    try:
        for tmp_db, run_label in zip(tmp_dbs, run_labels):
            if tmp_db is None:
                continue

            # Attach temp DB and copy rows for this run_label
            con.execute(f"ATTACH '{tmp_db}' AS tmp (READ_ONLY)")

            # Copy regime_strategies
            con.execute("""
                INSERT OR REPLACE INTO regime_strategies
                SELECT * FROM tmp.regime_strategies
                WHERE run_label = ?
            """, [run_label])

            # Copy regime_validated
            con.execute("""
                INSERT OR REPLACE INTO regime_validated
                SELECT * FROM tmp.regime_validated
                WHERE run_label = ?
            """, [run_label])

            con.execute("DETACH tmp")

        con.commit()
    finally:
        con.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallel rolling window evaluation (multi-core)"
    )
    parser.add_argument("--instrument", default="MGC")
    parser.add_argument("--train-months", type=int, nargs="+", default=[12, 18])
    parser.add_argument("--test-start", default="2017-01")
    parser.add_argument("--test-end", default="2026-01")
    parser.add_argument("--orb-minutes", type=int, default=5)
    parser.add_argument("--min-sample", type=int, default=30)
    parser.add_argument("--stress-multiplier", type=float, default=1.5)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH

    # Generate all windows
    all_windows = []
    for train_months in args.train_months:
        test_start_date = date.fromisoformat(args.test_start + "-01")
        test_end_date = date.fromisoformat(args.test_end + "-01")
        windows = generate_rolling_windows(train_months, test_start_date, test_end_date)
        all_windows.extend(windows)

    print(f"Total windows: {len(all_windows)}")
    print(f"Workers: {args.workers}")
    print(f"DB: {db_path}")

    # Build work items
    work_items = []
    for i, w in enumerate(all_windows):
        work_items.append({
            "src_db": str(db_path),
            "window": w,
            "instrument": args.instrument,
            "orb_minutes": args.orb_minutes,
            "min_sample": args.min_sample,
            "stress_multiplier": args.stress_multiplier,
            "worker_id": i,
        })

    # Run in parallel
    completed = []
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_window, item): item for item in work_items}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if "error" in result:
                errors.append(result)
                print(f"  [{i+1}/{len(all_windows)}] {result['run_label']}: ERROR - {result['error']}")
            else:
                completed.append(result)
                p = result["strategies_passed"]
                d = result["strategies_discovered"]
                db = result["strategies_degraded_double_break"]
                print(f"  [{i+1}/{len(all_windows)}] {result['run_label']}: "
                      f"{p} passed / {d} discovered"
                      f"{f' ({db} degraded)' if db else ''}")

    # Merge results into main DB
    print(f"\nMerging {len(completed)} window results into {db_path}...")
    tmp_dbs = [Path(r["tmp_db"]) for r in completed if r.get("tmp_db")]
    run_labels = [r["run_label"] for r in completed if r.get("tmp_db")]
    merge_results_to_main(db_path, tmp_dbs, run_labels)

    # Clean up temp dirs
    for r in completed:
        if r.get("tmp_db"):
            tmp_dir = Path(r["tmp_db"]).parent
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Summary
    print(f"\n{'='*60}")
    print("ROLLING EVALUATION SUMMARY (PARALLEL)")
    print(f"{'='*60}")
    print(f"  Windows completed: {len(completed)}")
    print(f"  Errors: {len(errors)}")
    total_passed = sum(r["strategies_passed"] for r in completed)
    total_degraded = sum(r["strategies_degraded_double_break"] for r in completed)
    print(f"  Total strategies passed (all windows): {total_passed}")
    print(f"  Total auto-degraded (double-break): {total_degraded}")

    # Group by train_months
    by_months = {}
    for r in completed:
        key = f"{r['train_months']}m"
        by_months.setdefault(key, []).append(r)

    for key, windows in sorted(by_months.items()):
        p = sum(w["strategies_passed"] for w in windows)
        print(f"\n  {key}: {len(windows)} windows, {p} total passed")

    if args.output:
        output_data = {"completed": completed, "errors": errors}
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
