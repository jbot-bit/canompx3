#!/usr/bin/env python3
"""Daily paper-trade pipeline runner.

Automates the forward eval pack daily checklist (steps 1-6):
  1. Refresh bars (ingest_dbn --resume)
  2. Build daily_features for today
  3. Build orb_outcomes for today
  4. Run paper-trade monitor
  5. Check kill triggers
  6. Print summary

Usage:
    python -m scripts.infra.daily_paper_run
    python -m scripts.infra.daily_paper_run --date 2026-03-25
    python -m scripts.infra.daily_paper_run --skip-ingest  # if bars already current

Output: stdout summary + CSV at data/paper_mnq_core5_replay.csv
No DB schema changes. No live execution paths.
"""

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Derive instruments and sessions from canonical sources
from trading_app.live_config import PAPER_TRADE_CANDIDATES
from trading_app.prop_profiles import ACCOUNT_PROFILES

_PAPER_SESSIONS = sorted({s.orb_label for s in PAPER_TRADE_CANDIDATES if s.tier == "core"})
_PAPER_INSTRUMENTS = {"MNQ"}  # All PAPER_TRADE_CANDIDATES are MNQ
_TOPSTEP = ACCOUNT_PROFILES.get("topstep_50k")
_TOPSTEP_INSTRUMENTS = set(_TOPSTEP.allowed_instruments) if _TOPSTEP else set()
_ALL_INSTRUMENTS = _PAPER_INSTRUMENTS | _TOPSTEP_INSTRUMENTS


def run_step(step_num: int, description: str, cmd: list[str], allow_fail: bool = False) -> bool:
    """Run a pipeline step and report pass/fail. Fail-closed on timeout."""
    print(f"\n[{step_num}/6] {description}")
    print(f"  CMD: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        if not allow_fail:
            print(f"  ABORT — fix before continuing")
            return False
        print(f"  (allow_fail=True, continuing)")
        return True
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-5:]:
                print(f"    {line}")
        if not allow_fail:
            print(f"  ABORT — fix before continuing")
            return False
        print(f"  (allow_fail=True, continuing)")
    else:
        last = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else "OK"
        print(f"  OK: {last[:120]}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Daily paper-trade pipeline runner")
    parser.add_argument("--date", type=date.fromisoformat, default=None,
                        help="Trading date to process (default: yesterday)")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip bar ingestion (if bars already current)")
    parser.add_argument("--monitor-only", action="store_true",
                        help="Skip pipeline steps, just run monitor")
    args = parser.parse_args()

    trading_date = args.date or (date.today() - timedelta(days=1))
    date_str = trading_date.isoformat()

    print("=" * 70)
    print(f"DAILY PAPER-TRADE RUN — {date_str}")
    print("=" * 70)
    print(f"Paper sessions: {', '.join(_PAPER_SESSIONS)}")
    print(f"Instruments: {', '.join(sorted(_ALL_INSTRUMENTS))}")
    if _TOPSTEP:
        print(f"TopStep: {', '.join(sorted(_TOPSTEP.allowed_sessions))} {', '.join(sorted(_TOPSTEP_INSTRUMENTS))} (conditional)")

    py = sys.executable

    if not args.monitor_only:
        # Primary instrument = MNQ (paper CORE). Secondary = TopStep instruments.
        primary = sorted(_PAPER_INSTRUMENTS)
        secondary = sorted(_TOPSTEP_INSTRUMENTS - _PAPER_INSTRUMENTS)

        # Step 1: Refresh bars
        if not args.skip_ingest:
            for inst in primary:
                ok = run_step(1, f"Refresh {inst} bars", [
                    py, "-m", "pipeline.ingest_dbn", "--instrument", inst, "--resume",
                ])
                if not ok:
                    return 1
            for inst in secondary:
                run_step(1, f"Refresh {inst} bars (secondary)", [
                    py, "-m", "pipeline.ingest_dbn", "--instrument", inst, "--resume",
                ], allow_fail=True)
        else:
            print(f"\n[1/6] Skipped (--skip-ingest)")

        # Step 2: Build features
        for inst in primary:
            ok = run_step(2, f"Build daily_features {inst} {date_str}", [
                py, "-m", "pipeline.build_daily_features",
                "--instrument", inst, "--start", date_str, "--end", date_str,
            ])
            if not ok:
                return 1
        for inst in secondary:
            run_step(2, f"Build daily_features {inst} {date_str}", [
                py, "-m", "pipeline.build_daily_features",
                "--instrument", inst, "--start", date_str, "--end", date_str,
            ], allow_fail=True)

        # Step 3: Build outcomes
        for inst in primary:
            ok = run_step(3, f"Build orb_outcomes {inst} {date_str}", [
                py, "-m", "trading_app.outcome_builder",
                "--instrument", inst, "--start", date_str, "--end", date_str,
            ])
            if not ok:
                return 1
        for inst in secondary:
            run_step(3, f"Build orb_outcomes {inst} {date_str}", [
                py, "-m", "trading_app.outcome_builder",
                "--instrument", inst, "--start", date_str, "--end", date_str,
            ], allow_fail=True)

    # Step 4: Run paper monitor
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "paper_mnq_core5_replay.csv"

    ok = run_step(4, "Run paper-trade monitor", [
        py, "-m", "scripts.reports.monitor_paper_forward",
        "--output", str(csv_path),
    ])
    if not ok:
        return 1

    # Step 5: Kill trigger check (already in monitor output, but explicit)
    print(f"\n[5/6] Kill trigger check")
    print(f"  Review monitor output above for:")
    print(f"    - 3 consecutive negative months?")
    print(f"    - Cumulative R <= -10?")
    print(f"    - Any session anomaly?")
    print(f"  If any trigger fires: STOP and log in incident log.")

    # Step 6: Summary
    print(f"\n[6/6] Summary")
    print(f"  Date processed: {date_str}")
    print(f"  CSV output: {csv_path}")
    print(f"  Next: if Friday, complete weekly risk memo")
    print(f"\n{'=' * 70}")
    print(f"DAILY RUN COMPLETE")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
