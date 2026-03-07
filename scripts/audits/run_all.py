#!/usr/bin/env python3
"""Orchestrator — Runs all audit phases in order.

Source: SYSTEM_AUDIT.md Execution Order (lines 525-534)

Runs phases 0-10 sequentially. Stops on CRITICAL (exit 1).
Supports --start N to resume from a specific phase.
Supports --quick for quick mode (phases 0, 1, 6, 3 only per SYSTEM_AUDIT.md).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Phase registry: (number, name, script filename)
PHASES = [
    (0, "Triage", "phase_0_triage.py"),
    (1, "Automated Checks", "phase_1_automated.py"),
    (2, "Infra Config", "phase_2_infra_config.py"),
    (3, "Docs vs Reality", "phase_3_docs.py"),
    (4, "Config Sync", "phase_4_config_sync.py"),
    (5, "Database Integrity", "phase_5_database.py"),
    (6, "Build Chain", "phase_6_build_chain.py"),
    (7, "Live Trading", "phase_7_live_trading.py"),
    (8, "Test Suite", "phase_8_test_suite.py"),
    (9, "Research Hygiene", "phase_9_research.py"),
    (10, "Git & CI", "phase_10_git_ci.py"),
]

# Quick mode: Phase 0B → 1 → 6 → 3A (per SYSTEM_AUDIT.md line 527)
QUICK_PHASES = {0, 1, 6, 3}


def run_phase(phase_num: int, name: str, script: str) -> tuple[int, float]:
    """Run a single phase script. Returns (exit_code, duration_seconds)."""
    script_path = Path(__file__).parent / script
    if not script_path.exists():
        print(f"\n  [SKIP] Phase {phase_num} ({name}): script not found ({script})")
        return -1, 0.0

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        timeout=600,  # 10 min max per phase
    )
    duration = time.time() - start
    return result.returncode, duration


def main():
    parser = argparse.ArgumentParser(description="Run all audit phases")
    parser.add_argument("--start", type=int, default=0, help="Resume from phase N")
    parser.add_argument("--quick", action="store_true", help="Quick mode: phases 0, 1, 3, 6 only")
    parser.add_argument("--phase", type=int, help="Run a single phase only")
    args = parser.parse_args()

    print("=" * 70)
    print("SYSTEM AUDIT — FULL RUN")
    print("=" * 70)

    if args.quick:
        print("Mode: QUICK (phases 0, 1, 3, 6)")
        phases = [(n, name, script) for n, name, script in PHASES if n in QUICK_PHASES]
    elif args.phase is not None:
        print(f"Mode: SINGLE PHASE ({args.phase})")
        phases = [(n, name, script) for n, name, script in PHASES if n == args.phase]
        if not phases:
            print(f"Phase {args.phase} not found. Available: {[n for n, _, _ in PHASES]}")
            sys.exit(1)
    else:
        print(f"Mode: FULL (phases {args.start}-10)")
        phases = [(n, name, script) for n, name, script in PHASES if n >= args.start]

    results = {}
    total_start = time.time()

    for phase_num, name, script in phases:
        print(f"\n{'─' * 70}")
        print(f"  Running Phase {phase_num}: {name}")
        print(f"{'─' * 70}")

        rc, duration = run_phase(phase_num, name, script)
        results[phase_num] = (name, rc, duration)

        if rc == 1:
            print(f"\n  *** CRITICAL failure in Phase {phase_num} ({name}) — stopping ***")
            print(
                f"  Fix the CRITICAL issue(s), then resume with: python scripts/audits/run_all.py --start {phase_num}"
            )
            break
        elif rc == -1:
            print(f"  Phase {phase_num} skipped (script not found)")

    total_duration = time.time() - total_start

    # ── Scorecard ──
    print(f"\n\n{'=' * 70}")
    print("SYSTEM AUDIT SCORECARD")
    print("=" * 70)

    for phase_num, (name, rc, duration) in sorted(results.items()):
        if rc == 0:
            status = "PASS"
        elif rc == 1:
            status = "FAIL (CRITICAL)"
        elif rc == -1:
            status = "SKIP"
        else:
            status = f"EXIT {rc}"
        print(f"  Phase {phase_num:>2} ({name:<20}): {status:<20} ({duration:.1f}s)")

    print(f"\n  Total time: {total_duration:.1f}s")

    passed = sum(1 for _, rc, _ in results.values() if rc == 0)
    failed = sum(1 for _, rc, _ in results.values() if rc == 1)
    skipped = sum(1 for _, rc, _ in results.values() if rc == -1)

    print(f"  Passed: {passed}, Failed: {failed}, Skipped: {skipped}")
    print("=" * 70)

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
