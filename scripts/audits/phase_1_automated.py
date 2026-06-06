#!/usr/bin/env python3
"""Phase 1 — Automated Checks: Orchestrate existing tools.

Source: SYSTEM_AUDIT.md Phase 1 (lines 67-119)

Runs all existing automated tools and captures results.
Does NOT re-implement any check — just orchestrates and records.
CRITICAL if any subprocess returns non-zero.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.audits import PROJECT_ROOT, AuditPhase, Severity

QUICK_PYTEST_TARGETS = [
    "tests/test_pipeline/test_health_check.py",
    "tests/test_tools/test_audit_integrity.py",
    "tests/test_tools/test_git_hooks_env.py",
    "tests/test_pipeline/test_drift_cache.py",
    "tests/test_pipeline/test_check_drift_skip_crg.py::TestCrgAdvisoryLabelInvariants",
    "tests/test_pipeline/test_check_drift_skip_crg.py::TestSkipAllAdvisoryRuntime",
]


def _run_tool(cmd: list[str], timeout: int = 300) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    return r.returncode, r.stdout or "", r.stderr or ""


def _git_common_dir() -> Path | None:
    r = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=10,
        check=False,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    common = Path(r.stdout.strip())
    if not common.is_absolute():
        common = PROJECT_ROOT / common
    return common.resolve()


def _pytest_basetemp() -> Path:
    """Keep pytest temp cleanup out of the user-global temp symlink namespace."""
    dirname = f"audit-phase-1-{os.getpid()}"
    common = _git_common_dir()
    if common is not None:
        base = common / "pytest-tmp" / dirname
    else:
        base = Path(tempfile.gettempdir()) / "canompx3-pytest-tmp" / dirname
    base.parent.mkdir(parents=True, exist_ok=True)
    return base


def _pytest_command(quick: bool) -> list[str]:
    targets = QUICK_PYTEST_TARGETS if quick else ["tests/"]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-x",
        "-q",
        "--basetemp",
        str(_pytest_basetemp()),
    ]
    if quick:
        cmd.extend(["-m", "not slow"])
    return cmd


def _parse_drift_counts(combined: str) -> tuple[int, int, int]:
    match = re.search(r"(\d+) checks passed \[OK\].*?(\d+) advisory", combined)
    if match:
        passed = int(match.group(1))
        advisory = int(match.group(2))
        return passed, 0, passed + advisory
    match = re.search(r"SUMMARY: clean passed=(\d+) advisory=(\d+)", combined)
    if match:
        passed = int(match.group(1))
        advisory = int(match.group(2))
        return passed, 0, passed + advisory
    match = re.search(r"SUMMARY: drift_detected violations=(\d+) passed=(\d+)", combined)
    if match:
        failed = int(match.group(1))
        passed = int(match.group(2))
        return passed, failed, passed + failed
    match = re.search(r"(\d+)\s+passed,\s+(\d+)\s+failed", combined)
    if match:
        passed = int(match.group(1))
        failed = int(match.group(2))
        return passed, failed, passed + failed
    return 0, 0, 0


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 automated audit checks")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run bounded quick Phase 1 gates; full Phase 1 remains the default.",
    )
    args = parser.parse_args()

    audit = AuditPhase(phase_num=1, name="Automated Checks")
    audit.print_header()
    if args.quick:
        print("\nMode: QUICK (bounded pytest/health smoke; full Phase 1 and full drift remain default)")

    # ── 1A: Test Suite ──
    print("\n--- 1A. Test Suite (pytest) ---")
    rc, out, err = _run_tool(_pytest_command(args.quick), timeout=180 if args.quick else 120)
    # Parse test count from pytest output
    combined = out + err
    match = re.search(r"(\d+) passed", combined)
    passed = int(match.group(1)) if match else 0
    match_fail = re.search(r"(\d+) failed", combined)
    failed = int(match_fail.group(1)) if match_fail else 0

    if rc == 0:
        audit.check_passed(f"pytest: {passed} passed")
    else:
        audit.check_failed(f"pytest: {passed} passed, {failed} failed")
        # Show last 10 lines of output for diagnosis
        tail = "\n".join(combined.splitlines()[-10:])
        print(f"         {tail}")
        audit.add_finding(
            Severity.CRITICAL,
            "SMOKE_TEST_FAILURE",
            claimed="All tests pass",
            actual=f"{failed} test(s) failed",
            evidence=f"pytest tests/ -x -q → exit {rc}",
            fix_type="CODE_FIX",
        )

    # ── 1B: Drift Detection ──
    print("\n--- 1B. Drift Detection ---")
    if args.quick:
        drift_cmd = [sys.executable, "-m", "py_compile", "pipeline/check_drift.py"]
        drift_label = "Drift smoke"
        drift_evidence = "python -m py_compile pipeline/check_drift.py"
        drift_timeout = 30
    else:
        drift_cmd = [sys.executable, "pipeline/check_drift.py"]
        drift_label = "Drift"
        drift_evidence = f"{sys.executable} pipeline/check_drift.py"
        drift_timeout = 700
    rc, out, err = _run_tool(drift_cmd, timeout=drift_timeout)
    combined = out + err
    if args.quick:
        d_passed, d_failed, total = (1, 0, 1) if rc == 0 else (0, 1, 1)
    else:
        d_passed, d_failed, total = _parse_drift_counts(combined)

    if rc == 0:
        audit.check_passed(f"{drift_label}: {d_passed}/{total} checks passing")
    else:
        audit.check_failed(f"{drift_label}: {d_passed}/{total} passing, {d_failed} failed")
        # Show failed checks
        for line in combined.splitlines():
            if "FAILED" in line:
                print(f"         {line.strip()[:100]}")
        audit.add_finding(
            Severity.CRITICAL,
            "CONFIG_DRIFT",
            claimed="All drift checks pass",
            actual=f"{d_failed} drift check(s) failed",
            evidence=f"{drift_evidence} -> exit {rc}",
            fix_type="CODE_FIX",
        )

    # ── 1C: Health Check ──
    print("\n--- 1C. Health Check (health_check.py) ---")
    health_cmd = [sys.executable, "-m", "pipeline.health_check"]
    if args.quick:
        health_cmd.append("--quick")
    rc, out, err = _run_tool(health_cmd, timeout=120 if args.quick else 300)
    combined = out + err
    ok_count = combined.count("[OK]")
    fail_count = combined.count("[FAIL]")

    if rc == 0:
        audit.check_passed(f"Health: {ok_count} checks OK")
    else:
        audit.check_failed(f"Health: {ok_count} OK, {fail_count} FAIL")
        for line in combined.splitlines():
            if "[FAIL]" in line:
                print(f"         {line.strip()[:100]}")
        audit.add_finding(
            Severity.CRITICAL,
            "SMOKE_TEST_FAILURE",
            claimed="Health check passes",
            actual=f"{fail_count} health check(s) failed",
            evidence=f"python -m pipeline.health_check -> exit {rc}",
            fix_type="CODE_FIX",
        )

    # ── 1D: Data Integrity ──
    print("\n--- 1D. Data Integrity (audit_integrity.py) ---")
    rc, out, err = _run_tool([sys.executable, "scripts/tools/audit_integrity.py"], timeout=120)
    combined = out + err
    if rc == 0:
        # Parse check count from summary line
        match = re.search(r"all (\d+) checks clean", combined)
        check_count = match.group(1) if match else "?"
        audit.check_passed(f"Integrity: {check_count} enforcing checks clean")
    else:
        # Count violations
        v_count = combined.count("FAILED")
        audit.check_failed(f"Integrity: {v_count} check(s) failed")
        for line in combined.splitlines():
            if "FAILED" in line or line.strip().startswith("  "):
                print(f"         {line.strip()[:100]}")
        audit.add_finding(
            Severity.CRITICAL,
            "DATA_INTEGRITY_VIOLATION",
            claimed="All integrity checks pass",
            actual="Integrity audit failed",
            evidence=f"python scripts/tools/audit_integrity.py → exit {rc}",
            fix_type="DATA_FIX",
        )

    # Print informational stats from integrity output (checks 7-17)
    for line in combined.splitlines():
        if (
            line.startswith("--- 7.")
            or line.startswith("--- 8.")
            or line.startswith("--- 9.")
            or line.startswith("--- 1")
        ) and ("ROW" in line or "DATE" in line or "FDR" in line or "TABLE" in line):
            print(f"\n  [INFO] {line.strip()}")

    # ── 1E: Behavioral Audit ──
    print("\n--- 1E. Behavioral Audit (audit_behavioral.py) ---")
    rc, out, err = _run_tool([sys.executable, "scripts/tools/audit_behavioral.py"], timeout=120)
    combined = out + err
    if rc == 0:
        match = re.search(r"all (\d+) checks clean", combined)
        check_count = match.group(1) if match else "?"
        audit.check_passed(f"Behavioral: {check_count} checks clean")
    else:
        v_count = combined.count("FAILED")
        audit.check_failed(f"Behavioral: {v_count} check(s) failed")
        for line in combined.splitlines():
            if "FAILED" in line:
                print(f"         {line.strip()[:100]}")
        audit.add_finding(
            Severity.CRITICAL,
            "DATA_INTEGRITY_VIOLATION",
            claimed="All behavioral checks pass",
            actual="Behavioral audit failed",
            evidence=f"python scripts/tools/audit_behavioral.py → exit {rc}",
            fix_type="CODE_FIX",
        )

    # ── 1F: Smoke Test (paper trader) ──
    print("\n--- 1F. Smoke Test (paper_trader.py) ---")
    rc, out, err = _run_tool(
        [
            sys.executable,
            "-m",
            "trading_app.paper_trader",
            "--instrument",
            "MGC",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-03",
        ],
        timeout=120,
    )
    combined = out + err
    if rc == 0:
        audit.check_passed("Paper trader smoke test: clean exit")
    else:
        audit.check_failed("Paper trader smoke test: non-zero exit")
        tail = "\n".join(combined.splitlines()[-5:])
        print(f"         {tail}")
        audit.add_finding(
            Severity.HIGH,
            "SMOKE_TEST_FAILURE",
            claimed="Paper trader runs without error",
            actual=f"Exit code {rc}",
            evidence="python -m trading_app.paper_trader --instrument MGC --start 2025-01-02 --end 2025-01-03",
            fix_type="CODE_FIX",
        )

    audit.run_and_exit()


if __name__ == "__main__":
    main()
