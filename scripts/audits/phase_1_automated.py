#!/usr/bin/env python3
"""Phase 1 — Automated Checks: Orchestrate existing tools.

Source: SYSTEM_AUDIT.md Phase 1 (lines 67-119)

Runs all existing automated tools and captures results.
Does NOT re-implement any check — just orchestrates and records.
CRITICAL if any subprocess returns non-zero.
"""

import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.audits import PROJECT_ROOT, AuditPhase, Severity


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


def main():
    audit = AuditPhase(phase_num=1, name="Automated Checks")
    audit.print_header()

    # ── 1A: Test Suite ──
    print("\n--- 1A. Test Suite (pytest) ---")
    rc, out, err = _run_tool(["python", "-m", "pytest", "tests/", "-x", "-q"], timeout=120)
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
    print("\n--- 1B. Drift Detection (check_drift.py) ---")
    rc, out, err = _run_tool(["python", "pipeline/check_drift.py"], timeout=120)
    combined = out + err
    # Parse summary line: "DRIFT CHECK SUMMARY: N passed, M failed"
    match = re.search(r"(\d+)\s+passed,\s+(\d+)\s+failed", combined)
    if match:
        d_passed, d_failed = int(match.group(1)), int(match.group(2))
        total = d_passed + d_failed
    else:
        d_passed, d_failed, total = 0, 0, 0

    if rc == 0:
        audit.check_passed(f"Drift: {d_passed}/{total} checks passing")
    else:
        audit.check_failed(f"Drift: {d_passed}/{total} passing, {d_failed} failed")
        # Show failed checks
        for line in combined.splitlines():
            if "FAILED" in line:
                print(f"         {line.strip()[:100]}")
        audit.add_finding(
            Severity.CRITICAL,
            "CONFIG_DRIFT",
            claimed="All drift checks pass",
            actual=f"{d_failed} drift check(s) failed",
            evidence=f"python pipeline/check_drift.py → exit {rc}",
            fix_type="CODE_FIX",
        )

    # ── 1C: Health Check ──
    print("\n--- 1C. Health Check (health_check.py) ---")
    rc, out, err = _run_tool(["python", "pipeline/health_check.py"], timeout=300)
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
            evidence=f"python pipeline/health_check.py → exit {rc}",
            fix_type="CODE_FIX",
        )

    # ── 1D: Data Integrity ──
    print("\n--- 1D. Data Integrity (audit_integrity.py) ---")
    rc, out, err = _run_tool(["python", "scripts/tools/audit_integrity.py"], timeout=120)
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
        if line.startswith("--- 7.") or line.startswith("--- 8.") or line.startswith("--- 9.") or line.startswith("--- 1") and ("ROW" in line or "DATE" in line or "FDR" in line or "TABLE" in line):
            print(f"\n  [INFO] {line.strip()}")

    # ── 1E: Behavioral Audit ──
    print("\n--- 1E. Behavioral Audit (audit_behavioral.py) ---")
    rc, out, err = _run_tool(["python", "scripts/tools/audit_behavioral.py"], timeout=120)
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
            "python",
            "trading_app/paper_trader.py",
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
            evidence="python trading_app/paper_trader.py --instrument MGC --start 2025-01-02 --end 2025-01-03",
            fix_type="CODE_FIX",
        )

    audit.run_and_exit()


if __name__ == "__main__":
    main()
