#!/usr/bin/env python3
"""Phase 8 — Test Suite Deep Check.

Source: SYSTEM_AUDIT.md Phase 8 (lines 358-373)

Coverage gaps, stale references (E0, old sessions), brittle tests.
"""

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.paths import PROJECT_ROOT

from scripts.audits import AuditPhase, Severity


# Critical modules that MUST have tests (from SYSTEM_AUDIT.md Phase 8)
CRITICAL_MODULES = [
    "pipeline/dst.py",
    "pipeline/calendar_filters.py",
    "pipeline/cost_model.py",
    "trading_app/config.py",
    "trading_app/strategy_fitness.py",
    "trading_app/live_config.py",
    "trading_app/mcp_server.py",
]

# Old session names that should not appear in tests
OLD_SESSION_NAMES = {"0900", "1000", "1100", "1800", "2300", "0030"}


def main():
    audit = AuditPhase(phase_num=8, name="Test Suite")
    audit.print_header()

    _check_coverage_gaps(audit)
    _check_stale_references(audit)
    _check_critical_coverage(audit)

    audit.run_and_exit()


def _check_coverage_gaps(audit: AuditPhase):
    """8A — Coverage gaps: production modules without test files."""
    print("\n--- 8A. Coverage Gaps ---")

    prod_dirs = [PROJECT_ROOT / "pipeline", PROJECT_ROOT / "trading_app"]
    tests_dir = PROJECT_ROOT / "tests"

    # Collect all test files
    test_files = set()
    if tests_dir.exists():
        for tf in tests_dir.rglob("test_*.py"):
            test_files.add(tf.stem)  # e.g., "test_dst"
        for tf in tests_dir.rglob("*_test.py"):
            test_files.add(tf.stem)

    untested = []
    tested = []
    skip_names = {"__init__", "__pycache__", "conftest"}

    for d in prod_dirs:
        if not d.exists():
            continue
        for py in sorted(d.glob("*.py")):
            if py.stem in skip_names or py.stem.startswith("_"):
                continue
            # Check for test_<module>.py
            expected_test = f"test_{py.stem}"
            if expected_test in test_files:
                tested.append(py.stem)
            else:
                untested.append(str(py.relative_to(PROJECT_ROOT)))

    audit.check_info(f"{len(tested)} modules with tests, {len(untested)} without")

    if untested:
        for m in untested:
            print(f"         UNTESTED: {m}")
        # Only flag as finding if there are many
        if len(untested) > len(tested):
            audit.add_finding(
                Severity.MEDIUM,
                "UNTESTED_MODULE",
                claimed="Most production modules have test coverage",
                actual=f"{len(untested)} modules without tests",
                evidence="Glob pipeline/*.py trading_app/*.py vs tests/test_*.py",
                fix_type="CODE_FIX",
            )


def _check_stale_references(audit: AuditPhase):
    """8B — Stale references in tests."""
    print("\n--- 8B. Stale References ---")

    tests_dir = PROJECT_ROOT / "tests"
    if not tests_dir.exists():
        audit.check_info("tests/ directory not found")
        return

    # Grep for E0
    e0_files = []
    old_session_files = []

    for tf in tests_dir.rglob("*.py"):
        try:
            content = tf.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        # E0 references (excluding comments about E0 being dead)
        if re.search(r"""['"]E0['"]""", content):
            e0_files.append(str(tf.relative_to(PROJECT_ROOT)))

        # Old session names
        for name in OLD_SESSION_NAMES:
            if re.search(rf"""['\"]{name}['\"]""", content):
                old_session_files.append((str(tf.relative_to(PROJECT_ROOT)), name))
                break  # One finding per file

    if e0_files:
        audit.check_failed(f"{len(e0_files)} test file(s) reference 'E0'")
        for f in e0_files:
            print(f"         STALE_TEST: {f}")
        audit.add_finding(
            Severity.MEDIUM,
            "STALE_TEST",
            claimed="No E0 references in tests",
            actual=f"{len(e0_files)} file(s) with E0",
            evidence="grep 'E0' tests/**/*.py",
            fix_type="CODE_FIX",
        )
    else:
        audit.check_passed("No E0 references in tests")

    if old_session_files:
        audit.check_failed(f"{len(old_session_files)} test file(s) reference old session names")
        for f, name in old_session_files:
            print(f"         STALE_TEST: {f} ('{name}')")
        audit.add_finding(
            Severity.MEDIUM,
            "STALE_TEST",
            claimed="No old session names in tests",
            actual=f"{len(old_session_files)} file(s) with old names",
            evidence="grep old session names in tests/**/*.py",
            fix_type="CODE_FIX",
        )
    else:
        audit.check_passed("No old session names in tests")


def _check_critical_coverage(audit: AuditPhase):
    """8C — Critical module test coverage."""
    print("\n--- 8C. Critical Module Coverage ---")

    tests_dir = PROJECT_ROOT / "tests"
    test_files = set()
    if tests_dir.exists():
        for tf in tests_dir.rglob("test_*.py"):
            test_files.add(tf.stem)

    for module_path in CRITICAL_MODULES:
        module_name = Path(module_path).stem
        expected_test = f"test_{module_name}"
        if expected_test in test_files:
            audit.check_passed(f"{module_path}: test file exists ({expected_test}.py)")
        else:
            audit.check_failed(f"{module_path}: NO test file ({expected_test}.py)")
            audit.add_finding(
                Severity.MEDIUM,
                "UNTESTED_MODULE",
                claimed=f"Critical module {module_path} has tests",
                actual=f"No {expected_test}.py found",
                evidence=f"ls tests/{expected_test}.py",
                fix_type="CODE_FIX",
            )


if __name__ == "__main__":
    main()
