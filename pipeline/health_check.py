#!/usr/bin/env python3
"""
Pipeline health check — quick CLI that checks everything at once.

Usage:
    python pipeline/health_check.py
"""

import subprocess
import sys
from pathlib import Path

import duckdb

from pipeline.paths import GOLD_DB_PATH, DAILY_DBN_DIR

PROJECT_ROOT = Path(__file__).parent.parent

def check_python_deps() -> tuple[bool, str]:
    """Check Python version and key dependencies."""
    version = f"Python {sys.version.split()[0]}"
    missing = []
    for pkg in ["duckdb", "databento", "pandas", "numpy", "pyarrow", "zstandard", "pytest"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        return False, f"{version}, missing: {', '.join(missing)}"
    return True, f"{version}, all deps installed"

def check_database() -> tuple[bool, str]:
    """Check gold.db exists and has data.

    Fail-closed: only duckdb.IOException (lock contention) passes.
    Missing tables, corrupt DB, or other errors return False.
    """
    if not GOLD_DB_PATH.exists():
        return False, "gold.db does not exist"
    size_mb = round(GOLD_DB_PATH.stat().st_size / (1024 * 1024), 1)
    try:
        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    except duckdb.IOException:
        return True, f"gold.db exists ({size_mb}MB, locked by another process)"
    except Exception as e:
        return False, f"gold.db exists ({size_mb}MB) but cannot open: {type(e).__name__}: {e}"
    try:
        counts = {}
        errors = []
        for table in ("bars_1m", "bars_5m", "daily_features"):
            try:
                counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except duckdb.CatalogException:
                errors.append(f"{table} missing")
            except Exception as e:
                errors.append(f"{table}: {type(e).__name__}")
        if errors:
            return False, f"gold.db ({size_mb}MB): {', '.join(errors)}"
    finally:
        con.close()
    return True, (
        f"gold.db exists ({size_mb}MB, {counts['bars_1m']:,} bars_1m, "
        f"{counts['bars_5m']:,} bars_5m, {counts['daily_features']:,} daily_features)"
    )

def check_dbn_files() -> tuple[bool, str]:
    """Check DBN files are present."""
    if not DAILY_DBN_DIR.exists():
        return False, f"Data dir missing: {DAILY_DBN_DIR}"
    count = len(list(DAILY_DBN_DIR.glob("glbx-mdp3-*.ohlcv-1m.dbn.zst")))
    if count == 0:
        return False, "No .dbn.zst files found"
    return True, f"{count:,} DBN files present"

def check_drift() -> tuple[bool, str]:
    """Run drift detection."""
    try:
        proc = subprocess.run(
            [sys.executable, "pipeline/check_drift.py"],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode == 0:
            # Count actual pass/fail from output
            passed = proc.stdout.count("PASSED [OK]")
            failed = proc.stdout.count("FAILED:")
            total = passed + failed
            return True, f"Drift detection: {passed}/{total} passing"
        # Non-zero exit: parse pass/fail counts for diagnostics
        passed = proc.stdout.count("PASSED [OK]")
        failed = proc.stdout.count("FAILED:")
        total = passed + failed
        if total > 0:
            return False, f"Drift detection: {passed}/{total} passing ({failed} FAILED)"
        return False, "Drift detection: FAILED"
    except Exception as e:
        return False, f"Drift detection error: {e}"

def check_tests() -> tuple[bool, str]:
    """Run test suite."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-x", "-q", "--tb=no"],
            capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        # Parse "N passed" from output
        for line in proc.stdout.splitlines():
            if "passed" in line:
                return proc.returncode == 0, f"Tests: {line.strip()}"
        return proc.returncode == 0, f"Tests: exit code {proc.returncode}"
    except Exception as e:
        return False, f"Tests error: {e}"

def check_integrity() -> tuple[bool, str]:
    """Run data integrity audit."""
    try:
        proc = subprocess.run(
            [sys.executable, "scripts/tools/audit_integrity.py"],
            capture_output=True, text=True, timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode == 0:
            # Parse dynamic check count from audit output
            for line in reversed(proc.stdout.splitlines()):
                if "checks clean" in line:
                    return True, f"Integrity audit: {line.strip()}"
            return True, "Integrity audit: passed"
        # Extract violation count from output
        for line in reversed(proc.stdout.splitlines()):
            if "violation" in line:
                return False, f"Integrity audit: {line.strip()}"
        return False, "Integrity audit: FAILED"
    except Exception as e:
        return False, f"Integrity audit error: {e}"

def check_git_hooks() -> tuple[bool, str]:
    """Check git hooks are configured."""
    hooks_dir = PROJECT_ROOT / ".githooks"
    pre_commit = hooks_dir / "pre-commit"
    if not pre_commit.exists():
        return False, "Git hooks: .githooks/pre-commit missing"

    try:
        proc = subprocess.run(
            ["git", "config", "core.hooksPath"],
            capture_output=True, text=True, timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        hooks_path = proc.stdout.strip()
        if hooks_path == ".githooks":
            return True, "Git hooks: configured"
        return False, f"Git hooks: core.hooksPath is '{hooks_path}', expected '.githooks'"
    except Exception:
        return False, "Git hooks: cannot read git config"

def main():
    print("=" * 50)
    print("PIPELINE HEALTH CHECK")
    print("=" * 50)
    print()

    checks = [
        check_python_deps,
        check_database,
        check_dbn_files,
        check_drift,
        check_integrity,
        check_tests,
        check_git_hooks,
    ]

    all_ok = True
    for check in checks:
        ok, msg = check()
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {msg}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
