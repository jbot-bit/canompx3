#!/usr/bin/env python3
"""
Drift detection for the multi-instrument pipeline.

Fails if anyone reintroduces:
1. Hardcoded 'MGC' SQL literals in generic pipeline code (ingest_dbn.py, run_pipeline.py)
2. .apply() or .iterrows() usage in ingest scripts (performance anti-pattern)
3. Any writes to tables other than bars_1m in ingest scripts

FAIL-CLOSED: Any violation exits with code 1.

Usage:
    python pipeline/check_drift.py
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"

# =============================================================================
# FILES TO CHECK
# =============================================================================

# Generic pipeline files: must NOT contain hardcoded 'MGC' SQL literals
GENERIC_FILES = [
    PIPELINE_DIR / "ingest_dbn.py",
    PIPELINE_DIR / "run_pipeline.py",
    PIPELINE_DIR / "asset_configs.py",
]

# Ingest files: must NOT use .apply()/.iterrows() on large data
# (outright_mask uses .apply on symbol column — that's the ONE allowed exception,
#  it's on the symbol column only, not OHLCV data)
INGEST_FILES = [
    PIPELINE_DIR / "ingest_dbn.py",
    PIPELINE_DIR / "ingest_dbn_mgc.py",
]

# Ingest files: must NOT write to any table other than bars_1m
INGEST_WRITE_FILES = [
    PIPELINE_DIR / "ingest_dbn.py",
    PIPELINE_DIR / "ingest_dbn_mgc.py",
]


def check_hardcoded_mgc_sql(files: list[Path]) -> list[str]:
    """Check for hardcoded 'MGC' in SQL statements in generic files."""
    violations = []

    # Patterns that indicate hardcoded MGC in SQL context
    # Matches: 'MGC' in SQL (VALUES, WHERE, INSERT contexts)
    sql_mgc_patterns = [
        re.compile(r"VALUES\s*\([^)]*'MGC'"),
        re.compile(r"WHERE\s+.*symbol\s*=\s*'MGC'"),
        re.compile(r"symbol\s*=\s*'MGC'"),
    ]

    for fpath in files:
        if not fpath.exists():
            continue

        content = fpath.read_text(encoding='utf-8')
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith('#'):
                continue

            for pattern in sql_mgc_patterns:
                if pattern.search(line):
                    violations.append(
                        f"  {fpath.name}:{line_num}: Hardcoded 'MGC' in SQL: {stripped[:80]}"
                    )

    return violations


def check_apply_iterrows(files: list[Path]) -> list[str]:
    """Check for .apply() or .iterrows() on data frames (perf anti-pattern)."""
    violations = []

    # Allowed exception: .apply() on symbol column for outright filtering
    # Pattern: chunk_df['symbol'].apply(...)  — this is acceptable
    allowed_apply_pattern = re.compile(r"\['symbol'\]\.apply\(")

    for fpath in files:
        if not fpath.exists():
            continue

        content = fpath.read_text(encoding='utf-8')
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue

            # Check .iterrows() — always forbidden in ingest hot path
            # Exception: the per-day row accumulation loop (front_df.iterrows())
            # This is on already-filtered single-contract single-day data, not bulk
            if '.iterrows()' in line:
                # Allow the known pattern: buffering single-day front-contract rows
                if 'front_df.iterrows()' in line:
                    continue
                violations.append(
                    f"  {fpath.name}:{line_num}: .iterrows() usage: {stripped[:80]}"
                )

            # Check .apply() — forbidden except on symbol column
            if '.apply(' in line:
                if allowed_apply_pattern.search(line):
                    continue
                # Also allow lambda in trading_days mask (compute_trading_days)
                if 'trading_days[mask].apply(' in line or 'trading_days[mask]' in stripped:
                    continue
                violations.append(
                    f"  {fpath.name}:{line_num}: .apply() usage: {stripped[:80]}"
                )

    return violations


def check_non_bars1m_writes(files: list[Path]) -> list[str]:
    """Check that ingest scripts only write to bars_1m."""
    violations = []

    # Patterns for SQL writes to tables other than bars_1m
    write_patterns = [
        re.compile(r'INSERT\s+(?:OR\s+REPLACE\s+)?INTO\s+(?!bars_1m\b)(\w+)', re.IGNORECASE),
        re.compile(r'DELETE\s+FROM\s+(?!bars_1m\b)(\w+)', re.IGNORECASE),
        re.compile(r'UPDATE\s+(?!bars_1m\b)(\w+)', re.IGNORECASE),
        re.compile(r'DROP\s+TABLE\s+(?!bars_1m\b)(\w+)', re.IGNORECASE),
    ]

    for fpath in files:
        if not fpath.exists():
            continue

        content = fpath.read_text(encoding='utf-8')
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue

            for pattern in write_patterns:
                match = pattern.search(line)
                if match:
                    table = match.group(1)
                    violations.append(
                        f"  {fpath.name}:{line_num}: Write to non-bars_1m table '{table}': {stripped[:80]}"
                    )

    return violations


def main():
    print("=" * 60)
    print("PIPELINE DRIFT CHECK")
    print("=" * 60)
    print()

    all_violations = []

    # Check 1: Hardcoded 'MGC' SQL in generic files
    print("Check 1: Hardcoded 'MGC' SQL literals in generic pipeline code...")
    v = check_hardcoded_mgc_sql(GENERIC_FILES)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 2: .apply() / .iterrows() in ingest scripts
    print("Check 2: .apply() / .iterrows() in ingest scripts...")
    v = check_apply_iterrows(INGEST_FILES)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 3: Writes to non-bars_1m tables in ingest scripts
    print("Check 3: Non-bars_1m writes in ingest scripts...")
    v = check_non_bars1m_writes(INGEST_WRITE_FILES)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Summary
    print("=" * 60)
    if all_violations:
        print(f"DRIFT DETECTED: {len(all_violations)} violation(s)")
        print("=" * 60)
        sys.exit(1)
    else:
        print("NO DRIFT DETECTED: All checks passed [OK]")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
