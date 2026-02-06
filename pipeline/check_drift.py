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


def check_schema_query_consistency(pipeline_dir: Path) -> list[str]:
    """Check that all SQL table references exist in init_db.py schema."""
    violations = []

    init_db_path = pipeline_dir / "init_db.py"
    if not init_db_path.exists():
        return violations

    # Extract tables defined in init_db.py
    init_content = init_db_path.read_text(encoding='utf-8')
    create_tables = set(re.findall(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', init_content, re.IGNORECASE))

    if not create_tables:
        return violations

    # Scan all pipeline files for SQL table references
    table_ref_patterns = [
        re.compile(r'(?:FROM|INTO|UPDATE|JOIN)\s+(\w+)', re.IGNORECASE),
    ]

    for fpath in pipeline_dir.glob("*.py"):
        if fpath.name == "init_db.py" or fpath.name == "check_drift.py":
            continue

        content = fpath.read_text(encoding='utf-8')
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue

            for pattern in table_ref_patterns:
                for match in pattern.finditer(line):
                    table = match.group(1)
                    # Skip SQL keywords and common false positives
                    if table.upper() in ('SELECT', 'WHERE', 'AND', 'OR', 'NOT', 'NULL',
                                          'SET', 'VALUES', 'ORDER', 'GROUP', 'BY', 'AS',
                                          'HAVING', 'LIMIT', 'OFFSET', 'DISTINCT', 'ON',
                                          'TRANSACTION', 'TABLE', 'REPLACE', 'EXISTS',
                                          'SCHEMA', 'COLUMNS', 'INFORMATION_SCHEMA',
                                          'MAIN', 'INDEX', 'VIEW', 'TYPE'):
                        continue
                    # Skip information_schema references
                    if 'information_schema' in line.lower():
                        continue
                    if table not in create_tables:
                        violations.append(
                            f"  {fpath.name}:{line_num}: References table '{table}' not in init_db.py schema"
                        )

    return violations


def check_import_cycles(pipeline_dir: Path) -> list[str]:
    """Check for import cycles between pipeline modules."""
    violations = []

    # ingest_dbn_mgc.py must NOT import from ingest_dbn.py (reverse dependency)
    mgc_path = pipeline_dir / "ingest_dbn_mgc.py"
    if not mgc_path.exists():
        return violations

    content = mgc_path.read_text(encoding='utf-8')
    lines = content.splitlines()

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if 'from pipeline.ingest_dbn import' in line or 'import pipeline.ingest_dbn' in line:
            violations.append(
                f"  ingest_dbn_mgc.py:{line_num}: Imports from ingest_dbn.py (circular dependency risk)"
            )
        if 'from .ingest_dbn import' in line or 'from . import ingest_dbn' in line:
            violations.append(
                f"  ingest_dbn_mgc.py:{line_num}: Relative import from ingest_dbn (circular dependency risk)"
            )

    return violations


def check_hardcoded_paths(pipeline_dir: Path) -> list[str]:
    """Check for hardcoded absolute Windows paths in pipeline code."""
    violations = []

    path_patterns = [
        re.compile(r'["\']C:\\Users', re.IGNORECASE),
        re.compile(r'["\']C:/Users', re.IGNORECASE),
        re.compile(r'["\']D:\\', re.IGNORECASE),
        re.compile(r'["\']D:/', re.IGNORECASE),
    ]

    for fpath in pipeline_dir.glob("*.py"):
        if fpath.name == "check_drift.py":
            continue

        content = fpath.read_text(encoding='utf-8')
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue

            for pattern in path_patterns:
                if pattern.search(line):
                    violations.append(
                        f"  {fpath.name}:{line_num}: Hardcoded absolute path: {stripped[:80]}"
                    )

    return violations


def check_connection_leaks(pipeline_dir: Path) -> list[str]:
    """Check that duckdb.connect() calls have proper cleanup."""
    violations = []

    for fpath in pipeline_dir.glob("*.py"):
        if fpath.name in ("check_drift.py", "check_db.py", "dashboard.py", "health_check.py", "__init__.py"):
            continue

        content = fpath.read_text(encoding='utf-8')

        # Count duckdb.connect() calls
        connect_count = len(re.findall(r'duckdb\.connect\(', content))
        if connect_count == 0:
            continue

        # Check for cleanup mechanisms
        has_close = 'con.close()' in content or '.close()' in content
        has_finally = 'finally:' in content
        has_atexit = 'atexit' in content
        has_with = 'with duckdb' in content

        if not (has_close or has_finally or has_atexit or has_with):
            violations.append(
                f"  {fpath.name}: {connect_count} duckdb.connect() calls but no close/finally/atexit/with"
            )

    return violations


def check_dashboard_readonly(pipeline_dir: Path) -> list[str]:
    """Check that dashboard.py only reads from the database (no writes)."""
    violations = []

    dash_path = pipeline_dir / "dashboard.py"
    if not dash_path.exists():
        return violations

    content = dash_path.read_text(encoding='utf-8')

    write_patterns = [
        re.compile(r'INSERT\s+', re.IGNORECASE),
        re.compile(r'DELETE\s+FROM', re.IGNORECASE),
        re.compile(r'UPDATE\s+\w+\s+SET', re.IGNORECASE),
        re.compile(r'DROP\s+TABLE', re.IGNORECASE),
        re.compile(r'CREATE\s+TABLE', re.IGNORECASE),
    ]

    lines = content.splitlines()
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        for pattern in write_patterns:
            if pattern.search(line):
                violations.append(
                    f"  dashboard.py:{line_num}: DB write detected (must be read-only): {stripped[:80]}"
                )

    # Also verify read_only=True is used
    if 'duckdb.connect(' in content and 'read_only=True' not in content:
        violations.append(
            "  dashboard.py: duckdb.connect() without read_only=True"
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

    # Check 4: Import cycle prevention
    print("Check 4: Import cycle prevention...")
    v = check_import_cycles(PIPELINE_DIR)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 5: Hardcoded absolute paths
    print("Check 5: Hardcoded absolute paths...")
    v = check_hardcoded_paths(PIPELINE_DIR)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 6: Connection leak detection
    print("Check 6: Connection leak detection...")
    v = check_connection_leaks(PIPELINE_DIR)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 7: Dashboard must be read-only
    print("Check 7: Dashboard read-only enforcement...")
    v = check_dashboard_readonly(PIPELINE_DIR)
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
