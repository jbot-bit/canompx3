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
TRADING_APP_DIR = PROJECT_ROOT / "trading_app"

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
    """Check that SQL table references in triple-quoted strings match init_db.py schema.

    Only scans inside triple-quoted SQL strings (containing SELECT/INSERT/DELETE/UPDATE)
    to avoid false positives from Python import statements and comments.
    """
    violations = []

    init_db_path = pipeline_dir / "init_db.py"
    if not init_db_path.exists():
        return violations

    # Extract tables defined in init_db.py
    init_content = init_db_path.read_text(encoding='utf-8')
    create_tables = set(re.findall(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', init_content, re.IGNORECASE))

    if not create_tables:
        return violations

    # SQL keywords and known non-table identifiers to skip
    sql_keywords = {
        'SELECT', 'WHERE', 'AND', 'OR', 'NOT', 'NULL', 'SET', 'VALUES',
        'ORDER', 'GROUP', 'BY', 'AS', 'HAVING', 'LIMIT', 'OFFSET',
        'DISTINCT', 'ON', 'TRANSACTION', 'TABLE', 'REPLACE', 'EXISTS',
        'SCHEMA', 'COLUMNS', 'INFORMATION_SCHEMA', 'MAIN', 'INDEX',
        'VIEW', 'TYPE', 'INTERVAL', 'ZONE', 'CAST', 'COUNT', 'MIN',
        'MAX', 'SUM', 'AVG', 'FIRST', 'LAST', 'EXTRACT', 'EPOCH',
        'BEGIN', 'COMMIT', 'ROLLBACK', 'DROP', 'CREATE', 'IF',
    }

    # Extract CTE names: WITH x AS (...), y AS (...), z AS (...)
    # Matches both the first CTE (after WITH) and subsequent ones (after comma)
    cte_pattern = re.compile(r'(?:WITH|,)\s+(\w+)\s+AS\s*\(', re.IGNORECASE)

    # Pattern for table references in SQL
    table_ref_pattern = re.compile(r'(?:FROM|INTO|UPDATE|JOIN)\s+(\w+)', re.IGNORECASE)

    # Extract triple-quoted strings that contain SQL
    sql_string_pattern = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.DOTALL)
    sql_indicator = re.compile(r'\b(SELECT|INSERT|DELETE|UPDATE|CREATE)\b', re.IGNORECASE)

    for fpath in pipeline_dir.glob("*.py"):
        if fpath.name in ("init_db.py", "check_drift.py"):
            continue

        content = fpath.read_text(encoding='utf-8')

        for match in sql_string_pattern.finditer(content):
            sql_text = match.group(1) or match.group(2)

            # Only check strings that look like SQL
            if not sql_indicator.search(sql_text):
                continue

            # Extract CTE names from this SQL block
            cte_names = {m.group(1) for m in cte_pattern.finditer(sql_text)}

            # Find line number of the match start
            line_num = content[:match.start()].count('\n') + 1

            for ref_match in table_ref_pattern.finditer(sql_text):
                table = ref_match.group(1)
                if table.upper() in sql_keywords:
                    continue
                if 'information_schema' in sql_text.lower():
                    continue
                # Skip CTE names and known DataFrame variable patterns
                if table in cte_names:
                    continue
                if table.endswith('_df') or table.endswith('_frame'):
                    continue
                # Skip column names used after EXTRACT(... FROM col)
                extract_ctx = sql_text[max(0, ref_match.start()-30):ref_match.start()]
                if 'EXTRACT' in extract_ctx.upper() or 'EPOCH' in extract_ctx.upper():
                    continue
                if table not in create_tables:
                    violations.append(
                        f"  {fpath.name}:~{line_num}: SQL references table '{table}' not in init_db.py schema"
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


def check_pipeline_never_imports_trading_app(pipeline_dir: Path) -> list[str]:
    """Check that pipeline/ modules NEVER import from trading_app/.

    One-way dependency rule:
      trading_app CAN import from pipeline (cost_model, paths, init_db)
      pipeline NEVER imports from trading_app
    """
    violations = []

    import_patterns = [
        re.compile(r'from\s+trading_app'),
        re.compile(r'import\s+trading_app'),
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

            for pattern in import_patterns:
                if pattern.search(line):
                    violations.append(
                        f"  {fpath.name}:{line_num}: Imports from trading_app "
                        f"(one-way dependency violation): {stripped[:80]}"
                    )

    return violations


def check_trading_app_connection_leaks(trading_app_dir: Path) -> list[str]:
    """Check that duckdb.connect() calls in trading_app/ have proper cleanup."""
    violations = []

    if not trading_app_dir.exists():
        return violations

    for fpath in trading_app_dir.rglob("*.py"):
        if fpath.name == "__init__.py":
            continue

        content = fpath.read_text(encoding='utf-8')

        connect_count = len(re.findall(r'duckdb\.connect\(', content))
        if connect_count == 0:
            continue

        has_close = 'con.close()' in content or '.close()' in content
        has_finally = 'finally:' in content
        has_atexit = 'atexit' in content
        has_with = 'with duckdb' in content

        if not (has_close or has_finally or has_atexit or has_with):
            violations.append(
                f"  {fpath.name}: {connect_count} duckdb.connect() calls "
                f"but no close/finally/atexit/with"
            )

    return violations


def check_trading_app_hardcoded_paths(trading_app_dir: Path) -> list[str]:
    """Check for hardcoded absolute Windows paths in trading_app code."""
    violations = []

    if not trading_app_dir.exists():
        return violations

    path_patterns = [
        re.compile(r'["\']C:\\Users', re.IGNORECASE),
        re.compile(r'["\']C:/Users', re.IGNORECASE),
        re.compile(r'["\']D:\\', re.IGNORECASE),
        re.compile(r'["\']D:/', re.IGNORECASE),
    ]

    for fpath in trading_app_dir.rglob("*.py"):
        content = fpath.read_text(encoding='utf-8')
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue

            for pattern in path_patterns:
                if pattern.search(line):
                    violations.append(
                        f"  {fpath.name}:{line_num}: Hardcoded absolute path: "
                        f"{stripped[:80]}"
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


def check_config_filter_sync() -> list[str]:
    """Check that ALL_FILTERS keys match filter_type inside each filter."""
    violations = []

    try:
        from trading_app.config import ALL_FILTERS

        for key, filt in ALL_FILTERS.items():
            if filt.filter_type != key:
                violations.append(
                    f"  ALL_FILTERS['{key}'].filter_type = '{filt.filter_type}' (mismatch)"
                )
    except ImportError:
        # trading_app may not exist yet
        pass

    return violations


def check_entry_models_sync() -> list[str]:
    """Check that ENTRY_MODELS constant matches expected values."""
    violations = []

    try:
        from trading_app.config import ENTRY_MODELS

        expected = ["E1", "E2", "E3"]
        if ENTRY_MODELS != expected:
            violations.append(
                f"  ENTRY_MODELS = {ENTRY_MODELS}, expected {expected}"
            )
    except ImportError:
        pass

    return violations


def check_nested_isolation() -> list[str]:
    """Check that trading_app/nested/ never imports from production modules.

    One-way dependency rule for nested subpackage:
      nested CAN import from: pipeline/, trading_app/config.py, trading_app/entry_rules.py,
        trading_app/outcome_builder.py (for compute_single_outcome, RR_TARGETS, etc.),
        trading_app/strategy_discovery.py (for compute_metrics, _load_daily_features, etc.),
        trading_app/strategy_validator.py (for validate_strategy)
      nested NEVER imports from: trading_app/db_manager.py (production schema)
    """
    violations = []

    nested_dir = TRADING_APP_DIR / "nested"
    if not nested_dir.exists():
        return violations

    # Forbidden: importing init_trading_app_schema or verify_trading_app_schema
    # (nested has its own schema module)
    forbidden_patterns = [
        re.compile(r'from\s+trading_app\.db_manager\s+import'),
        re.compile(r'import\s+trading_app\.db_manager'),
    ]

    for fpath in nested_dir.glob("*.py"):
        if fpath.name == "__init__.py":
            continue

        content = fpath.read_text(encoding='utf-8')
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            for pattern in forbidden_patterns:
                if pattern.search(line):
                    violations.append(
                        f"  nested/{fpath.name}:{line_num}: Imports from db_manager "
                        f"(nested must use own schema): {stripped[:80]}"
                    )

    return violations


def check_entry_price_sanity() -> list[str]:
    """Flag entry_price = orb_high/orb_low in outcome code without E3 guard.

    Catches regression to the broken behavior where entry_price was set to
    the ORB level regardless of entry model.
    """
    violations = []

    outcome_path = TRADING_APP_DIR / "outcome_builder.py"
    if not outcome_path.exists():
        return violations

    content = outcome_path.read_text(encoding='utf-8')
    lines = content.splitlines()

    # Pattern: entry_price = orb_high or entry_price = orb_low
    # This should only appear inside E3 logic in entry_rules.py, never in outcome_builder
    dangerous_patterns = [
        re.compile(r'entry_price\s*=\s*orb_high'),
        re.compile(r'entry_price\s*=\s*orb_low'),
    ]

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        for pattern in dangerous_patterns:
            if pattern.search(line):
                violations.append(
                    f"  outcome_builder.py:{line_num}: Direct entry_price = ORB level "
                    f"(must be set by entry model, not hardcoded): {stripped[:80]}"
                )

    return violations


def check_nested_production_writes() -> list[str]:
    """Check that nested/*.py never writes to production tables.

    Production tables: orb_outcomes, experimental_strategies, validated_setups.
    Nested code must only write to nested_outcomes, nested_strategies, nested_validated.
    """
    violations = []

    nested_dir = TRADING_APP_DIR / "nested"
    if not nested_dir.exists():
        return violations

    production_tables = ["orb_outcomes", "experimental_strategies", "validated_setups"]
    write_keywords = ["INSERT", "DELETE", "UPDATE", "DROP"]

    for fpath in nested_dir.glob("*.py"):
        if fpath.name == "__init__.py":
            continue

        content = fpath.read_text(encoding='utf-8')
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue

            for table in production_tables:
                for keyword in write_keywords:
                    pattern = re.compile(
                        rf'{keyword}\s+(?:OR\s+REPLACE\s+)?(?:INTO\s+|FROM\s+)?{table}\b',
                        re.IGNORECASE,
                    )
                    if pattern.search(line):
                        violations.append(
                            f"  nested/{fpath.name}:{line_num}: SQL write to production table "
                            f"'{table}': {stripped[:80]}"
                        )

    return violations


def check_schema_query_consistency_trading_app(trading_app_dir: Path) -> list[str]:
    """Extend Check 4 to scan trading_app/ for SQL table reference consistency."""
    violations = []

    # Gather ALL known tables from all schema files
    create_tables = set()
    schema_files = [
        PIPELINE_DIR / "init_db.py",
        trading_app_dir / "db_manager.py",
        trading_app_dir / "nested" / "schema.py",
    ]
    for sf in schema_files:
        if sf.exists():
            sf_content = sf.read_text(encoding='utf-8')
            create_tables.update(re.findall(
                r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', sf_content, re.IGNORECASE
            ))

    if not create_tables:
        return violations

    sql_keywords = {
        'SELECT', 'WHERE', 'AND', 'OR', 'NOT', 'NULL', 'SET', 'VALUES',
        'ORDER', 'GROUP', 'BY', 'AS', 'HAVING', 'LIMIT', 'OFFSET',
        'DISTINCT', 'ON', 'TRANSACTION', 'TABLE', 'REPLACE', 'EXISTS',
        'SCHEMA', 'COLUMNS', 'INFORMATION_SCHEMA', 'MAIN', 'INDEX',
        'VIEW', 'TYPE', 'INTERVAL', 'ZONE', 'CAST', 'COUNT', 'MIN',
        'MAX', 'SUM', 'AVG', 'FIRST', 'LAST', 'EXTRACT', 'EPOCH',
        'BEGIN', 'COMMIT', 'ROLLBACK', 'DROP', 'CREATE', 'IF',
        'TEXT', 'INTEGER', 'DOUBLE', 'BOOLEAN', 'DATE', 'TIMESTAMP',
        'TIMESTAMPTZ', 'VARCHAR', 'BIGINT', 'REAL', 'FLOAT',
    }

    cte_pattern = re.compile(r'(?:WITH|,)\s+(\w+)\s+AS\s*\(', re.IGNORECASE)
    table_ref_pattern = re.compile(r'(?:FROM|INTO|UPDATE|JOIN)\s+(\w+)', re.IGNORECASE)
    sql_string_pattern = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.DOTALL)
    sql_indicator = re.compile(r'\b(SELECT|INSERT|DELETE|UPDATE|CREATE)\b', re.IGNORECASE)

    if not trading_app_dir.exists():
        return violations

    for fpath in trading_app_dir.rglob("*.py"):
        if fpath.name in ("__init__.py", "schema.py", "db_manager.py"):
            continue

        content = fpath.read_text(encoding='utf-8')

        for match in sql_string_pattern.finditer(content):
            sql_text = match.group(1) or match.group(2)

            # Require at least 2 SQL keywords to distinguish real SQL from docstrings
            # (docstrings can contain "Update", "Create" etc. as English words)
            sql_hits = sql_indicator.findall(sql_text)
            if len(sql_hits) < 2:
                continue

            cte_names = {m.group(1) for m in cte_pattern.finditer(sql_text)}
            line_num = content[:match.start()].count('\n') + 1

            for ref_match in table_ref_pattern.finditer(sql_text):
                table = ref_match.group(1)
                if table.upper() in sql_keywords:
                    continue
                if 'information_schema' in sql_text.lower():
                    continue
                if table in cte_names:
                    continue
                if table.endswith('_df') or table.endswith('_frame'):
                    continue
                extract_ctx = sql_text[max(0, ref_match.start()-30):ref_match.start()]
                if 'EXTRACT' in extract_ctx.upper() or 'EPOCH' in extract_ctx.upper():
                    continue
                if table not in create_tables:
                    violations.append(
                        f"  {fpath.name}:~{line_num}: SQL references table '{table}' "
                        f"not in schema"
                    )

    return violations


def check_timezone_hygiene() -> list[str]:
    """Block pytz imports and hardcoded timedelta(hours=10) patterns.

    Known footguns:
      - pytz has surprising DST normalization behavior
      - timedelta(hours=10) is a hardcoded Brisbane offset instead of using zoneinfo
    """
    violations = []

    pytz_pattern = re.compile(r'import\s+pytz|from\s+pytz\s+import')
    # Match timedelta(hours=10) with optional spaces
    hardcoded_tz_pattern = re.compile(r'timedelta\s*\(\s*hours\s*=\s*10\s*\)')

    for base_dir in [PIPELINE_DIR, TRADING_APP_DIR]:
        if not base_dir.exists():
            continue

        for fpath in base_dir.rglob("*.py"):
            if fpath.name in ("__init__.py", "check_drift.py"):
                continue

            content = fpath.read_text(encoding='utf-8')
            lines = content.splitlines()

            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue

                if pytz_pattern.search(line):
                    violations.append(
                        f"  {fpath.name}:{line_num}: pytz import "
                        f"(use zoneinfo instead): {stripped[:80]}"
                    )
                if hardcoded_tz_pattern.search(line):
                    violations.append(
                        f"  {fpath.name}:{line_num}: Hardcoded timedelta(hours=10) "
                        f"(use timezone.utc or zoneinfo): {stripped[:80]}"
                    )

    return violations


def check_all_imports_resolve() -> list[str]:
    """Verify that all .py files in pipeline/ and trading_app/ can be imported.

    Catches typos in import statements, missing dependencies, and broken
    module references BEFORE runtime.
    """
    import importlib
    violations = []

    # Ensure project root is on sys.path (modules use sys.path.insert)
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    for base_dir in [PIPELINE_DIR, TRADING_APP_DIR]:
        if not base_dir.exists():
            continue

        for fpath in base_dir.rglob("*.py"):
            if fpath.name.startswith("_") and fpath.name != "__init__.py":
                continue
            if fpath.name == "__init__.py":
                continue
            # Skip check_drift itself (circular)
            if fpath.name == "check_drift.py":
                continue

            # Convert file path to module path
            rel = fpath.relative_to(PROJECT_ROOT)
            module = str(rel).replace("\\", "/").replace("/", ".").removesuffix(".py")

            try:
                importlib.import_module(module)
            except Exception as e:
                err_type = type(e).__name__
                violations.append(
                    f"  {module}: {err_type}: {str(e)[:100]}"
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

    # Check 4: Schema-query table name consistency
    print("Check 4: Schema-query table name consistency...")
    v = check_schema_query_consistency(PIPELINE_DIR)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 5: Import cycle prevention
    print("Check 5: Import cycle prevention...")
    v = check_import_cycles(PIPELINE_DIR)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 6: Hardcoded absolute paths
    print("Check 6: Hardcoded absolute paths...")
    v = check_hardcoded_paths(PIPELINE_DIR)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 7: Connection leak detection
    print("Check 7: Connection leak detection...")
    v = check_connection_leaks(PIPELINE_DIR)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 8: Dashboard must be read-only
    print("Check 8: Dashboard read-only enforcement...")
    v = check_dashboard_readonly(PIPELINE_DIR)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 9: pipeline/ must never import from trading_app/
    print("Check 9: Pipeline never imports trading_app (one-way dependency)...")
    v = check_pipeline_never_imports_trading_app(PIPELINE_DIR)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 10: Connection leak detection in trading_app/
    print("Check 10: Trading app connection leak detection...")
    v = check_trading_app_connection_leaks(TRADING_APP_DIR)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 11: Hardcoded paths in trading_app/
    print("Check 11: Trading app hardcoded paths...")
    v = check_trading_app_hardcoded_paths(TRADING_APP_DIR)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 12: Config/DB sync — filter_type keys match filter objects
    print("Check 12: Config filter_type sync...")
    v = check_config_filter_sync()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 13: ENTRY_MODELS sync
    print("Check 13: ENTRY_MODELS sync...")
    v = check_entry_models_sync()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 14: Entry price sanity (no hardcoded ORB level in outcome_builder)
    print("Check 14: Entry price sanity...")
    v = check_entry_price_sanity()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 15: Nested subpackage isolation (no db_manager imports)
    print("Check 15: Nested subpackage isolation...")
    v = check_nested_isolation()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 16: All imports resolve (no typos, missing deps)
    print("Check 16: All imports resolve...")
    v = check_all_imports_resolve()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 17: Nested code must never write to production tables
    print("Check 17: Nested production table write guard...")
    v = check_nested_production_writes()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 18: Schema-query consistency in trading_app/
    print("Check 18: Trading app schema-query consistency...")
    v = check_schema_query_consistency_trading_app(TRADING_APP_DIR)
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 19: Timezone hygiene (no pytz, no hardcoded UTC+10)
    print("Check 19: Timezone hygiene...")
    v = check_timezone_hygiene()
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
