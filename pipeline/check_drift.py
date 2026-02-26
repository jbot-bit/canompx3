#!/usr/bin/env python3
"""
Drift detection for the multi-instrument pipeline.

Fails if anyone reintroduces:
1. Hardcoded 'MGC' SQL literals in generic pipeline code (ingest_dbn.py, run_pipeline.py)
2. .apply() or .iterrows() usage in ingest scripts (performance anti-pattern)
3. Any writes to tables other than bars_1m in ingest scripts
   (covers: ingest_dbn.py, ingest_dbn_mgc.py, ingest_dbn_daily.py,
    scripts/infra/run_parallel_ingest.py)

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
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

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
    PIPELINE_DIR / "ingest_dbn_daily.py",
]

# Ingest files: must NOT write to any table other than bars_1m
INGEST_WRITE_FILES = [
    PIPELINE_DIR / "ingest_dbn.py",
    PIPELINE_DIR / "ingest_dbn_mgc.py",
    PIPELINE_DIR / "ingest_dbn_daily.py",
    SCRIPTS_DIR / "infra" / "run_parallel_ingest.py",
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

    # Gather tables from all schema files (pipeline + trading_app)
    create_tables = set()
    schema_files = [
        pipeline_dir / "init_db.py",
        pipeline_dir.parent / "trading_app" / "db_manager.py",
        pipeline_dir.parent / "trading_app" / "nested" / "schema.py",
    ]
    for sf in schema_files:
        if sf.exists():
            sf_content = sf.read_text(encoding='utf-8')
            create_tables.update(re.findall(
                r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', sf_content, re.IGNORECASE
            ))

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
    """Check that duckdb.connect() calls have proper cleanup.

    Improved: counts connect() vs close() calls. If connect > close
    AND no finally/atexit/with-statement, flags it.
    """
    violations = []

    for fpath in pipeline_dir.glob("*.py"):
        if fpath.name in ("check_drift.py", "check_db.py", "dashboard.py", "health_check.py", "__init__.py"):
            continue

        content = fpath.read_text(encoding='utf-8')

        connect_count = len(re.findall(r'duckdb\.connect\(', content))
        if connect_count == 0:
            continue

        close_count = len(re.findall(r'\.close\(\)', content))
        has_finally = 'finally:' in content
        has_atexit = 'atexit' in content
        # Also recognise wrapper helpers that return duckdb.connect() and are
        # called via ``with _connect(...) as con:``
        has_with = 'with duckdb' in content or (
            bool(re.search(r'return\s+duckdb\.connect\(', content))
            and bool(re.search(r'\bwith\s+\w+\(', content))
        )

        if has_finally or has_atexit or has_with:
            continue

        if close_count < connect_count:
            violations.append(
                f"  {fpath.name}: {connect_count} duckdb.connect() calls but only "
                f"{close_count} .close() calls and no finally/atexit/with"
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
    """Check that duckdb.connect() calls in trading_app/ have proper cleanup.

    Improved: counts connect() vs close() calls. If connect > close
    AND no finally/atexit/with-statement, flags it.
    """
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

        close_count = len(re.findall(r'\.close\(\)', content))
        has_finally = 'finally:' in content
        has_atexit = 'atexit' in content
        # Also recognise wrapper helpers that return duckdb.connect() and are
        # called via ``with _connect(...) as con:``
        has_with = 'with duckdb' in content or (
            bool(re.search(r'return\s+duckdb\.connect\(', content))
            and bool(re.search(r'\bwith\s+\w+\(', content))
        )

        if has_finally or has_atexit or has_with:
            continue

        if close_count < connect_count:
            violations.append(
                f"  {fpath.name}: {connect_count} duckdb.connect() calls but only "
                f"{close_count} .close() calls and no finally/atexit/with"
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

    # Ensure project root is on sys.path so trading_app is importable
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from trading_app.config import ALL_FILTERS

        for key, filt in ALL_FILTERS.items():
            if filt.filter_type != key:
                violations.append(
                    f"  ALL_FILTERS['{key}'].filter_type = '{filt.filter_type}' (mismatch)"
                )
    except ImportError as e:
        violations.append(f"  Cannot import trading_app.config.ALL_FILTERS: {e}")

    return violations


def check_entry_models_sync() -> list[str]:
    """Check that ENTRY_MODELS constant matches expected values."""
    violations = []

    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from trading_app.config import ENTRY_MODELS

        expected = ["E1", "E2", "E3"]
        if ENTRY_MODELS != expected:
            violations.append(
                f"  ENTRY_MODELS = {ENTRY_MODELS}, expected {expected}"
            )
    except ImportError as e:
        violations.append(f"  Cannot import trading_app.config.ENTRY_MODELS: {e}")

    return violations


def check_nested_isolation() -> list[str]:
    """Check that trading_app/nested/ and trading_app/regime/ never import from production modules.

    One-way dependency rule for nested/regime subpackages:
      CAN import from: pipeline/, trading_app/config.py, trading_app/entry_rules.py,
        trading_app/outcome_builder.py (for compute_single_outcome, RR_TARGETS, etc.),
        trading_app/strategy_discovery.py (for compute_metrics, _load_daily_features, etc.),
        trading_app/strategy_validator.py (for validate_strategy)
      NEVER imports from: trading_app/db_manager.py (production schema)
    """
    violations = []

    # Forbidden: importing init_trading_app_schema or verify_trading_app_schema
    # (nested/regime have their own schema modules)
    forbidden_patterns = [
        re.compile(r'from\s+trading_app\.db_manager\s+import'),
        re.compile(r'import\s+trading_app\.db_manager'),
    ]

    for subdir_name in ["nested", "regime"]:
        subdir = TRADING_APP_DIR / subdir_name
        if not subdir.exists():
            continue

        for fpath in subdir.glob("*.py"):
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
                            f"  {subdir_name}/{fpath.name}:{line_num}: Imports from db_manager "
                            f"({subdir_name} must use own schema): {stripped[:80]}"
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
    """Check that nested/*.py and regime/*.py never write to production tables.

    Production tables: orb_outcomes, experimental_strategies, validated_setups.
    Nested/regime code must only write to their own tables.
    """
    violations = []

    production_tables = ["orb_outcomes", "experimental_strategies", "validated_setups"]
    write_keywords = ["INSERT", "DELETE", "UPDATE", "DROP"]

    for subdir_name in ["nested", "regime"]:
        subdir = TRADING_APP_DIR / subdir_name
        if not subdir.exists():
            continue

        for fpath in subdir.glob("*.py"):
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
                                f"  {subdir_name}/{fpath.name}:{line_num}: SQL write to production table "
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
        trading_app_dir / "regime" / "schema.py",
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


def check_market_state_readonly() -> list[str]:
    """Check that market_state.py, scoring.py, cascade_table.py never write to DB.

    These modules are read-only consumers. Any SQL write keyword is a violation.
    """
    violations = []
    write_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE"]

    readonly_files = [
        TRADING_APP_DIR / "market_state.py",
        TRADING_APP_DIR / "scoring.py",
        TRADING_APP_DIR / "cascade_table.py",
    ]

    for fpath in readonly_files:
        if not fpath.exists():
            continue

        content = fpath.read_text(encoding='utf-8')
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            # Skip string-only lines (docstrings, comments in strings)
            if stripped.startswith(('"""', "'''", '"', "'")):
                continue

            for keyword in write_keywords:
                # Match SQL write keywords followed by whitespace (avoids
                # matching variable names like 'update_signals')
                pattern = re.compile(
                    rf'\b{keyword}\s+(?:OR\s+REPLACE\s+)?(?:INTO|FROM|TABLE|INDEX)\b',
                    re.IGNORECASE,
                )
                if pattern.search(line):
                    violations.append(
                        f"  {fpath.name}:{line_num}: SQL write keyword "
                        f"'{keyword}': {stripped[:80]}"
                    )

    return violations


def check_sharpe_ann_presence() -> list[str]:
    """Check that sharpe_ann is computed in discovery and shown in view_strategies.

    Prevents regression where someone removes annualized Sharpe from the
    pipeline, leaving only per-trade Sharpe (which is meaningless without
    trade frequency context).
    """
    violations = []

    # Check 1: strategy_discovery.py must compute sharpe_ann in return dict
    discovery_path = TRADING_APP_DIR / "strategy_discovery.py"
    if discovery_path.exists():
        content = discovery_path.read_text(encoding='utf-8')
        if '"sharpe_ann"' not in content:
            violations.append(
                "  strategy_discovery.py: Missing 'sharpe_ann' in compute_metrics return dict"
            )
        if '"trades_per_year"' not in content:
            violations.append(
                "  strategy_discovery.py: Missing 'trades_per_year' in compute_metrics return dict"
            )

    # Check 2: view_strategies.py must show sharpe_ann in user-facing output
    view_path = TRADING_APP_DIR / "view_strategies.py"
    if view_path.exists():
        content = view_path.read_text(encoding='utf-8')
        if 'sharpe_ann' not in content:
            violations.append(
                "  view_strategies.py: Missing 'sharpe_ann' in user-facing output"
            )

    return violations


def check_ingest_authority_notice() -> list[str]:
    """Check #22: ingest_dbn_mgc.py must have deprecation notice in __main__ block."""
    path = PIPELINE_DIR / "ingest_dbn_mgc.py"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")

    main_idx = text.find('if __name__ == "__main__"')
    if main_idx == -1:
        main_idx = text.find("if __name__ == '__main__'")
    if main_idx == -1:
        return []  # No __main__ block — library-only, fine

    main_block = text[main_idx:]
    # Accept either print() or logger.info() for the deprecation notice
    required_pairs = [
        ('print("NOTE: For multi-instrument support, prefer:")',
         'logger.info("NOTE: For multi-instrument support, prefer:")'),
        ('print("  python pipeline/ingest_dbn.py --instrument MGC")',
         'logger.info("  python pipeline/ingest_dbn.py --instrument MGC")'),
    ]
    missing = [p for p, lg in required_pairs if p not in main_block and lg not in main_block]
    if missing:
        return [f"  {path.name}: Missing deprecation notice in __main__: {missing}"]
    return []


def check_validation_gate_existence() -> list[str]:
    """Check #24: Verify critical validation gate functions still exist.

    If someone deletes a gate function, this fires. Simple static grep per file.
    """
    violations = []

    gate_map = {
        PIPELINE_DIR / "ingest_dbn_mgc.py": [
            "def validate_chunk(",
            "def validate_timestamp_utc(",
            "def check_pk_safety(",
            "def check_merge_integrity(",
            "def run_final_gates(",
        ],
        PIPELINE_DIR / "build_bars_5m.py": [
            "def verify_5m_integrity(",
        ],
        PIPELINE_DIR / "ingest_dbn.py": [
            "FAIL-CLOSED",
        ],
    }

    for fpath, required_strings in gate_map.items():
        if not fpath.exists():
            violations.append(f"  {fpath.name}: File not found (gate source missing)")
            continue

        content = fpath.read_text(encoding='utf-8')
        for gate in required_strings:
            if gate not in content:
                violations.append(
                    f"  {fpath.name}: Missing gate '{gate}'"
                )

    return violations


def check_naive_datetime() -> list[str]:
    """Check #25: Block deprecated naive datetime constructors.

    Catches:
    - datetime.utcnow() (deprecated in 3.12, returns naive UTC)
    - datetime.utcfromtimestamp() (same issue)

    NOTE: datetime.now() is NOT flagged — it's used legitimately for
    wall-clock elapsed timing throughout the codebase. The dangerous
    pattern is utcnow() which implies UTC intent but returns naive.
    """
    violations = []

    dangerous_patterns = [
        (re.compile(r'datetime\.utcnow\(\)'), 'datetime.utcnow()'),
        (re.compile(r'datetime\.utcfromtimestamp\('), 'datetime.utcfromtimestamp()'),
    ]

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

                for pattern, label in dangerous_patterns:
                    if pattern.search(line):
                        violations.append(
                            f"  {fpath.name}:{line_num}: {label} is deprecated "
                            f"(use datetime.now(timezone.utc)): {stripped[:80]}"
                        )

    return violations


def check_dst_session_coverage() -> list[str]:
    """Check that all non-alias sessions are classified as DST-affected or DST-clean.

    A developer could add a session to SESSION_CATALOG and forget to classify it,
    causing silent DST contamination. This check catches that at pre-commit time.
    """
    violations = []

    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from pipeline.dst import SESSION_CATALOG, DST_AFFECTED_SESSIONS, DST_CLEAN_SESSIONS

        non_alias = {
            label for label, entry in SESSION_CATALOG.items()
            if entry["type"] != "alias"
        }
        classified = set(DST_AFFECTED_SESSIONS.keys()) | DST_CLEAN_SESSIONS
        unclassified = non_alias - classified

        if unclassified:
            violations.append(
                f"  Sessions in SESSION_CATALOG but not in DST_AFFECTED_SESSIONS or "
                f"DST_CLEAN_SESSIONS: {sorted(unclassified)}"
            )

        # Also check for stale entries (in DST sets but not in catalog)
        stale = classified - non_alias
        if stale:
            violations.append(
                f"  Sessions in DST sets but not in SESSION_CATALOG: {sorted(stale)}"
            )

    except ImportError as e:
        violations.append(f"  Cannot import pipeline.dst: {e}")

    return violations


def check_db_config_usage() -> list[str]:
    """Check that every file calling duckdb.connect() also calls configure_connection().

    Any DuckDB consumer that connects without configuring gets default PRAGMAs,
    which means no memory cap, no temp directory, and slower inserts.
    """
    violations = []

    # Files that are allowed to connect without configure_connection:
    # - Infrastructure/utilities: check_drift, db_config, init_db, dashboard, health_check
    # - Deprecated: ingest_dbn_mgc.py
    # - Read-only consumers: market_state, cascade_table, view_strategies, etc.
    # - Nested subpackages: trading_app/nested/, trading_app/regime/
    # The 7 main write-path files are enforced (ingest_dbn, ingest_dbn_daily,
    # build_bars_5m, build_daily_features, outcome_builder, strategy_discovery,
    # strategy_validator).
    EXEMPT = {
        "check_drift.py", "db_config.py", "check_db.py",
        "init_db.py", "dashboard.py", "health_check.py",
        "__init__.py",
        # Deprecated
        "ingest_dbn_mgc.py",
        # Pipeline utilities (read-only)
        "audit_bars_coverage.py", "export_parquet.py",
        # Trading app read-only consumers
        "cascade_table.py", "db_manager.py", "live_config.py",
        "market_state.py", "paper_trader.py", "portfolio.py",
        "rolling_correlation.py", "rolling_portfolio.py",
        "strategy_fitness.py", "view_strategies.py",
        # validate_1800_*.py moved to research/archive/ (I12 audit cleanup)
        # Nested subpackages
        "corpus.py", "sql_adapter.py", "strategy_matcher.py",
        "asia_session_analyzer.py", "audit_outcomes.py",
        "builder.py", "compare.py", "discovery.py",
        "schema.py", "validator.py",
    }

    for base_dir in [PIPELINE_DIR, TRADING_APP_DIR]:
        if not base_dir.exists():
            continue

        for fpath in base_dir.rglob("*.py"):
            if fpath.name in EXEMPT:
                continue

            content = fpath.read_text(encoding='utf-8')

            if 'duckdb.connect(' not in content:
                continue

            if 'configure_connection' not in content:
                violations.append(
                    f"  {fpath.name}: calls duckdb.connect() but never calls "
                    f"configure_connection() (import from pipeline.db_config)"
                )

    return violations


def check_claude_md_size_cap() -> list[str]:
    """Check #23: CLAUDE.md must stay under 12KB."""
    path = PROJECT_ROOT / "CLAUDE.md"
    if not path.exists():
        return [f"  CLAUDE.md not found at {path}"]
    size = path.stat().st_size
    if size > 12288:
        return [f"  CLAUDE.md exceeds 12KB size cap ({size / 1024:.1f}KB). Compact before committing."]
    return []


def check_discovery_session_aware_filters() -> list[str]:
    """Check #28: Discovery scripts must use get_filters_for_grid(), not iterate ALL_FILTERS.

    Grid-search files (strategy_discovery.py, nested/discovery.py, regime/discovery.py)
    must use the session-aware get_filters_for_grid() instead of iterating ALL_FILTERS
    directly. Iterating ALL_FILTERS applies DOW composites to sessions that lack
    research justification.

    Allowed: ALL_FILTERS lookups by key (e.g., ALL_FILTERS.get(filter_type))
    Forbidden: ALL_FILTERS.items() or len(ALL_FILTERS) in discovery grid loops
    """
    violations = []
    discovery_files = [
        TRADING_APP_DIR / "strategy_discovery.py",
        TRADING_APP_DIR / "nested" / "discovery.py",
        TRADING_APP_DIR / "regime" / "discovery.py",
    ]

    # Pattern: iterating or sizing ALL_FILTERS (grid misuse)
    iter_pattern = re.compile(r'\bALL_FILTERS\.(items|values|keys)\(\)')
    len_pattern = re.compile(r'\blen\(ALL_FILTERS\)')

    for fpath in discovery_files:
        if not fpath.exists():
            continue
        content = fpath.read_text(encoding='utf-8')
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            for pat, desc in [(iter_pattern, "iterates ALL_FILTERS"),
                              (len_pattern, "uses len(ALL_FILTERS)")]:
                if pat.search(line):
                    violations.append(
                        f"  {fpath.relative_to(PROJECT_ROOT)}:{line_num}: "
                        f"{desc} — use get_filters_for_grid() instead: "
                        f"{stripped[:80]}"
                    )

    return violations


def check_validated_filters_registered() -> list[str]:
    """Check #29: Every filter_type in validated_setups must exist in ALL_FILTERS.

    ALL_FILTERS is the single source of truth for filter definitions.
    No script should ever reconstruct filters from naming conventions.
    """
    violations = []
    try:
        from trading_app.config import ALL_FILTERS
    except ImportError as e:
        violations.append(f"  Cannot import ALL_FILTERS: {e}")
        return violations

    try:
        import duckdb
        from pipeline.paths import GOLD_DB_PATH

        con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
        rows = con.execute(
            "SELECT DISTINCT filter_type FROM validated_setups ORDER BY filter_type"
        ).fetchall()
        con.close()

        db_filter_types = {r[0] for r in rows}
        missing = db_filter_types - set(ALL_FILTERS.keys())
        if missing:
            for ft in sorted(missing):
                violations.append(
                    f"  filter_type '{ft}' in validated_setups but NOT in ALL_FILTERS"
                )
    except Exception:
        # DB may not exist in CI — skip gracefully
        pass

    return violations


def check_e0_cb1_only() -> list[str]:
    """Check #30: E0 entry model must be restricted to CB1 only.

    E0 (limit-on-confirm) fills on the confirm bar itself. For CB2+, the confirm
    bar has already closed by the time you know it's the confirm bar — filling on
    it is look-ahead. outcome_builder.py must restrict E0 to cb_options=[1].
    Discovery scripts must skip E0+CB2+ in the grid iteration.
    """
    violations = []
    ob_file = TRADING_APP_DIR / "outcome_builder.py"
    if not ob_file.exists():
        return violations

    content = ob_file.read_text(encoding="utf-8")

    # Verify E0 is in the same branch as E3 for cb_options = [1]
    # Expected pattern: em in ("E0", "E3") or em in {"E0", "E3"}
    if 'em in ("E0", "E3")' not in content and 'em in {"E0", "E3"}' not in content:
        # Also accept reversed order
        if 'em in ("E3", "E0")' not in content and 'em in {"E3", "E0"}' not in content:
            violations.append(
                "  outcome_builder.py: E0 must be restricted to cb_options=[1] "
                "(same as E3). E0+CB2+ is look-ahead bias."
            )

    # Also verify discovery grid scripts skip E0+CB2+
    discovery_files = [
        TRADING_APP_DIR / "strategy_discovery.py",
        TRADING_APP_DIR / "nested" / "discovery.py",
        TRADING_APP_DIR / "regime" / "discovery.py",
    ]
    e0_skip_patterns = [
        'em in ("E0", "E3") and cb > 1',
        'em in {"E0", "E3"} and cb > 1',
        'em in ("E3", "E0") and cb > 1',
        'em in {"E3", "E0"} and cb > 1',
    ]
    for f in discovery_files:
        if not f.exists():
            continue
        fc = f.read_text(encoding="utf-8")
        # Only check files that iterate confirm_bars (have a grid loop)
        if "CONFIRM_BARS" not in fc:
            continue
        if not any(p in fc for p in e0_skip_patterns):
            violations.append(
                f"  {f.relative_to(PROJECT_ROOT)}: Grid iteration must skip E0+CB2+ "
                "(em in ('E0', 'E3') and cb > 1: continue)"
            )

    return violations


def check_orb_minutes_in_strategy_id() -> list[str]:
    """Check #31: Non-5m strategies must have _O{minutes} suffix in strategy_id.

    make_strategy_id() appends _O15 or _O30 when orb_minutes != 5.
    This prevents PK collisions between 5m and 15m/30m strategies in
    experimental_strategies (strategy_id is PK).
    """
    violations = []
    discovery_file = TRADING_APP_DIR / "strategy_discovery.py"
    if not discovery_file.exists():
        return violations

    content = discovery_file.read_text(encoding="utf-8")

    # Verify make_strategy_id accepts orb_minutes parameter
    if "orb_minutes" not in content.split("def make_strategy_id")[1].split("def ")[0]:
        violations.append(
            "  strategy_discovery.py: make_strategy_id() must accept orb_minutes parameter"
        )

    # Verify the call site passes orb_minutes
    # Look for make_strategy_id call that includes orb_minutes=
    call_section = content.split("strategy_id = make_strategy_id(")
    if len(call_section) >= 2:
        # Check the main discovery loop call (not all calls need it — 5m callers use default)
        main_call = call_section[1].split(")")[0]
        if "orb_minutes=" not in main_call and "orb_minutes =" not in main_call:
            violations.append(
                "  strategy_discovery.py: main discovery call to make_strategy_id() "
                "must pass orb_minutes= to prevent PK collisions"
            )

    return violations


def check_orb_labels_session_catalog_sync() -> list[str]:
    """Check #32: ORB_LABELS must match SESSION_CATALOG dynamic entries.

    ORB_LABELS (init_db.py) drives schema generation (column names).
    SESSION_CATALOG (dst.py) drives time resolution (per-day resolvers).
    If a session exists in one but not the other:
      - In ORB_LABELS only: schema has columns but build_daily_features raises ValueError
      - In SESSION_CATALOG only: time resolved but no columns to store results (silent data loss)
    """
    violations = []

    init_db_file = PIPELINE_DIR / "init_db.py"
    dst_file = PIPELINE_DIR / "dst.py"

    if not init_db_file.exists() or not dst_file.exists():
        return violations

    # Import both sources
    try:
        from pipeline.init_db import ORB_LABELS
        from pipeline.dst import SESSION_CATALOG
    except ImportError as e:
        violations.append(f"  Cannot import for sync check: {e}")
        return violations

    orb_set = set(ORB_LABELS)
    catalog_dynamic = {k for k, v in SESSION_CATALOG.items() if v.get("type") == "dynamic"}

    in_orb_only = orb_set - catalog_dynamic
    in_catalog_only = catalog_dynamic - orb_set

    if in_orb_only:
        violations.append(
            f"  ORB_LABELS has sessions not in SESSION_CATALOG: {sorted(in_orb_only)}"
        )
    if in_catalog_only:
        violations.append(
            f"  SESSION_CATALOG has dynamic sessions not in ORB_LABELS: {sorted(in_catalog_only)}"
        )

    return violations


def check_stale_session_names_in_code() -> list[str]:
    """Check #33: No old fixed-clock session names in Python source code.

    Old names (0900, 1000, 1100, 1130, 1800, 2300, 0030) were replaced with
    event-based names (CME_REOPEN, TOKYO_OPEN, etc.) in Feb 2026. Stale
    references in code (not comments) could cause silent bugs.
    Checks pipeline/ and trading_app/ Python files.
    """
    violations = []
    # Patterns that indicate old session names used as identifiers/strings
    # (not in comments — we check quoted strings and f-strings)
    old_names_pattern = re.compile(
        r"""['"](?:0900|1000|1100|1130|1800|2300|0030)['"]"""
    )
    # Directories to check
    dirs = [PIPELINE_DIR, TRADING_APP_DIR]
    # Files to skip (historical references are OK in these)
    skip_files = {
        "check_drift.py",  # This file (contains the pattern itself)
    }
    for d in dirs:
        for py_file in sorted(d.rglob("*.py")):
            if py_file.name in skip_files:
                continue
            content = py_file.read_text(encoding="utf-8")
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.lstrip()
                # Skip comment-only lines
                if stripped.startswith("#"):
                    continue
                if old_names_pattern.search(line):
                    violations.append(
                        f"  {py_file.relative_to(PROJECT_ROOT)}:{i}: "
                        f"stale session name in code: {line.strip()[:80]}"
                    )
    return violations


def check_sql_adapter_validation_sync() -> list[str]:
    """Check #34: sql_adapter.py VALID_* sets must match outcome_builder.py grids.

    VALID_RR_TARGETS and VALID_CONFIRM_BARS in sql_adapter.py gate which
    queries the AI CLI accepts. If outcome_builder.py adds new grid values,
    sql_adapter.py must be updated or legitimate queries get rejected.
    """
    violations = []
    adapter_file = TRADING_APP_DIR / "ai" / "sql_adapter.py"
    builder_file = TRADING_APP_DIR / "outcome_builder.py"
    if not adapter_file.exists() or not builder_file.exists():
        return violations

    adapter_content = adapter_file.read_text(encoding="utf-8")
    builder_content = builder_file.read_text(encoding="utf-8")

    # Extract RR_TARGETS from outcome_builder.py
    rr_match = re.search(r'RR_TARGETS\s*=\s*\[([\d.,\s]+)\]', builder_content)
    cb_match = re.search(r'CONFIRM_BARS_OPTIONS\s*=\s*\[([\d,\s]+)\]', builder_content)

    if rr_match:
        builder_rr = {float(x.strip()) for x in rr_match.group(1).split(",") if x.strip()}
        adapter_rr_match = re.search(r'VALID_RR_TARGETS\s*=\s*\{([^}]+)\}', adapter_content)
        if adapter_rr_match:
            adapter_rr = {float(x.strip()) for x in adapter_rr_match.group(1).split(",") if x.strip()}
            missing = builder_rr - adapter_rr
            if missing:
                violations.append(
                    f"  sql_adapter.py VALID_RR_TARGETS missing: {sorted(missing)} "
                    f"(present in outcome_builder.py RR_TARGETS)"
                )

    if cb_match:
        builder_cb = {int(x.strip()) for x in cb_match.group(1).split(",") if x.strip()}
        adapter_cb_match = re.search(r'VALID_CONFIRM_BARS\s*=\s*\{([^}]+)\}', adapter_content)
        if adapter_cb_match:
            adapter_cb = {int(x.strip()) for x in adapter_cb_match.group(1).split(",") if x.strip()}
            missing = builder_cb - adapter_cb
            if missing:
                violations.append(
                    f"  sql_adapter.py VALID_CONFIRM_BARS missing: {sorted(missing)} "
                    f"(present in outcome_builder.py CONFIRM_BARS_OPTIONS)"
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

    # Check 20: MarketState/scoring/cascade read-only guard
    print("Check 20: MarketState read-only SQL guard...")
    v = check_market_state_readonly()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 21: Analytical honesty guard (sharpe_ann presence)
    print("Check 21: Analytical honesty guard (sharpe_ann)...")
    v = check_sharpe_ann_presence()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 22: Ingest authority notice (deprecation in ingest_dbn_mgc.py)
    print("Check 22: Ingest authority notice (ingest_dbn_mgc.py deprecation)...")
    v = check_ingest_authority_notice()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 23: CLAUDE.md size cap
    print("Check 23: CLAUDE.md size cap...")
    v = check_claude_md_size_cap()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 24: Validation gate existence
    print("Check 24: Validation gate existence...")
    v = check_validation_gate_existence()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 25: Naive datetime detection
    print("Check 25: Naive datetime detection...")
    v = check_naive_datetime()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 26: DST session coverage
    print("Check 26: DST session coverage (all sessions classified)...")
    v = check_dst_session_coverage()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 27: DB config usage (all connect() calls must configure)
    print("Check 27: DB config usage (configure_connection after connect)...")
    v = check_db_config_usage()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 28: Discovery files must use session-aware filters
    print("Check 28: Discovery scripts use get_filters_for_grid (not ALL_FILTERS)...")
    v = check_discovery_session_aware_filters()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 29: All filter_types in validated_setups must be in ALL_FILTERS
    print("Check 29: All validated filter_types registered in ALL_FILTERS...")
    v = check_validated_filters_registered()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 30: E0 entry model restricted to CB1 only
    print("Check 30: E0 restricted to CB1 (no look-ahead CB2+ fills)...")
    v = check_e0_cb1_only()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 31: Non-5m strategy IDs include orb_minutes suffix
    print("Check 31: Non-5m strategy IDs include _O{minutes} suffix...")
    v = check_orb_minutes_in_strategy_id()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 32: ORB_LABELS vs SESSION_CATALOG sync
    print("Check 32: ORB_LABELS matches SESSION_CATALOG dynamic entries...")
    v = check_orb_labels_session_catalog_sync()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 33: Stale session names in code
    print("Check 33: No old fixed-clock session names in Python source...")
    v = check_stale_session_names_in_code()
    if v:
        print("  FAILED:")
        for line in v:
            print(line)
        all_violations.extend(v)
    else:
        print("  PASSED [OK]")
    print()

    # Check 34: sql_adapter validation set sync
    print("Check 34: sql_adapter VALID_* sets match outcome_builder grids...")
    v = check_sql_adapter_validation_sync()
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
