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
from datetime import UTC
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
TRADING_APP_DIR = PROJECT_ROOT / "trading_app"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESEARCH_DIR = PROJECT_ROOT / "research"

# Module-level override for tests; production uses GOLD_DB_PATH from pipeline.paths
GOLD_DB_PATH_FOR_CHECKS = None  # Set by tests; production uses GOLD_DB_PATH


def _import_duckdb_or_exit():
    """Import duckdb with a fail-closed environment hint."""
    try:
        import duckdb  # type: ignore
    except ModuleNotFoundError:
        print("FATAL: duckdb is not installed for the current interpreter.", file=sys.stderr)
        print(f"Interpreter: {Path(sys.executable).resolve()}", file=sys.stderr)
        venv_python = PROJECT_ROOT / ".venv-wsl" / "bin" / "python"
        if venv_python.exists():
            print("Use the repo-managed WSL environment instead:", file=sys.stderr)
            print(f"  {venv_python} pipeline/check_drift.py", file=sys.stderr)
            print("  uv run python pipeline/check_drift.py", file=sys.stderr)
            print("  scripts/infra/codex-project.sh", file=sys.stderr)
        else:
            print(
                "Create the WSL environment first with: "
                "UV_PROJECT_ENVIRONMENT=.venv-wsl uv sync --frozen --python 3.13 --group dev",
                file=sys.stderr,
            )
        sys.exit(2)
    return duckdb


def _get_db_path() -> Path:
    """Resolve DB path: test override > pipeline.paths default."""
    if GOLD_DB_PATH_FOR_CHECKS is not None:
        return Path(GOLD_DB_PATH_FOR_CHECKS)
    from pipeline.paths import GOLD_DB_PATH

    return GOLD_DB_PATH


def _table_exists(con, table_name: str) -> bool:
    """Return True if the table exists in the connected DuckDB database."""
    row = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_name = ?
        LIMIT 1
        """,
        [table_name],
    ).fetchone()
    return row is not None


def _missing_table_columns(con, table_name: str, required: list[str]) -> list[str]:
    """Return sorted required columns missing from a table."""
    rows = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = ?
        """,
        [table_name],
    ).fetchall()
    present = {row[0] for row in rows}
    return sorted(col for col in required if col not in present)


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

        content = fpath.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            # Skip comments and docstrings
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for pattern in sql_mgc_patterns:
                if pattern.search(line):
                    violations.append(f"  {fpath.name}:{line_num}: Hardcoded 'MGC' in SQL: {stripped[:80]}")

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

        content = fpath.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Check .iterrows() — always forbidden in ingest hot path
            # Exception: the per-day row accumulation loop (front_df.iterrows())
            # This is on already-filtered single-contract single-day data, not bulk
            if ".iterrows()" in line:
                # Allow the known pattern: buffering single-day front-contract rows
                if "front_df.iterrows()" in line:
                    continue
                violations.append(f"  {fpath.name}:{line_num}: .iterrows() usage: {stripped[:80]}")

            # Check .apply() — forbidden except on symbol column
            if ".apply(" in line:
                if allowed_apply_pattern.search(line):
                    continue
                # Also allow lambda in trading_days mask (compute_trading_days)
                if "trading_days[mask].apply(" in line or "trading_days[mask]" in stripped:
                    continue
                violations.append(f"  {fpath.name}:{line_num}: .apply() usage: {stripped[:80]}")

    return violations


def check_non_bars1m_writes(files: list[Path]) -> list[str]:
    """Check that ingest scripts only write to bars_1m."""
    violations = []

    # Patterns for SQL writes to tables other than bars_1m
    write_patterns = [
        re.compile(r"INSERT\s+(?:OR\s+REPLACE\s+)?INTO\s+(?!bars_1m\b)(\w+)", re.IGNORECASE),
        re.compile(r"DELETE\s+FROM\s+(?!bars_1m\b)(\w+)", re.IGNORECASE),
        re.compile(r"UPDATE\s+(?!bars_1m\b)(\w+)", re.IGNORECASE),
        re.compile(r"DROP\s+TABLE\s+(?!bars_1m\b)(\w+)", re.IGNORECASE),
    ]

    for fpath in files:
        if not fpath.exists():
            continue

        content = fpath.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
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
            sf_content = sf.read_text(encoding="utf-8")
            create_tables.update(
                re.findall(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", sf_content, re.IGNORECASE)
            )

    if not create_tables:
        return violations

    # SQL keywords and known non-table identifiers to skip
    sql_keywords = {
        "SELECT",
        "WHERE",
        "AND",
        "OR",
        "NOT",
        "NULL",
        "SET",
        "VALUES",
        "ORDER",
        "GROUP",
        "BY",
        "AS",
        "HAVING",
        "LIMIT",
        "OFFSET",
        "DISTINCT",
        "ON",
        "TRANSACTION",
        "TABLE",
        "REPLACE",
        "EXISTS",
        "SCHEMA",
        "COLUMNS",
        "INFORMATION_SCHEMA",
        "MAIN",
        "INDEX",
        "VIEW",
        "TYPE",
        "INTERVAL",
        "ZONE",
        "CAST",
        "COUNT",
        "MIN",
        "MAX",
        "SUM",
        "AVG",
        "FIRST",
        "LAST",
        "EXTRACT",
        "EPOCH",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "DROP",
        "CREATE",
        "IF",
    }

    # Extract CTE names: WITH x AS (...), y AS (...), z AS (...)
    # Matches both the first CTE (after WITH) and subsequent ones (after comma)
    cte_pattern = re.compile(r"(?:WITH|,)\s+(\w+)\s+AS\s*\(", re.IGNORECASE)

    # Pattern for table references in SQL
    table_ref_pattern = re.compile(r"(?:FROM|INTO|UPDATE|JOIN)\s+(\w+)", re.IGNORECASE)

    # Extract triple-quoted strings that contain SQL
    sql_string_pattern = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.DOTALL)
    sql_indicator = re.compile(r"\b(SELECT|INSERT|DELETE|UPDATE|CREATE)\b", re.IGNORECASE)

    for fpath in pipeline_dir.glob("*.py"):
        if fpath.name in ("init_db.py", "check_drift.py"):
            continue

        content = fpath.read_text(encoding="utf-8")

        for match in sql_string_pattern.finditer(content):
            sql_text = match.group(1) or match.group(2)

            # Only check strings that look like SQL
            if not sql_indicator.search(sql_text):
                continue

            # Extract CTE names from this SQL block
            cte_names = {m.group(1) for m in cte_pattern.finditer(sql_text)}

            # Find line number of the match start
            line_num = content[: match.start()].count("\n") + 1

            for ref_match in table_ref_pattern.finditer(sql_text):
                table = ref_match.group(1)
                if table.upper() in sql_keywords:
                    continue
                if "information_schema" in sql_text.lower():
                    continue
                # Skip CTE names and known DataFrame variable patterns
                if table in cte_names:
                    continue
                if table.endswith("_df") or table.endswith("_frame"):
                    continue
                # Skip column names used after EXTRACT(... FROM col)
                extract_ctx = sql_text[max(0, ref_match.start() - 30) : ref_match.start()]
                if "EXTRACT" in extract_ctx.upper() or "EPOCH" in extract_ctx.upper():
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

    content = mgc_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "from pipeline.ingest_dbn import" in line or "import pipeline.ingest_dbn" in line:
            violations.append(f"  ingest_dbn_mgc.py:{line_num}: Imports from ingest_dbn.py (circular dependency risk)")
        if "from .ingest_dbn import" in line or "from . import ingest_dbn" in line:
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

        content = fpath.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for pattern in path_patterns:
                if pattern.search(line):
                    violations.append(f"  {fpath.name}:{line_num}: Hardcoded absolute path: {stripped[:80]}")

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

        content = fpath.read_text(encoding="utf-8")

        connect_count = len(re.findall(r"duckdb\.connect\(", content))
        if connect_count == 0:
            continue

        close_count = len(re.findall(r"\.close\(\)", content))
        has_finally = "finally:" in content
        has_atexit = "atexit" in content
        # Also recognise wrapper helpers that return duckdb.connect() and are
        # called via ``with _connect(...) as con:``
        has_with = "with duckdb" in content or (
            bool(re.search(r"return\s+duckdb\.connect\(", content)) and bool(re.search(r"\bwith\s+\w+\(", content))
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
        re.compile(r"from\s+trading_app"),
        re.compile(r"import\s+trading_app"),
    ]

    for fpath in pipeline_dir.glob("*.py"):
        if fpath.name == "check_drift.py":
            continue

        content = fpath.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
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

        content = fpath.read_text(encoding="utf-8")

        connect_count = len(re.findall(r"duckdb\.connect\(", content))
        if connect_count == 0:
            continue

        close_count = len(re.findall(r"\.close\(\)", content))
        has_finally = "finally:" in content
        has_atexit = "atexit" in content
        # Also recognise wrapper helpers that return duckdb.connect() and are
        # called via ``with _connect(...) as con:``
        has_with = "with duckdb" in content or (
            bool(re.search(r"return\s+duckdb\.connect\(", content)) and bool(re.search(r"\bwith\s+\w+\(", content))
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
        content = fpath.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for pattern in path_patterns:
                if pattern.search(line):
                    violations.append(f"  {fpath.name}:{line_num}: Hardcoded absolute path: {stripped[:80]}")

    return violations


def check_dashboard_readonly(pipeline_dir: Path) -> list[str]:
    """Check that dashboard.py only reads from the database (no writes)."""
    violations = []

    dash_path = pipeline_dir / "dashboard.py"
    if not dash_path.exists():
        return violations

    content = dash_path.read_text(encoding="utf-8")

    write_patterns = [
        re.compile(r"INSERT\s+", re.IGNORECASE),
        re.compile(r"DELETE\s+FROM", re.IGNORECASE),
        re.compile(r"UPDATE\s+\w+\s+SET", re.IGNORECASE),
        re.compile(r"DROP\s+TABLE", re.IGNORECASE),
        re.compile(r"CREATE\s+TABLE", re.IGNORECASE),
    ]

    lines = content.splitlines()
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pattern in write_patterns:
            if pattern.search(line):
                violations.append(f"  dashboard.py:{line_num}: DB write detected (must be read-only): {stripped[:80]}")

    # Also verify read_only=True is used
    if "duckdb.connect(" in content and "read_only=True" not in content:
        violations.append("  dashboard.py: duckdb.connect() without read_only=True")

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
                violations.append(f"  ALL_FILTERS['{key}'].filter_type = '{filt.filter_type}' (mismatch)")
    except ImportError as e:
        violations.append(f"  Cannot import trading_app.config.ALL_FILTERS: {e}")

    return violations


def check_entry_models_sync() -> list[str]:
    """Check that ENTRY_MODELS in config is consistent with VALID_ENTRY_MODELS in sql_adapter.

    Cross-system sync: trading_app.config.ENTRY_MODELS is the canonical definition;
    trading_app.ai.sql_adapter.VALID_ENTRY_MODELS gates which entry models the AI CLI
    accepts. If they diverge, legitimate queries get rejected or invalid models get queried.
    """
    violations = []

    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from trading_app.ai.sql_adapter import VALID_ENTRY_MODELS
        from trading_app.config import ENTRY_MODELS

        config_set = set(ENTRY_MODELS)
        if config_set != VALID_ENTRY_MODELS:
            diff = config_set.symmetric_difference(VALID_ENTRY_MODELS)
            violations.append(
                f"  sql_adapter.VALID_ENTRY_MODELS {sorted(VALID_ENTRY_MODELS)} "
                f"!= config.ENTRY_MODELS {sorted(config_set)} (diff: {sorted(diff)})"
            )
    except ImportError as e:
        violations.append(f"  Cannot import ENTRY_MODELS or VALID_ENTRY_MODELS: {e}")

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
        re.compile(r"from\s+trading_app\.db_manager\s+import"),
        re.compile(r"import\s+trading_app\.db_manager"),
    ]

    for subdir_name in ["nested", "regime"]:
        subdir = TRADING_APP_DIR / subdir_name
        if not subdir.exists():
            continue

        for fpath in subdir.glob("*.py"):
            if fpath.name == "__init__.py":
                continue

            content = fpath.read_text(encoding="utf-8")
            lines = content.splitlines()

            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
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

    content = outcome_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Pattern: entry_price = orb_high or entry_price = orb_low
    # This should only appear inside E3 logic in entry_rules.py, never in outcome_builder
    dangerous_patterns = [
        re.compile(r"entry_price\s*=\s*orb_high"),
        re.compile(r"entry_price\s*=\s*orb_low"),
    ]

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
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

            content = fpath.read_text(encoding="utf-8")
            lines = content.splitlines()

            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                for table in production_tables:
                    for keyword in write_keywords:
                        pattern = re.compile(
                            rf"{keyword}\s+(?:OR\s+REPLACE\s+)?(?:INTO\s+|FROM\s+)?{table}\b",
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
            sf_content = sf.read_text(encoding="utf-8")
            create_tables.update(
                re.findall(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", sf_content, re.IGNORECASE)
            )

    if not create_tables:
        return violations

    sql_keywords = {
        "SELECT",
        "WHERE",
        "AND",
        "OR",
        "NOT",
        "NULL",
        "SET",
        "VALUES",
        "ORDER",
        "GROUP",
        "BY",
        "AS",
        "HAVING",
        "LIMIT",
        "OFFSET",
        "DISTINCT",
        "ON",
        "TRANSACTION",
        "TABLE",
        "REPLACE",
        "EXISTS",
        "SCHEMA",
        "COLUMNS",
        "INFORMATION_SCHEMA",
        "MAIN",
        "INDEX",
        "VIEW",
        "TYPE",
        "INTERVAL",
        "ZONE",
        "CAST",
        "COUNT",
        "MIN",
        "MAX",
        "SUM",
        "AVG",
        "FIRST",
        "LAST",
        "EXTRACT",
        "EPOCH",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "DROP",
        "CREATE",
        "IF",
        "TEXT",
        "INTEGER",
        "DOUBLE",
        "BOOLEAN",
        "DATE",
        "TIMESTAMP",
        "TIMESTAMPTZ",
        "VARCHAR",
        "BIGINT",
        "REAL",
        "FLOAT",
        "FROM",
        "IS",
        "IN",
        "INTO",
        "DELETE",
        "INSERT",
        "JOIN",
        "INNER",
        "LEFT",
        "RIGHT",
        "OUTER",
        "UNION",
        "ALL",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "LIKE",
        "BETWEEN",
        "TRUE",
        "FALSE",
        "DEFAULT",
        "PRIMARY",
        "KEY",
        "FOREIGN",
        "REFERENCES",
        "CASCADE",
        "CHECK",
        "UNIQUE",
        "ALTER",
        "ADD",
        "COLUMN",
        "WITH",
    }

    cte_pattern = re.compile(r"(?:WITH|,)\s+(\w+)\s+AS\s*\(", re.IGNORECASE)
    table_ref_pattern = re.compile(r"(?:FROM|INTO|UPDATE|JOIN)\s+(\w+)", re.IGNORECASE)
    sql_string_pattern = re.compile(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', re.DOTALL)
    sql_indicator = re.compile(r"\b(SELECT|INSERT|DELETE|UPDATE|CREATE)\b", re.IGNORECASE)

    if not trading_app_dir.exists():
        return violations

    for fpath in trading_app_dir.rglob("*.py"):
        if fpath.name in ("__init__.py", "schema.py", "db_manager.py"):
            continue

        content = fpath.read_text(encoding="utf-8")

        for match in sql_string_pattern.finditer(content):
            sql_text = match.group(1) or match.group(2)

            # Require at least 2 SQL keywords to distinguish real SQL from docstrings
            # (docstrings can contain "Update", "Create" etc. as English words)
            sql_hits = sql_indicator.findall(sql_text)
            if len(sql_hits) < 2:
                continue

            cte_names = {m.group(1) for m in cte_pattern.finditer(sql_text)}
            line_num = content[: match.start()].count("\n") + 1

            for ref_match in table_ref_pattern.finditer(sql_text):
                table = ref_match.group(1)
                if table.upper() in sql_keywords:
                    continue
                if "information_schema" in sql_text.lower():
                    continue
                if table in cte_names:
                    continue
                if table.endswith("_df") or table.endswith("_frame"):
                    continue
                extract_ctx = sql_text[max(0, ref_match.start() - 30) : ref_match.start()]
                if "EXTRACT" in extract_ctx.upper() or "EPOCH" in extract_ctx.upper():
                    continue
                if table not in create_tables:
                    violations.append(f"  {fpath.name}:~{line_num}: SQL references table '{table}' not in schema")

    return violations


def check_timezone_hygiene() -> list[str]:
    """Block pytz imports and hardcoded timedelta(hours=10) patterns.

    Known footguns:
      - pytz has surprising DST normalization behavior
      - timedelta(hours=10) is a hardcoded Brisbane offset instead of using zoneinfo
      - pd.Timedelta(hours=10) added to DuckDB timestamps double-converts
        (DuckDB fetchdf() returns Brisbane-localized timestamps, not naive UTC)
    """
    violations = []

    pytz_pattern = re.compile(r"import\s+pytz|from\s+pytz\s+import")
    # Match timedelta(hours=10) with optional spaces — both stdlib and pandas
    hardcoded_tz_pattern = re.compile(r"timedelta\s*\(\s*hours\s*=\s*10\s*\)")
    # Match pd.Timedelta(hours=10) — the double-conversion footgun in research scripts
    pd_timedelta_pattern = re.compile(r"pd\.Timedelta\s*\(\s*hours\s*=\s*10\s*\)")

    for base_dir in [PIPELINE_DIR, TRADING_APP_DIR, RESEARCH_DIR]:
        if not base_dir.exists():
            continue

        for fpath in base_dir.rglob("*.py"):
            if fpath.name in ("__init__.py", "check_drift.py"):
                continue

            content = fpath.read_text(encoding="utf-8")
            lines = content.splitlines()

            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue

                if pytz_pattern.search(line):
                    violations.append(f"  {fpath.name}:{line_num}: pytz import (use zoneinfo instead): {stripped[:80]}")
                if hardcoded_tz_pattern.search(line):
                    violations.append(
                        f"  {fpath.name}:{line_num}: Hardcoded timedelta(hours=10) "
                        f"(use timezone.utc or zoneinfo): {stripped[:80]}"
                    )
                if pd_timedelta_pattern.search(line):
                    violations.append(
                        f"  {fpath.name}:{line_num}: pd.Timedelta(hours=10) "
                        f"double-converts DuckDB Brisbane timestamps: {stripped[:80]}"
                    )

    return violations


# Modules with optional dependencies (feature-gated, not broken)
_OPTIONAL_DEP_MODULES = {
    "trading_app.live.data_feed",  # websockets
    "trading_app.live.session_orchestrator",  # websockets
    "trading_app.live.webhook_server",  # fastapi
    "trading_app.ml.cpcv",  # sklearn
    "trading_app.ml.evaluate",  # joblib
    "trading_app.ml.evaluate_validated",  # joblib
    "trading_app.ml.importance",  # sklearn
    "trading_app.ml.meta_label",  # joblib
    "trading_app.ml.predict_live",  # joblib
}


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

            # Skip modules already loaded — their imports resolved successfully
            if module in sys.modules:
                continue
            # Skip modules with optional dependencies (feature-gated, not broken)
            if module in _OPTIONAL_DEP_MODULES:
                continue
            try:
                importlib.import_module(module)
            except Exception as e:
                err_type = type(e).__name__
                violations.append(f"  {module}: {err_type}: {str(e)[:100]}")

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

        content = fpath.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # Skip string-only lines (docstrings, comments in strings)
            if stripped.startswith(('"""', "'''", '"', "'")):
                continue

            for keyword in write_keywords:
                # Match SQL write keywords followed by whitespace (avoids
                # matching variable names like 'update_signals')
                pattern = re.compile(
                    rf"\b{keyword}\s+(?:OR\s+REPLACE\s+)?(?:INTO|FROM|TABLE|INDEX)\b",
                    re.IGNORECASE,
                )
                if pattern.search(line):
                    violations.append(f"  {fpath.name}:{line_num}: SQL write keyword '{keyword}': {stripped[:80]}")

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
        content = discovery_path.read_text(encoding="utf-8")
        if '"sharpe_ann"' not in content:
            violations.append("  strategy_discovery.py: Missing 'sharpe_ann' in compute_metrics return dict")
        if '"trades_per_year"' not in content:
            violations.append("  strategy_discovery.py: Missing 'trades_per_year' in compute_metrics return dict")

    # Check 2: view_strategies.py must show sharpe_ann in user-facing output
    view_path = TRADING_APP_DIR / "view_strategies.py"
    if view_path.exists():
        content = view_path.read_text(encoding="utf-8")
        if "sharpe_ann" not in content:
            violations.append("  view_strategies.py: Missing 'sharpe_ann' in user-facing output")

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
        (
            'print("NOTE: For multi-instrument support, prefer:")',
            'logger.info("NOTE: For multi-instrument support, prefer:")',
        ),
        (
            'print("  python pipeline/ingest_dbn.py --instrument MGC")',
            'logger.info("  python pipeline/ingest_dbn.py --instrument MGC")',
        ),
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

        content = fpath.read_text(encoding="utf-8")
        for gate in required_strings:
            if gate not in content:
                violations.append(f"  {fpath.name}: Missing gate '{gate}'")

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
        (re.compile(r"datetime\.utcnow\(\)"), "datetime.utcnow()"),
        (re.compile(r"datetime\.utcfromtimestamp\("), "datetime.utcfromtimestamp()"),
    ]

    for base_dir in [PIPELINE_DIR, TRADING_APP_DIR]:
        if not base_dir.exists():
            continue

        for fpath in base_dir.rglob("*.py"):
            if fpath.name in ("__init__.py", "check_drift.py"):
                continue

            content = fpath.read_text(encoding="utf-8")
            lines = content.splitlines()

            for line_num, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
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
        from pipeline.dst import DST_AFFECTED_SESSIONS, DST_CLEAN_SESSIONS, SESSION_CATALOG

        non_alias = {label for label, entry in SESSION_CATALOG.items() if entry["type"] != "alias"}
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
            violations.append(f"  Sessions in DST sets but not in SESSION_CATALOG: {sorted(stale)}")

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
        "check_drift.py",
        "db_config.py",
        "check_db.py",
        "init_db.py",
        "dashboard.py",
        "health_check.py",
        "__init__.py",
        # Deprecated
        "ingest_dbn_mgc.py",
        # Pipeline utilities (read-only)
        "audit_bars_coverage.py",
        "export_parquet.py",
        # Trading app read-only consumers
        "cascade_table.py",
        "db_manager.py",
        "live_config.py",
        "market_state.py",
        "paper_trader.py",
        "portfolio.py",
        "rolling_correlation.py",
        "rolling_portfolio.py",
        "strategy_fitness.py",
        "view_strategies.py",
        "dsr.py",  # read-only DSR estimation utilities
        # validate_1800_*.py moved to research/archive/ (I12 audit cleanup)
        # Nested subpackages
        "corpus.py",
        "sql_adapter.py",
        "strategy_matcher.py",
        "asia_session_analyzer.py",
        "audit_outcomes.py",
        "builder.py",
        "compare.py",
        "discovery.py",
        "schema.py",
        "validator.py",
    }

    for base_dir in [PIPELINE_DIR, TRADING_APP_DIR]:
        if not base_dir.exists():
            continue

        for fpath in base_dir.rglob("*.py"):
            if fpath.name in EXEMPT:
                continue

            content = fpath.read_text(encoding="utf-8")

            if "duckdb.connect(" not in content:
                continue

            if "configure_connection" not in content:
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
    iter_pattern = re.compile(r"\bALL_FILTERS\.(items|values|keys)\(\)")
    len_pattern = re.compile(r"\blen\(ALL_FILTERS\)")

    for fpath in discovery_files:
        if not fpath.exists():
            continue
        content = fpath.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pat, desc in [(iter_pattern, "iterates ALL_FILTERS"), (len_pattern, "uses len(ALL_FILTERS)")]:
                if pat.search(line):
                    violations.append(
                        f"  {fpath.relative_to(PROJECT_ROOT)}:{line_num}: "
                        f"{desc} — use get_filters_for_grid() instead: "
                        f"{stripped[:80]}"
                    )

    return violations


def check_validated_filters_registered(con=None) -> list[str]:
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

    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        rows = con.execute("SELECT DISTINCT filter_type FROM validated_setups ORDER BY filter_type").fetchall()

        db_filter_types = {r[0] for r in rows}
        missing = db_filter_types - set(ALL_FILTERS.keys())
        if missing:
            for ft in sorted(missing):
                violations.append(f"  filter_type '{ft}' in validated_setups but NOT in ALL_FILTERS")
    except (ImportError, OSError) as e:
        print(f"    SKIP check_validated_filters_registered: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()

    return violations


def check_e2_e3_cb1_only() -> list[str]:
    """Check #30: E2 and E3 entry models must be restricted to CB1 only.

    E2 (stop-market) has no confirm bars concept — always CB1.
    E3 (limit-at-ORB) always uses CB1 (higher CBs produce identical outcomes).
    outcome_builder.py must restrict E2/E3 to cb_options=[1].
    Discovery scripts must skip E2+CB2+ and E3+CB2+ in the grid iteration.
    """
    violations = []
    ob_file = TRADING_APP_DIR / "outcome_builder.py"
    if not ob_file.exists():
        return violations

    content = ob_file.read_text(encoding="utf-8")

    # Verify E3 has cb_options = [1] in outcome_builder
    if 'em == "E3"' not in content:
        violations.append("  outcome_builder.py: E3 must be restricted to cb_options=[1].")

    # Also verify discovery grid scripts skip E2+CB2+ and E3+CB2+
    discovery_files = [
        TRADING_APP_DIR / "strategy_discovery.py",
        TRADING_APP_DIR / "nested" / "discovery.py",
        TRADING_APP_DIR / "regime" / "discovery.py",
    ]
    e2_skip_patterns = [
        'em in ("E2", "E3") and cb > 1',
        'em in {"E2", "E3"} and cb > 1',
        'em in ("E3", "E2") and cb > 1',
        'em in {"E3", "E2"} and cb > 1',
    ]
    for f in discovery_files:
        if not f.exists():
            continue
        fc = f.read_text(encoding="utf-8")
        # Only check files that iterate confirm_bars (have a grid loop)
        if "CONFIRM_BARS" not in fc:
            continue
        if not any(p in fc for p in e2_skip_patterns):
            violations.append(
                f"  {f.relative_to(PROJECT_ROOT)}: Grid iteration must skip E2+CB2+ "
                "(em in ('E2', 'E3') and cb > 1: continue)"
            )

    return violations


def check_no_e0_in_db(con=None) -> list[str]:
    """Check #35: No E0 rows should exist in any trading table.

    E0 (limit-on-confirm) was purged Feb 2026. Replaced by E2 (stop-market).
    NOTE: This check will fail until Phase C DB purge is complete. Commented
    out in the main() runner until then.
    """
    violations = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True
        for table in ["orb_outcomes", "experimental_strategies", "validated_setups"]:
            count = con.execute(f"SELECT COUNT(*) FROM {table} WHERE entry_model = 'E0'").fetchone()[0]
            if count > 0:
                violations.append(f"  {table}: {count} rows with entry_model='E0' (purged Feb 2026)")
    except (ImportError, OSError) as e:
        print(f"    SKIP check_no_e0_in_db: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_doc_stats_consistency(con=None) -> list[str]:
    """Check #36: Doc files must match ground-truth DB counts.

    Parses key stats (validated active, FDR significant, edge families,
    drift check count) from documentation and compares to gold.db.
    Prevents narrative-math divergence after rebuilds.
    """
    violations = []

    # Ground-truth from DB
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations  # Skip if no DB (CI)
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True
        validated_active = con.execute("SELECT COUNT(*) FROM validated_setups WHERE status = 'active'").fetchone()[0]
        fdr_significant = con.execute(
            "SELECT COUNT(*) FROM validated_setups WHERE status = 'active' AND fdr_significant"
        ).fetchone()[0]
        edge_families_count = con.execute("SELECT COUNT(*) FROM edge_families").fetchone()[0]
        # Per-aperture validated counts
        aperture_rows = con.execute(
            "SELECT orb_minutes, COUNT(*) FROM validated_setups WHERE status = 'active' GROUP BY orb_minutes"
        ).fetchall()
        aperture_counts = {int(r[0]): r[1] for r in aperture_rows}
        # Edge family robustness tiers
        tier_rows = con.execute(
            "SELECT robustness_status, COUNT(*) FROM edge_families "
            "WHERE robustness_status IN ('ROBUST','WHITELISTED') "
            "GROUP BY robustness_status"
        ).fetchall()
        tier_counts = {r[0]: r[1] for r in tier_rows}
    except (ImportError, OSError) as e:
        print(f"    SKIP check_doc_stats_consistency: {type(e).__name__}: {e}")
        return violations
    finally:
        if _own_con and con is not None:
            con.close()

    # Count drift checks from the CHECKS registry (single source of truth)
    drift_check_count = len(CHECKS)

    # Doc files to check: (file, regex, expected_key)
    DOC_STATS_CHECKS = [
        # TRADING_RULES.md no longer hardcodes aggregate counts — query DB via MCP.
        # Only guard against someone re-adding a "Post-rebuild results" line with counts.
        # Guard fires only if someone re-adds a hardcoded count to these files
        ("ROADMAP.md", r"([\d,]+)\s+validated\s+active", "validated_active"),
        ("CLAUDE.md", r"(\d+)\s+(?:drift|static)\s+checks", "drift_check_count"),
    ]

    ground_truth = {
        "validated_active": validated_active,
        "fdr_significant": fdr_significant,
        "edge_families": edge_families_count,
        "drift_check_count": drift_check_count,
        "aperture_5m": aperture_counts.get(5, 0),
        "aperture_15m": aperture_counts.get(15, 0),
        "aperture_30m": aperture_counts.get(30, 0),
        "tier_robust": tier_counts.get("ROBUST", 0),
        "tier_whitelisted": tier_counts.get("WHITELISTED", 0),
    }

    for filename, pattern, key in DOC_STATS_CHECKS:
        filepath = PROJECT_ROOT / filename
        if not filepath.exists():
            continue
        content = filepath.read_text(encoding="utf-8")
        matches = re.findall(pattern, content)
        if not matches:
            continue  # No match — don't flag (doc may not mention this stat)
        # Check first match only (headline/summary number).  Docs also
        # mention per-session counts (e.g. "4 validated strategies") which
        # are correct but not the total — checking all matches would
        # false-positive on those.
        doc_value = int(matches[0].replace(",", ""))
        expected = ground_truth[key]
        if doc_value != expected:
            violations.append(f"  {filename}: says {doc_value} {key} but DB has {expected}")

    return violations


def check_stale_scratch_db() -> list[str]:
    """Check #37: Canonical gold.db must exist at project root.

    Canonical DB is <project>/gold.db (via pipeline.paths.GOLD_DB_PATH).
    C:/db/gold.db scratch copy is DEPRECATED (Mar 24 2026) — caused stale-data
    decisions across terminals. No auto-sync. Use canonical only.
    """
    violations = []
    project_root_db = PROJECT_ROOT / "gold.db"
    if not project_root_db.exists():
        violations.append(f"  Canonical DB not found at project root ({project_root_db}). Run pipeline to create it.")
        return violations
    scratch_db = Path("C:/db/gold.db")
    if scratch_db.exists():
        violations.append(
            "  DEPRECATED scratch DB still exists at C:/db/gold.db. "
            "Delete it to prevent stale-data bugs: del C:\\db\\gold.db"
        )
    return violations


def check_old_session_names() -> list[str]:
    """Check #38: No old fixed-clock session names in active code.

    Old names (0900, 1000, 1100, 1800, 2300, 0030) were replaced by
    dynamic event-based names (CME_REOPEN, TOKYO_OPEN, etc.) in Feb 2026.
    Active code must use the new names from pipeline/dst.py SESSION_CATALOG.

    Frozen historical files are excluded — they document what was tested
    at runtime and serve as immutable experimental records.
    """
    violations = []
    # Old session name literals that should not appear in active code
    old_names = {"0900", "1000", "1100", "1800", "2300", "0030"}
    # Pattern: quoted string literals containing old session names
    # Matches "0900", '0900', etc. as standalone values
    pattern = re.compile(r"""(?:["'])({names})(?:["'])""".format(names="|".join(old_names)))

    frozen_files = {
        "scripts/tools/migrate_session_names.py",
        "scripts/tools/volume_session_analysis.py",
        "scripts/tools/audit_ib_single_break.py",
        "scripts/tools/audit_integrity.py",
        "scripts/tools/backtest_1100_early_exit.py",
        "scripts/tools/explore.py",
        "scripts/tools/profile_1000_runners.py",
        "pipeline/check_drift.py",  # this file defines the old names
        "scripts/audits/phase_3_docs.py",  # detects old names in rule files
        "scripts/audits/phase_8_test_suite.py",  # detects old names in tests
        "scripts/audits/phase_9_research.py",  # detects old names in research scripts
    }
    # Active research scripts that were fixed — NOT frozen
    active_research = {
        "research/cross_validate_strategies.py",
        "research/analyze_double_break.py",
    }

    # Scan only production directories (not venv/.git/.auto-claude)
    _scan_dirs = [PIPELINE_DIR, TRADING_APP_DIR, SCRIPTS_DIR]
    all_py_files = []
    for scan_dir in _scan_dirs:
        if scan_dir.exists():
            all_py_files.extend(scan_dir.rglob("*.py"))
    # Also include active research scripts explicitly
    for ar in active_research:
        ar_path = PROJECT_ROOT / ar
        if ar_path.exists():
            all_py_files.append(ar_path)

    # Frozen subdirectories within scanned dirs
    frozen_subdirs = {
        "scripts/walkforward",
    }

    for py_file in sorted(all_py_files):
        rel = py_file.relative_to(PROJECT_ROOT).as_posix()

        # Skip frozen subdirectories
        if any(rel.startswith(d + "/") for d in frozen_subdirs):
            continue
        # Skip explicitly frozen files
        if rel in frozen_files:
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        for i, line in enumerate(content.splitlines(), 1):
            # Skip comments
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            matches = pattern.findall(line)
            for m in matches:
                violations.append(f"  {rel}:{i}: old session name '{m}' — replace with new name from SESSION_CATALOG")

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
        violations.append("  strategy_discovery.py: make_strategy_id() must accept orb_minutes parameter")

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
        from pipeline.dst import SESSION_CATALOG
        from pipeline.init_db import ORB_LABELS
    except ImportError as e:
        violations.append(f"  Cannot import for sync check: {e}")
        return violations

    orb_set = set(ORB_LABELS)
    catalog_dynamic = {k for k, v in SESSION_CATALOG.items() if v.get("type") == "dynamic"}

    in_orb_only = orb_set - catalog_dynamic
    in_catalog_only = catalog_dynamic - orb_set

    if in_orb_only:
        violations.append(f"  ORB_LABELS has sessions not in SESSION_CATALOG: {sorted(in_orb_only)}")
    if in_catalog_only:
        violations.append(f"  SESSION_CATALOG has dynamic sessions not in ORB_LABELS: {sorted(in_catalog_only)}")

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
    old_names_pattern = re.compile(r"""['"](?:0900|1000|1100|1130|1800|2300|0030)['"]""")
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
                        f"  {py_file.relative_to(PROJECT_ROOT)}:{i}: stale session name in code: {line.strip()[:80]}"
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
    rr_match = re.search(r"RR_TARGETS\s*=\s*\[([\d.,\s]+)\]", builder_content)
    cb_match = re.search(r"CONFIRM_BARS_OPTIONS\s*=\s*\[([\d,\s]+)\]", builder_content)

    if rr_match:
        builder_rr = {float(x.strip()) for x in rr_match.group(1).split(",") if x.strip()}
        adapter_rr_match = re.search(r"VALID_RR_TARGETS\s*=\s*\{([^}]+)\}", adapter_content)
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
        adapter_cb_match = re.search(r"VALID_CONFIRM_BARS\s*=\s*\{([^}]+)\}", adapter_content)
        if adapter_cb_match:
            adapter_cb = {int(x.strip()) for x in adapter_cb_match.group(1).split(",") if x.strip()}
            missing = builder_cb - adapter_cb
            if missing:
                violations.append(
                    f"  sql_adapter.py VALID_CONFIRM_BARS missing: {sorted(missing)} "
                    f"(present in outcome_builder.py CONFIRM_BARS_OPTIONS)"
                )

    return violations


def check_no_active_e3(con=None) -> list[str]:
    """Check #39: No active E3 strategies in validated_setups.

    E3 (retrace limit entry) was soft-retired Feb 2026: 0/50 FDR-significant,
    no timeout mechanism (100% fill rate = late garbage included).
    """
    violations = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations  # Skip if no DB (CI)
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True
        count = con.execute(
            "SELECT COUNT(*) FROM validated_setups WHERE entry_model = 'E3' AND status = 'active'"
        ).fetchone()[0]
        if count > 0:
            violations.append(f"  validated_setups: {count} active E3 strategies (soft-retired Feb 2026)")
    except (ImportError, OSError) as e:
        print(f"    SKIP check_no_active_e3: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_no_active_e2_lookahead_filters(con=None) -> list[str]:
    """Check that active E2 strategies never use break-bar-derived filters.

    E2 stop-market entries fire on first touch after ORB completion, before the
    break bar closes. Any filter that depends on break-bar properties therefore
    becomes look-ahead for E2 and must not survive into active
    ``validated_setups``.

    This check delegates to the same canonical exclusion constants used by
    discovery and execution so the shelf audit cannot drift from runtime policy.
    """
    violations = []
    _own_con = False
    try:
        from trading_app.config import is_e2_lookahead_filter

        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations  # Skip if no DB (CI)
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        rows = con.execute(
            """
            SELECT strategy_id, instrument, orb_label, filter_type
            FROM validated_setups
            WHERE entry_model = 'E2'
              AND status = 'active'
            ORDER BY instrument, orb_label, filter_type, strategy_id
            """
        ).fetchall()

        contaminated = []
        for strategy_id, instrument, orb_label, filter_type in rows:
            if is_e2_lookahead_filter(filter_type):
                contaminated.append((strategy_id, instrument, orb_label, filter_type))

        for strategy_id, instrument, orb_label, filter_type in contaminated:
            violations.append(
                "  validated_setups: active E2 strategy "
                f"{strategy_id} uses look-ahead filter_type {filter_type!r} "
                f"({instrument} {orb_label}). Break-bar-derived filters are "
                "invalid for E2 and must not survive on the active shelf."
            )
    except (ImportError, OSError) as e:
        print(f"    SKIP check_no_active_e2_lookahead_filters: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_active_validated_filters_routable(con=None) -> list[str]:
    """Check active shelf rows still belong to the canonical session grid.

    The session-aware discovery universe is defined exclusively by
    ``trading_app.config.get_filters_for_grid(instrument, session)``. An active
    ``validated_setups`` row whose ``filter_type`` is no longer present in that
    canonical grid for its `(instrument, orb_label)` lane indicates one of:

    - session-dependent look-ahead contamination, such as OVNRNG on Asian lanes
    - DOW/session misalignment that escaped routing discipline
    - stale active rows using filters intentionally removed from the lane

    This delegates to the same canonical router used by discovery instead of
    re-encoding per-family allowlists in drift.
    """
    violations = []
    _own_con = False
    try:
        from trading_app.config import get_filters_for_grid

        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations  # Skip if no DB (CI)
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        rows = con.execute(
            """
            SELECT strategy_id, instrument, orb_label, filter_type
            FROM validated_setups
            WHERE status = 'active'
            ORDER BY instrument, orb_label, filter_type, strategy_id
            """
        ).fetchall()

        lane_cache: dict[tuple[str, str], set[str]] = {}
        for strategy_id, instrument, orb_label, filter_type in rows:
            lane = (instrument, orb_label)
            if lane not in lane_cache:
                try:
                    lane_cache[lane] = set(get_filters_for_grid(instrument, orb_label).keys())
                except Exception as exc:  # noqa: BLE001 - drift boundary, fail-closed
                    violations.append(
                        "  validated_setups: could not resolve canonical grid for "
                        f"{instrument} {orb_label}: {type(exc).__name__}: {exc}"
                    )
                    lane_cache[lane] = set()
                    continue
            if filter_type not in lane_cache[lane]:
                violations.append(
                    "  validated_setups: active strategy "
                    f"{strategy_id} uses filter_type {filter_type!r} which is not "
                    f"routable for canonical lane ({instrument}, {orb_label}). "
                    "Active shelf must remain a subset of get_filters_for_grid()."
                )
    except (ImportError, OSError) as e:
        print(f"    SKIP check_active_validated_filters_routable: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_active_micro_only_filters_on_real_micros(con=None) -> list[str]:
    """Check active shelf rows never use micro-only filters on proxy lanes.

    ``StrategyFilter.requires_micro_data`` is the canonical declaration that a
    filter's signal is only meaningful on REAL micro contract data. An active
    ``validated_setups`` row using such a filter must therefore target an
    instrument where ``pipeline.data_era.is_micro(instrument)`` is True.

    This catches the class of future contamination where a volume-based filter
    (e.g. rel-vol or ORB-volume) is promoted on GC/NQ/ES parent lanes or dead
    proxy micros whose bars_1m still come from the parent contract.

    Important limit: this check enforces instrument-level era compatibility
    only. Precise pre-launch date enforcement still needs per-strategy trading
    date provenance on the shelf; drift must not pretend that metadata exists
    when it does not.
    """
    violations = []
    _own_con = False
    try:
        from pipeline.data_era import is_micro
        from trading_app.config import ALL_FILTERS

        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations  # Skip if no DB (CI)
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        rows = con.execute(
            """
            SELECT strategy_id, instrument, orb_label, filter_type
            FROM validated_setups
            WHERE status = 'active'
            ORDER BY instrument, orb_label, filter_type, strategy_id
            """
        ).fetchall()

        micro_cache: dict[str, bool] = {}
        for strategy_id, instrument, orb_label, filter_type in rows:
            filter_obj = ALL_FILTERS.get(filter_type)
            if filter_obj is None:
                # Registered-filter drift is handled by check_validated_filters_registered().
                continue
            if not filter_obj.requires_micro_data:
                continue

            if instrument not in micro_cache:
                try:
                    micro_cache[instrument] = is_micro(instrument)
                except Exception as exc:  # noqa: BLE001 - drift boundary, fail-closed
                    violations.append(
                        "  validated_setups: could not resolve micro-era status for "
                        f"{instrument} on strategy {strategy_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    micro_cache[instrument] = False

            if not micro_cache[instrument]:
                violations.append(
                    "  validated_setups: active strategy "
                    f"{strategy_id} uses micro-only filter_type {filter_type!r} "
                    f"on non-micro instrument {instrument!r} ({orb_label}). "
                    "requires_micro_data filters are invalid on parent/proxy lanes."
                )
    except (ImportError, OSError) as e:
        print(f"    SKIP check_active_micro_only_filters_on_real_micros: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_active_micro_only_filters_after_micro_launch(con=None) -> list[str]:
    """Recompute first trade day for active micro-only filters from canonical facts.

    This is the precise Stage 3d-style honesty gate that the active shelf can
    actually support without trusting stale metadata tables. For every active
    ``validated_setups`` row whose filter declares ``requires_micro_data=True``:

    1. Load the lane's canonical ``daily_features`` rows.
    2. Recompute filter-eligible trade days using canonical filter logic.
    3. Intersect with canonical ``orb_outcomes`` for the exact lane.
    4. Verify the first traded day is on/after ``micro_launch_day(instrument)``.

    The check deliberately ignores ``strategy_trade_days`` because that table is
    known to be stale and is not authoritative for active-shelf audit work.
    """
    violations = []
    _own_con = False
    try:
        from pipeline.data_era import is_micro, micro_launch_day
        from trading_app.config import ALL_FILTERS
        from trading_app.validation_provenance import StrategyTradeWindowResolver

        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations  # Skip if no DB (CI)
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        rows = con.execute(
            """
            SELECT strategy_id, instrument, orb_label, orb_minutes,
                   entry_model, rr_target, confirm_bars, filter_type
            FROM validated_setups
            WHERE status = 'active'
            ORDER BY instrument, orb_minutes, orb_label, filter_type, strategy_id
            """
        ).fetchall()

        micro_rows = []
        for row in rows:
            filter_obj = ALL_FILTERS.get(row[7])
            if filter_obj is None or not filter_obj.requires_micro_data:
                continue
            # Non-micro instruments are handled by the instrument-level gate.
            try:
                if not is_micro(row[1]):
                    continue
            except Exception:
                continue
            micro_rows.append(row)

        if not micro_rows:
            return violations

        resolver = StrategyTradeWindowResolver(con)
        for strategy_id, instrument, orb_label, orb_minutes, entry_model, rr_target, confirm_bars, filter_type in micro_rows:
            trade_window = resolver.resolve(
                instrument=instrument,
                orb_label=orb_label,
                orb_minutes=orb_minutes,
                entry_model=entry_model,
                rr_target=rr_target,
                confirm_bars=confirm_bars,
                filter_type=filter_type,
            )

            if trade_window.trade_day_count <= 0 or trade_window.first_trade_day is None:
                violations.append(
                    "  validated_setups: active micro-only strategy "
                    f"{strategy_id} has no recomputable traded days from canonical "
                    "daily_features/orb_outcomes. Era discipline cannot be proven."
                )
                continue

            first_day = trade_window.first_trade_day
            launch_day = micro_launch_day(instrument)
            if first_day < launch_day:
                violations.append(
                    "  validated_setups: active micro-only strategy "
                    f"{strategy_id} first trades on {first_day} before "
                    f"{instrument} micro launch {launch_day}. Pre-launch "
                    "parent/proxy data is invalid for requires_micro_data filters."
                )
    except (ImportError, OSError) as e:
        print(f"    SKIP check_active_micro_only_filters_after_micro_launch: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_active_native_promotion_provenance_populated(con=None) -> list[str]:
    """Native promotion provenance fields must be populated and linkable."""
    violations = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        missing_validated_cols = _missing_table_columns(
            con,
            "validated_setups",
            [
                "status",
                "promotion_provenance",
                "validation_run_id",
                "promotion_git_sha",
                "first_trade_day",
                "last_trade_day",
                "trade_day_count",
            ],
        )
        if missing_validated_cols:
            violations.append(
                "  validated_setups: missing native promotion provenance columns "
                f"{missing_validated_cols}. Run init_trading_app_schema() migration "
                "before enforcing native provenance drift checks."
            )
            return violations

        if not _table_exists(con, "validation_run_log"):
            violations.append(
                "  validation_run_log: missing table required for native promotion "
                "provenance linkage checks."
            )
            return violations

        rows = con.execute(
            """
            SELECT strategy_id, validation_run_id, promotion_git_sha,
                   promotion_provenance, first_trade_day, last_trade_day,
                   trade_day_count
            FROM validated_setups
            WHERE status = 'active'
              AND promotion_provenance = 'VALIDATOR_NATIVE'
            ORDER BY strategy_id
            """
        ).fetchall()

        for sid, run_id, git_sha, provenance, first_day, last_day, trade_day_count in rows:
            missing = []
            if run_id in (None, ""):
                missing.append("validation_run_id")
            if git_sha in (None, ""):
                missing.append("promotion_git_sha")
            if provenance != "VALIDATOR_NATIVE":
                missing.append("promotion_provenance")
            if first_day is None:
                missing.append("first_trade_day")
            if last_day is None:
                missing.append("last_trade_day")
            if trade_day_count is None or trade_day_count <= 0:
                missing.append("trade_day_count")
            if missing:
                violations.append(
                    f"  validated_setups: active native row {sid} missing "
                    f"promotion provenance fields: {missing}"
                )
                continue

            run_row = con.execute(
                "SELECT 1 FROM validation_run_log WHERE run_id = ?",
                [run_id],
            ).fetchone()
            if run_row is None:
                violations.append(
                    f"  validated_setups: active native row {sid} references "
                    f"missing validation_run_log.run_id {run_id!r}"
                )
    except (ImportError, OSError) as e:
        print(f"    SKIP check_active_native_promotion_provenance_populated: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_active_native_trade_windows_match_provenance(con=None) -> list[str]:
    """Stored native trade-window provenance must match canonical recomputation."""
    violations = []
    _own_con = False
    try:
        from trading_app.validation_provenance import StrategyTradeWindowResolver

        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        missing_validated_cols = _missing_table_columns(
            con,
            "validated_setups",
            [
                "strategy_id",
                "instrument",
                "orb_label",
                "orb_minutes",
                "entry_model",
                "rr_target",
                "confirm_bars",
                "filter_type",
                "status",
                "promotion_provenance",
                "first_trade_day",
                "last_trade_day",
                "trade_day_count",
            ],
        )
        if missing_validated_cols:
            violations.append(
                "  validated_setups: missing native trade-window provenance columns "
                f"{missing_validated_cols}. Run init_trading_app_schema() migration "
                "before enforcing canonical provenance drift checks."
            )
            return violations

        rows = con.execute(
            """
            SELECT strategy_id, instrument, orb_label, orb_minutes, entry_model,
                   rr_target, confirm_bars, filter_type,
                   first_trade_day, last_trade_day, trade_day_count
            FROM validated_setups
            WHERE status = 'active'
              AND promotion_provenance = 'VALIDATOR_NATIVE'
            ORDER BY strategy_id
            """
        ).fetchall()
        if not rows:
            return violations

        resolver = StrategyTradeWindowResolver(con)
        for (
            sid,
            instrument,
            orb_label,
            orb_minutes,
            entry_model,
            rr_target,
            confirm_bars,
            filter_type,
            first_day,
            last_day,
            trade_day_count,
        ) in rows:
            canonical = resolver.resolve(
                instrument=instrument,
                orb_label=orb_label,
                orb_minutes=orb_minutes,
                entry_model=entry_model,
                rr_target=rr_target,
                confirm_bars=confirm_bars,
                filter_type=filter_type,
            )
            if (
                canonical.first_trade_day != first_day
                or canonical.last_trade_day != last_day
                or canonical.trade_day_count != trade_day_count
            ):
                violations.append(
                    "  validated_setups: active native row "
                    f"{sid} has stored trade window "
                    f"({first_day}, {last_day}, N={trade_day_count}) but "
                    f"canonical recompute is "
                    f"({canonical.first_trade_day}, {canonical.last_trade_day}, "
                    f"N={canonical.trade_day_count})"
                )
    except (ImportError, OSError) as e:
        print(f"    SKIP check_active_native_trade_windows_match_provenance: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_wf_coverage(con=None) -> list[str]:
    """Check #40: WF coverage for MGC/MES (soft gate — WARNING ONLY, never blocks).

    MGC and MES have enough data years for walk-forward testing.
    MNQ/M2K do not (~2-5 years), so they are skipped.
    Prints warnings if any active strategy in MGC/MES lacks wf_tested=TRUE,
    but returns empty violations so it never blocks commits.
    """
    WF_REQUIRED_INSTRUMENTS = {"MGC", "MES"}
    warnings = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return []  # Skip if no DB (CI)
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True
        for inst in sorted(WF_REQUIRED_INSTRUMENTS):
            row = con.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN wf_tested = TRUE THEN 1 ELSE 0 END) AS tested "
                "FROM validated_setups "
                "WHERE instrument = ? AND status = 'active'",
                [inst],
            ).fetchone()
            total, tested = row[0], row[1]
            if total > 0 and tested < total:
                warnings.append(f"  {inst}: {total - tested}/{total} active strategies missing WF test (soft gate)")
    except (ImportError, OSError) as e:
        print(f"    SKIP check_wf_coverage: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()

    # Print warnings but never block — this is a soft gate
    if warnings:
        for w in warnings:
            print(f"  WARNING (non-blocking): {w.strip()}")
    return []  # Always pass — soft gate only warns


def check_data_years_disclosure() -> list[str]:
    """Check #41: Warn on instruments with years_tested < 7 (WARNING ONLY, never blocks).

    MNQ (~5yr) and M2K (~5yr) have shorter histories than MGC (10yr)
    and MES (7yr). Strategies validated on shorter data may not survive
    regime changes. Advisory warning only — prints but doesn't block.
    """
    MIN_YEARS = 7
    warnings = []
    try:
        import duckdb

        db_path = GOLD_DB_PATH_FOR_CHECKS
        if db_path is None:
            from pipeline.paths import GOLD_DB_PATH

            db_path = GOLD_DB_PATH
        if not Path(db_path).exists():
            return []
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute(
                """SELECT instrument, MIN(years_tested) as min_years,
                          COUNT(*) as n_strategies
                   FROM validated_setups
                   WHERE status = 'active'
                   GROUP BY instrument
                   HAVING MIN(years_tested) < ?""",
                [MIN_YEARS],
            ).fetchall()
            for inst, min_years, n_strats in rows:
                warnings.append(
                    f"  {inst}: {n_strats} active strategies with "
                    f"min years_tested={min_years} (< {MIN_YEARS}). "
                    f"Short data history — monitor for regime fragility"
                )
        finally:
            con.close()
    except Exception as e:
        # duckdb.IOException (DB locked) is not a subclass of OSError — catch all
        msg = str(e)
        if "being used by another process" in msg or "Cannot open file" in msg:
            print("    SKIP: DB busy — another process holds the lock")
        else:
            print(f"    SKIP check_data_years_disclosure: {type(e).__name__}: {e}")

    if warnings:
        for w in warnings:
            print(f"  WARNING (non-blocking): {w.strip()}")
    return []  # Always pass — soft gate only warns


def check_uncovered_fdr_strategies(con=None) -> list[str]:
    """Check #43: Warn on FDR+WF strategies with no live_config spec (WARNING ONLY, never blocks).

    Detects strategies that passed full validation (fdr_significant=True, wf_passed=True,
    expectancy_r >= LIVE_MIN_EXPECTANCY_R) but are not covered by any spec in LIVE_PORTFOLIO.

    A strategy is "covered" if LIVE_PORTFOLIO has a spec matching its
    (orb_label, entry_model, filter_type) combination.

    Advisory only — the portfolio is intentionally curated, not exhaustive.
    Run after every validator rebuild to catch new FDR+WF winners not yet in the portfolio.
    """
    _own_con = False
    try:
        from trading_app.live_config import LIVE_MIN_EXPECTANCY_R, LIVE_PORTFOLIO

        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return []
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        covered = {(spec.orb_label, spec.entry_model, spec.filter_type) for spec in LIVE_PORTFOLIO}

        rows = con.execute(
            """SELECT DISTINCT orb_label, entry_model, filter_type,
                      COUNT(*) OVER (PARTITION BY orb_label, entry_model, filter_type) AS combo_count,
                      MAX(expectancy_r) OVER (PARTITION BY orb_label, entry_model, filter_type) AS best_exp_r
               FROM validated_setups
               WHERE status = 'active'
               AND fdr_significant = TRUE
               AND wf_passed = TRUE
               AND expectancy_r >= ?
               ORDER BY best_exp_r DESC, orb_label, entry_model, filter_type""",
            [LIVE_MIN_EXPECTANCY_R],
        ).fetchall()

        seen_combos: set[tuple] = set()
        warnings = []
        for orb_label, entry_model, filter_type, combo_count, best_exp_r in rows:
            combo = (orb_label, entry_model, filter_type)
            if combo in seen_combos:
                continue
            seen_combos.add(combo)
            if combo not in covered:
                warnings.append(
                    f"{orb_label} {entry_model} {filter_type}: "
                    f"{combo_count} FDR+WF strategy(ies) (best ExpR={best_exp_r:.3f}) "
                    f"— consider adding LiveStrategySpec to live_config.py"
                )

        if warnings:
            for w in warnings:
                print(f"  WARNING (non-blocking): {w}")
    except (ImportError, OSError) as e:
        print(f"    SKIP check_uncovered_fdr_strategies: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return []  # Always pass — advisory only


def check_variant_selection_metric() -> list[str]:
    """Check #44: All variant selection ORDER BY clauses must use expectancy_r.

    Prevents regression where one code path sorts by sharpe_ratio while another
    sorts by expectancy_r. Within a family, all variants share the same trade
    days, so ExpR = (WR*RR)-(1-WR) is the correct metric. Sharpe mechanically
    favors RR 1.0 (lower variance).

    Scans: live_config.py, rolling_portfolio.py
    """
    violations = []
    files_to_check = [
        TRADING_APP_DIR / "live_config.py",
        TRADING_APP_DIR / "rolling_portfolio.py",
    ]

    # Allowed: ORDER BY ... expectancy_r DESC
    # Forbidden: ORDER BY ... sharpe_ratio DESC (in variant selection contexts)
    forbidden = re.compile(r"ORDER\s+BY\s+\S*sharpe_ratio\s+DESC", re.IGNORECASE)

    for fpath in files_to_check:
        if not fpath.exists():
            continue
        content = fpath.read_text(encoding="utf-8")
        for i, line in enumerate(content.splitlines(), 1):
            if forbidden.search(line):
                violations.append(
                    f"  {fpath.name}:{i}: ORDER BY sharpe_ratio in variant "
                    f"selection — must use expectancy_r (Sharpe favors RR 1.0)"
                )

    return violations


def check_research_provenance_annotations() -> list[str]:
    """Check #45: Research-derived config values must have provenance annotations.

    Config constants derived from research need structured comments documenting
    which entry models were in the dataset. When entry models change (e.g.
    E0→E2), stale thresholds silently drive live execution if not re-validated.

    Scans config.py for @research-source blocks and verifies they include
    @entry-models and @revalidated-for annotations.
    """
    violations = []
    config_path = TRADING_APP_DIR / "config.py"
    if not config_path.exists():
        return violations

    content = config_path.read_text(encoding="utf-8")

    # Find all @research-source annotations
    source_pattern = re.compile(r"#\s*@research-source:\s*(.+)")
    entry_model_pattern = re.compile(r"#\s*@entry-models:\s*(.+)")

    sources = source_pattern.findall(content)
    entry_models = entry_model_pattern.findall(content)

    # Every @research-source must have a corresponding @entry-models
    if sources and not entry_models:
        violations.append(
            "  config.py: Found @research-source annotations but no @entry-models tags — add entry model provenance"
        )

    # Check that active entry models (E1, E2) appear in annotations
    for em_line in entry_models:
        models = [m.strip() for m in em_line.split(",")]
        if "E2" not in models:
            violations.append(
                f"  config.py: @entry-models '{em_line}' does not include E2 "
                f"— research may be stale for current entry model"
            )

    return violations


def check_orphaned_validated_strategies(con=None) -> list[str]:
    """Check #42: Validated strategies must have corresponding outcome data.

    Detects orphaned validated strategies — strategies in validated_setups for
    (instrument, orb_minutes) with no matching rows in orb_outcomes.

    This happens when outcome_builder is run for a subset of apertures (e.g.
    5m only) but older 15m/30m validated strategies survive the validator's
    per-aperture DELETE because the validator only removes the apertures it
    actually processed (derived from experimental_strategies).

    Fix: run outcome_builder --instrument <inst> --orb-minutes <n> --force
    then rerun strategy_discovery, strategy_validator, build_edge_families.
    """
    violations = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations  # Skip if no DB (CI)
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True
        rows = con.execute(
            """SELECT v.instrument, v.orb_minutes, COUNT(*) AS n_strategies
               FROM validated_setups v
               WHERE v.status = 'active'
               AND NOT EXISTS (
                   SELECT 1 FROM orb_outcomes o
                   WHERE o.symbol = v.instrument
                   AND o.orb_minutes = v.orb_minutes
               )
               GROUP BY v.instrument, v.orb_minutes
               ORDER BY v.instrument, v.orb_minutes"""
        ).fetchall()
        for inst, orb_min, n in rows:
            violations.append(
                f"  {inst} {orb_min}m: {n} active strategies with no "
                f"orb_outcomes rows — rebuild: outcome_builder "
                f"--instrument {inst} --orb-minutes {orb_min} --force"
            )
    except (ImportError, OSError) as e:
        print(f"    SKIP check_orb_outcomes_coverage: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_cost_model_completeness() -> list[str]:
    """Check that COST_SPECS covers all tradeable instruments and has valid values."""
    violations = []

    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from pipeline.cost_model import COST_SPECS, SESSION_SLIPPAGE_MULT

        # 1. Every active instrument must have a CostSpec
        for inst in ACTIVE_ORB_INSTRUMENTS:
            if inst not in COST_SPECS:
                violations.append(f"  {inst} is in ACTIVE_ORB_INSTRUMENTS but missing from COST_SPECS")

        # 2. Every CostSpec must have positive friction values
        for inst, spec in COST_SPECS.items():
            if spec.total_friction <= 0:
                violations.append(f"  {inst}: total_friction = {spec.total_friction} (must be > 0)")
            if spec.point_value <= 0:
                violations.append(f"  {inst}: point_value = {spec.point_value} (must be > 0)")
            if spec.instrument != inst:
                violations.append(f"  COST_SPECS['{inst}'].instrument = '{spec.instrument}' (key mismatch)")

        # 3. SESSION_SLIPPAGE_MULT keys must be subset of COST_SPECS keys
        for inst in SESSION_SLIPPAGE_MULT:
            if inst not in COST_SPECS:
                violations.append(f"  SESSION_SLIPPAGE_MULT has '{inst}' but COST_SPECS does not")

    except ImportError as e:
        violations.append(f"  Cannot import cost_model or asset_configs: {e}")

    return violations


def check_trading_rules_authority() -> list[str]:
    """Check that TRADING_RULES.md canonical values match code.

    Validates critical trading constants that, if diverged, would cause
    incorrect trades, portfolio misconfiguration, or logical inconsistency.
    """
    violations = []

    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from pipeline.cost_model import COST_SPECS
        from pipeline.dst import SESSION_CATALOG
        from trading_app.config import (
            E2_SLIPPAGE_TICKS,
            EARLY_EXIT_MINUTES,
            ENTRY_MODELS,
            TRADEABLE_INSTRUMENTS,
        )
        from trading_app.outcome_builder import RR_TARGETS

        # 1. Session catalog must have all 12 dynamic sessions
        expected_sessions = {
            "CME_REOPEN",
            "TOKYO_OPEN",
            "SINGAPORE_OPEN",
            "LONDON_METALS",
            "EUROPE_FLOW",
            "US_DATA_830",
            "NYSE_OPEN",
            "US_DATA_1000",
            "COMEX_SETTLE",
            "CME_PRECLOSE",
            "NYSE_CLOSE",
            "BRISBANE_1025",
        }
        catalog_sessions = {k for k, v in SESSION_CATALOG.items() if v.get("type") == "dynamic"}
        missing = expected_sessions - catalog_sessions
        if missing:
            violations.append(f"  SESSION_CATALOG missing dynamic sessions: {sorted(missing)}")
        extra = catalog_sessions - expected_sessions
        if extra:
            violations.append(f"  SESSION_CATALOG has unexpected dynamic sessions: {sorted(extra)}")

        # 2. Entry models
        if list(ENTRY_MODELS) != ["E1", "E2", "E3"]:
            violations.append(f"  ENTRY_MODELS = {list(ENTRY_MODELS)}, expected ['E1', 'E2', 'E3']")

        # 3. RR targets
        expected_rr = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        if list(RR_TARGETS) != expected_rr:
            violations.append(f"  RR_TARGETS = {list(RR_TARGETS)}, expected {expected_rr}")

        # 4. Tradeable instruments — canonical source is ACTIVE_ORB_INSTRUMENTS
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        if sorted(TRADEABLE_INSTRUMENTS) != sorted(ACTIVE_ORB_INSTRUMENTS):
            violations.append(
                f"  TRADEABLE_INSTRUMENTS = {sorted(TRADEABLE_INSTRUMENTS)}, "
                f"expected {sorted(ACTIVE_ORB_INSTRUMENTS)} (from ACTIVE_ORB_INSTRUMENTS)"
            )

        # 5. E2 slippage ticks
        if E2_SLIPPAGE_TICKS != 1:
            violations.append(f"  E2_SLIPPAGE_TICKS = {E2_SLIPPAGE_TICKS}, expected 1")

        # 6. MGC cost model total friction (TRADING_RULES: $5.74/RT)
        if "MGC" in COST_SPECS:
            mgc_friction = COST_SPECS["MGC"].total_friction
            if abs(mgc_friction - 5.74) > 0.01:
                violations.append(f"  MGC total_friction = {mgc_friction}, TRADING_RULES says $5.74")

        # 7. Early exit thresholds: values must be None or positive int.
        # T80 disabled 2026-03-18 (OOS validation NO-GO). All values are None.
        for session, t80 in EARLY_EXIT_MINUTES.items():
            if t80 is not None and t80 <= 0:
                violations.append(f"  EARLY_EXIT_MINUTES['{session}'] = {t80}, must be None or positive")

        # 8. All EARLY_EXIT_MINUTES keys must be valid sessions
        for session in EARLY_EXIT_MINUTES:
            if session not in catalog_sessions:
                violations.append(f"  EARLY_EXIT_MINUTES key '{session}' not in SESSION_CATALOG")

    except ImportError as e:
        violations.append(f"  Cannot import required modules: {e}")

    return violations


def check_ml_config_canonical_sources() -> list[str]:
    """Check ML config ACTIVE_INSTRUMENTS matches pipeline canonical source,
    and no features appear in both feature lists and LOOKAHEAD_BLACKLIST."""
    violations = []
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from pipeline.dst import SESSION_CATALOG
        from trading_app.ml.config import (
            ACTIVE_INSTRUMENTS,
            GLOBAL_FEATURES,
            LOOKAHEAD_BLACKLIST,
            REL_VOL_SESSIONS,
            SESSION_CHRONOLOGICAL_ORDER,
            TRADE_CONFIG_FEATURES,
        )

        # ACTIVE_INSTRUMENTS must be a subset of pipeline instruments
        # (ML excludes instruments with no validated strategies, e.g. MBT)
        extra = set(ACTIVE_INSTRUMENTS) - set(ACTIVE_ORB_INSTRUMENTS)
        if extra:
            violations.append(f"  ml/config.ACTIVE_INSTRUMENTS has instruments not in pipeline: {extra}")

        # No feature in both GLOBAL_FEATURES and LOOKAHEAD_BLACKLIST
        overlap = set(GLOBAL_FEATURES) & LOOKAHEAD_BLACKLIST
        if overlap:
            violations.append(f"  ml/config: Features in both GLOBAL_FEATURES and LOOKAHEAD_BLACKLIST: {overlap}")

        # No feature in both TRADE_CONFIG and LOOKAHEAD_BLACKLIST
        overlap2 = set(TRADE_CONFIG_FEATURES) & LOOKAHEAD_BLACKLIST
        if overlap2:
            violations.append(
                f"  ml/config: Features in both TRADE_CONFIG_FEATURES and LOOKAHEAD_BLACKLIST: {overlap2}"
            )

        # REL_VOL_SESSIONS must match SESSION_CATALOG dynamic entries
        catalog_dynamic = {name for name, cfg in SESSION_CATALOG.items() if cfg.get("type") == "dynamic"}
        if set(REL_VOL_SESSIONS) != catalog_dynamic:
            extra = set(REL_VOL_SESSIONS) - catalog_dynamic
            missing = catalog_dynamic - set(REL_VOL_SESSIONS)
            violations.append(
                f"  ml/config.REL_VOL_SESSIONS mismatch with SESSION_CATALOG: extra={extra}, missing={missing}"
            )

        # SESSION_CHRONOLOGICAL_ORDER membership must match SESSION_CATALOG dynamic entries.
        # Order is intentionally manual (Brisbane time order), but the SET must stay in sync.
        # A missing session causes index=-1 in features.py, silently corrupting cross-session counts.
        if set(SESSION_CHRONOLOGICAL_ORDER) != catalog_dynamic:
            extra = set(SESSION_CHRONOLOGICAL_ORDER) - catalog_dynamic
            missing = catalog_dynamic - set(SESSION_CHRONOLOGICAL_ORDER)
            violations.append(
                f"  ml/config.SESSION_CHRONOLOGICAL_ORDER mismatch with SESSION_CATALOG: extra={extra}, missing={missing}"
            )

    except ImportError as e:
        print(f"    SKIP check_ml_config_canonical: {e}")
        return violations

    return violations


def check_ml_lookahead_blacklist() -> list[str]:
    """Check ML LOOKAHEAD_BLACKLIST includes all required outcome targets."""
    violations = []
    try:
        from trading_app.ml.config import LOOKAHEAD_BLACKLIST

        required = {
            "outcome",
            "pnl_r",
            "pnl_dollars",
            "mae_r",
            "mfe_r",
            "double_break",
            "exit_ts",
            "exit_price",
        }
        missing = required - LOOKAHEAD_BLACKLIST
        if missing:
            violations.append(f"  ml/config.LOOKAHEAD_BLACKLIST missing required targets: {missing}")
    except ImportError as e:
        print(f"    SKIP (optional dep): {e}")
        return violations

    return violations


def check_audit_columns_populated(con=None) -> list[str]:
    """Check #50: Audit columns (n_trials, fst_hurdle, DSR) must be populated.

    Verifies that strategy_discovery has been run with the new audit code
    for each active instrument. Old rows from prior runs (E0/E3, old combos)
    legitimately have NULLs — we only fail if ZERO rows are populated,
    meaning discovery was never re-run after the schema change.
    """
    violations = []
    _own_con = False
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True
        for inst in ACTIVE_ORB_INSTRUMENTS:
            total = con.execute(
                "SELECT COUNT(*) FROM experimental_strategies WHERE instrument = ?",
                [inst],
            ).fetchone()[0]
            if total == 0:
                continue
            # At least some rows must have audit columns populated
            for col in ["n_trials_at_discovery", "fst_hurdle"]:
                populated = con.execute(
                    f"SELECT COUNT(*) FROM experimental_strategies WHERE instrument = ? AND {col} IS NOT NULL",
                    [inst],
                ).fetchone()[0]
                if populated == 0:
                    violations.append(f"  {inst}: 0/{total} rows have {col} (re-run strategy_discovery)")
            # sharpe_haircut: at least some rows with sample_size>=30
            # should have DSR computed
            eligible = con.execute(
                "SELECT COUNT(*) FROM experimental_strategies WHERE instrument = ? AND sample_size >= 30",
                [inst],
            ).fetchone()[0]
            populated_dsr = con.execute(
                "SELECT COUNT(*) FROM experimental_strategies WHERE instrument = ? AND sharpe_haircut IS NOT NULL",
                [inst],
            ).fetchone()[0]
            if eligible > 0 and populated_dsr == 0:
                violations.append(
                    f"  {inst}: 0/{eligible} eligible rows have sharpe_haircut (re-run strategy_discovery)"
                )
    except (ImportError, OSError) as e:
        print(f"    SKIP check_experimental_strategies_audit: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def _find_ml_model_path(model_dir, inst: str):
    """Find ML model path: prefer hybrid, fall back to per-instrument."""
    hybrid = model_dir / f"meta_label_{inst}_hybrid.joblib"
    legacy = model_dir / f"meta_label_{inst}.joblib"
    if hybrid.exists():
        return hybrid
    if legacy.exists():
        return legacy
    return None


def check_ml_model_files_exist() -> list[str]:
    """Check ML model .joblib files exist for all active ML instruments."""
    violations = []
    try:
        from trading_app.ml.config import ACTIVE_INSTRUMENTS, MODEL_DIR

        for inst in ACTIVE_INSTRUMENTS:
            if _find_ml_model_path(MODEL_DIR, inst) is None:
                violations.append(f"  Missing ML model for {inst} (checked hybrid + legacy)")
    except ImportError as e:
        print(f"    SKIP (optional dep): {e}")
        return violations
    return violations


def check_ml_config_hash_match() -> list[str]:
    """Check ML model config hashes match current config."""
    violations = []
    try:
        import joblib

        from trading_app.ml.config import (
            ACTIVE_INSTRUMENTS,
            MODEL_DIR,
            compute_config_hash,
        )

        current_hash = compute_config_hash()
        for inst in ACTIVE_INSTRUMENTS:
            path = _find_ml_model_path(MODEL_DIR, inst)
            if path is None:
                continue
            try:
                bundle = joblib.load(path)
                model_hash = bundle.get("config_hash")
                if model_hash and model_hash != current_hash:
                    violations.append(f"  {inst}: model hash={model_hash}, current={current_hash} — retrain needed")
            except Exception as e:
                violations.append(f"  {inst}: failed to load model: {e}")
    except ImportError as e:
        print(f"    SKIP (optional dep): {e}")
        return violations
    return violations


def check_ml_model_freshness() -> list[str]:
    """Check ML models are < 90 days old."""
    violations = []
    try:
        from datetime import datetime

        import joblib

        from trading_app.ml.config import ACTIVE_INSTRUMENTS, MODEL_DIR

        for inst in ACTIVE_INSTRUMENTS:
            path = _find_ml_model_path(MODEL_DIR, inst)
            if path is None:
                continue
            try:
                bundle = joblib.load(path)
                trained_at_str = bundle.get("trained_at")
                if not trained_at_str:
                    violations.append(f"  {inst}: model missing trained_at timestamp")
                    continue
                trained_at = datetime.fromisoformat(trained_at_str)
                age_days = (datetime.now(UTC) - trained_at).days
                if age_days > 90:
                    violations.append(f"  {inst}: model is {age_days} days old (>90 day limit)")
            except Exception as e:
                violations.append(f"  {inst}: failed to check freshness: {e}")
    except ImportError as e:
        print(f"    SKIP (optional dep): {e}")
        return violations
    return violations


# =============================================================================
# Check 54+: New checks added by deep audit (Mar 2026)
# =============================================================================


def check_live_config_spec_validity() -> list[str]:
    """Validate that every LiveStrategySpec references real sessions, entry models,
    filter types, and valid tiers. A misspelled orb_label silently produces
    an empty portfolio with zero error."""
    violations = []
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from pipeline.dst import SESSION_CATALOG
        from trading_app.config import ALL_FILTERS, ENTRY_MODELS
        from trading_app.live_config import LIVE_PORTFOLIO

        valid_sessions = {k for k, v in SESSION_CATALOG.items() if v.get("type") == "dynamic"}
        valid_entry_models = set(ENTRY_MODELS)
        valid_filters = set(ALL_FILTERS.keys()) | {"NO_FILTER"}
        valid_tiers = {"core", "regime", "hot"}

        for spec in LIVE_PORTFOLIO:
            if spec.orb_label not in valid_sessions:
                violations.append(
                    f"  LiveStrategySpec '{spec.family_id}': orb_label '{spec.orb_label}' not in SESSION_CATALOG"
                )
            if spec.entry_model not in valid_entry_models:
                violations.append(
                    f"  LiveStrategySpec '{spec.family_id}': entry_model '{spec.entry_model}' not in ENTRY_MODELS"
                )
            if spec.filter_type and spec.filter_type not in valid_filters:
                # filter_type can be composite — check base name
                base = spec.filter_type.split("_O")[0]  # strip aperture suffix
                if base not in valid_filters:
                    violations.append(
                        f"  LiveStrategySpec '{spec.family_id}': filter_type '{spec.filter_type}' not in ALL_FILTERS"
                    )
            if spec.tier not in valid_tiers:
                violations.append(f"  LiveStrategySpec '{spec.family_id}': tier '{spec.tier}' not in {valid_tiers}")
    except ImportError as e:
        violations.append(f"  Cannot import live_config: {e}")
    return violations


def check_cost_model_field_ranges() -> list[str]:
    """Validate cost model fields are within sane ranges.

    A typo like slippage=20.0 instead of 2.0 would silently make
    every strategy look unprofitable. No check previously guarded this.
    """
    violations = []
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from pipeline.cost_model import COST_SPECS, SESSION_SLIPPAGE_MULT

        for inst, spec in COST_SPECS.items():
            if inst not in ACTIVE_ORB_INSTRUMENTS:
                continue  # Skip proxy-only instruments (e.g. GC for MGC discovery)
            # Commission: $0.50 - $10.00 per RT is the sane range for micro futures
            if not (0.50 <= spec.commission_rt <= 10.0):
                violations.append(f"  {inst}: commission_rt={spec.commission_rt} outside [0.50, 10.00]")
            # Spread (doubled): $0.50 - $20.00
            if not (0.50 <= spec.spread_doubled <= 20.0):
                violations.append(f"  {inst}: spread_doubled={spec.spread_doubled} outside [0.50, 20.00]")
            # Slippage: $0.50 - $20.00
            if not (0.50 <= spec.slippage <= 20.0):
                violations.append(f"  {inst}: slippage={spec.slippage} outside [0.50, 20.00]")
            # Total friction: $2.00 - $50.00 for micro futures
            if not (2.0 <= spec.total_friction <= 50.0):
                violations.append(f"  {inst}: total_friction={spec.total_friction} outside [2.00, 50.00]")

        # Session slippage multipliers: must be in [0.5, 3.0]
        for inst, sessions in SESSION_SLIPPAGE_MULT.items():
            for session, mult in sessions.items():
                if not (0.5 <= mult <= 3.0):
                    violations.append(f"  SESSION_SLIPPAGE_MULT[{inst}][{session}]={mult} outside [0.5, 3.0]")
    except ImportError as e:
        violations.append(f"  Cannot import cost_model: {e}")
    return violations


def check_session_resolver_sanity() -> list[str]:
    """Call every session resolver for a winter and summer date and verify
    the output is a valid (hour, minute) tuple. A broken resolver would
    crash the pipeline at runtime with no pre-commit detection."""
    violations = []
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from datetime import date as dt_date

        from pipeline.dst import SESSION_CATALOG

        test_dates = [
            dt_date(2025, 1, 15),
            dt_date(2025, 7, 15),
            dt_date(2025, 3, 9),
            dt_date(2025, 11, 2),
        ]  # + DST transition days

        for label, entry in SESSION_CATALOG.items():
            if entry.get("type") != "dynamic":
                continue
            resolver = entry.get("resolver")
            if resolver is None:
                violations.append(f"  {label}: type=dynamic but no resolver function")
                continue
            for td in test_dates:
                try:
                    result = resolver(td)
                    if not isinstance(result, tuple) or len(result) != 2:
                        violations.append(f"  {label}: resolver({td}) returned {result}, expected (hour, minute)")
                        continue
                    h, m = result
                    if not (0 <= h <= 23 and 0 <= m <= 59):
                        violations.append(f"  {label}: resolver({td}) returned ({h}, {m}) — invalid time")
                except Exception as e:
                    violations.append(f"  {label}: resolver({td}) raised {type(e).__name__}: {e}")
    except ImportError as e:
        violations.append(f"  Cannot import dst: {e}")
    return violations


def check_daily_features_row_integrity(con=None) -> list[str]:
    """Verify daily_features has exactly N rows per (trading_day, symbol).

    N = len(VALID_ORB_MINUTES) — one row per aperture. A partial rebuild
    can leave days with fewer rows, causing strategy discovery to silently
    compute on wrong N.
    """
    violations = []
    _own_con = False
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from pipeline.build_daily_features import VALID_ORB_MINUTES

        expected_rows = len(VALID_ORB_MINUTES)
        # Only check active instruments — proxy-only symbols (e.g. GC for MGC
        # discovery) may legitimately have fewer apertures.
        active_syms = ", ".join(f"'{s}'" for s in ACTIVE_ORB_INSTRUMENTS)

        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True
        # Find (trading_day, symbol) pairs with != expected_rows rows
        bad = con.execute(f"""
            SELECT symbol, COUNT(*) as n_bad_days
            FROM (
                SELECT trading_day, symbol, COUNT(*) as row_count
                FROM daily_features
                WHERE symbol IN ({active_syms})
                GROUP BY trading_day, symbol
                HAVING COUNT(*) != {expected_rows}
            )
            GROUP BY symbol
        """).fetchall()
        for symbol, n_bad in bad:
            violations.append(f"  {symbol}: {n_bad} trading day(s) with != {expected_rows} rows in daily_features")
    except (ImportError, OSError) as e:
        print(f"    SKIP check_daily_features_row_integrity: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_data_continuity(con=None) -> list[str]:
    """Check #58: Warn on unexpected gaps in trading days per instrument.

    Queries daily_features for each active instrument and flags gaps > 7
    calendar days (~5 business days). Normal market closures (weekends,
    holidays) produce 2-4 day gaps; longer gaps may indicate missing data
    or incomplete ingestion.

    Advisory only — market closures are legitimate.
    """
    warnings = []
    _own_con = False
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return []
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True
        for inst in sorted(ACTIVE_ORB_INSTRUMENTS):
            rows = con.execute(
                """
                WITH days AS (
                    SELECT DISTINCT trading_day
                    FROM daily_features
                    WHERE symbol = ?
                ),
                gaps AS (
                    SELECT trading_day,
                           LEAD(trading_day) OVER (ORDER BY trading_day) as next_day
                    FROM days
                )
                SELECT trading_day, next_day,
                       next_day - trading_day as gap_days
                FROM gaps
                WHERE next_day IS NOT NULL
                AND next_day - trading_day > 7
                ORDER BY gap_days DESC
            """,
                [inst],
            ).fetchall()

            for start_day, end_day, gap in rows:
                warnings.append(f"  {inst}: {gap}-day gap from {start_day} to {end_day}")
    except (ImportError, OSError) as e:
        print(f"    SKIP check_data_continuity: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()

    if warnings:
        for w in warnings:
            print(f"  WARNING (non-blocking): {w.strip()}")
    return []  # Always pass — advisory only


def check_family_rr_locks_coverage(con=None) -> list[str]:
    """Every active instrument must have family_rr_locks rows covering its validated strategies."""
    errors = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return ["SKIPPED"]
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True
        # Check table exists
        tables = [
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name = 'family_rr_locks'"
            ).fetchall()
        ]
        if not tables:
            return ["SKIPPED"]

        # Count families in validated_setups without a matching lock
        missing = con.execute("""
            SELECT DISTINCT vs.instrument, vs.orb_label, vs.filter_type,
                   vs.entry_model, vs.orb_minutes, vs.confirm_bars
            FROM validated_setups vs
            LEFT JOIN family_rr_locks frl
              ON vs.instrument = frl.instrument
              AND vs.orb_label = frl.orb_label
              AND vs.filter_type = frl.filter_type
              AND vs.entry_model = frl.entry_model
              AND vs.orb_minutes = frl.orb_minutes
              AND vs.confirm_bars = frl.confirm_bars
            WHERE vs.status = 'active'
              AND frl.locked_rr IS NULL
        """).fetchall()
        if missing:
            errors.append(
                f"{len(missing)} active families missing from family_rr_locks "
                f"(run: python scripts/tools/select_family_rr.py)"
            )
    except (ImportError, OSError):
        return ["SKIPPED"]
    finally:
        if _own_con and con is not None:
            con.close()
    return errors


def check_frl_join_key_completeness() -> list[str]:
    """Check #60: Every family_rr_locks JOIN must use the full 6-column key.

    The 6-column key is: (instrument, orb_label, filter_type, entry_model,
    orb_minutes, confirm_bars). If any JOIN is missing a column, the query
    silently matches wrong families — catastrophic for RR lock enforcement.

    Scans all .py files that contain 'family_rr_locks' JOINs.
    """
    violations = []
    required_columns = {"instrument", "orb_label", "filter_type", "entry_model", "orb_minutes", "confirm_bars"}

    # Scan all Python files in production paths (exclude self to avoid
    # matching our own docstrings/comments/regex patterns)
    scan_dirs = [TRADING_APP_DIR, PIPELINE_DIR, SCRIPTS_DIR]
    this_file = Path(__file__).resolve()
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for fpath in scan_dir.rglob("*.py"):
            if fpath.resolve() == this_file:
                continue  # don't scan self
            content = fpath.read_text(encoding="utf-8")
            if "family_rr_locks" not in content:
                continue

            lines = content.splitlines()
            for i, line in enumerate(lines):
                # Match actual SQL JOINs: require 'frl' alias after table name.
                # This skips comments/docstrings that merely mention the table.
                if re.search(r"JOIN\s+family_rr_locks\s+frl\b", line, re.IGNORECASE):
                    # Collect the next 10 lines to find ON clause columns
                    block = "\n".join(lines[i : i + 12])
                    found_cols = set()
                    for col in required_columns:
                        if re.search(rf"frl\.{col}\b", block):
                            found_cols.add(col)
                    missing = required_columns - found_cols
                    if missing:
                        rel = fpath.relative_to(PROJECT_ROOT)
                        violations.append(f"  {rel}:{i + 1}: family_rr_locks JOIN missing columns: {sorted(missing)}")

    return violations


def check_rr_resolution_paths_locked() -> list[str]:
    """Check #61: Production RR-resolution queries must JOIN family_rr_locks.

    Files that SELECT from validated_setups with ROW_NUMBER() or LIMIT 1
    (i.e., picking ONE variant per family) must JOIN family_rr_locks to
    enforce the locked RR. Without the JOIN, the query can pick any RR.

    Scans production files only (not research/, not tests/).
    """
    violations = []

    # Production files that resolve a single variant from validated_setups
    # (these are the only files where LIMIT 1 or ROW_NUMBER() on
    # validated_setups is a valid pattern)
    production_files = [
        TRADING_APP_DIR / "live_config.py",
        TRADING_APP_DIR / "portfolio.py",
        TRADING_APP_DIR / "rolling_portfolio.py",
        TRADING_APP_DIR / "ml" / "features.py",
        SCRIPTS_DIR / "tools" / "generate_trade_sheet.py",
    ]

    # Extract individual SQL string blocks (triple-quoted) and check each
    # independently. Previous DOTALL regex matched across function boundaries,
    # causing false positives (e.g. FROM validated_setups in function A matching
    # LIMIT 1 in function B which queries experimental_strategies).
    sql_block_pattern = re.compile(r'"""(.*?)"""', re.DOTALL)
    variant_in_block = re.compile(r"FROM\s+validated_setups", re.IGNORECASE)
    pick_pattern = re.compile(r"LIMIT\s+1|ROW_NUMBER", re.IGNORECASE)

    for fpath in production_files:
        if not fpath.exists():
            continue
        content = fpath.read_text(encoding="utf-8")

        # Check each SQL string block independently
        for block_match in sql_block_pattern.finditer(content):
            block = block_match.group(1)
            if not variant_in_block.search(block):
                continue
            if not pick_pattern.search(block):
                continue
            # This block selects from validated_setups with LIMIT 1 / ROW_NUMBER
            if "family_rr_locks" not in block and "frl_join" not in block:
                line_num = content[: block_match.start()].count("\n") + 1
                rel = fpath.relative_to(PROJECT_ROOT)
                violations.append(
                    f"  {rel}:{line_num}: variant selection (LIMIT 1 / "
                    f"ROW_NUMBER) from validated_setups without "
                    f"family_rr_locks JOIN"
                )

    return violations


def check_no_hardcoded_scratch_db() -> list[str]:
    """Check #62: No hardcoded C:/db/gold.db defaults in active Python code.

    Research/script files must use pipeline.paths.GOLD_DB_PATH as their default,
    not a hardcoded scratch path. Docstrings and archive/ are excluded.
    """
    violations = []
    # Match both forward-slash (C:/db/gold.db) and backslash (C:\db\gold.db, C:\\db\\gold.db)
    _sep = r"[/\\]{1,2}"  # matches /, \, or \\
    _path = rf"C:{_sep}db{_sep}gold\.db"
    scratch_pattern = re.compile(
        rf"""(?:default\s*=\s*(?:Path\s*\(\s*)?["']{_path}["']|"""
        rf"""^DB_PATH\s*=\s*Path\s*\(\s*r?["']{_path}["'])""",
        re.MULTILINE,
    )
    scan_dirs = [RESEARCH_DIR, SCRIPTS_DIR]
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for py_file in scan_dir.rglob("*.py"):
            # Skip archive directories
            if "archive" in py_file.parts:
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for match in scratch_pattern.finditer(content):
                line_no = content[: match.start()].count("\n") + 1
                rel = py_file.relative_to(PROJECT_ROOT)
                violations.append(
                    f"  {rel}:{line_no} — hardcoded scratch DB default. Use pipeline.paths.GOLD_DB_PATH instead."
                )
    return violations


def check_db_reader_cached_connection() -> list[str]:
    """Check #63: ui/db_reader.py must use cached DB connections, not connection-per-query.

    The cached _DB_CONNECTIONS pattern eliminates 45ms overhead per query.
    Individual functions must NOT close the shared connection.
    """
    violations = []
    db_reader = PROJECT_ROOT / "ui" / "db_reader.py"
    if not db_reader.exists():
        return violations
    content = db_reader.read_text(encoding="utf-8")

    # Must have the cached connection dict
    if "_DB_CONNECTIONS" not in content:
        violations.append(
            "  ui/db_reader.py: missing _DB_CONNECTIONS cache — connection-per-query anti-pattern detected"
        )

    # Individual functions must NOT close the shared connection
    # (Only _cleanup_connections and atexit should close)
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if "con.close()" in stripped or "conn.close()" in stripped:
            # Check if we're inside _cleanup_connections (allowed)
            in_cleanup = False
            for j in range(max(0, i - 20), i):
                if "def _cleanup_connections" in lines[j - 1]:
                    in_cleanup = True
                    break
            if not in_cleanup:
                violations.append(
                    f"  ui/db_reader.py:{i}: conn.close() outside _cleanup_connections — "
                    f"shared connection must not be closed by individual callers"
                )

    return violations


def check_drift_shared_db_connection() -> list[str]:
    """Check #64: All requires_db drift checks must accept con= parameter.

    The shared connection pattern saves ~400ms (11x45ms per connect).
    """
    violations = []
    import inspect

    # Get all requires_db check functions from CHECKS
    for _label, check_fn, _is_advisory, requires_db in CHECKS:
        if not requires_db:
            continue
        sig = inspect.signature(check_fn)
        if "con" not in sig.parameters:
            violations.append(
                f"  {check_fn.__name__}: requires_db=True but missing con= parameter — cannot use shared DB connection"
            )

    return violations


def check_no_broad_rglob_in_drift_checks() -> list[str]:
    """Check #65: check_old_session_names must not use PROJECT_ROOT.rglob.

    Scoped rglob (pipeline/, trading_app/, scripts/ only) saves ~1,800ms
    by skipping venv/.git/.auto-claude tree walks.
    """
    violations = []
    drift_file = PROJECT_ROOT / "pipeline" / "check_drift.py"
    content = drift_file.read_text(encoding="utf-8")

    # Find the check_old_session_names function and look for broad rglob
    in_function = False
    for i, line in enumerate(content.splitlines(), 1):
        if "def check_old_session_names" in line:
            in_function = True
        elif in_function and line.strip().startswith("def "):
            break  # Next function — stop scanning
        elif in_function and "PROJECT_ROOT.rglob" in line:
            violations.append(
                f"  pipeline/check_drift.py:{i}: PROJECT_ROOT.rglob in "
                f"check_old_session_names — must use scoped _scan_dirs instead"
            )

    return violations


def check_stop_multiplier_consistency(con=None) -> list[str]:
    """Check: _S075 in strategy_id must match stop_multiplier=0.75 in column (and vice versa)."""
    violations = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        for table in ["experimental_strategies", "validated_setups"]:
            # Check if column exists
            cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
            if "stop_multiplier" not in cols:
                continue  # Column not yet migrated — skip

            # IDs containing _S075 but column != 0.75
            bad_id = con.execute(f"""
                SELECT strategy_id, stop_multiplier FROM {table}
                WHERE strategy_id LIKE '%\\_S075%' ESCAPE '\\'
                  AND (stop_multiplier IS NULL OR ABS(stop_multiplier - 0.75) > 0.001)
            """).fetchall()
            for sid, sm in bad_id:
                violations.append(f"  {table}: {sid} has _S075 in ID but stop_multiplier={sm}")

            # Column = 0.75 but no _S075 in ID
            bad_col = con.execute(f"""
                SELECT strategy_id, stop_multiplier FROM {table}
                WHERE ABS(stop_multiplier - 0.75) < 0.001
                  AND strategy_id NOT LIKE '%\\_S075%' ESCAPE '\\'
            """).fetchall()
            for sid, sm in bad_col:
                violations.append(f"  {table}: {sid} has stop_multiplier={sm} but no _S075 in ID")

    except (ImportError, OSError) as e:
        print(f"    SKIP check_stop_multiplier_consistency: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


# =============================================================================
# TOOLING CONFIG CHECKS
# =============================================================================


def check_pyright_config_exists(project_root: Path) -> list[str]:
    """Ensure pyrightconfig.json exists and has basic mode."""
    config_path = project_root / "pyrightconfig.json"
    if not config_path.exists():
        return ["pyrightconfig.json missing — type checking not configured"]
    import json

    config = json.loads(config_path.read_text())
    mode = config.get("typeCheckingMode", "off")
    if mode not in ("basic", "standard", "strict"):
        return [f"pyrightconfig.json typeCheckingMode={mode}, expected basic/standard/strict"]
    return []


def check_ruff_rules_minimum(project_root: Path) -> list[str]:
    """Ensure ruff.toml has minimum required rule sets."""
    ruff_path = project_root / "ruff.toml"
    if not ruff_path.exists():
        return ["ruff.toml missing"]
    import tomllib

    config = tomllib.loads(ruff_path.read_text())
    selected = config.get("lint", {}).get("select", [])
    required = ["I", "B", "UP"]
    missing = [r for r in required if r not in selected]
    if missing:
        return [f"ruff.toml missing required rule sets: {missing}"]
    return []


def check_python_version_file(project_root: Path) -> list[str]:
    """Ensure .python-version exists and matches pyproject.toml."""
    pv_path = project_root / ".python-version"
    if not pv_path.exists():
        return [".python-version file missing"]
    version = pv_path.read_text().strip()
    if not version.startswith("3.13"):
        return [f".python-version says {version}, expected 3.13"]
    return []


def check_tradovate_api_urls() -> list[str]:
    """Ensure all Tradovate URLs use tradovateapi.com (not tradovate.com).

    Per official docs: REST = {demo,live}.tradovateapi.com, WS = md.tradovateapi.com.
    The domain tradovate.com is the marketing site, not the API.
    """
    violations = []
    live_dir = TRADING_APP_DIR / "live"
    if not live_dir.exists():
        return []
    # Pattern: any URL with tradovate.com that is NOT tradovateapi.com
    bad_url = re.compile(r"""https?://[a-z-]*\.tradovate\.com/|wss?://[a-z-]*\.tradovate\.com/""")
    good_url = re.compile(r"""tradovateapi\.com""")
    for py_file in live_dir.glob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(content.splitlines(), 1):
            if bad_url.search(line) and not good_url.search(line):
                rel = py_file.relative_to(PROJECT_ROOT)
                violations.append(f"  {rel}:{i} — wrong Tradovate domain. Use tradovateapi.com per official API docs.")
    return violations


def check_uv_lock_exists(project_root: Path) -> list[str]:
    """Ensure uv.lock exists and is not a skeleton."""
    lock_path = project_root / "uv.lock"
    if not lock_path.exists():
        return ["uv.lock missing — run 'uv lock' to generate"]
    content = lock_path.read_text()
    if content.count("[[package]]") < 5:
        return ["uv.lock appears to be a skeleton — run 'uv lock' to regenerate"]
    return []


def check_pipeline_staleness(con=None) -> list[str]:
    """Fail if any active instrument has orb_outcomes > 7 trading days behind daily_features."""
    violations = []
    _own_con = False
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return []
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        stale_instruments = []
        for inst in sorted(ACTIVE_ORB_INSTRUMENTS):
            df_max = con.execute(
                "SELECT MAX(trading_day) FROM daily_features WHERE symbol = ? AND orb_minutes = 5",
                [inst],
            ).fetchone()[0]
            oo_max = con.execute(
                "SELECT MAX(trading_day) FROM orb_outcomes WHERE symbol = ?",
                [inst],
            ).fetchone()[0]

            if df_max is None or oo_max is None:
                continue  # No data yet — not a staleness issue

            # Count trading days (weekdays) between oo_max and df_max
            from scripts.tools.pipeline_status import _trading_days_between

            gap = _trading_days_between(oo_max, df_max)

            if gap > 7:
                stale_instruments.append(f"{inst} ({gap}d)")

        if stale_instruments:
            violations.append(f"  orb_outcomes stale: {', '.join(stale_instruments)}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_dead_instruments_doc_sync() -> list[str]:
    """Verify docs referencing dead instruments match the canonical DEAD_ORB_INSTRUMENTS set."""
    import re

    from pipeline.asset_configs import DEAD_ORB_INSTRUMENTS

    violations = []
    canonical = sorted(DEAD_ORB_INSTRUMENTS)
    canonical_str = ", ".join(canonical)

    # Files that should list dead instruments — pattern: "MCL, SIL, M6E" or "MCL/SIL/M6E"
    doc_files = [
        PROJECT_ROOT / "CLAUDE.md",
        PROJECT_ROOT / ".claude" / "rules" / "quant-agent-identity.md",
        PROJECT_ROOT / "docs" / "prompts" / "SYSTEM_AUDIT.md",
        PROJECT_ROOT / "docs" / "prompts" / "PIPELINE_DATA_GUARDIAN.md",
        PROJECT_ROOT / "docs" / "STRATEGY_DISCOVERY_AUDIT.md",
    ]

    # Extract ALL uppercase instrument-like symbols from lines containing at least
    # MCL and SIL, intersect with known dead set. Skip lines that also list active
    # instruments (general instrument coverage lists, not dead-specific lists).
    for fpath in doc_files:
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if "MCL" not in line or "SIL" not in line:
                continue
            all_symbols = set(re.findall(r"\b([A-Z][A-Z0-9]{1,3})\b", line))
            found_dead = all_symbols & DEAD_ORB_INSTRUMENTS
            # Skip lines that also list active instruments (general coverage lists)
            active_in_line = all_symbols & {"MGC", "MNQ", "MES"}
            if len(found_dead) >= 3 and not active_in_line and found_dead != DEAD_ORB_INSTRUMENTS:
                missing = DEAD_ORB_INSTRUMENTS - found_dead
                violations.append(
                    f"  {fpath.relative_to(PROJECT_ROOT)}:{i} — "
                    f"lists {sorted(found_dead)} but canonical is [{canonical_str}], "
                    f"missing: {sorted(missing)}"
                )

    return violations


# =============================================================================
# ML layer Bloomey fixes drift checks (Mar 2026)
# =============================================================================


def check_ml_evaluate_hybrid_support() -> list[str]:
    """V1 evaluate scripts removed in V2 cleanup (Mar 2026). No-op."""
    return []


def check_ml_bundle_full_delta() -> list[str]:
    """ML bundles must include total_full_delta_r alongside total_honest_delta_r (White 2000)."""
    violations = []
    meta_label = PROJECT_ROOT / "trading_app" / "ml" / "meta_label.py"
    if not meta_label.exists():
        violations.append("  meta_label.py: file not found")
        return violations
    text = meta_label.read_text(encoding="utf-8")
    if "total_full_delta_r" not in text:
        violations.append("  meta_label.py: missing total_full_delta_r computation (White 2000 OOS selection bias fix)")
    return violations


def check_ml_sharpe_jk_pvalue() -> list[str]:
    """Evaluate scripts must display JK p-value alongside Sharpe delta."""
    violations = []
    eval_files = [
        PROJECT_ROOT / "trading_app" / "ml" / "evaluate.py",
        PROJECT_ROOT / "trading_app" / "ml" / "evaluate_validated.py",
    ]
    for fpath in eval_files:
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8")
        if "jobson_korkie" not in text.lower():
            violations.append(f"  {fpath.name}: Sharpe delta displayed without Jobson-Korkie p-value")
    return violations


def check_ml_no_iterrows_filters() -> list[str]:
    """features.py filter application must not use iterrows (vectorized matches_df instead)."""
    violations = []
    features = PROJECT_ROOT / "trading_app" / "ml" / "features.py"
    if not features.exists():
        return violations
    text = features.read_text(encoding="utf-8")
    # Check for iterrows inside the filter application sections
    in_filter_section = False
    for i, line in enumerate(text.splitlines(), 1):
        if "filter eligibility" in line.lower() or "filter_type" in line.lower() and "keep_mask" in line:
            in_filter_section = True
        if in_filter_section and ".iterrows()" in line and "matches_row" in line:
            violations.append(f"  features.py:{i}: iterrows() still used in filter application (should use matches_df)")
        if in_filter_section and "reset_index" in line:
            in_filter_section = False
    return violations


# =============================================================================
# CHECK REGISTRY — single source of truth for all drift checks
# =============================================================================
_PIPELINE_TABLES = frozenset({"orb_outcomes", "daily_features", "bars_1m", "bars_5m", "prospective_signals"})
_TRADING_APP_TABLES = frozenset(
    {
        "validated_setups",
        "edge_families",
        "experimental_strategies",
        "family_rr_locks",
        "regime_strategies",
        "rebuild_manifest",
    }
)
_SQL_KW_RE = re.compile(r"\b(SELECT|FROM|JOIN|WHERE|AND|ON|GROUP BY|ORDER BY|INSERT|UPDATE|DELETE)\b", re.I)


def check_noise_floor_active() -> list[str]:
    """Noise floor is no longer a hard gate (2026-03-21 canon lock).

    Phase 2b removed. Noise check is now a post-validation flag (noise_risk).
    NOISE_EXPR_FLOOR being zeroed is the expected canonical state.
    This check is retained as a no-op for registry compatibility.
    """
    return []


def check_session_guard_sync() -> list[str]:
    """Verify pipeline.session_guard._SESSION_ORDER matches trading_app.ml.config.SESSION_CHRONOLOGICAL_ORDER."""
    violations = []
    try:
        from pipeline.session_guard import _SESSION_ORDER
        from trading_app.ml.config import SESSION_CHRONOLOGICAL_ORDER

        if list(_SESSION_ORDER) != list(SESSION_CHRONOLOGICAL_ORDER):
            violations.append(
                f"  session_guard._SESSION_ORDER ({len(_SESSION_ORDER)} items) != "
                f"ml.config.SESSION_CHRONOLOGICAL_ORDER ({len(SESSION_CHRONOLOGICAL_ORDER)} items). "
                f"These MUST be identical or look-ahead protection breaks."
            )
    except ImportError as e:
        violations.append(f"  Cannot import session ordering: {e}")
    return violations


def check_noise_floor_compliance(con=None) -> list[str]:
    """Verify no validated strategy has ExpR at or below its entry-model noise floor.

    Floors defined in NOISE_EXPR_FLOOR (trading_app.config) — derived from
    100-seed null test (White's Reality Check 2026-03-19).
    """
    from trading_app.config import NOISE_EXPR_FLOOR

    violations = []
    if con is None:
        return violations

    for entry_model, floor in NOISE_EXPR_FLOOR.items():
        rows = con.execute(
            """SELECT strategy_id, expectancy_r
               FROM validated_setups
               WHERE entry_model = ?
               AND expectancy_r <= ?
               AND (status IS NULL OR status NOT IN ('RETIRED', 'PURGED'))""",
            [entry_model, floor],
        ).fetchall()
        for sid, expr in rows:
            violations.append(f"  {sid}: ExpR={expr:.4f} <= noise floor {floor} for {entry_model}")

    return violations


def check_symbol_instrument_sql_convention() -> list[str]:
    """Pipeline tables use 'symbol'; trading app tables use 'instrument'.

    Flags SQL-context lines in pipeline/, trading_app/, and scripts/ that use
    the wrong column name for their table layer — the class of bug that caused
    the BinderException on orb_outcomes in generate_promotion_candidates.py.

    Only checks lines that also contain a SQL keyword to avoid false positives
    on Python attribute access (e.g. self.instrument, self.trading_day).
    """
    violations: list[str] = []
    for search_dir in (PIPELINE_DIR, TRADING_APP_DIR, SCRIPTS_DIR):
        for py_file in sorted(search_dir.rglob("*.py")):
            try:
                lines = py_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if not _SQL_KW_RE.search(stripped):
                    continue
                lo = stripped.lower()
                if any(t in lo for t in _PIPELINE_TABLES) and ".instrument" in lo:
                    violations.append(
                        f"  {py_file.relative_to(PROJECT_ROOT)}:{lineno} — "
                        f"pipeline table with '.instrument' (use '.symbol'): "
                        f"{stripped[:120]}"
                    )
                if any(t in lo for t in _TRADING_APP_TABLES) and ".symbol" in lo:
                    violations.append(
                        f"  {py_file.relative_to(PROJECT_ROOT)}:{lineno} — "
                        f"trading app table with '.symbol' (use '.instrument'): "
                        f"{stripped[:120]}"
                    )
    return violations


def check_holdout_contamination(con=None) -> list[str]:
    """Detect sacred-holdout contamination in MNQ/MES/MGC (Mode A, Amendment 2.7).

    Authority: docs/institutional/pre_registered_criteria.md Amendment 2.7 (2026-04-08)
    Decision: docs/plans/2026-04-07-holdout-policy-decision.md (top-of-file rescission)
    Canonical source: trading_app.holdout_policy

    Policy: HOLDOUT_SACRED_FROM onwards is the sacred holdout window. Any
    discovery run that touches sacred-window data without
    --holdout-date <= HOLDOUT_SACRED_FROM is contaminated. The existing
    validated_setups discovered during the Apr 3 -> Apr 8 Mode B deviation
    window are grandfathered as research-provisional per Amendment 2.4 —
    NOT OOS-clean, but not rejected either.

    Enforcement mechanism: HOLDOUT_GRANDFATHER_CUTOFF is the Amendment 2.7
    commit moment. Any experimental_strategies row with
    created_at > HOLDOUT_GRANDFATHER_CUTOFF that contains a sacred-year key
    in yearly_results was discovered without respecting --holdout-date and
    is flagged as contamination. Rows created at or before the grandfather
    cutoff are silently grandfathered.

    Fail-closed: if DB unavailable, returns a violation (not a silent pass).
    """
    violations = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return ["  HOLDOUT CHECK SKIPPED: gold.db not found — cannot verify holdout integrity"]
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        # Canonical source for Mode A policy (Amendment 2.7).
        # HOLDOUT_GRANDFATHER_CUTOFF is the enforcement moment (Apr 8 2026).
        # HOLDOUT_SACRED_FROM is the sacred window boundary (Jan 1 2026).
        # Instrument list comes from ACTIVE_ORB_INSTRUMENTS so new active
        # instruments automatically inherit the holdout.
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from trading_app.holdout_policy import (
            HOLDOUT_GRANDFATHER_CUTOFF,
            HOLDOUT_SACRED_FROM,
        )

        sacred_year_key = str(HOLDOUT_SACRED_FROM.year)

        for instrument in ACTIVE_ORB_INSTRUMENTS:
            contaminated = con.execute(
                """SELECT COUNT(*) FROM experimental_strategies
                   WHERE instrument = ?
                   AND created_at > ?
                   AND yearly_results IS NOT NULL
                   AND json_extract_string(yearly_results, '$."' || ? || '"') IS NOT NULL""",
                [instrument, HOLDOUT_GRANDFATHER_CUTOFF, sacred_year_key],
            ).fetchone()[0]
            if contaminated > 0:
                violations.append(
                    f"  HOLDOUT CONTAMINATION: {instrument} has {contaminated} experimental_strategies "
                    f"created after {HOLDOUT_GRANDFATHER_CUTOFF.date()} containing "
                    f"{sacred_year_key} trade data. "
                    f"Discovery was run without --holdout-date {HOLDOUT_SACRED_FROM.isoformat()}. "
                    f"Authority: docs/institutional/pre_registered_criteria.md Amendment 2.7. "
                    f"Canonical source: trading_app.holdout_policy"
                )
    except (ImportError, OSError) as e:
        violations.append(f"  HOLDOUT CHECK FAILED: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()

    return violations


def check_holdout_policy_declaration_consistency() -> list[str]:
    """Declaration-consistency check for Mode A holdout policy (Amendment 2.7).

    Catches drift between the canonical source (trading_app.holdout_policy),
    the binding policy doc (pre_registered_criteria.md Amendment 2.7), and
    the top-level research rules (RESEARCH_RULES.md).

    Asserts:
    1. trading_app/holdout_policy.py exists and exports the three canonical
       names (HOLDOUT_SACRED_FROM, HOLDOUT_GRANDFATHER_CUTOFF,
       enforce_holdout_date).
    2. HOLDOUT_SACRED_FROM equals date(2026, 1, 1) — Amendment 2.7 lock.
    3. docs/institutional/pre_registered_criteria.md contains the string
       "Amendment 2.7" (the rescission of Amendment 2.6).
    4. RESEARCH_RULES.md references the canonical sacred-from date
       (2026-01-01) and cites Amendment 2.7.

    Any failure indicates the canonical source and its documentation are out
    of sync, which is the exact class of drift that produced the Apr 7 Mode B
    autonomous error (the HANDOFF entry said one thing, the Apr 2 plan said
    another, and the code had no single source of truth).

    Fail-closed: if the canonical import fails, that IS the violation.
    """
    from datetime import date as _date
    from pathlib import Path

    violations = []
    project_root = Path(__file__).parent.parent

    # (1) Canonical module exists and exports the three names. We import the
    # module as an opaque object and check exports via hasattr so the import
    # block does not need ruff I001 hand-holding for aliased multi-imports.
    try:
        import trading_app.holdout_policy as _hp
    except ImportError as e:
        return [
            f"  HOLDOUT POLICY CANONICAL SOURCE MISSING: {e}. "
            "Expected trading_app/holdout_policy.py per Amendment 2.7."
        ]

    required_exports = ("HOLDOUT_SACRED_FROM", "HOLDOUT_GRANDFATHER_CUTOFF", "enforce_holdout_date")
    for name in required_exports:
        if not hasattr(_hp, name):
            violations.append(
                f"  HOLDOUT POLICY EXPORT MISSING: trading_app.holdout_policy "
                f"does not define {name!r} (required by Amendment 2.7)."
            )
    if violations:
        return violations  # bail early — downstream checks rely on the exports

    # (2) Sacred-from value is the Amendment 2.7 lock
    if _date(2026, 1, 1) != _hp.HOLDOUT_SACRED_FROM:
        violations.append(
            f"  HOLDOUT_SACRED_FROM drifted from Amendment 2.7 lock: "
            f"got {_hp.HOLDOUT_SACRED_FROM.isoformat()}, expected 2026-01-01. "
            "Changing this requires a new Amendment in pre_registered_criteria.md."
        )

    # (3) pre_registered_criteria.md cites Amendment 2.7
    criteria_path = project_root / "docs" / "institutional" / "pre_registered_criteria.md"
    if not criteria_path.exists():
        violations.append(f"  {criteria_path} missing — cannot verify Amendment 2.7 citation.")
    else:
        criteria_text = criteria_path.read_text(encoding="utf-8", errors="replace")
        if "Amendment 2.7" not in criteria_text:
            violations.append(
                f"  {criteria_path.relative_to(project_root)} does not mention "
                "'Amendment 2.7' — Mode A declaration missing from binding policy doc."
            )

    # (4) RESEARCH_RULES.md references the sacred-from date and cites Amendment 2.7
    rules_path = project_root / "RESEARCH_RULES.md"
    if not rules_path.exists():
        violations.append(f"  {rules_path} missing — cannot verify Mode A declaration.")
    else:
        rules_text = rules_path.read_text(encoding="utf-8", errors="replace")
        sacred_str = _hp.HOLDOUT_SACRED_FROM.isoformat()
        if sacred_str not in rules_text:
            violations.append(
                f"  RESEARCH_RULES.md does not mention the sacred-from date "
                f"'{sacred_str}' — doc-code drift against trading_app.holdout_policy."
            )
        if "Amendment 2.7" not in rules_text:
            violations.append(
                "  RESEARCH_RULES.md does not cite 'Amendment 2.7' — "
                "Mode A declaration missing from research rules top-level file."
            )

    return violations


def check_no_raw_orb_active_reads() -> list[str]:
    """No direct orb_active flag reads outside pipeline/asset_configs.py.

    The raw orb_active flag in ASSET_CONFIGS is dangerous because
    DEAD_ORB_INSTRUMENTS can override it (e.g. M2K has orb_active=True
    but is dead). All code must use ACTIVE_ORB_INSTRUMENTS or
    get_active_instruments() instead of reading the flag directly.

    Allowed: pipeline/asset_configs.py (defines the flag and derives the canonical list),
    tests (may test the flag directly), docs/prompts (documentation).
    """
    violations = []
    # Match cfg.get("orb_active" or ["orb_active"] or .orb_active patterns
    pattern = re.compile(r"""(?:\.get\s*\(\s*["']orb_active["']|\["orb_active"\]|\.orb_active)""")
    scan_dirs = [PIPELINE_DIR, SCRIPTS_DIR, PROJECT_ROOT / "trading_app"]
    allowed_files = {"asset_configs.py", "check_drift.py"}
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for py_file in scan_dir.rglob("*.py"):
            if py_file.name in allowed_files:
                continue
            if "archive" in py_file.parts or "test" in py_file.name.lower():
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for match in pattern.finditer(content):
                line_no = content[: match.start()].count("\n") + 1
                rel = py_file.relative_to(PROJECT_ROOT)
                violations.append(
                    f"  {rel}:{line_no} — raw orb_active read. "
                    "Use ACTIVE_ORB_INSTRUMENTS or get_active_instruments() instead."
                )
    return violations


def check_filter_self_description_coverage() -> list[str]:
    """Every ALL_FILTERS entry must produce valid AtomDescription via describe().

    The canonical-filter-self-description refactor (2026-04-07) made each
    StrategyFilter class own its own decomposition via the describe()
    method. This check enforces that contract: any new filter added to
    ALL_FILTERS must implement describe() and return a list of
    AtomDescription instances with valid enum-string fields.

    Without this check, a researcher could add a new filter to
    ALL_FILTERS, forget to implement describe(), and silently get the
    base class default — re-introducing the parallel-model drift bug
    that the refactor eliminated.

    What this check enforces:
      1. ALL_FILTERS[ft].describe(sample_row, "CME_REOPEN", "E2") does not raise
      2. The return value is a list (possibly empty for NoFilter)
      3. Every element is an AtomDescription instance
      4. NoFilter is the only filter allowed to return zero atoms
      5. CompositeFilter atoms come from leaf filters (recursive walk)
      6. No concrete filter inherits the base class default describe()
      7. atom.category ∈ {PRE_SESSION, INTRA_SESSION, OVERLAY, DIRECTIONAL}
      8. atom.resolves_at ∈ {STARTUP, ORB_FORMATION, BREAK_DETECTED,
         CONFIRM_COMPLETE, TRADE_ENTERED}
      9. atom.confidence_tier ∈ {PROVEN, PLAUSIBLE, LEGACY, UNKNOWN}

    Rules 7-9 close a silent-default gap in the eligibility adapter: the
    _CATEGORY_MAP / _RESOLVES_AT_MAP lookups silently coerce typos to
    PRE_SESSION / STARTUP (losing filter-specific semantics). By asserting
    valid string membership at the drift layer, we catch the typo at
    check time instead of at runtime on a mis-labeled eligibility report.

    @rule canonical-filter-self-description
    @stage canonical-filter-self-description (Phase 5 + post-review hardening)
    """
    # Hardcoded enum-string sets. Kept in sync with
    # trading_app.eligibility.types.ConditionCategory / ResolvesAt and
    # trading_app.eligibility.types.ConfidenceTier. If those enums change,
    # this drift check must update in lock-step (single source of truth is
    # the enum; this set is the contract-level mirror for pipeline-side
    # validation without a cross-package import).
    _VALID_CATEGORIES = frozenset({
        "PRE_SESSION", "INTRA_SESSION", "OVERLAY", "DIRECTIONAL",
    })
    _VALID_RESOLVES_AT = frozenset({
        "STARTUP", "ORB_FORMATION", "BREAK_DETECTED",
        "CONFIRM_COMPLETE", "TRADE_ENTERED",
    })
    _VALID_CONFIDENCE_TIERS = frozenset({
        "PROVEN", "PLAUSIBLE", "LEGACY", "UNKNOWN",
    })
    violations: list[str] = []
    try:
        from trading_app.config import (
            ALL_FILTERS,
            AtomDescription,
            CompositeFilter,
            NoFilter,
            StrategyFilter,
        )
    except ImportError as exc:
        return [f"  Could not import trading_app.config: {exc}"]

    # Sample row with plausible values for all common feature columns.
    # Using CME_REOPEN as the orb_label exercises the most filters.
    sample_row = {
        "orb_CME_REOPEN_size": 8.0,
        "orb_CME_REOPEN_break_delay_min": 3.0,
        "orb_CME_REOPEN_break_bar_continues": True,
        "orb_CME_REOPEN_break_dir": "long",
        "orb_CME_REOPEN_compression_tier": "Compressed",
        "orb_volume_ratio_CME_REOPEN": 1.5,
        "rel_vol_CME_REOPEN": 1.4,
        "symbol": "MGC",
        "pit_range_atr": 0.15,
        "prev_day_range": 200.0,
        "atr_20": 150.0,
        "gap_open_points": 5.0,
        "overnight_range": 80.0,
        "overnight_range_pct": 60.0,
        "atr_20_pct": 75.0,
        "cross_atr_MES_pct": 75.0,
        "cross_atr_MGC_pct": 75.0,
        "atr_vel_regime": "Stable",
        "day_of_week": 2,
        "is_nfp_day": False,
        "is_opex_day": False,
        "is_friday": False,
        "double_break": 0,
    }

    def _walk_leaves(filt: StrategyFilter) -> list[StrategyFilter]:
        if isinstance(filt, CompositeFilter):
            return _walk_leaves(filt.base) + _walk_leaves(filt.overlay)
        return [filt]

    for filter_type, filter_inst in sorted(ALL_FILTERS.items()):
        # Iterate leaves so the check sees the actual class implementing
        # describe(), not just the composite wrapper.
        leaves = _walk_leaves(filter_inst)
        for leaf in leaves:
            cls_name = type(leaf).__name__
            try:
                atoms = leaf.describe(sample_row, "CME_REOPEN", "E2")
            except Exception as exc:
                violations.append(
                    f"  {filter_type} ({cls_name}): describe() raised "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            if not isinstance(atoms, list):
                violations.append(
                    f"  {filter_type} ({cls_name}): describe() returned "
                    f"{type(atoms).__name__}, expected list"
                )
                continue

            # NoFilter is the only filter allowed zero atoms.
            if not atoms and not isinstance(leaf, NoFilter):
                violations.append(
                    f"  {filter_type} ({cls_name}): describe() returned empty "
                    f"list — only NoFilter may have zero atoms"
                )
                continue

            for atom in atoms:
                if not isinstance(atom, AtomDescription):
                    violations.append(
                        f"  {filter_type} ({cls_name}): describe() yielded "
                        f"{type(atom).__name__}, expected AtomDescription"
                    )
                    continue
                # Validate enum-string field membership. The eligibility
                # adapter uses _CATEGORY_MAP.get(...) with silent defaults,
                # so a typo in a filter's describe() would silently coerce
                # to PRE_SESSION / STARTUP at runtime. Catch it here at
                # check time instead.
                if atom.category not in _VALID_CATEGORIES:
                    violations.append(
                        f"  {filter_type} ({cls_name}): atom.category="
                        f"{atom.category!r} not in "
                        f"{sorted(_VALID_CATEGORIES)}"
                    )
                if atom.resolves_at not in _VALID_RESOLVES_AT:
                    violations.append(
                        f"  {filter_type} ({cls_name}): atom.resolves_at="
                        f"{atom.resolves_at!r} not in "
                        f"{sorted(_VALID_RESOLVES_AT)}"
                    )
                if atom.confidence_tier not in _VALID_CONFIDENCE_TIERS:
                    violations.append(
                        f"  {filter_type} ({cls_name}): atom.confidence_tier="
                        f"{atom.confidence_tier!r} not in "
                        f"{sorted(_VALID_CONFIDENCE_TIERS)}"
                    )

            # Verify the leaf has not silently inherited the base class
            # describe(): the default returns one atom with category
            # 'INTRA_SESSION' and resolves_at 'ORB_FORMATION' regardless
            # of filter semantics. Concrete filters MUST override
            # describe() — except NoFilter (returns empty) and the
            # base StrategyFilter itself (which is never instantiated).
            uses_base_default = type(leaf).describe is StrategyFilter.describe
            if uses_base_default and not isinstance(leaf, NoFilter):
                violations.append(
                    f"  {filter_type} ({cls_name}): inherits the base class "
                    f"describe() default — concrete filters MUST override "
                    f"describe() to surface filter-specific semantics"
                )

    return violations


def check_no_scratch_db_in_docstrings() -> list[str]:
    """No C:/db/gold.db in docstring usage examples (stale since Mar 2026).

    Scratch DB at C:/db/gold.db is deprecated. Usage examples in docstrings
    that suggest it as a --db or --db-path argument will mislead users.
    Intentional scratch tooling (scratch_run.py, scratch_ingest.py) is excluded.
    """
    violations = []
    scratch_re = re.compile(r"C:[/\\]{1,2}db[/\\]{1,2}gold\.db")
    scan_dirs = [PIPELINE_DIR, SCRIPTS_DIR, PROJECT_ROOT / "trading_app"]
    # These files intentionally reference scratch DB (tooling, deprecation docs, scratch workflows)
    scratch_tooling = {"scratch_run.py", "scratch_ingest.py", "check_drift.py", "paths.py", "ingest_mnq.py"}
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for py_file in scan_dir.rglob("*.py"):
            if py_file.name in scratch_tooling:
                continue
            if "archive" in py_file.parts:
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            # Only flag matches inside docstrings (triple-quoted strings)
            in_docstring = False
            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if '"""' in stripped or "'''" in stripped:
                    # Toggle docstring state (simplified — handles most cases)
                    count = stripped.count('"""') + stripped.count("'''")
                    if count % 2 == 1:
                        in_docstring = not in_docstring
                if in_docstring and scratch_re.search(line):
                    rel = py_file.relative_to(PROJECT_ROOT)
                    violations.append(
                        f"  {rel}:{i} — C:/db/gold.db in docstring. Remove or replace with canonical gold.db reference."
                    )
    return violations


# =============================================================================
# E2 canonical-window fix — structural locks (added 2026-04-07, Stage 8)
# =============================================================================
#
# These five checks lock in the Stage 5 fail-closed E2 fix from the
# E2 canonical-window refactor. Each one prevents a specific regression
# vector that would re-introduce fakeout-blind backtests (Chan Ch 1 p4
# violation: backtest must equal live execution). See
# docs/postmortems/2026-04-07-e2-canonical-window-fix.md for the full
# postmortem of why these checks exist.
#
# Each check is paired with a negative test in
# tests/test_pipeline/test_check_drift.py that injects a controlled
# violation and asserts the check detects it (Generation Is Not Validation
# rule from .claude/rules/integrity-guardian.md).


def check_canonical_source_annotations() -> list[str]:
    """Verify @canonical-source annotations point to existing files.

    @canonical-source comments in production code (pipeline/, trading_app/,
    scripts/) must reference real paths under docs/research-input/ or
    docs/institutional/literature/. This catches stale citations after
    canonical sources are renamed, moved, or deleted (e.g. quarterly
    re-scrape that replaces a help-center article with a new article ID).

    The annotation format expected by this check is:
        # @canonical-source <relative path from project root>

    Trailing parenthetical comments (article ID, scrape date) are allowed
    after the path. Examples that match:
        # @canonical-source docs/research-input/topstep/topstep_dll_article.md
        # @canonical-source docs/research-input/topstep/topstep_mll_article.md  (8284204, 2026-04-08)

    The check is non-fatal for paths in `tests/` (test fixtures may
    reference deliberately-fake paths) and ignores files under archive/.

    Established 2026-04-08 by stage 8 of docs/plans/2026-04-08-topstep-canonical-fixes.md.
    """
    violations = []
    pattern = re.compile(r"@canonical-source\s+([^\s]+)")
    scan_dirs = [PIPELINE_DIR, TRADING_APP_DIR, SCRIPTS_DIR]
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for py_file in scan_dir.rglob("*.py"):
            if "archive" in py_file.parts:
                continue
            if py_file.name == "check_drift.py":
                # Don't scan ourselves — this file references the regex pattern
                # itself and would self-flag.
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for match in pattern.finditer(content):
                ref = match.group(1).strip().rstrip(",.;:")
                # Skip placeholder/example refs (start with `<` or contain `example`)
                if ref.startswith("<") or "example" in ref.lower():
                    continue
                # Resolve relative to project root
                target = (PROJECT_ROOT / ref).resolve()
                if not target.exists():
                    line_no = content[: match.start()].count("\n") + 1
                    try:
                        rel_path = py_file.relative_to(PROJECT_ROOT).as_posix()
                    except ValueError:
                        # Test fixtures may live outside PROJECT_ROOT (tmp_path)
                        rel_path = py_file.as_posix()
                    violations.append(
                        f"{rel_path}:{line_no}: @canonical-source points to missing file: {ref}"
                    )
    return violations


def check_canonical_orb_utc_window_source() -> list[str]:
    """Only pipeline/dst.py may define `def orb_utc_window(`.

    Stage 1 of the E2 canonical-window refactor consolidated three parallel
    implementations of "compute ORB window end UTC" into one canonical
    function in pipeline.dst. This check prevents accidental re-encoding
    in build_daily_features, execution_engine, outcome_builder, or any
    new file — a regression that would re-create the parallel-models
    drift that originally caused the fakeout-blind backtest bug.

    Allowed: pipeline/dst.py (the canonical home), check_drift.py (this
    file, which references the symbol name in regex), and tests
    (test fixtures may import or reference the symbol).
    """
    violations = []
    pattern = re.compile(r"^\s*def\s+orb_utc_window\s*\(", re.MULTILINE)
    canonical_file = PIPELINE_DIR / "dst.py"
    scan_dirs = [PIPELINE_DIR, TRADING_APP_DIR, SCRIPTS_DIR]
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for py_file in scan_dir.rglob("*.py"):
            if py_file.resolve() == canonical_file.resolve():
                continue
            if py_file.name == "check_drift.py":
                continue
            if "archive" in py_file.parts:
                continue
            try:
                content = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for match in pattern.finditer(content):
                line_no = content[: match.start()].count("\n") + 1
                try:
                    rel: Path | str = py_file.relative_to(PROJECT_ROOT)
                except ValueError:
                    rel = py_file  # tmp_path during testing — fall back to absolute
                violations.append(
                    f"  {rel}:{line_no} — defines orb_utc_window outside canonical "
                    f"pipeline/dst.py. The E2 canonical-window refactor (2026-04-07) "
                    f"requires a single source of truth. Import from pipeline.dst instead."
                )
    return violations


def check_no_silent_break_ts_fallback() -> list[str]:
    """trading_app/outcome_builder.py must not silently fall back to break_ts.

    Stage 5 of the E2 canonical-window refactor deleted these lookahead-bias
    patterns from compute_single_outcome:
      - `if orb_end_utc is not None else break_ts` (the L455 silent fallback)
      - `orb_end_utc or break_ts` (a shorter equivalent)
      - `= break_ts - timedelta(minutes=break_delay)` (the L782 derivation
        from break_delay_min, which was a different shape of the same bug)

    Each pattern would scan from the close-confirmed break bar instead of
    the canonical ORB window close, missing fakeout entries and producing
    backtest results that diverge from live execution (Chan Ch 1 p4
    violation). This check prevents reintroduction.
    """
    violations = []
    forbidden_patterns = [
        ("if orb_end_utc is not None else break_ts",
         "silent fallback to break_ts — re-creates Stage 5 lookahead bias"),
        ("orb_end_utc or break_ts",
         "shorthand silent fallback to break_ts — re-creates Stage 5 lookahead bias"),
        ("= break_ts - timedelta(minutes=break_delay)",
         "L782-style derivation from break_delay_min — re-creates Stage 5 lookahead bias"),
    ]
    target = TRADING_APP_DIR / "outcome_builder.py"
    if not target.exists():
        return [f"  {target} — missing (cannot verify Stage 5 fix)"]
    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return [f"  {target} — read failed: {e}"]
    for needle, reason in forbidden_patterns:
        if needle in content:
            # Find first line number for the report
            line_no = content[: content.find(needle)].count("\n") + 1
            try:
                rel: Path | str = target.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = target  # tmp_path during testing — fall back to absolute
            violations.append(
                f"  {rel}:{line_no} — forbidden pattern '{needle}': {reason}"
            )
    return violations


def check_compute_single_outcome_canonical_kwargs() -> list[str]:
    """compute_single_outcome must accept (trading_day, orb_label, orb_minutes, orb_end_utc).

    Stage 5 added these parameters as the canonical fail-closed path for E2
    entries. A regression that removed any of them would break the
    fail-closed contract — callers would silently revert to the old
    break_ts derivation. This check uses inspect.signature to read the
    actual function parameters at import time, catching renames that a
    text-based regex would miss.
    """
    violations = []
    try:
        import inspect

        from trading_app.outcome_builder import compute_single_outcome
    except ImportError as e:
        return [f"  trading_app.outcome_builder.compute_single_outcome — import failed: {e}"]
    required = {"trading_day", "orb_label", "orb_minutes", "orb_end_utc"}
    sig = inspect.signature(compute_single_outcome)
    missing = required - set(sig.parameters.keys())
    if missing:
        violations.append(
            f"  trading_app.outcome_builder.compute_single_outcome — missing canonical "
            f"kwargs {sorted(missing)}. Stage 5 of the E2 canonical-window refactor "
            f"requires all four (trading_day, orb_label, orb_minutes, orb_end_utc) "
            f"to enforce the fail-closed contract. Re-adding any will silently break "
            f"the lookahead-bias guard."
        )
    return violations


def check_nested_builder_absent() -> list[str]:
    """trading_app/nested/builder.py must not exist.

    Stage 7 of the E2 canonical-window refactor deleted this 536-line
    module. It targeted a `nested_outcomes` table that was never created
    in init_db.py — every build_nested_outcomes() invocation crashed on
    the missing table, so the module was structurally dead from day one.
    Worse, it embedded a buggy E2 path (the same lookahead-bias bug as
    Stage 5 fixed in outcome_builder).

    Two real helpers (resample_to_5m, _verify_e3_sub_bar_fill) were
    rescued to trading_app/entry_rules.py in Stage 4 before the deletion.
    Re-creating the file would re-introduce the dead-table bug AND the
    duplicate E2 implementation, both of which Stage 7 eliminated.
    """
    target = TRADING_APP_DIR / "nested" / "builder.py"
    if target.exists():
        try:
            rel: Path | str = target.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = target  # tmp_path during testing — fall back to absolute
        return [
            f"  {rel} — file must not exist. Stage 7 of the E2 canonical-window "
            f"refactor (2026-04-07) deleted this 536-line dead module. The two real "
            f"helpers it contained (resample_to_5m, _verify_e3_sub_bar_fill) live in "
            f"trading_app/entry_rules.py — import from there. See "
            f"docs/postmortems/2026-04-07-e2-canonical-window-fix.md."
        ]
    return []


def check_phase_4_validator_gates_present() -> list[str]:
    """Verify the strategy_validator exposes the Phase 4 Stage 4.0 gates.

    Phase 4 Stage 4.0 (2026-04-08) adds pre-flight gates for institutional
    criteria 1 (hypothesis file presence), 2 (MinBTL bound), 8 (2026 OOS
    positive, N/A-safe), and 9 (era stability enforced). This drift check
    asserts the gate functions still exist as module-level callables in
    ``trading_app/strategy_validator.py`` so a future refactor cannot
    silently drop them while leaving the locked criteria text in
    ``docs/institutional/pre_registered_criteria.md`` asserting they are
    enforced.

    Criteria 4 (Chordia) and 5 (DSR) are DEFERRED to Stage 4.0b and are
    NOT checked here. Amendment 2.1 (locked) downgraded Criterion 5 to
    cross-check only until N_eff is formally solved (the existing
    informational DSR block at the bottom of run_validation already
    implements this correctly). Amendment 2.2 (locked) reframed Criterion 4
    as a 4-band ladder that requires BH FDR + WFE + 2026 OOS composition
    and therefore cannot fire as a pre-flight gate. Stage 4.0b will
    implement the banded Chordia rule as a post-validation check. Until
    then, this drift check intentionally does NOT assert those gates exist.

    The check is structural (presence of named functions in the source),
    not behavioral (does the gate fire correctly). Behavioral coverage is
    in ``tests/test_trading_app/test_strategy_validator.py``.

    @canonical-source: trading_app/strategy_validator.py
    @canonical-source: docs/institutional/pre_registered_criteria.md
    """
    violations = []
    validator_path = TRADING_APP_DIR / "strategy_validator.py"
    if not validator_path.is_file():
        return [f"strategy_validator.py not found at {validator_path}"]
    src = validator_path.read_text(encoding="utf-8")
    required_gates = [
        "_is_phase_4_grandfathered",
        "_check_criterion_1_hypothesis_file",
        "_check_criterion_2_minbtl",
        "_check_criterion_8_oos",
        "_check_criterion_9_era_stability",
        "_check_phase_4_pre_flight_gates",
    ]
    for gate in required_gates:
        if f"def {gate}(" not in src:
            violations.append(
                f"strategy_validator.py missing Phase 4 gate function: {gate}. "
                f"Phase 4 Stage 4.0 enforces this gate per "
                f"docs/institutional/pre_registered_criteria.md."
            )
    # Also assert the orchestrator is wired into run_validation's loop.
    if "_check_phase_4_pre_flight_gates(" not in src:
        violations.append(
            "strategy_validator.py defines _check_phase_4_pre_flight_gates "
            "but does not invoke it. The orchestrator must be called inside "
            "the run_validation row loop or the gates are dead code."
        )
    # Assert C4 and C5 gate bodies are NOT present — they were removed from
    # Stage 4.0 to comply with Amendments 2.1 (DSR cross-check only) and
    # 2.2 (Chordia banded post-validation). If they reappear as pre-flight
    # reject gates, drift has occurred and Amendment 2.8 is required.
    for deferred_gate in ("_check_criterion_4_chordia", "_check_criterion_5_dsr"):
        if f"def {deferred_gate}(" in src:
            violations.append(
                f"strategy_validator.py defines {deferred_gate} which is "
                f"DEFERRED to Stage 4.0b per Amendments 2.1/2.2 of the "
                f"locked criteria. Restoring it as a pre-flight reject gate "
                f"requires a new amendment with literature justification."
            )
    return violations


def check_phase_4_sha_integrity(con=None) -> list[str]:
    """Verify every stamped hypothesis_file_sha references a real file on disk.

    Phase 4 Stage 4.1 (2026-04-08) adds discovery-side SHA stamping: every
    experimental_strategies row written by ``run_discovery`` with
    ``hypothesis_file=<Path>`` carries the content SHA of the hypothesis
    YAML in ``experimental_strategies.hypothesis_file_sha``. This check is
    the INTEGRITY guard: any row whose SHA does NOT resolve to a real file
    in ``docs/audit/hypotheses/`` indicates either (a) tampering (the file
    was deleted after the run), (b) a rebase that dropped the hypothesis
    commit, or (c) a test-fixture leak into gold.db.

    Scope: rows with ``created_at >= PHASE_4_1_SHIP_DATE`` AND
    ``hypothesis_file_sha IS NOT NULL``. Rows created BEFORE the ship date
    are grandfathered (they come from legacy discovery runs that pre-date
    Stage 4.1 enforcement). Rows with NULL SHA are legacy-mode runs and
    are outside this check's scope — the validator's
    ``_is_phase_4_grandfathered`` handles them.

    This check is INTENTIONALLY narrow: it does NOT assert "post-ship-date
    rows must have a SHA". That assertion would conflict with legitimate
    legacy-mode callers (run_full_pipeline.py, parallel_rebuild.py, null
    seed tests) that intentionally call run_discovery without a hypothesis
    file. The write-time discipline enforcement for "post-ship runs should
    stamp a SHA" is the operator's responsibility at the CLI layer; this
    drift check only catches INTEGRITY violations where a stamped SHA has
    become orphaned.

    regime/nested/legacy rows with NULL hypothesis_file_sha are filtered
    out at query time so the check never flags them.

    Fail-closed: if DB unavailable, returns a violation (not a silent pass).

    @canonical-source: trading_app/holdout_policy.py
    @canonical-source: trading_app/hypothesis_loader.py
    """
    violations = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return [
                    "  PHASE 4 SHA INTEGRITY SKIPPED: gold.db not found — "
                    "cannot verify hypothesis file SHA integrity"
                ]
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        from trading_app.holdout_policy import PHASE_4_1_SHIP_DATE
        from trading_app.hypothesis_loader import find_hypothesis_file_by_sha

        # Fetch every stamped SHA that is subject to integrity enforcement.
        # Pre-ship-date rows are grandfathered (legacy). Post-ship rows with
        # NULL SHA are legacy-mode (out of scope — see docstring). Post-ship
        # rows with non-null SHA are the enforcement target.
        rows = con.execute(
            """
            SELECT DISTINCT hypothesis_file_sha
            FROM experimental_strategies
            WHERE hypothesis_file_sha IS NOT NULL
              AND created_at >= ?
            """,
            [PHASE_4_1_SHIP_DATE],
        ).fetchall()

        for (sha,) in rows:
            if not isinstance(sha, str) or not sha:
                violations.append(
                    f"  PHASE 4 SHA INTEGRITY: malformed SHA value {sha!r} "
                    f"in experimental_strategies.hypothesis_file_sha"
                )
                continue
            resolved = find_hypothesis_file_by_sha(sha)
            if resolved is None:
                # Count rows affected for operator context.
                row_count = con.execute(
                    """SELECT COUNT(*) FROM experimental_strategies
                       WHERE hypothesis_file_sha = ?""",
                    [sha],
                ).fetchone()[0]
                violations.append(
                    f"  PHASE 4 SHA INTEGRITY: orphaned SHA {sha[:12]}... "
                    f"({row_count} row(s)) — no file in "
                    f"docs/audit/hypotheses/ matches this SHA. Either the "
                    f"hypothesis file was deleted/rebased after discovery, "
                    f"or a test fixture leaked into gold.db. Investigate "
                    f"via: SELECT strategy_id, created_at FROM "
                    f"experimental_strategies WHERE hypothesis_file_sha = "
                    f"'{sha[:12]}...' LIMIT 5;"
                )
    except Exception as exc:
        violations.append(f"  PHASE 4 SHA INTEGRITY CHECK FAILED: {exc}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_prop_profiles_validated_alignment(con=None) -> list[str]:
    """Every DailyLaneSpec in an ``active=True`` AccountProfile must exist in
    ``validated_setups`` with ``status='active'``.

    Rationale: the 2026-04-09 alignment audit found that all 5 deployed lanes
    in ``topstep_50k_mnq_auto`` were ghosts — strategy_ids not present in the
    current validated book. The bot was operating with zero current validation
    backing against real capital. This check prevents the class of drift from
    recurring.

    Scope: only profiles with ``active=True`` are audited. Inactive profiles
    are exempt because they don't affect runtime — they're held as reference
    templates. A lane in an inactive profile is re-validated at the point of
    activation (the profile flip to ``active=True`` will re-run the check).

    Fail mode: returns a violation per mismatched lane, naming both the
    profile and the offending strategy_id so the operator knows where to look.
    If DB is unavailable, returns a SKIPPED-style violation so the
    ``requires_db=True`` runner handles it correctly.

    @canonical-source: trading_app/prop_profiles.py
    @canonical-source: trading_app/validated_setups schema
    """
    violations: list[str] = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return [
                    "  PROP PROFILES ALIGNMENT SKIPPED: gold.db not found — "
                    "cannot verify deployed lane validation backing"
                ]
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        from trading_app.prop_profiles import ACCOUNT_PROFILES

        for profile_id, profile in sorted(ACCOUNT_PROFILES.items()):
            if not profile.active:
                continue
            for lane in profile.daily_lanes:
                row = con.execute(
                    "SELECT status FROM validated_setups WHERE strategy_id = ?",
                    [lane.strategy_id],
                ).fetchone()
                if row is None:
                    violations.append(
                        f"  prop_profiles.{profile_id}: lane "
                        f"{lane.strategy_id!r} is NOT in validated_setups. "
                        f"Either the strategy was discovered, validated, "
                        f"and promoted OR the profile references a stale "
                        f"lane from an older discovery. Run discovery + "
                        f"validator for the affected strategy or remove the "
                        f"lane from the profile."
                    )
                    continue
                status = row[0]
                if status != "active":
                    violations.append(
                        f"  prop_profiles.{profile_id}: lane "
                        f"{lane.strategy_id!r} exists in validated_setups "
                        f"but status={status!r} (not active). A retired or "
                        f"suspended strategy cannot back a live lane."
                    )
    except Exception as exc:
        violations.append(f"  PROP PROFILES ALIGNMENT CHECK FAILED: {exc}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_shared_profile_fingerprint_canonical() -> list[str]:
    """Ensure the profile fingerprint helper lives in one canonical runtime module."""
    violations = []

    derived_state_path = TRADING_APP_DIR / "derived_state.py"
    account_survival_path = TRADING_APP_DIR / "account_survival.py"

    if not derived_state_path.exists():
        return ["  trading_app/derived_state.py missing (canonical derived-state helper required)"]

    derived_text = derived_state_path.read_text(encoding="utf-8")
    account_text = account_survival_path.read_text(encoding="utf-8") if account_survival_path.exists() else ""

    if "def build_profile_fingerprint(" not in derived_text:
        violations.append("  trading_app/derived_state.py must define build_profile_fingerprint()")

    runtime_defs = 0
    for path in TRADING_APP_DIR.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        runtime_defs += text.count("def build_profile_fingerprint(")
    if runtime_defs != 1:
        violations.append(f"  Expected exactly one runtime build_profile_fingerprint() definition, found {runtime_defs}")

    if "from trading_app.derived_state import build_profile_fingerprint" not in account_text:
        violations.append("  trading_app/account_survival.py must import build_profile_fingerprint from trading_app.derived_state")

    return violations


def check_sr_state_contract_writer() -> list[str]:
    """Ensure SR monitor persists a versioned derived-state envelope."""
    violations = []
    path = TRADING_APP_DIR / "sr_monitor.py"
    if not path.exists():
        return violations

    text = path.read_text(encoding="utf-8")
    required_tokens = [
        "build_state_envelope(",
        "schema_version=1",
        'state_type="sr_monitor"',
        "git_head=",
        'tool="sr_monitor"',
        "canonical_inputs={",
        "freshness={",
        "payload={",
        '"profile_fingerprint"',
        '"lane_ids"',
        '"db_identity"',
        '"code_fingerprint"',
    ]
    for token in required_tokens:
        if token not in text:
            violations.append(f"  trading_app/sr_monitor.py missing SR contract token: {token}")

    return violations


def check_sr_state_contract_reader() -> list[str]:
    """Ensure the SR state reader validates the envelope before trust."""
    violations = []
    pulse_path = SCRIPTS_DIR / "tools" / "project_pulse.py"
    lifecycle_path = TRADING_APP_DIR / "lifecycle_state.py"
    if not pulse_path.exists():
        return violations

    pulse_text = pulse_path.read_text(encoding="utf-8")
    collect_idx = pulse_text.find("def collect_sr_state(")
    if collect_idx == -1:
        return ["  scripts/tools/project_pulse.py missing collect_sr_state()"]

    collect_text = pulse_text[
        collect_idx : pulse_text.find("\ndef ", collect_idx + 1) if pulse_text.find("\ndef ", collect_idx + 1) != -1 else None
    ]
    uses_shared_reader = "read_criterion12_state" in collect_text
    validates_locally = "validate_state_envelope(" in collect_text and 'payload.get("results"' in collect_text

    if not validates_locally and not uses_shared_reader:
        violations.append(
            "  project_pulse.collect_sr_state() must validate SR envelope directly or delegate to "
            "trading_app.lifecycle_state.read_criterion12_state()"
        )
    if 'data.get("results"' in collect_text:
        violations.append("  project_pulse.collect_sr_state() may not trust top-level data.get('results') directly")

    if uses_shared_reader:
        if not lifecycle_path.exists():
            violations.append("  trading_app/lifecycle_state.py missing read_criterion12_state()")
            return violations
        lifecycle_text = lifecycle_path.read_text(encoding="utf-8")
        reader_idx = lifecycle_text.find("def read_criterion12_state(")
        if reader_idx == -1:
            violations.append("  trading_app.lifecycle_state.read_criterion12_state() missing")
            return violations
        reader_text = lifecycle_text[
            reader_idx
            : lifecycle_text.find("\ndef ", reader_idx + 1) if lifecycle_text.find("\ndef ", reader_idx + 1) != -1 else None
        ]
        if "validate_state_envelope(" not in reader_text:
            violations.append(
                "  trading_app.lifecycle_state.read_criterion12_state() must validate SR envelope before trust"
            )
        if 'payload.get("results"' not in reader_text:
            violations.append(
                "  trading_app.lifecycle_state.read_criterion12_state() must read SR results from validated payload"
            )
        if 'data.get("results"' in reader_text:
            violations.append(
                "  trading_app.lifecycle_state.read_criterion12_state() may not trust top-level data.get('results') directly"
            )

    return violations


def check_preflight_launcher_modes() -> list[str]:
    """Ensure launcher entrypoints pass explicit preflight claim modes."""
    violations = []
    required_tokens = {
        PROJECT_ROOT / "scripts" / "infra" / "codex-project.sh": "--claim codex --mode mutating",
        PROJECT_ROOT / "scripts" / "infra" / "codex-project-search.sh": "--claim codex-search --mode read-only",
        PROJECT_ROOT / "scripts" / "infra" / "wsl-env.sh": "--claim wsl-shell --mode read-only",
        PROJECT_ROOT / "scripts" / "infra" / "claude-worktree.sh": "--claim claude --mode mutating",
    }
    for path, token in required_tokens.items():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if token not in text:
            violations.append(f"  {path.relative_to(PROJECT_ROOT)} missing explicit preflight mode token: {token}")

    windows_launcher = PROJECT_ROOT / "scripts" / "infra" / "windows_agent_launch.py"
    if windows_launcher.exists():
        text = windows_launcher.read_text(encoding="utf-8")
        if '"--mode", mode' not in text:
            violations.append("  scripts/infra/windows_agent_launch.py must pass --mode through run_preflight()")

    return violations


def check_resample_helpers_in_entry_rules() -> list[str]:
    """resample_to_5m and _verify_e3_sub_bar_fill must be defined in trading_app.entry_rules.

    Stage 4 of the E2 canonical-window refactor extracted these helpers
    from the doomed nested/builder.py to trading_app/entry_rules.py before
    Stage 7 deleted the original. They're used by trading_app/nested/audit_outcomes.py.
    Moving them to a different module would break that import without
    necessarily breaking tests (audit_outcomes is rarely run). This check
    asserts the canonical home by importing the helpers and reading
    `__module__` — catching any silent relocation.
    """
    violations = []
    expected_module = "trading_app.entry_rules"
    try:
        from trading_app.entry_rules import (
            _verify_e3_sub_bar_fill,
            resample_to_5m,
        )
    except ImportError as e:
        return [
            f"  trading_app.entry_rules — could not import resample_to_5m or "
            f"_verify_e3_sub_bar_fill: {e}. Stage 4 of the E2 canonical-window refactor "
            f"placed these helpers here as the canonical home."
        ]
    for fn, name in [(resample_to_5m, "resample_to_5m"), (_verify_e3_sub_bar_fill, "_verify_e3_sub_bar_fill")]:
        if fn.__module__ != expected_module:
            violations.append(
                f"  trading_app.entry_rules.{name} — __module__ is {fn.__module__!r}, "
                f"expected {expected_module!r}. Stage 4 of the E2 canonical-window "
                f"refactor placed these helpers in trading_app/entry_rules.py."
            )
    return violations


# Each entry: (description, callable, is_advisory).
# is_advisory=True → prints warnings but never blocks (shown as ADVISORY).
# Check number is derived from position (1-indexed).

# Tuple format: (description, callable, is_advisory, requires_db)
# requires_db=True means check can return "SKIPPED" if DB unavailable
CHECKS = [
    (
        "Hardcoded 'MGC' SQL literals in generic pipeline code",
        lambda: check_hardcoded_mgc_sql(GENERIC_FILES),
        False,
        False,
    ),
    (".apply() / .iterrows() in ingest scripts", lambda: check_apply_iterrows(INGEST_FILES), False, False),
    ("Non-bars_1m writes in ingest scripts", lambda: check_non_bars1m_writes(INGEST_WRITE_FILES), False, False),
    ("Schema-query table name consistency", lambda: check_schema_query_consistency(PIPELINE_DIR), False, False),
    ("Import cycle prevention", lambda: check_import_cycles(PIPELINE_DIR), False, False),
    ("Hardcoded absolute paths", lambda: check_hardcoded_paths(PIPELINE_DIR), False, False),
    ("Connection leak detection", lambda: check_connection_leaks(PIPELINE_DIR), False, False),
    ("Dashboard read-only enforcement", lambda: check_dashboard_readonly(PIPELINE_DIR), False, False),
    (
        "Pipeline never imports trading_app (one-way dependency)",
        lambda: check_pipeline_never_imports_trading_app(PIPELINE_DIR),
        False,
        False,
    ),
    (
        "Trading app connection leak detection",
        lambda: check_trading_app_connection_leaks(TRADING_APP_DIR),
        False,
        False,
    ),
    ("Trading app hardcoded paths", lambda: check_trading_app_hardcoded_paths(TRADING_APP_DIR), False, False),
    ("Config filter_type sync", check_config_filter_sync, False, False),
    ("ENTRY_MODELS sync", check_entry_models_sync, False, False),
    ("Entry price sanity", check_entry_price_sanity, False, False),
    ("Nested subpackage isolation", check_nested_isolation, False, False),
    ("All imports resolve", check_all_imports_resolve, False, False),
    ("Nested production table write guard", check_nested_production_writes, False, False),
    (
        "Trading app schema-query consistency",
        lambda: check_schema_query_consistency_trading_app(TRADING_APP_DIR),
        False,
        False,
    ),
    ("Timezone hygiene", check_timezone_hygiene, False, False),
    ("MarketState read-only SQL guard", check_market_state_readonly, False, False),
    ("Analytical honesty guard (sharpe_ann)", check_sharpe_ann_presence, False, False),
    ("Ingest authority notice (ingest_dbn_mgc.py deprecation)", check_ingest_authority_notice, False, False),
    ("CLAUDE.md size cap", check_claude_md_size_cap, False, False),
    ("Validation gate existence", check_validation_gate_existence, False, False),
    ("Naive datetime detection", check_naive_datetime, False, False),
    ("DST session coverage (all sessions classified)", check_dst_session_coverage, False, False),
    ("DB config usage (configure_connection after connect)", check_db_config_usage, False, False),
    (
        "Discovery scripts use get_filters_for_grid (not ALL_FILTERS)",
        check_discovery_session_aware_filters,
        False,
        False,
    ),
    (
        "All validated filter_types registered in ALL_FILTERS",
        check_validated_filters_registered,
        False,
        True,
    ),  # requires_db
    ("E2+E3 restricted to CB1 (no CB2+ for stop-market/retrace)", check_e2_e3_cb1_only, False, False),
    ("Non-5m strategy IDs include _O{minutes} suffix", check_orb_minutes_in_strategy_id, False, False),
    ("ORB_LABELS matches SESSION_CATALOG dynamic entries", check_orb_labels_session_catalog_sync, False, False),
    ("No old fixed-clock session names in Python source", check_stale_session_names_in_code, False, False),
    ("sql_adapter VALID_* sets match outcome_builder grids", check_sql_adapter_validation_sync, False, False),
    ("No E0 rows in trading tables", check_no_e0_in_db, False, True),  # requires_db
    ("Doc stats match DB ground truth", check_doc_stats_consistency, False, True),  # requires_db
    ("No duplicate gold.db at project root", check_stale_scratch_db, False, False),
    ("No old session names in active code", check_old_session_names, False, False),
    ("No active E3 strategies (soft-retired Feb 2026)", check_no_active_e3, False, True),  # requires_db
    (
        "No active E2 strategies with look-ahead filter families",
        check_no_active_e2_lookahead_filters,
        False,
        True,
    ),  # requires_db
    (
        "Active validated filters remain canonical-routable for their lane",
        check_active_validated_filters_routable,
        False,
        True,
    ),  # requires_db
    (
        "Active micro-only filters run only on real-micro instruments",
        check_active_micro_only_filters_on_real_micros,
        False,
        True,
    ),  # requires_db
    (
        "Active micro-only filters first trade on/after micro launch",
        check_active_micro_only_filters_after_micro_launch,
        False,
        True,
    ),  # requires_db
    (
        "Active native promotion provenance fields are populated",
        check_active_native_promotion_provenance_populated,
        False,
        True,
    ),  # requires_db
    (
        "Active native trade-window provenance matches canonical recomputation",
        check_active_native_trade_windows_match_provenance,
        False,
        True,
    ),  # requires_db
    ("WF coverage for MGC/MES (soft gate)", check_wf_coverage, True, True),  # ADVISORY, requires_db
    ("Data years disclosure (years_tested < 7)", check_data_years_disclosure, True, False),  # ADVISORY only
    (
        "Orphaned validated strategies (no outcome data for aperture)",
        check_orphaned_validated_strategies,
        False,
        True,
    ),  # requires_db
    (
        "Uncovered FDR+WF strategies (FDR-validated but no live_config spec)",
        check_uncovered_fdr_strategies,
        True,
        True,
    ),  # ADVISORY, requires_db
    (
        "Variant selection ORDER BY must use expectancy_r (not sharpe_ratio)",
        check_variant_selection_metric,
        False,
        False,
    ),
    (
        "Research-derived config values need @entry-models provenance",
        check_research_provenance_annotations,
        False,
        False,
    ),
    ("Cost model completeness (COST_SPECS covers all active instruments)", check_cost_model_completeness, False, False),
    ("TRADING_RULES.md authority values match code", check_trading_rules_authority, False, False),
    (
        "ML config canonical sources (instruments, sessions, no blacklist overlap)",
        check_ml_config_canonical_sources,
        False,
        False,
    ),
    ("ML lookahead blacklist includes all outcome targets", check_ml_lookahead_blacklist, False, False),
    ("Audit columns populated (n_trials, fst_hurdle, DSR)", check_audit_columns_populated, False, True),  # requires_db
    (
        "ML model files exist for all active instruments",
        check_ml_model_files_exist,
        True,
        False,
    ),  # advisory: ML frozen (V2 gate), MES/MGC models intentionally absent
    ("ML model config hashes match current config", check_ml_config_hash_match, True, False),
    ("ML model freshness < 90 days", check_ml_model_freshness, False, False),
    # ── New checks from deep audit (Mar 2026) ──────────────────────────
    ("Live config spec validity (orb_label, entry_model, filter, tier)", check_live_config_spec_validity, False, False),
    (
        "Cost model field ranges (commission, spread, slippage, multipliers)",
        check_cost_model_field_ranges,
        False,
        False,
    ),
    (
        "Session resolver sanity (valid hour/minute, incl. DST transition days)",
        check_session_resolver_sanity,
        False,
        False,
    ),
    (
        "Daily features row integrity (one row per aperture per trading_day × symbol)",
        check_daily_features_row_integrity,
        False,
        True,
    ),  # requires_db
    (
        "Data continuity (gaps > 7 calendar days in trading days per instrument)",
        check_data_continuity,
        True,
        True,
    ),  # ADVISORY, requires_db
    (
        "family_rr_locks coverage (all active families have locked RR)",
        check_family_rr_locks_coverage,
        False,
        True,
    ),  # requires_db
    (
        "family_rr_locks JOIN key completeness (6-column key in every JOIN)",
        check_frl_join_key_completeness,
        False,
        False,
    ),
    (
        "RR resolution paths locked (LIMIT 1 / ROW_NUMBER must JOIN family_rr_locks)",
        check_rr_resolution_paths_locked,
        False,
        False,
    ),
    ("No hardcoded scratch DB defaults in active code", check_no_hardcoded_scratch_db, False, False),
    ("db_reader cached connection enforcement", check_db_reader_cached_connection, False, False),
    ("Drift check shared DB connection enforcement", check_drift_shared_db_connection, False, False),
    ("No broad rglob in drift checks", check_no_broad_rglob_in_drift_checks, False, False),
    (
        "Stop multiplier ID-column consistency (_S075 ↔ stop_multiplier)",
        check_stop_multiplier_consistency,
        False,
        True,
    ),  # requires_db
    ("pyrightconfig.json exists with basic+ mode", lambda: check_pyright_config_exists(PROJECT_ROOT), False, False),
    ("ruff.toml has minimum required rules (I, B, UP)", lambda: check_ruff_rules_minimum(PROJECT_ROOT), False, False),
    (".python-version file exists and matches 3.13", lambda: check_python_version_file(PROJECT_ROOT), False, False),
    ("uv.lock exists and is not a skeleton", lambda: check_uv_lock_exists(PROJECT_ROOT), False, False),
    ("Tradovate API URLs use tradovateapi.com (not tradovate.com)", check_tradovate_api_urls, False, False),
    (
        "Pipeline staleness: orb_outcomes not >7 trading days behind daily_features",
        check_pipeline_staleness,
        True,
        True,
    ),  # requires_db — advisory: staleness is a pipeline status issue, not code drift
    ("Dead instruments doc sync (docs match DEAD_ORB_INSTRUMENTS)", check_dead_instruments_doc_sync, False, False),
    # ── ML layer Bloomey fixes (Mar 2026) ─────────────────────────
    ("ML evaluate scripts support hybrid model format", check_ml_evaluate_hybrid_support, False, False),
    ("ML bundles include total_full_delta_r (White 2000)", check_ml_bundle_full_delta, False, False),
    ("ML Sharpe deltas include JK p-value", check_ml_sharpe_jk_pvalue, False, False),
    ("No iterrows in features.py filter application", check_ml_no_iterrows_filters, False, False),
    (
        "SQL column convention: pipeline tables use 'symbol', trading app tables use 'instrument'",
        check_symbol_instrument_sql_convention,
        False,
        False,
    ),
    ("Session guard ordering matches ML config", check_session_guard_sync, False, False),
    ("Noise floor gate removed — no-op since 2026-03-21 canon lock", check_noise_floor_active, False, False),
    (
        "No validated strategies below entry-model noise floor (per-strategy null, not global max)",
        check_noise_floor_compliance,
        False,
        True,
    ),  # requires_db
    (
        "2026 holdout not contaminated (pre-registered strategies sacred)",
        check_holdout_contamination,
        False,
        True,
    ),  # requires_db
    (
        "Holdout policy declaration consistency (docs <-> canonical source)",
        check_holdout_policy_declaration_consistency,
        False,
        False,
    ),
    ("No raw orb_active reads outside asset_configs.py", check_no_raw_orb_active_reads, False, False),
    ("No deprecated C:/db/gold.db in docstring usage examples", check_no_scratch_db_in_docstrings, False, False),
    (
        "Filter self-description coverage (every ALL_FILTERS entry has describe())",
        check_filter_self_description_coverage,
        False,
        False,
    ),
    # ── E2 canonical-window fix structural locks (2026-04-07, Stage 8) ────────
    (
        "Canonical orb_utc_window source (only pipeline/dst.py may define it)",
        check_canonical_orb_utc_window_source,
        False,
        False,
    ),
    (
        "No silent break_ts fallback in outcome_builder (Stage 5 fail-closed lock)",
        check_no_silent_break_ts_fallback,
        False,
        False,
    ),
    (
        "compute_single_outcome canonical kwargs present (trading_day/orb_label/orb_minutes/orb_end_utc)",
        check_compute_single_outcome_canonical_kwargs,
        False,
        False,
    ),
    (
        "trading_app/nested/builder.py absent (Stage 7 dead-code deletion)",
        check_nested_builder_absent,
        False,
        False,
    ),
    (
        "resample_to_5m + _verify_e3_sub_bar_fill canonical home is trading_app.entry_rules",
        check_resample_helpers_in_entry_rules,
        False,
        False,
    ),
    (
        "@canonical-source annotations point to existing files",
        check_canonical_source_annotations,
        False,
        False,
    ),
    (
        "Phase 4 validator gate functions present (criteria 1, 2, 8, 9; 4 and 5 deferred per Amendments 2.1/2.2)",
        check_phase_4_validator_gates_present,
        False,
        False,
    ),
    (
        "Phase 4 discovery SHA integrity (stamped hypothesis_file_sha must reference real file)",
        check_phase_4_sha_integrity,
        False,
        False,
    ),
    (
        "prop_profiles deployed lanes must exist in validated_setups (active=True profiles only)",
        check_prop_profiles_validated_alignment,
        False,
        True,
    ),
    ("Shared profile fingerprint helper is canonical", check_shared_profile_fingerprint_canonical, False, False),
    ("SR state writer uses derived-state contract envelope", check_sr_state_contract_writer, False, False),
    ("SR state reader validates envelope before trust", check_sr_state_contract_reader, False, False),
    ("Preflight launchers pass explicit claim modes", check_preflight_launcher_modes, False, False),
]  # end CHECKS


def main():
    print("=" * 60)
    print("PIPELINE DRIFT CHECK")
    print("=" * 60)
    print()

    all_violations = []
    advisory_count = 0
    blocking_count = 0
    skip_count = 0

    # Open shared read-only DB connection for all requires_db checks
    duckdb = _import_duckdb_or_exit()

    _shared_con = None
    db_path = _get_db_path()
    if db_path.exists():
        try:
            _shared_con = duckdb.connect(str(db_path), read_only=True)
        except Exception as exc:
            print(f"  WARNING: could not open DB ({exc}) — DB-dependent checks will skip")

    for i, (label, check_fn, is_advisory, requires_db) in enumerate(CHECKS, 1):
        print(f"Check {i}: {label}...")

        # DB-dependent checks can be skipped if DB unavailable.
        # duckdb.IOException is NOT a subclass of OSError, so we inspect
        # the message to distinguish "DB busy" from real code failures.
        if requires_db:
            try:
                v = check_fn(con=_shared_con)
            except Exception as e:
                msg = str(e)
                if "being used by another process" in msg or "Cannot open file" in msg:
                    skip_count += 1
                    print("  SKIPPED (DB busy — another process holds the lock)")
                    print()
                    continue
                v = [f"  EXCEPTION: {type(e).__name__}: {e}"]
        else:
            v = check_fn()

        if is_advisory:
            advisory_count += 1
            # Advisory checks print their own warnings; show ADVISORY tag
            print("  ADVISORY (non-blocking)")
        elif v:
            print("  FAILED:")
            for line in v:
                print(line)
            all_violations.extend(v)
        else:
            blocking_count += 1
            print("  PASSED [OK]")
        print()

    # Cleanup shared connection
    if _shared_con is not None:
        _shared_con.close()

    # Summary — blocking_count tracks actual passes (not computed from total)
    print("=" * 60)
    summary_line = (
        f"{blocking_count} checks passed [OK], {skip_count} skipped (DB unavailable), {advisory_count} advisory"
    )
    if all_violations:
        print(f"DRIFT DETECTED: {len(all_violations)} violation(s) across {summary_line}")
        print("=" * 60)
        sys.exit(1)
    else:
        print(f"NO DRIFT DETECTED: {summary_line}")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
