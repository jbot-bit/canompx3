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

import ast
import contextlib
import io
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
TRADING_APP_DIR = PROJECT_ROOT / "trading_app"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESEARCH_DIR = PROJECT_ROOT / "research"

# Ensure PROJECT_ROOT is first on sys.path so `pipeline.X` and `trading_app.X`
# imports resolve before sys.path[0] (which Python sets to the script's parent
# dir, i.e. `pipeline/`, when invoked as `python pipeline/check_drift.py`).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Module-level override for tests; production uses GOLD_DB_PATH from pipeline.paths
GOLD_DB_PATH_FOR_CHECKS = None  # Set by tests; production uses GOLD_DB_PATH

ACTIVE_DB_PATH_SCAN_DIRS = [
    RESEARCH_DIR,
    SCRIPTS_DIR / "research",
]

ACTIVE_DB_PATH_SKIP_FILES = {
    "pipeline/paths.py",
    "research/research_alt_stops.py",
    "research/research_false_breakout_bqs_tests.py",
    "research/research_gold_compass.py",
    "research/research_mnq_singapore_avoid.py",
    "research/research_shinies_bqs_overlay_tests.py",
    "research/research_shinies_universal_overlays.py",
    "research/research_universal_hypothesis_pool.py",
    "research/research_v3_mechanism.py",
}


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


class _QuietSink(io.StringIO):
    """Drop-in sys.stdout replacement for --quiet mode.

    Swallows every print() emitted by per-check helpers so the sanitized
    PASS/FAIL stream stays clean. Provides ``reconfigure()`` as a no-op
    because some imported modules (e.g. ``trading_app.outcome_builder``)
    call ``sys.stdout.reconfigure(line_buffering=True)`` at import time.
    Bare ``io.StringIO`` does not expose that method.
    """

    def reconfigure(self, **_kwargs: object) -> None:  # noqa: D401
        return None


def _safe_label_for_quiet(label: str) -> str:
    """Strip non-ASCII glyphs from a CHECKS label for --quiet output.

    --quiet output is meant for LLM consumption and pre-commit subprocess
    capture where the parent stdout encoding may be cp1252 (Windows
    default). Some legacy labels contain glyphs like ``↔`` that break
    cp1252; we ASCII-fold them here rather than mutating the canonical
    CHECKS list.
    """
    return label.encode("ascii", "replace").decode("ascii")


def _skip_db_check_for_ci(skip_msg: str) -> list[str]:
    """When CI=true (set automatically by GitHub Actions) or SKIP_DB_CHECKS=1
    is set, return [] (silent skip — runner counts it as skip_count).
    Otherwise return [skip_msg] as a violation (preserves local fail-closed
    behavior — missing DB locally indicates broken setup).

    Per CLAUDE.md "Database Location & Workflow": gold.db lives on local
    disk only ("no cloud sync"). CI runners legitimately have no DB; local
    devs are expected to. Same check, two contexts, env-controlled response.
    No `continue-on-error` band-aid — drift stays a hard gate everywhere;
    only the missing-DB case is treated correctly per context.
    """
    import os

    if os.environ.get("CI", "").lower() == "true" or os.environ.get("SKIP_DB_CHECKS", "") == "1":
        # Print so CI logs show WHICH check skipped and WHY — avoids the
        # misleading "PASSED" label when nothing was actually verified.
        print(f"  SKIPPED (CI/SKIP_DB_CHECKS env set): {skip_msg.strip()}")
        return []
    return [skip_msg]


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


def _looks_like_sql_block(text: str) -> bool:
    """Return True only for real SQL statement blocks, not prose/docstrings."""
    statement_patterns = (
        re.compile(r"\bSELECT\b[\s\S]*\bFROM\s+\w+\b", re.IGNORECASE),
        re.compile(r"\bINSERT\s+INTO\s+\w+\b", re.IGNORECASE),
        re.compile(r"\bDELETE\s+FROM\s+\w+\b", re.IGNORECASE),
        re.compile(r"^\s*UPDATE\s+\w+\s+SET\b", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*CREATE\s+(TABLE|VIEW|INDEX|SCHEMA)\b", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*WITH\s+\w+\s+AS\s*\(", re.IGNORECASE | re.MULTILINE),
    )
    return any(pattern.search(text) for pattern in statement_patterns)


def _has_sql_literal_context(content: str, match_start: int) -> bool:
    """Return True when a triple-quoted string is used as a SQL literal."""
    prefix = content[max(0, match_start - 160) : match_start]
    execute_call = re.search(r"(?:\.|^)\s*(execute|executemany|sql)\s*\(\s*$", prefix, re.IGNORECASE)
    sql_assignment = re.search(r"[\w_]*(sql|query|schema)[\w_]*\s*=\s*$", prefix, re.IGNORECASE)
    return bool(execute_call or sql_assignment)


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

    for fpath in pipeline_dir.glob("*.py"):
        if fpath.name in ("init_db.py", "check_drift.py"):
            continue

        content = fpath.read_text(encoding="utf-8")

        for match in sql_string_pattern.finditer(content):
            sql_text = match.group(1) or match.group(2)

            if not _has_sql_literal_context(content, match.start()):
                continue
            # Only check strings that actually start like SQL. This avoids
            # prose false positives such as "update the baton", which would
            # otherwise be parsed as UPDATE <table>.
            if not _looks_like_sql_block(sql_text):
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


def check_aperture_hardcode_in_scoring_paths(trading_app_dir: Path) -> list[str]:
    """Block the PR #189 class bug from recurring in lane-iterating scoring paths.

    Background — three recurrences of the same fingerprint:
      - PR #189 (commit d73bab28): trading_app/lane_allocator.py — 5 query sites
        hardcoded `WHERE orb_minutes=5` while iterating O15/O30 lanes.
      - PR #231: trading_app/paper_trade_logger.py — _load_features and
        _inject_cross_asset_atrs same fingerprint.
      - PR #232: trading_app/paper_trader.py — _inject_cross_asset_atrs_for_replay
        same fingerprint.

    Root cause: no canonical "load aperture-correct daily_features for a lane"
    helper; every lane-iterating consumer rolls its own SQL with `WHERE orb_minutes=5`
    literal.

    Scope of this check: lane-iterating scoring/execution paths —
    `trading_app/paper_*.py`, `trading_app/lane_*.py`, and
    `trading_app/live/*.py`. Research scripts, allocator session-regime
    gates, and DISTINCT/aggregate reads of non-aperture columns are exempt.
    Flag deliberate uses with `# canonical-cte-guard:`
    (per .claude/rules/daily-features-joins.md) or
    `# session-regime-gate:` (per lane_allocator.py:454) within 20 lines
    above the violating line — the deliberate `lane_allocator.py:454` SQL
    has its annotation 16 lines above, so a small window (e.g. 5) misses it.

    Pattern: SQL fragment containing `daily_features` AND `orb_minutes = 5`
    literal on the same line, AND `daily_features` appears within a 6-line
    window. Bounded window avoids matching unrelated string literals.
    """
    violations: list[str] = []

    if not trading_app_dir.exists():
        return violations

    target_files: list[Path] = []
    for glob in ("paper_*.py", "lane_*.py"):
        target_files.extend(trading_app_dir.glob(glob))
    live_dir = trading_app_dir / "live"
    if live_dir.exists():
        target_files.extend(live_dir.glob("*.py"))

    # Pattern: orb_minutes literal-5 (handles `=5`, `= 5`, `IN (5)` minimal forms).
    pat_hardcode = re.compile(r"\borb_minutes\s*(=|IN\s*\(\s*)\s*5\b")
    pat_daily_features = re.compile(r"\bdaily_features\b")
    exempt_markers = ("canonical-cte-guard:", "session-regime-gate:")
    exempt_lookback = 20  # lines

    for fpath in target_files:
        content = fpath.read_text(encoding="utf-8")
        lines = content.splitlines()

        for line_num, line in enumerate(lines, 1):
            if not pat_hardcode.search(line):
                continue

            # Exemption: any of the previous `exempt_lookback` lines (or the
            # violating line itself) carries a canonical marker comment.
            window_start = max(0, line_num - 1 - exempt_lookback)
            window = "\n".join(lines[window_start:line_num])
            if any(marker in window for marker in exempt_markers):
                continue

            # Anchor: `daily_features` must appear within a 6-line window so
            # we don't flag unrelated string literals that happen to mention
            # `orb_minutes = 5` in a comment about a different module.
            anchor_start = max(0, line_num - 6)
            anchor_end = min(len(lines), line_num + 5)
            anchor_text = "\n".join(lines[anchor_start:anchor_end])
            if not pat_daily_features.search(anchor_text):
                continue

            rel_path = fpath.relative_to(trading_app_dir).as_posix()
            stripped = line.strip()
            violations.append(
                f"  trading_app/{rel_path}:{line_num}: hardcoded `orb_minutes = 5` in "
                f"lane-iterating scoring/execution path (PR #189 class bug). Plumb the "
                f"lane's orb_minutes; or annotate within 20 lines above with "
                f"`# canonical-cte-guard:` (CTE dedup) or "
                f"`# session-regime-gate:` (deliberate cross-aperture comparator). "
                f"See PR #231/#232. Source: {stripped[:80]}"
            )

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


def check_dashboard_localhost_only_binding(trading_app_dir: Path) -> list[str]:
    """Bot dashboard MUST default to localhost-only binding.

    Rationale: bot_dashboard.py serves SSE (live position state) and exposes
    /api/action/kill (real-money mutation). Binding 0.0.0.0 would expose both
    over LAN. The runtime guard in run_dashboard() raises on non-localhost,
    but this drift check catches the SOURCE-of-truth — anyone editing the
    argparse default to "0.0.0.0" trips this check before commit.

    Scope: file `trading_app/live/bot_dashboard.py`. Looks for the function
    signature `def run_dashboard(host: str = "..."` and the argparse `--host`
    default; both must be loopback.
    """
    violations: list[str] = []
    dash = trading_app_dir / "live" / "bot_dashboard.py"
    if not dash.exists():
        return violations
    content = dash.read_text(encoding="utf-8")
    loopback = {"127.0.0.1", "localhost", "::1"}

    # Defence 1: function signature default
    fn_match = re.search(r'def\s+run_dashboard\s*\(\s*host\s*:\s*str\s*=\s*"([^"]+)"', content)
    if not fn_match:
        violations.append(
            '  bot_dashboard.py: run_dashboard signature with host: str="..." not found — '
            "regression on the localhost-only contract."
        )
    elif fn_match.group(1) not in loopback:
        violations.append(
            f"  bot_dashboard.py: run_dashboard host default {fn_match.group(1)!r} is not "
            f"a loopback address — SSE + kill endpoint must not bind to LAN."
        )

    # Defence 2: argparse --host default must exist and be loopback.
    # Absence of the --host argument is also a violation: removing it silently
    # disables CLI-level host restriction without any other guard firing.
    cli_match = re.search(r'add_argument\(\s*"--host"\s*,\s*default\s*=\s*"([^"]+)"', content)
    if not cli_match:
        violations.append(
            "  bot_dashboard.py: argparse --host argument not found — CLI-level localhost restriction has been removed."
        )
    elif cli_match.group(1) not in loopback:
        violations.append(
            f"  bot_dashboard.py: argparse --host default {cli_match.group(1)!r} is not a loopback address."
        )

    # Defence 3: ensure the RuntimeError guard inside run_dashboard is still present.
    if "Refusing to start dashboard on non-localhost host" not in content:
        violations.append(
            "  bot_dashboard.py: run_dashboard no longer carries the non-localhost RuntimeError "
            "guard — removed without explicit replacement."
        )
    return violations


def check_dashboard_sse_single_worker(trading_app_dir: Path) -> list[str]:
    """Bot dashboard SSE assumes single uvicorn worker.

    Rationale: _SSEBroker subscriber set is in-process. Multi-worker would
    fragment subscribers across worker processes, breaking ring-buffer replay
    and the subscriber cap. Catch any future refactor that adds `workers=N`
    where N>1 or removes the explicit `workers=1` pin.
    """
    violations: list[str] = []
    dash = trading_app_dir / "live" / "bot_dashboard.py"
    if not dash.exists():
        return violations
    content = dash.read_text(encoding="utf-8")

    # Locate `uvicorn.run(...)` invocations and inspect the workers= arg.
    for match in re.finditer(r"uvicorn\.run\s*\(([^)]+)\)", content, flags=re.DOTALL):
        call_args = match.group(1)
        workers_match = re.search(r"workers\s*=\s*(\d+)", call_args)
        if workers_match is None:
            violations.append(
                "  bot_dashboard.py: uvicorn.run(...) call missing explicit workers=1 — "
                "SSE broker subscriber set is in-process; multi-worker would split it."
            )
        elif int(workers_match.group(1)) != 1:
            violations.append(
                f"  bot_dashboard.py: uvicorn.run(workers={workers_match.group(1)}) — SSE invariant requires workers=1."
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
        trading_app_dir / "live" / "trade_journal.py",
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

    if not trading_app_dir.exists():
        return violations

    for fpath in trading_app_dir.rglob("*.py"):
        if fpath.name in ("__init__.py", "schema.py", "db_manager.py"):
            continue

        content = fpath.read_text(encoding="utf-8")

        for match in sql_string_pattern.finditer(content):
            sql_text = match.group(1) or match.group(2)

            if not _has_sql_literal_context(content, match.start()):
                continue
            if not _looks_like_sql_block(sql_text):
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
    # ML subsystem (trading_app.ml.*) removed 2026-04-11 (V1/V2/V3 DEAD)
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


def check_cryptography_pin_holds() -> list[str]:
    """Two-phase guard for the cryptography<47 sidecar pin.

    Phase 1 — fail-closed: if cryptography>=47 is installed alongside fastmcp,
    every FastMCP MCP server crashes on startup (Authlib 1.7.0 imports
    `cryptography.hazmat.backends.default_backend`, removed in cryptography 47).

    Phase 2 — advisory staleness: if the pin's `revisit-by:` date in
    `constraints.txt` has passed, emit a non-blocking advisory prompting a
    check of Authlib's latest release. Stops the pin from silently blocking
    security updates after upstream fixes itself.

    Pin lives in `constraints.txt` (not pyproject.toml — sidecar pip-installs).
    Detail: memory/feedback_mcp_venv_drift_cryptography47.md
    """
    violations: list[str] = []

    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return violations

    # ---- Phase 1: fail-closed regression check ---------------------------
    try:
        crypto_ver_str = version("cryptography")
    except PackageNotFoundError:
        crypto_ver_str = None
    except Exception:
        crypto_ver_str = None

    fastmcp_present = False
    try:
        version("fastmcp")
        fastmcp_present = True
    except PackageNotFoundError:
        pass
    except Exception:
        pass

    if crypto_ver_str is not None and fastmcp_present:
        try:
            major = int(crypto_ver_str.split(".", 1)[0])
        except (ValueError, IndexError):
            major = 0
        if major >= 47:
            violations.append(
                f"  cryptography=={crypto_ver_str} installed alongside fastmcp; this breaks "
                f"every FastMCP MCP server (Authlib 1.7.0 imports the removed hazmat.backends). "
                f"Install with `pip install -c constraints.txt ...` to honor the pin. "
                f"Detail: memory/feedback_mcp_venv_drift_cryptography47.md"
            )
            return violations

    # ---- Phase 2: advisory staleness check -------------------------------
    constraints_path = PROJECT_ROOT / "constraints.txt"
    if not constraints_path.exists():
        return violations

    try:
        text = constraints_path.read_text(encoding="utf-8")
    except Exception:
        return violations

    if "cryptography<47" not in text:
        # Pin already removed — staleness signal moot.
        return violations

    revisit_iso = None
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped.startswith("revisit-by:"):
            revisit_iso = stripped.split(":", 1)[1].strip()
            break

    if revisit_iso is None:
        return violations

    try:
        from datetime import date

        revisit_date = date.fromisoformat(revisit_iso)
    except Exception:
        return violations

    today = date.today()
    if today >= revisit_date:
        days_overdue = (today - revisit_date).days
        # NOTE: print() + don't append. This check is registered with
        # is_advisory=False so Phase 1 (cryptography>=47 regression) can
        # fail-closed. If we appended to violations here, the staleness
        # signal would also block — directly contradicting the
        # "ADVISORY" wording. Surfacing via stdout matches how the
        # CRG D1-D5 checks already emit advisories.
        print(
            f"  ADVISORY: constraints.txt revisit-by:{revisit_iso} has passed "
            f"({days_overdue} day(s) overdue). Check Authlib release notes — if the "
            f"hazmat.backends import has been removed upstream, drop the cryptography<47 "
            f"pin and re-test MCPs. If still broken, bump revisit-by forward 180 days. "
            f"Detail: memory/feedback_mcp_venv_drift_cryptography47.md"
        )

    return violations


def check_garch_dependency_importable() -> list[str]:
    """Fail-closed if `arch` is not importable.

    `arch>=8.0.0` is a hard pyproject.toml dep used by
    `pipeline.build_daily_features.compute_garch_forecast`. The function
    swallows ImportError and returns None (now WARN-level), so a venv that
    drifts away from pyproject silently NULLs `garch_forecast_vol` /
    `garch_forecast_vol_pct` on every daily build until Check 65 surfaces
    the late-history NULLs.

    This drift check fails up-front so the regression is caught before any
    daily-build NULLs accumulate.

    Detail: 2026-04-29 incident — `arch` missing from canonical
    `C:\\Users\\joshd\\canompx3\\.venv` produced 1 day of NULL GARCH on
    MES/MGC/MNQ across O5/O15/O30 before Check 65 caught it.
    """
    violations: list[str] = []
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return violations

    try:
        version("arch")
    except PackageNotFoundError:
        violations.append(
            "  arch package not installed in active venv but is a hard pyproject.toml "
            "dep (>=8.0.0). Daily builds will silently NULL garch_forecast_vol on every "
            "new row. Run: pip install 'arch>=8.0.0'  (or `uv sync --frozen`)."
        )
        return violations
    except Exception as exc:
        violations.append(
            f"  arch package metadata lookup failed: {type(exc).__name__}: {exc}. "
            f"Likely venv corruption — re-run `uv sync --frozen`."
        )
        return violations

    # Belt-and-suspenders: actually try to import, in case version() succeeds
    # but the package is unimportable (broken install, partial wheel).
    try:
        import arch  # noqa: F401
    except ImportError as exc:
        violations.append(
            f"  arch importable check failed: ImportError: {exc}. Run: pip install --force-reinstall 'arch>=8.0.0'."
        )
    except Exception as exc:
        violations.append(f"  arch import raised {type(exc).__name__}: {exc}. Investigate before next daily build.")

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
                return _skip_db_check_for_ci(
                    "  check_doc_stats_consistency SKIPPED: gold.db not found "
                    "— cannot verify validated/FDR-significant counts vs docs"
                )
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


def _doc_hygiene_rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def _read_yaml_doc(path: Path):
    try:
        import yaml
    except ImportError as exc:
        return None, f"PyYAML unavailable: {exc}"

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _looks_like_python_entrypoint(value: str) -> bool:
    return value.endswith(".py") or ".py " in value


def _entrypoint_path(value: str) -> Path | None:
    entry = value.strip()
    if not _looks_like_python_entrypoint(entry):
        return None
    first_token = entry.split()[0]
    path = Path(first_token)
    return path if path.is_absolute() else PROJECT_ROOT / path


def check_doc_hygiene_contracts() -> list[str]:
    """Enforce doc truth hygiene for preregs/results and generated docs.

    This intentionally targets high-signal failure modes instead of linting all
    prose. Broad historical-doc scanning belongs in scripts/tools/stale_doc_scanner.py.
    """
    violations: list[str] = []

    # 1. Preregs/results must not carry placeholder provenance stamps.
    # Bare-word markers (always violations).
    stamp_dirs = [
        PROJECT_ROOT / "docs" / "audit" / "hypotheses",
        PROJECT_ROOT / "docs" / "audit" / "results",
    ]
    stamp_patterns = ("UNSTAMPED", "TO_BE_STAMPED")
    # commit_sha placeholder regex. Per .claude/rules/research-truth-protocol.md
    # § 2a, `commit_sha: TO_FILL_AFTER_COMMIT` is legitimate (chicken-and-egg
    # before first commit). Any OTHER placeholder value on commit_sha is rot.
    commit_sha_placeholder = re.compile(
        r"""commit_sha\s*:\s*["']?(PENDING|TO_FILL_(?!AFTER_COMMIT\b)[A-Z_]+|UNSTAMPED|TO_BE_STAMPED)["']?""",
        re.IGNORECASE,
    )
    for directory in stamp_dirs:
        if not directory.exists():
            continue
        for path in sorted((*directory.glob("*.md"), *directory.glob("*.yaml"), *directory.glob("*.yml"))):
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_num, line in enumerate(text.splitlines(), 1):
                hits = [marker for marker in stamp_patterns if marker in line]
                sha_match = commit_sha_placeholder.search(line)
                if sha_match:
                    hits.append(f"commit_sha={sha_match.group(1)}")
                if hits:
                    violations.append(
                        f"  {_doc_hygiene_rel(path)}:{line_num}: placeholder provenance marker ({', '.join(hits)})"
                    )

    # 2. Execution metadata must not imply an unavailable or fake runner.
    hyp_dir = PROJECT_ROOT / "docs" / "audit" / "hypotheses"
    if hyp_dir.exists():
        for path in sorted((*hyp_dir.glob("*.yaml"), *hyp_dir.glob("*.yml"))):
            text = path.read_text(encoding="utf-8", errors="replace")
            if "execution:" not in text:
                # Many older hypothesis files are prose-heavy YAML-ish archives.
                # Only enforce executable/design-only contracts once a prereg
                # declares the execution block this check owns.
                continue
            body, error = _read_yaml_doc(path)
            if error:
                violations.append(f"  {_doc_hygiene_rel(path)}: YAML parse failed in doc hygiene check: {error}")
                continue
            if not isinstance(body, dict):
                continue
            execution = body.get("execution")
            if not isinstance(execution, dict):
                continue
            mode = str(execution.get("mode", "")).strip().lower()
            entrypoint = execution.get("entrypoint")
            rel = _doc_hygiene_rel(path)
            if mode == "design_only":
                if entrypoint not in (None, "", "null"):
                    violations.append(
                        f"  {rel}: execution.mode=design_only but execution.entrypoint is populated ({entrypoint!r})"
                    )
                continue
            if entrypoint in (None, "", "null"):
                violations.append(f"  {rel}: executable prereg missing execution.entrypoint")
                continue
            if not isinstance(entrypoint, str):
                violations.append(f"  {rel}: execution.entrypoint must be a string or null")
                continue
            # Deferred-authoring exemption: preregs with execution_gate.allowed_now=false
            # are B1 contracts whose B2 runner script is intentionally pending. The
            # entrypoint is a forward reference, not yet load-bearing. When allowed_now
            # flips to true, this check resumes path-existence enforcement.
            execution_gate = body.get("execution_gate")
            allowed_now = execution_gate.get("allowed_now") if isinstance(execution_gate, dict) else None
            if allowed_now is False:
                continue
            entry_path = _entrypoint_path(entrypoint)
            if entry_path is not None and not entry_path.exists():
                violations.append(
                    f"  {rel}: execution.entrypoint references missing path "
                    f"{_doc_hygiene_rel(entry_path) if entry_path.is_relative_to(PROJECT_ROOT) else entry_path}"
                )

    # 3. Generated docs must name their source and block hand edits.
    generated_docs = [
        PROJECT_ROOT / "docs" / "governance" / "system_authority_map.md",
        PROJECT_ROOT / "docs" / "context" / "README.md",
        PROJECT_ROOT / "docs" / "context" / "source-catalog.md",
        PROJECT_ROOT / "docs" / "context" / "task-routes.md",
        PROJECT_ROOT / "docs" / "context" / "institutional-contracts.md",
    ]
    for path in generated_docs:
        if not path.exists():
            violations.append(f"  {_doc_hygiene_rel(path)} missing")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Generated from" not in text and "Auto-generated from" not in text:
            violations.append(f"  {_doc_hygiene_rel(path)} missing generated-source marker")
        if "Do not edit by hand" not in text and "do not edit by hand" not in text:
            violations.append(f"  {_doc_hygiene_rel(path)} missing do-not-edit marker")

    return violations


def check_stale_scratch_db() -> list[str]:
    """Check #37: Canonical gold.db must exist where pipeline.paths points.

    Resolves the canonical DB location via _get_db_path(), which honors:
      1. GOLD_DB_PATH_FOR_CHECKS module-level test override (set by tests)
      2. pipeline.paths.GOLD_DB_PATH (production), which honors:
         - DUCKDB_PATH env var (worktree / override scenario)
         - PROJECT_ROOT/gold.db (default canonical location)

    Per integrity-guardian.md rule 2 (canonical-source delegation), this
    check delegates to the canonical resolver rather than recomputing
    the path independently. Worktree scenario: f5 worktree has no local
    gold.db; DUCKDB_PATH points to canonical canompx3/gold.db.

    C:/db/gold.db scratch copy is DEPRECATED (Mar 24 2026) — caused stale-data
    decisions across terminals. No auto-sync. Use canonical only.
    """
    violations = []
    canonical_db = _get_db_path()
    if not canonical_db.exists():
        return _skip_db_check_for_ci(f"  Canonical DB not found at {canonical_db}. Run pipeline to create it.")
    scratch_db = Path("C:/db/gold.db")
    if scratch_db.exists():
        violations.append(
            "  DEPRECATED scratch DB still exists at C:/db/gold.db. "
            "Delete it to prevent stale-data bugs: del C:\\db\\gold.db"
        )
    return violations


def check_active_code_uses_canonical_db_path() -> list[str]:
    """Check: active research code must delegate DB selection to pipeline.paths."""
    violations = []
    patterns = [
        (
            re.compile(r"""\bDB_PATH\s*=\s*["']gold\.db["']"""),
            "direct DB_PATH literal",
        ),
        (
            re.compile(r"""\b(?:DB_PATH|GOLD_DB_PATH)\s*=\s*(?:ROOT|PROJECT_ROOT|REPO_ROOT)\s*/\s*["']gold\.db["']"""),
            "repo-root gold.db join",
        ),
        (
            re.compile(
                r"""os\.environ\.get\(\s*["']DUCKDB_PATH["']\s*,\s*str\((?:ROOT|PROJECT_ROOT|REPO_ROOT)\s*/\s*["']gold\.db["']\)\s*\)"""
            ),
            "manual DUCKDB_PATH fallback",
        ),
        (
            re.compile(
                r"""parser\.add_argument\(["']--db-path["'][^)]*default\s*=\s*(?:ROOT|PROJECT_ROOT|REPO_ROOT)\s*/\s*["']gold\.db["']"""
            ),
            "parser default bypasses pipeline.paths",
        ),
    ]

    for scan_dir in ACTIVE_DB_PATH_SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for py_file in sorted(scan_dir.rglob("*.py")):
            rel = py_file.relative_to(PROJECT_ROOT).as_posix()
            if rel in ACTIVE_DB_PATH_SKIP_FILES:
                continue
            if rel.startswith("research/archive/") or rel.startswith("scripts/archive/"):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            for i, line in enumerate(content.splitlines(), 1):
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                for pattern, reason in patterns:
                    if pattern.search(line):
                        violations.append(f"  {rel}:{i}: {reason} — delegate to pipeline.paths.GOLD_DB_PATH")
                        break
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
                return _skip_db_check_for_ci(
                    "  check_no_active_e3 SKIPPED: gold.db not found "
                    "— cannot verify E3 entry-model deactivation on active shelf"
                )
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
                return _skip_db_check_for_ci(
                    "  check_no_active_e2_lookahead_filters SKIPPED: gold.db not found "
                    "— cannot verify E2 look-ahead filter contamination on active shelf"
                )
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
                return _skip_db_check_for_ci(
                    "  check_active_validated_filters_routable SKIPPED: gold.db not found "
                    "— cannot verify session-aware filter routing on active shelf"
                )
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
                return _skip_db_check_for_ci(
                    "  check_active_micro_only_filters_on_real_micros SKIPPED: gold.db not found "
                    "— cannot verify MICRO-only filter routing"
                )
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
                return _skip_db_check_for_ci(
                    "  check_active_micro_only_filters_after_micro_launch SKIPPED: gold.db not found "
                    "— cannot verify MICRO-only filters' validation windows"
                )
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
        for (
            strategy_id,
            instrument,
            orb_label,
            orb_minutes,
            entry_model,
            rr_target,
            confirm_bars,
            filter_type,
        ) in micro_rows:
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
                "  validation_run_log: missing table required for native promotion provenance linkage checks."
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
                    f"  validated_setups: active native row {sid} missing promotion provenance fields: {missing}"
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


def check_hypothesis_minbtl_compliance() -> list[str]:
    """Check: every pre-reg yaml under docs/audit/hypotheses/ passes the
    Bailey 2013 MinBTL K-budget gate.

    Authority: ``pre_registered_criteria.md`` Criterion 2 (locked
    operational cap N<=300 clean / N<=2000 proxy; strict-Bailey
    horizon ``2*Ln[N]/E[max_N]^2`` must fit within the instrument's
    clean-data window).

    Delegates math to ``scripts.tools.estimate_k_budget`` (which
    delegates further to ``scripts.tools.minbtl_retro_report.strict_bailey_n``
    — single source of truth, per ``integrity-guardian.md`` § 2).

    Enforcement tiers (mirrors ``check_pooled_finding_annotations``):
    All gates are **binding for files dated >= sentinel (2026-05-12)**
    and **advisory for older files**. The sentinel matches the date this
    drift check landed; pre-existing pre-regs were written under earlier
    doctrine and are grandfathered. Editing an old file after the
    sentinel re-asserts the full gate (drift check sees current content).

    1. Operational cap (N<=300 clean / N<=2000 proxy) — Criterion 2.
    2. Bailey horizon (``2*Ln[N]/E^2 <= clean_years``) — Criterion 2.
    3. Missing trial count when instruments are declared — Criterion 1.
    4. Unknown instruments (6A/6B/6J, GC proxy) are **always advisory** —
       extending coverage requires a Criterion 2 amendment + horizons
       table update, never a silent pass.

    Pathway B K=1 files and audit-only N=0 files trivially PASS
    (ln(1)=ln(0+epsilon)=0 short-circuited in ``required_minbtl_years``).

    Silently skips files that declare neither trial count nor instrument
    scope — these are documentation stubs / templates.
    """
    from scripts.tools.estimate_k_budget import (
        CLEAN_YEARS_BY_INSTRUMENT,
        check_hypothesis_file,
        load_hypothesis,
    )

    # Date >= this string activates strict enforcement for new pre-regs.
    # Bump forward only via an explicit Criterion 2 amendment + commit;
    # never back-date to silently grandfather a new violation.
    sentinel = "2026-05-12"

    def _is_binding(path: Path) -> bool:
        return path.name[:10] >= sentinel

    violations: list[str] = []
    advisories: list[str] = []
    hyp_dir = PROJECT_ROOT / "docs" / "audit" / "hypotheses"
    if not hyp_dir.exists():
        return violations
    for path in sorted(hyp_dir.glob("*.yaml")):
        binding = _is_binding(path)
        rel = path.relative_to(PROJECT_ROOT)
        try:
            summary = load_hypothesis(path)
        except Exception as exc:
            msg = f"  {rel}: failed to parse ({type(exc).__name__}: {exc})"
            (violations if binding else advisories).append(msg)
            continue
        # Documentation stubs / templates — skip.
        if summary.n_trials is None and not summary.instruments:
            continue
        if summary.n_trials is None:
            msg = (
                f"  {rel}: declares instruments {list(summary.instruments)} "
                "but no total_expected_trials / primary_selection_trials / "
                "n_trials — Criterion 1 requires a pre-committed N."
            )
            (violations if binding else advisories).append(msg)
            continue
        # Unknown instruments — always advisory; doctrine gap to surface.
        recognized = [i for i in summary.instruments if i in CLEAN_YEARS_BY_INSTRUMENT]
        if summary.instruments and not recognized:
            advisories.append(
                f"  {rel}: instruments {list(summary.instruments)} not in "
                "CLEAN_YEARS_BY_INSTRUMENT — horizon gate skipped. "
                "Extend horizons table to enforce."
            )
            continue
        reports = check_hypothesis_file(path)
        for report in reports:
            if report.passed:
                continue
            cap_violated = report.n_trials > report.operational_cap
            horizon_violated = report.minbtl_years_required > report.clean_years
            if cap_violated:
                msg = (
                    f"  {rel} [{report.instrument}]: N={report.n_trials} "
                    f"exceeds operational cap {report.operational_cap} "
                    "(Criterion 2)."
                )
                (violations if binding else advisories).append(msg)
            elif horizon_violated:
                msg = (
                    f"  {rel} [{report.instrument}]: N={report.n_trials}, "
                    f"requires {report.minbtl_years_required:.2f}yr but "
                    f"{report.instrument} has {report.clean_years:.2f}yr "
                    f"clean data (max N={report.n_max_at_horizon} at "
                    f"E={report.e_max})."
                )
                (violations if binding else advisories).append(msg)
    if advisories:
        print(f"  WARNING (non-blocking, pre-{sentinel} grandfathered) — MinBTL advisory: {len(advisories)} pre-reg(s)")
        for a in advisories[:5]:
            print(f"    {a.strip()}")
        if len(advisories) > 5:
            print(f"    ... and {len(advisories) - 5} more")
    return violations


def check_chordia_result_threshold_matches_prereg() -> list[str]:
    """Result MD `MEASURED threshold applied` must match prereg `chordia_threshold_basis`.

    Origin: 2026-05-12 MGC LONDON_METALS Stage 1 run. The prereg's
    ``hypotheses[0].theory_citation`` field contained prose ("No filter-mechanism
    theory citation available...") which is a truthy string. The loader at
    ``trading_app/hypothesis_loader.py:262-269`` correctly emitted
    ``has_theory=True`` and the runner applied 3.00 instead of the prereg's
    declared strict 3.79. The result MD's ``MEASURED threshold applied`` line
    surfaces the mismatch but only post-run. This check catches the same class
    of authoring error at commit time, by comparing the numeric threshold
    declared in the prereg's ``chordia_threshold_basis`` string against the
    numeric ``MEASURED threshold applied`` in the matching result MD.

    The check is **paired-file** (prereg + matching result MD by stem) and
    silently passes when either file is missing — preregs without result MDs
    are pre-run state and not yet auditable. Likewise result MDs without
    prereg are out-of-scope.

    Authority: ``pre_registered_criteria.md`` Criterion 4 (Chordia threshold
    contract); ``docs/audit/results/2026-05-12-deployed-lanes-chordia-strict-379-exposure-audit.md``
    (root cause analysis).

    Date sentinel mirrors check_hypothesis_minbtl_compliance — binding for
    files dated >= 2026-05-12; advisory for older files. Grandfathered
    behaviour because pre-sentinel preregs were authored without this
    expectation.
    """
    import re

    sentinel = "2026-05-12"

    def _is_binding(stem: str) -> bool:
        return stem[:10] >= sentinel

    def _extract_prereg_threshold(prereg_text: str) -> float | None:
        """Pull the numeric threshold from chordia_threshold_basis prose.

        Matches phrasing like ``t >= 3.79`` or ``t >= 3.00``. Returns None if
        the string is absent or unparseable (silent skip — not every prereg
        carries chordia_threshold_basis; only Chordia unlock / revalidation
        preregs do).
        """
        m = re.search(
            r"chordia_threshold_basis\s*:\s*[\"']?[^\n\"']*?t\s*>=\s*(\d+\.\d+)",
            prereg_text,
        )
        if m is None:
            return None
        return float(m.group(1))

    def _extract_result_threshold(result_text: str) -> float | None:
        """Pull the numeric ``MEASURED threshold applied`` from the result MD."""
        m = re.search(
            r"\*\*MEASURED threshold applied:\*\*\s*`(\d+\.\d+)`",
            result_text,
        )
        if m is None:
            return None
        return float(m.group(1))

    violations: list[str] = []
    advisories: list[str] = []
    hyp_dir = PROJECT_ROOT / "docs" / "audit" / "hypotheses"
    res_dir = PROJECT_ROOT / "docs" / "audit" / "results"
    if not hyp_dir.exists() or not res_dir.exists():
        return violations

    for prereg_path in sorted(hyp_dir.glob("*.yaml")):
        stem = prereg_path.stem
        result_path = res_dir / f"{stem}.md"
        if not result_path.exists():
            continue  # pre-run state — no result to compare against

        try:
            prereg_text = prereg_path.read_text(encoding="utf-8")
            result_text = result_path.read_text(encoding="utf-8")
        except Exception as exc:
            advisories.append(f"  {prereg_path.name}: failed to read prereg/result pair ({type(exc).__name__}: {exc})")
            continue

        prereg_threshold = _extract_prereg_threshold(prereg_text)
        result_threshold = _extract_result_threshold(result_text)
        if prereg_threshold is None or result_threshold is None:
            # Not a Chordia-style audit (no threshold to compare). Silent skip.
            continue

        if abs(prereg_threshold - result_threshold) < 0.001:
            continue

        binding = _is_binding(stem)
        # Format with .2f so the violation message mirrors the literal phrasing
        # in the MD (e.g., "3.00" not Python's float repr "3.0"). Operator
        # greps the violation against the prereg + result file contents.
        msg = (
            f"  {prereg_path.name}: prereg declares t>={prereg_threshold:.2f} "
            f"but result MD applied t>={result_threshold:.2f}. "
            "Likely cause: hypotheses[i].theory_citation contains a truthy "
            "string for a no-theory prereg (the loader treats field-presence "
            "as the boolean). Omit theory_citation entirely for no-theory "
            "preregs, OR amend the prereg's chordia_threshold_basis if the "
            "applied threshold is the institutionally correct one. "
            "See docs/audit/results/2026-05-12-deployed-lanes-chordia-strict-379-exposure-audit.md."
        )
        (violations if binding else advisories).append(msg)

    if advisories:
        print(
            f"  WARNING (non-blocking, pre-{sentinel} grandfathered) — "
            f"Chordia threshold mismatch advisory: {len(advisories)} prereg/result pair(s)"
        )
        for a in advisories[:5]:
            print(f"    {a.strip()}")
        if len(advisories) > 5:
            print(f"    ... and {len(advisories) - 5} more")

    return violations


def check_verdict_vocabulary_md_matches_code(
    md_path: Path | None = None,
) -> list[str]:
    """Parity check: RESEARCH_RULES.md § Verdict Token Vocabulary <-> code constants.

    Source-of-truth is the Python constants ``_VERDICT_NORMALIZE`` and
    ``_VERDICT_PRIORITY_TOKENS`` in
    ``scripts/tools/research_catalog_mcp_server.py``. The doctrine MD section
    must mirror them exactly. Binding tier — divergence is a doctrine-integrity
    bug, not advisory.

    Origin: 2026-05-14 promotion of the de-facto verdict registry to
    ``RESEARCH_RULES.md``. Action-queue item
    ``research_catalog_verdict_vocabulary_doctrine_2026_05_12``.

    Parameters
    ----------
    md_path : Path | None
        Override path to RESEARCH_RULES.md (test seam). Defaults to the
        canonical project location.
    """
    import re

    violations: list[str] = []

    target = md_path if md_path is not None else PROJECT_ROOT / "RESEARCH_RULES.md"
    try:
        text = target.read_text(encoding="utf-8")
    except Exception as exc:
        violations.append(f"VERDICT_VOCAB_MD_READ_FAILED: {target} ({type(exc).__name__}: {exc})")
        return violations

    try:
        from scripts.tools.research_catalog_mcp_server import (
            _VERDICT_NORMALIZE,
            _VERDICT_PRIORITY_TOKENS,
        )
    except Exception as exc:
        violations.append(f"VERDICT_VOCAB_MODULE_IMPORT_FAILED: {type(exc).__name__}: {exc}")
        return violations

    section_re = re.compile(
        r"^##\s+Verdict Token Vocabulary\s*$(.*?)(?=^##\s+\S|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    sec_match = section_re.search(text)
    if sec_match is None:
        violations.append("VERDICT_VOCAB_SECTION_MISSING: '## Verdict Token Vocabulary' not found in RESEARCH_RULES.md")
        return violations
    section = sec_match.group(1)

    # Layout lock: only two sub-sections are permitted inside § Verdict Token
    # Vocabulary. Any additional `### ` sub-header is a doctrine-drift signal
    # (someone added a side channel that escapes the parity check).
    allowed_subsection_prefixes = ("Raw-Spelling", "Priority Resolution Order")
    for header_match in re.finditer(r"^###\s+(.+?)\s*$", section, re.MULTILINE):
        header = header_match.group(1)
        if not any(header.startswith(prefix) for prefix in allowed_subsection_prefixes):
            violations.append(
                f"VERDICT_VOCAB_UNEXPECTED_SUBSECTION: '### {header}' inside "
                f"§ Verdict Token Vocabulary; sub-headers must start with one "
                f"of {list(allowed_subsection_prefixes)}"
            )

    def _slice_table(sub_header_prefix: str) -> str | None:
        sub_re = re.compile(
            r"^###\s+" + re.escape(sub_header_prefix) + r"[^\n]*$(.*?)(?=^###\s+|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        m = sub_re.search(section)
        return m.group(1) if m is not None else None

    row_re = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", re.MULTILINE)
    sep_re = re.compile(r"^[\s\-:]+$")

    def _strip_backticks(cell: str) -> str:
        return cell.strip().strip("`").strip()

    map_slice = _slice_table("Raw-Spelling")
    if map_slice is None:
        violations.append("VERDICT_VOCAB_TABLE_MISSING: '### Raw-Spelling' not found within § Verdict Token Vocabulary")
    else:
        md_map: dict[str, str] = {}
        for raw_cell, canon_cell in row_re.findall(map_slice):
            raw, canon = raw_cell.strip(), canon_cell.strip()
            if sep_re.match(raw) or sep_re.match(canon):
                continue
            if raw.lower() == "raw spelling" and canon.lower() == "canonical tag":
                continue
            md_map[_strip_backticks(raw)] = _strip_backticks(canon)
        code_map = dict(_VERDICT_NORMALIZE)
        for raw, canonical in code_map.items():
            if raw not in md_map:
                violations.append(
                    f"VERDICT_VOCAB_MISSING_IN_MD: '{raw}' -> '{canonical}' present in code, absent in MD"
                )
            elif md_map[raw] != canonical:
                violations.append(
                    f"VERDICT_VOCAB_MAPPING_MISMATCH: '{raw}' maps to '{canonical}' in code but '{md_map[raw]}' in MD"
                )
        for raw, canonical in md_map.items():
            if raw not in code_map:
                violations.append(f"VERDICT_VOCAB_EXTRA_IN_MD: '{raw}' -> '{canonical}' present in MD, absent in code")

    prio_slice = _slice_table("Priority Resolution Order")
    if prio_slice is None:
        violations.append(
            "VERDICT_VOCAB_TABLE_MISSING: '### Priority Resolution Order' not found within § Verdict Token Vocabulary"
        )
    else:
        md_prio: list[str] = []
        for index_cell, token_cell in row_re.findall(prio_slice):
            index_s, token = index_cell.strip(), token_cell.strip()
            if sep_re.match(index_s) or sep_re.match(token):
                continue
            if index_s == "#" and token.lower() == "token":
                continue
            if not index_s.lstrip("-").isdigit():
                continue
            md_prio.append(_strip_backticks(token))
        code_prio = list(_VERDICT_PRIORITY_TOKENS)
        md_set = set(md_prio)
        code_set = set(code_prio)

        # Duplicate detection MUST run before any zip-based comparison.
        # Without this, a duplicate row in MD would pass set-equality but
        # diverge in list length — and any zip(strict=True) here would crash
        # the entire drift runner because non-DB checks are not wrapped in
        # try/except by the runner (see main() ~line 9818).
        if len(md_prio) != len(md_set):
            seen: set[str] = set()
            dupes: list[str] = []
            for tok in md_prio:
                if tok in seen and tok not in dupes:
                    dupes.append(tok)
                seen.add(tok)
            violations.append(
                f"VERDICT_PRIORITY_DUPLICATE_IN_MD: token(s) {dupes} appear more than once in MD priority list"
            )

        for tok in code_prio:
            if tok not in md_set:
                violations.append(
                    f"VERDICT_PRIORITY_MISSING_IN_MD: '{tok}' (code index "
                    f"{code_prio.index(tok) + 1}) absent in MD priority list"
                )
        for tok in md_prio:
            if tok not in code_set:
                violations.append(f"VERDICT_PRIORITY_EXTRA_IN_MD: '{tok}' present in MD priority list, absent in code")

        # Order check on the intersection of code-side and MD-side tokens —
        # walk both filtered lists in their respective relative orders and
        # flag the first position where they disagree. This surfaces order
        # bugs even when MISSING/EXTRA entries are present, so divergence
        # events emit complete information in one run instead of cascading
        # across multiple fix cycles.
        #
        # md_prio may contain duplicates (already reported above as
        # VERDICT_PRIORITY_DUPLICATE_IN_MD); dedupe-preserving-first-occurrence
        # for the order-check input so the equal-length precondition for
        # zip(strict=True) holds even on duplicate-MD inputs.
        intersection = code_set & md_set
        seen_md: set[str] = set()
        md_prio_dedup: list[str] = []
        for tok in md_prio:
            if tok in seen_md:
                continue
            seen_md.add(tok)
            md_prio_dedup.append(tok)
        common_code_order = [t for t in code_prio if t in intersection]
        common_md_order = [t for t in md_prio_dedup if t in intersection]
        # strict=True is safe here: both lists are filtered through the same
        # symmetric intersection AND md_prio is deduped before filtering, so
        # length equality holds for every input shape. Loud-crash preferred
        # over silent truncation if a future refactor breaks the invariant.
        for i, (code_tok, md_tok) in enumerate(zip(common_code_order, common_md_order, strict=True)):
            if code_tok != md_tok:
                violations.append(
                    f"VERDICT_PRIORITY_ORDER_MISMATCH: first divergence in "
                    f"shared tokens at position {i + 1}: code='{code_tok}' "
                    f"vs md='{md_tok}'"
                )
                break

    return violations


def check_checks_list_labels_are_ascii() -> list[str]:
    """All CHECKS labels must be pure ASCII (codepoints 0x20-0x7E).

    Origin: 2026-05-14. The drift runner emits each label via
    ``print(f"Check {i}: {label}...")`` from ``main()``. On Windows the
    default console encoding is cp1252 and any non-ASCII codepoint raises
    UnicodeEncodeError mid-loop, taking down the whole drift gate. The
    crash is order-sensitive — adding a check shifts indices and the FIRST
    label with a non-ASCII glyph becomes the first one to crash. Five
    labels carried glyphs (`x`/`<->`/`--` substitutions landed in the same
    commit). This guard prevents the class from regressing.

    Labels are operator-facing strings; emoji and decorative glyphs add no
    information. Substitute ASCII conventions: ``x`` for cross-product,
    ``<->`` for "and back", ``--`` for em-dash, ``->`` for arrow.
    """
    violations: list[str] = []
    checks_list = globals().get("CHECKS")
    if checks_list is None:
        # Module not fully initialized — only happens if a test imports
        # this function before CHECKS is constructed. Fail-open.
        return violations
    for label, *_ in checks_list:
        offenders = [(i, c) for i, c in enumerate(label) if ord(c) > 127]
        if offenders:
            hex_codes = ", ".join(f"0x{ord(c):x}" for _, c in offenders)
            violations.append(
                f"NON_ASCII_CHECK_LABEL: label contains codepoints "
                f"{hex_codes}; substitute ASCII (x, <->, --, ->, etc.). "
                f"Offending label (truncated): {label[:80]!r}"
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
                return _skip_db_check_for_ci(
                    "  check_orphaned_validated_strategies SKIPPED: gold.db not found "
                    "— cannot verify validated strategies have orb_outcomes coverage"
                )
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


def check_htf_levels_integrity(con=None) -> list[str]:
    """Verify all 12 prev_week_* / prev_month_* fields agree with canonical SQL.

    Canonical semantics: Monday-anchor week via DuckDB DATE_TRUNC('week', ...),
    calendar-month anchor via DATE_TRUNC('month', ...). Prior period only.

    For every row, confirms against the canonical per-period aggregate:
      prev_week_high   == MAX(daily_high)                      over prior Mon-Sun week
      prev_week_low    == MIN(daily_low)                       over prior Mon-Sun week
      prev_week_open   == arg_min(daily_open, trading_day)     over prior week
      prev_week_close  == arg_max(daily_close, trading_day)    over prior week
      prev_week_range  == ROUND(high - low, 4)
      prev_week_mid    == ROUND((high + low) / 2.0, 4)
      (same six for prev_month_*)

    Divergence classes flagged:
      - symmetric  : stored != canonical (both non-NULL)
      - phantom    : stored non-NULL but canonical IS NULL
      - stale miss : stored IS NULL but canonical non-NULL

    Rounds to 4dp for floating-point comparisons (matches Python post-pass).
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

        active_syms = ", ".join(f"'{s}'" for s in ACTIVE_ORB_INSTRUMENTS)

        # Per-field check SQL fragment: flags symmetric, phantom, and stale-miss
        # classes. NULL-safety handled explicitly (DuckDB "a != b" is NULL when
        # either side is NULL, so symmetric case needs both-non-NULL guard).
        def _diff_case(stored_col: str, canonical_col: str, label: str, round_to: int = 4) -> str:
            return (
                f"CASE "
                f"WHEN df.{stored_col} IS NOT NULL AND {canonical_col} IS NOT NULL "
                f"     AND ROUND(df.{stored_col}, {round_to}) != ROUND({canonical_col}, {round_to}) "
                f"THEN '{label}(diff)' "
                f"WHEN df.{stored_col} IS NOT NULL AND {canonical_col} IS NULL "
                f"THEN '{label}(phantom)' "
                f"WHEN df.{stored_col} IS NULL AND {canonical_col} IS NOT NULL "
                f"THEN '{label}(stale_miss)' "
                f"END AS {stored_col}"
            )

        week_fields = [
            ("prev_week_high", "w.wh"),
            ("prev_week_low", "w.wl"),
            ("prev_week_open", "w.wo"),
            ("prev_week_close", "w.wc"),
            ("prev_week_range", "w.wr"),
            ("prev_week_mid", "w.wm"),
        ]
        month_fields = [
            ("prev_month_high", "m.mh"),
            ("prev_month_low", "m.ml"),
            ("prev_month_open", "m.mo"),
            ("prev_month_close", "m.mc"),
            ("prev_month_range", "m.mr"),
            ("prev_month_mid", "m.mm"),
        ]

        diff_cases = [_diff_case(s, c, s) for s, c in week_fields + month_fields]
        diff_concat = ",\n                       ".join(diff_cases)

        # Explicit CASE-per-field SQL: each field yields a string token when
        # divergent (symmetric diff, phantom, or stale miss) or NULL when OK.
        # Python filters out the OK columns when formatting the violation line.
        explicit_sql = f"""
            WITH df AS (
                SELECT symbol,
                       trading_day,
                       daily_open, daily_high, daily_low, daily_close,
                       DATE_TRUNC('week', trading_day)::DATE  AS week_key,
                       DATE_TRUNC('month', trading_day)::DATE AS month_key,
                       prev_week_high, prev_week_low, prev_week_open,
                       prev_week_close, prev_week_range, prev_week_mid,
                       prev_month_high, prev_month_low, prev_month_open,
                       prev_month_close, prev_month_range, prev_month_mid
                FROM daily_features
                WHERE symbol IN ({active_syms})
                  AND orb_minutes = 5
            ),
            week_aggs AS (
                SELECT symbol,
                       week_key,
                       MAX(daily_high)                     AS wh,
                       MIN(daily_low)                      AS wl,
                       arg_min(daily_open, trading_day)    AS wo,
                       arg_max(daily_close, trading_day)   AS wc,
                       ROUND(MAX(daily_high) - MIN(daily_low), 4) AS wr,
                       ROUND((MAX(daily_high) + MIN(daily_low)) / 2.0, 4) AS wm
                FROM df
                WHERE daily_high IS NOT NULL AND daily_low IS NOT NULL
                GROUP BY symbol, week_key
            ),
            month_aggs AS (
                SELECT symbol,
                       month_key,
                       MAX(daily_high)                     AS mh,
                       MIN(daily_low)                      AS ml,
                       arg_min(daily_open, trading_day)    AS mo,
                       arg_max(daily_close, trading_day)   AS mc,
                       ROUND(MAX(daily_high) - MIN(daily_low), 4) AS mr,
                       ROUND((MAX(daily_high) + MIN(daily_low)) / 2.0, 4) AS mm
                FROM df
                WHERE daily_high IS NOT NULL AND daily_low IS NOT NULL
                GROUP BY symbol, month_key
            ),
            joined AS (
                SELECT df.symbol,
                       df.trading_day,
                       {diff_concat}
                FROM df
                LEFT JOIN week_aggs w
                       ON w.symbol = df.symbol
                      AND w.week_key = df.week_key - INTERVAL '7 days'
                LEFT JOIN month_aggs m
                       ON m.symbol = df.symbol
                      AND m.month_key = CASE
                            WHEN EXTRACT(MONTH FROM df.month_key) = 1
                                THEN MAKE_DATE(CAST(EXTRACT(YEAR FROM df.month_key) AS INT) - 1, 12, 1)
                            ELSE MAKE_DATE(CAST(EXTRACT(YEAR FROM df.month_key) AS INT),
                                           CAST(EXTRACT(MONTH FROM df.month_key) AS INT) - 1, 1)
                      END
            )
            SELECT symbol, trading_day,
                   COALESCE(prev_week_high, '')   AS e1,
                   COALESCE(prev_week_low, '')    AS e2,
                   COALESCE(prev_week_open, '')   AS e3,
                   COALESCE(prev_week_close, '')  AS e4,
                   COALESCE(prev_week_range, '')  AS e5,
                   COALESCE(prev_week_mid, '')    AS e6,
                   COALESCE(prev_month_high, '')  AS e7,
                   COALESCE(prev_month_low, '')   AS e8,
                   COALESCE(prev_month_open, '')  AS e9,
                   COALESCE(prev_month_close, '') AS e10,
                   COALESCE(prev_month_range, '') AS e11,
                   COALESCE(prev_month_mid, '')   AS e12
            FROM joined
            WHERE prev_week_high  IS NOT NULL
               OR prev_week_low   IS NOT NULL
               OR prev_week_open  IS NOT NULL
               OR prev_week_close IS NOT NULL
               OR prev_week_range IS NOT NULL
               OR prev_week_mid   IS NOT NULL
               OR prev_month_high IS NOT NULL
               OR prev_month_low  IS NOT NULL
               OR prev_month_open IS NOT NULL
               OR prev_month_close IS NOT NULL
               OR prev_month_range IS NOT NULL
               OR prev_month_mid   IS NOT NULL
            LIMIT 20
        """
        bad = con.execute(explicit_sql).fetchall()
        for row in bad:
            sym, td = row[0], row[1]
            diverged = [msg for msg in row[2:] if msg and msg != ""]
            violations.append(f"  {sym} {td}: HTF level divergence → {'; '.join(diverged)}")
    except (ImportError, OSError) as e:
        print(f"    SKIP check_htf_levels_integrity: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_htf_aperture_consistency(con=None) -> list[str]:
    """Verify HTF fields are identical across apertures for each day/symbol.

    ``prev_week_*`` / ``prev_month_*`` are derived from completed higher-timeframe
    daily OHLC aggregates. They are orb-agnostic by definition, so a single
    ``(symbol, trading_day)`` must never carry different HTF values across the
    O5/O15/O30 duplicate rows. This catches partial stale-miss / partial repair
    states that can survive when one aperture is rebuilt or repaired and a
    sibling aperture is left untouched.
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

        active_syms = ", ".join(f"'{s}'" for s in ACTIVE_ORB_INSTRUMENTS)
        htf_cols = [
            "prev_week_high",
            "prev_week_low",
            "prev_week_open",
            "prev_week_close",
            "prev_week_range",
            "prev_week_mid",
            "prev_month_high",
            "prev_month_low",
            "prev_month_open",
            "prev_month_close",
            "prev_month_range",
            "prev_month_mid",
        ]

        diff_exprs = ",\n                       ".join(
            [
                (
                    "CASE WHEN COUNT(DISTINCT COALESCE(CAST("
                    f"{col} AS VARCHAR), '__NULL__')) > 1 THEN '{col}(aperture_diff)' END AS {col}"
                )
                for col in htf_cols
            ]
        )

        sql = f"""
            WITH grouped AS (
                SELECT symbol,
                       trading_day,
                       {diff_exprs}
                FROM daily_features
                WHERE symbol IN ({active_syms})
                GROUP BY symbol, trading_day
                HAVING COUNT(*) > 1
            )
            SELECT symbol,
                   trading_day,
                   COALESCE(prev_week_high, '')   AS e1,
                   COALESCE(prev_week_low, '')    AS e2,
                   COALESCE(prev_week_open, '')   AS e3,
                   COALESCE(prev_week_close, '')  AS e4,
                   COALESCE(prev_week_range, '')  AS e5,
                   COALESCE(prev_week_mid, '')    AS e6,
                   COALESCE(prev_month_high, '')  AS e7,
                   COALESCE(prev_month_low, '')   AS e8,
                   COALESCE(prev_month_open, '')  AS e9,
                   COALESCE(prev_month_close, '') AS e10,
                   COALESCE(prev_month_range, '') AS e11,
                   COALESCE(prev_month_mid, '')   AS e12
            FROM grouped
            WHERE prev_week_high  IS NOT NULL
               OR prev_week_low   IS NOT NULL
               OR prev_week_open  IS NOT NULL
               OR prev_week_close IS NOT NULL
               OR prev_week_range IS NOT NULL
               OR prev_week_mid   IS NOT NULL
               OR prev_month_high IS NOT NULL
               OR prev_month_low  IS NOT NULL
               OR prev_month_open IS NOT NULL
               OR prev_month_close IS NOT NULL
               OR prev_month_range IS NOT NULL
               OR prev_month_mid   IS NOT NULL
            LIMIT 20
        """
        bad = con.execute(sql).fetchall()
        for row in bad:
            sym, td = row[0], row[1]
            diverged = [msg for msg in row[2:] if msg and msg != ""]
            violations.append(f"  {sym} {td}: HTF aperture divergence → {'; '.join(diverged)}")
    except (ImportError, OSError) as e:
        print(f"    SKIP check_htf_aperture_consistency: {type(e).__name__}: {e}")
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


def check_recent_garch_feature_coverage(con=None) -> list[str]:
    """Fail if late-history GARCH state goes NULL on active instruments/apertures.

    `garch_forecast_vol` and `garch_forecast_vol_pct` are rolling prior-only
    features. Once a symbol × aperture partition has well past the warm-up
    period, recent rows should not revert to NULL unless the feature builder
    lost prior seed history or ordering.

    This is intentionally scoped to late-series recent rows only:
    early-history warm-up NULLs are legitimate; sudden recent NULLs are not.
    """
    violations = []
    _own_con = False
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from pipeline.build_daily_features import GARCH_MIN_PRIOR_CLOSES, GARCH_PCT_MIN_PRIOR_VALUES

        recent_rows = 20
        min_total_rows = GARCH_MIN_PRIOR_CLOSES + GARCH_PCT_MIN_PRIOR_VALUES + recent_rows

        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return violations
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        active_syms = ", ".join(f"'{s}'" for s in ACTIVE_ORB_INSTRUMENTS)
        rows = con.execute(
            f"""
            WITH ranked AS (
                SELECT symbol,
                       orb_minutes,
                       trading_day,
                       garch_forecast_vol,
                       garch_forecast_vol_pct,
                       ROW_NUMBER() OVER (
                           PARTITION BY symbol, orb_minutes
                           ORDER BY trading_day DESC
                       ) AS rn_recent,
                       COUNT(*) OVER (
                           PARTITION BY symbol, orb_minutes
                       ) AS n_total
                FROM daily_features
                WHERE symbol IN ({active_syms})
            )
            SELECT symbol,
                   orb_minutes,
                   MIN(trading_day) AS first_bad_day,
                   MAX(trading_day) AS last_bad_day,
                   COUNT(*) AS bad_rows
            FROM ranked
            WHERE n_total >= {min_total_rows}
              AND rn_recent <= {recent_rows}
              AND (garch_forecast_vol IS NULL OR garch_forecast_vol_pct IS NULL)
            GROUP BY symbol, orb_minutes
            ORDER BY symbol, orb_minutes
            """
        ).fetchall()
        for sym, orb_minutes, first_bad_day, last_bad_day, bad_rows in rows:
            violations.append(
                f"  {sym} O{orb_minutes}: recent late-history GARCH coverage has {bad_rows} NULL row(s) "
                f"from {first_bad_day} to {last_bad_day}. Rolling post-pass state likely lost seed history "
                f"or ordering."
            )
    except (ImportError, OSError) as e:
        print(f"    SKIP check_recent_garch_feature_coverage: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_family_rr_locks_coverage(con=None) -> list[str]:
    """Every active instrument must have family_rr_locks rows covering its validated strategies."""
    errors = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return _skip_db_check_for_ci(
                    "  FAMILY RR LOCKS SKIPPED: gold.db not found — cannot verify lock coverage"
                )
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
            return _skip_db_check_for_ci("  FAMILY RR LOCKS SKIPPED: family_rr_locks table not present in DB")

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
# ML-layer drift checks removed 2026-04-11 (ML V1/V2/V3 DEAD — V3 sprint Stage 4).
# These previously validated trading_app/ml/ subsystem invariants:
#   check_ml_evaluate_hybrid_support, check_ml_bundle_full_delta,
#   check_ml_sharpe_jk_pvalue, check_ml_no_iterrows_filters.
# All removed with the ML subsystem. See docs/audit/hypotheses/
# 2026-04-11-ml-v3-pooled-confluence-postmortem.md for the terminal verdict.
# =============================================================================


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
    """Deprecated 2026-04-11 — previously verified pipeline.session_guard._SESSION_ORDER
    against trading_app.ml.config.SESSION_CHRONOLOGICAL_ORDER. The ML subsystem was
    removed in the V3 sprint Stage 4 (V1/V2/V3 all DEAD). session_guard now stands
    alone as the canonical chronological ordering source. Retained as a no-op for
    registry stability."""
    return []


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
                return _skip_db_check_for_ci(
                    "  HOLDOUT CHECK SKIPPED: gold.db not found — cannot verify holdout integrity"
                )
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
            f"  HOLDOUT POLICY CANONICAL SOURCE MISSING: {e}. Expected trading_app/holdout_policy.py per Amendment 2.7."
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


def check_prereg_present_for_recent_runs(con=None) -> list[str]:
    """Pre-registration file presence for post-Phase-0 discovery runs.

    Authority: docs/institutional/pre_registered_criteria.md Criterion 1
    (line 42-53). Every discovery run that writes to validated_setups must
    have a corresponding pre-registered hypothesis file at
    ``docs/audit/hypotheses/YYYY-MM-DD-<instrument>-*.yaml``.

    Definition of "recent run" for this check:
      * an experimental_strategies row with ``created_at >
        HOLDOUT_GRANDFATHER_CUTOFF`` exists,
      * its ``CAST(created_at AS DATE)`` has no matching prereg yaml on disk.

    Legacy carve-out: rows created at or before the Amendment 2.7 grandfather
    cutoff (currently 2026-04-08 UTC) are exempt -- they pre-date the
    prereg-discipline regime.

    Reports per (instrument, discovery_date) combinations missing a prereg
    file. Fail-closed: missing DB returns SKIPPED (not silent pass).
    """
    violations: list[str] = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return _skip_db_check_for_ci(
                    "  PREREG CHECK SKIPPED: gold.db not found -- cannot verify Criterion 1 compliance"
                )
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from trading_app.holdout_policy import HOLDOUT_GRANDFATHER_CUTOFF

        hyp_dir = PROJECT_ROOT / "docs" / "audit" / "hypotheses"
        if not hyp_dir.exists():
            violations.append(
                f"  PREREG: hypotheses dir missing at {hyp_dir.relative_to(PROJECT_ROOT)} -- "
                "Criterion 1 cannot be enforced. Authority: pre_registered_criteria.md."
            )
            return violations

        for instrument in ACTIVE_ORB_INSTRUMENTS:
            instr_lower = instrument.lower()
            rows = con.execute(
                """SELECT DISTINCT CAST(created_at AS DATE) AS d
                   FROM experimental_strategies
                   WHERE instrument = ?
                   AND created_at > ?
                   ORDER BY d""",
                [instrument, HOLDOUT_GRANDFATHER_CUTOFF],
            ).fetchall()

            for (d,) in rows:
                if d is None:
                    continue
                ds = d.isoformat() if hasattr(d, "isoformat") else str(d)
                if not list(hyp_dir.glob(f"{ds}-{instr_lower}-*.yaml")):
                    violations.append(
                        f"  PREREG MISSING: {instrument} discovery on {ds} has no "
                        f"prereg yaml at docs/audit/hypotheses/{ds}-{instr_lower}-*.yaml. "
                        f"Authority: pre_registered_criteria.md Criterion 1."
                    )
    except (ImportError, OSError) as e:
        violations.append(f"  PREREG CHECK FAILED: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()

    return violations


def check_validator_pool_freshness(con=None, drift_threshold: float = 0.10) -> list[str]:
    """Report drift between a row's frozen ``discovery_k`` and live per-session pool.

    Authority chain:
    - ``trading_app/strategy_validator.py:2216-2219`` -- ``discovery_k`` is
      written ONLY on first promotion (UPDATE...CASE WHEN discovery_k IS NULL).
    - ``docs/institutional/pre_registered_criteria.md`` Criterion 3 (BH-FDR
      stratified per session). The frozen K is the audit-trail anchor; live
      pool size shifts as MNQ/MES/MGC discovery rewrites peer rows.

    Behavior: for every promoted row in ``validated_setups`` written in the
    last 7 days, recompute the live per-session pool size from
    ``experimental_strategies`` and compare to the frozen ``discovery_k``.
    Report (advisory, do not fail) any drift > ``drift_threshold`` (default
    10%). Surfaces silent K mutation across instrument-discovery cross-runs.

    Fail-closed: missing DB returns SKIPPED.
    """
    violations: list[str] = []
    _own_con = False
    try:
        if con is None:
            import duckdb

            db_path = _get_db_path()
            if not db_path.exists():
                return _skip_db_check_for_ci("  POOL-FRESHNESS CHECK SKIPPED: gold.db not found")
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        # Recent promotions (last 7 days) with frozen discovery_k.
        rows = con.execute(
            """SELECT strategy_id, instrument, orb_label, discovery_k
               FROM validated_setups
               WHERE discovery_k IS NOT NULL
               AND promoted_at IS NOT NULL
               AND promoted_at >= now() - INTERVAL 7 DAY
               ORDER BY promoted_at DESC"""
        ).fetchall()

        if not rows:
            return violations

        # Live per-session pool sizes for the instruments present.
        instrument_pools: dict[tuple[str, str], int] = {}
        live_query = con.execute(
            """SELECT instrument, orb_label, COUNT(*) AS k
               FROM experimental_strategies
               WHERE is_canonical = TRUE
               AND p_value IS NOT NULL
               GROUP BY instrument, orb_label"""
        ).fetchall()
        for instr_, orb_, k_ in live_query:
            instrument_pools[(instr_, orb_)] = int(k_)

        for sid, instr, orb, frozen_k in rows:
            live_k = instrument_pools.get((instr, orb))
            if live_k is None or frozen_k is None or frozen_k <= 0:
                continue
            drift = abs(live_k - frozen_k) / float(frozen_k)
            if drift > drift_threshold:
                violations.append(
                    f"  POOL-FRESHNESS: {sid} frozen discovery_k={frozen_k}, "
                    f"live K={live_k} (drift {drift:.1%}). Frozen K is the "
                    "audit-trail anchor; live drift is informational. "
                    "Authority: pre_registered_criteria.md Criterion 3."
                )
    except (ImportError, OSError) as e:
        violations.append(f"  POOL-FRESHNESS CHECK FAILED: {type(e).__name__}: {e}")
    finally:
        if _own_con and con is not None:
            con.close()

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
    _VALID_CATEGORIES = frozenset(
        {
            "PRE_SESSION",
            "INTRA_SESSION",
            "OVERLAY",
            "DIRECTIONAL",
        }
    )
    _VALID_RESOLVES_AT = frozenset(
        {
            "STARTUP",
            "ORB_FORMATION",
            "BREAK_DETECTED",
            "CONFIRM_COMPLETE",
            "TRADE_ENTERED",
        }
    )
    _VALID_CONFIDENCE_TIERS = frozenset(
        {
            "PROVEN",
            "PLAUSIBLE",
            "LEGACY",
            "UNKNOWN",
        }
    )
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
                violations.append(f"  {filter_type} ({cls_name}): describe() raised {type(exc).__name__}: {exc}")
                continue

            if not isinstance(atoms, list):
                violations.append(
                    f"  {filter_type} ({cls_name}): describe() returned {type(atoms).__name__}, expected list"
                )
                continue

            # NoFilter is the only filter allowed zero atoms.
            if not atoms and not isinstance(leaf, NoFilter):
                violations.append(
                    f"  {filter_type} ({cls_name}): describe() returned empty list — only NoFilter may have zero atoms"
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
                    violations.append(f"{rel_path}:{line_no}: @canonical-source points to missing file: {ref}")
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
        ("if orb_end_utc is not None else break_ts", "silent fallback to break_ts — re-creates Stage 5 lookahead bias"),
        ("orb_end_utc or break_ts", "shorthand silent fallback to break_ts — re-creates Stage 5 lookahead bias"),
        (
            "= break_ts - timedelta(minutes=break_delay)",
            "L782-style derivation from break_delay_min — re-creates Stage 5 lookahead bias",
        ),
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
            violations.append(f"  {rel}:{line_no} — forbidden pattern '{needle}': {reason}")
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


_SHA_MIGRATION_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "audit" / "check_107_sha_migrations.yaml"
)


def _load_sha_migration_manifest(
    manifest_path: Path | None = None,
) -> tuple[set[str], list[str], list[dict]]:
    """Load the SHA migration manifest. Returns (accepted_orphan_shas, errors, entries).

    The manifest at ``docs/audit/check_107_sha_migrations.yaml`` records
    orphan SHAs that have been verified as legitimate re-stamping artifacts
    (e.g., a doctrine migration like Amendment 3.3 that edited the
    hypothesis YAML in place after discovery). Each entry must carry:

    - ``orphan_sha``: 64-char hex; the stale SHA still stamped in experimental_strategies
    - ``current_file``: path to the file whose content used to hash to that SHA
    - ``current_sha``: the file's current on-disk SHA (informational)
    - ``migration_commit``: git commit that re-stamped the file
    - ``introducing_commit``: git commit at which the file content hashed to ``orphan_sha``
    - ``rationale``: short prose justification
    - ``audit_ref``: path to the audit MD that documents this entry

    Fail-closed on syntax errors: returns empty set + error list, so
    Check 107 keeps flagging the underlying orphans. Semantic verification
    (commits exist, blob SHA matches the orphan, etc.) is performed by
    ``check_phase_4_sha_migration_manifest_integrity``.
    """
    path = manifest_path if manifest_path is not None else _SHA_MIGRATION_MANIFEST_PATH
    errors: list[str] = []
    accepted: set[str] = set()
    entries_out: list[dict] = []

    if not path.exists():
        return accepted, errors, entries_out

    try:
        import yaml

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:
        errors.append(f"SHA MIGRATION MANIFEST PARSE ERROR: {path} — {exc}")
        return accepted, errors, entries_out

    if not isinstance(data, dict):
        errors.append(f"SHA MIGRATION MANIFEST: {path} root is not a mapping")
        return accepted, errors, entries_out

    entries = data.get("entries", [])
    if not isinstance(entries, list):
        errors.append(f"SHA MIGRATION MANIFEST: {path} `entries` is not a list")
        return accepted, errors, entries_out

    required = {
        "orphan_sha",
        "current_file",
        "current_sha",
        "migration_commit",
        "introducing_commit",
        "rationale",
        "audit_ref",
    }
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"SHA MIGRATION MANIFEST: entry[{i}] is not a mapping")
            continue
        missing = required - set(entry.keys())
        if missing:
            errors.append(f"SHA MIGRATION MANIFEST: entry[{i}] missing fields: {sorted(missing)}")
            continue
        sha = entry["orphan_sha"]
        if not isinstance(sha, str) or len(sha) != 64:
            errors.append(f"SHA MIGRATION MANIFEST: entry[{i}] orphan_sha is not a 64-char hex string")
            continue
        accepted.add(sha)
        entries_out.append(entry)

    return accepted, errors, entries_out


def check_phase_4_sha_integrity(con=None) -> list[str]:
    """Verify every stamped hypothesis_file_sha references a real file on disk.

    Phase 4 Stage 4.1 (2026-04-08) adds discovery-side SHA stamping: every
    experimental_strategies row written by ``run_discovery`` with
    ``hypothesis_file=<Path>`` carries the content SHA of the hypothesis
    YAML in ``experimental_strategies.hypothesis_file_sha``. This check is
    the INTEGRITY guard: any row whose SHA does NOT resolve to a real file
    in ``docs/audit/hypotheses/`` indicates either (a) tampering (the file
    was deleted after the run), (b) a rebase that dropped the hypothesis
    commit, (c) a test-fixture leak into gold.db, or (d) a legitimate
    in-place edit of the hypothesis YAML after discovery (e.g., bulk
    doctrine migration). Class (d) is recorded in
    ``docs/audit/check_107_sha_migrations.yaml`` and accepted by this
    check via ``_load_sha_migration_manifest``; the sibling check
    ``check_phase_4_sha_migration_manifest_integrity`` proves each
    manifest entry is well-formed and evidence-grounded.

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
                return _skip_db_check_for_ci(
                    "  PHASE 4 SHA INTEGRITY SKIPPED: gold.db not found — cannot verify hypothesis file SHA integrity"  # noqa: E501
                )
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        from trading_app.holdout_policy import PHASE_4_1_SHIP_DATE
        from trading_app.hypothesis_loader import find_hypothesis_file_by_sha

        # Migration-manifest-accepted orphan SHAs (class (d) in the docstring).
        # Manifest parse errors propagate as violations so a malformed
        # manifest never silently grants acceptance.
        accepted_orphans, manifest_errors, _entries = _load_sha_migration_manifest()
        for err in manifest_errors:
            violations.append(f"  PHASE 4 SHA INTEGRITY: {err}")

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
            if resolved is not None:
                continue
            if sha in accepted_orphans:
                # Verified migration artifact — not a violation. The sibling
                # check ``check_phase_4_sha_migration_manifest_integrity``
                # proves the manifest entry is evidence-grounded.
                continue
            # Count rows affected for operator context.
            row_count = con.execute(
                """SELECT COUNT(*) FROM experimental_strategies
                   WHERE hypothesis_file_sha = ?""",
                [sha],
            ).fetchone()[0]
            violations.append(
                f"  PHASE 4 SHA INTEGRITY: orphaned SHA {sha[:12]}... "
                f"({row_count} row(s)) — no file in "
                f"docs/audit/hypotheses/ matches this SHA, and the SHA is "
                f"not recorded in docs/audit/check_107_sha_migrations.yaml. "
                f"Likely causes: (a) hypothesis file deleted/rebased after "
                f"discovery, (b) test fixture leaked into gold.db, or (d) "
                f"legitimate in-place edit — if (d), add an entry to the "
                f"migration manifest with introducing_commit/migration_commit. "
                f"Investigate via: SELECT strategy_id, created_at FROM "
                f"experimental_strategies WHERE hypothesis_file_sha = "
                f"'{sha[:12]}...' LIMIT 5;"
            )
    except Exception as exc:
        violations.append(f"  PHASE 4 SHA INTEGRITY CHECK FAILED: {exc}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_phase_4_sha_migration_manifest_integrity(
    manifest_path: Path | None = None,
) -> list[str]:
    """Verify ``docs/audit/check_107_sha_migrations.yaml`` is evidence-grounded.

    The manifest grants Check 107 acceptance to orphan SHAs that are
    legitimate re-stamping artifacts. Without this sibling check, a
    malformed or fabricated entry could silence the orphan detector
    without proof. This check fails closed on each:

    1. ``current_file`` must exist on disk.
    2. ``current_sha`` must equal the file's current SHA-256 (informational
       field; recomputed here for parity).
    3. ``migration_commit`` and ``introducing_commit`` must exist as
       resolvable git revisions.
    4. ``migration_commit`` must have touched ``current_file``.
    5. The blob SHA-256 of ``current_file`` at ``introducing_commit``
       must equal ``orphan_sha`` (the entry's central claim).
    6. ``audit_ref`` must exist on disk.

    No fallbacks. Each failed assertion is one violation line. Manifest
    parse-level errors (already surfaced by Check 107) are NOT
    re-reported here; this check only runs the semantic checks on
    well-formed entries.
    """
    violations: list[str] = []
    import subprocess

    _accepted, parse_errors, entries = _load_sha_migration_manifest(manifest_path)
    if parse_errors and not entries:
        # Manifest unparseable — Check 107 already surfaced the parse error.
        return violations

    repo_root = Path(__file__).resolve().parent.parent

    def _git(args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            cwd=str(repo_root),
        )

    for i, entry in enumerate(entries):
        orphan = entry["orphan_sha"]
        rel = entry["current_file"]
        declared_current = entry["current_sha"]
        mig = entry["migration_commit"]
        intro = entry["introducing_commit"]
        audit_ref = entry["audit_ref"]

        file_path = (repo_root / rel).resolve()
        if not file_path.exists():
            violations.append(f"  SHA MIGRATION MANIFEST: entry[{i}] current_file does not exist on disk: {rel}")
            continue

        try:
            import hashlib

            on_disk_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
        except Exception as exc:
            violations.append(f"  SHA MIGRATION MANIFEST: entry[{i}] failed to read {rel}: {exc}")
            continue

        if on_disk_sha != declared_current:
            violations.append(
                f"  SHA MIGRATION MANIFEST: entry[{i}] current_sha mismatch — "
                f"manifest says {declared_current[:12]}..., on-disk is {on_disk_sha[:12]}... "
                f"for {rel}"
            )

        for label, commit in (("migration_commit", mig), ("introducing_commit", intro)):
            result = _git(["cat-file", "-e", commit])
            if result.returncode != 0:
                violations.append(
                    f"  SHA MIGRATION MANIFEST: entry[{i}] {label} {commit[:12]}... does not exist in this repository"
                )

        # migration_commit must touch current_file. Use --name-only on the
        # commit and check for the path in the changed-file list.
        result = _git(["show", "--name-only", "--pretty=format:", mig])
        if result.returncode == 0:
            touched = {
                line.strip() for line in result.stdout.decode("utf-8", errors="replace").splitlines() if line.strip()
            }
            if rel not in touched:
                violations.append(
                    f"  SHA MIGRATION MANIFEST: entry[{i}] migration_commit {mig[:12]}... "
                    f"did not touch {rel} — the cited migration cannot be the cause "
                    f"of this orphan."
                )

        # introducing_commit's blob SHA-256 of current_file must equal orphan_sha.
        result = _git(["show", f"{intro}:{rel}"])
        if result.returncode != 0:
            violations.append(
                f"  SHA MIGRATION MANIFEST: entry[{i}] cannot read {rel} at "
                f"introducing_commit {intro[:12]}... — git show failed."
            )
        else:
            import hashlib as _hashlib

            blob_sha = _hashlib.sha256(result.stdout).hexdigest()
            if blob_sha != orphan:
                violations.append(
                    f"  SHA MIGRATION MANIFEST: entry[{i}] orphan_sha {orphan[:12]}... "
                    f"does not match the SHA-256 of {rel} at introducing_commit "
                    f"{intro[:12]}... (got {blob_sha[:12]}...). The entry's central "
                    f"claim is FALSE."
                )

        audit_path = (repo_root / audit_ref).resolve()
        if not audit_path.exists():
            violations.append(f"  SHA MIGRATION MANIFEST: entry[{i}] audit_ref does not exist on disk: {audit_ref}")

    return violations


def check_prop_profiles_validated_alignment(con=None) -> list[str]:
    """Every DailyLaneSpec in an ``active=True`` AccountProfile must exist in
    the deployable validated shelf.

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
                return _skip_db_check_for_ci(
                    "  PROP PROFILES ALIGNMENT SKIPPED: gold.db not found — "
                    "cannot verify deployed lane validation backing"
                )
            con = duckdb.connect(str(db_path), read_only=True)
            _own_con = True

        from trading_app.prop_profiles import ACCOUNT_PROFILES
        from trading_app.validated_shelf import validated_setups_has_deployment_scope

        has_scope = validated_setups_has_deployment_scope(con)
        lane_query = (
            "SELECT status, LOWER(COALESCE(deployment_scope, 'deployable')) FROM validated_setups WHERE strategy_id = ?"
            if has_scope
            else "SELECT status, NULL FROM validated_setups WHERE strategy_id = ?"
        )

        from trading_app.prop_profiles import load_allocation_lanes

        for profile_id, profile in sorted(ACCOUNT_PROFILES.items()):
            if not profile.active:
                continue
            lanes_to_check = profile.daily_lanes
            if not lanes_to_check:
                lanes_to_check = load_allocation_lanes(profile_id)
            for lane in lanes_to_check:
                row = con.execute(
                    lane_query,
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
                status, deployment_scope = row
                if status != "active":
                    violations.append(
                        f"  prop_profiles.{profile_id}: lane "
                        f"{lane.strategy_id!r} exists in validated_setups "
                        f"but status={status!r} (not active). A retired or "
                        f"suspended strategy cannot back a live lane."
                    )
                    continue
                if has_scope and deployment_scope != "deployable":
                    violations.append(
                        f"  prop_profiles.{profile_id}: lane "
                        f"{lane.strategy_id!r} exists with deployment_scope="
                        f"{deployment_scope!r} (not deployable). Live lanes "
                        f"must point at deployable shelf rows only."
                    )
    except Exception as exc:
        violations.append(f"  PROP PROFILES ALIGNMENT CHECK FAILED: {exc}")
    finally:
        if _own_con and con is not None:
            con.close()
    return violations


def check_account_profiles_declare_is_express_funded() -> list[str]:
    """Every ACCOUNT_PROFILES entry in trading_app/prop_profiles.py must
    explicitly declare ``is_express_funded=`` rather than rely on the
    dataclass default.

    Rationale: AccountProfile.is_express_funded defaults to False (fail-closed
    after 2026-05-18 — see telemetry_maturity.py module docstring DOCTRINE
    NOTE). A new profile that forgets the field is silently classified as
    real-capital, so safety gates (e.g. telemetry-maturity preflight) refuse
    to demote to advisory WARN. That default is the safe direction, but only
    works if every profile entry forces the author to think about the field
    explicitly. This drift check asserts each AccountProfile(...) call inside
    the ACCOUNT_PROFILES dict literal contains the keyword.

    Fail mode: AST-parses prop_profiles.py, walks the ACCOUNT_PROFILES
    assignment, and reports any AccountProfile(...) call that omits
    ``is_express_funded=``.

    @canonical-source: trading_app/prop_profiles.py
    """
    import ast

    violations: list[str] = []
    target = TRADING_APP_DIR / "prop_profiles.py"
    if not target.exists():
        return [f"  prop_profiles.py not found at {target} — drift check cannot run"]

    try:
        tree = ast.parse(target.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"  prop_profiles.py parse failed: {exc}"]

    profiles_dict: ast.Dict | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "ACCOUNT_PROFILES":
            if isinstance(node.value, ast.Dict):
                profiles_dict = node.value
                break
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "ACCOUNT_PROFILES" and isinstance(node.value, ast.Dict):
                    profiles_dict = node.value
                    break
            if profiles_dict is not None:
                break

    if profiles_dict is None:
        return [
            "  ACCOUNT_PROFILES dict literal not found in prop_profiles.py — "
            "drift check cannot verify is_express_funded declarations"
        ]

    for key_node, value_node in zip(profiles_dict.keys, profiles_dict.values, strict=False):
        if not isinstance(value_node, ast.Call):
            continue
        if not (isinstance(value_node.func, ast.Name) and value_node.func.id == "AccountProfile"):
            continue
        declared = any(kw.arg == "is_express_funded" for kw in value_node.keywords)
        if declared:
            continue
        profile_id = "<unknown>"
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            profile_id = key_node.value
        else:
            for kw in value_node.keywords:
                if kw.arg == "profile_id" and isinstance(kw.value, ast.Constant):
                    profile_id = str(kw.value.value)
                    break
        violations.append(
            f"  prop_profiles.ACCOUNT_PROFILES[{profile_id!r}]: AccountProfile(...) "
            f"omits is_express_funded=. Default is False (fail-closed real-capital); "
            f"declare True only for Express-Funded prop accounts (Topstep XFA) where "
            f"the funded-account wrapper insulates real capital."
        )
    return violations


def check_validated_setups_writer_allowlist() -> list[str]:
    """Only canonical validator/maintenance paths may mutate validated_setups."""
    violations: list[str] = []
    allowed_exact = {
        Path("trading_app/db_manager.py"),
        Path("trading_app/strategy_validator.py"),
        Path("trading_app/edge_families.py"),
        Path("scripts/infra/parallel_rebuild.py"),
        Path("scripts/infra/revalidate_null_seeds.py"),
        Path("scripts/tools/backfill_dollar_columns.py"),
        Path("scripts/tools/backfill_deployability_evidence.py"),
        Path("scripts/tools/migrate_fairness_audit.py"),
    }
    allowed_prefixes = (Path("scripts/migrations"),)
    write_pattern = re.compile(
        r"\b(?:INSERT(?:\s+OR\s+REPLACE)?\s+INTO|UPDATE|DELETE\s+FROM)\s+validated_setups\b",
        re.IGNORECASE,
    )

    for base_dir in (TRADING_APP_DIR, SCRIPTS_DIR):
        if not base_dir.exists():
            continue
        for fpath in sorted(base_dir.rglob("*.py")):
            rel = fpath.relative_to(PROJECT_ROOT)
            if rel in allowed_exact or any(str(rel).startswith(str(prefix)) for prefix in allowed_prefixes):
                continue
            content = fpath.read_text(encoding="utf-8")
            for idx, line in enumerate(content.splitlines(), 1):
                if write_pattern.search(line):
                    violations.append(f"  {rel}:{idx}: writes validated_setups outside canonical allowlist")

    return violations


def check_critical_deployable_shelf_consumers() -> list[str]:
    """Critical production readers must encode deployable-shelf semantics canonically."""
    violations: list[str] = []
    critical_files = [
        PIPELINE_DIR / "dashboard.py",
        TRADING_APP_DIR / "live_config.py",
        TRADING_APP_DIR / "prop_portfolio.py",
        TRADING_APP_DIR / "lane_allocator.py",
        TRADING_APP_DIR / "portfolio.py",
        TRADING_APP_DIR / "pbo.py",
        TRADING_APP_DIR / "edge_families.py",
        TRADING_APP_DIR / "strategy_fitness.py",
        TRADING_APP_DIR / "sr_monitor.py",
        TRADING_APP_DIR / "sprt_monitor.py",
        TRADING_APP_DIR / "view_strategies.py",
        SCRIPTS_DIR / "tools" / "backtest_allocator.py",
        SCRIPTS_DIR / "tools" / "build_optimal_profiles.py",
        SCRIPTS_DIR / "tools" / "forward_monitor.py",
        SCRIPTS_DIR / "tools" / "generate_profile_lanes.py",
        SCRIPTS_DIR / "tools" / "generate_promotion_candidates.py",
        SCRIPTS_DIR / "tools" / "generate_trade_sheet.py",
        SCRIPTS_DIR / "tools" / "optimal_lanes.py",
        SCRIPTS_DIR / "tools" / "pipeline_status.py",
        SCRIPTS_DIR / "tools" / "project_pulse.py",
        SCRIPTS_DIR / "tools" / "rolling_portfolio_assembly.py",
        SCRIPTS_DIR / "tools" / "score_lanes.py",
        SCRIPTS_DIR / "tools" / "select_family_rr.py",
        TRADING_APP_DIR / "ai" / "sql_adapter.py",
    ]
    relation_required = {
        fpath.relative_to(PROJECT_ROOT)
        for fpath in critical_files
        if fpath != TRADING_APP_DIR / "ai" / "sql_adapter.py"
    }
    raw_active_pattern = re.compile(r"(?:LOWER\([^)]*status\)|status)\s*=\s*'active'", re.IGNORECASE)

    for fpath in critical_files:
        if not fpath.exists():
            violations.append(f"  {fpath.relative_to(PROJECT_ROOT)} missing")
            continue
        rel = fpath.relative_to(PROJECT_ROOT)
        content = fpath.read_text(encoding="utf-8")
        uses_relation_helper = "deployable_validated_relation" in content or "active_validated_relation" in content
        uses_explicit_scope = "deployment_scope" in content
        if rel in relation_required and not uses_relation_helper:
            violations.append(f"  {rel}: critical validated_setups reader must use published relation helper")
        elif rel == Path("trading_app/ai/sql_adapter.py") and not (uses_relation_helper or uses_explicit_scope):
            violations.append(f"  {rel}: critical validated_setups reader lacks canonical deployable-shelf semantics")
        if rel != Path("trading_app/ai/sql_adapter.py"):
            lines = content.splitlines()
            for idx, line in enumerate(lines, 1):
                if not raw_active_pattern.search(line):
                    continue
                start = max(0, idx - 6)
                end = min(len(lines), idx + 5)
                block = "\n".join(lines[start:end])
                if "validated_setups" not in block:
                    continue
                violations.append(f"  {rel}:{idx}: raw status='active' predicate in critical shelf reader")

    return violations


def check_document_authority_registry() -> list[str]:
    """Core authority docs must exist and advertise their roles explicitly."""
    violations: list[str] = []

    registry = PROJECT_ROOT / "docs" / "governance" / "document_authority.md"
    if not registry.exists():
        return ["  docs/governance/document_authority.md missing"]

    registry_text = registry.read_text(encoding="utf-8")
    required_registry_refs = [
        "CLAUDE.md",
        "TRADING_RULES.md",
        "RESEARCH_RULES.md",
        "ROADMAP.md",
        "HANDOFF.md",
        "docs/plans/",
        "docs/institutional/pre_registered_criteria.md",
        "docs/governance/system_authority_map.md",
        "docs/ARCHITECTURE.md",
        "docs/MONOREPO_ARCHITECTURE.md",
        "docs/context/*.md",
        "REPO_MAP.md",
    ]
    for ref in required_registry_refs:
        if ref not in registry_text:
            violations.append(f"  docs/governance/document_authority.md missing registry reference {ref!r}")

    required_markers = {
        Path("CLAUDE.md"): "## Document Authority",
        Path("TRADING_RULES.md"): "Single source of truth for live trading.",
        Path("RESEARCH_RULES.md"): "**Authority:**",
        Path("ROADMAP.md"): "Features planned but NOT YET BUILT.",
        Path("HANDOFF.md"): "Cross-Tool Session Baton",
        Path("docs/ARCHITECTURE.md"): "Reference guide only.",
        Path("docs/MONOREPO_ARCHITECTURE.md"): "Reference guide only.",
        Path("REPO_MAP.md"): "Auto-generated by `scripts/tools/gen_repo_map.py`.",
    }
    for rel_path, marker in required_markers.items():
        doc_path = PROJECT_ROOT / rel_path
        if not doc_path.exists():
            violations.append(f"  {rel_path} missing")
            continue
        content = doc_path.read_text(encoding="utf-8")
        if marker not in content:
            violations.append(f"  {rel_path} missing authority marker {marker!r}")

    return violations


def check_system_authority_map() -> list[str]:
    """Whole-project authority map must stay generated from the canonical registry."""
    violations: list[str] = []

    authority_map = PROJECT_ROOT / "docs" / "governance" / "system_authority_map.md"
    if not authority_map.exists():
        return ["  docs/governance/system_authority_map.md missing"]

    from pipeline.system_authority import render_system_authority_map

    content = authority_map.read_text(encoding="utf-8")
    expected = render_system_authority_map()
    if content != expected:
        violations.append(
            "  docs/governance/system_authority_map.md drifted from pipeline/system_authority.py; "
            "re-render via scripts/tools/render_system_authority_map.py"
        )
    return violations


def check_context_routing_registry() -> list[str]:
    """Task-context registry must resolve to real paths and known domain/profile IDs."""
    try:
        from context.registry import validate_registry
    except ImportError:
        return ["  context/registry.py not found — context package not yet on this branch"]

    return [f"  {violation}" for violation in validate_registry()]


def check_context_generated_docs() -> list[str]:
    """Generated context-routing docs must stay in sync with the canonical registry."""
    try:
        from context.registry import (
            render_institutional_markdown,
            render_readme_markdown,
            render_source_catalog_markdown,
            render_task_routes_markdown,
        )
    except ImportError:
        return ["  context/registry.py not found — context package not yet on this branch"]

    expected_by_path = {
        PROJECT_ROOT / "docs" / "context" / "README.md": render_readme_markdown(),
        PROJECT_ROOT / "docs" / "context" / "source-catalog.md": render_source_catalog_markdown(),
        PROJECT_ROOT / "docs" / "context" / "task-routes.md": render_task_routes_markdown(),
        PROJECT_ROOT / "docs" / "context" / "institutional-contracts.md": render_institutional_markdown(),
    }
    violations: list[str] = []
    for path, expected in expected_by_path.items():
        if not path.exists():
            violations.append(f"  {path.relative_to(PROJECT_ROOT)} missing")
            continue
        content = path.read_text(encoding="utf-8")
        if content != expected:
            violations.append(
                f"  {path.relative_to(PROJECT_ROOT)} drifted from context/registry.py; "
                "re-render via scripts/tools/render_context_catalog.py"
            )
    return violations


def check_context_view_contracts() -> list[str]:
    """Generated task views must preserve strict truth-class boundaries."""
    try:
        from scripts.tools.context_views import VIEW_BUILDERS, build_view, validate_view_payload
    except (ImportError, ModuleNotFoundError, SystemExit):
        # SystemExit: context_views.py runs argparse at module level
        return ["  context_views not importable — context package not yet wired on this branch"]

    violations: list[str] = []
    db_path = PROJECT_ROOT / "data" / "gold.db"
    for view in VIEW_BUILDERS:
        try:
            payload = build_view(view, PROJECT_ROOT, db_path)
        except Exception as exc:
            violations.append(f"  context view {view} failed to build: {type(exc).__name__}: {exc}")
            continue
        for violation in validate_view_payload(payload):
            violations.append(f"  context view {view}: {violation}")
    return violations


def check_agents_mentions_context_resolver() -> list[str]:
    """AGENTS.md should point cold-start agents to the deterministic context resolver."""
    agents_path = PROJECT_ROOT / "AGENTS.md"
    if not agents_path.exists():
        return ["  AGENTS.md missing"]
    content = agents_path.read_text(encoding="utf-8")
    required_refs = [
        "scripts/tools/context_resolver.py",
        "docs/governance/document_authority.md",
        "docs/governance/system_authority_map.md",
    ]
    violations: list[str] = []
    for ref in required_refs:
        if ref not in content:
            violations.append(f"  AGENTS.md missing context-routing reference {ref!r}")
    return violations


def check_startup_docs_reference_context_router() -> list[str]:
    """Startup/orientation docs must point non-trivial tasks at the deterministic router."""
    docs = {
        PROJECT_ROOT / "CLAUDE.md": ["scripts/tools/context_resolver.py", "## Task Routing"],
        PROJECT_ROOT / "CODEX.md": ["scripts/tools/context_resolver.py"],
    }
    violations: list[str] = []
    for path, markers in docs.items():
        if not path.exists():
            violations.append(f"  {path.relative_to(PROJECT_ROOT)} missing")
            continue
        content = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in content:
                violations.append(f"  {path.relative_to(PROJECT_ROOT)} missing context-router marker {marker!r}")
    return violations


def check_live_audit_uses_runtime_authority() -> list[str]:
    """Phase 7 audit must use runtime profile lanes + deployable shelf, not deprecated LIVE_PORTFOLIO."""
    audit_path = SCRIPTS_DIR / "audits" / "phase_7_live_trading.py"
    if not audit_path.exists():
        return ["  scripts/audits/phase_7_live_trading.py missing"]

    content = audit_path.read_text(encoding="utf-8")
    violations: list[str] = []
    if "LIVE_PORTFOLIO" in content:
        violations.append("  scripts/audits/phase_7_live_trading.py still references deprecated LIVE_PORTFOLIO")
    if "get_active_profile_ids" not in content or "get_profile_lane_definitions" not in content:
        violations.append(
            "  scripts/audits/phase_7_live_trading.py must source active runtime lanes from trading_app.prop_profiles"
        )
    if "deployable_validated_relation" not in content:
        violations.append(
            "  scripts/audits/phase_7_live_trading.py must validate lanes against deployable_validated_relation()"
        )
    return violations


def check_project_pulse_uses_authority_registry() -> list[str]:
    """Project pulse must expose repo identity from canonical authority registry + path/config surfaces."""
    pulse_path = SCRIPTS_DIR / "tools" / "project_pulse.py"
    if not pulse_path.exists():
        return ["  scripts/tools/project_pulse.py missing"]

    content = pulse_path.read_text(encoding="utf-8")
    violations: list[str] = []
    required_refs = [
        "collect_system_identity",
        "pipeline.system_authority",
        "ACTIVE_ORB_INSTRUMENTS",
        "GOLD_DB_PATH",
        "SYSTEM_AUTHORITY_BACKBONE_MODULES",
    ]
    for ref in required_refs:
        if ref not in content:
            violations.append(f"  scripts/tools/project_pulse.py missing canonical identity reference {ref!r}")
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
        violations.append(
            f"  Expected exactly one runtime build_profile_fingerprint() definition, found {runtime_defs}"
        )

    # AST-based import detection — robust to single-line, multi-line, and
    # parenthesized `from X import (a, b, c)` forms. Literal string match
    # was brittle: ruff's I001 import-sort consolidates separate imports
    # into the existing multi-line block, breaking the literal pattern
    # while preserving semantics. See drift-check-96-ast-aware stage.
    canonical_module = "trading_app.derived_state"
    canonical_name = "build_profile_fingerprint"
    has_canonical_import = False
    if account_text:
        try:
            tree = ast.parse(account_text)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == canonical_module
                    and any(alias.name == canonical_name for alias in node.names)
                ):
                    has_canonical_import = True
                    break
        except SyntaxError:
            # Unparseable file — let other checks handle it; don't double-violate.
            has_canonical_import = True
    if not has_canonical_import:
        violations.append(
            "  trading_app/account_survival.py must import build_profile_fingerprint from trading_app.derived_state"
        )

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
        collect_idx : pulse_text.find("\ndef ", collect_idx + 1)
        if pulse_text.find("\ndef ", collect_idx + 1) != -1
        else None
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
            reader_idx : lifecycle_text.find("\ndef ", reader_idx + 1)
            if lifecycle_text.find("\ndef ", reader_idx + 1) != -1
            else None
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


def check_sr_pauses_have_recent_evidence() -> list[str]:
    """ADVISORY: SR-monitor pauses on lane_overrides_*.json should be supported by
    current evidence (post-alarm recovery NOT observed) — surfaces stale peak-SR
    alarms that mask recovered streams.

    Conjunction predicate (per 2026-05-11 plan agent finding): all three must hold
    for the advisory to fire on a paused lane:
      1. current_sr_stat < threshold * 0.5  (SR has fully recovered)
      2. trades_since_alarm >= 10            (enough post-alarm trades to be informative)
      3. recent_10_mean_r > 0                (last 10 trades are net positive)

    Origin: 2026-05-11 sr_monitor misread incident. Operator (Claude) ran
    `--apply-pauses` reading `sr_stat` as current state when it is the peak / trigger
    value. All 3 deployed MNQ lanes were paused; 2 had >30 trades of clean recovery.
    Even with the registry override, `sr_monitor --apply-pauses` can re-shadow the
    review at `lifecycle_state.py:232` because pause_info short-circuits the
    registry consultation.

    Advisory (does not block): the canonical override surface is
    `trading_app/sr_review_registry.py`; this check just surfaces drift between the
    SR-monitor's autonomous pause writes and recovered streams. Remediation is to
    register a `watch` outcome and delete the `lane_overrides_*.json` entry.
    """
    import json

    violations: list[str] = []
    state_dir = PROJECT_ROOT / "data" / "state"
    sr_state_path = state_dir / "sr_state.json"
    if not sr_state_path.exists():
        return violations

    try:
        sr = json.loads(sr_state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return violations

    try:
        profile_id = sr["canonical_inputs"]["profile_id"]
        results = sr["payload"]["results"]
    except (KeyError, TypeError):
        return violations

    overrides_path = state_dir / f"lane_overrides_{profile_id}.json"
    if not overrides_path.exists():
        return violations

    try:
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return violations

    sr_by_sid = {row.get("strategy_id"): row for row in results if isinstance(row, dict)}
    K_TRADES_SINCE = 10

    for sid, ov in overrides.items():
        if not isinstance(ov, dict):
            continue
        if ov.get("active", True):
            continue
        if ov.get("source") != "sr_monitor":
            continue
        sr_lane = sr_by_sid.get(sid)
        if not sr_lane:
            continue
        current = sr_lane.get("current_sr_stat")
        threshold = sr_lane.get("threshold")
        trades_since = sr_lane.get("trades_since_alarm")
        recent_mean_r = sr_lane.get("recent_10_mean_r")
        if current is None or threshold is None:
            continue
        if trades_since is None or recent_mean_r is None:
            continue
        recovered_sr = current < threshold * 0.5
        enough_trades = trades_since >= K_TRADES_SINCE
        positive_recent = recent_mean_r > 0
        if recovered_sr and enough_trades and positive_recent:
            violations.append(
                f"  {sid}: sr_monitor pause looks stale "
                f"(current_sr={current:.2f} << threshold={threshold:.2f}, "
                f"trades_since_alarm={trades_since}, recent_10_mean_r={recent_mean_r:+.3f}). "
                f"Remediate: register a 'watch' outcome in trading_app/sr_review_registry.py "
                f"and remove the entry from data/state/lane_overrides_{profile_id}.json."
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


def check_deployable_subset_of_active() -> list[str]:
    """DEPLOYABLE_ORB_INSTRUMENTS must be a strict subset of ACTIVE_ORB_INSTRUMENTS.

    The canonical taxonomy is that deployable instruments are the subset of
    active instruments that are expected to have validated strategies on the
    deployable shelf. Any instrument with deployable_expected=True must first
    be orb_active=True (otherwise it's not in the run set at all). This
    invariant is enforced at definition time by deriving DEPLOYABLE from
    ACTIVE, but we defend-in-depth here so that any future refactor that
    changes the derivation pattern still preserves the invariant.
    """
    from pipeline.asset_configs import (
        ACTIVE_ORB_INSTRUMENTS,
        ASSET_CONFIGS,
        DEPLOYABLE_ORB_INSTRUMENTS,
    )

    violations = []
    active_set = set(ACTIVE_ORB_INSTRUMENTS)
    deployable_set = set(DEPLOYABLE_ORB_INSTRUMENTS)
    rogue = deployable_set - active_set
    if rogue:
        violations.append(
            f"  DEPLOYABLE_ORB_INSTRUMENTS contains instruments not in "
            f"ACTIVE_ORB_INSTRUMENTS: {sorted(rogue)}. Every deployable "
            f"instrument must first be orb_active=True."
        )

    # A separate failure mode: an instrument carries deployable_expected=True
    # (default or explicit) but somehow didn't make it into the derived list.
    # Catches anyone who manually builds DEPLOYABLE_ORB_INSTRUMENTS in the
    # future and drops a valid entry.
    expected_deployable = {k for k in active_set if ASSET_CONFIGS[k].get("deployable_expected", True)}
    missing = expected_deployable - deployable_set
    if missing:
        violations.append(
            f"  DEPLOYABLE_ORB_INSTRUMENTS is missing active instruments whose "
            f"deployable_expected flag is True: {sorted(missing)}."
        )
    return violations


def check_signal_log_rotation_not_bypassed() -> list[str]:
    """_write_signal_record must delegate to SignalLogRotator, not raw open().

    R4 fix (Ralph iter 181): live_signals.jsonl was written via a bare `open(..., "a")`
    call inside `_write_signal_record`. This bypassed rotation and swallowed disk-full
    errors silently (institutional-rigor.md § 6 violation).

    After the fix, `_write_signal_record` must:
      (1) NOT contain `open(self.SIGNALS_FILE` (raw file open bypasses rotator).
      (2) Contain `_signal_rotator` (delegates to SignalLogRotator).
      (3) SIGNALS_FILE must NOT appear in the class body (replaced by SIGNALS_DIR).

    A future refactor that reverts to raw open() trips this check.
    """
    violations = []
    target = TRADING_APP_DIR / "live" / "session_orchestrator.py"
    if not target.exists():
        violations.append(f"  {target}: missing — cannot verify signal log rotation guard")
        return violations

    try:
        source = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        violations.append(f"  {target.name}: failed to read for signal rotation check: {exc}")
        return violations

    # Offense: raw open() on a monolithic signals file in _write_signal_record.
    if "open(self.SIGNALS_FILE" in source:
        violations.append(
            "  session_orchestrator.py: `open(self.SIGNALS_FILE` found — _write_signal_record "
            "must delegate to SignalLogRotator, not write directly. R4 rotation bypass."
        )
    # Offense: SIGNALS_FILE class attribute still present (replaced by SIGNALS_DIR in R4).
    if "SIGNALS_FILE = " in source:
        violations.append(
            "  session_orchestrator.py: `SIGNALS_FILE = ` found — R4 replaced this with "
            "SIGNALS_DIR. Remove SIGNALS_FILE or the rotation invariant is broken."
        )
    # Required: _signal_rotator delegation present.
    if "_signal_rotator" not in source:
        violations.append(
            "  session_orchestrator.py: `_signal_rotator` not found — _write_signal_record "
            "must delegate to SignalLogRotator (R4 fix). Rotation is absent."
        )

    return violations


def check_c1_kill_switch_guards_intact() -> list[str]:
    """C1 kill-switch guards must remain in place at the canonical insertion points.

    The C1 race (iter 174 F4 audit, 2026-04-25) was: `_handle_event` had no
    `_kill_switch_fired` guard, so an entry-creating event N+1 in the same bar
    could submit a NEW broker entry after the kill-switch fired for event N.
    The fix (commit f8f993b7) added a guard at the top of the ENTRY branch.

    This check enforces TWO regression-prevention invariants in
    `trading_app/live/session_orchestrator.py`:

    (1) `_on_bar` body must contain `_kill_switch_fired` near its top.
        Canonical guard added in pre-history; protects bar-level dispatch.

    (2) `_handle_event` body must contain `_kill_switch_fired` AND must contain
        a check on `event.event_type == "ENTRY"` near the same line — the
        guard must be ENTRY-scoped so EXIT/SCRATCH events still wind down
        existing exposure during a halt (do-not-touch from iter 178 audit).

    A future refactor that removes either guard, OR widens the C1 guard to
    blanket all event types (breaking EOD wind-down), trips this check.
    """
    violations = []
    target = TRADING_APP_DIR / "live" / "session_orchestrator.py"
    if not target.exists():
        violations.append(f"  {target}: missing — cannot verify C1 kill-switch guards")
        return violations

    try:
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        violations.append(f"  {target.name}: failed to parse for C1 guard verification: {exc}")
        return violations

    methods_to_check = {"_on_bar", "_handle_event"}
    found: dict[str, ast.AsyncFunctionDef | ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name in methods_to_check:
            found[node.name] = node

    for missing in methods_to_check - set(found):
        violations.append(
            f"  session_orchestrator.py: method `{missing}` not found — "
            f"C1 guard cannot be verified. If renamed, update check 114 to match."
        )

    if "_on_bar" in found:
        body_src = ast.get_source_segment(source, found["_on_bar"]) or ""
        if "_kill_switch_fired" not in body_src:
            violations.append(
                f"  session_orchestrator.py:{found['_on_bar'].lineno} `_on_bar`: "
                f"`_kill_switch_fired` guard missing. C1 race re-opened — "
                f"events arriving on a halted orchestrator can reach broker."
            )

    if "_handle_event" in found:
        body_src = ast.get_source_segment(source, found["_handle_event"]) or ""
        if "_kill_switch_fired" not in body_src:
            violations.append(
                f"  session_orchestrator.py:{found['_handle_event'].lineno} `_handle_event`: "
                f"C1 ENTRY-branch `_kill_switch_fired` guard missing. "
                f"See iter 174 audit + iter 177 fix `f8f993b7`."
            )
        elif 'event_type == "ENTRY"' not in body_src and "event_type == 'ENTRY'" not in body_src:
            violations.append(
                f"  session_orchestrator.py:{found['_handle_event'].lineno} `_handle_event`: "
                f"C1 guard present but ENTRY-branch discriminator missing — "
                f"a blanket guard would break EOD wind-down (iter 178 audit do-not-touch)."
            )

    return violations


def check_no_crlf_in_tracked_text_blobs() -> list[str]:
    """No tracked text file may have CRLF line endings in its committed blob.

    Defense-in-depth for the pre-commit `[0b]` auto-renormalize hook.
    The hook prevents new CRLF entering commits via the normal commit path; this
    check catches anything that bypassed the hook — `--no-verify`, direct API push,
    `git lfs`, hook tampering, or a missing hooksPath setting on a contributor box.

    Scope: every file with `.gitattributes` `text=set` or `text=auto` whose blob
    at HEAD contains a `\\r\\n` byte sequence is a violation.

    Why blob-not-WT: working-tree state is environment-specific (Windows checkout
    can produce CRLF on disk even from LF blobs). The COMMITTED blob is the
    canonical contract — that is what reaches CI, other contributors, and merge
    conflict resolution. Phantom-modified WT files do NOT trigger this check;
    only actual CRLF in HEAD's tree does.

    History: PR #130 (2026-04-26) renormalized 83 historical CRLF blobs to LF
    after multiple sessions hit the recurring `pre-rebase CRLF noise` pattern
    (stash@{6-10} in repo). This check exists so that debt class never returns.
    """
    # Perf-tuned 2026-05-02: previous implementation spawned 2 subprocesses per
    # tracked file (git check-attr + git show), ~20k spawns on this repo, ~70s
    # on Windows + Git Bash. New shape: O(1) subprocess calls via batched stdin
    # for check-attr and a single `git grep` against HEAD's tree. Verdict
    # semantics are identical — same set of files (text=set/auto per
    # .gitattributes) is examined for CRLF in their HEAD-committed blobs.
    violations: list[str] = []
    try:
        ls_files = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            timeout=20,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []  # not a git repo / git unavailable — skip silently

    tracked = [p for p in ls_files.stdout.decode("utf-8", "replace").split("\x00") if p]
    if not tracked:
        return []

    # 1. Batched check-attr: feed every path on stdin in NUL-separated form.
    # `git check-attr --stdin -z text` returns one record per path:
    # "<path>\0text\0<value>\0".
    try:
        attr_input = b"\x00".join(p.encode("utf-8", "replace") for p in tracked)
        attr_proc = subprocess.run(
            ["git", "check-attr", "--stdin", "-z", "text"],
            cwd=PROJECT_ROOT,
            input=attr_input,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []  # git unavailable mid-run — skip silently per pre-existing fail-open

    # Parse: stream is path\0attr\0value\0path\0attr\0value\0…
    text_paths: set[str] = set()
    fields = attr_proc.stdout.split(b"\x00")
    # Drop trailing empty produced by the final \0
    if fields and fields[-1] == b"":
        fields.pop()
    for i in range(0, len(fields), 3):
        if i + 2 >= len(fields):
            break
        path = fields[i].decode("utf-8", "replace")
        value = fields[i + 2].decode("utf-8", "replace")
        if value in ("set", "auto"):
            text_paths.add(path)

    if not text_paths:
        return []

    # 2. Single `git grep` over HEAD's tree for CRLF. -l = list matching files,
    # -I = skip binary, -P = PCRE (needed for \r), --cached vs treeish: use
    # HEAD directly so we get COMMITTED blobs (the contract this check enforces).
    # Output is NUL-terminated paths via -z.
    try:
        grep_proc = subprocess.run(
            ["git", "grep", "-lI", "-z", "-P", r"\r$", "HEAD", "--"] + sorted(text_paths),
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    # `git grep <treeish>` prefixes each match with "<treeish>:". -z stays NUL-
    # separated. Strip the "HEAD:" prefix for reporting.
    matches = [m for m in grep_proc.stdout.split(b"\x00") if m]
    crlf_paths: list[str] = []
    head_prefix = b"HEAD:"
    for raw in matches:
        if raw.startswith(head_prefix):
            crlf_paths.append(raw[len(head_prefix) :].decode("utf-8", "replace"))
        else:
            # Defensive: should not happen with `git grep HEAD --`, but if a
            # different output shape ever appears, surface it as a finding so
            # silent fail-open never masks a real CRLF blob.
            crlf_paths.append(raw.decode("utf-8", "replace"))

    # 3. For each CRLF-in-HEAD path, count CRLF lines for the violation message.
    # We only `git show` the offenders — typically zero, never more than a
    # handful. This is the only per-file subprocess and bounded by the
    # offender count, not the tree size.
    for rel_path in crlf_paths:
        try:
            blob = subprocess.run(
                ["git", "show", f"HEAD:{rel_path}"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                check=False,
                timeout=5,
            ).stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
        crlf_lines = blob.count(b"\r\n")
        if crlf_lines == 0:
            # Should not happen if grep matched, but be defensive about
            # cross-tool encoding/normalization edge cases.
            continue
        violations.append(
            f"  {rel_path} — committed blob has {crlf_lines} CRLF line(s). "
            f"Per .gitattributes eol=lf, must be LF. "
            f"Fix: `git add --renormalize -- {rel_path} && git commit`."
        )

    return violations


def check_canonical_claude_client_source() -> list[str]:
    """Only `trading_app/ai/claude_client.py` may hardcode Claude model IDs
    or instantiate `anthropic.Anthropic(...)` directly.

    Stage 4 of the claude-api-modernization refactor (2026-04-17) consolidated
    model pins and client construction into a single canonical module. This
    check prevents regressions where a new file hardcodes a Claude model
    string (e.g. `claude-opus-4-7`) or calls `anthropic.Anthropic()` directly,
    bypassing `CLAUDE_STRUCTURED_MODEL` / `CLAUDE_REASONING_MODEL` / `get_client()`.

    Allowed: `trading_app/ai/claude_client.py` (canonical home), `check_drift.py`
    (this file, which references the literal patterns in regex), and `tests/**`
    (stale-ID fixtures for testing). Scan covers `pipeline/`, `trading_app/`,
    `scripts/`, `research/`.

    Two offense patterns:
      1. `claude-(opus|sonnet|haiku)-\\d(?:[\\d-]*\\d)?` — any hardcoded Claude model ID
         (covers `claude-opus-4-7`, `claude-sonnet-4-5-20250929`, etc.)
      2. `anthropic.Anthropic(` — any direct client construction
    """
    violations = []
    model_pattern = re.compile(r"claude-(opus|sonnet|haiku)-\d(?:[\d-]*\d)?")
    client_pattern = re.compile(r"anthropic\.Anthropic\s*\(")

    canonical_file = TRADING_APP_DIR / "ai" / "claude_client.py"
    scan_dirs = [PIPELINE_DIR, TRADING_APP_DIR, SCRIPTS_DIR, RESEARCH_DIR]

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

            try:
                rel: Path | str = py_file.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = py_file  # tmp_path during testing — fall back to absolute

            for match in model_pattern.finditer(content):
                line_no = content[: match.start()].count("\n") + 1
                violations.append(
                    f"  {rel}:{line_no} — hardcoded Claude model ID `{match.group(0)}`. "
                    f"Import CLAUDE_STRUCTURED_MODEL or CLAUDE_REASONING_MODEL from "
                    f"trading_app.ai.claude_client instead."
                )

            for match in client_pattern.finditer(content):
                line_no = content[: match.start()].count("\n") + 1
                violations.append(
                    f"  {rel}:{line_no} — direct `anthropic.Anthropic(` instantiation. "
                    f"Use trading_app.ai.claude_client.get_client() instead."
                )

    return violations


def check_routed_filter_columns_populated(con=None) -> list[str]:
    """Every routed filter's required daily_features columns must be populated.

    Catches ghost deployments — a filter registered in ALL_FILTERS and
    routed to a session via get_session_filters, where the daily_features
    column it depends on is 0%-populated (or near-zero). The filter is
    fail-closed on NULL, so such a lane silently produces zero trades.

    Canonical example: PIT_MIN was routed to CME_REOPEN in commit
    f776bc13 (2026-04-06) but daily_features.pit_range_atr was never
    populated — the backfill code did not land with the schema. Zero
    live-trade impact only because no deployed lane used PIT_MIN; the
    trap would have fired on the first deployment.

    What this check enforces:
      1. Collect the routed set: every filter_type string that appears
         in get_session_filters output across all SESSION_CATALOG entries.
      2. For each routed filter, walk composites to leaves and call
         describe() to collect every feature_column reporting a scalar
         daily_features column (skip orb_* per-aperture columns — those
         are session-conditional by design and checked elsewhere).
      3. Query the live DB for population fraction of each column scoped
         to ACTIVE_ORB_INSTRUMENTS at orb_minutes=5.
      4. Flag when fraction < 0.50 — catches both 0%-populated ghost
         deployments and writer regressions while tolerating sparse
         early-history warmups.

    @rule backfill-integrity
    @stage hardening-three-fixes (2026-04-20)
    """
    violations: list[str] = []
    if con is None:
        return violations  # DB-required check skips cleanly when unavailable

    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from pipeline.dst import SESSION_CATALOG
        from trading_app.config import (
            ALL_FILTERS,
            CompositeFilter,
            NoFilter,
            StrategyFilter,
            get_filters_for_grid,
        )
    except ImportError as exc:
        return [f"  Could not import required modules: {exc}"]

    # Collect the ROUTED set: filter_type strings appearing in any
    # (instrument, session) filter map. This is the production-contract set.
    routed: set[str] = set()
    for inst in ACTIVE_ORB_INSTRUMENTS:
        for session_label in SESSION_CATALOG:
            try:
                filters = get_filters_for_grid(inst, session_label)
            except Exception:
                continue
            routed.update(filters.keys())

    if not routed:
        return ["  Could not enumerate routed filters — SESSION_CATALOG or get_session_filters failed"]

    # Walk composites to leaves and collect scalar daily_features columns.
    # Sample row gives describe() enough context to return atoms without
    # raising. CME_REOPEN chosen because it routes the largest filter set.
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
        "garch_forecast_vol_pct": 80.0,
    }

    def _walk_leaves(filt: StrategyFilter) -> list[StrategyFilter]:
        if isinstance(filt, CompositeFilter):
            return _walk_leaves(filt.base) + _walk_leaves(filt.overlay)
        return [filt]

    # Map each required scalar column back to the filter(s) that need it.
    col_to_filters: dict[str, set[str]] = {}
    for ft in sorted(routed):
        if ft not in ALL_FILTERS:
            continue
        if isinstance(ALL_FILTERS[ft], NoFilter):
            continue
        for leaf in _walk_leaves(ALL_FILTERS[ft]):
            if isinstance(leaf, NoFilter):
                continue
            try:
                atoms = leaf.describe(sample_row, "CME_REOPEN", "E2")
            except Exception:
                continue
            for atom in atoms:
                col = getattr(atom, "feature_column", None)
                if col is None:
                    continue
                # Skip per-aperture nested columns — they are session-specific
                # by construction and covered by other integrity checks.
                if col.startswith("orb_"):
                    continue
                col_to_filters.setdefault(col, set()).add(ft)

    # Verify each scalar column is present in daily_features schema AND
    # populated at >= 50% for active instruments at orb_minutes=5.
    try:
        known_cols = {
            row[0]
            for row in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'daily_features'"
            ).fetchall()
        }
    except Exception as exc:
        return [f"  Could not introspect daily_features schema: {exc}"]

    active = tuple(ACTIVE_ORB_INSTRUMENTS)
    if not active:
        return []  # nothing to check

    placeholders = ",".join(["?"] * len(active))
    # Runtime-injected columns (not backed by daily_features schema) are
    # tracked separately. They fail-closed at eligibility time if the
    # injector is broken, so visibility matters. The primary violation
    # surface of this check is schema-population; runtime-injected
    # coverage is a separate check class, emitted as advisory via stderr.
    runtime_injected: list[str] = []
    for col in sorted(col_to_filters):
        if col not in known_cols:
            filters_list = ", ".join(sorted(col_to_filters[col]))
            runtime_injected.append(f"    {col} (required by [{filters_list}])")
            continue
        try:
            row = con.execute(
                f"SELECT COUNT(*) total, COUNT({col}) populated "
                f"FROM daily_features WHERE orb_minutes = 5 "
                f"AND symbol IN ({placeholders})",
                list(active),
            ).fetchone()
        except Exception as exc:
            violations.append(f"  Could not query {col} population: {exc}")
            continue
        if row is None:
            continue
        total, populated = int(row[0]), int(row[1])
        if total == 0:
            continue
        pct = populated / total
        if pct < 0.50:
            filters_list = ", ".join(sorted(col_to_filters[col]))
            violations.append(
                f"  Column '{col}' is {pct:.1%} populated across "
                f"ACTIVE_ORB_INSTRUMENTS but is required by routed filters "
                f"[{filters_list}] — likely ghost deployment or writer regression"
            )

    if runtime_injected:
        # Informational only — these columns are not in daily_features and
        # must be covered by runtime-injection integrity checks elsewhere.
        # Emit to stderr so they appear in the audit log but do not block.
        import sys as _sys

        print(
            "  INFO: routed filters require runtime-injected columns (not daily_features-backed):",
            file=_sys.stderr,
        )
        for entry in runtime_injected:
            print(entry, file=_sys.stderr)

    return violations


def check_pooled_finding_annotations() -> list[str]:
    """Audit-result files claiming pooled-universe findings must carry schema.

    RULE 14 (2026-04-20 retroactive heterogeneity audit, commit aa3399b3):
    pooled-universe p-values and ExpR averages hide opposite-sign
    per-cell behavior. A pooled claim without a per-cell breakdown is
    a silent heterogeneity artefact.

    What this check enforces on audit-result files created or modified
    after the sentinel date 2026-04-20:
      1. If YAML front-matter sets pooled_finding: true, the file must
         also carry per_cell_breakdown_path: <repo-relative path> and
         flip_rate_pct: <0-100 numeric>.
      2. If flip_rate_pct >= 25, the file must carry
         heterogeneity_ack: true to acknowledge the heterogeneity finding.

    Historical audits modified before the sentinel are exempt — editing
    an older file after the sentinel opts into the schema.

    @rule pooled-finding-rule
    @stage hardening-three-fixes (2026-04-20)
    """
    sentinel_date = "2026-04-20"
    results_dir = PROJECT_ROOT / "docs" / "audit" / "results"
    if not results_dir.is_dir():
        return []

    violations: list[str] = []

    for md_file in sorted(results_dir.glob("*.md")):
        # Skip the template itself
        if md_file.name.startswith("TEMPLATE-"):
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # Only files whose name starts with the sentinel date or later.
        # Filename convention: YYYY-MM-DD-<slug>.md
        fname = md_file.name
        date_prefix = fname[:10] if len(fname) >= 10 else ""
        if date_prefix < sentinel_date:
            continue

        # Extract YAML front-matter (between leading --- delimiters).
        # Minimal parser: only looks at top-level key: value lines.
        if not content.startswith("---"):
            continue  # no front-matter — rule does not apply
        end_marker = content.find("\n---", 3)
        if end_marker < 0:
            continue
        fm_text = content[3:end_marker]
        fm: dict[str, str] = {}
        for line in fm_text.splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition(":")
                fm[k.strip()] = v.strip().strip('"').strip("'")

        pooled_flag = fm.get("pooled_finding", "").lower()
        if pooled_flag not in ("true", "yes", "1"):
            continue  # not a pooled-finding file

        rel = md_file.relative_to(PROJECT_ROOT)

        breakdown_path = fm.get("per_cell_breakdown_path", "")
        if not breakdown_path:
            violations.append(f"  {rel}: pooled_finding: true but per_cell_breakdown_path missing")
        flip_rate_str = fm.get("flip_rate_pct", "")
        flip_rate: float | None = None
        if flip_rate_str:
            try:
                flip_rate = float(flip_rate_str)
            except ValueError:
                violations.append(f"  {rel}: flip_rate_pct={flip_rate_str!r} is not a number")
        else:
            violations.append(f"  {rel}: pooled_finding: true but flip_rate_pct missing")

        if flip_rate is not None and flip_rate >= 25.0:
            ack = fm.get("heterogeneity_ack", "").lower()
            if ack not in ("true", "yes", "1"):
                violations.append(
                    f"  {rel}: flip_rate_pct={flip_rate:.1f} >= 25 but "
                    f"heterogeneity_ack not set to true — RULE 14 requires "
                    f"explicit acknowledgement when pooled framing hides "
                    f"≥25% cell-level sign flips"
                )

    return violations


def check_magic_number_rationale(trading_app_dir: Path) -> list[str]:
    """Pass Three: magic-number rationale audit on capital-class trading_app paths.

    Every UPPER_SNAKE_CASE assignment (class-body or module-level) in the
    scoped paths below whose value is a numeric literal with
    abs(value) > RATIONALE_THRESHOLD must have either:
      (a) a comment containing "Rationale:" or "rationale" (case-insensitive)
          within +/- 10 lines of the assignment (covers multi-paragraph
          comment blocks where the explicit "Rationale:" tag lands a few
          lines above the literal), OR
      (b) the constant name appears in RATIONALE_WHITELIST below.

    Scope (single canonical list — extend carefully, every new path is a
    new gate that fails commits with bare magic numbers):
      - trading_app/live/ — order routing, kill switch, dashboards, webhook
      - trading_app/account_hwm_tracker.py — DD tier boundaries, staleness gates
      - trading_app/pre_session_check.py — pre-flight gate constants

    Why: parameter-justification discipline per Robert Carver,
    "Systematic Trading" Ch. 4 (resources/Robert Carver - Systematic
    Trading.pdf) — every magic number in production code should encode a
    documented decision, not an undefended choice. Untraceable constants
    are how prop-desk strategies silently overfit and how operators lose
    track of why a timeout / threshold / cap was set.

    Introduced: 2026-04-26 as part of v6.1 open-work burndown (Pass Three).
    Scope extended 2026-05-15 (Batch 4 review) to cover HWM tracker +
    pre_session_check — both carry capital-class boundaries (30-day
    staleness, DD warn/block, inactivity window) outside trading_app/live/.
    """
    import ast

    RATIONALE_THRESHOLD = 10
    # Names whose meaning is self-evident by context. Add here only after
    # confirming there is no plausible operator question of "why this value".
    RATIONALE_WHITELIST: set[str] = set()

    violations: list[str] = []

    # Capital-class path scope. live/ is recursive; the two sibling files
    # are explicit because they are not under live/ but carry DD/inactivity
    # gates with the same operator-safety profile.
    scoped_files: list[Path] = []
    live_dir = trading_app_dir / "live"
    if not live_dir.exists():
        return [f"  trading_app/live/ not found at {live_dir}"]
    scoped_files.extend(sorted(live_dir.rglob("*.py")))
    for sibling in ("account_hwm_tracker.py", "pre_session_check.py"):
        sib_path = trading_app_dir / sibling
        if not sib_path.exists():
            return [f"  trading_app/{sibling} not found at {sib_path}"]
        scoped_files.append(sib_path)

    name_re = re.compile(r"[A-Z][A-Z0-9_]+")
    rationale_re = re.compile(r"#.*\brationale\b", re.IGNORECASE)

    for fpath in scoped_files:
        try:
            content = fpath.read_text(encoding="utf-8")
        except Exception:
            continue
        lines = content.splitlines()
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Name) or not name_re.fullmatch(tgt.id):
                continue
            val = node.value
            if not isinstance(val, ast.Constant):
                continue
            v = val.value
            # bool is a subclass of int — exclude explicitly.
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if abs(v) <= RATIONALE_THRESHOLD:
                continue
            if tgt.id in RATIONALE_WHITELIST:
                continue
            line_no = node.lineno
            window_start = max(0, line_no - 11)
            window_end = min(len(lines), line_no + 10)
            window = lines[window_start:window_end]
            if any(rationale_re.search(line) for line in window):
                continue
            try:
                rel = fpath.relative_to(trading_app_dir.parent)
            except ValueError:
                rel = fpath
            violations.append(
                f"  {rel}:{line_no}: magic number {tgt.id}={v} lacks "
                f"'Rationale:' comment within +/- 10 lines (Carver Ch. 4)"
            )
    return violations


def check_nq_mini_substitution_wired_or_unused(trading_app_dir: Path) -> list[str]:
    """Fail-open guard: NQ-mini symbol-substitution must be wired before use.

    The contract added in PR #158 (commit 8bef5eb1, 2026-04-27) introduced
    AccountProfile.execution_symbol_map + .execution_qty_divisor to allow a
    profile to declare e.g. {"MNQ": "NQ"} so a strategy generated against
    the micro is executed on the mini for ~77% commission reduction. Stage
    1 of 3 was the contract; Stage 2 was supposed to wire the call sites in
    trading_app/live/session_orchestrator.py and webhook_server.py via
    resolve_execution_symbol(profile, strategy_symbol) but never landed.

    Failure mode this check closes: if any ACCOUNT_PROFILES entry in
    trading_app/prop_profiles.py populates `execution_symbol_map=...` while
    no production callsite of resolve_execution_symbol() exists in
    trading_app/live/, the broker silently receives the strategy_symbol
    unchanged and original qty — wrong-instrument fills sized for the wrong
    contract. Capital-class silent failure per integrity-guardian § 3.

    Invariant enforced:
      - No ACCOUNT_PROFILES row populates execution_symbol_map: PASS regardless
      - At least one ACCOUNT_PROFILES row populates execution_symbol_map AND
        at least one resolve_execution_symbol() callsite exists under
        trading_app/live/: PASS (Stage 2 wired)
      - Any ACCOUNT_PROFILES row populates execution_symbol_map AND no
        resolve_execution_symbol() callsite exists under trading_app/live/:
        FAIL (Stage 1 active without Stage 2 — silent mis-route trap)

    Test fixtures (tests/) are intentionally exempt — they validate the
    contract surface and SHOULD populate the field to exercise the
    happy path + 8 invalid configurations.

    Introduced: 2026-05-15 (Batch 4 code review). Companion to
    feedback_code_review_dead_class_detection.md (Sonnet missed the dead
    BrokerDispatcher; this check forecloses the equivalent NQ-mini hole).
    """
    import ast

    profiles_path = trading_app_dir / "prop_profiles.py"
    live_dir = trading_app_dir / "live"
    if not profiles_path.exists():
        return [f"  trading_app/prop_profiles.py not found at {profiles_path}"]
    if not live_dir.exists():
        return [f"  trading_app/live/ not found at {live_dir}"]

    try:
        profiles_src = profiles_path.read_text(encoding="utf-8")
    except Exception as exc:
        return [f"  could not read {profiles_path}: {exc}"]

    try:
        tree = ast.parse(profiles_src)
    except SyntaxError as exc:
        return [f"  could not parse {profiles_path}: {exc}"]

    # Find every keyword arg `execution_symbol_map=<expr>` in the source
    # whose <expr> is not the literal None. Restrict to AccountProfile
    # constructions (call_func.id == "AccountProfile") to avoid spurious
    # matches elsewhere in the module.
    populated_lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # AccountProfile(...) — Name node form
        if not isinstance(func, ast.Name) or func.id != "AccountProfile":
            continue
        for kw in node.keywords:
            if kw.arg != "execution_symbol_map":
                continue
            # Literal None means "identity behaviour" — fine.
            if isinstance(kw.value, ast.Constant) and kw.value.value is None:
                continue
            # Empty dict matches loader's identity-behaviour interpretation
            # (AccountProfile.__post_init__ short-circuits on empty maps).
            if isinstance(kw.value, ast.Dict) and len(kw.value.keys) == 0:
                continue
            populated_lines.append(node.lineno)

    if not populated_lines:
        # No profile populates the field — Stage 1 dead infrastructure but
        # not a silent-mis-route hazard. PASS.
        return []

    # Profile is populated — Stage 2 wiring MUST exist somewhere under live/.
    callsite_re = re.compile(r"\bresolve_execution_symbol\s*\(")
    for fpath in sorted(live_dir.rglob("*.py")):
        try:
            text = fpath.read_text(encoding="utf-8")
        except Exception:
            continue
        if callsite_re.search(text):
            # Stage 2 wired — PASS.
            return []

    # Populated AND not wired — fail-open trap.
    profiles_rel = profiles_path.relative_to(trading_app_dir.parent)
    line_list = ", ".join(f"{profiles_rel}:{ln}" for ln in populated_lines)
    return [
        f"  NQ-mini fail-open trap: ACCOUNT_PROFILES populates "
        f"execution_symbol_map at [{line_list}] but no production callsite of "
        f"resolve_execution_symbol() exists under trading_app/live/. Stage 2 "
        f"wiring (session_orchestrator + webhook_server) is missing — broker "
        f"would receive strategy_symbol unchanged. See PR #158 driver memo "
        f"memory/mini_vs_micro_commission_fix.md and the Stage 2 design doc "
        f"docs/runtime/stages/nq-mini-execution-stage1-account-profile.md."
    ]


def check_orb_outcomes_scratch_pnl(con=None) -> list[str]:
    """Post Stage 5 fix: ≥99% of scratch rows must have non-NULL ``pnl_r``.

    The canonical ``trading_app/outcome_builder.py`` fix (Stage 5 of
    ``docs/runtime/stages/scratch-eod-mtm-canonical-fix.md``) populates
    ``pnl_r`` / ``exit_ts`` / ``exit_price`` for ``outcome='scratch'``
    rows from realized session-end close MTM. Pre-fix baseline was
    0% of scratch rows populated. Post-fix target is ≥99% — the <1% gap
    is the pathological no-post-bars edge case explicitly documented in
    ``docs/specs/outcome_builder_scratch_eod_mtm.md``.

    Pre-rebuild this check fires on stale data; it is registered as
    advisory until Stage 5b rebuild completes.

    @rule scratch-eod-mtm-canonical-fix
    @stage stage-5b drift companion
    """
    if con is None:
        return []  # SKIPPED message handled by caller via requires_db=True
    try:
        rows = con.execute("SELECT COUNT(*), COUNT(pnl_r) FROM orb_outcomes WHERE outcome = 'scratch'").fetchone()
    except Exception as exc:
        return [f"  query failed: {exc!r}"]
    if rows is None:
        return []
    total = int(rows[0])
    populated = int(rows[1])
    if total == 0:
        return []
    pct_populated = 100.0 * populated / total
    if pct_populated < 99.0:
        return [
            f"  scratch rows with non-NULL pnl_r: {populated}/{total} = "
            f"{pct_populated:.2f}% (expected ≥ 99% post-Stage-5 fix; "
            f"rebuild outcome_builder for affected instruments — see "
            f"docs/specs/outcome_builder_scratch_eod_mtm.md and "
            f".claude/rules/validation-workflow.md § Full Rebuild Chain)"
        ]
    return []


def check_research_scratch_policy_annotation() -> list[str]:
    """Research scripts using ``pnl_r IS NOT NULL`` must annotate scratch policy.

    Class bug discovered 2026-04-27: ``trading_app/outcome_builder.py`` produces
    ``outcome='scratch'`` rows with ``pnl_r=NULL`` when neither stop nor target
    hits by trading-day-end. Research scripts using ``WHERE pnl_r IS NOT NULL``
    silently drop those rows, inflating measured ExpR by 10-45% on the
    survivor lanes (verified MNQ NYSE_OPEN 15m: 9.9% scratch at RR=1.0,
    44.6% at RR=4.0).

    This check fires on any ``research/**.py`` file (excluding
    ``research/archive/``) that contains the literal string
    ``pnl_r IS NOT NULL`` AND does NOT also contain a canonical
    scratch-policy comment marker. Acceptable markers (case-insensitive):
      - ``# scratch-policy: drop`` (deliberate exclusion of scratches)
      - ``# scratch-policy: include-as-zero`` (count scratches as 0R)
      - ``# scratch-policy: realized-eod`` (count scratches at EOD MTM)

    Companion to Stage 5 fix in ``trading_app/outcome_builder.py`` and
    Criterion 13 in ``docs/institutional/pre_registered_criteria.md``.

    @rule scratch-policy-annotation
    @stage stage-2 of 2026-04-27 scratch-eod-mtm plan
    """
    research_dir = PROJECT_ROOT / "research"
    if not research_dir.is_dir():
        return []

    violations: list[str] = []
    needle = "pnl_r IS NOT NULL"
    marker_prefix = "# scratch-policy:"
    valid_markers = ("drop", "include-as-zero", "realized-eod")

    for py_file in sorted(research_dir.rglob("*.py")):
        try:
            rel = py_file.relative_to(research_dir)
        except ValueError:
            continue
        # Skip archive — frozen scans are not retro-edited per Backtesting Rule 11.
        if rel.parts and rel.parts[0] == "archive":
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        if needle not in content:
            continue

        marker_found_with_valid_value = False
        for line in content.splitlines():
            stripped = line.strip().lower()
            idx = stripped.find(marker_prefix)
            if idx < 0:
                continue
            value = stripped[idx + len(marker_prefix) :].strip()
            if any(value.startswith(m) for m in valid_markers):
                marker_found_with_valid_value = True
                break

        if not marker_found_with_valid_value:
            rel_repo = py_file.relative_to(PROJECT_ROOT)
            violations.append(
                f"  {rel_repo}: contains 'pnl_r IS NOT NULL' but no "
                f"'# scratch-policy: drop|include-as-zero|realized-eod' annotation. "
                f"Class bug 2026-04-27: scratch rows have NULL pnl_r and are silently "
                f"dropped. See memory/feedback_scratch_pnl_null_class_bug.md."
            )

    return violations


def check_e2_lookahead_research_contamination() -> list[str]:
    """Research scripts combining ``entry_model='E2'`` with break-bar predictors.

    Class bug catalogued 2026-04-28: 18 of 73 E2-using research scripts use
    ``rel_vol``, ``break_bar_volume``, ``break_bar_continues``, or
    ``break_delay_min`` as predictors. Real-data verification shows ~41% of
    E2 trades have ``entry_ts < break_ts`` (the canonical "break bar"
    defined by close-outside-ORB is later than E2's range-cross entry on
    that subset), so break-bar features are post-entry data on ~4 in 10
    E2 rows. Canonical authority for the gate at the registered-filter
    layer: ``trading_app/config.py`` ``E2_EXCLUDED_FILTER_PREFIXES`` /
    ``E2_EXCLUDED_FILTER_SUBSTRINGS``. Research scripts that bypass
    ``ALL_FILTERS`` and read ``daily_features`` directly are not gated.

    Fires on any ``research/**.py`` file (excluding ``research/archive/``)
    that contains BOTH:
      1. an ``entry_model='E2'`` or ``entry_model="E2"`` literal, AND
      2. a tainted feature reference: ``rel_vol``, ``break_bar_volume``,
         ``break_bar_continues``, or ``break_delay_min``.

    Skips and policies:
      - ``research/archive/`` is skipped (frozen scans not retro-edited).
      - The 18 known-tainted scripts from the 2026-04-28 registry are
        grandfathered via ``known_tainted_registry`` — the registry doc
        is the canonical record of their status.
      - Scripts with a canonical annotation are cleared.

    Acceptable annotations (case-insensitive):
      - ``# e2-lookahead-policy: cleared`` (verified clean post-fix)
      - ``# e2-lookahead-policy: late-fill-only`` (filter ``entry_ts >= break_ts``)
      - ``# e2-lookahead-policy: not-predictor`` (feature is window-sizing only)
      - ``# e2-lookahead-policy: tainted`` (acknowledges, kept for audit trail)

    See ``docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md``.
    @rule e2-lookahead-policy
    """
    research_dir = PROJECT_ROOT / "research"
    if not research_dir.is_dir():
        return []

    known_tainted_registry = frozenset(
        {
            "comprehensive_deployed_lane_scan.py",
            "participation_optimum_universality_v1.py",
            "participation_optimum_mes_universality_v1.py",
            "participation_optimum_mgc_universality_v1.py",
            "participation_shape_cross_instrument_v1.py",
            "pr48_participation_shape_oos_replication_v1.py",
            "q1_h04_mechanism_shape_validation_v1.py",
            "close_h2_book_path_c.py",
            "h2_exploitation_audit.py",
            "audit_comex_settle_orb_g5_failure_pocket.py",
            "rel_vol_is_only_quantile_sensitivity.py",
            "rel_vol_mechanism_decomposition.py",
            "research_vol_regime_wf.py",
            "stress_test_rel_vol_finding.py",
            "stress_test_rel_vol_finding_v2.py",
            "t0_t8_audit_volume_cells.py",
            "vwap_comprehensive_family_scan.py",
            "research_mgc_e2_microstructure_pilot.py",
        }
    )

    # Canonical contamination patterns from `daily_features` columns.
    # `rel_vol_<UPPER>` matches `rel_vol_NYSE_OPEN` etc but NOT `rel_vol_session_norm`
    # (a script-local clean-replacement name). break_bar / break_delay substrings
    # are unambiguous — no clean features share those substrings.
    # `orb_\w+_break_dir` added 2026-04-28: break_dir is post-entry on ~42% of E2
    # fills (range-touch-then-reverse fakeouts) — same class as break_bar_volume.
    # See backtesting-methodology.md § 6.3 and postmortem 2026-04-21-e2-break-bar-lookahead.md.
    tainted_feature_re = re.compile(
        r"\brel_vol_[A-Z]"  # rel_vol_NYSE_OPEN, rel_vol_TOKYO_OPEN, etc
        r"|break_bar_volume"
        r"|break_bar_continues"
        r"|break_delay_min"
        r"|orb_\w+_break_dir"  # post-entry on ~42% of E2 fills when used as selector
    )
    marker_prefix = "# e2-lookahead-policy:"
    valid_markers = ("cleared", "late-fill-only", "not-predictor", "tainted")

    # Match permissively: any file mentioning entry_model AND an 'E2' / "E2"
    # literal AND a tainted feature. This catches kwarg form (entry_model='E2'),
    # SQL form (entry_model = 'E2' with spaces), and dict form ('entry_model':
    # 'E2'). False positives are tolerable because the check is advisory.
    e2_literals = ("'E2'", '"E2"')

    violations: list[str] = []
    for py_file in sorted(research_dir.rglob("*.py")):
        try:
            rel = py_file.relative_to(research_dir)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] == "archive":
            continue
        if py_file.name in known_tainted_registry:
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        if "entry_model" not in content:
            continue
        if not any(lit in content for lit in e2_literals):
            continue
        if not tainted_feature_re.search(content):
            continue

        marker_found = False
        for line in content.splitlines():
            stripped = line.strip().lower()
            idx = stripped.find(marker_prefix)
            if idx < 0:
                continue
            value = stripped[idx + len(marker_prefix) :].strip()
            if any(value.startswith(m) for m in valid_markers):
                marker_found = True
                break

        if not marker_found:
            rel_repo = py_file.relative_to(PROJECT_ROOT)
            violations.append(
                f"  {rel_repo}: combines entry_model='E2' with break-bar feature "
                f"(rel_vol/break_bar_volume/break_bar_continues/break_delay_min) "
                f"but no '# e2-lookahead-policy: cleared|late-fill-only|"
                f"not-predictor|tainted' annotation. ~41% of E2 trades have "
                f"entry_ts<break_ts; break-bar features are post-entry on that "
                f"subset. See docs/audit/results/"
                f"2026-04-28-e2-lookahead-contamination-registry.md."
            )

    return violations


# ── iso_utc silent-None formatter class bug (2026-05-04) ──
# Class memo: memory/feedback_iso_utc_silent_none_class_pattern.md
# Pattern: operator-visible formatter helper takes Any → does isinstance branch
# → returns formatted on match, returns None on miss without log.warning. Result
# is silent type drops on the live dashboard / state file. Canonical fix:
# `_iso_utc` in trading_app/live/bot_state.py logs a warning on the unsupported
# branch (line 95). This check enforces the same shape on related operator-
# visible files. Annotation `# silent-none-policy: <reason>` exempts a function
# whose silent-None tail is intentional (e.g., try/except-pass publishers).
_ISO_UTC_FORMATTER_SCAN_FILES = (
    "trading_app/live/bot_state.py",
    "trading_app/live/bot_dashboard.py",
    "trading_app/pre_session_check.py",
    "scripts/tools/live_readiness_report.py",
    "trading_app/lifecycle_state.py",
    "trading_app/live/session_orchestrator.py",
    "trading_app/lane_allocator.py",
)
_SILENT_NONE_POLICY_MARKER = "# silent-none-policy:"


def _function_has_isinstance_then_silent_none(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Detect the iso_utc silent-None formatter shape on a function body.

    Returns True iff ALL of:
      1. function body contains at least one ``isinstance(...)`` call (anywhere)
      2. last statement of body is an explicit ``return None`` or bare
         ``return`` (matches the canonical ``_iso_utc`` shape — explicit
         silent-None tail). Implicit fall-off-end is not flagged because
         most functions whose last statement is a ``try``/``with``/``for``
         actually return values from inner branches and AST cannot prove
         the tail is reachable.
      3. function body contains NO ``log.warning|critical|error`` /
         ``logger.*`` / ``self.log.*`` / ``cls.log.*`` call AND NO
         ``raise`` statement at any nesting level

    The combined predicate distinguishes the silent-formatter shape from the
    pure null-passthrough idiom (``if v is None: return None; return str(v)``)
    which has no isinstance call.
    """
    body = func_node.body
    if not body:
        return False

    last_stmt = body[-1]
    if not isinstance(last_stmt, ast.Return):
        return False
    if last_stmt.value is not None:
        if not (isinstance(last_stmt.value, ast.Constant) and last_stmt.value.value is None):
            return False

    has_isinstance = False
    has_warn_or_raise = False

    for sub in ast.walk(func_node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name) and func.id == "isinstance":
                has_isinstance = True
            elif isinstance(func, ast.Attribute) and func.attr in {"warning", "critical", "error"}:
                # log.warning / logger.warning / self.log.warning / cls.log.error etc
                value = func.value
                target_names = {"log", "logger"}
                if (isinstance(value, ast.Name) and value.id in target_names) or (
                    isinstance(value, ast.Attribute) and value.attr in target_names
                ):
                    has_warn_or_raise = True
        elif isinstance(sub, ast.Raise):
            has_warn_or_raise = True

    return has_isinstance and not has_warn_or_raise


def check_routine_tbbo_slippage_registry_coverage() -> list[str]:
    """Every PASS routine-TBBO slippage pilot must be in ROUTINE_TBBO_SLIPPAGE_REGISTRY.

    `trading_app/deployability.py` reads
    ``ROUTINE_TBBO_SLIPPAGE_REGISTRY`` to decide whether a row's missing
    ``slippage_validation_status`` should be inferred as
    ``PENDING_EVENT_TAIL`` (controlled-pilot eligible, warning) versus
    ``slippage_missing`` (hard-blocked). The registry is the single source
    of truth for "this instrument's routine-liquidity slippage pilot has
    shipped a PASS."

    This check globs ``docs/audit/results/*slippage*pilot*v1*.md``,
    parses the verdict line, and fails closed when:

      * a pilot doc carries verdict ``PASS`` but its instrument is absent
        from the registry — silent under-coverage, qualifying rows would
        be over-blocked as ``slippage_missing`` despite shipped evidence;
      * a pilot doc carries verdict ``WARN`` or ``FAIL`` but its instrument
        IS in the registry — silent over-coverage, qualifying rows would
        be inferred-passed despite shipped evidence saying otherwise.

    Verdict format is the project convention used by both pilot docs:
    a level-2 markdown heading ``## Verdict: **PASS**`` (or WARN/FAIL).
    Surrounding markdown bold (``**``) is tolerated; surrounding spaces
    are tolerated; case is tolerated.

    Pilot doc filename convention: ``YYYY-MM-DD-<instrument>-...slippage-pilot-v1.md``
    where ``<instrument>`` is the lowercase ticker (mnq/mes/mgc).

    @rule routine-tbbo-slippage-registry
    """

    violations: list[str] = []
    results_dir = PROJECT_ROOT / "docs" / "audit" / "results"
    if not results_dir.is_dir():
        return violations

    try:
        from trading_app.deployability import ROUTINE_TBBO_SLIPPAGE_REGISTRY
    except Exception as exc:  # pragma: no cover - import-side surfaces in upstream check
        violations.append(
            f"  trading_app.deployability.ROUTINE_TBBO_SLIPPAGE_REGISTRY import failed: {type(exc).__name__}: {exc}"
        )
        return violations

    registry_instruments = {key.upper() for key in ROUTINE_TBBO_SLIPPAGE_REGISTRY}

    pilot_re = re.compile(
        r"^(?P<date>\d{4}-\d{2}-\d{2})-(?P<instrument>[a-z0-9]+)-.*slippage[-_]?pilot[-_]?v1\.md$",
        re.IGNORECASE,
    )
    verdict_re = re.compile(
        r"^##\s*Verdict\s*:\s*\**\s*(?P<verdict>PASS|WARN|FAIL)\b",
        re.IGNORECASE | re.MULTILINE,
    )

    pilot_docs: list[tuple[str, str, str, Path]] = []  # (instrument, date, verdict, path)
    for path in sorted(results_dir.glob("*slippage*pilot*v1*.md")):
        match = pilot_re.match(path.name)
        if match is None:
            violations.append(
                f"  pilot doc {path.name} does not match expected filename pattern "
                f"`YYYY-MM-DD-<instrument>-...slippage-pilot-v1.md`; cannot be classified."
            )
            continue
        instrument = match.group("instrument").upper()
        date_str = match.group("date")  # YYYY-MM-DD, sortable as string
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            violations.append(f"  failed to read {path.name}: {exc}")
            continue
        verdict_match = verdict_re.search(text)
        if verdict_match is None:
            violations.append(
                f"  pilot doc {path.name} has no `## Verdict: **PASS|WARN|FAIL**` line; cannot be classified."
            )
            continue
        pilot_docs.append((instrument, date_str, verdict_match.group("verdict").upper(), path))

    # Per instrument the LATEST pilot v1 doc (by filename date) is authoritative.
    # A stale older PASS does not justify a registry entry if a newer WARN/FAIL
    # refutes it; equally, a stale older WARN does not block a newer PASS.
    # Filename date is YYYY-MM-DD so string sort = chronological sort.
    latest_by_instrument: dict[str, tuple[str, str, Path]] = {}  # inst -> (date, verdict, path)
    for instrument, date_str, verdict, path in pilot_docs:
        prior = latest_by_instrument.get(instrument)
        if prior is None or date_str > prior[0]:
            latest_by_instrument[instrument] = (date_str, verdict, path)

    # Symmetric registry-vs-evidence assertion: one loop per direction, each
    # iterating the registry / evidence side and demanding the other side
    # carries matching support. Earlier set-difference loops missed the
    # registered-but-no-doc case (silent over-coverage). Same class of bug
    # the chronology fix closed; this loop structure prevents recurrence.

    # Direction 1: every PASS pilot must have a matching registry entry.
    for inst, (date_str, verdict, path) in sorted(latest_by_instrument.items()):
        if verdict != "PASS":
            continue
        if inst in registry_instruments:
            continue
        violations.append(
            f"  routine-TBBO slippage pilot v1 PASS for {inst} (latest: {path.name} dated {date_str}) "
            f"but {inst} is missing from trading_app.deployability."
            f"ROUTINE_TBBO_SLIPPAGE_REGISTRY. Add a RoutineTbboPilot entry covering the "
            f"pilot's session set so deployability infers PENDING_EVENT_TAIL instead of "
            f"hard-blocking on slippage_missing."
        )

    # Direction 2: every registry entry must be backed by a current PASS pilot.
    # Catches both (a) registered with latest verdict WARN/FAIL (chronology
    # fix), and (b) registered with NO matching pilot doc at all (no evidence,
    # registry entry with nothing supporting it).
    for inst in sorted(registry_instruments):
        latest = latest_by_instrument.get(inst)
        if latest is None:
            violations.append(
                f"  {inst} is registered in ROUTINE_TBBO_SLIPPAGE_REGISTRY but no "
                f"`{inst.lower()}-...slippage-pilot-v1.md` doc was found in "
                f"docs/audit/results/. Either commit the pilot v1 evidence doc that "
                f"justifies this registry entry, or remove the entry — registry membership "
                f"without committed evidence is silent over-coverage."
            )
            continue
        date_str, verdict, path = latest
        if verdict == "PASS":
            continue
        violations.append(
            f"  {inst} is registered in ROUTINE_TBBO_SLIPPAGE_REGISTRY but its LATEST "
            f"slippage pilot v1 doc is non-PASS ({path.name} dated {date_str}, verdict={verdict}). "
            f"Remove the registry entry or land a newer PASS pilot v1 for {inst} before "
            f"continuing to infer PENDING_EVENT_TAIL on its rows."
        )

    return violations


def check_iso_utc_formatter_silent_none() -> list[str]:
    """Operator-visible formatter helpers must warn on unrecognized types.

    Class bug catalogued 2026-05-04 (memo:
    ``memory/feedback_iso_utc_silent_none_class_pattern.md``): formatter
    helpers that take ``Any`` and silently return ``None`` on the off-branch
    of an ``isinstance`` test let upstream type drift propagate to the
    operator surface (``bot_state.json``, dashboard, lifecycle, allocator)
    as missing fields rather than visible warnings.

    Canonical fix: ``trading_app/live/bot_state.py:_iso_utc`` logs a warning
    before returning None on the unsupported-type branch (line 95). The check
    fires on functions in the scan set whose body contains an ``isinstance``
    call AND falls off the end / returns None at the tail AND has no
    ``log.warning|critical|error`` (or ``logger.*``, ``self.log.*``,
    ``cls.log.*``) call AND no ``raise``.

    Annotation exemption: prepend
    ``# silent-none-policy: <reason>`` on any line within the function (or
    on the line above its ``def``) to declare an intentional silent-None
    tail (e.g., ``try/except: pass`` publishers, cleanup utilities). The
    reason is documentation for future readers.

    @rule iso-utc-silent-none-formatter
    @class-memo memory/feedback_iso_utc_silent_none_class_pattern.md
    """
    violations: list[str] = []
    for rel_path in _ISO_UTC_FORMATTER_SCAN_FILES:
        py_file = PROJECT_ROOT / rel_path
        if not py_file.is_file():
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        source_lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not _function_has_isinstance_then_silent_none(node):
                continue

            # Annotation exemption: scan from one line above the def through
            # the end of the function body. Match the marker prefix literally.
            start = max(0, node.lineno - 2)
            end_lineno = getattr(node, "end_lineno", node.lineno) or node.lineno
            end = min(len(source_lines), end_lineno)
            window = source_lines[start:end]
            if any(_SILENT_NONE_POLICY_MARKER in line for line in window):
                continue

            violations.append(
                f"  {rel_path}:{node.lineno} {node.name}: operator-visible formatter "
                f"with isinstance + silent None tail and no log.warning|critical|error / "
                f"raise. Add `log.warning(...)` before the tail return OR annotate the "
                f"def with `# silent-none-policy: <reason>` if the silent tail is "
                f"intentional. Class memo: "
                f"memory/feedback_iso_utc_silent_none_class_pattern.md."
            )

    return violations


def check_parked_cells_registry_completeness() -> list[str]:
    """Every Pathway B / Phase D result file must have a matching parked-cells.yaml entry.

    Authored 2026-04-29 to close the structural gap where parked Phase D cells
    (PARK_PENDING_OOS_POWER, KILL, PARK_CONDITIONAL_DEPLOY_RETAINED, etc.) lived
    in 4 disjoint surfaces (validated_setups, experimental_strategies,
    docs/audit/results/, docs/runtime/decision-ledger.md) with no code-enforced
    invariant binding them. A cell could go missing or drift between surfaces
    and the only detection was human review.

    The fix: docs/runtime/parked-cells.yaml is the single durable registry of
    every cell with a locked Pathway B verdict. This drift check enforces three
    invariants:

      1. Every result file under docs/audit/results/*pathway-b*-result.md or
         docs/audit/results/*-d0-v2-*backtest.md (Phase D backtest variants) that
         contains a verdict token in {PARK_PENDING_OOS_POWER,
         PARK_CONDITIONAL_DEPLOY_RETAINED, PARK_ABSOLUTE_FLOOR_FAIL, KILL,
         CANDIDATE_READY_FOR_PHASE_2, KEEP_PARKED_INDEFINITELY} MUST have a
         corresponding cell entry in parked-cells.yaml with `result:` pointing
         at it.

      2. Every cell entry's `pre_reg:` and `result:` paths MUST exist on disk.
         Catches the failure mode where a result is renamed or moved and the
         registry goes stale silently.

      3. Every cell entry's `status:` token MUST appear in its result file.
         Catches the failure mode where the registry says PARK but the result
         file says KILL (e.g., a successor pre-reg moved the verdict and the
         registry was not updated).

    Authority: `docs/runtime/parked-cells.yaml` schema_version=1.
    Origin: 2026-04-29 D6 session — user asked "do we have an automatic
    system that keeps track of the parked trades, so we don't lose them
    randomly over the repo and shit." This check is the answer.

    BLOCKING (not advisory): registry drift is a real correctness issue.
    """
    issues: list[str] = []

    registry_path = PROJECT_ROOT / "docs" / "runtime" / "parked-cells.yaml"
    results_dir = PROJECT_ROOT / "docs" / "audit" / "results"
    if not registry_path.exists():
        issues.append(f"  parked-cells registry missing: {registry_path.relative_to(PROJECT_ROOT)}")
        return issues
    if not results_dir.is_dir():
        return issues  # nothing to check

    # Permissive YAML parse (avoid pyyaml dep — minimal flat-block reader).
    # We only need the `cells:` list of dicts with `cell_id`, `status`,
    # `pre_reg`, `result` keys.
    try:
        registry_text = registry_path.read_text(encoding="utf-8")
    except OSError as e:
        issues.append(f"  parked-cells registry unreadable: {e}")
        return issues

    # Parse `cells:` block — each entry begins with "  - cell_id:" at 2-space indent.
    # Extract the four fields we care about per entry.
    valid_status_tokens = {
        "PARK_PENDING_OOS_POWER",
        "PARK_CONDITIONAL_DEPLOY_RETAINED",
        "PARK_ABSOLUTE_FLOOR_FAIL",
        "KILL",
        "CANDIDATE_READY_FOR_PHASE_2",
        "KEEP_PARKED_INDEFINITELY",
    }
    cells: list[dict[str, str]] = []
    current: dict[str, str] = {}
    in_cells_block = False
    for raw_line in registry_text.splitlines():
        if raw_line.startswith("cells:"):
            in_cells_block = True
            continue
        if not in_cells_block:
            continue
        if raw_line.startswith("events:") or (
            raw_line and not raw_line.startswith(" ") and not raw_line.startswith("#")
        ):
            # Left the cells block
            if current:
                cells.append(current)
                current = {}
            in_cells_block = False
            continue
        stripped = raw_line.strip()
        if stripped.startswith("- cell_id:"):
            if current:
                cells.append(current)
            current = {"cell_id": stripped.partition(":")[2].strip().strip('"').strip("'")}
            continue
        for field in ("status", "pre_reg", "result"):
            prefix = f"{field}:"
            if stripped.startswith(prefix) and field not in current:
                value = stripped[len(prefix) :].strip().strip('"').strip("'")
                if value and value not in ("|", ">", "null", "None"):
                    current[field] = value
    if current:
        cells.append(current)

    # Build {result_path -> cell} map for invariant 1 reverse lookup
    result_to_cell: dict[str, dict] = {}
    for cell in cells:
        cell_id = cell.get("cell_id", "<unknown>")
        # Invariant 2: paths must exist
        for path_field in ("pre_reg", "result"):
            rel = cell.get(path_field, "")
            if not rel:
                issues.append(f"  parked-cells: cell {cell_id!r} missing required field {path_field!r}")
                continue
            full = PROJECT_ROOT / rel
            if not full.exists():
                issues.append(f"  parked-cells: cell {cell_id!r} {path_field}={rel!r} does NOT exist on disk")
        result = cell.get("result")
        if result:
            result_to_cell[result] = cell
        # Invariant 3: status token must appear in result file
        status = cell.get("status", "")
        if status and result:
            full_result = PROJECT_ROOT / result
            if full_result.exists():
                try:
                    body = full_result.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    body = ""
                if status not in body:
                    issues.append(
                        f"  parked-cells: cell {cell_id!r} status={status!r} does NOT appear in {result!r} "
                        f"(registry-result divergence — verdict moved or registry stale)"
                    )
        if status and status not in valid_status_tokens:
            issues.append(
                f"  parked-cells: cell {cell_id!r} status={status!r} not in canonical set {sorted(valid_status_tokens)}"
            )

    # Invariant 1: every result file with a verdict token must have a registry entry
    surfaced_patterns = [
        "*pathway-b*-result.md",
        "*-d0-v2-*backtest.md",
    ]
    surfaced_files: set[Path] = set()
    for pat in surfaced_patterns:
        for path in results_dir.glob(pat):
            surfaced_files.add(path)
    for result_path in sorted(surfaced_files):
        try:
            text = result_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Verdict token must be present (otherwise file is not a result-doc; skip)
        verdict_present = any(tok in text for tok in valid_status_tokens)
        if not verdict_present:
            continue
        rel_path = result_path.relative_to(PROJECT_ROOT).as_posix()
        if rel_path not in result_to_cell:
            issues.append(
                f"  parked-cells: result file {rel_path!r} contains a Pathway B verdict but has "
                f"NO matching cell entry in parked-cells.yaml. Add an entry referencing this "
                f"result + its pre_reg path."
            )

    return issues


_STAGE_ACCEPTANCE_CMD_ALLOWLIST = ("pytest", "python", "ls", "grep", "test")


def _parse_stage_acceptance_commands(text: str) -> list[str]:
    """Extract runnable acceptance commands from a stage file body.

    Handles two shapes:
      (1) YAML ``acceptance:`` list under frontmatter — bullets that begin
          with one of the allowlisted commands.
      (2) ``## Acceptance`` markdown section — bullets containing backtick-
          quoted commands like `` `pytest tests/...` `` or
          `` `python pipeline/check_drift.py` ``.

    Returns commands as raw strings; caller validates against the allowlist
    again before subprocess-shelling. Anything containing shell metacharacters
    that suggest mutation (``rm``, ``>``, ``|``, ``;``, ``&``, ``$(``) is
    dropped.
    """
    commands: list[str] = []
    bad_chars = (">", "|", ";", "&", "$(", "`rm ", " rm ", "rm -")
    # Shape 2: markdown section — pull every backtick-fenced span and keep
    # ones that start with an allowlisted command.
    in_acceptance = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.lower().startswith("## acceptance"):
            in_acceptance = True
            continue
        if in_acceptance and stripped.startswith("## "):
            in_acceptance = False
            continue
        if in_acceptance:
            # backtick-fenced command on a bullet line
            if "`" in stripped:
                segments = stripped.split("`")
                for idx in range(1, len(segments), 2):
                    cmd = segments[idx].strip()
                    if not cmd:
                        continue
                    head = cmd.split()[0] if cmd.split() else ""
                    if head in _STAGE_ACCEPTANCE_CMD_ALLOWLIST and not any(b in cmd for b in bad_chars):
                        commands.append(cmd)
    # Shape 1: YAML acceptance: list — only fires if frontmatter contained
    # the key. Detect by scanning for a top-level "acceptance:" line followed
    # by indented "- " bullets that contain backtick-fenced commands.
    lines = text.splitlines()
    for i, raw in enumerate(lines):
        if raw.strip() == "acceptance:" or raw.strip().startswith("acceptance:"):
            for j in range(i + 1, min(i + 40, len(lines))):
                item = lines[j]
                if item and not item.startswith((" ", "\t", "-")):
                    break  # left the YAML block
                stripped_item = item.strip()
                if not stripped_item.startswith("-"):
                    continue
                # Look for backtick-fenced command in the bullet text
                if "`" in stripped_item:
                    segs = stripped_item.split("`")
                    for k in range(1, len(segs), 2):
                        cmd = segs[k].strip()
                        head = cmd.split()[0] if cmd.split() else ""
                        if head in _STAGE_ACCEPTANCE_CMD_ALLOWLIST and not any(b in cmd for b in bad_chars):
                            commands.append(cmd)
    return commands


def _stage_acceptance_all_pass(commands: list[str], timeout_s: int = 5) -> tuple[bool, int]:
    """Run each acceptance command and return (all_passed, count_run).

    Returns (False, 0) on empty input — caller treats as "no parseable
    acceptance", falls through to slug-grep heuristic.
    Returns (False, n) if any command exits non-zero or times out.
    Returns (True, n) only if every command exits 0.
    """
    if not commands:
        return (False, 0)
    ran = 0
    for cmd in commands:
        head = cmd.split()[0] if cmd.split() else ""
        if head not in _STAGE_ACCEPTANCE_CMD_ALLOWLIST:
            return (False, ran)
        try:
            # shlex.split for safety; never shell=True
            import shlex

            argv = shlex.split(cmd)
            if argv and argv[0] == "python":
                argv[0] = sys.executable
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=str(PROJECT_ROOT),
            )
            ran += 1
            if result.returncode != 0:
                return (False, ran)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return (False, ran)
    return (True, ran)


def check_stage_file_landed_drift() -> list[str]:
    """ADVISORY: stage files claiming work-in-progress while either (a) their
    own acceptance criteria all pass when executed, or (b) git log shows
    landings.

    Catches the "I thought we sorted this already" failure mode where a
    stage file under docs/runtime/stages/ describes work as in-progress
    but the actual artifacts are already on disk and tests already pass.
    Without this surface, /next and /orient resume already-landed work,
    wasting cycles.

    Two detection modes (file fires advisory if EITHER trips):

      Mode A — acceptance verification (added 2026-05-05):
        Parse the stage's ``acceptance:`` YAML list or ``## Acceptance``
        markdown section. Extract backtick-fenced commands whose head is
        in ``_STAGE_ACCEPTANCE_CMD_ALLOWLIST``. Execute each with a 5s
        timeout. If ≥1 command ran AND all commands exit 0 AND the stage
        is ≥3 days old, emit advisory.

      Mode B — slug-grep landings (original 2026-04-28 behavior):
        If `updated:` ≥7 days old, count git commits since whose messages
        reference the stage's `slug:`. If ≥3, emit advisory.

    Files without YAML frontmatter, auto_trivial.md, and files flagged
    operator-gated in body ("Status: implemented", "operator-gated") are
    skipped in both modes.

    Why Mode A was added: 2026-05-05 manual /next pass discovered 13 stale
    stages — most lacked a `slug:` field OR landed via single squash-merge
    PR (so slug-grep saw 1 commit, not ≥3) OR were under the 7-day age
    floor. Mode A closes those gaps by trusting the stage's own acceptance
    contract instead of inferring from commit archaeology.
    """
    issues: list[str] = []
    stages_dir = PROJECT_ROOT / "docs" / "runtime" / "stages"
    if not stages_dir.exists():
        return issues

    import datetime as _dt

    today = _dt.date.today()
    for stage_file in sorted(stages_dir.glob("*.md")):
        if stage_file.name == "auto_trivial.md":
            continue
        try:
            text = stage_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.startswith("---"):
            continue  # plain stage file (no frontmatter); skip
        end = text.find("\n---", 3)
        if end < 0:
            continue
        fm_text = text[3:end]
        fm: dict[str, str] = {}
        for raw in fm_text.splitlines():
            if ":" in raw and not raw.lstrip().startswith("#"):
                k, _, v = raw.partition(":")
                fm[k.strip()] = v.strip().strip('"').strip("'")
        slug = fm.get("slug", "").strip()
        updated_str = fm.get("updated", "").strip()
        if not updated_str:
            continue  # Mode A and Mode B both need an `updated:` date
        # Operator-gated exemption — body declares it's waiting on a person.
        body_lower = text[end:].lower()
        if (
            "operator-gated" in body_lower
            or "operator action" in body_lower
            or "operator observation" in body_lower
            or ("status:" in body_lower and "implemented" in body_lower and "pending" in body_lower)
        ):
            continue
        try:
            updated_date = _dt.date.fromisoformat(updated_str[:10])
        except ValueError:
            continue
        age_days = (today - updated_date).days

        # ── Mode A: acceptance-verification (≥3 days old, runnable acceptance) ──
        if age_days >= 3:
            commands = _parse_stage_acceptance_commands(text)
            all_pass, n_ran = _stage_acceptance_all_pass(commands)
            if all_pass and n_ran >= 1:
                issues.append(
                    f"  {stage_file.relative_to(PROJECT_ROOT).as_posix()}: "
                    f"updated {age_days}d ago, all {n_ran} acceptance command(s) "
                    f"exit 0 — close the stage"
                )
                continue  # don't double-fire on Mode B

        # ── Mode B: slug-grep landings (≥7 days old) ──
        if age_days < 7:
            continue
        if not slug:
            continue  # Mode B requires slug; Mode A above already ran
        try:
            result = subprocess.run(
                [
                    "git",
                    "log",
                    f"--since={updated_date.isoformat()}",
                    "--all",
                    "--oneline",
                    f"--grep={slug}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(PROJECT_ROOT),
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        commit_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        if len(commit_lines) >= 3:
            issues.append(
                f"  {stage_file.relative_to(PROJECT_ROOT).as_posix()}: "
                f"updated {age_days}d ago, {len(commit_lines)} commits since "
                f"reference slug '{slug}' — verify still active or close stage"
            )
    return issues


# ── CRG-backed drift checks (D1-D5) ──────────────────────────────────────
# All checks are ADVISORY (is_advisory=True): when CRG is unavailable they
# print "ADVISORY: CRG unavailable" and return without blocking commits.
# Authority: docs/plans/2026-04-29-crg-integration-spec.md Phase 2.


def check_crg_cross_layer_surprising_connections() -> list[str]:
    """D1: Catch surprising cross-layer edges between pipeline/ and trading_app/.

    Uses ``code_review_graph.tools.analysis_tools.get_surprising_connections_func``
    to fetch CRG's full surprise list, then filters to edges where one endpoint
    is in pipeline/ and the other in trading_app/ AND neither endpoint is a
    canonical surface (SESSION_CATALOG, COST_SPECS, ACTIVE_ORB_INSTRUMENTS, etc.).

    ADVISORY: emits warnings; never blocks when CRG is unavailable.
    """
    from pipeline.check_drift_crg_helpers import (
        CRG_UNAVAILABLE,
        crg_is_available,
        get_surprising_connections,
    )

    if not crg_is_available():
        print("  ADVISORY: CRG unavailable — run `code-review-graph build` to enable cross-layer check")
        return []

    result = get_surprising_connections(top_n=50)
    if result is CRG_UNAVAILABLE:
        print("  ADVISORY: CRG query failed — cross-layer check skipped")
        return []

    # Canonical surfaces — when an edge passes through one of these, it's an
    # expected cross-layer link, not a Project Truth Protocol violation.
    canonical_surfaces = (
        "/pipeline/dst.py",
        "/pipeline/cost_model.py",
        "/pipeline/asset_configs.py",
        "/pipeline/paths.py",
        "/trading_app/holdout_policy.py",
        "/trading_app/eligibility/builder.py",
    )

    def _norm(qn: str) -> str:
        return str(qn).replace("\\", "/")

    surprises = []
    for conn in result:
        # CRG returns flat dicts with source_qualified / target_qualified holding
        # absolute "file::symbol" paths.  Older formats may use source/target as
        # nested dicts — we tolerate both for forward compatibility.
        src_q = _norm(conn.get("source_qualified", ""))
        tgt_q = _norm(conn.get("target_qualified", ""))
        if not src_q or not tgt_q:
            src_field, tgt_field = conn.get("source"), conn.get("target")
            if isinstance(src_field, dict):
                src_q = _norm(src_field.get("file_path", src_field.get("path", "")))
            if isinstance(tgt_field, dict):
                tgt_q = _norm(tgt_field.get("file_path", tgt_field.get("path", "")))

        # Only flag pipeline <-> trading_app crossings
        is_cross = ("/pipeline/" in src_q and "/trading_app/" in tgt_q) or (
            "/trading_app/" in src_q and "/pipeline/" in tgt_q
        )
        if not is_cross:
            continue

        # Skip if either endpoint is a known canonical surface
        if any(surf in src_q or surf in tgt_q for surf in canonical_surfaces):
            continue

        score = conn.get("surprise_score", conn.get("score", "?"))

        # Trim absolute project-root prefix for readability. Derived from
        # PROJECT_ROOT (not a hardcoded directory name) so it works in any
        # worktree or alternate checkout location.
        root_prefix = str(PROJECT_ROOT).replace("\\", "/").rstrip("/") + "/"

        def _rel(qn: str, _prefix: str = root_prefix) -> str:
            return qn[len(_prefix) :] if qn.startswith(_prefix) else qn

        surprises.append(f"  Surprising cross-layer edge: {_rel(src_q)} -> {_rel(tgt_q)} (score={score})")

    if surprises:
        print(f"  ADVISORY: {len(surprises)} surprising pipeline/trading_app edge(s) found")
        for s in surprises[:10]:  # cap output
            print(s)
    return []  # advisory — never blocks


def check_crg_canonical_import_enforcement() -> list[str]:
    """D2: Research scripts must import canonical functions, not re-implement them.

    AST-based check: walks research/ scripts, finds any that DEFINE a function
    whose name matches a canonical-source name (parse_strategy_id, orb_utc_window,
    etc.) AND do not import the canonical version. CRG can't express this purely
    as a graph query (no IMPORTS_FROM-with-name-collision pattern), so we use
    Python's ast module directly.  This check has no CRG dependency and runs
    even when CRG is unavailable.

    ADVISORY: emits warnings; never blocks.
    """
    # Functions whose canonical module is verified to actually contain the
    # definition. Audit trail vs spec original list of 5:
    #   * ``reprice_e2_entry`` — lives in ``research/databento_microstructure.py``,
    #     not yet promoted to pipeline.cost_model. Flagging research/ as
    #     "re-implementing" itself would be a false positive. Excluded.
    #   * ``_load_strategy_outcomes`` — canonical in
    #     ``trading_app.strategy_fitness``; consumed by 27 files including 4
    #     research scripts that all import correctly today. Included for
    #     forward coverage.
    canonical: dict[str, str] = {
        "parse_strategy_id": "trading_app.eligibility.builder",
        "orb_utc_window": "pipeline.dst",
        "detect_break_touch": "trading_app.entry_rules",
        "_load_strategy_outcomes": "trading_app.strategy_fitness",
    }

    if not RESEARCH_DIR.is_dir():
        return []

    violations: list[str] = []
    for py_file in RESEARCH_DIR.rglob("*.py"):
        # Skip archive — frozen, exempt
        if "archive" in py_file.parts:
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

        defined: set[str] = set()
        imported_from: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if node.name in canonical:
                    defined.add(node.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    imported_from[alias.name] = node.module

        for func, canon_module in canonical.items():
            if func in defined:
                src_of_import = imported_from.get(func)
                if src_of_import != canon_module:
                    rel = py_file.relative_to(PROJECT_ROOT).as_posix()
                    violations.append(f"  {rel}: locally defines `{func}` (canonical: import from {canon_module})")

    if violations:
        print(f"  ADVISORY: {len(violations)} canonical-import violation(s) in research/")
        for v in violations[:20]:
            print(v)
    return []  # advisory


def check_crg_canonical_functions_have_tests() -> list[str]:
    """D3: Canonical functions must each have at least one test that imports them.

    AST-based check: walks tests/ tree and records every (module, symbol)
    imported via ``from <module> import <symbol>`` (or ``import <module>``)
    AND actually called somewhere in the file.  A canonical function is
    flagged if NO test file imports-and-calls it.

    Replaces an earlier graph-based version that called
    ``query_graph(pattern="tests_for")`` directly. CRG v2.1.0's ``tests_for``
    pattern is incomplete: it returns 0 results for functions that ARE tested
    (verified 2026-04-29: ``reprice_e2_entry``, ``parse_strategy_id``,
    ``detect_break_touch`` all have unit tests but graph reports 0 edges).
    AST scan is the correctness-first alternative.

    SCOPE — callables only.
    The Phase 2 spec originally listed canonical CONSTANTS too
    (SESSION_CATALOG, COST_SPECS, ACTIVE_ORB_INSTRUMENTS, HOLDOUT_SACRED_FROM).
    AST cannot detect ``Call`` nodes against constants — they are referenced,
    not called — so this check covers callables only. Constant test-coverage
    is a separate, currently-unbuilt drift-check class; tracked as
    follow-up in spec §"Open follow-ups". D1 (surprising connections)
    intentionally SKIPS canonical surfaces, so it does not cover constants
    either; do not assume D1 fills the gap.

    ADVISORY: emits warnings; never blocks. No CRG dependency — runs even
    when CRG is uninstalled.
    """
    # symbol → canonical module (only entries whose canonical module is
    # verified to contain the definition — see D2 docstring for the
    # reprice_e2_entry exclusion rationale).
    canonical_callable: dict[str, str] = {
        "orb_utc_window": "pipeline.dst",
        "parse_strategy_id": "trading_app.eligibility.builder",
        "detect_break_touch": "trading_app.entry_rules",
    }

    tests_dir = PROJECT_ROOT / "tests"
    if not tests_dir.is_dir():
        return []

    # Map canonical symbol -> set of test files that import-and-call it
    coverage: dict[str, set[str]] = {sym: set() for sym in canonical_callable}

    for py_file in tests_dir.rglob("test_*.py"):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue

        # Track which canonical symbols this test imports from canonical modules
        imported_canonical: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    name = alias.name
                    if name in canonical_callable and node.module == canonical_callable[name]:
                        imported_canonical.add(name)

        if not imported_canonical:
            continue

        # Verify at least one Call node references the imported symbol.  This
        # filters out import-only files (e.g., re-exports) that don't actually
        # exercise the function.
        called: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in imported_canonical:
                    called.add(func.id)

        rel = py_file.relative_to(PROJECT_ROOT).as_posix()
        for sym in called:
            coverage[sym].add(rel)

    untested: list[str] = []
    for sym, canon_module in canonical_callable.items():
        if not coverage[sym]:
            untested.append(f"  No test imports-and-calls `{canon_module}.{sym}`")

    if untested:
        print(f"  ADVISORY: {len(untested)} canonical function(s) lack import-and-call test coverage")
        for u in untested:
            print(u)
    return []  # advisory


def check_crg_canonical_path_function_size() -> list[str]:
    """D4: Canonical-path functions must not exceed 200 lines (monster-function cap).

    Fetches all functions exceeding 200 lines (no path filter — CRG's
    ``file_path_pattern`` is a substring match against the stored path, which
    is brittle on Windows where paths use backslashes), then filters
    client-side via ``Path.parts`` to ``pipeline/`` and ``trading_app/`` only.

    ADVISORY: emits warnings; never blocks when CRG is unavailable.
    """
    from pipeline.check_drift_crg_helpers import (
        CRG_UNAVAILABLE,
        crg_is_available,
        find_large_functions,
    )

    if not crg_is_available():
        print("  ADVISORY: CRG unavailable — function-size cap check skipped")
        return []

    result = find_large_functions(min_lines=200, limit=200)
    if result is CRG_UNAVAILABLE:
        print("  ADVISORY: CRG query failed — function-size cap check skipped")
        return []

    canonical_prefixes = ("pipeline", "trading_app")
    flagged: list[dict] = []
    for fn in result:
        rel_path = fn.get("relative_path", "")
        # Normalize to forward-slash for cross-platform Path.parts behaviour
        norm = rel_path.replace("\\", "/")
        first_part = norm.split("/", 1)[0] if norm else ""
        if first_part in canonical_prefixes:
            flagged.append(fn)

    if flagged:
        print(f"  ADVISORY: {len(flagged)} function(s) exceed 200 lines in canonical paths")
        for fn in flagged[:10]:
            name = fn.get("name", fn.get("qualified_name", "?"))
            fpath = fn.get("relative_path", "?")
            lines = fn.get("line_count", "?")
            print(f"    {fpath}:{name} ({lines} lines)")
    return []  # advisory


def check_crg_bridge_node_test_coverage() -> list[str]:
    """D5: Top-10 production-code bridge nodes must each have a TESTED_BY edge.

    Bridge nodes are architectural chokepoints — if they break, the whole
    pipeline breaks. Fetches a wider sample (top_n=50) and filters to
    ``pipeline/`` and ``trading_app/`` files only (test files dominate raw
    betweenness rankings, but their lack of TESTED_BY edges is by design,
    not a finding).

    ADVISORY: emits warnings; never blocks when CRG is unavailable.
    """
    from pipeline.check_drift_crg_helpers import (
        CRG_UNAVAILABLE,
        crg_is_available,
        get_bridge_nodes,
        query_tests_for,
    )

    if not crg_is_available():
        print("  ADVISORY: CRG unavailable — bridge-node coverage check skipped")
        return []

    raw_bridges = get_bridge_nodes(top_n=50)
    if raw_bridges is CRG_UNAVAILABLE:
        print("  ADVISORY: CRG query failed — bridge-node coverage check skipped")
        return []

    # Filter to production-code bridges only.  ``file`` is absolute path on the
    # indexing host; we test for "pipeline/" / "trading_app/" substring after
    # forward-slash normalization, which works on both Windows and POSIX.
    canonical_markers = ("/pipeline/", "/trading_app/")
    canonical_bridges: list[dict] = []
    for node in raw_bridges:
        fpath = str(node.get("file", "")).replace("\\", "/")
        if any(marker in fpath for marker in canonical_markers):
            canonical_bridges.append(node)
        if len(canonical_bridges) >= 10:
            break

    untested: list[str] = []
    crg_errored: list[str] = []
    for node in canonical_bridges:
        node_name = node.get("qualified_name", node.get("name", ""))
        if not node_name:
            continue
        result = query_tests_for(node_name)
        if result is CRG_UNAVAILABLE:
            continue
        # Tagged status (post-audit 2026-04-29): distinguish "CRG returned
        # cleanly but found no tests" from "CRG response was malformed /
        # not_found / ambiguous". The first is a real D5 finding; the second
        # is a CRG-graph completeness issue and is reported separately so
        # readers don't conflate the two false-positive classes.
        if not isinstance(result, dict):
            continue
        status = result.get("status")
        fpath = str(node.get("file", "")).replace("\\", "/")
        root_prefix = str(PROJECT_ROOT).replace("\\", "/").rstrip("/") + "/"
        if fpath.startswith(root_prefix):
            fpath = fpath[len(root_prefix) :]
        betweenness = node.get("betweenness", "?")
        sym_only = node_name.split("::", 1)[-1]
        if status == "error":
            crg_errored.append(
                f"  CRG-uncertain: {sym_only} (betweenness={betweenness}, file={fpath}, raw={result.get('raw_status')!r})"
            )
        elif status == "empty":
            untested.append(f"  No TESTED_BY edge: {sym_only} (betweenness={betweenness}, file={fpath})")

    if untested:
        print(f"  ADVISORY: {len(untested)} bridge node(s) in top-10 lack TESTED_BY edges")
        for u in untested:
            print(u)
    if crg_errored:
        print(
            f"  ADVISORY: {len(crg_errored)} bridge node(s) had non-ok CRG response (graph-completeness issue, not a missing-test finding)"
        )
        for u in crg_errored:
            print(u)
    return []  # advisory


def check_referenced_paths_in_rules() -> list[str]:
    """Validate backtick-quoted path references in CLAUDE.md and .claude/rules/*.md.

    Delegates to scripts.tools.check_referenced_paths.main() — the canonical
    implementation. Treats the tool's stdout (one broken ref per line) as
    advisory diagnostics.

    ADVISORY: emits warnings but never blocks. Refs may transiently break
    during refactors or worktree-specific work; blocking would create
    cross-worktree friction.

    Why this check exists: 2026-04-29 audit found 3 real broken refs in
    canonical guardrail rules (integrity-guardian, research-truth-protocol,
    adversarial-audit-gate) — pointers to renamed/archived files that
    silently invalidate the rule's authority. Drift check surfaces these.
    """
    try:
        import contextlib
        import io

        from scripts.tools import check_referenced_paths
    except ImportError:
        print("  ADVISORY: check_referenced_paths unavailable — referenced-paths check skipped")
        return []

    # Capture the tool's stdout (one broken ref per line in non-verbose mode).
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            check_referenced_paths.main(verbose=False)
    except Exception as e:  # pragma: no cover — defensive
        print(f"  ADVISORY: referenced-paths check raised {type(e).__name__}: {e}")
        return []

    broken = [line.strip() for line in buf.getvalue().splitlines() if line.strip()]
    if not broken:
        return []

    print(f"  ADVISORY: {len(broken)} broken reference(s) in rule docs:")
    for ref in broken:
        print(f"    - {ref}")
    print("    Run: python scripts/tools/check_referenced_paths.py --verbose")
    return []  # advisory — surface but do not block


def check_lane_allocation_chordia_gate() -> list[str]:
    """Every lane in ``docs/runtime/lane_allocation.json`` must carry a passing
    Chordia verdict and a fresh audit.

    Rationale: 2026-05-01 chordia revalidation found 6 of 7 candidate lanes
    had never been audited against Criterion 4. The fix landed via the
    allocator gate (``trading_app.lane_allocator.apply_chordia_gate``) which
    refuses DEPLOY for FAIL_BOTH / FAIL_CHORDIA / MISSING and stale audits.
    This drift check is the second layer of defense: even if a developer
    hand-edits the JSON or rolls back the gate, the check fails the build.

    Fail conditions:
      - any lane's ``chordia_verdict`` is missing / None / not in
        (PASS_CHORDIA, PASS_PROTOCOL_A)
      - any lane's ``chordia_audit_age_days`` is missing / None / exceeds the
        doctrine freshness threshold (default 90, sourced live from
        ``trading_app.chordia.load_chordia_audit_log()``)

    Skip mode: if ``docs/runtime/lane_allocation.json`` does not exist, the
    check returns no violations — the file is generated by
    ``rebalance_lanes.py`` and may legitimately be absent in a fresh worktree
    or a CI checkout. The ``check_allocation_staleness`` mechanism in
    ``trading_app.lane_allocator`` handles missing-file enforcement at
    runtime.

    @canonical-source: trading_app/lane_allocator.apply_chordia_gate
    @canonical-source: trading_app/chordia.chordia_verdict_label
    @canonical-source: docs/runtime/chordia_audit_log.yaml
    """
    import json

    # Resolve via PROJECT_ROOT (module-top constant) — never CWD-relative.
    # A CWD-relative path would silently fail-open from any non-root cwd
    # (worktree subdir, CI step that cd's into pipeline/, editor "run check"
    # action). This drift check is the second-layer defense for a
    # capital-class gate; it must match the path-resolution discipline of
    # every other check in this file.
    allocation_path = PROJECT_ROOT / "docs" / "runtime" / "lane_allocation.json"
    if not allocation_path.exists():
        return []

    try:
        data = json.loads(allocation_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"  BAD lane_allocation.json: {exc}"]

    lanes = data.get("lanes", []) or []
    if not lanes:
        # An empty lanes[] is NOT a legitimate state — `rebalance_lanes.py`
        # always writes at least one lane in normal operation. Empty array
        # indicates either (a) producer crashed mid-write, (b) every
        # strategy got demoted by the gate (hand-edit) and the file was
        # left in place, or (c) test fixture leaked. Fail-closed: file
        # exists -> must contain real lanes. Operators wanting "no lanes"
        # should delete the file rather than empty its contents.
        return [
            "  lane_allocation.json has empty lanes[] — no active lanes to audit; "
            "if intentional, delete the file rather than leaving it empty"
        ]

    # Source freshness threshold from the doctrine YAML so they cannot drift.
    # Fail-closed: if the doctrine module cannot be loaded or returns an
    # unexpected shape, we cannot verify the audit threshold and must NOT
    # silently fall back to a hardcoded 90 — that hides doctrine corruption
    # behind a passing-looking check.
    try:
        from trading_app.chordia import load_chordia_audit_log

        freshness = load_chordia_audit_log().audit_freshness_days
    except Exception as exc:
        return [
            f"  Cannot load chordia freshness threshold from doctrine "
            f"(trading_app.chordia.load_chordia_audit_log raised {type(exc).__name__}: {exc}) — "
            f"audit threshold unverified; fix the import / YAML before this check can run"
        ]

    allow = ("PASS_CHORDIA", "PASS_PROTOCOL_A")
    violations: list[str] = []
    for lane in lanes:
        sid = lane.get("strategy_id", "<unknown>")
        verdict = lane.get("chordia_verdict")
        age = lane.get("chordia_audit_age_days")
        if verdict is None:
            violations.append(
                f"  {sid}: missing chordia_verdict in lane_allocation.json — "
                f"rerun scripts/tools/rebalance_lanes.py to repopulate"
            )
            continue
        if verdict not in allow:
            violations.append(
                f"  {sid}: chordia_verdict={verdict} is not in {{PASS_CHORDIA, PASS_PROTOCOL_A}} — "
                f"allocator gate must refuse DEPLOY"
            )
            continue
        if age is None:
            violations.append(
                f"  {sid}: missing chordia_audit_age_days — strategy not in "
                f"docs/runtime/chordia_audit_log.yaml; add an audit entry or refuse DEPLOY"
            )
            continue
        try:
            age_int = int(age)
        except (TypeError, ValueError):
            violations.append(f"  {sid}: chordia_audit_age_days={age!r} is not an integer")
            continue
        if age_int > freshness:
            violations.append(
                f"  {sid}: chordia audit is stale ({age_int}d > {freshness}d freshness) — "
                f"re-audit and update docs/runtime/chordia_audit_log.yaml"
            )

    return violations


def check_lane_allocation_c8_gate() -> list[str]:
    """Every lane in ``docs/runtime/lane_allocation.json`` must carry a
    Criterion-8 OOS-status verdict of ``PASSED`` (or NULL for Phase-4
    grandfather rows).

    Rationale: 2026-05-14 capital review of a proposed rebalance surfaced
    two MNQ lanes carrying ``c8_oos_status = FAILED_RATIO`` — one deployed
    (OVNRNG_25) and one proposed-add (ORB_VOL_8K). Investigation found the
    allocator never read ``validated_setups.c8_oos_status``: a fail-open
    architectural gap identical in shape to the pre-PR-#197 chordia
    situation. The fix landed via the allocator gate
    (``trading_app.lane_allocator.apply_c8_gate``) which refuses DEPLOY for
    every non-PASSED non-NULL label. This drift check is the second layer:
    even if a developer hand-edits the JSON or rolls back the gate, the
    check fails the build.

    Fail conditions:
      - any lane's ``c8_oos_status`` is in the C8-fail set (FAILED_RATIO,
        NEGATIVE_OOS_EXPR, NO_OOS_DATA, INSUFFICIENT_N_PATHWAY_B_REJECT,
        INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH, REJECTED), or empty string

    Pass conditions:
      - ``c8_oos_status == "PASSED"`` (validator confirmed the OOS ratio
        meets Criterion 8)
      - ``c8_oos_status is None`` / key absent (Phase-4 grandfather row
        written before c8 became a validator write target; matches
        validator's NULL emission on SKIPPED aliases at
        strategy_validator.py:1778)

    Skip mode: if ``docs/runtime/lane_allocation.json`` does not exist,
    the check returns no violations — the file is generated by
    ``rebalance_lanes.py`` and may legitimately be absent in a fresh
    worktree or a CI checkout. The ``check_allocation_staleness``
    mechanism in ``trading_app.lane_allocator`` handles missing-file
    enforcement at runtime.

    @canonical-source: trading_app/lane_allocator.apply_c8_gate
    @canonical-source: trading_app/strategy_validator.py (c8_oos_status writer, lines 1061-1138)
    @canonical-source: docs/institutional/pre_registered_criteria.md (Criterion 8 + Amendments 3.0/3.1)
    """
    import json

    # Resolve via PROJECT_ROOT (module-top constant) — never CWD-relative.
    # Matches check_lane_allocation_chordia_gate path-resolution discipline.
    allocation_path = PROJECT_ROOT / "docs" / "runtime" / "lane_allocation.json"
    if not allocation_path.exists():
        return []

    try:
        data = json.loads(allocation_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"  BAD lane_allocation.json: {exc}"]

    lanes = data.get("lanes", []) or []
    if not lanes:
        # An empty lanes[] is NOT a legitimate state — handled by the
        # chordia drift check's empty-lanes guard. We don't double-report.
        return []

    # Labels that mean "not deployable" per Criterion 8 + Amendment 3.1.
    # Mirrors apply_c8_gate._C8_FAIL_LABELS — keep in sync if extended.
    c8_fail_labels = {
        "FAILED_RATIO",
        "NEGATIVE_OOS_EXPR",
        "NO_OOS_DATA",
        "INSUFFICIENT_N_PATHWAY_B_REJECT",
        "INSUFFICIENT_N_PATHWAY_A_PASS_THROUGH",
        "REJECTED",
    }

    violations: list[str] = []
    for lane in lanes:
        sid = lane.get("strategy_id", "<unknown>")
        # Key absent OR explicit None = Phase-4 grandfather, pass through.
        if "c8_oos_status" not in lane:
            continue
        c8 = lane.get("c8_oos_status")
        if c8 is None:
            continue
        if c8 == "PASSED":
            continue
        if c8 == "" or c8 in c8_fail_labels:
            violations.append(
                f"  {sid}: c8_oos_status={c8!r} (Criterion 8 OOS deployment gate) — "
                f"allocator gate must refuse DEPLOY; rerun scripts/tools/rebalance_lanes.py"
            )
            continue
        # Defensive: any future label not in the allowlist.
        violations.append(
            f"  {sid}: c8_oos_status={c8!r} is not PASSED / NULL / known-fail; "
            f"unknown label — extend allowlist or refuse DEPLOY"
        )

    return violations


def check_lane_allocation_displaced_bucket() -> list[str]:
    """`displaced` array in lane_allocation.json must have valid entries.

    Rationale: 2026-05-17 audit of MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30
    found the allocator silently drops candidates that cleared all hard gates
    (chordia, c8, live-tradeability) but lost a soft gate (correlation,
    dd_budget, hysteresis). The JSON had no `displaced[]` bucket so a future
    auditor could not distinguish "never had a chordia audit" from "passed
    chordia, beaten by correlation". This check enforces the new bucket's
    structural contract once the rebalancer has written it.

    Grandfather: if `displaced` key is absent, return no violations — the
    canonical JSON may pre-date this field. Once the first post-fix
    rebalance commits, the key is present and the check enforces.

    Fail conditions (only when `displaced` key present):
      - any entry missing `strategy_id` or with non-string strategy_id
      - any entry with `rejection_gate` not in the locked enum
      - any entry with `rejection_gate == "missing_cost_spec"` (LOUD FAIL —
        config drift trip-wire; this should never fire in production)

    Pass conditions:
      - `displaced` key absent (grandfather)
      - `displaced` is an empty list (no soft-gate rejections this rebalance)
      - every entry has valid strategy_id + rejection_gate in the enum and
        no entry trips the missing_cost_spec trip-wire

    @canonical-source: trading_app/lane_allocator.build_allocation (displaced_out)
    @canonical-source: docs/runtime/stages/lane-allocator-displaced-bucket.md
    """
    import json

    ALLOWED_GATES = {"correlation", "dd_budget", "hysteresis", "missing_cost_spec"}

    allocation_path = PROJECT_ROOT / "docs" / "runtime" / "lane_allocation.json"
    if not allocation_path.exists():
        return []

    try:
        data = json.loads(allocation_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"  BAD lane_allocation.json: {exc}"]

    if "displaced" not in data:
        return []

    displaced = data.get("displaced") or []
    if not isinstance(displaced, list):
        return [f"  lane_allocation.json: 'displaced' must be a list, got {type(displaced).__name__}"]

    violations: list[str] = []
    for i, entry in enumerate(displaced):
        if not isinstance(entry, dict):
            violations.append(f"  displaced[{i}]: not a dict")
            continue
        sid = entry.get("strategy_id")
        if not isinstance(sid, str) or not sid:
            violations.append(f"  displaced[{i}]: missing or invalid strategy_id")
            continue
        gate = entry.get("rejection_gate")
        if gate not in ALLOWED_GATES:
            violations.append(
                f"  {sid}: rejection_gate={gate!r} not in {sorted(ALLOWED_GATES)}; extend allowlist or fix writer"
            )
            continue
        if gate == "missing_cost_spec":
            violations.append(
                f"  {sid}: rejection_gate='missing_cost_spec' fired — "
                f"pipeline.cost_model.COST_SPECS is missing instrument "
                f"{entry.get('instrument')!r}; config drift, fix immediately"
            )

    return violations


def check_strategy_lab_no_fitness_endpoint() -> list[str]:
    """`scripts/tools/strategy_lab_mcp_server.py` must NOT register a
    `get_recent_fitness` MCP endpoint.

    Origin: 2026-05-13 Nugget 3 closure. `get_recent_fitness` overlapped
    `gold-db.get_strategy_fitness` and created two surfaces for the same
    fact, which the project's MCP doctrine
    (`.claude/rules/mcp-usage.md`) explicitly designates as canonical on
    gold-db. The MCP-staleness class bug
    (`feedback_strategy_lab_mcp_vs_lane_allocation_json_divergence.md`,
    2026-05-13) is the proof case for why two parallel fitness surfaces
    cannot exist.

    The check AST-walks ToolSpec(...) call sites; doc/comment mentions
    of the removed name are intentionally NOT flagged (those are
    historical references, not registration). `_compute_fitness_payload`
    is NOT removed — it is still consumed by `_get_strategy_readiness`
    and `_list_promotable_candidates`.
    """
    target = PROJECT_ROOT / "scripts" / "tools" / "strategy_lab_mcp_server.py"
    if not target.exists():
        return []
    try:
        tree = ast.parse(target.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"  Could not parse {target}: {exc}"]

    offending: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_toolspec = (isinstance(func, ast.Name) and func.id == "ToolSpec") or (
            isinstance(func, ast.Attribute) and func.attr == "ToolSpec"
        )
        if not is_toolspec or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == "get_recent_fitness":
            offending.append(f"  Line {node.lineno}: ToolSpec re-registered 'get_recent_fitness'")

    if offending:
        return [
            "  scripts/tools/strategy_lab_mcp_server.py re-registered the "
            "deprecated 'get_recent_fitness' MCP endpoint. Use "
            "gold-db.get_strategy_fitness instead (see .claude/rules/mcp-usage.md).",
            *offending,
        ]
    return []


def check_literature_extracts_mode_a_b_framing() -> list[str]:
    """Literature extracts that cite internal `research/output/` artifacts
    must carry explicit Mode A / Mode B / holdout-window framing.

    Origin: 2026-05-07 self-review (commit 2ea6fc5e) caught two storytelling
    issues in `yordanov_2026_nq_orb_value_area_breakouts.md` initial draft:
    (1) composite-N conflation (cited aggregate N=90 alongside test-only
    +0.46R, which actually came from N=42 OOS-on); (2) the cited "test
    2025+" block aggregates 2025 (Mode A IS under
    `trading_app.holdout_policy.HOLDOUT_SACRED_FROM=2026-01-01`) with 2026
    (Mode A sacred OOS), which per `research-truth-protocol.md` § "Mode B
    grandfathered baselines" CANNOT be cited as Mode A OOS evidence.

    Both classes of error fire when an extract references internal
    pre-Phase-0 research output without naming the holdout regime under
    which the cited results were produced. This check enforces that any
    new/edited extract citing `research/output/` mentions Mode A, Mode B,
    `HOLDOUT_SACRED_FROM`, or "grandfathered" at least once.

    Forward-looking: at adoption (2026-05-07) only 2 existing extracts
    cite `research/output/` and both already comply — zero violations at
    first run. PENDING_ACQUISITION_*.md files are exempt by filename
    prefix because they are sourcing-status reports, not extracts.
    """
    lit_dir = PROJECT_ROOT / "docs" / "institutional" / "literature"
    if not lit_dir.exists():
        return []
    framing_pattern = re.compile(r"Mode\s*A|Mode\s*B|HOLDOUT_SACRED_FROM|grandfathered", re.IGNORECASE)
    citation_pattern = re.compile(r"research/output/")
    violations: list[str] = []
    for md_file in sorted(lit_dir.glob("*.md")):
        if md_file.name.startswith("PENDING_ACQUISITION_"):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not citation_pattern.search(content):
            continue  # extract does not cite internal output — exempt
        if not framing_pattern.search(content):
            try:
                rel = md_file.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                rel = md_file.as_posix()
            violations.append(
                f"{rel}: cites research/output/ but lacks Mode A / Mode B / "
                f"HOLDOUT_SACRED_FROM / grandfathered framing. Add explicit "
                f"holdout-regime framing per .claude/rules/research-truth-protocol.md "
                f"section 'Mode B grandfathered baselines'. (Class bug: "
                f"composite-N + Mode A/B conflation, 2026-05-07 self-review "
                f"commit 2ea6fc5e)"
            )
    return violations


def check_fast_lane_template_doctrine_fields() -> list[str]:
    """Verify FAST_LANE templates carry doctrine self-labels honestly.

    v5 (superseded) must carry `deprecated: true` so a future operator copying
    from it gets a loud signal. v5.1 (live) must carry `is_triage_screen: true`,
    `validation_status_explicit` starting with `NOT_VALIDATED`, and
    `promotion_target: heavyweight_chordia_prereg`. These are the doctrine
    claims that downstream consumers rely on to refuse capital paths.

    Doctrine source: TEMPLATE-fast-lane-v5.1.yaml `validation_status_explicit`
    field; commit 277b643b (2026-05-18) supersession of v5.

    Fail-closed: missing field or unexpected value is a drift violation.
    Templates that no longer exist are skipped — the check has nothing to
    enforce.

    Added 2026-05-18 per FAST_LANE v6 backlog item L1 (code-review follow-up).
    """
    hypotheses = PROJECT_ROOT / "docs" / "audit" / "hypotheses"
    v5_path = hypotheses / "TEMPLATE-fast-lane-v5.yaml"
    v51_path = hypotheses / "TEMPLATE-fast-lane-v5.1.yaml"

    violations: list[str] = []

    if v5_path.exists():
        try:
            v5_text = v5_path.read_text(encoding="utf-8")
        except OSError as e:
            return [f"check_fast_lane_template_doctrine_fields: read error v5: {e}"]
        # v5 must self-label as deprecated. We grep for the exact YAML line
        # rather than parsing the file because templates use `<MNQ|MES|MGC>`
        # placeholder syntax that pyyaml accepts but instantiated linters reject.
        # Tolerate trailing comments (YAML allows `key: value  # comment`).
        if not re.search(r"^\s*deprecated:\s*true\s*(?:#.*)?$", v5_text, re.MULTILINE):
            violations.append(
                "TEMPLATE-fast-lane-v5.yaml missing `deprecated: true` in metadata. "
                "v5 was superseded by v5.1 (commit 277b643b, 2026-05-18). Add "
                "`deprecated: true` so a future operator copying from v5 gets a "
                "loud signal. See docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml."
            )

    if v51_path.exists():
        try:
            v51_text = v51_path.read_text(encoding="utf-8")
        except OSError as e:
            return [f"check_fast_lane_template_doctrine_fields: read error v5.1: {e}"]

        if not re.search(r"^\s*is_triage_screen:\s*true\s*(?:#.*)?$", v51_text, re.MULTILINE):
            violations.append(
                "TEMPLATE-fast-lane-v5.1.yaml missing `is_triage_screen: true` in "
                "metadata. This flag is the doctrine claim downstream consumers "
                "rely on to refuse capital paths."
            )

        if not re.search(
            r"^\s*validation_status_explicit:\s*\"?NOT_VALIDATED[^\"\n#]*\"?\s*(?:#.*)?$",
            v51_text,
            re.MULTILINE,
        ):
            violations.append(
                "TEMPLATE-fast-lane-v5.1.yaml `validation_status_explicit` field "
                "missing or does not start with `NOT_VALIDATED`. FAST_LANE is a "
                "triage screen, not validated; the explicit self-label is the "
                "strongest anti-narrative guard against future drift toward "
                "treating PROMOTE as a deploy verdict."
            )

        if not re.search(
            r"^\s*promotion_target:\s*\"?heavyweight_chordia_prereg\"?\s*(?:#.*)?$",
            v51_text,
            re.MULTILINE,
        ):
            violations.append(
                "TEMPLATE-fast-lane-v5.1.yaml `promotion_target` field must equal "
                "`heavyweight_chordia_prereg`. PROMOTE outcome routes to heavyweight "
                "pre-reg, never to capital — the field encodes this contract."
            )

    return violations


def check_fast_lane_runner_template_routing() -> list[str]:
    """Verify FAST_LANE v5.1 preregs route through the runner's v5.1 branch.

    Any prereg under ``docs/audit/hypotheses/`` whose ``metadata.template_version``
    is ``fast_lane_v5.1`` MUST have a corresponding result MD that carries BOTH
    the automated FAST_LANE block AND the heavyweight Chordia verdict line.

    This catches two failure classes:
    1. Operator runs a pre-v5.1-aware copy of ``chordia_strict_unlock_v1.py``
       against a v5.1 prereg — heavyweight verdict appears but FAST_LANE block
       does not (silent bypass of the v5.1 gate table).
    2. Operator hand-edits a result MD and accidentally deletes the FAST_LANE
       block, leaving only the heavyweight verdict that does not represent
       the prereg's actual screening verdict.

    Sentinel-date gate: only applies to prereg files dated >= 2026-05-20.
    This grandfathers the existing manual v5.1 instance
    ``2026-05-18-mnq-usdata1000-e1-rr10-pd-clear-long-o30-fast-lane-v1`` whose
    FAST_LANE verdict block was operator-written before the runner branch
    landed. Matches the date-sentinel pattern used by
    ``check_hypothesis_minbtl_compliance`` and
    ``check_chordia_result_threshold_matches_prereg``.

    Sentinel string ``## FAST_LANE v5.1 verdict (automated)`` and the line
    ``**FAST_LANE verdict:** `<VERDICT>``` are produced by
    ``research/chordia_strict_unlock_v1.py::_fast_lane_block_lines``. Renaming
    the heading there requires updating this check (paired by design).

    Added 2026-05-18 alongside the FAST_LANE runner-branch automation
    (stage ``fast-lane-runner-automation``).
    """
    import datetime as _dt

    hypotheses_dir = PROJECT_ROOT / "docs" / "audit" / "hypotheses"
    results_dir = PROJECT_ROOT / "docs" / "audit" / "results"
    if not hypotheses_dir.is_dir() or not results_dir.is_dir():
        return []

    sentinel = _dt.date(2026, 5, 20)
    fname_date = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-")
    violations: list[str] = []

    for prereg_path in sorted(hypotheses_dir.glob("*.yaml")):
        stem = prereg_path.stem
        if stem.startswith("TEMPLATE-"):
            continue
        m = fname_date.match(stem)
        if m is None:
            continue
        try:
            file_date = _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
        if file_date < sentinel:
            continue
        try:
            text = prereg_path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Cheap line-level scan — avoids loading PyYAML for a ~100 KB sweep.
        if not re.search(
            r"^\s*template_version:\s*[\"']?fast_lane_v5\.1[\"']?\s*(?:#.*)?$",
            text,
            re.MULTILINE,
        ):
            continue
        result_md = results_dir / f"{stem}.md"
        if not result_md.exists():
            # Prereg landed but runner has not been invoked yet — operator task,
            # not a runner-bypass. Skip rather than block.
            continue
        try:
            md_text = result_md.read_text(encoding="utf-8")
        except OSError as e:
            violations.append(
                f"check_fast_lane_runner_template_routing: cannot read {result_md.relative_to(PROJECT_ROOT)}: {e}"
            )
            continue
        if "## FAST_LANE v5.1 verdict (automated)" not in md_text:
            violations.append(
                f"{result_md.relative_to(PROJECT_ROOT).as_posix()}: prereg declares "
                f"`template_version: fast_lane_v5.1` but result MD missing automated "
                f"FAST_LANE block (`## FAST_LANE v5.1 verdict (automated)`). "
                f"Re-run `research/chordia_strict_unlock_v1.py --hypothesis-file "
                f"{prereg_path.relative_to(PROJECT_ROOT).as_posix()}` to regenerate."
            )
            continue
        if not re.search(
            r"^\*\*FAST_LANE verdict:\*\*\s+`(PROMOTE|NEEDS-MORE|KILL)`",
            md_text,
            re.MULTILINE,
        ):
            violations.append(
                f"{result_md.relative_to(PROJECT_ROOT).as_posix()}: FAST_LANE block "
                f"present but verdict line missing or value not in "
                f"{{PROMOTE, NEEDS-MORE, KILL}}. Operator may have hand-edited; "
                f"re-run the runner to regenerate."
            )
            continue
        if "**MEASURED verdict:**" not in md_text:
            violations.append(
                f"{result_md.relative_to(PROJECT_ROOT).as_posix()}: heavyweight "
                f"Chordia verdict line missing. Heavyweight and FAST_LANE must "
                f"coexist; neither overrides the other per "
                f"TEMPLATE-fast-lane-v5.1.yaml line 8-9."
            )
    return violations


def check_intent_router_routing_parity() -> list[str]:
    """Verify .claude/rules/auto-skill-routing.md ↔ .claude/hooks/intent-router.py parity.

    The hook compiles a hardcoded INTENT_RULES regex table, while the rule file
    documents the same routing surface as human-readable bullets. If they drift,
    routing cues point to skills the rule says aren't routed (or vice versa).

    Compares the set of slash-command targets on both sides. Fail-closed:
    any divergence is a drift violation.

    Added 2026-05-17 per operating-layer audit plan §G.1 (mutation probe).
    """
    rule_file = PROJECT_ROOT / ".claude" / "rules" / "auto-skill-routing.md"
    hook_file = PROJECT_ROOT / ".claude" / "hooks" / "intent-router.py"
    if not rule_file.exists() or not hook_file.exists():
        return []  # if either is missing, the hook itself fails-open — no drift to detect

    try:
        rule_text = rule_file.read_text(encoding="utf-8")
        hook_text = hook_file.read_text(encoding="utf-8")
    except OSError as e:
        return [f"check_intent_router_routing_parity: read error: {e}"]

    # Collect skill targets from the rule file. Matches both arrow styles:
    #   `-> \`/skill-name\``   (Intent Map / CRG bullets)
    #   `→ \`/skill-name\``    (unicode arrow if introduced)
    rule_targets: set[str] = set()
    for match in re.finditer(r"(?:->|→)\s*`(/[a-z][a-z0-9 _-]*?)`", rule_text):
        rule_targets.add(match.group(1).strip())

    # Skill targets from the hook's INTENT_RULES list.
    hook_targets: set[str] = set()
    try:
        start = hook_text.index("INTENT_RULES: list[")
        end = hook_text.index("\n]", start)
        rules_block = hook_text[start:end]
    except ValueError:
        return ["check_intent_router_routing_parity: could not locate INTENT_RULES block in intent-router.py"]
    for match in re.finditer(r',\s*"(/[a-z][a-z0-9 _-]*?)"\s*,', rules_block):
        hook_targets.add(match.group(1).strip())

    violations: list[str] = []

    # Direction 1: every hook target must appear in the rule file (no orphan routes).
    rule_missing = sorted(hook_targets - rule_targets)
    if rule_missing:
        violations.append(
            f"intent-router.py routes to skills not documented in auto-skill-routing.md: "
            f"{rule_missing}. Either add a bullet to the rule file's Intent Map / CRG "
            f"section, or remove the hook entry. Class-bug protection: hook + rule must "
            f"agree on the routing surface (plan 2026-05-17 §G.1)."
        )

    # Direction 2: every rule target SHOULD appear in the hook (otherwise the cue
    # never fires). Some rule entries are intentionally documentary and never
    # bind to a single literal trigger.
    DOCUMENTED_RULE_ONLY = {
        "/code-review",  # rule documents the route; hook regex line not unique
        "/crg-context",  # rule § CRG line — meta-routing, not a single literal trigger
    }
    hook_missing = sorted(rule_targets - hook_targets - DOCUMENTED_RULE_ONLY)
    if hook_missing:
        violations.append(
            f"auto-skill-routing.md documents skill routes not present in intent-router.py "
            f"INTENT_RULES: {hook_missing}. Either add a regex pattern to the hook, or "
            f"add the skill to DOCUMENTED_RULE_ONLY allow-list in this check with a "
            f"justification. Class-bug protection: silent rule additions never fire."
        )

    return violations


# Each entry: (description, callable, is_advisory).
# is_advisory=True → prints warnings but never blocks (shown as ADVISORY).
# Check number is derived from position (1-indexed).

# Tuple format: (description, callable, is_advisory, requires_db)
# requires_db=True means check can return "SKIPPED" if DB unavailable
def check_fast_lane_promote_orphans() -> list[str]:
    """Reconstruct the FAST_LANE PROMOTE queue and diff against the cache.

    Three failure classes caught:

    1. ERROR entry — a PROMOTE result MD whose per-direction sanity gate
       flags pooling-artifact (both directions KILL_AS_STANDALONE) without
       a revocation sidecar. Pooling-inflation PROMOTEs cannot be left in
       the queue — the deployment claim is structurally invalid.
    2. Cache drift — ``docs/runtime/promote_queue.yaml`` exists but its
       entries disagree with the reconstruction. Either the cache is
       stale or someone hand-edited it (e.g. to flip REVOKED back to
       QUEUED). Either way, operator must rerun ``--write`` to refresh.
    3. Orphan PROMOTE — a result MD with FAST_LANE verdict PROMOTE that
       would not appear in the queue at all (e.g. malformed title line).
       The scanner returns it as ERROR; this check forwards that.

    Added 2026-05-18 alongside ``scripts/research/fast_lane_promote_queue.py``
    and the stage ``2026-05-18-fast-lane-promote-queue-scanner.md``.
    """
    try:
        from scripts.research.fast_lane_promote_queue import (
            QUEUE_CACHE,
            diff_against_cache,
            scan,
        )
    except Exception as exc:
        return [f"check_fast_lane_promote_orphans: import error: {exc}"]

    try:
        entries = scan()
    except Exception as exc:
        return [f"check_fast_lane_promote_orphans: scan failure: {exc}"]

    violations: list[str] = []

    for e in entries:
        if e.status == "ERROR":
            violations.append(
                f"FAST_LANE PROMOTE in ERROR state: {e.strategy_id} -> {e.error_reason} "
                f"(file: {e.result_md})"
            )

    if QUEUE_CACHE.exists():
        diff_lines = diff_against_cache(entries, QUEUE_CACHE)
        for line in diff_lines:
            stripped = line.strip()
            if stripped in {"(cache up to date)", "(no cache file on disk; first run)"}:
                continue
            violations.append(
                f"FAST_LANE PROMOTE queue cache out of sync: {stripped} "
                f"(rerun `python scripts/research/fast_lane_promote_queue.py --write`)"
            )

    return violations


def check_fast_lane_promote_threshold_parity(template_path: Path | None = None) -> list[str]:
    """Parity check: fast_lane_promote_queue scanner constants vs TEMPLATE-fast-lane-v5.1.yaml.

    The PROMOTE-queue scanner at ``scripts/research/fast_lane_promote_queue.py``
    inlines six gating constants (T_KILL_FLOOR, T_PROMOTE_FLOOR, EXPR_FLOOR,
    N_FLOOR, FIRE_MIN, FIRE_MAX) with a prose-comment cite to the canonical
    FAST_LANE v5.1 template. Comment-cited inlines silently drift when the
    template is amended (4th confirmed instance of the
    [[canonical-inline-copy-parity-bug-class]] — see
    ``memory/feedback_canonical_inline_copy_parity_bug_class.md``).

    This check asserts each scanner constant equals its template-derived value:

      - T_KILL_FLOOR     == screen.promote_threshold
      - T_PROMOTE_FLOOR  == screen.promote_threshold + screen.needs_more_band
      - EXPR_FLOOR       == screen.expr_min
      - N_FLOOR          == screen.n_IS_on_min
      - FIRE_MIN, FIRE_MAX parsed from screen.fire_rate_gate.kill_if string

    Authority: ``docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml`` lines
    102-145 are the single source of truth for FAST_LANE v5.1 promotion
    thresholds. Scanner inlines must mirror.

    Parameters
    ----------
    template_path : Path | None
        Override path to the v5.1 template (test seam for the missing-template
        injection test). Defaults to the canonical project location.

    Returns
    -------
    list[str]
        Violation strings — empty when all six constants match canonical.
        Fail-closed when the template cannot be read or parsed.
    """
    import re

    try:
        import yaml  # local import — yaml is already a project dep but keep optional surface tight
    except Exception as exc:  # pragma: no cover — yaml is a hard dep
        return [f"check_fast_lane_promote_threshold_parity: PyYAML import failed: {exc}"]

    try:
        from scripts.research import fast_lane_promote_queue as flpq
    except Exception as exc:
        return [f"check_fast_lane_promote_threshold_parity: scanner import failed: {exc}"]

    target = (
        template_path
        if template_path is not None
        else PROJECT_ROOT / "docs" / "audit" / "hypotheses" / "TEMPLATE-fast-lane-v5.1.yaml"
    )

    if not target.exists():
        return [
            f"check_fast_lane_promote_threshold_parity: canonical template missing at {target} "
            "(scanner threshold parity cannot be verified — fail-closed)"
        ]

    try:
        spec = yaml.safe_load(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            f"check_fast_lane_promote_threshold_parity: failed to parse {target.name}: "
            f"{type(exc).__name__}: {exc}"
        ]

    if not isinstance(spec, dict) or "screen" not in spec:
        return [
            f"check_fast_lane_promote_threshold_parity: {target.name} missing top-level `screen:` "
            "block (template structure has drifted — investigate before promoting the check)"
        ]

    screen = spec["screen"]
    violations: list[str] = []

    def _require(key: str) -> float | int | None:
        if key not in screen:
            violations.append(
                f"check_fast_lane_promote_threshold_parity: canonical template missing "
                f"`screen.{key}` (FAST_LANE v5.1 contract broken)"
            )
            return None
        value = screen[key]
        # Distinguish "key absent" (above) from "key present but null/non-numeric"
        # — YAML `key:` with no value parses as None, and `key: [0.5]` parses as
        # a list. Both would crash `float(value)` downstream — fail-closed with
        # a structural violation instead.
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            violations.append(
                f"check_fast_lane_promote_threshold_parity: canonical template "
                f"`screen.{key}` is {type(value).__name__} ({value!r}); expected numeric "
                "(FAST_LANE v5.1 contract broken)"
            )
            return None
        return value

    promote_threshold = _require("promote_threshold")
    needs_more_band = _require("needs_more_band")
    expr_min = _require("expr_min")
    n_is_on_min = _require("n_IS_on_min")

    fire_min_canonical: float | None = None
    fire_max_canonical: float | None = None
    gate = screen.get("fire_rate_gate")
    if not isinstance(gate, dict) or "kill_if" not in gate:
        violations.append(
            "check_fast_lane_promote_threshold_parity: canonical template missing "
            "`screen.fire_rate_gate.kill_if`"
        )
    else:
        kill_if = str(gate["kill_if"])
        m = re.search(r"<\s*(\d+\.\d+).*?>\s*(\d+\.\d+)", kill_if)
        if m is None:
            violations.append(
                f"check_fast_lane_promote_threshold_parity: could not parse fire-rate bounds "
                f"from kill_if={kill_if!r} (expected `fire_rate < X OR fire_rate > Y`)"
            )
        else:
            fire_min_canonical = float(m.group(1))
            fire_max_canonical = float(m.group(2))

    # If structural parse already failed, stop — comparing partial values
    # would emit misleading "constant != None" violations.
    if violations:
        return violations

    # Type guards — _require / fire-rate parse have already populated these
    # to non-None when violations is empty, but assert to make Pyright happy
    # and to fail loudly if a future refactor breaks the invariant.
    assert promote_threshold is not None
    assert needs_more_band is not None
    assert expr_min is not None
    assert n_is_on_min is not None
    assert fire_min_canonical is not None
    assert fire_max_canonical is not None

    expected_t_kill = float(promote_threshold)
    expected_t_promote = float(promote_threshold) + float(needs_more_band)
    expected_expr = float(expr_min)
    expected_n = int(n_is_on_min)
    expected_fire_min = float(fire_min_canonical)
    expected_fire_max = float(fire_max_canonical)

    # Tolerance: scanner values are typed as float; equality with .yaml-decoded
    # floats is exact when the template uses literal `2.5`/`0.5`/etc.
    TOL = 1e-9

    def _flag(name: str, actual: float | int, expected: float | int, canonical_source: str) -> None:
        violations.append(
            f"check_fast_lane_promote_threshold_parity: scanner {name}={actual!r} "
            f"does not match canonical {expected!r} (source: {canonical_source}). "
            f"Inline-copy drift — see [[canonical-inline-copy-parity-bug-class]]; "
            f"update scripts/research/fast_lane_promote_queue.py or amend the template."
        )

    if abs(float(flpq.T_KILL_FLOOR) - expected_t_kill) > TOL:
        _flag("T_KILL_FLOOR", flpq.T_KILL_FLOOR, expected_t_kill, "screen.promote_threshold")
    if abs(float(flpq.T_PROMOTE_FLOOR) - expected_t_promote) > TOL:
        _flag(
            "T_PROMOTE_FLOOR",
            flpq.T_PROMOTE_FLOOR,
            expected_t_promote,
            "screen.promote_threshold + screen.needs_more_band",
        )
    if abs(float(flpq.EXPR_FLOOR) - expected_expr) > TOL:
        _flag("EXPR_FLOOR", flpq.EXPR_FLOOR, expected_expr, "screen.expr_min")
    if int(flpq.N_FLOOR) != expected_n:
        _flag("N_FLOOR", flpq.N_FLOOR, expected_n, "screen.n_IS_on_min")
    if abs(float(flpq.FIRE_MIN) - expected_fire_min) > TOL:
        _flag(
            "FIRE_MIN",
            flpq.FIRE_MIN,
            expected_fire_min,
            "screen.fire_rate_gate.kill_if (lower bound)",
        )
    if abs(float(flpq.FIRE_MAX) - expected_fire_max) > TOL:
        _flag(
            "FIRE_MAX",
            flpq.FIRE_MAX,
            expected_fire_max,
            "screen.fire_rate_gate.kill_if (upper bound)",
        )

    return violations


def check_cherry_pick_ranker_threshold_parity(
    criteria_path: Path | None = None,
) -> list[str]:
    """Parity check: cherry_pick_ranker.HEAVYWEIGHT_T_THRESHOLD vs pre_registered_criteria.md Criterion 4.

    The cherry-pick ranker at ``scripts/research/cherry_pick_ranker.py``
    inlines the Chordia strict no-theory t-threshold as
    ``HEAVYWEIGHT_T_THRESHOLD`` with a prose-comment cite to
    ``pre_registered_criteria.md`` Criterion 4. Comment-cited inlines
    silently drift when the doctrine is amended (5th confirmed instance of
    the [[canonical-inline-copy-parity-bug-class]] -- see
    ``memory/feedback_canonical_inline_copy_parity_bug_class.md``).

    This check parses the Criterion 4 section of the doctrine and asserts
    the ranker's threshold equals the no-theory value cited verbatim from
    ``literature/chordia_et_al_2018_two_million_strategies.md:20``.

    Authority: ``docs/institutional/pre_registered_criteria.md`` Criterion 4
    is the single source of truth for the heavyweight Chordia threshold.

    Parameters
    ----------
    criteria_path : Path | None
        Override path to the doctrine doc (test seam for missing-doctrine
        and parse-failure injection tests). Defaults to canonical location.

    Returns
    -------
    list[str]
        Violation strings -- empty when ranker constant matches canonical.
        Fail-closed when the doctrine cannot be read or parsed.
    """
    try:
        from scripts.research import cherry_pick_ranker as cpr
    except Exception as exc:
        return [
            f"check_cherry_pick_ranker_threshold_parity: ranker import failed: {exc}"
        ]

    target = (
        criteria_path
        if criteria_path is not None
        else PROJECT_ROOT
        / "docs"
        / "institutional"
        / "pre_registered_criteria.md"
    )

    if not target.exists():
        return [
            f"check_cherry_pick_ranker_threshold_parity: canonical doctrine missing at "
            f"{target} (ranker threshold parity cannot be verified -- fail-closed)"
        ]

    try:
        text = target.read_text(encoding="utf-8")
    except Exception as exc:
        return [
            f"check_cherry_pick_ranker_threshold_parity: failed to read {target.name}: "
            f"{type(exc).__name__}: {exc}"
        ]

    # Locate the Criterion 4 section. Use the canonical heading exactly as
    # written in the doctrine; supersession-banner amendments do not move it
    # (per feedback_doctrine_supersession_banner_pattern.md).
    crit4_marker = "## Criterion 4 — Chordia t-statistic threshold"
    idx = text.find(crit4_marker)
    if idx < 0:
        return [
            "check_cherry_pick_ranker_threshold_parity: Criterion 4 section not found "
            f"in {target.name} (heading drifted or doctrine restructured -- "
            "investigate before promoting the check)"
        ]
    # Bound the search to the section body (until the next H2 or EOF).
    next_h2 = text.find("\n## ", idx + len(crit4_marker))
    section = text[idx : next_h2 if next_h2 > 0 else len(text)]

    # Match the no-theory threshold. The doctrine wording (line 116 at landing
    # time) is verbatim: "Require t >= 3.79 (Chordia et al 2018, ...) for
    # strategies without such theoretical support." We anchor on the phrase
    # "t >= <FLOAT>" close to "without".
    no_theory_match = re.search(
        r"t\s*[>=≥]+\s*(\d+\.\d+)[^.]*?(?:without|no.?theory)",
        section,
        re.IGNORECASE | re.DOTALL,
    )
    if no_theory_match is None:
        return [
            "check_cherry_pick_ranker_threshold_parity: could not parse no-theory "
            f"threshold from Criterion 4 in {target.name} (doctrine wording drifted)"
        ]

    try:
        canonical = float(no_theory_match.group(1))
    except ValueError:
        return [
            "check_cherry_pick_ranker_threshold_parity: parsed no-theory threshold "
            f"{no_theory_match.group(1)!r} is not a valid float"
        ]

    actual = float(cpr.HEAVYWEIGHT_T_THRESHOLD)
    TOL = 1e-9
    if abs(actual - canonical) > TOL:
        return [
            f"check_cherry_pick_ranker_threshold_parity: ranker "
            f"HEAVYWEIGHT_T_THRESHOLD={actual!r} does not match canonical "
            f"{canonical!r} from pre_registered_criteria.md Criterion 4 "
            "(no-theory threshold). Inline-copy drift -- see "
            "[[canonical-inline-copy-parity-bug-class]]; update "
            "scripts/research/cherry_pick_ranker.py or amend the doctrine."
        ]

    return []


def check_fast_lane_oos_power_gate_constants_grounded() -> list[str]:
    """Sanity gate on FAST_LANE pre-flight OOS-power gate constants.

    Asserts that ``scripts/research/fast_lane_promote_queue.py`` continues to
    expose:

      - ``OOS_POWER_FLOOR`` as a finite float in (0.0, 1.0]
      - ``OOS_COHEN_D_TARGET`` as a finite float > 0
      - ``REJECTED_OOS_UNPOWERED`` listed in ``STATUS_VALUES``

    These three together define the gate's behavior. Drift in any of them —
    deleting the floor, flipping the Cohen's d sign, or dropping the status
    enum value — would silently re-promote structurally-unbuildable cells
    that the gate exists to catch. Mutation-probe per
    ``feedback_regex_alternation_sibling_coverage.md``: every constant has
    its own dedicated injection test in
    ``tests/test_pipeline/test_check_drift_fast_lane_oos_power_gate.py``.

    Literature grounding for the gate itself:
      - ``research/oos_power.py::POWER_TIERS`` (0.50 DIRECTIONAL_ONLY floor —
        the canonical source the gate constants mirror)
      - ``backtesting-methodology.md`` RULE 3.3 (canonical OOS-power doctrine)
      - ``docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md``
        Thm 1 / Eq. 6 (MinBTL bound — trial-budget preservation)
    """
    import math as _math

    try:
        from scripts.research.fast_lane_promote_queue import (  # type: ignore[import-untyped]
            OOS_COHEN_D_TARGET,
            OOS_POWER_FLOOR,
            STATUS_VALUES,
        )
    except Exception as exc:
        return [
            f"check_fast_lane_oos_power_gate_constants_grounded: scanner import "
            f"failed: {exc}. The gate constants must remain importable from "
            "scripts/research/fast_lane_promote_queue.py."
        ]

    errors: list[str] = []

    if not isinstance(OOS_POWER_FLOOR, (int, float)) or isinstance(OOS_POWER_FLOOR, bool):
        errors.append(
            f"check_fast_lane_oos_power_gate_constants_grounded: "
            f"OOS_POWER_FLOOR must be a numeric (not bool); got "
            f"{type(OOS_POWER_FLOOR).__name__}={OOS_POWER_FLOOR!r}."
        )
    elif _math.isnan(float(OOS_POWER_FLOOR)) or _math.isinf(float(OOS_POWER_FLOOR)):
        errors.append(
            "check_fast_lane_oos_power_gate_constants_grounded: "
            f"OOS_POWER_FLOOR must be finite; got {OOS_POWER_FLOOR!r}."
        )
    elif not (0.0 < float(OOS_POWER_FLOOR) <= 1.0):
        errors.append(
            "check_fast_lane_oos_power_gate_constants_grounded: "
            f"OOS_POWER_FLOOR={OOS_POWER_FLOOR!r} out of valid range (0.0, 1.0]. "
            "Power floor below 0 or above 1 has no statistical meaning; mirror "
            "research/oos_power.py::POWER_TIERS (canonical 0.50)."
        )

    if not isinstance(OOS_COHEN_D_TARGET, (int, float)) or isinstance(
        OOS_COHEN_D_TARGET, bool
    ):
        errors.append(
            f"check_fast_lane_oos_power_gate_constants_grounded: "
            f"OOS_COHEN_D_TARGET must be a numeric (not bool); got "
            f"{type(OOS_COHEN_D_TARGET).__name__}={OOS_COHEN_D_TARGET!r}."
        )
    elif _math.isnan(float(OOS_COHEN_D_TARGET)) or _math.isinf(
        float(OOS_COHEN_D_TARGET)
    ):
        errors.append(
            "check_fast_lane_oos_power_gate_constants_grounded: "
            f"OOS_COHEN_D_TARGET must be finite; got {OOS_COHEN_D_TARGET!r}."
        )
    elif float(OOS_COHEN_D_TARGET) <= 0.0:
        errors.append(
            "check_fast_lane_oos_power_gate_constants_grounded: "
            f"OOS_COHEN_D_TARGET={OOS_COHEN_D_TARGET!r} must be > 0. Cohen's d "
            "below zero inverts the power calc and zero would silently pass all "
            "cells (one_sample_power returns 0 at d<=0). Cohen 1988 conventions: "
            "0.2 small, 0.5 medium, 0.8 large; project default 0.3."
        )

    if "REJECTED_OOS_UNPOWERED" not in STATUS_VALUES:
        errors.append(
            "check_fast_lane_oos_power_gate_constants_grounded: STATUS_VALUES "
            "does not list 'REJECTED_OOS_UNPOWERED' -- the gate emits this "
            "status but the scanner's enum tuple would render no bucket for it, "
            "silently dropping rejected entries from the queue report. "
            f"Current STATUS_VALUES={STATUS_VALUES!r}."
        )

    return errors


def check_bridge_methodology_rules_parity(
    rules_path: Path | None = None,
) -> list[str]:
    """Parity check: bridge METHODOLOGY_RULES_APPLIED vs backtesting-methodology.md.

    The fast-lane -> heavyweight bridge at
    ``scripts/research/fast_lane_to_heavyweight_bridge.py`` inlines a tuple
    of methodology-rule slugs (``rule_1_temporal_alignment``,
    ``rule_3_is_oos_discipline``, etc.) and uses them as keys in the
    ``methodology_rules_applied`` block of every emitted draft heavyweight
    prereg. The slugs must correspond to real RULE headings in the canonical
    methodology doc; an extra slug here means the bridge emits a fake rule
    citation that no doctrine backs, a removed canonical rule means a stale
    citation lingers in every draft (6th confirmed instance of the
    [[canonical-inline-copy-parity-bug-class]] -- see
    ``memory/feedback_canonical_inline_copy_parity_bug_class.md``).

    This check parses ``.claude/rules/backtesting-methodology.md`` for
    ``## RULE N: ...`` headings, derives a canonical set of slug stems
    (``rule_<N>_*``), and asserts every bridge slug is a subset of that
    canonical set.

    Authority: ``.claude/rules/backtesting-methodology.md`` is the single
    source of truth for methodology RULE numbering.

    Parameters
    ----------
    rules_path : Path | None
        Override path to the methodology doc (test seam). Defaults to the
        canonical project location.

    Returns
    -------
    list[str]
        Violation strings -- empty when every bridge slug maps to a real RULE.
        Fail-closed when the doc cannot be read or no RULE headings parse.
    """
    try:
        from scripts.research import fast_lane_to_heavyweight_bridge as bridge
    except Exception as exc:
        return [
            f"check_bridge_methodology_rules_parity: bridge import failed: {exc}"
        ]

    target = (
        rules_path
        if rules_path is not None
        else PROJECT_ROOT / ".claude" / "rules" / "backtesting-methodology.md"
    )

    if not target.exists():
        return [
            f"check_bridge_methodology_rules_parity: canonical doctrine missing at "
            f"{target} (bridge methodology parity cannot be verified -- fail-closed)"
        ]

    try:
        text = target.read_text(encoding="utf-8")
    except Exception as exc:
        return [
            f"check_bridge_methodology_rules_parity: failed to read {target.name}: "
            f"{type(exc).__name__}: {exc}"
        ]

    # Parse ``## RULE <N>:`` headings. Each match produces an integer rule
    # number; the canonical slug stem is ``rule_<N>_``.
    rule_numbers: set[int] = set()
    for m in re.finditer(r"^##\s+RULE\s+(\d+)\b", text, re.MULTILINE):
        try:
            rule_numbers.add(int(m.group(1)))
        except ValueError:
            continue

    if not rule_numbers:
        return [
            f"check_bridge_methodology_rules_parity: no `## RULE N:` headings "
            f"parsed from {target.name} (doctrine structure drifted -- "
            "investigate before promoting the check)"
        ]

    violations: list[str] = []
    for slug in bridge.METHODOLOGY_RULES_APPLIED:
        m = re.match(r"^rule_(\d+)_", slug)
        if m is None:
            violations.append(
                f"check_bridge_methodology_rules_parity: bridge slug "
                f"{slug!r} does not match the canonical rule_<N>_<name> "
                "shape (expected `rule_1_temporal_alignment`, etc.)"
            )
            continue
        n = int(m.group(1))
        if n not in rule_numbers:
            violations.append(
                f"check_bridge_methodology_rules_parity: bridge slug "
                f"{slug!r} cites RULE {n}, which is not present in "
                f"{target.name} (canonical rule set: "
                f"{sorted(rule_numbers)!r}). Inline-copy drift -- see "
                "[[canonical-inline-copy-parity-bug-class]]; "
                "update METHODOLOGY_RULES_APPLIED or amend the doctrine."
            )

    return violations


def check_canonical_inline_copies_have_parity_check() -> list[str]:
    """Meta-check: every CANONICAL_INLINE_COPIES entry has a live parity guard.

    Layer 2 of the canonical-inline-copy-parity-bug-class defense
    (see ``memory/feedback_canonical_inline_copy_parity_bug_class.md``).
    Iterates ``pipeline.canonical_inline_copies.CANONICAL_INLINE_COPIES``
    and for each registered ``InlineCopyPair`` asserts:

      (a) ``entry.parity_check`` exists in ``pipeline.check_drift`` module
          globals and is callable.
      (b) ``entry.test_file`` exists on disk and is non-empty.
      (c) The test file contains at least ``len(entry.gated_constants)``
          ``def test_`` functions (sibling-coverage doctrine per
          ``feedback_regex_alternation_sibling_coverage.md``).

    Fail-closed: a missing ``pipeline.canonical_inline_copies`` module,
    an unreadable test file, or an entry whose ``parity_check`` is not
    a callable are all violations -- never a silent pass.

    The check does NOT execute the per-pair parity functions itself; those
    run via their own CHECKS entries. This is structural enforcement only:
    the registry must not list orphan entries, and registered entries must
    carry sibling-coverage tests.
    """
    try:
        from pipeline.canonical_inline_copies import (
            CANONICAL_INLINE_COPIES,
            InlineCopyPair,
        )
    except Exception as exc:
        return [
            "check_canonical_inline_copies_have_parity_check: "
            f"could not import pipeline.canonical_inline_copies: "
            f"{type(exc).__name__}: {exc} "
            "(Layer 2 registry missing — Stage 2 not landed?)"
        ]

    if not isinstance(CANONICAL_INLINE_COPIES, list):
        return [
            "check_canonical_inline_copies_have_parity_check: "
            "CANONICAL_INLINE_COPIES is not a list "
            f"(got {type(CANONICAL_INLINE_COPIES).__name__})"
        ]

    if len(CANONICAL_INLINE_COPIES) == 0:
        return [
            "check_canonical_inline_copies_have_parity_check: "
            "CANONICAL_INLINE_COPIES is empty -- registry must seed at least "
            "the fast_lane_promote_threshold pair (Check #158 anchor). "
            "If you intentionally cleared it, remove this check too."
        ]

    violations: list[str] = []
    module_globals = globals()

    for idx, entry in enumerate(CANONICAL_INLINE_COPIES):
        # Build prefix defensively — entry may not have `.name` if the
        # type check below fires (caller passed a dict by accident).
        entry_name = getattr(entry, "name", None)
        prefix = (
            f"check_canonical_inline_copies_have_parity_check: "
            f"entry[{idx}] name={entry_name!r}"
        )

        if not isinstance(entry, InlineCopyPair):
            violations.append(
                f"{prefix}: not an InlineCopyPair (got "
                f"{type(entry).__name__})"
            )
            continue

        # (a) parity_check must exist in module globals and be callable.
        check_fn = module_globals.get(entry.parity_check)
        if check_fn is None:
            violations.append(
                f"{prefix}: parity check {entry.parity_check!r} not found "
                "in pipeline.check_drift module globals (registry entry "
                "orphaned -- add the parity-check function or remove the "
                "entry)"
            )
        elif not callable(check_fn):
            violations.append(
                f"{prefix}: parity check {entry.parity_check!r} exists but "
                f"is not callable (type={type(check_fn).__name__})"
            )

        # (b) test_file must exist and be non-empty.
        test_path = PROJECT_ROOT / entry.test_file
        if not test_path.exists():
            violations.append(
                f"{prefix}: injection-test file {entry.test_file!r} "
                "not found (sibling-coverage doctrine requires "
                "tests/test_pipeline/<slug>.py for every registered pair)"
            )
            continue
        try:
            test_text = test_path.read_text(encoding="utf-8")
        except Exception as exc:
            violations.append(
                f"{prefix}: could not read injection-test file "
                f"{entry.test_file!r}: {type(exc).__name__}: {exc}"
            )
            continue
        if not test_text.strip():
            violations.append(
                f"{prefix}: injection-test file {entry.test_file!r} is empty"
            )
            continue

        # (c) sibling-coverage: >= one test function per gated constant.
        expected_n = max(1, len(entry.gated_constants))
        # Robust count: any line where a stripped `def test_` appears.
        # We do NOT parse the AST here -- regex on stripped lines is
        # sufficient and matches the cheap-grep convention used by
        # neighbouring checks in this file.
        test_fn_count = sum(
            1
            for line in test_text.splitlines()
            if line.lstrip().startswith("def test_")
        )
        if test_fn_count < expected_n:
            violations.append(
                f"{prefix}: expected >= {expected_n} test functions "
                f"in {entry.test_file!r} (one per gated constant: "
                f"{', '.join(entry.gated_constants)}), found {test_fn_count} "
                "-- sibling-coverage doctrine violation "
                "(see memory/feedback_regex_alternation_sibling_coverage.md)"
            )

    return violations


def check_triage_provenance_completeness(
    drafts_dir: Path | None = None,
) -> list[str]:
    """Triage-draft provenance integrity (Improvement 3 / Stage C, 2026-05-19).

    Every yaml under ``docs/audit/hypotheses/drafts/`` that contains a
    ``triage_provenance:`` block MUST declare
    ``triage_provenance.source_validated_setup_strategy_id``. Without that
    field, an auditor cannot trace the draft back to its canonical source
    row in ``validated_setups`` — the entire feedback loop the triage
    script supplies becomes opaque.

    Drafts hand-authored without a ``triage_provenance:`` block are not
    affected — the check fires only when the marker is present (i.e. when
    the operator OR the triage script asserts this is triage-provenance
    data).

    Fail-closed on YAML parse failure. Drafts directory missing is an
    expected fresh-tree state and emits no violations.

    Parameters
    ----------
    drafts_dir : Path | None
        Override drafts dir (test seam).
    """
    target = (
        drafts_dir
        if drafts_dir is not None
        else PROJECT_ROOT / "docs" / "audit" / "hypotheses" / "drafts"
    )
    if not target.exists():
        return []

    try:
        import yaml as _yaml
    except Exception as exc:
        return [
            "check_triage_provenance_completeness: pyyaml import failed: "
            f"{type(exc).__name__}: {exc}"
        ]

    violations: list[str] = []
    for path in sorted(target.glob("*.yaml")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            violations.append(
                f"check_triage_provenance_completeness: cannot read {path.name}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        # Fast skip: no triage_provenance marker -> not a triage draft.
        if "triage_provenance:" not in text:
            continue

        try:
            parsed = _yaml.safe_load(text)
        except Exception as exc:
            violations.append(
                f"check_triage_provenance_completeness: {path.name} carries "
                f"triage_provenance marker but failed to parse: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if not isinstance(parsed, dict):
            violations.append(
                f"check_triage_provenance_completeness: {path.name} root is not "
                f"a YAML mapping (got {type(parsed).__name__})"
            )
            continue

        block = parsed.get("triage_provenance")
        if not isinstance(block, dict):
            violations.append(
                f"check_triage_provenance_completeness: {path.name} "
                f"triage_provenance must be a mapping (got {type(block).__name__})"
            )
            continue

        sid = block.get("source_validated_setup_strategy_id")
        if not isinstance(sid, str) or not sid.strip():
            violations.append(
                f"check_triage_provenance_completeness: {path.name} "
                "triage_provenance.source_validated_setup_strategy_id is "
                "required and must be a non-empty string "
                "(every triage draft must trace back to a canonical "
                "validated_setups row; see Stage C / Improvement 3 plan)"
            )

    return violations


def check_cherry_pick_journal_integrity(
    journal_path: Path | None = None,
    queue_path: Path | None = None,
) -> list[str]:
    """Cherry-pick journal entry-integrity gate.

    Every promote_queue.yaml entry that has been ESCALATED (``heavyweight_prereg``
    is non-null, signalling the operator authored a heavyweight draft via the
    fast_lane_to_heavyweight_bridge) MUST have a corresponding journal entry
    for that ``strategy_id`` in ``docs/runtime/cherry_pick_journal.yaml``.

    Rationale: the cherry-pick journal is the data substrate the ranker's
    era_stability_proxy will eventually consume. If escalations land without
    journal entries, the substrate develops silent gaps and the eventual
    proxy-weight calibration is invalidated. Caught early via this check.

    Additional structural assertions:
      (a) Journal entries have monotonically-increasing ``iter`` starting at 1
          (append-only contract — no insertions, no deletions).
      (b) Required fields are present per entry (``iter``, ``date``,
          ``strategy_id``, ``rank_score``, ``components``, ``pooled_t``,
          ``pooled_n``, ``oos_n``, ``oos_power_tier``).
      (c) ``oos_power_tier`` is one of the allowed categorical values.

    Fail-closed when files are missing or malformed.

    Parameters
    ----------
    journal_path : Path | None
        Override the journal path (test seam).
    queue_path : Path | None
        Override the promote queue path (test seam).
    """
    j_path = (
        journal_path
        if journal_path is not None
        else PROJECT_ROOT / "docs" / "runtime" / "cherry_pick_journal.yaml"
    )
    q_path = (
        queue_path
        if queue_path is not None
        else PROJECT_ROOT / "docs" / "runtime" / "promote_queue.yaml"
    )

    if not j_path.exists():
        return [
            "check_cherry_pick_journal_integrity: journal missing at "
            f"{j_path} (cherry_pick_journal.yaml is required substrate -- "
            "create it via scripts/research/cherry_pick_ranker.py --write-journal "
            "or seed an empty schema_version: 1, entries: [] file)"
        ]

    try:
        import yaml as _yaml
    except Exception as exc:
        return [
            "check_cherry_pick_journal_integrity: pyyaml import failed: "
            f"{type(exc).__name__}: {exc}"
        ]

    try:
        journal = _yaml.safe_load(j_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            f"check_cherry_pick_journal_integrity: failed to parse {j_path.name}: "
            f"{type(exc).__name__}: {exc}"
        ]

    if not isinstance(journal, dict):
        return [
            "check_cherry_pick_journal_integrity: journal payload is not a YAML "
            f"mapping (got {type(journal).__name__})"
        ]

    entries = journal.get("entries")
    if not isinstance(entries, list):
        return [
            "check_cherry_pick_journal_integrity: journal `entries` field must "
            f"be a list (got {type(entries).__name__})"
        ]

    violations: list[str] = []

    ALLOWED_TIERS = {
        "CAN_REFUTE",
        "DIRECTIONAL_ONLY",
        "STATISTICALLY_USELESS",
        "NA_NO_OOS",
        "NA_N_BELOW_FLOOR",
    }
    REQUIRED_FIELDS = (
        "iter",
        "date",
        "strategy_id",
        "rank_score",
        "components",
        "pooled_n",
        "oos_n",
        "oos_power_tier",
    )

    prev_iter = 0
    seen_strategy_ids: set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            violations.append(
                f"check_cherry_pick_journal_integrity: entry[{idx}] is not a "
                f"YAML mapping (got {type(entry).__name__})"
            )
            continue

        for field in REQUIRED_FIELDS:
            if field not in entry:
                violations.append(
                    f"check_cherry_pick_journal_integrity: entry[{idx}] missing "
                    f"required field {field!r}"
                )

        cur_iter = entry.get("iter")
        if not isinstance(cur_iter, int):
            violations.append(
                f"check_cherry_pick_journal_integrity: entry[{idx}] iter is not "
                f"an int (got {type(cur_iter).__name__})"
            )
        elif cur_iter != prev_iter + 1:
            violations.append(
                f"check_cherry_pick_journal_integrity: entry[{idx}] iter={cur_iter} "
                f"breaks monotonic append-only contract (expected {prev_iter + 1})"
            )
        if isinstance(cur_iter, int):
            prev_iter = cur_iter

        tier = entry.get("oos_power_tier")
        if isinstance(tier, str) and tier not in ALLOWED_TIERS:
            violations.append(
                f"check_cherry_pick_journal_integrity: entry[{idx}] "
                f"oos_power_tier={tier!r} is not one of "
                f"{sorted(ALLOWED_TIERS)} "
                "(see backtesting-methodology.md § RULE 3.3 + ranker oos_power_tier_for)"
            )

        sid = entry.get("strategy_id")
        if isinstance(sid, str):
            seen_strategy_ids.add(sid)

    if not q_path.exists():
        # No promote_queue is acceptable on a fresh tree -- skip the escalation
        # cross-check rather than fail. The integrity assertions above still apply.
        return violations

    try:
        queue = _yaml.safe_load(q_path.read_text(encoding="utf-8"))
    except Exception as exc:
        violations.append(
            f"check_cherry_pick_journal_integrity: failed to parse {q_path.name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return violations

    if not isinstance(queue, dict):
        return violations

    queue_entries = queue.get("entries")
    if not isinstance(queue_entries, list):
        return violations

    # Cross-check: escalated promote-queue rows (heavyweight_prereg set) must
    # have a journal entry. REVOKED, PARKED, and plain-QUEUED rows are exempt
    # -- they have not yet flowed through the ranker.
    for idx, qe in enumerate(queue_entries):
        if not isinstance(qe, dict):
            continue
        hp = qe.get("heavyweight_prereg")
        if not hp:
            continue
        sid = qe.get("strategy_id")
        if not isinstance(sid, str):
            continue
        if sid not in seen_strategy_ids:
            violations.append(
                f"check_cherry_pick_journal_integrity: promote_queue entry[{idx}] "
                f"strategy_id={sid!r} has heavyweight_prereg={hp!r} but no matching "
                "journal entry -- escalations must journal "
                "(run scripts/research/cherry_pick_ranker.py --write-journal "
                "before bridging to heavyweight, or backfill the entry by hand)"
            )

    return violations


def check_am33_audit_log_theory_grant_parity(
    audit_log_path: Path | None = None,
    hypotheses_dir: Path | None = None,
) -> list[str]:
    """Check #165: chordia_audit_log.yaml theory_grants must match prereg metadata.theory_grant.

    Amendment 3.3 (2026-05-17) introduced ``metadata.theory_grant`` as the
    EXPLICIT bool governing the Chordia t-threshold gate in the hypothesis
    loader.  The live allocator gate (``trading_app/chordia.py``) reads
    ``has_theory`` from ``chordia_audit_log.yaml`` via ``ChordiaAuditLog`` --
    an INDEPENDENT trust surface (documented deferred finding AM3.3-AUDIT-LOG-DRIFT,
    evidence-auditor 2026-05-17).

    If an operator manually writes ``has_theory: true`` in the audit log for a
    strategy whose prereg declares ``theory_grant: false``, the allocator silently
    applies t>=3.00 instead of the strict t>=3.79 declared by the prereg -- a
    capital-class miscalculation.

    This check asserts parity for every strategy_id that appears in BOTH surfaces:

    (a) Audit-log side: every entry in ``theory_grants`` with ``has_theory: true``
        (has_theory=false entries are the default and do not need a journal entry).
    (b) Prereg side: every active prereg YAML under ``docs/audit/hypotheses/``
        (NOT in the ``drafts/`` sub-directory -- drafts are quarantined and not
        yet preregistered) whose ``scope.strategy_id`` OR any
        ``primary_schema.family_cells[*].strategy_id`` matches a theory_grants entry.

    Violation cases:
    - Audit log has ``has_theory: true`` for SID, active prereg declares
      ``theory_grant: false`` (or is missing the field) -> VIOLATION.
    - Active prereg declares ``theory_grant: true`` for SID, audit log entry for
      that SID has ``has_theory: false`` (explicit entry in ``theory_grants`` with
      false, OR SID absent from ``theory_grants`` while prereg says true) -> VIOLATION.

    Fail-closed: missing or unparseable audit log returns a violation rather than
    silently passing.  Missing hypotheses dir returns a violation.  A prereg that
    cannot be parsed is skipped with a warning (not a hard violation -- the YAML
    validator catches broken syntax separately).

    Parameters
    ----------
    audit_log_path : Path | None
        Override the audit log path (test seam).
    hypotheses_dir : Path | None
        Override the hypotheses directory (test seam).
    """
    try:
        import yaml as _yaml
    except ImportError as exc:
        return [f"check_am33_audit_log_theory_grant_parity: pyyaml import failed: {exc}"]

    _audit_path = (
        audit_log_path
        if audit_log_path is not None
        else PROJECT_ROOT / "docs" / "runtime" / "chordia_audit_log.yaml"
    )
    _hyp_dir = (
        hypotheses_dir
        if hypotheses_dir is not None
        else PROJECT_ROOT / "docs" / "audit" / "hypotheses"
    )

    # --- Load and validate audit log ---
    if not _audit_path.exists():
        return [
            f"check_am33_audit_log_theory_grant_parity: chordia_audit_log.yaml not found at "
            f"{_audit_path} -- this file is required for the Chordia allocator gate "
            "(trading_app/chordia.py::ChordiaAuditLog)"
        ]
    try:
        audit_raw = _yaml.safe_load(_audit_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            f"check_am33_audit_log_theory_grant_parity: failed to parse chordia_audit_log.yaml: "
            f"{type(exc).__name__}: {exc}"
        ]
    if not isinstance(audit_raw, dict):
        return [
            "check_am33_audit_log_theory_grant_parity: chordia_audit_log.yaml payload is not "
            f"a YAML mapping (got {type(audit_raw).__name__})"
        ]

    # Build the audit-log theory-grant map: {strategy_id: bool}
    # Only entries explicitly listed in theory_grants[] are tracked; the default
    # (default_has_theory, typically false) is not materialised here because the
    # check only fires on strategy_ids that appear in BOTH surfaces.
    audit_theory_map: dict[str, bool] = {}
    theory_grants_raw = audit_raw.get("theory_grants") or []
    if not isinstance(theory_grants_raw, list):
        return [
            "check_am33_audit_log_theory_grant_parity: chordia_audit_log.yaml "
            f"theory_grants is not a list (got {type(theory_grants_raw).__name__})"
        ]
    for entry in theory_grants_raw:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("strategy_id")
        if not isinstance(sid, str) or not sid:
            continue
        has_theory = entry.get("has_theory")
        if isinstance(has_theory, bool):
            audit_theory_map[sid] = has_theory

    # --- Scan active prereqs ---
    if not _hyp_dir.exists():
        return [
            f"check_am33_audit_log_theory_grant_parity: hypotheses dir not found at {_hyp_dir}"
        ]

    # Active prereqs: docs/audit/hypotheses/*.yaml (NOT drafts/ sub-directory).
    active_yamls = [
        p for p in _hyp_dir.glob("*.yaml")
        if p.is_file() and "drafts" not in str(p)
    ]

    violations: list[str] = []

    for prereq_path in sorted(active_yamls):
        try:
            prereq_raw = _yaml.safe_load(prereq_path.read_text(encoding="utf-8"))
        except Exception:
            # Broken YAML is caught by the YAML-syntax validator; skip silently here.
            continue
        if not isinstance(prereq_raw, dict):
            continue

        meta = prereq_raw.get("metadata") or {}
        if not isinstance(meta, dict):
            continue
        prereq_theory_grant = meta.get("theory_grant")
        if not isinstance(prereq_theory_grant, bool):
            # Missing theory_grant is caught by hypothesis_loader; skip here.
            continue

        # Skip multi-lane diagnostic/triage/diligence prereqs.
        # These reference existing deployed lanes as objects of study but do not
        # independently assert theory grants -- the audit log is the ground truth
        # for deployed lanes.  Only single-SID (testing_mode: individual or family
        # without multi-lane scope) prereqs make theory claims on specific SIDs.
        rqt = meta.get("research_question_type", "")
        if rqt in ("capital_diligence", "triage"):
            continue

        # Collect strategy_ids this prereg covers (scope.strategy_id + family_cells).
        # NOTE: scope.lanes[] shape is intentionally NOT included here -- those are
        # multi-lane audit/triage docs (see exclusion above).  The scope.lanes shape
        # references lanes as objects of study, not theory-grant claims.
        covered_sids: set[str] = set()
        scope = prereq_raw.get("scope") or {}
        if isinstance(scope, dict):
            # Skip entirely if this prereg references multiple lanes in scope.lanes[].
            # That shape is a multi-lane diagnostic even when rqt is not explicitly set.
            lanes_list = scope.get("lanes")
            if isinstance(lanes_list, list) and len(lanes_list) > 1:
                continue
            sid = scope.get("strategy_id")
            if isinstance(sid, str) and sid:
                covered_sids.add(sid)
        primary = prereq_raw.get("primary_schema") or {}
        if isinstance(primary, dict):
            cells = primary.get("family_cells") or []
            if isinstance(cells, list):
                for cell in cells:
                    if isinstance(cell, dict):
                        csid = cell.get("strategy_id")
                        if isinstance(csid, str) and csid:
                            covered_sids.add(csid)

        # For each covered SID that appears in the audit-log theory_grants map,
        # assert parity.
        for sid in sorted(covered_sids):
            if sid not in audit_theory_map:
                # SID not listed in theory_grants at all; audit log implicitly
                # treats it as has_theory=default_has_theory (typically false).
                # Only flag if the prereg claims true (that's the dangerous direction).
                if prereq_theory_grant is True:
                    default_ht = bool(audit_raw.get("default_has_theory", False))
                    if not default_ht:
                        violations.append(
                            f"check_am33_audit_log_theory_grant_parity: PARITY MISMATCH "
                            f"strategy_id={sid!r} -- prereg {prereq_path.name} declares "
                            f"metadata.theory_grant=true but chordia_audit_log.yaml has NO "
                            f"theory_grants entry for this strategy_id (implicit "
                            f"has_theory={default_ht}). "
                            f"Allocator applies t>={3.0 if default_ht else 3.79:.2f}; "
                            f"prereg declares t>=3.00 hurdle. "
                            f"Add strategy_id to chordia_audit_log.yaml theory_grants with "
                            f"has_theory: true AND theory_ref, OR flip prereg to "
                            f"theory_grant: false. "
                            f"Doctrine: AM3.3-AUDIT-LOG-DRIFT deferred finding "
                            f"(docs/ralph-loop/deferred-findings.md)."
                        )
                continue

            audit_ht = audit_theory_map[sid]
            if audit_ht != prereq_theory_grant:
                allocator_t = 3.00 if audit_ht else 3.79
                prereg_t = 3.00 if prereq_theory_grant else 3.79
                violations.append(
                    f"check_am33_audit_log_theory_grant_parity: PARITY MISMATCH "
                    f"strategy_id={sid!r} -- "
                    f"chordia_audit_log.yaml theory_grants has_theory={audit_ht} "
                    f"(allocator threshold t>={allocator_t:.2f}) but "
                    f"prereg {prereq_path.name} declares "
                    f"metadata.theory_grant={prereq_theory_grant} "
                    f"(loader threshold t>={prereg_t:.2f}). "
                    f"Mismatch silently applies different t-hurdles on the two "
                    f"independent trust surfaces (Amendment 3.3 evidence-auditor "
                    f"finding, 2026-05-17). Fix by aligning both to the same bool. "
                    f"Doctrine: AM3.3-AUDIT-LOG-DRIFT (docs/ralph-loop/deferred-findings.md)."
                )

    return violations


def check_fast_lane_state_graph_node_parity(
    spec_path: Path | None = None,
    runtime_dir: Path | None = None,
) -> list[str]:
    """Check #166: fast-lane state-graph node list must match on-disk derived state.

    Parses the ``## Node Inventory`` YAML block in
    ``docs/specs/fast_lane_state_graph.md`` and verifies symmetric parity
    with the on-disk derived-state files referenced by the fast-lane prereg
    pipeline (Stage 1 of
    ``docs/plans/2026-05-19-fast-lane-pipeline-connective-tissue-design.md``).

    Two violation directions:

    (a) ORPHAN-NODE: an ``active`` node (``proposed: true`` absent or false)
        in the spec names a file/dir path that does not exist on disk.

    (b) ORPHAN-FILE: a derived-state file under the known glob roots
        (``docs/runtime/promote_queue.yaml``,
        ``docs/runtime/cherry_pick_journal.yaml``,
        ``docs/runtime/cherry_pick_ranking_*.csv``,
        ``docs/audit/hypotheses/drafts/``) is not named by any spec node.

    Reserved (``proposed: true``) nodes are excluded from BOTH directions —
    Stage 2 and Stage 3 schemas live in the spec but are not yet shipped.

    Fail-closed:
    - Missing or unparseable spec returns a violation (no silent pass).
    - Missing ``## Node Inventory`` heading returns a violation.
    - YAML block that fails to parse returns a violation.

    Parameters
    ----------
    spec_path : Path | None
        Override spec doc path (test seam).
    runtime_dir : Path | None
        Override repo root for derived-state globs (test seam). Defaults to
        ``PROJECT_ROOT`` so globs resolve against the live repo.
    """
    try:
        import yaml as _yaml
    except ImportError as exc:
        return [f"check_fast_lane_state_graph_node_parity: pyyaml import failed: {exc}"]

    _spec = (
        spec_path
        if spec_path is not None
        else PROJECT_ROOT / "docs" / "specs" / "fast_lane_state_graph.md"
    )
    _root = runtime_dir if runtime_dir is not None else PROJECT_ROOT

    if not _spec.exists():
        return [
            f"check_fast_lane_state_graph_node_parity: spec doc not found at {_spec} -- "
            "this file is the single source of truth for the fast-lane chain "
            "(Stage 1 of docs/plans/2026-05-19-fast-lane-pipeline-connective-tissue-design.md)."
        ]

    try:
        raw = _spec.read_text(encoding="utf-8")
    except Exception as exc:
        return [
            f"check_fast_lane_state_graph_node_parity: failed to read {_spec}: "
            f"{type(exc).__name__}: {exc}"
        ]

    # Extract the YAML block under "## 2. Node Inventory". The block is
    # delimited by triple-backtick yaml fences; we take the FIRST yaml fence
    # appearing after the Node Inventory heading.
    heading_match = re.search(r"^##\s+2\.\s+Node Inventory\s*$", raw, re.MULTILINE)
    if not heading_match:
        return [
            f"check_fast_lane_state_graph_node_parity: spec doc {_spec} missing "
            "'## 2. Node Inventory' heading -- parser cannot locate the canonical "
            "node list. Restore the heading or update the parser."
        ]

    after_heading = raw[heading_match.end():]
    fence_match = re.search(r"```yaml\s*\n(.*?)\n```", after_heading, re.DOTALL)
    if not fence_match:
        return [
            f"check_fast_lane_state_graph_node_parity: spec doc {_spec} has no "
            "```yaml fenced block after '## 2. Node Inventory' -- node list cannot "
            "be parsed."
        ]

    yaml_body = fence_match.group(1)
    try:
        parsed = _yaml.safe_load(yaml_body)
    except Exception as exc:
        return [
            f"check_fast_lane_state_graph_node_parity: failed to parse Node Inventory "
            f"YAML block in {_spec}: {type(exc).__name__}: {exc}"
        ]

    if not isinstance(parsed, dict) or "nodes" not in parsed:
        return [
            f"check_fast_lane_state_graph_node_parity: Node Inventory YAML in {_spec} "
            "is not a mapping with a top-level 'nodes:' key."
        ]

    nodes_raw = parsed.get("nodes")
    if not isinstance(nodes_raw, list):
        return [
            f"check_fast_lane_state_graph_node_parity: 'nodes' in {_spec} is not a list "
            f"(got {type(nodes_raw).__name__})."
        ]

    violations: list[str] = []
    active_paths: set[str] = set()  # normalised repo-relative path strings

    # --- Direction (a): ORPHAN-NODE — every active node path must resolve on disk ---
    for idx, node in enumerate(nodes_raw):
        if not isinstance(node, dict):
            violations.append(
                f"check_fast_lane_state_graph_node_parity: node[{idx}] in {_spec} "
                f"is not a mapping (got {type(node).__name__})."
            )
            continue
        node_id = node.get("id", f"<unnamed-node-{idx}>")
        node_path = node.get("path")
        if not isinstance(node_path, str) or not node_path:
            violations.append(
                f"check_fast_lane_state_graph_node_parity: node[{idx}] (id={node_id!r}) "
                f"missing required 'path' string."
            )
            continue
        is_proposed = bool(node.get("proposed", False))
        if is_proposed:
            continue  # Stage 2/3 reserved nodes — excluded from parity until they ship.

        active_paths.add(node_path)
        # Resolve path on disk. Treat trailing '/' as directory; '*' as glob.
        target = _root / node_path
        if "*" in node_path:
            matches = list(_root.glob(node_path))
            if not matches:
                violations.append(
                    f"check_fast_lane_state_graph_node_parity: ORPHAN-NODE -- spec node "
                    f"id={node_id!r} declares glob path {node_path!r} but no file matches "
                    f"on disk under {_root}. Either remove the node, mark it "
                    "'proposed: true', or write a matching file."
                )
        elif node_path.endswith("/"):
            if not target.is_dir():
                violations.append(
                    f"check_fast_lane_state_graph_node_parity: ORPHAN-NODE -- spec node "
                    f"id={node_id!r} declares directory path {node_path!r} but the "
                    f"directory does not exist at {target}. Either remove the node, "
                    "mark it 'proposed: true', or create the directory."
                )
        else:
            if not target.exists():
                violations.append(
                    f"check_fast_lane_state_graph_node_parity: ORPHAN-NODE -- spec node "
                    f"id={node_id!r} declares path {node_path!r} but the file does not "
                    f"exist at {target}. Either remove the node, mark it "
                    "'proposed: true', or write the file."
                )

    # --- Direction (b): ORPHAN-FILE — every derived-state file in known glob roots
    # must be named by an active spec node. ---
    # Glob roots that participate in the fast-lane chain (matches design § 3 Stage 3).
    #
    # ASSUMPTION: spec_path_pattern and disk_glob must be IDENTICAL strings so that
    # the `spec_path_pattern not in active_paths` membership test works correctly.
    # active_paths stores node['path'] verbatim from the spec YAML, so a node with
    # path "docs/runtime/cherry_pick_ranking_*.csv" must use the exact same glob
    # string here.  If spec authors vary the glob syntax (e.g. "cherry_pick_ranking_
    # 2026-05-*.csv"), this literal-match will fire a false ORPHAN-FILE violation.
    # To generalise, consider fnmatch-based set intersection instead of membership
    # lookup — deferred until a concrete divergence is observed.
    known_globs: list[tuple[str, str]] = [
        ("docs/runtime/promote_queue.yaml", "docs/runtime/promote_queue.yaml"),
        ("docs/runtime/cherry_pick_journal.yaml", "docs/runtime/cherry_pick_journal.yaml"),
        ("docs/runtime/cherry_pick_ranking_*.csv", "docs/runtime/cherry_pick_ranking_*.csv"),
    ]
    for spec_path_pattern, disk_glob in known_globs:
        matches = list(_root.glob(disk_glob))
        if not matches:
            continue  # absent on disk: that's an ORPHAN-NODE concern, handled above
        if spec_path_pattern not in active_paths:
            violations.append(
                f"check_fast_lane_state_graph_node_parity: ORPHAN-FILE -- on-disk "
                f"derived-state matches {disk_glob!r} ({len(matches)} file(s)) but no "
                f"spec node names {spec_path_pattern!r}. Add a node to "
                "'## 2. Node Inventory' in docs/specs/fast_lane_state_graph.md."
            )
    # Drafts directory is a special case: directory presence, not glob.
    drafts_dir = _root / "docs/audit/hypotheses/drafts"
    if drafts_dir.is_dir() and "docs/audit/hypotheses/drafts/" not in active_paths:
        violations.append(
            "check_fast_lane_state_graph_node_parity: ORPHAN-FILE -- "
            "docs/audit/hypotheses/drafts/ exists on disk but no spec node names "
            "it. Add a node to '## 2. Node Inventory'."
        )

    return violations


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
        "Bot dashboard localhost-only host binding",
        lambda: check_dashboard_localhost_only_binding(TRADING_APP_DIR),
        False,
        False,
    ),
    (
        "Bot dashboard SSE single-worker invariant",
        lambda: check_dashboard_sse_single_worker(TRADING_APP_DIR),
        False,
        False,
    ),
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
    (
        "Aperture hardcode (orb_minutes=5) in lane-iterating scoring paths (PR #189 class bug)",
        lambda: check_aperture_hardcode_in_scoring_paths(TRADING_APP_DIR),
        False,
        False,
    ),
    ("Config filter_type sync", check_config_filter_sync, False, False),
    ("ENTRY_MODELS sync", check_entry_models_sync, False, False),
    ("Entry price sanity", check_entry_price_sanity, False, False),
    ("Nested subpackage isolation", check_nested_isolation, False, False),
    ("All imports resolve", check_all_imports_resolve, False, False),
    ("Cryptography pin holds (FastMCP/Authlib compat)", check_cryptography_pin_holds, False, False),
    ("GARCH dependency importable (`arch` package present)", check_garch_dependency_importable, False, False),
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
    ("Doc hygiene contracts (stamps, design-only, generated markers)", check_doc_hygiene_contracts, False, False),
    ("No duplicate gold.db at project root", check_stale_scratch_db, False, False),
    (
        "Active research code uses pipeline.paths for DB selection",
        check_active_code_uses_canonical_db_path,
        False,
        False,
    ),
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
    (
        "Pre-reg hypothesis files pass Bailey 2013 MinBTL K-budget gate",
        check_hypothesis_minbtl_compliance,
        False,
        False,
    ),
    (
        "Chordia result MD threshold matches prereg chordia_threshold_basis (theory_citation field-presence trap)",
        check_chordia_result_threshold_matches_prereg,
        False,
        False,
    ),
    (
        "Verdict-token vocabulary parity (RESEARCH_RULES.md <-> MCP server constants)",
        check_verdict_vocabulary_md_matches_code,
        False,
        False,
    ),
    (
        "CHECKS list labels are ASCII-only (Windows cp1252 console crash class)",
        check_checks_list_labels_are_ascii,
        False,
        False,
    ),
    ("Cost model completeness (COST_SPECS covers all active instruments)", check_cost_model_completeness, False, False),
    ("TRADING_RULES.md authority values match code", check_trading_rules_authority, False, False),
    # ML drift checks removed 2026-04-11 (ML V1/V2/V3 DEAD — V3 sprint Stage 4):
    #   check_ml_config_canonical_sources, check_ml_lookahead_blacklist,
    #   check_ml_model_files_exist, check_ml_config_hash_match,
    #   check_ml_model_freshness — all validated invariants of a dead subsystem.
    #   See docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md.
    ("Audit columns populated (n_trials, fst_hurdle, DSR)", check_audit_columns_populated, False, True),  # requires_db
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
        "Daily features row integrity (one row per aperture per trading_day x symbol)",
        check_daily_features_row_integrity,
        False,
        True,
    ),  # requires_db
    (
        "HTF level fields match canonical week/month SQL aggregation",
        check_htf_levels_integrity,
        False,
        True,
    ),  # requires_db
    (
        "HTF fields consistent across apertures for each trading_day x symbol",
        check_htf_aperture_consistency,
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
        "Recent late-history GARCH feature coverage is intact",
        check_recent_garch_feature_coverage,
        False,
        True,
    ),  # requires_db
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
        "Stop multiplier ID-column consistency (_S075 <-> stop_multiplier)",
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
    # ML-layer drift checks removed 2026-04-11 (ML V1/V2/V3 DEAD):
    #   check_ml_evaluate_hybrid_support, check_ml_bundle_full_delta,
    #   check_ml_sharpe_jk_pvalue, check_ml_no_iterrows_filters
    (
        "SQL column convention: pipeline tables use 'symbol', trading app tables use 'instrument'",
        check_symbol_instrument_sql_convention,
        False,
        False,
    ),
    ("Session guard ordering canonical source retained after ML removal", check_session_guard_sync, False, False),
    ("Noise floor gate removed -- no-op since 2026-03-21 canon lock", check_noise_floor_active, False, False),
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
    (
        "Pre-registration file present for post-Phase-0 discovery runs (Criterion 1)",
        check_prereg_present_for_recent_runs,
        True,  # advisory: many already-promoted rows lack prereg files; report, do not block
        True,  # requires_db
    ),
    (
        "Validator pool freshness (frozen discovery_k vs live per-session pool)",
        check_validator_pool_freshness,
        True,  # advisory: drift surfaces silent K mutation, does not block
        True,  # requires_db
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
        "Phase 4 SHA migration manifest integrity (every check_107_sha_migrations.yaml entry is evidence-grounded)",
        check_phase_4_sha_migration_manifest_integrity,
        False,
        False,
    ),
    (
        "validated_setups writes stay on canonical allowlist",
        check_validated_setups_writer_allowlist,
        False,
        False,
    ),
    (
        "critical validated_setups readers use canonical deployable-shelf semantics",
        check_critical_deployable_shelf_consumers,
        False,
        False,
    ),
    (
        "prop_profiles deployed lanes must exist in validated_setups (active=True profiles only)",
        check_prop_profiles_validated_alignment,
        False,
        True,
    ),
    (
        "ACCOUNT_PROFILES entries must declare is_express_funded explicitly (fail-closed default)",
        check_account_profiles_declare_is_express_funded,
        False,
        False,
    ),
    ("Shared profile fingerprint helper is canonical", check_shared_profile_fingerprint_canonical, False, False),
    ("SR state writer uses derived-state contract envelope", check_sr_state_contract_writer, False, False),
    ("SR state reader validates envelope before trust", check_sr_state_contract_reader, False, False),
    ("Preflight launchers pass explicit claim modes", check_preflight_launcher_modes, False, False),
    (
        "Document authority registry exists and core docs advertise their roles",
        check_document_authority_registry,
        False,
        False,
    ),
    ("System authority map exists and classifies linked truth surfaces", check_system_authority_map, False, False),
    (
        "Context-routing registry resolves only to valid domains, profiles, views, and files",
        check_context_routing_registry,
        False,
        False,
    ),
    ("Generated context-routing docs stay in sync with the registry", check_context_generated_docs, False, False),
    (
        "Generated task views preserve strict truth-class boundaries",
        check_context_view_contracts,
        True,
        False,
    ),  # ADVISORY: context/ package not yet committed
    (
        "AGENTS.md points cold-start agents to the deterministic context router",
        check_agents_mentions_context_resolver,
        False,
        False,
    ),
    (
        "Startup docs point non-trivial tasks at the deterministic context router",
        check_startup_docs_reference_context_router,
        False,
        False,
    ),
    ("Phase 7 live audit uses canonical runtime authorities", check_live_audit_uses_runtime_authority, False, False),
    (
        "Project pulse exposes repo identity from canonical authority registry",
        check_project_pulse_uses_authority_registry,
        False,
        False,
    ),
    (
        "DEPLOYABLE_ORB_INSTRUMENTS is a strict subset of ACTIVE_ORB_INSTRUMENTS",
        check_deployable_subset_of_active,
        False,
        False,
    ),
    (
        "Canonical Claude client source (claude_client.py is the only place for Claude model IDs + anthropic.Anthropic)",
        check_canonical_claude_client_source,
        False,
        False,
    ),
    (
        "No CRLF in tracked text blobs (defense-in-depth for pre-commit [0b] auto-renormalize)",
        check_no_crlf_in_tracked_text_blobs,
        False,
        False,
    ),
    (
        "C1 kill-switch guards intact at _on_bar and _handle_event ENTRY branch",
        check_c1_kill_switch_guards_intact,
        False,
        False,
    ),
    (
        "Signal log rotation: _write_signal_record delegates to SignalLogRotator (R4 fix)",
        check_signal_log_rotation_not_bypassed,
        False,
        False,
    ),
    (
        "Routed filter required columns populated (catches ghost deployments)",
        check_routed_filter_columns_populated,
        False,
        True,
    ),
    (
        "Pooled-finding audit files carry per-cell breakdown annotation (RULE 14)",
        check_pooled_finding_annotations,
        False,
        False,
    ),
    (
        "Magic-number rationale audit on trading_app/live/ + HWM tracker + pre_session_check (Carver Ch. 4)",
        lambda: check_magic_number_rationale(TRADING_APP_DIR),
        False,
        False,
    ),
    (
        "NQ-mini symbol-substitution: ACCOUNT_PROFILES populated implies live wiring (PR #158 Stage 2 gate)",
        lambda: check_nq_mini_substitution_wired_or_unused(TRADING_APP_DIR),
        False,
        False,
    ),
    (
        "Research scripts using 'pnl_r IS NOT NULL' annotate scratch policy (2026-04-27 class bug)",
        check_research_scratch_policy_annotation,
        True,  # advisory until Stage 6 of 2026-04-27 plan annotates the 131 pre-existing scripts
        False,
    ),
    (
        "orb_outcomes scratch rows have non-NULL pnl_r post Stage-5 canonical fix",
        check_orb_outcomes_scratch_pnl,
        True,  # advisory until Stage 5b rebuild completes for all instruments
        True,  # requires_db
    ),
    (
        "Stage-file staleness vs landed commits (advisory; prevents re-litigation)",
        check_stage_file_landed_drift,
        True,
        False,
    ),
    (
        "Parked-cells registry completeness (every Pathway B result has matching docs/runtime/parked-cells.yaml entry)",
        check_parked_cells_registry_completeness,
        False,  # BLOCKING — registry drift is a real correctness issue
        False,
    ),
    (
        "E2 entry_model + break-bar features in research/ require e2-lookahead-policy annotation (2026-04-28 class bug)",
        check_e2_lookahead_research_contamination,
        True,  # advisory — surfaces new contamination without blocking commits
        False,
    ),
    (
        "Routine-TBBO slippage pilot v1 PASS coverage in ROUTINE_TBBO_SLIPPAGE_REGISTRY (deployability inference parity)",
        check_routine_tbbo_slippage_registry_coverage,
        False,  # BLOCKING — registry under/over-coverage silently mis-classifies slippage_missing
        False,
    ),
    (
        "Operator-visible formatter helpers must warn on unrecognized types (iso_utc class bug, 2026-05-04)",
        check_iso_utc_formatter_silent_none,
        True,  # advisory — first-week shakeout may surface unenumerated false positives
        False,
    ),
    (
        "SR-monitor pauses on lane_overrides_*.json should be supported by current evidence (2026-05-11 stale-pause class bug)",
        check_sr_pauses_have_recent_evidence,
        True,  # advisory — surfaces stale peak-SR alarms masking recovered streams
        False,
    ),
    # ── CRG-backed checks (D1-D5) — all ADVISORY; fail-open when CRG unavailable ──
    (
        "CRG D1: Surprising cross-layer connections between pipeline/ and trading_app/ (bypassing canonical surfaces)",
        check_crg_cross_layer_surprising_connections,
        True,  # advisory — CRG may be unavailable; never blocks commits
        False,
    ),
    (
        "CRG D2: Canonical-import enforcement -- research scripts must not re-implement canonical functions",
        check_crg_canonical_import_enforcement,
        True,
        False,
    ),
    (
        "CRG D3: Canonical functions must have at least one import-and-call test (AST-based; CRG tests_for graph proven incomplete)",
        check_crg_canonical_functions_have_tests,
        True,
        False,
    ),
    (
        "CRG D4: Canonical-path function size cap (>200 lines = monster-function class)",
        check_crg_canonical_path_function_size,
        True,
        False,
    ),
    (
        "CRG D5: Top-10 bridge nodes (betweenness-centrality chokepoints) must have TESTED_BY edges",
        check_crg_bridge_node_test_coverage,
        True,
        False,
    ),
    (
        "Referenced paths in CLAUDE.md / .claude/rules/ must exist (canonical-rule pointer integrity)",
        check_referenced_paths_in_rules,
        True,  # advisory — refs may transiently break during refactors
        False,
    ),
    (
        "lane_allocation.json lanes[] must pass Chordia gate (verdict + audit freshness)",
        check_lane_allocation_chordia_gate,
        False,  # blocking — capital-class gate, not advisory
        False,
    ),
    (
        "lane_allocation.json lanes[] must pass C8 OOS-status gate (PASSED or NULL grandfather)",
        check_lane_allocation_c8_gate,
        False,  # blocking — capital-class gate, mirrors chordia gate doctrine
        False,
    ),
    (
        "lane_allocation.json displaced[] entries must have valid rejection_gate",
        check_lane_allocation_displaced_bucket,
        False,  # blocking — schema contract for soft-gate provenance bucket
        False,
    ),
    (
        "Literature extracts citing research/output/ carry Mode A/B framing",
        check_literature_extracts_mode_a_b_framing,
        False,  # blocking — composite-N / Mode A/B conflation class bug
        False,
    ),
    (
        "strategy-lab MCP must not re-register deprecated get_recent_fitness endpoint",
        check_strategy_lab_no_fitness_endpoint,
        False,  # blocking — MCP overlap with gold-db.get_strategy_fitness
        False,
    ),
    (
        "intent-router.py routing table matches auto-skill-routing.md documented skills",
        check_intent_router_routing_parity,
        False,  # blocking — silent drift between hook + rule = unreachable routes
        False,
    ),
    (
        "FAST_LANE templates carry doctrine self-labels (v5 deprecated, v5.1 triage+not-validated)",
        check_fast_lane_template_doctrine_fields,
        False,  # blocking — doctrine drift toward treating PROMOTE as deploy is capital-class risk
        False,
    ),
    (
        "FAST_LANE v5.1 preregs route through runner v5.1 branch (sentinel >= 2026-05-20)",
        check_fast_lane_runner_template_routing,
        False,  # blocking — silent bypass of v5.1 gate table is capital-relevant when PROMOTE feeds heavyweight queue
        False,
    ),
    (
        "FAST_LANE PROMOTE queue: no orphan PROMOTEs, no ERROR entries, cache up to date",
        check_fast_lane_promote_orphans,
        False,  # blocking — orphan PROMOTE is the missing-pipe class we just closed
        False,
    ),
    (
        "FAST_LANE promote-queue scanner thresholds match TEMPLATE-fast-lane-v5.1.yaml (canonical-inline-copy parity)",
        check_fast_lane_promote_threshold_parity,
        False,  # blocking — silent drift between scanner constants and v5.1 template would mis-promote heavyweight candidates
        False,
    ),
    (
        "Cherry-pick ranker HEAVYWEIGHT_T_THRESHOLD matches pre_registered_criteria.md Criterion 4 (canonical-inline-copy parity)",
        check_cherry_pick_ranker_threshold_parity,
        False,  # blocking — silent drift would mis-score deflation_headroom and bias which candidates the operator escalates
        False,
    ),
    (
        "Bridge METHODOLOGY_RULES_APPLIED slugs map to real RULE headings in backtesting-methodology.md (canonical-inline-copy parity)",
        check_bridge_methodology_rules_parity,
        False,  # blocking — stale rule citations in generated drafts would propagate into every heavyweight prereg the operator authors
        False,
    ),
    (
        "FAST_LANE pre-flight OOS-power gate constants are present, finite, and grounded in research/oos_power.py POWER_TIERS",
        check_fast_lane_oos_power_gate_constants_grounded,
        False,  # blocking — drift in the gate constants would silently re-promote structurally-unbuildable cells the gate exists to catch
        False,
    ),
    (
        "Canonical-inline-copy registry: every InlineCopyPair has a live parity check + sibling-coverage tests (Layer 2 meta-check)",
        check_canonical_inline_copies_have_parity_check,
        False,  # blocking — an orphan registry entry means a class-pattern fix has silently rotted
        False,
    ),
    (
        "Cherry-pick journal integrity: escalated promote_queue rows have journal entries; entries are append-only monotonic with allowed power tiers",
        check_cherry_pick_journal_integrity,
        False,  # blocking — silent journal gaps invalidate the era_stability_proxy substrate (Improvement 1 / Plan / Stage A)
        False,
    ),
    (
        "Triage-draft provenance: every drafts/*.yaml with a triage_provenance block declares source_validated_setup_strategy_id",
        check_triage_provenance_completeness,
        False,  # blocking — orphan triage drafts break the audit lineage back to canonical validated_setups (Improvement 3 / Stage C)
        False,
    ),
    (
        "AM3.3 audit-log/prereg theory_grant parity: chordia_audit_log.yaml theory_grants must match active prereg metadata.theory_grant (Check #162)",
        check_am33_audit_log_theory_grant_parity,
        False,  # blocking — mismatch silently applies wrong Chordia t-threshold (3.00 vs 3.79) at allocator gate
        False,
    ),
    (
        "Fast-lane state-graph node parity: docs/specs/fast_lane_state_graph.md '## 2. Node Inventory' must match on-disk derived state (Stage 1 of fast-lane connective-tissue plan)",
        check_fast_lane_state_graph_node_parity,
        False,  # blocking — orphan nodes/files mean the canonical chain spec drifted from reality
        False,
    ),
]  # end CHECKS


# Checks measured >0.3s by scripts/tools/profile_check_drift.py (2026-04-19).
# `--fast` mode (used by post-edit hook) skips these for sub-5s real-time coverage.
# Pre-commit hook + CI run the full set — no coverage loss end-to-end.
SLOW_CHECK_LABELS = frozenset(
    {
        "All imports resolve",
        "Generated task views preserve strict truth-class boundaries",
        "ENTRY_MODELS sync",
        "Phase 4 discovery SHA integrity (stamped hypothesis_file_sha must reference real file)",
        "Canonical Claude client source (claude_client.py is the only place for Claude model IDs + anthropic.Anthropic)",
        "SQL column convention: pipeline tables use 'symbol', trading app tables use 'instrument'",
        "Timezone hygiene",
        "Canonical orb_utc_window source (only pipeline/dst.py may define it)",
        "validated_setups writes stay on canonical allowlist",
        "No hardcoded scratch DB defaults in active code",
        "family_rr_locks JOIN key completeness (6-column key in every JOIN)",
        "No old session names in active code",
        "Trading app hardcoded paths",
        "No deprecated C:/db/gold.db in docstring usage examples",
        "No raw orb_active reads outside asset_configs.py",
        "@canonical-source annotations point to existing files",
        "Naive datetime detection",
        "No old fixed-clock session names in Python source",
        "Trading app schema-query consistency",
        "No CRLF in tracked text blobs (defense-in-depth for pre-commit [0b] auto-renormalize)",
        # CRG D1/D2/D3/D5 exceed 0.3s — measured 2026-04-29: D1=0.98s, D2=1.20s,
        # D3=0.49s, D5=9.76s. D2/D3 are AST-only (no graph DB traversal) but the
        # tree-walks over research/ and tests/ still cross the threshold. D4
        # ran <0.3s and stays in the fast path. Pre-commit and CI run the full
        # set; only the post-edit hook's --fast path skips.
        "CRG D1: Surprising cross-layer connections between pipeline/ and trading_app/ (bypassing canonical surfaces)",
        "CRG D2: Canonical-import enforcement -- research scripts must not re-implement canonical functions",
        "CRG D3: Canonical functions must have at least one import-and-call test (AST-based; CRG tests_for graph proven incomplete)",
        "CRG D5: Top-10 bridge nodes (betweenness-centrality chokepoints) must have TESTED_BY edges",
    }
)


def _assert_slow_labels_valid() -> None:
    """Fail closed when the fast-skip registry drifts from the canonical checks.

    Without this guard, renaming a CHECKS label silently removes it from the
    ``--fast`` skip set. Fast mode then starts running the slow check on every
    edit hook invocation, which can exceed the 30s hook timeout and degrade
    coverage without an obvious failure.
    """
    known_labels = {label for label, *_ in CHECKS}
    stale = SLOW_CHECK_LABELS - known_labels
    if stale:
        raise RuntimeError(
            "SLOW_CHECK_LABELS references label(s) not present in CHECKS: "
            f"{sorted(stale)}. Either the check was renamed/removed without "
            "updating SLOW_CHECK_LABELS, or the label string has a typo. "
            "Re-run scripts/tools/profile_check_drift.py and update the set."
        )


_assert_slow_labels_valid()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline drift detection")
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Skip slow checks (>0.3s) — used by post-edit hook for real-time coverage. "
            "Pre-commit and CI run the full set so coverage is preserved end-to-end."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Sanitized output for LLM consumption: emit only PASS / FAIL: <name> (count=N) "
            "lines per check, plus a single summary line. Suppresses banner, file paths, "
            "SQL fragments, DB internals, and any per-check diagnostic detail. "
            "Exit code semantics are unchanged (0 = clean, 1 = drift detected)."
        ),
    )
    args = parser.parse_args()
    fast_mode = args.fast
    quiet_mode = args.quiet

    if not quiet_mode:
        print("=" * 60)
        if fast_mode:
            print(f"PIPELINE DRIFT CHECK (FAST — skipping {len(SLOW_CHECK_LABELS)} slow checks)")
        else:
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
            if not quiet_mode:
                print(f"  WARNING: could not open DB ({exc}) — DB-dependent checks will skip")

    fast_skipped = 0  # tracks --fast skips separately from DB-unavailable skips
    for i, (label, check_fn, is_advisory, requires_db) in enumerate(CHECKS, 1):
        if fast_mode and label in SLOW_CHECK_LABELS:
            fast_skipped += 1
            continue
        if not quiet_mode:
            print(f"Check {i}: {label}...")

        # DB-dependent checks can be skipped if DB unavailable.
        # duckdb.IOException is NOT a subclass of OSError, so we inspect
        # the message to distinguish "DB busy" from real code failures.
        # In --quiet mode, redirect stdout for the duration of the check call
        # so per-check inline prints (advisory WARNINGs, doc-stat dumps, etc.)
        # never leak to the sanitized output stream. _QuietSink mimics the
        # subset of TextIOBase that imported modules call at import-time
        # (e.g. trading_app.outcome_builder calls sys.stdout.reconfigure()).
        suppress_ctx = contextlib.redirect_stdout(_QuietSink()) if quiet_mode else contextlib.nullcontext()
        if requires_db:
            try:
                with suppress_ctx:
                    v = check_fn(con=_shared_con)
            except Exception as e:
                msg = str(e)
                if "being used by another process" in msg or "Cannot open file" in msg:
                    skip_count += 1
                    if quiet_mode:
                        print(f"SKIP: {_safe_label_for_quiet(label)}")
                    else:
                        print("  SKIPPED (DB busy — another process holds the lock)")
                        print()
                    continue
                v = [f"  EXCEPTION: {type(e).__name__}: {e}"]
        else:
            with suppress_ctx:
                v = check_fn()

        if is_advisory:
            advisory_count += 1
            if quiet_mode:
                print(f"ADVISORY: {_safe_label_for_quiet(label)}")
            else:
                # Advisory checks print their own warnings; show ADVISORY tag
                print("  ADVISORY (non-blocking)")
        elif v:
            if quiet_mode:
                # Sanitized — count only, no per-violation detail.
                print(f"FAIL: {_safe_label_for_quiet(label)} (count={len(v)})")
            else:
                print("  FAILED:")
                for line in v:
                    print(line)
            all_violations.extend(v)
        else:
            blocking_count += 1
            if quiet_mode:
                print(f"PASS: {_safe_label_for_quiet(label)}")
            else:
                print("  PASSED [OK]")
        if not quiet_mode:
            print()

    # Cleanup shared connection
    if _shared_con is not None:
        _shared_con.close()

    # Summary — blocking_count tracks actual passes (not computed from total)
    fast_part = f", {fast_skipped} skipped (--fast)" if fast_skipped else ""
    summary_line = (
        f"{blocking_count} checks passed [OK], "
        f"{skip_count} skipped (DB unavailable){fast_part}, "
        f"{advisory_count} advisory"
    )
    if all_violations:
        if quiet_mode:
            print(f"SUMMARY: drift_detected violations={len(all_violations)} passed={blocking_count}")
        else:
            print("=" * 60)
            print(f"DRIFT DETECTED: {len(all_violations)} violation(s) across {summary_line}")
            print("=" * 60)
        sys.exit(1)
    else:
        if quiet_mode:
            print(f"SUMMARY: clean passed={blocking_count} advisory={advisory_count}")
        else:
            print("=" * 60)
            print(f"NO DRIFT DETECTED: {summary_line}")
            print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
