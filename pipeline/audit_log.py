"""
Immutable audit log for pipeline operations.

Append-only table tracking every write operation: what changed, when, how many rows,
by which rebuild. Provides forensic capability for answering "what happened to my data?"

Usage:
    from pipeline.audit_log import log_operation, get_previous_counts

    log_id = log_operation(con, "OUTCOME_BUILDER", "orb_outcomes",
                           instrument="MGC", rows_before=1000, rows_after=1050,
                           duration_s=12.3, rebuild_id="abc123")
"""

import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from pipeline.init_db import PIPELINE_AUDIT_LOG_SCHEMA

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_audit_table(con: duckdb.DuckDBPyConnection) -> None:
    """Create pipeline_audit_log if it doesn't exist (idempotent)."""
    con.execute(PIPELINE_AUDIT_LOG_SCHEMA)


# ---------------------------------------------------------------------------
# Git SHA helper
# ---------------------------------------------------------------------------


_cached_git_sha: str | None = None
_git_sha_resolved: bool = False


def get_git_sha() -> str | None:
    """Return current HEAD commit SHA, or None if not in a git repo.

    Result is cached for the lifetime of the process — git SHA doesn't change
    during a single pipeline run. Also avoids interfering with subprocess mocks
    in tests.
    """
    global _cached_git_sha, _git_sha_resolved
    if _git_sha_resolved:
        return _cached_git_sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0 and hasattr(result, "stdout") and result.stdout:
            _cached_git_sha = result.stdout.strip()[:12]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    _git_sha_resolved = True
    return _cached_git_sha


# ---------------------------------------------------------------------------
# Core API — append-only
# ---------------------------------------------------------------------------


def log_operation(
    con: duckdb.DuckDBPyConnection,
    operation: str,
    table_name: str,
    *,
    instrument: str | None = None,
    date_start=None,
    date_end=None,
    rows_before: int | None = None,
    rows_after: int | None = None,
    duration_s: float | None = None,
    git_sha: str | None = None,
    rebuild_id: str | None = None,
    status: str = "SUCCESS",
) -> str:
    """Append one audit log row. Returns log_id.

    This function ONLY does INSERT. No UPDATE, no DELETE. Ever.
    If the insert fails, logs to stderr and returns empty string (audit failure
    should never block data writes).
    """

    log_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    if git_sha is None:
        git_sha = get_git_sha()

    try:
        ensure_audit_table(con)
        con.execute(
            """
            INSERT INTO pipeline_audit_log
                (log_id, timestamp, operation, table_name, instrument,
                 date_start, date_end, rows_before, rows_after,
                 duration_s, git_sha, rebuild_id, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            """,
            [
                log_id,
                now,
                operation,
                table_name,
                instrument,
                date_start,
                date_end,
                rows_before,
                rows_after,
                duration_s,
                git_sha,
                rebuild_id,
                status,
            ],
        )
        con.commit()
    except Exception as e:
        print(f"[AUDIT] WARNING: failed to log operation: {e}", file=sys.stderr)
        return ""

    return log_id


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_previous_counts(
    con: duckdb.DuckDBPyConnection,
    instrument: str,
    table_name: str,
) -> int | None:
    """Return rows_after from the most recent SUCCESS log for this instrument+table.

    Returns None if no prior log entry exists.
    """
    try:
        ensure_audit_table(con)
        row = con.execute(
            """
            SELECT rows_after FROM pipeline_audit_log
            WHERE instrument = $1 AND table_name = $2 AND status = 'SUCCESS'
                  AND rows_after IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            [instrument, table_name],
        ).fetchone()
        return row[0] if row else None
    except Exception as e:
        print(f"[AUDIT] WARNING: get_previous_counts failed: {e}", file=sys.stderr)
        return None


# Allowlist of tables safe to query in get_table_row_count.
# Must match tables defined in pipeline/init_db.py.
_ALLOWED_TABLES: frozenset[str] = frozenset(
    {
        "bars_1m",
        "bars_5m",
        "daily_features",
        "orb_outcomes",
        "experimental_strategies",
        "validated_setups",
        "edge_families",
        "family_rr_locks",
        "prospective_signals",
        "pipeline_audit_log",
        "rebuild_manifest",
    }
)

# Tables that use 'instrument' instead of 'symbol' for filtering
_INSTRUMENT_COLUMN_TABLES: frozenset[str] = frozenset(
    {
        "experimental_strategies",
        "validated_setups",
        "edge_families",
        "family_rr_locks",
    }
)


def get_table_row_count(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    instrument: str | None = None,
) -> int:
    """Count rows in a table, optionally filtered by instrument.

    Uses the correct column name per table: 'symbol' for pipeline tables,
    'instrument' for trading_app tables. Table name is validated against an
    allowlist to prevent SQL injection.
    """
    if table_name not in _ALLOWED_TABLES:
        raise ValueError(f"Table '{table_name}' not in allowlist: {sorted(_ALLOWED_TABLES)}")

    if instrument is None:
        row = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    else:
        col = "instrument" if table_name in _INSTRUMENT_COLUMN_TABLES else "symbol"
        row = con.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} = $1", [instrument]).fetchone()

    return row[0] if row else 0
