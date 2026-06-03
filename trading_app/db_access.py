"""Read-safe gold.db access helpers for MCP and remote-agent snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from pipeline.db_config import configure_connection
from pipeline.paths import CANONICAL_RUNTIME_ROOT, GOLD_DB_PATH

APPROVED_SNAPSHOT_TABLES: tuple[str, ...] = (
    "daily_features",
    "orb_outcomes",
    "validated_setups",
    "edge_families",
)
DEFAULT_SNAPSHOT_ROOT = CANONICAL_RUNTIME_ROOT / "data" / "snapshots" / "gold_db"
DEPRECATED_SCRATCH_DB = Path("C:/db/gold.db")
MANIFEST_VERSION = 1
DEFAULT_MAX_DB_AGE_HOURS = 168

_FRESHNESS_COLUMNS: dict[str, tuple[str, ...]] = {
    "daily_features": ("trading_day",),
    "orb_outcomes": ("trading_day", "entry_ts", "exit_ts"),
    "validated_setups": ("updated_at", "created_at", "validation_date"),
    "edge_families": ("updated_at", "created_at"),
}


def _iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC).isoformat()
        return value.astimezone(UTC).isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _safe_resolve(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_deprecated_scratch_db(path: Path) -> bool:
    return _safe_resolve(path) == _safe_resolve(DEPRECATED_SCRATCH_DB)


def _path_source(db_path: Path) -> dict[str, Any]:
    env_value = os.environ.get("DUCKDB_PATH")
    if not env_value:
        return {"source": "canonical", "duckdb_path_env": None}

    env_path = Path(env_value).expanduser()
    if _is_deprecated_scratch_db(env_path):
        return {
            "source": "canonical_fallback",
            "duckdb_path_env": env_value,
            "warning": "DUCKDB_PATH points to deprecated scratch DB and must be ignored by pipeline.paths",
        }
    if _safe_resolve(env_path) == _safe_resolve(db_path):
        return {"source": "DUCKDB_PATH", "duckdb_path_env": env_value}
    return {
        "source": "canonical_fallback",
        "duckdb_path_env": env_value,
        "warning": "DUCKDB_PATH is set but did not resolve to the active DB path",
    }


def db_access_policy() -> dict[str, Any]:
    """Return the gold-db MCP access policy agents should assume."""
    return {
        "default_transport": "stdio",
        "http_enabled": False,
        "http_bind": None,
        "write_enabled": False,
        "raw_sql_writes_enabled": False,
        "database_switching_enabled": False,
        "github_live_db_access": "forbidden",
        "snapshot_access": "approved_manifest_only",
        "write_broker": "future_named_jobs_only",
        "notes": [
            "gold-db MCP is a local read-only agent surface.",
            "GitHub and remote jobs consume stamped snapshots, not live gold.db.",
            "Future writes must use a single-writer named-job broker, never raw SQL.",
        ],
    }


@contextmanager
def _read_only_connection(db_path: Path):
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        configure_connection(con)
        yield con
    finally:
        con.close()


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
        """,
        [table],
    ).fetchone()
    return bool(row and row[0])


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = ?
        """,
        [table],
    ).fetchall()
    return {str(row[0]) for row in rows}


def _table_freshness(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, Any]:
    if not _table_exists(con, table):
        return {"exists": False, "row_count": 0}

    row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    payload: dict[str, Any] = {"exists": True, "row_count": int(row_count)}
    cols = _columns(con, table)
    for column in _FRESHNESS_COLUMNS.get(table, ()):
        if column not in cols:
            continue
        value = con.execute(f"SELECT MAX({column}) FROM {table}").fetchone()[0]
        payload[f"max_{column}"] = _json_scalar(value)
    return payload


def db_freshness(db_path: Path | None = None) -> dict[str, Any]:
    """Return row-count and horizon information for approved DB tables."""
    path = Path(db_path) if db_path is not None else GOLD_DB_PATH
    if not path.exists():
        return {
            "status": "MISSING",
            "db_path": str(path),
            "tables": {table: {"exists": False, "row_count": 0} for table in APPROVED_SNAPSHOT_TABLES},
            "error": "gold.db missing",
        }

    try:
        with _read_only_connection(path) as con:
            tables = {table: _table_freshness(con, table) for table in APPROVED_SNAPSHOT_TABLES}
    except Exception as exc:
        return {
            "status": "ERROR",
            "db_path": str(path),
            "tables": {},
            "error": str(exc),
        }

    return {"status": "OK", "db_path": str(path), "tables": tables}


def db_health(db_path: Path | None = None) -> dict[str, Any]:
    """Return fail-closed health and access information for gold.db."""
    path = Path(db_path) if db_path is not None else GOLD_DB_PATH
    payload: dict[str, Any] = {
        "db_path": str(path),
        "path_source": _path_source(path),
        "access": db_access_policy(),
        "exists": path.exists(),
        "read_only_open_ok": False,
        "open_error": None,
        "size_bytes": None,
        "mtime_utc": None,
        "horizon": {},
    }

    if _is_deprecated_scratch_db(path):
        payload.update(
            {
                "status": "BLOCKED_DEPRECATED_SCRATCH_DB",
                "open_error": "C:/db/gold.db is deprecated and blocked",
            }
        )
        return payload

    if not path.exists():
        payload.update({"status": "MISSING", "open_error": "gold.db missing"})
        return payload

    stat = path.stat()
    payload["size_bytes"] = stat.st_size
    payload["mtime_utc"] = _iso_utc(stat.st_mtime)

    try:
        with _read_only_connection(path) as con:
            payload["read_only_open_ok"] = True
            payload["horizon"] = {table: _table_freshness(con, table) for table in APPROVED_SNAPSHOT_TABLES}
    except Exception as exc:
        payload.update({"status": "ERROR", "open_error": str(exc)})
        return payload

    payload["status"] = "OK"
    return payload


def validate_snapshot_manifest(manifest: dict[str, Any]) -> list[str]:
    required_top = {
        "manifest_version",
        "snapshot_id",
        "generated_at_utc",
        "source_db",
        "tables",
        "horizon",
    }
    errors = [f"missing {key}" for key in sorted(required_top - set(manifest))]
    source = manifest.get("source_db")
    if not isinstance(source, dict):
        errors.append("source_db must be an object")
    else:
        for key in ("path", "mtime_utc"):
            if not source.get(key):
                errors.append(f"source_db missing {key}")
    if not isinstance(manifest.get("tables"), dict):
        errors.append("tables must be an object")
    if not isinstance(manifest.get("horizon"), dict):
        errors.append("horizon must be an object")
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        errors.append(f"manifest_version must be {MANIFEST_VERSION}")
    return errors


def snapshot_manifest(snapshot_root: Path | None = None) -> dict[str, Any]:
    root = Path(snapshot_root) if snapshot_root is not None else DEFAULT_SNAPSHOT_ROOT
    if not root.exists():
        return {"status": "MISSING", "snapshot_root": str(root), "snapshots": [], "errors": []}

    snapshots: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue

        validation_errors = validate_snapshot_manifest(manifest)
        if validation_errors:
            errors.append(
                {
                    "path": str(path),
                    "snapshot_id": manifest.get("snapshot_id"),
                    "errors": validation_errors,
                }
            )
            continue
        snapshots.append(manifest)

    status = "OK"
    if errors and snapshots:
        status = "OK_WITH_ERRORS"
    elif errors:
        status = "ERROR"
    return {"status": status, "snapshot_root": str(root), "snapshots": snapshots, "errors": errors}


def _assert_output_under_root(output_dir: Path, snapshot_root: Path) -> None:
    resolved_output = _safe_resolve(output_dir)
    resolved_root = _safe_resolve(snapshot_root)
    if resolved_output != resolved_root and resolved_root not in resolved_output.parents:
        raise ValueError(f"output_dir must be under approved snapshot root: {resolved_root}")


def _assert_db_not_stale(path: Path, now: datetime, max_age_hours: int | None) -> None:
    if max_age_hours is None:
        return
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    age_hours = (now - mtime).total_seconds() / 3600
    if age_hours > max_age_hours:
        raise ValueError(
            f"source gold.db is stale: age_hours={age_hours:.1f}, "
            f"max_age_hours={max_age_hours}, mtime_utc={mtime.isoformat()}"
        )


def _db_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_snapshot(
    *,
    db_path: Path | None = None,
    output_dir: Path | None = None,
    snapshot_root: Path | None = None,
    tables: list[str] | None = None,
    now: datetime | None = None,
    max_age_hours: int | None = DEFAULT_MAX_DB_AGE_HOURS,
) -> dict[str, Any]:
    """Export approved read-only tables to Parquet with a stamped manifest."""
    source = Path(db_path) if db_path is not None else GOLD_DB_PATH
    root = Path(snapshot_root) if snapshot_root is not None else DEFAULT_SNAPSHOT_ROOT
    generated_at = now.astimezone(UTC) if now is not None else datetime.now(UTC)
    destination = Path(output_dir) if output_dir is not None else root / generated_at.strftime("%Y%m%dT%H%M%SZ")
    selected_tables = list(tables) if tables is not None else list(APPROVED_SNAPSHOT_TABLES)

    if _is_deprecated_scratch_db(source):
        raise ValueError("C:/db/gold.db is a deprecated scratch DB and cannot be snapshotted")
    if not source.exists():
        raise FileNotFoundError(f"gold.db missing: {source}")
    _assert_output_under_root(destination, root)
    _assert_db_not_stale(source, generated_at, max_age_hours)

    unapproved = sorted(set(selected_tables) - set(APPROVED_SNAPSHOT_TABLES))
    if unapproved:
        raise ValueError(f"tables not approved for snapshot export: {unapproved}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    table_payload: dict[str, Any] = {}
    horizon_payload: dict[str, Any] = {}
    with _read_only_connection(source) as con:
        for table in selected_tables:
            freshness = _table_freshness(con, table)
            horizon_payload[table] = freshness
            row_count = int(freshness.get("row_count", 0))
            table_payload[table] = {"row_count": row_count, "format": "parquet"}
            if not freshness.get("exists") or row_count == 0:
                table_payload[table]["exported"] = False
                continue

            table_dir = destination / table
            table_dir.mkdir(parents=True, exist_ok=True)
            out_file = table_dir / f"{table}.parquet"
            escaped = str(out_file).replace("'", "''")
            con.execute(f"COPY {table} TO '{escaped}' (FORMAT PARQUET)")
            table_payload[table]["exported"] = True
            table_payload[table]["path"] = str(out_file)

    stat = source.stat()
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "snapshot_id": destination.name,
        "generated_at_utc": generated_at.isoformat(),
        "source_db": {
            "path": str(source),
            "size_bytes": stat.st_size,
            "mtime_utc": _iso_utc(stat.st_mtime),
            "sha256": _db_fingerprint(source),
        },
        "access_policy": db_access_policy(),
        "tables": table_payload,
        "horizon": horizon_payload,
    }
    validation_errors = validate_snapshot_manifest(manifest)
    if validation_errors:
        raise ValueError(f"internal manifest validation failed: {validation_errors}")

    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
