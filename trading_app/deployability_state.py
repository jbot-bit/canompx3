"""Append-only deployment readiness state derived from deployability audits.

This table is not a research source of truth. It materializes the latest
deployment-readiness evaluation so rebuilds and operators can fail closed
without mutating ``validated_setups`` or restamping research evidence.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEPLOYMENT_READINESS_TABLE = "deployment_readiness_evaluations"

DEPLOYMENT_READINESS_EVALUATIONS_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {DEPLOYMENT_READINESS_TABLE} (
    evaluation_id TEXT PRIMARY KEY,
    generated_at TIMESTAMPTZ NOT NULL,
    rebuild_id TEXT,
    git_sha TEXT,
    scope TEXT NOT NULL,
    profile_id TEXT,
    strategy_id TEXT NOT NULL,
    instrument TEXT,
    verdict TEXT NOT NULL,
    deployable BOOLEAN NOT NULL,
    institutional_language_allowed BOOLEAN NOT NULL,
    hard_issue_ids TEXT NOT NULL,
    warning_issue_ids TEXT NOT NULL,
    info_issue_ids TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL
)
"""


def ensure_deployment_readiness_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create deployment readiness state table if missing."""

    con.execute(DEPLOYMENT_READINESS_EVALUATIONS_SCHEMA)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _issue_ids(row: dict[str, Any], severity: str) -> list[str]:
    return [
        str(issue["id"])
        for issue in row.get("issues", [])
        if issue.get("severity") == severity and issue.get("id") is not None
    ]


def _validate_report_shape(report: dict[str, Any]) -> list[dict[str, Any]]:
    scope = report.get("scope")
    if scope not in {"all-active", "profile"}:
        raise ValueError(f"deployability report has invalid scope: {scope!r}")

    strategies = report.get("strategies")
    if not isinstance(strategies, list):
        raise ValueError("deployability report missing strategies list")

    required_strategy_fields = {
        "strategy_id",
        "verdict",
        "deployable",
        "institutional_language_allowed",
    }
    for idx, strategy in enumerate(strategies, start=1):
        if not isinstance(strategy, dict):
            raise ValueError(f"deployability strategy row {idx} is not an object")
        missing = [field for field in required_strategy_fields if field not in strategy]
        if missing:
            raise ValueError(f"deployability strategy row {idx} missing fields: {missing}")
        if strategy.get("strategy_id") in (None, ""):
            raise ValueError(f"deployability strategy row {idx} missing strategy_id")
        if strategy.get("verdict") in (None, ""):
            raise ValueError(f"deployability strategy row {idx} missing verdict")
        if not isinstance(strategy.get("deployable"), bool):
            raise ValueError(f"deployability strategy row {idx} has non-boolean deployable")
        if not isinstance(strategy.get("institutional_language_allowed"), bool):
            raise ValueError(f"deployability strategy row {idx} has non-boolean institutional_language_allowed")
    return strategies


def write_deployability_state(
    report: dict[str, Any],
    *,
    db_path: Path | str = GOLD_DB_PATH,
    rebuild_id: str | None = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Append strategy-level deployability audit rows and return write metadata."""

    strategies = _validate_report_shape(report)
    now = datetime.now(UTC)
    resolved_git_sha = git_sha if git_sha is not None else _git_sha()
    provenance = {
        "report_generated_at": report.get("generated_at"),
        "db_path": report.get("db_path"),
        "source_truth": report.get("source_truth"),
        "resource_lit": report.get("resource_lit"),
        "summary": report.get("summary"),
        "no_double_calculation": (
            "Deployment readiness stores derived audit output only; research metrics remain owned by "
            "validator/family/replay/account/runtime canonical surfaces."
        ),
    }

    rows = []
    for idx, strategy in enumerate(strategies, start=1):
        evidence = {
            "replay": strategy.get("replay"),
            "current_k_fdr": strategy.get("current_k_fdr"),
            "c8_oos": strategy.get("c8_oos"),
            "trade_context": strategy.get("trade_context"),
            "runtime_control": strategy.get("runtime_control"),
            "metrics": strategy.get("metrics"),
            "issues": strategy.get("issues"),
            "account_state": report.get("account_state"),
        }
        evaluation_id = f"dre-{now.strftime('%Y%m%dT%H%M%S%f')}-{idx:04d}-{uuid.uuid4().hex[:10]}"
        rows.append(
            [
                evaluation_id,
                now,
                rebuild_id,
                resolved_git_sha,
                report.get("scope"),
                report.get("profile_id"),
                strategy.get("strategy_id"),
                strategy.get("instrument"),
                strategy.get("verdict"),
                bool(strategy.get("deployable")),
                bool(strategy.get("institutional_language_allowed")),
                _json_dumps(_issue_ids(strategy, "hard")),
                _json_dumps(_issue_ids(strategy, "warning")),
                _json_dumps(_issue_ids(strategy, "info")),
                _json_dumps(evidence),
                _json_dumps(provenance),
            ]
        )

    with duckdb.connect(str(db_path)) as con:
        configure_connection(con, writing=True)
        ensure_deployment_readiness_schema(con)
        if rows:
            con.executemany(
                f"""
                INSERT INTO {DEPLOYMENT_READINESS_TABLE}
                    (evaluation_id, generated_at, rebuild_id, git_sha, scope, profile_id,
                     strategy_id, instrument, verdict, deployable,
                     institutional_language_allowed, hard_issue_ids, warning_issue_ids,
                     info_issue_ids, evidence_json, provenance_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        con.commit()

    return {
        "table": DEPLOYMENT_READINESS_TABLE,
        "rows_written": len(rows),
        "rebuild_id": rebuild_id,
        "git_sha": resolved_git_sha,
    }


def load_latest_deployment_readiness(
    *,
    db_path: Path | str = GOLD_DB_PATH,
    profile_id: str | None = None,
    scope: str | None = None,
    strategy_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return latest append-only readiness row per strategy/profile/scope."""

    with duckdb.connect(str(db_path), read_only=True) as con:
        table = con.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            LIMIT 1
            """,
            [DEPLOYMENT_READINESS_TABLE],
        ).fetchone()
        if table is None:
            return []

        predicates: list[str] = []
        params: list[Any] = []
        if profile_id is not None:
            predicates.append("profile_id = ?")
            params.append(profile_id)
        if scope is not None:
            predicates.append("scope = ?")
            params.append(scope)
        if strategy_ids:
            placeholders = ", ".join("?" for _ in strategy_ids)
            predicates.append(f"strategy_id IN ({placeholders})")
            params.extend(sorted(strategy_ids))

        where_sql = f"WHERE {' AND '.join(predicates)}" if predicates else ""
        rows = con.execute(
            f"""
            SELECT evaluation_id, generated_at, rebuild_id, git_sha, scope, profile_id,
                   strategy_id, instrument, verdict, deployable,
                   institutional_language_allowed, hard_issue_ids, warning_issue_ids,
                   info_issue_ids, evidence_json, provenance_json
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY strategy_id, COALESCE(profile_id, ''), scope
                           ORDER BY generated_at DESC, evaluation_id DESC
                       ) AS rn
                FROM {DEPLOYMENT_READINESS_TABLE}
                {where_sql}
            )
            WHERE rn = 1
            ORDER BY instrument, strategy_id
            """,
            params,
        ).fetchall()
        cols = [desc[0] for desc in con.description]
    return [dict(zip(cols, row, strict=False)) for row in rows]
