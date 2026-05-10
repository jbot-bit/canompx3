#!/usr/bin/env python3
"""Read-only adversarial gate before strategy routing or allocator expansion.

This tool deliberately separates research shelf size from deployable capacity.
It combines existing repo checks with direct DB evidence and fails closed when
the evidence chain is incomplete.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS  # noqa: E402
from pipeline.paths import GOLD_DB_PATH  # noqa: E402
from scripts.tools.live_readiness_report import build_live_readiness_report  # noqa: E402
from trading_app.deployability import build_deployability_audit  # noqa: E402

GO = "GO"
BLOCKED = "BLOCKED"
NO_GO = "NO_GO"
TOOL_ERROR = "tool_error"

EXIT_CODES = {GO: 0, NO_GO: 1, BLOCKED: 2}


@dataclass(frozen=True)
class CommandResult:
    name: str
    argv: list[str]
    returncode: int | None
    stdout_tail: str
    stderr_tail: str
    timed_out: bool = False


CommandRunner = Callable[[str, list[str], int], CommandResult]
DeployabilityRunner = Callable[..., dict[str, Any]]


def _tail(text: str, *, max_chars: int = 4000) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def run_command(name: str, argv: list[str], timeout_seconds: int) -> CommandResult:
    try:
        result = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            name=name,
            argv=argv,
            returncode=None,
            stdout_tail=_tail(exc.stdout or ""),
            stderr_tail=_tail(exc.stderr or ""),
            timed_out=True,
        )
    return CommandResult(
        name=name,
        argv=argv,
        returncode=result.returncode,
        stdout_tail=_tail(result.stdout),
        stderr_tail=_tail(result.stderr),
    )


def _connect_ro(db_path: Path):
    return duckdb.connect(str(db_path), read_only=True)


def _has_table(con, table_name: str) -> bool:
    return (
        con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [table_name],
        ).fetchone()[0]
        > 0
    )


def _active_in_sql() -> str:
    return ", ".join(f"'{inst}'" for inst in ACTIVE_ORB_INSTRUMENTS)


def load_counts(con) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    counts["raw_validated"] = con.execute(
        "SELECT COUNT(*) FROM validated_setups WHERE LOWER(status) = 'active'"
    ).fetchone()[0]
    if _has_table(con, "deployable_validated_setups"):
        counts["deployable_rows"] = con.execute("SELECT COUNT(*) FROM deployable_validated_setups").fetchone()[0]
    else:
        counts["deployable_rows"] = con.execute(
            """
            SELECT COUNT(*) FROM validated_setups
            WHERE LOWER(status) = 'active'
              AND LOWER(COALESCE(deployment_scope, 'deployable')) = 'deployable'
            """
        ).fetchone()[0]

    counts["unique_streams"] = con.execute(
        """
        SELECT COUNT(DISTINCT e.trade_day_hash)
        FROM validated_setups vs
        JOIN experimental_strategies e USING (strategy_id)
        WHERE LOWER(vs.status) = 'active' AND e.trade_day_hash IS NOT NULL
        """
    ).fetchone()[0]
    counts["edge_families"] = con.execute("SELECT COUNT(*) FROM edge_families").fetchone()[0]
    counts["edge_families_non_purged"] = con.execute(
        """
        SELECT COUNT(*) FROM edge_families
        WHERE COALESCE(robustness_status, '') <> 'PURGED'
        """
    ).fetchone()[0]
    counts["rr_locked_non_purged"] = con.execute(
        """
        SELECT COUNT(*)
        FROM validated_setups vs
        JOIN family_rr_locks frl
          ON frl.instrument = vs.instrument
         AND frl.orb_label = vs.orb_label
         AND frl.filter_type = vs.filter_type
         AND frl.entry_model = vs.entry_model
         AND frl.orb_minutes = vs.orb_minutes
         AND frl.confirm_bars = vs.confirm_bars
         AND frl.locked_rr = vs.rr_target
        LEFT JOIN edge_families ef ON ef.family_hash = vs.family_hash
        WHERE LOWER(vs.status) = 'active'
          AND COALESCE(ef.robustness_status, '') <> 'PURGED'
        """
    ).fetchone()[0]
    counts["by_instrument"] = [
        {"instrument": row[0], "raw_validated": row[1]}
        for row in con.execute(
            """
            SELECT instrument, COUNT(*)
            FROM validated_setups
            WHERE LOWER(status) = 'active'
            GROUP BY instrument
            ORDER BY instrument
            """
        ).fetchall()
    ]
    return counts


def _load_strategy_rows(con, strategy_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not strategy_ids:
        return {}
    placeholders = ", ".join("?" for _ in strategy_ids)
    rows = con.execute(
        f"""
        SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.orb_minutes,
               vs.entry_model, vs.rr_target, vs.confirm_bars, vs.filter_type,
               LOWER(vs.status) AS status,
               LOWER(COALESCE(vs.deployment_scope, 'deployable')) AS deployment_scope,
               vs.sample_size, vs.years_tested, vs.expectancy_r, vs.oos_exp_r,
               vs.wfe, vs.dsr_score, vs.fdr_adjusted_p, vs.discovery_k,
               vs.slippage_validation_status, vs.c8_oos_status,
               ef.robustness_status, ef.trade_tier, ef.pbo
        FROM validated_setups vs
        LEFT JOIN edge_families ef ON ef.family_hash = vs.family_hash
        WHERE vs.strategy_id IN ({placeholders})
        """,
        strategy_ids,
    ).fetchall()
    cols = [d[0] for d in con.description]
    return {str(row[0]): dict(zip(cols, row, strict=False)) for row in rows}


def _bh_adjusted_for_session(con, orb_label: str) -> dict[str, float]:
    rows = con.execute(
        f"""
        SELECT strategy_id, p_value
        FROM experimental_strategies
        WHERE is_canonical = TRUE
          AND orb_label = ?
          AND p_value IS NOT NULL
          AND instrument IN ({_active_in_sql()})
        ORDER BY p_value
        """,
        [orb_label],
    ).fetchall()
    if not rows:
        return {}
    m = len(rows)
    adjusted = [0.0] * len(rows)
    for idx in range(len(rows) - 1, -1, -1):
        rank = idx + 1
        raw = min(float(rows[idx][1]) * m / rank, 1.0)
        if idx < len(rows) - 1:
            raw = min(raw, adjusted[idx + 1])
        adjusted[idx] = raw
    return {str(row[0]): adj for row, adj in zip(rows, adjusted, strict=False)}


def current_k_fdr(con, strategy_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not strategy_ids:
        return {}
    rows = _load_strategy_rows(con, strategy_ids)
    by_session = sorted({row["orb_label"] for row in rows.values() if row.get("orb_label")})
    adjusted_by_session = {session: _bh_adjusted_for_session(con, session) for session in by_session}
    out: dict[str, dict[str, Any]] = {}
    for sid, row in rows.items():
        current_adj = adjusted_by_session.get(row["orb_label"], {}).get(sid)
        out[sid] = {
            "stored_adj_p": row.get("fdr_adjusted_p"),
            "stored_k": row.get("discovery_k"),
            "current_adj_p": current_adj,
            "current_pass": None if current_adj is None else current_adj < 0.05,
        }
    return out


def _active_lane_ids(live_report: dict[str, Any]) -> list[str]:
    return [str(lane.get("strategy_id")) for lane in live_report.get("active_lanes", []) if lane.get("strategy_id")]


def _classify_commands(
    commands: list[CommandResult], hard_blockers: list[dict[str, Any]], edge_cases: list[dict[str, Any]]
):
    for result in commands:
        if result.timed_out:
            hard_blockers.append(
                {
                    "id": "tool_timeout",
                    "severity": BLOCKED,
                    "detail": f"{result.name} timed out",
                }
            )
            continue
        if result.returncode in (0, None):
            continue
        if result.name == "check_drift":
            hard_blockers.append(
                {
                    "id": "drift_failure",
                    "severity": NO_GO,
                    "detail": "pipeline/check_drift.py exited non-zero",
                }
            )
        elif result.name == "fdr_integrity" and result.returncode == 1:
            hard_blockers.append(
                {
                    "id": "fdr_integrity_failure",
                    "severity": NO_GO,
                    "detail": "FDR audit reported correctness failure",
                }
            )
        elif result.name == "fdr_integrity":
            edge_cases.append(
                {
                    "id": "fdr_integrity_warn",
                    "detail": "FDR audit reported warnings; inspect command output",
                }
            )
        elif result.name == "chain_integrity":
            combined_tail = f"{result.stdout_tail}\n{result.stderr_tail}"
            if "OVERALL: SUSPECT" in combined_tail and "critical=0" in combined_tail:
                edge_cases.append(
                    {
                        "id": "chain_integrity_warn",
                        "detail": "Chain integrity reported SUSPECT with zero critical failures; inspect warnings before scaling.",
                    }
                )
                continue
            hard_blockers.append(
                {
                    "id": "chain_integrity_not_clean",
                    "severity": BLOCKED,
                    "detail": "Chain integrity stress did not exit cleanly",
                }
            )
        else:
            hard_blockers.append(
                {
                    "id": f"{result.name}_tool_failure",
                    "severity": BLOCKED,
                    "detail": f"{result.name} exited {result.returncode}",
                }
            )


def _inspect_live_report(
    live_report: dict[str, Any],
    strategy_rows: dict[str, dict[str, Any]],
    fdr_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    hard_blockers: list[dict[str, Any]] = []
    silences: list[dict[str, Any]] = []
    edge_cases: list[dict[str, Any]] = []

    deployment = live_report.get("deployment_summary", {})
    if deployment.get("deployed_not_validated"):
        hard_blockers.append(
            {
                "id": "deployed_not_validated",
                "severity": NO_GO,
                "detail": deployment["deployed_not_validated"],
            }
        )

    allocator = live_report.get("allocator_summary") or {}
    if allocator.get("available") is False:
        hard_blockers.append({"id": "allocator_missing", "severity": BLOCKED, "detail": allocator.get("source_path")})
    if allocator.get("profile_match") is False:
        hard_blockers.append(
            {
                "id": "allocator_profile_mismatch",
                "severity": BLOCKED,
                "detail": {
                    "allocation_profile_id": allocator.get("allocation_profile_id"),
                    "requested": live_report.get("profile_id"),
                },
            }
        )

    criterion12 = live_report.get("criterion12") or {}
    if criterion12.get("valid") is False:
        hard_blockers.append(
            {
                "id": "criterion12_invalid",
                "severity": BLOCKED,
                "detail": {
                    "reason": criterion12.get("reason"),
                    "state_age_days": criterion12.get("state_age_days"),
                },
            }
        )

    for lane in live_report.get("active_lanes", []):
        sid = str(lane.get("strategy_id"))
        if lane.get("lifecycle_blocked") or lane.get("paused"):
            hard_blockers.append(
                {
                    "id": "selected_lane_paused_or_blocked",
                    "severity": BLOCKED,
                    "strategy_id": sid,
                    "detail": lane.get("lifecycle_block_reason")
                    or lane.get("pause_reason")
                    or lane.get("status_reason"),
                }
            )
        elif lane.get("sr_status") == "ALARM":
            edge_cases.append(
                {
                    "id": "selected_lane_sr_alarm_watch",
                    "strategy_id": sid,
                    "detail": lane.get("lifecycle_block_reason") or lane.get("sr_review_summary") or "reviewed watch",
                }
            )
        row = strategy_rows.get(sid)
        if not row:
            continue
        if row.get("status") != "active" or row.get("deployment_scope") != "deployable":
            hard_blockers.append(
                {
                    "id": "selected_lane_not_deployable",
                    "severity": NO_GO,
                    "strategy_id": sid,
                    "detail": {"status": row.get("status"), "deployment_scope": row.get("deployment_scope")},
                }
            )
        if fdr_rows.get(sid, {}).get("current_pass") is False:
            hard_blockers.append(
                {
                    "id": "selected_lane_current_k_fdr_fail",
                    "severity": BLOCKED,
                    "strategy_id": sid,
                    "detail": fdr_rows[sid],
                }
            )
        for field in ("slippage_validation_status", "c8_oos_status"):
            if row.get(field) in (None, ""):
                silences.append(
                    {
                        "id": "missing_validation_field",
                        "strategy_id": sid,
                        "field": field,
                    }
                )
        if row.get("dsr_score") in (None, 0, 0.0):
            edge_cases.append(
                {
                    "id": "dsr_no_comfort",
                    "strategy_id": sid,
                    "detail": row.get("dsr_score"),
                }
            )
        if row.get("years_tested") is not None and row["years_tested"] < 7:
            edge_cases.append(
                {
                    "id": "short_history",
                    "strategy_id": sid,
                    "detail": row["years_tested"],
                }
            )
        if row.get("wfe") is not None and row["wfe"] > 2.0:
            edge_cases.append(
                {
                    "id": "wfe_over_amplified",
                    "strategy_id": sid,
                    "detail": row["wfe"],
                }
            )
        if row.get("robustness_status") in (None, "PURGED", "SINGLETON"):
            edge_cases.append(
                {
                    "id": "weak_family_status",
                    "strategy_id": sid,
                    "detail": row.get("robustness_status"),
                }
            )

    return hard_blockers, silences, edge_cases


def _decide_verdict(hard_blockers: list[dict[str, Any]], silences: list[dict[str, Any]]) -> str:
    if any(blocker.get("severity") == NO_GO for blocker in hard_blockers):
        return NO_GO
    if hard_blockers or silences:
        return BLOCKED
    return GO


def build_gate_report(
    *,
    profile_id: str,
    db_path: Path = GOLD_DB_PATH,
    runner: CommandRunner = run_command,
    deployability_runner: DeployabilityRunner | None = None,
    run_external_checks: bool = True,
    run_deployability_audit: bool = True,
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    commands: list[CommandResult] = []
    hard_blockers: list[dict[str, Any]] = []
    silences: list[dict[str, Any]] = []
    edge_cases: list[dict[str, Any]] = []

    if run_external_checks:
        py = sys.executable
        commands = [
            runner("check_drift", [py, "pipeline/check_drift.py"], timeout_seconds),
            runner("fdr_integrity", [py, "scripts/tools/audit_fdr_integrity.py"], timeout_seconds),
            runner("chain_integrity", [py, "scripts/tools/stress_test_chain_integrity.py"], timeout_seconds),
        ]
        _classify_commands(commands, hard_blockers, edge_cases)

    live_report = build_live_readiness_report(profile_id=profile_id, db_path=db_path)
    lane_ids = _active_lane_ids(live_report)

    with _connect_ro(db_path) as con:
        counts = load_counts(con)
        strategy_rows = _load_strategy_rows(con, lane_ids)
        fdr_rows = current_k_fdr(con, lane_ids)

    live_blockers, live_silences, live_edges = _inspect_live_report(live_report, strategy_rows, fdr_rows)
    hard_blockers.extend(live_blockers)
    silences.extend(live_silences)
    edge_cases.extend(live_edges)

    deployability_report: dict[str, Any] | None = None
    if run_deployability_audit:
        active_deployability_runner = deployability_runner or build_deployability_audit
        try:
            deployability_report = active_deployability_runner(
                db_path=db_path,
                scope="profile",
                profile_id=profile_id,
                strict=True,
            )
        except Exception as exc:
            hard_blockers.append(
                {
                    "id": "full_shelf_deployability_tool_failure",
                    "severity": BLOCKED,
                    "detail": str(exc),
                }
            )
        else:
            by_sid = {row["strategy_id"]: row for row in deployability_report.get("strategies", [])}
            for sid in lane_ids:
                row = by_sid.get(sid)
                if row is None:
                    hard_blockers.append(
                        {
                            "id": "selected_lane_missing_from_deployability_audit",
                            "severity": BLOCKED,
                            "strategy_id": sid,
                            "detail": "full-shelf audit did not return selected lane",
                        }
                    )
                elif not row.get("deployable"):
                    hard_blockers.append(
                        {
                            "id": "selected_lane_not_full_shelf_deployable",
                            "severity": BLOCKED,
                            "strategy_id": sid,
                            "detail": {
                                "verdict": row.get("verdict"),
                                "hard_issues": [
                                    issue.get("id")
                                    for issue in row.get("issues", [])
                                    if issue.get("severity") == "hard"
                                ],
                            },
                        }
                    )

    verdict = _decide_verdict(hard_blockers, silences)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "profile_id": profile_id,
        "verdict": verdict,
        "source_truth": {
            "db_path": str(db_path),
            "git_head": live_report.get("git_head"),
            "git_branch": live_report.get("git_branch"),
            "external_checks_run": run_external_checks,
            "full_shelf_deployability_audit_run": run_deployability_audit,
        },
        "counts": counts,
        "active_lane_ids": lane_ids,
        "strategy_rows": strategy_rows,
        "current_k_fdr": fdr_rows,
        "hard_blockers": hard_blockers,
        "silences": silences,
        "edge_cases": edge_cases,
        "commands": [asdict(c) for c in commands],
        "live_readiness": live_report,
        "full_shelf_deployability": deployability_report,
    }


def render_text(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        f"Adversarial Stress Gate | profile={report['profile_id']} | verdict={report['verdict']}",
        f"DB: {report['source_truth']['db_path']}",
        (
            "Counts: "
            f"raw_validated={counts.get('raw_validated')} "
            f"deployable_rows={counts.get('deployable_rows')} "
            f"unique_streams={counts.get('unique_streams')} "
            f"edge_families={counts.get('edge_families')} "
            f"rr_locked_non_purged={counts.get('rr_locked_non_purged')}"
        ),
        "Active lanes:",
    ]
    for sid in report.get("active_lane_ids", []):
        fdr = report.get("current_k_fdr", {}).get(sid, {})
        lines.append(f"  - {sid} current_k_fdr={fdr.get('current_pass')} current_adj={fdr.get('current_adj_p')}")
    deployability = report.get("full_shelf_deployability")
    if deployability:
        lines.append(f"Full-shelf deployability: {deployability.get('summary')}")

    for label in ("hard_blockers", "silences", "edge_cases"):
        lines.append(f"{label}:")
        items = report.get(label, [])
        if not items:
            lines.append("  - none")
        for item in items:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run read-only adversarial stress gate before strategy routing.")
    parser.add_argument("--profile", default="topstep_50k_mnq_auto", help="Profile id to gate.")
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH), help="DuckDB path. Defaults to canonical gold.db.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", default=None, help="Optional output path.")
    parser.add_argument("--skip-external-checks", action="store_true", help="Only run DB/live-readiness gate logic.")
    parser.add_argument(
        "--skip-deployability-audit",
        action="store_true",
        help="Diagnostic only: skip full-shelf deployability audit. Never use for promotion.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        report = build_gate_report(
            profile_id=args.profile,
            db_path=Path(args.db_path),
            run_external_checks=not args.skip_external_checks,
            run_deployability_audit=not args.skip_deployability_audit,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        payload = {"verdict": TOOL_ERROR, "error": str(exc)}
        rendered = json.dumps(payload, indent=2, sort_keys=True) if args.format == "json" else f"tool_error: {exc}"
        if args.output:
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 3

    rendered = (
        json.dumps(report, indent=2, sort_keys=True, default=str) if args.format == "json" else render_text(report)
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + ("\n" if not rendered.endswith("\n") else ""), encoding="utf-8")
    else:
        print(rendered)
    return EXIT_CODES[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
