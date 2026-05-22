#!/usr/bin/env python3
"""Fast Lane Phase 5 report-only research review.

See ``docs/specs/fast_lane_state_graph.md`` for the canonical fast-lane chain.

This tool reads Fast Lane chain state plus bounded per-strategy strategy-lab
payloads and emits a research review report to stdout. It does not write
canonical runtime, validator, allocation, broker, or live-control state.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "docs" / "runtime"
STATUS_ROLLUP_PATH = RUNTIME_DIR / "fast_lane_status.yaml"
JOURNAL_PATH = RUNTIME_DIR / "cherry_pick_journal.yaml"

SCHEMA_VERSION = 1
CAPITAL_BOUNDARY = "REPORT_ONLY_NOT_DEPLOYMENT_AUTHORITY"

RECOMMENDATIONS = (
    "KILL",
    "PARK",
    "BULLPEN",
    "RECOMMEND_RESEARCH_REVIEW",
    "ESCALATE_CAPITAL_REVIEW",
)

BANNED_ACTIVE_SURFACE_PHRASES = (
    "_".join(("DEPLOYMENT", "CANDIDATE")),
    "_".join(("operator", "deployment_decision")),
    " ".join(("ready", "to", "deploy")),
    " ".join(("go", "live")),
    " ".join(("start", "trading")),
    " ".join(("allocate", "capital")),
)

PASS_HEAVYWEIGHT_VERDICTS = frozenset({"PASS_CHORDIA", "PASS_PROTOCOL_A", "PASS"})
FAIL_HEAVYWEIGHT_VERDICTS = frozenset({"FAIL_STRICT", "FAIL_STRICT_CHORDIA", "KILL", "FAIL"})
UNDERPOWERED_OOS_TIERS = frozenset({"NA_N_BELOW_FLOOR", "STATISTICALLY_USELESS", "NA_NO_OOS"})

type StrategyLabProvider = Callable[[str, int], dict[str, Any]]


def _safe_load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None


def load_status_entries(path: Path = STATUS_ROLLUP_PATH) -> list[dict[str, Any]]:
    payload = _safe_load_yaml(path)
    if not isinstance(payload, dict):
        return []
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def load_latest_journal_entries(path: Path = JOURNAL_PATH) -> dict[str, dict[str, Any]]:
    payload = _safe_load_yaml(path)
    if not isinstance(payload, dict):
        return {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        sid = entry.get("strategy_id")
        if isinstance(sid, str) and sid:
            latest[sid] = entry
    return latest


def _default_strategy_lab_provider(strategy_id: str, rolling_months: int = 18) -> dict[str, Any]:
    from scripts.tools.strategy_lab_mcp_server import get_strategy_readiness_payload

    return get_strategy_readiness_payload(strategy_id, rolling_months)


def _selected_status_entries(
    status_entries: Iterable[dict[str, Any]],
    *,
    strategy_ids: set[str] | None,
    include_direct_heavyweight: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for entry in status_entries:
        sid = entry.get("strategy_id")
        if not isinstance(sid, str) or not sid:
            continue
        if strategy_ids is not None and sid not in strategy_ids:
            continue
        lineage = entry.get("lineage_class")
        if lineage == "FAST_LANE" or (include_direct_heavyweight and lineage == "DIRECT_HEAVYWEIGHT"):
            selected.append(entry)
    return selected


def _is_underpowered(status_entry: dict[str, Any], journal_entry: dict[str, Any] | None) -> bool:
    return (
        status_entry.get("current_stage") == "REJECTED_OOS_UNPOWERED"
        or status_entry.get("blocker_class") == "UNDERPOWERED_OOS"
        or (journal_entry or {}).get("oos_power_tier") in UNDERPOWERED_OOS_TIERS
    )


def _heavyweight_verdict(journal_entry: dict[str, Any] | None) -> str | None:
    verdict = (journal_entry or {}).get("heavyweight_verdict")
    return verdict if isinstance(verdict, str) and verdict else None


def _classify_recommendation(
    status_entry: dict[str, Any],
    journal_entry: dict[str, Any] | None,
    strategy_lab_payload: dict[str, Any],
) -> dict[str, Any]:
    stage = status_entry.get("current_stage")
    lineage = status_entry.get("lineage_class")
    blocker = status_entry.get("blocker_class")
    strategy_lab_verdict = strategy_lab_payload.get("verdict") if isinstance(strategy_lab_payload, dict) else None
    heavyweight_verdict = _heavyweight_verdict(journal_entry)

    reason_codes: list[str] = []
    next_review_action = "no_capital_action"

    if lineage == "DIRECT_HEAVYWEIGHT":
        reason_codes.append("direct_heavyweight_context_only")

    if blocker == "ERROR" or stage == "ERROR":
        return {
            "recommendation": "PARK",
            "next_review_action": "resolve_state_error",
            "reason_codes": [*reason_codes, "state_error"],
        }

    if blocker in {"INVALID_ARTIFACT", "PROVENANCE_SUPPRESSED"} or stage == "REVOKED":
        return {
            "recommendation": "KILL",
            "next_review_action": "record_research_rejection",
            "reason_codes": [*reason_codes, "invalid_or_suppressed_artifact"],
        }

    if _is_underpowered(status_entry, journal_entry):
        return {
            "recommendation": "PARK",
            "next_review_action": "open_research_remedy_review",
            "reason_codes": [*reason_codes, "underpowered_oos"],
        }

    if heavyweight_verdict in FAIL_HEAVYWEIGHT_VERDICTS:
        return {
            "recommendation": "KILL",
            "next_review_action": "record_research_rejection",
            "reason_codes": [*reason_codes, "heavyweight_failed"],
        }

    if heavyweight_verdict not in PASS_HEAVYWEIGHT_VERDICTS:
        return {
            "recommendation": "PARK",
            "next_review_action": "wait_for_heavyweight_verdict",
            "reason_codes": [*reason_codes, "missing_heavyweight_pass"],
        }

    if lineage == "DIRECT_HEAVYWEIGHT":
        return {
            "recommendation": "RECOMMEND_RESEARCH_REVIEW",
            "next_review_action": "open_research_review",
            "reason_codes": reason_codes,
        }

    if strategy_lab_verdict == "PROMOTABLE":
        return {
            "recommendation": "ESCALATE_CAPITAL_REVIEW",
            "next_review_action": "open_separate_capital_review",
            "reason_codes": [*reason_codes, "heavyweight_passed", "strategy_lab_promotable"],
        }

    if strategy_lab_verdict in {"DEPLOYED", "PAUSED"}:
        return {
            "recommendation": "RECOMMEND_RESEARCH_REVIEW",
            "next_review_action": "review_existing_allocator_context",
            "reason_codes": [*reason_codes, "heavyweight_passed", f"strategy_lab_{str(strategy_lab_verdict).lower()}"],
        }

    if isinstance(strategy_lab_verdict, str) and strategy_lab_verdict.startswith("VALIDATED_BUT_"):
        return {
            "recommendation": "BULLPEN",
            "next_review_action": "monitor_research_context",
            "reason_codes": [*reason_codes, "heavyweight_passed", f"strategy_lab_{strategy_lab_verdict.lower()}"],
        }

    if strategy_lab_verdict in {"NOT_VALIDATED", "VALIDATED_FITNESS_UNAVAILABLE", "VALIDATED_UNKNOWN"}:
        return {
            "recommendation": "PARK",
            "next_review_action": "resolve_validation_context",
            "reason_codes": [*reason_codes, "heavyweight_passed", "strategy_lab_not_clear"],
        }

    return {
        "recommendation": "PARK",
        "next_review_action": next_review_action,
        "reason_codes": [*reason_codes, "strategy_lab_missing_or_unknown"],
    }


def build_review_report(
    *,
    status_entries: list[dict[str, Any]],
    journal_by_strategy: dict[str, dict[str, Any]],
    strategy_lab_provider: StrategyLabProvider = _default_strategy_lab_provider,
    strategy_ids: Iterable[str] | None = None,
    include_direct_heavyweight: bool = False,
    rolling_months: int = 18,
) -> dict[str, Any]:
    requested = set(strategy_ids) if strategy_ids is not None else None
    selected = _selected_status_entries(
        status_entries,
        strategy_ids=requested,
        include_direct_heavyweight=include_direct_heavyweight,
    )

    rows: list[dict[str, Any]] = []
    for status_entry in selected:
        sid = str(status_entry["strategy_id"])
        journal_entry = journal_by_strategy.get(sid)
        try:
            strategy_lab_payload = strategy_lab_provider(sid, rolling_months)
        except Exception as exc:  # pragma: no cover - defensive transport boundary
            strategy_lab_payload = {"error": f"{type(exc).__name__}: {exc}", "verdict": "STRATEGY_LAB_PROVIDER_ERROR"}
        classification = _classify_recommendation(status_entry, journal_entry, strategy_lab_payload)
        rows.append(
            {
                "strategy_id": sid,
                "current_stage": status_entry.get("current_stage"),
                "lineage_class": status_entry.get("lineage_class"),
                "blocker_class": status_entry.get("blocker_class"),
                "primary_blocker": status_entry.get("primary_blocker"),
                "heavyweight_verdict": _heavyweight_verdict(journal_entry),
                "oos_power_tier": (journal_entry or {}).get("oos_power_tier"),
                "strategy_lab_verdict": strategy_lab_payload.get("verdict")
                if isinstance(strategy_lab_payload, dict)
                else None,
                "strategy_lab_reason": strategy_lab_payload.get("reason")
                if isinstance(strategy_lab_payload, dict)
                else None,
                **classification,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source": "scripts/tools/fast_lane_research_review.py",
        "generated_at": datetime.now(tz=UTC).date().isoformat(),
        "capital_boundary": CAPITAL_BOUNDARY,
        "entries": rows,
    }


def _assert_render_safe(text: str) -> None:
    found = [phrase for phrase in BANNED_ACTIVE_SURFACE_PHRASES if phrase in text]
    if found:
        raise ValueError(f"Phase 5 report contains banned active-surface phrase(s): {found}")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fast Lane Research Review",
        "",
        f"schema_version: {report.get('schema_version')}",
        f"source: {report.get('source')}",
        f"generated_at: {report.get('generated_at')}",
        f"capital_boundary: {report.get('capital_boundary')}",
        "",
        "Boundary: report-only research evidence. `ESCALATE_CAPITAL_REVIEW` means open a separate human review packet; this report is not trade, account, broker, allocator, or live-control authority.",
        "",
        "## Recommendations",
        "",
        "| strategy_id | stage | lineage | blocker | heavyweight | strategy_lab | recommendation | next_review_action | reasons |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("entries") or []:
        if not isinstance(row, dict):
            continue
        reasons = ",".join(str(x) for x in row.get("reason_codes") or [])
        lines.append(
            "| {strategy_id} | {stage} | {lineage} | {blocker} | {heavyweight} | {strategy_lab} | {recommendation} | {next_action} | {reasons} |".format(
                strategy_id=row.get("strategy_id"),
                stage=row.get("current_stage"),
                lineage=row.get("lineage_class"),
                blocker=row.get("blocker_class"),
                heavyweight=row.get("heavyweight_verdict"),
                strategy_lab=row.get("strategy_lab_verdict"),
                recommendation=row.get("recommendation"),
                next_action=row.get("next_review_action"),
                reasons=reasons,
            )
        )
    if not report.get("entries"):
        lines.append("| _none_ | _none_ | _none_ | _none_ | _none_ | _none_ | _none_ | _none_ | _none_ |")

    out = "\n".join(lines) + "\n"
    _assert_render_safe(out)
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fast_lane_research_review",
        description="Emit Fast Lane Phase 5 report-only research review to stdout.",
    )
    parser.add_argument("--strategy-id", action="append", default=None, help="Limit report to one strategy_id.")
    parser.add_argument(
        "--include-direct-heavyweight",
        action="store_true",
        help="Include direct heavyweight rows as context only; they cannot exceed research review recommendation.",
    )
    parser.add_argument("--rolling-months", type=int, default=18)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_review_report(
        status_entries=load_status_entries(),
        journal_by_strategy=load_latest_journal_entries(),
        strategy_ids=args.strategy_id,
        include_direct_heavyweight=args.include_direct_heavyweight,
        rolling_months=args.rolling_months,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
