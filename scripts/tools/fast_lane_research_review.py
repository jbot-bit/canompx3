#!/usr/bin/env python3
"""Fast Lane Phase 5 report-only research review.

See ``docs/specs/fast_lane_state_graph.md`` for the canonical fast-lane chain.

This tool reads Fast Lane chain state plus bounded per-strategy strategy-lab
payloads and emits a research review report to stdout. It does not write
canonical runtime, validator, allocation, broker, or live-control state.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from trading_app.prop_profiles import legacy_lane_allocation_path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPO_ROOT / "docs" / "runtime"
STATUS_ROLLUP_PATH = RUNTIME_DIR / "fast_lane_status.yaml"
JOURNAL_PATH = RUNTIME_DIR / "cherry_pick_journal.yaml"
CAPITAL_PACKET_JSON_PATH = RUNTIME_DIR / "fast_lane_capital_packet.json"
TRUTH_LEDGER_JSON_PATH = RUNTIME_DIR / "fast_lane_truth_ledger.json"
TRUTH_LEDGER_CSV_PATH = RUNTIME_DIR / "fast_lane_truth_ledger.csv"

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

_TITLE_RE = re.compile(r"^#\s+Chordia strict unlock audit\s+[—-]\s+(?P<strategy_id>\S+)\s*$", re.MULTILINE)
_MEASURED_RE = re.compile(r"\*\*MEASURED (?P<key>[^:*]+):\*\*\s*`?(?P<value>[^`\n]+)`?")

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
            prior = latest.get(sid)
            if prior is None or (entry.get("iter") or 0) > (prior.get("iter") or 0):
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


def _repo_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _markdown_table(text: str, heading: str) -> list[dict[str, str]]:
    marker = f"## {heading}"
    start = text.find(marker)
    if start < 0:
        return []
    lines = text[start:].splitlines()[1:]
    table_lines: list[str] = []
    in_table = False
    for line in lines:
        if line.startswith("|"):
            table_lines.append(line)
            in_table = True
        elif in_table:
            break
    if len(table_lines) < 3:
        return []
    headers = _cells(table_lines[0])
    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        cells = _cells(line)
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells, strict=True)))
    return rows


def _floatish(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).replace("%", "").strip()
        if text.lower() in {"nan", "none", ""}:
            return None
        return float(text)
    except ValueError:
        return None


def _intish(value: Any) -> int | None:
    parsed = _floatish(value)
    return int(parsed) if parsed is not None else None


def _parse_scope_from_strategy_id(strategy_id: str) -> dict[str, Any]:
    parts = strategy_id.split("_")
    if len(parts) < 5:
        return {
            "instrument": None,
            "session": None,
            "orb_minutes": None,
            "entry_model": None,
            "rr_target": None,
            "confirm_bars": None,
            "filter_type": None,
            "feature_family": "UNKNOWN",
        }
    instrument = parts[0]
    entry_idx = next((idx for idx, part in enumerate(parts) if re.fullmatch(r"E\d+", part)), None)
    if entry_idx is None or entry_idx + 2 >= len(parts):
        return {
            "instrument": instrument,
            "session": None,
            "orb_minutes": None,
            "entry_model": None,
            "rr_target": None,
            "confirm_bars": None,
            "filter_type": None,
            "feature_family": "UNKNOWN",
        }
    session = "_".join(parts[1:entry_idx])
    rr_raw = parts[entry_idx + 1].removeprefix("RR")
    cb_raw = parts[entry_idx + 2].removeprefix("CB")
    filter_parts = parts[entry_idx + 3 :]
    orb_minutes = None
    if filter_parts and re.fullmatch(r"O\d+", filter_parts[-1]):
        orb_minutes = _intish(filter_parts[-1].removeprefix("O"))
        filter_parts = filter_parts[:-1]
    filter_type = "_".join(filter_parts) or None
    return {
        "instrument": instrument,
        "session": session,
        "orb_minutes": orb_minutes,
        "entry_model": parts[entry_idx],
        "rr_target": _floatish(rr_raw),
        "confirm_bars": _intish(cb_raw),
        "filter_type": filter_type,
        "feature_family": _feature_family(filter_type),
    }


def _feature_family(filter_type: str | None) -> str:
    if not filter_type:
        return "UNKNOWN"
    family_prefixes = (
        ("NO_FILTER", "NO_FILTER"),
        ("COST_LT", "COST"),
        ("ORB_VOL", "ORB_VOLUME"),
        ("ORB_G", "ORB_SIZE"),
        ("ATR_P", "ATR_PERCENTILE"),
        ("VWAP_MID", "VWAP_MID"),
        ("X_MES_ATR", "CROSS_ASSET_ATR"),
        ("PD_CLEAR", "PRIOR_DAY_CLEAR"),
        ("OVNRNG", "OVERNIGHT_RANGE"),
        ("VOL_RV20", "RELATIVE_VOLUME"),
    )
    for prefix, family in family_prefixes:
        if filter_type.startswith(prefix):
            return family
    return filter_type.split("_", maxsplit=1)[0]


def parse_result_md(path: Path) -> dict[str, Any] | None:
    """Parse the bounded Chordia result MD into report evidence.

    The parser is intentionally narrow: if the expected measured fields or
    summary table are missing, the caller must fail closed instead of inferring
    a pass from prose.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    title_match = _TITLE_RE.search(text)
    if title_match is None:
        return None
    measured = {
        m.group("key").strip().lower().replace(" ", "_"): m.group("value").strip() for m in _MEASURED_RE.finditer(text)
    }
    split_rows = {row.get("Split"): row for row in _markdown_table(text, "Split summary")}
    is_row = split_rows.get("IS")
    oos_row = split_rows.get("OOS")
    if not is_row:
        return None
    strategy_id = title_match.group("strategy_id")
    is_t = _floatish(is_row.get("t"))
    is_n = _intish(is_row.get("N_fired"))
    is_expr = _floatish(is_row.get("ExpR"))
    oos_n = _intish((oos_row or {}).get("N_fired"))
    oos_expr = _floatish((oos_row or {}).get("ExpR"))
    threshold = _floatish(measured.get("threshold_applied"))
    verdict = measured.get("verdict")
    dir_match = None
    if is_expr is not None and oos_expr is not None and oos_n is not None and oos_n >= 30:
        dir_match = (is_expr >= 0 and oos_expr >= 0) or (is_expr < 0 and oos_expr < 0)
    blocker = None
    if verdict in FAIL_HEAVYWEIGHT_VERDICTS:
        blocker = "heavyweight_failed"
    elif verdict == "PARK":
        blocker = "oos_sign_flip_or_incomplete_confirmation"
    elif oos_n is not None and oos_n < 30:
        blocker = "oos_n_below_confirmation_floor"

    scope = _parse_scope_from_strategy_id(strategy_id)
    return {
        "strategy_id": strategy_id,
        "md_path": str(path.relative_to(REPO_ROOT)).replace("\\", "/") if path.is_relative_to(REPO_ROOT) else str(path),
        "heavyweight_verdict": verdict,
        "is_t": is_t,
        "threshold_applied": threshold,
        "c8_oos_status": _oos_status(verdict, oos_n=oos_n, dir_match=dir_match),
        "dir_match": dir_match,
        "n_unique_days": is_n,
        "is_n_fired": is_n,
        "oos_n_fired": oos_n,
        "is_expr": is_expr,
        "oos_expr": oos_expr,
        "blocker": blocker,
        **scope,
    }


def _oos_status(verdict: str | None, *, oos_n: int | None, dir_match: bool | None) -> str:
    if oos_n is None:
        return "OOS_MISSING"
    if oos_n < 30:
        return "OOS_N_BELOW_30"
    if dir_match is True:
        return "OOS_SIGN_MATCH_N_GE_30"
    if dir_match is False:
        return "OOS_SIGN_FLIP_N_GE_30"
    if verdict in PASS_HEAVYWEIGHT_VERDICTS:
        return "OOS_CONFIRMATION_UNKNOWN"
    return "OOS_NOT_CONFIRMING"


def result_evidence_for_status(status_entry: dict[str, Any]) -> dict[str, Any] | None:
    path = _repo_path(status_entry.get("upstream_artifact_path"))
    if path is None:
        return None
    return parse_result_md(path)


def _evidence_for_row(
    status_entry: dict[str, Any],
    journal_entry: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str, str | None]:
    if journal_entry is not None and _heavyweight_verdict(journal_entry) is not None:
        return (
            {
                "strategy_id": status_entry.get("strategy_id"),
                "heavyweight_verdict": _heavyweight_verdict(journal_entry),
                "oos_power_tier": journal_entry.get("oos_power_tier"),
                "is_t": journal_entry.get("t_observed_post_clustered_se"),
            },
            "journal",
            None,
        )
    parsed = result_evidence_for_status(status_entry)
    if parsed is None:
        return None, "missing", "journal_missing_result_md_missing_or_unparseable"
    if status_entry.get("lineage_class") == "FAST_LANE" and status_entry.get("current_stage") in {
        "HEAVYWEIGHT_COMPLETE",
        "ENRICHED",
    }:
        return parsed, "result_md_integrity_fallback", "journal_missing_integrity_break"
    return parsed, "result_md", None


def _classify_recommendation(
    status_entry: dict[str, Any],
    journal_entry: dict[str, Any] | None,
    strategy_lab_payload: dict[str, Any],
    result_evidence: dict[str, Any] | None = None,
    evidence_problem: str | None = None,
) -> dict[str, Any]:
    stage = status_entry.get("current_stage")
    lineage = status_entry.get("lineage_class")
    blocker = status_entry.get("blocker_class")
    strategy_lab_verdict = strategy_lab_payload.get("verdict") if isinstance(strategy_lab_payload, dict) else None
    heavyweight_verdict = _heavyweight_verdict(journal_entry) or (result_evidence or {}).get("heavyweight_verdict")

    reason_codes: list[str] = []
    next_review_action = "no_capital_action"

    if lineage == "DIRECT_HEAVYWEIGHT":
        reason_codes.append("direct_heavyweight_context_only")

    if evidence_problem == "journal_missing_integrity_break":
        return {
            "recommendation": "PARK",
            "next_review_action": "repair_fast_lane_journal_lineage",
            "reason_codes": [*reason_codes, evidence_problem],
        }

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
            "reason_codes": [*reason_codes, evidence_problem or "missing_heavyweight_pass"],
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


TRUTH_LEDGER_FIELDS = (
    "strategy_id",
    "lineage",
    "verdict",
    "is_t",
    "threshold_applied",
    "c8_oos_status",
    "dir_match",
    "n_unique_days",
    "feature_family",
    "session",
    "instrument",
    "md_path",
    "blocker",
)


def build_truth_ledger(
    status_entries: list[dict[str, Any]],
    *,
    journal_by_strategy: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    journal = journal_by_strategy or {}
    rows: list[dict[str, Any]] = []
    for entry in status_entries:
        if entry.get("current_stage") != "HEAVYWEIGHT_COMPLETE":
            continue
        sid = str(entry.get("strategy_id") or "")
        parsed = result_evidence_for_status(entry)
        if parsed is None:
            rows.append(
                {
                    "strategy_id": entry.get("strategy_id"),
                    "lineage": entry.get("lineage_class"),
                    "verdict": None,
                    "is_t": None,
                    "threshold_applied": None,
                    "c8_oos_status": "RESULT_MD_MISSING_OR_UNPARSEABLE",
                    "dir_match": None,
                    "n_unique_days": None,
                    "feature_family": "UNKNOWN",
                    "session": None,
                    "instrument": None,
                    "md_path": entry.get("upstream_artifact_path"),
                    "blocker": "result_md_missing_or_unparseable",
                }
            )
            continue
        blocker = parsed.get("blocker")
        if entry.get("lineage_class") == "FAST_LANE" and sid not in journal:
            blocker = "journal_missing_integrity_break"
        rows.append(
            {
                "strategy_id": parsed.get("strategy_id"),
                "lineage": entry.get("lineage_class"),
                "verdict": parsed.get("heavyweight_verdict"),
                "is_t": parsed.get("is_t"),
                "threshold_applied": parsed.get("threshold_applied"),
                "c8_oos_status": parsed.get("c8_oos_status"),
                "dir_match": parsed.get("dir_match"),
                "n_unique_days": parsed.get("n_unique_days"),
                "feature_family": parsed.get("feature_family"),
                "session": parsed.get("session"),
                "instrument": parsed.get("instrument"),
                "md_path": parsed.get("md_path"),
                "blocker": blocker,
                "orb_minutes": parsed.get("orb_minutes"),
                "entry_model": parsed.get("entry_model"),
                "rr_target": parsed.get("rr_target"),
                "confirm_bars": parsed.get("confirm_bars"),
                "filter_type": parsed.get("filter_type"),
                "oos_n_fired": parsed.get("oos_n_fired"),
                "is_expr": parsed.get("is_expr"),
                "oos_expr": parsed.get("oos_expr"),
            }
        )
    rows.sort(key=lambda row: str(row.get("strategy_id") or ""))
    return rows


def _ranking_eligible(row: dict[str, Any]) -> bool:
    verdict = row.get("verdict") or row.get("heavyweight_verdict")
    return (
        verdict == "PASS_CHORDIA"
        and row.get("c8_oos_status") == "OOS_SIGN_MATCH_N_GE_30"
        and row.get("blocker") in (None, "")
    )


def _filter_reason(row: dict[str, Any]) -> str:
    verdict = row.get("verdict") or row.get("heavyweight_verdict")
    if row.get("blocker"):
        return str(row["blocker"])
    if verdict != "PASS_CHORDIA":
        return f"verdict_{str(verdict or 'missing').lower()}"
    if row.get("c8_oos_status") != "OOS_SIGN_MATCH_N_GE_30":
        return f"c8_{str(row.get('c8_oos_status') or 'missing').lower()}"
    return "included_for_ranking"


def filtered_out_summary(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(_filter_reason(row) for row in ledger if not _ranking_eligible(row))
    return {
        "total_rows": len(ledger),
        "included_for_ranking": sum(1 for row in ledger if _ranking_eligible(row)),
        "filtered_out": sum(reasons.values()),
        "by_reason": dict(sorted(reasons.items())),
    }


def _rank_key(row: dict[str, Any]) -> tuple[float, int, str]:
    is_t = row.get("is_t")
    n_days = row.get("n_unique_days")
    return (
        float(is_t) if isinstance(is_t, int | float) else float("-inf"),
        int(n_days) if isinstance(n_days, int) else -1,
        str(row.get("strategy_id") or ""),
    )


def build_capital_packet(
    ledger: list[dict[str, Any]],
    *,
    current_lanes: list[str] | None = None,
) -> dict[str, Any]:
    current = set(current_lanes or [])
    ranked = sorted([row for row in ledger if _ranking_eligible(row)], key=_rank_key, reverse=True)
    decisions: dict[str, dict[str, Any]] = {}
    for row in ledger:
        sid = str(row.get("strategy_id") or "")
        reason = _filter_reason(row)
        verdict = row.get("verdict") or row.get("heavyweight_verdict")
        if _ranking_eligible(row):
            bucket = "APPROVE_DRY_RUN"
            reason = "pass_chordia_and_oos_sign_match"
        elif verdict in FAIL_HEAVYWEIGHT_VERDICTS or row.get("blocker") == "heavyweight_failed":
            bucket = "REJECT"
        else:
            bucket = "WATCH"
            if row.get("blocker") == "oos_sign_flip_or_incomplete_confirmation":
                bucket = "REJECT"
        decisions[sid] = {"bucket": bucket, "reason": reason, "md_path": row.get("md_path")}

    ranked_ids = [str(row.get("strategy_id")) for row in ranked]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "scripts/tools/fast_lane_research_review.py",
        "generated_at": datetime.now(tz=UTC).date().isoformat(),
        "capital_boundary": CAPITAL_BOUNDARY,
        "methodology": {
            "ranking_rule": "rank only PASS_CHORDIA rows with OOS_SIGN_MATCH_N_GE_30",
            "allocator_action": "dry_run_only_no_profile_or_lane_allocation_write",
        },
        "filtered_out_summary": filtered_out_summary(ledger),
        "ranked_candidates": ranked,
        "decisions": decisions,
        "rebalance_dry_run_diff": {
            "current_lanes": sorted(current),
            "would_add": [sid for sid in ranked_ids if sid not in current],
            "would_keep": [sid for sid in ranked_ids if sid in current],
            "would_remove": [sid for sid in sorted(current) if sid not in ranked_ids],
        },
    }


def load_current_lane_ids(path: Path | None = None) -> list[str]:
    if path is None:
        path = legacy_lane_allocation_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    lanes = payload.get("lanes") if isinstance(payload, dict) else None
    if not isinstance(lanes, list):
        return []
    out: list[str] = []
    for lane in lanes:
        if isinstance(lane, dict) and isinstance(lane.get("strategy_id"), str):
            out.append(lane["strategy_id"])
    return out


def write_truth_ledger_json(ledger: list[dict[str, Any]], path: Path = TRUTH_LEDGER_JSON_PATH) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": "scripts/tools/fast_lane_research_review.py",
        "generated_at": datetime.now(tz=UTC).date().isoformat(),
        "capital_boundary": CAPITAL_BOUNDARY,
        "entries": ledger,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_truth_ledger_csv(ledger: list[dict[str, Any]], path: Path = TRUTH_LEDGER_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TRUTH_LEDGER_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ledger)


def write_capital_packet_json(packet: dict[str, Any], path: Path = CAPITAL_PACKET_JSON_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")


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
        result_evidence, evidence_source, evidence_problem = _evidence_for_row(status_entry, journal_entry)
        try:
            strategy_lab_payload = strategy_lab_provider(sid, rolling_months)
        except Exception as exc:  # pragma: no cover - defensive transport boundary
            strategy_lab_payload = {"error": f"{type(exc).__name__}: {exc}", "verdict": "STRATEGY_LAB_PROVIDER_ERROR"}
        classification = _classify_recommendation(
            status_entry,
            journal_entry,
            strategy_lab_payload,
            result_evidence=result_evidence,
            evidence_problem=evidence_problem,
        )
        rows.append(
            {
                "strategy_id": sid,
                "current_stage": status_entry.get("current_stage"),
                "lineage_class": status_entry.get("lineage_class"),
                "blocker_class": status_entry.get("blocker_class"),
                "primary_blocker": status_entry.get("primary_blocker"),
                "heavyweight_verdict": _heavyweight_verdict(journal_entry)
                or (result_evidence or {}).get("heavyweight_verdict"),
                "oos_power_tier": (journal_entry or {}).get("oos_power_tier"),
                "evidence_source": evidence_source,
                "evidence_problem": evidence_problem,
                "result_md_path": (result_evidence or {}).get("md_path"),
                "is_t": (result_evidence or {}).get("is_t"),
                "c8_oos_status": (result_evidence or {}).get("c8_oos_status"),
                "dir_match": (result_evidence or {}).get("dir_match"),
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
        "filtered_out_summary": filtered_out_summary(
            build_truth_ledger(selected, journal_by_strategy=journal_by_strategy)
        ),
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
        "## Filtered-out summary",
        "",
        f"`{report.get('filtered_out_summary') or {}}`",
        "",
        "## Recommendations",
        "",
        "| strategy_id | stage | lineage | blocker | evidence | heavyweight | c8_oos | strategy_lab | recommendation | next_review_action | reasons |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.get("entries") or []:
        if not isinstance(row, dict):
            continue
        reasons = ",".join(str(x) for x in row.get("reason_codes") or [])
        lines.append(
            "| {strategy_id} | {stage} | {lineage} | {blocker} | {evidence} | {heavyweight} | {c8_oos} | {strategy_lab} | {recommendation} | {next_action} | {reasons} |".format(
                strategy_id=row.get("strategy_id"),
                stage=row.get("current_stage"),
                lineage=row.get("lineage_class"),
                blocker=row.get("blocker_class"),
                evidence=row.get("evidence_source"),
                heavyweight=row.get("heavyweight_verdict"),
                c8_oos=row.get("c8_oos_status"),
                strategy_lab=row.get("strategy_lab_verdict"),
                recommendation=row.get("recommendation"),
                next_action=row.get("next_review_action"),
                reasons=reasons,
            )
        )
    if not report.get("entries"):
        lines.append(
            "| _none_ | _none_ | _none_ | _none_ | _none_ | _none_ | _none_ | _none_ | _none_ | _none_ | _none_ |"
        )

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
    parser.add_argument(
        "--write-audit-packet",
        action="store_true",
        help=(
            "Write report-only derived artifacts: truth ledger CSV/JSON and "
            "dry-run capital packet JSON under docs/runtime/."
        ),
    )
    parser.add_argument(
        "--packet-only",
        action="store_true",
        help="With --write-audit-packet, write artifacts and print only their paths.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    status_entries = load_status_entries()
    if args.write_audit_packet:
        journal = load_latest_journal_entries()
        requested = set(args.strategy_id) if args.strategy_id is not None else None
        selected = _selected_status_entries(
            status_entries,
            strategy_ids=requested,
            include_direct_heavyweight=True,
        )
        ledger = build_truth_ledger(selected, journal_by_strategy=journal)
        packet = build_capital_packet(ledger, current_lanes=load_current_lane_ids())
        write_truth_ledger_json(ledger)
        write_truth_ledger_csv(ledger)
        write_capital_packet_json(packet)
        if args.packet_only:
            print(f"wrote {TRUTH_LEDGER_JSON_PATH.relative_to(REPO_ROOT)}")
            print(f"wrote {TRUTH_LEDGER_CSV_PATH.relative_to(REPO_ROOT)}")
            print(f"wrote {CAPITAL_PACKET_JSON_PATH.relative_to(REPO_ROOT)}")
            return 0
    report = build_review_report(
        status_entries=status_entries,
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
