#!/usr/bin/env python3
"""Validate live trade validity packets.

This is an operator reasoning gate, not a live runner. It checks that any claim
that a setup is live-valid carries the required research, profile, runtime, and
risk evidence before the claim can be trusted.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

ALLOWED_DECISIONS = {
    "IDEA_ONLY",
    "RESEARCH_READY",
    "PAPER_READY",
    "CONTROLLED_LIVE_PILOT",
    "LIVE_VALID",
    "LIVE_BLOCKED",
    "TOPSTEP_BLOCKED",
}

BLOCKED_DECISIONS = {"LIVE_BLOCKED", "TOPSTEP_BLOCKED"}
NON_LIVE_DECISIONS = {"IDEA_ONLY", "RESEARCH_READY", "PAPER_READY"}

FORBIDDEN_EVIDENCE_EXACT = {"HANDOFF.md"}
FORBIDDEN_EVIDENCE_PREFIXES = ("memory/",)

TOPSTEP_REQUIRED_CONSTRAINTS = {
    "scaling_stage_checked",
    "daily_loss_limit_checked",
    "max_loss_limit_checked",
    "xfa_aggregate_cap_checked",
}

LIVE_LANGUAGE_RE = re.compile(r"\b(live[- ]valid|valid live|okay to trade live|take live|live trade)\b", re.I)


def _load_yaml(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, [f"{path}: invalid YAML: {exc}"]
    except OSError as exc:
        return None, [f"{path}: cannot read file: {exc}"]
    if not isinstance(loaded, dict):
        return None, [f"{path}: top-level YAML must be a mapping"]
    return loaded, []


def _is_blank(value: Any) -> bool:
    return value is None or value == "" or value == []


def _require_mapping(record: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be a mapping")
        return {}
    return value


def _require_nonblank(mapping: dict[str, Any], dotted_key: str, errors: list[str]) -> None:
    _, key = dotted_key.rsplit(".", 1)
    if _is_blank(mapping.get(key)):
        errors.append(f"{dotted_key} is required")


def _require_positive_int(mapping: dict[str, Any], dotted_key: str, errors: list[str]) -> None:
    _, key = dotted_key.rsplit(".", 1)
    value = mapping.get(key)
    if not isinstance(value, int) or value <= 0:
        errors.append(f"{dotted_key} must be a positive integer")


def _validate_decision(record: dict[str, Any], errors: list[str]) -> str:
    decision = record.get("decision")
    if decision not in ALLOWED_DECISIONS:
        errors.append(f"decision must be one of {sorted(ALLOWED_DECISIONS)}")
        return ""
    return str(decision)


def _validate_common_sections(record: dict[str, Any], errors: list[str]) -> tuple[dict[str, Any], ...]:
    trade = _require_mapping(record, "trade_context", errors)
    research = _require_mapping(record, "research_status", errors)
    profile = _require_mapping(record, "profile_context", errors)
    runtime = _require_mapping(record, "runtime_evidence", errors)
    risk = _require_mapping(record, "risk_context", errors)

    _require_nonblank(trade, "trade_context.instrument", errors)
    _require_nonblank(trade, "trade_context.session", errors)
    _require_nonblank(trade, "trade_context.date_reviewed", errors)
    _require_nonblank(record, "record.next_action", errors)

    blockers = record.get("blockers")
    if not isinstance(blockers, list):
        errors.append("blockers must be a list")

    return trade, research, profile, runtime, risk


def _evidence_refs(research: dict[str, Any], runtime: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    raw_refs = research.get("evidence_refs", [])
    if isinstance(raw_refs, list):
        refs.extend(ref for ref in raw_refs if isinstance(ref, str))
    pre_session = runtime.get("pre_session", {})
    if isinstance(pre_session, dict) and isinstance(pre_session.get("ref"), str):
        refs.append(pre_session["ref"])
    for key in ("live_readiness_ref", "adversarial_gate_ref"):
        if isinstance(runtime.get(key), str):
            refs.append(runtime[key])
    return refs


def _validate_forbidden_evidence(research: dict[str, Any], runtime: dict[str, Any], errors: list[str]) -> None:
    for ref in _evidence_refs(research, runtime):
        if ref in FORBIDDEN_EVIDENCE_EXACT:
            errors.append(f"{ref} is not valid live-validity evidence")
        if ref.startswith(FORBIDDEN_EVIDENCE_PREFIXES):
            errors.append(f"{ref} is not valid evidence; memory/ is scratch context")


def _validate_topstep(profile: dict[str, Any], errors: list[str]) -> None:
    if str(profile.get("firm", "")).lower() != "topstep":
        return
    constraints = profile.get("topstep_constraints")
    if not isinstance(constraints, dict):
        errors.append("profile_context.topstep_constraints is required for Topstep packets")
        return
    missing = sorted(k for k in TOPSTEP_REQUIRED_CONSTRAINTS if constraints.get(k) is not True)
    if missing:
        errors.append(f"profile_context.topstep_constraints missing checked flags: {missing}")


def _validate_live_valid(
    trade: dict[str, Any],
    research: dict[str, Any],
    profile: dict[str, Any],
    runtime: dict[str, Any],
    risk: dict[str, Any],
    blockers: list[Any],
    errors: list[str],
) -> None:
    _require_nonblank(trade, "trade_context.strategy_id", errors)
    _require_research_refs(research, "LIVE_VALID", errors)
    _require_nonblank(profile, "profile_context.profile_id", errors)
    _require_nonblank(profile, "profile_context.firm", errors)
    _require_allowed_profile_scope(profile, "LIVE_VALID", errors)

    _require_runtime_controls(runtime, "LIVE_VALID", require_pre_session_pass=True, errors=errors)
    _require_positive_int(risk, "risk_context.size_contracts", errors)
    if risk.get("dd_state_checked") is not True:
        errors.append("risk_context.dd_state_checked must be true for LIVE_VALID")
    if blockers:
        errors.append("LIVE_VALID cannot have blockers")


def _require_research_refs(research: dict[str, Any], decision: str, errors: list[str]) -> None:
    refs = research.get("evidence_refs")
    if not isinstance(refs, list) or not refs:
        errors.append(f"research_status.evidence_refs must be a non-empty list for {decision}")


def _require_allowed_profile_scope(profile: dict[str, Any], decision: str, errors: list[str]) -> None:
    if profile.get("allowed_lane") is not True:
        errors.append(f"profile_context.allowed_lane must be true for {decision}")
    if profile.get("allowed_session") is not True:
        errors.append(f"profile_context.allowed_session must be true for {decision}")


def _require_runtime_controls(
    runtime: dict[str, Any],
    decision: str,
    *,
    require_pre_session_pass: bool,
    errors: list[str],
) -> None:
    pre_session = runtime.get("pre_session")
    if not isinstance(pre_session, dict):
        errors.append(f"runtime_evidence.pre_session.ref is required for {decision}")
        return
    if _is_blank(pre_session.get("ref")):
        errors.append(f"runtime_evidence.pre_session.ref is required for {decision}")
    if require_pre_session_pass:
        if pre_session.get("status") != "pass":
            errors.append(f"runtime_evidence.pre_session.status must be pass for {decision}")

    if _is_blank(runtime.get("live_readiness_ref")) and _is_blank(runtime.get("adversarial_gate_ref")):
        errors.append(f"runtime_evidence.live_readiness_ref or adversarial_gate_ref is required for {decision}")
    if runtime.get("kill_flatten_available") is not True:
        errors.append(f"runtime_evidence.kill_flatten_available must be true for {decision}")
    if runtime.get("monitoring_available") is not True:
        errors.append(f"runtime_evidence.monitoring_available must be true for {decision}")


def _validate_controlled_pilot(
    research: dict[str, Any],
    profile: dict[str, Any],
    runtime: dict[str, Any],
    risk: dict[str, Any],
    errors: list[str],
) -> None:
    _require_research_refs(research, "CONTROLLED_LIVE_PILOT", errors)
    _require_allowed_profile_scope(profile, "CONTROLLED_LIVE_PILOT", errors)
    _require_runtime_controls(runtime, "CONTROLLED_LIVE_PILOT", require_pre_session_pass=False, errors=errors)
    _require_positive_int(risk, "risk_context.size_contracts", errors)
    _require_positive_int(risk, "risk_context.max_contracts", errors)
    constraints = risk.get("pilot_constraints")
    if not isinstance(constraints, list) or not constraints:
        errors.append("risk_context.pilot_constraints must be a non-empty list for CONTROLLED_LIVE_PILOT")
    if risk.get("dd_state_checked") is not True:
        errors.append("risk_context.dd_state_checked must be true for CONTROLLED_LIVE_PILOT")


def _validate_decision_semantics(record: dict[str, Any], decision: str, errors: list[str]) -> None:
    blockers = record.get("blockers")
    blocker_list = blockers if isinstance(blockers, list) else []
    text = " ".join(str(record.get(key, "")) for key in ("next_action", "decision"))
    if decision in NON_LIVE_DECISIONS and LIVE_LANGUAGE_RE.search(text):
        errors.append(f"{decision} packet contains live-valid language")
    if decision in BLOCKED_DECISIONS and not blocker_list:
        errors.append(f"{decision} requires at least one blocker")


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    decision = _validate_decision(record, errors)
    trade, research, profile, runtime, risk = _validate_common_sections(record, errors)
    _validate_forbidden_evidence(research, runtime, errors)
    _validate_topstep(profile, errors)
    _validate_decision_semantics(record, decision, errors)

    blockers = record.get("blockers")
    blocker_list = blockers if isinstance(blockers, list) else []
    if decision == "LIVE_VALID":
        _validate_live_valid(trade, research, profile, runtime, risk, blocker_list, errors)
    elif decision == "CONTROLLED_LIVE_PILOT":
        _validate_controlled_pilot(research, profile, runtime, risk, errors)

    return errors


def validate_file(path: Path) -> list[str]:
    record, errors = _load_yaml(path)
    if record is None:
        return errors
    return [f"{path}: {error}" for error in validate_record(record)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="YAML live-validity packet files to validate")
    args = parser.parse_args(argv)

    errors: list[str] = []
    for path in args.paths:
        errors.extend(validate_file(path))

    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"PASS live trade validity validation ({len(args.paths)} file(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
