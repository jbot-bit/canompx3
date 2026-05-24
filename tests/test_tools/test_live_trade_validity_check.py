"""Tests for live trade validity packet validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.tools import live_trade_validity_check as validity


def _write_packet(tmp_path: Path, packet: dict) -> Path:
    path = tmp_path / "packet.yaml"
    path.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")
    return path


def _base_packet(**overrides: object) -> dict:
    packet: dict = {
        "trade_context": {
            "instrument": "MNQ",
            "session": "COMEX_SETTLE",
            "strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5",
            "date_reviewed": "2026-05-11",
        },
        "research_status": {
            "classification": "validated",
            "evidence_refs": ["docs/audit/results/2026-05-11-mnq-all-active-deployability.json"],
        },
        "profile_context": {
            "profile_id": "topstep_50k_mnq_auto",
            "firm": "topstep",
            "account_type": "XFA",
            "allowed_lane": True,
            "allowed_session": True,
            "topstep_constraints": {
                "scaling_stage_checked": True,
                "daily_loss_limit_checked": True,
                "max_loss_limit_checked": True,
                "xfa_aggregate_cap_checked": True,
            },
        },
        "runtime_evidence": {
            "pre_session": {"status": "pass", "ref": "data/state/session_checklist_2026-05-11_COMEX_SETTLE.json"},
            "live_readiness_ref": "docs/audit/results/2026-05-11-topstep-50k-mnq-profile-deployability.json",
            "kill_flatten_available": True,
            "monitoring_available": True,
        },
        "risk_context": {
            "size_contracts": 1,
            "max_contracts": 1,
            "dd_state_checked": True,
        },
        "decision": "PAPER_READY",
        "blockers": ["Needs fresh pre-session immediately before real order entry."],
        "next_action": "Paper only until live preflight is fresh.",
    }
    packet.update(overrides)
    return packet


def _errors_for(tmp_path: Path, packet: dict) -> list[str]:
    return validity.validate_file(_write_packet(tmp_path, packet))


def _live_valid_packet() -> dict:
    return _base_packet(
        decision="LIVE_VALID",
        blockers=[],
        next_action="Allowed only for the declared profile/session/size.",
    )


def _controlled_pilot_packet() -> dict:
    return _base_packet(
        decision="CONTROLLED_LIVE_PILOT",
        blockers=["Controlled warning: MNQ slippage event-tail review pending."],
        risk_context={
            "size_contracts": 1,
            "max_contracts": 1,
            "dd_state_checked": True,
            "pilot_constraints": ["one micro contract", "no scaling", "fresh pre-session required"],
        },
        next_action="Run as controlled pilot only at one micro contract.",
    )


def test_valid_idea_only_packet_passes(tmp_path: Path) -> None:
    packet = _base_packet(
        decision="IDEA_ONLY",
        trade_context={"instrument": "MNQ", "session": "COMEX_SETTLE", "date_reviewed": "2026-05-11"},
        profile_context={"profile_id": "", "firm": "none", "account_type": "none"},
        runtime_evidence={},
        risk_context={},
        blockers=["No research validation yet."],
        next_action="Route to external strategy intake first.",
    )
    assert _errors_for(tmp_path, packet) == []


def test_valid_paper_ready_packet_passes(tmp_path: Path) -> None:
    assert _errors_for(tmp_path, _base_packet()) == []


def test_valid_controlled_live_pilot_packet_passes(tmp_path: Path) -> None:
    assert _errors_for(tmp_path, _controlled_pilot_packet()) == []


def test_valid_live_blocked_packet_passes(tmp_path: Path) -> None:
    packet = _base_packet(
        decision="LIVE_BLOCKED",
        runtime_evidence={"pre_session": {"status": "fail", "ref": "data/state/session_checklist_fail.json"}},
        blockers=["Pre-session failed."],
        next_action="Do not trade until pre-session passes.",
    )
    assert _errors_for(tmp_path, packet) == []


def test_valid_topstep_blocked_packet_passes(tmp_path: Path) -> None:
    packet = _base_packet(
        decision="TOPSTEP_BLOCKED",
        profile_context={
            **_base_packet()["profile_context"],
            "allowed_lane": False,
        },
        blockers=["Strategy is not in the selected Topstep profile."],
        next_action="Do not trade this profile.",
    )
    assert _errors_for(tmp_path, packet) == []


def test_live_valid_requires_profile_id(tmp_path: Path) -> None:
    packet = _live_valid_packet()
    packet["profile_context"] = {**packet["profile_context"], "profile_id": ""}
    errors = _errors_for(tmp_path, packet)
    assert any("profile_context.profile_id" in error for error in errors)


def test_live_valid_requires_strategy_id(tmp_path: Path) -> None:
    packet = _live_valid_packet()
    packet["trade_context"] = {**packet["trade_context"], "strategy_id": ""}
    errors = _errors_for(tmp_path, packet)
    assert any("trade_context.strategy_id" in error for error in errors)


def test_live_valid_requires_research_evidence_refs(tmp_path: Path) -> None:
    packet = _live_valid_packet()
    packet["research_status"] = {"classification": "validated", "evidence_refs": []}
    errors = _errors_for(tmp_path, packet)
    assert any("research_status.evidence_refs" in error for error in errors)


def test_live_valid_requires_pre_session_pass(tmp_path: Path) -> None:
    packet = _live_valid_packet()
    packet["runtime_evidence"] = {
        **packet["runtime_evidence"],
        "pre_session": {"status": "missing", "ref": ""},
    }
    errors = _errors_for(tmp_path, packet)
    assert any("runtime_evidence.pre_session.status must be pass" in error for error in errors)


def test_live_valid_rejects_blockers(tmp_path: Path) -> None:
    packet = _live_valid_packet()
    packet["blockers"] = ["Monitor state is stale."]
    errors = _errors_for(tmp_path, packet)
    assert any("LIVE_VALID cannot have blockers" in error for error in errors)


def test_topstep_packet_requires_constraints(tmp_path: Path) -> None:
    packet = _live_valid_packet()
    packet["profile_context"] = {
        **packet["profile_context"],
        "topstep_constraints": {"scaling_stage_checked": True},
    }
    errors = _errors_for(tmp_path, packet)
    assert any("topstep_constraints" in error for error in errors)


def test_controlled_pilot_requires_size_and_constraints(tmp_path: Path) -> None:
    packet = _controlled_pilot_packet()
    packet["risk_context"] = {"size_contracts": 0, "dd_state_checked": True}
    errors = _errors_for(tmp_path, packet)
    assert any("risk_context.size_contracts" in error for error in errors)
    assert any("risk_context.pilot_constraints" in error for error in errors)


def test_controlled_pilot_requires_runtime_controls(tmp_path: Path) -> None:
    packet = _controlled_pilot_packet()
    packet["runtime_evidence"] = {
        "pre_session": {"status": "required_fresh_before_trade", "ref": "trading_app/pre_session_check.py"},
        "kill_flatten_available": False,
        "monitoring_available": False,
    }
    errors = _errors_for(tmp_path, packet)
    assert any("live_readiness_ref or adversarial_gate_ref" in error for error in errors)
    assert any("kill_flatten_available" in error for error in errors)
    assert any("monitoring_available" in error for error in errors)


def test_controlled_pilot_topstep_must_use_allowed_profile_lane(tmp_path: Path) -> None:
    packet = _controlled_pilot_packet()
    packet["profile_context"] = {**packet["profile_context"], "allowed_lane": False}
    errors = _errors_for(tmp_path, packet)
    assert any("allowed_lane" in error for error in errors)


def test_paper_ready_rejects_live_valid_language(tmp_path: Path) -> None:
    packet = _base_packet(next_action="This is live valid for Topstep.")
    errors = _errors_for(tmp_path, packet)
    assert any("live-valid language" in error for error in errors)


def test_forbidden_evidence_refs_fail(tmp_path: Path) -> None:
    packet = _base_packet(
        research_status={"classification": "validated", "evidence_refs": ["HANDOFF.md", "memory/2026-05-11.md"]}
    )
    errors = _errors_for(tmp_path, packet)
    assert any("HANDOFF.md" in error for error in errors)
    assert any("memory/" in error for error in errors)
