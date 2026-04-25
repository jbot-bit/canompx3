from __future__ import annotations

from datetime import date
from pathlib import Path

from trading_app import lifecycle_state


def test_read_criterion12_state_hashes_shared_derived_helper(tmp_path, monkeypatch):
    state_file = tmp_path / "sr_state.json"
    state_file.write_text("{}", encoding="utf-8")

    captured: list[Path] = []

    monkeypatch.setattr(lifecycle_state, "SR_STATE_PATH", state_file)
    monkeypatch.setattr(lifecycle_state, "build_db_identity", lambda _db_path: "db-identity")
    monkeypatch.setattr(lifecycle_state, "build_profile_fingerprint", lambda _profile: "profile-fingerprint")
    monkeypatch.setattr(
        lifecycle_state,
        "build_code_fingerprint",
        lambda paths: captured.extend(paths) or "code-identity",
    )
    monkeypatch.setattr(
        lifecycle_state,
        "validate_state_envelope",
        lambda *_args, **_kwargs: (False, "code fingerprint mismatch", None),
    )

    state = lifecycle_state.read_criterion12_state(
        "topstep_50k_mnq_auto",
        db_path=Path("/tmp/gold.db"),
        today=date(2026, 4, 12),
    )

    assert state["valid"] is False
    assert state["reason"] == "code fingerprint mismatch"
    assert any(path.name == "derived_state.py" for path in captured)


def test_read_lifecycle_state_includes_conditional_overlays(monkeypatch):
    overlay_state = {
        "profile_id": "topstep_50k",
        "available": True,
        "valid": True,
        "overlays": [{"overlay_id": "pr48_mgc_cont_exec_v1", "status": "ready", "valid": True}],
    }

    monkeypatch.setattr(
        lifecycle_state,
        "read_criterion11_state",
        lambda *args, **kwargs: {"profile_id": "topstep_50k", "available": False, "valid": True, "reason": None},
    )
    monkeypatch.setattr(
        lifecycle_state,
        "read_criterion12_state",
        lambda *args, **kwargs: {
            "profile_id": "topstep_50k",
            "available": False,
            "valid": True,
            "reason": None,
            "status_by_strategy": {},
            "alarm_strategy_ids": [],
            "no_data_strategy_ids": [],
        },
    )
    monkeypatch.setattr(
        lifecycle_state,
        "read_pause_state",
        lambda *args, **kwargs: {
            "profile_id": "topstep_50k",
            "paused_count": 0,
            "paused_strategy_ids": [],
            "paused_details": {},
        },
    )
    monkeypatch.setattr(lifecycle_state, "read_overlay_states", lambda *args, **kwargs: overlay_state)

    state = lifecycle_state.read_lifecycle_state("topstep_50k", today=date(2026, 4, 23))

    assert state["conditional_overlays"] == overlay_state
