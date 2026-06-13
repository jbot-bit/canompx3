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


def test_read_lifecycle_state_includes_opportunity_awareness(monkeypatch):
    opportunity_state = {
        "profile_id": "topstep_50k",
        "available": True,
        "valid": True,
        "summary": {"prime_shadow_count": 1, "watch_count": 0, "blocked_count": 0, "lane_count": 1},
        "lanes": [],
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
    monkeypatch.setattr(
        lifecycle_state,
        "read_overlay_states",
        lambda *args, **kwargs: {"profile_id": "topstep_50k", "available": False, "valid": True, "overlays": []},
    )
    monkeypatch.setattr(lifecycle_state, "read_opportunity_state", lambda *args, **kwargs: opportunity_state)

    state = lifecycle_state.read_lifecycle_state("topstep_50k", today=date(2026, 5, 13))

    assert state["opportunity_awareness"] == opportunity_state


def test_project_pulse_c12_branch_uses_classifier_for_both_classes():
    """Same classify_state_reason drives the C12 pulse branch — one taxonomy, both criteria.

    A fingerprint-mismatch C12 reason reads EXPECTED-regen; a legacy-envelope C12 reason
    reads DEFECT. Proves finding #9 (C11 and C12 share one reason vocabulary + classifier).
    """
    from scripts.tools.project_pulse import _collect_control_items_from_lifecycle

    def _pulse_c12_item(reason: str):
        lifecycle = {
            "criterion11": None,
            "criterion12": {
                "profile_id": "topstep_50k_mnq_auto",
                "available": True,
                "valid": False,
                "reason": reason,
                "counts": {},
                "status_by_strategy": {},
            },
            "pauses": None,
            "strategy_states": {},
        }
        _s, _sr, _p, items = _collect_control_items_from_lifecycle(lifecycle)
        sr_items = [i for i in items if i.source == "sr_monitor"]
        assert len(sr_items) == 1
        return sr_items[0]

    # EXPECTED-stale: code fingerprint mismatch -> regen, don't debug.
    expected = _pulse_c12_item("code fingerprint mismatch")
    assert "invalidated" in expected.summary
    assert "EXPECTED" in expected.summary
    assert "don't debug" in expected.summary
    assert "DEFECT" not in expected.summary

    # DEFECT: legacy envelope -> investigate, never 'don't debug'.
    defect = _pulse_c12_item("legacy state: missing versioned envelope")
    assert "DEFECT" in defect.summary
    assert "investigate" in defect.summary
    assert "don't debug" not in defect.summary
