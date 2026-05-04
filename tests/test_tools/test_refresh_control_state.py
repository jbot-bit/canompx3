from __future__ import annotations

from scripts.tools import refresh_control_state


def test_refresh_control_state_refreshes_only_invalid_surfaces(monkeypatch):
    before = {
        "profile_id": "topstep_50k_mnq_auto",
        "criterion11": {"available": True, "valid": False, "gate_ok": False, "reason": "code fingerprint mismatch"},
        "criterion12": {"available": True, "valid": True, "reason": None},
        "blocked_strategy_ids": [],
    }
    after = {
        "profile_id": "topstep_50k_mnq_auto",
        "criterion11": {"available": True, "valid": True, "gate_ok": True, "reason": None},
        "criterion12": {"available": True, "valid": True, "reason": None},
        "blocked_strategy_ids": [],
    }

    calls: list[str] = []
    states = [before, after]

    monkeypatch.setattr(refresh_control_state, "resolve_profile_id", lambda *_args, **_kwargs: "topstep_50k_mnq_auto")
    monkeypatch.setattr(refresh_control_state, "read_lifecycle_state", lambda *_args, **_kwargs: states.pop(0))
    monkeypatch.setattr(
        refresh_control_state,
        "evaluate_profile_survival",
        lambda **_kwargs: calls.append("c11"),
    )
    monkeypatch.setattr(
        refresh_control_state,
        "run_monitor",
        lambda **_kwargs: calls.append("c12"),
    )

    result = refresh_control_state.refresh_control_state()

    assert calls == ["c11"]
    assert result["criterion11"].refreshed is True
    assert result["criterion12"].refreshed is False
    assert result["after"]["criterion11"]["gate_ok"] is True


def test_refresh_control_state_force_runs_both(monkeypatch):
    before = {
        "profile_id": "topstep_50k_mnq_auto",
        "criterion11": {"available": True, "valid": True, "gate_ok": True, "reason": None},
        "criterion12": {"available": True, "valid": True, "reason": None},
        "blocked_strategy_ids": [],
    }
    after = {
        "profile_id": "topstep_50k_mnq_auto",
        "criterion11": {"available": True, "valid": True, "gate_ok": True, "reason": None},
        "criterion12": {"available": True, "valid": True, "reason": None},
        "blocked_strategy_ids": [],
    }

    calls: list[str] = []
    states = [before, after]

    monkeypatch.setattr(refresh_control_state, "resolve_profile_id", lambda *_args, **_kwargs: "topstep_50k_mnq_auto")
    monkeypatch.setattr(refresh_control_state, "read_lifecycle_state", lambda *_args, **_kwargs: states.pop(0))
    monkeypatch.setattr(
        refresh_control_state,
        "evaluate_profile_survival",
        lambda **_kwargs: calls.append("c11"),
    )
    monkeypatch.setattr(
        refresh_control_state,
        "run_monitor",
        lambda **_kwargs: calls.append("c12"),
    )

    result = refresh_control_state.refresh_control_state(force=True)

    assert calls == ["c11", "c12"]
    assert result["criterion11"].refreshed is True
    assert result["criterion12"].refreshed is True
