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
