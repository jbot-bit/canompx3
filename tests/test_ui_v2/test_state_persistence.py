"""Tests for ui_v2/state_persistence.py — JSON state persistence."""

from __future__ import annotations

import json

from ui_v2.state_persistence import (
    load_commitment_state,
    load_cooling_state,
    load_state,
    save_commitment_state,
    save_cooling_state,
    save_state,
)


def test_load_state_returns_empty_on_missing(tmp_path):
    result = load_state(tmp_path / "nonexistent.json")
    assert result == {}


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    data = {"cooling": {"active": True, "remaining": 300}, "commitment": {"items": {"chart_open": True}}}
    assert save_state(data, path)

    loaded = load_state(path)
    assert loaded["cooling"]["active"] is True
    assert loaded["cooling"]["remaining"] == 300
    assert "_saved_at" in loaded  # Metadata added by save


def test_load_state_handles_corrupt_json(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("not valid json {{{", encoding="utf-8")
    result = load_state(path)
    assert result == {}


def test_cooling_state_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    cooling = {"active": True, "start_ts": "2026-03-08T10:00:00Z", "duration_s": 180}
    assert save_cooling_state(cooling, path)

    loaded = load_cooling_state(path)
    assert loaded["active"] is True
    assert loaded["duration_s"] == 180


def test_commitment_state_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    commitment = {"items": {"chart_open": True, "order_ready": False}, "date": "2026-03-08"}
    assert save_commitment_state(commitment, path)

    loaded = load_commitment_state(path)
    assert loaded["items"]["chart_open"] is True
    assert loaded["date"] == "2026-03-08"


def test_save_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "state.json"
    assert save_state({"test": True}, path)
    assert path.exists()
    loaded = load_state(path)
    assert loaded["test"] is True


def test_save_overwrites_existing(tmp_path):
    path = tmp_path / "state.json"
    save_state({"v": 1}, path)
    save_state({"v": 2}, path)
    loaded = load_state(path)
    assert loaded["v"] == 2


def test_cooling_and_commitment_coexist(tmp_path):
    """Both sub-states share the same file without clobbering each other."""
    path = tmp_path / "state.json"
    save_cooling_state({"active": True}, path)
    save_commitment_state({"items": {"risk_sized": True}, "date": "2026-03-08"}, path)

    # Both should still be present
    state = load_state(path)
    assert state["cooling"]["active"] is True
    assert state["commitment"]["items"]["risk_sized"] is True
