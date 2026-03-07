"""Tests for ui_v2/discipline_api.py — JSONL I/O with tmp_path."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

# ── JSONL loading ────────────────────────────────────────────────────────────


def test_load_jsonl_missing_file():
    from ui_v2.discipline_api import _load_jsonl

    result = _load_jsonl(Path("/nonexistent/file.jsonl"))
    assert result == []


def test_load_jsonl_valid(tmp_path):
    from ui_v2.discipline_api import _load_jsonl

    path = tmp_path / "test.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    result = _load_jsonl(path)
    assert len(result) == 2
    assert result[0] == {"a": 1}


def test_load_jsonl_skips_corrupt_lines(tmp_path):
    from ui_v2.discipline_api import _load_jsonl

    path = tmp_path / "test.jsonl"
    path.write_text('{"a": 1}\nNOT JSON\n{"b": 2}\n', encoding="utf-8")
    result = _load_jsonl(path)
    assert len(result) == 2  # skips corrupt line


# ── append_debrief ───────────────────────────────────────────────────────────


def test_append_debrief_creates_file(tmp_path):
    from ui_v2.discipline_api import append_debrief

    path = tmp_path / "debriefs.jsonl"
    record = {"strategy_id": "test_1", "adherence": "followed", "pnl_r": 1.5}
    result = append_debrief(record, path=path)
    assert result is True
    assert path.exists()

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["strategy_id"] == "test_1"


def test_append_debrief_appends(tmp_path):
    from ui_v2.discipline_api import append_debrief

    path = tmp_path / "debriefs.jsonl"
    append_debrief({"id": 1}, path=path)
    append_debrief({"id": 2}, path=path)

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


# ── load_debriefs ────────────────────────────────────────────────────────────


def test_load_debriefs_empty_when_missing(tmp_path):
    from ui_v2.discipline_api import load_debriefs

    result = load_debriefs(path=tmp_path / "missing.jsonl")
    assert result == []


def test_load_debriefs_reads_all(tmp_path):
    from ui_v2.discipline_api import append_debrief, load_debriefs

    path = tmp_path / "debriefs.jsonl"
    append_debrief({"id": 1}, path=path)
    append_debrief({"id": 2}, path=path)

    result = load_debriefs(path=path)
    assert len(result) == 2


# ── discipline events ────────────────────────────────────────────────────────


def test_append_discipline_event(tmp_path):
    from ui_v2.discipline_api import append_discipline_event

    path = tmp_path / "state.jsonl"
    result = append_discipline_event("cooling_triggered", {"pnl_r": -1.0}, path=path)
    assert result is True

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    record = json.loads(lines[0])
    assert record["event"] == "cooling_triggered"
    assert record["pnl_r"] == -1.0
    assert "ts" in record


# ── pending debriefs ─────────────────────────────────────────────────────────


def test_get_pending_debriefs_finds_unmatched(tmp_path):
    from ui_v2.discipline_api import get_pending_debriefs

    signals_path = tmp_path / "signals.jsonl"
    debriefs_path = tmp_path / "debriefs.jsonl"

    # Write 2 exit signals
    signals = [
        {"type": "SIGNAL_EXIT", "strategy_id": "s1", "ts": "2026-03-04T10:00:00"},
        {"type": "SIGNAL_EXIT", "strategy_id": "s2", "ts": "2026-03-04T11:00:00"},
    ]
    signals_path.write_text("\n".join(json.dumps(s) for s in signals) + "\n", encoding="utf-8")

    # Write 1 matching debrief
    debrief = {"strategy_id": "s1", "signal_exit_ts": "2026-03-04T10:00:00"}
    debriefs_path.write_text(json.dumps(debrief) + "\n", encoding="utf-8")

    pending = get_pending_debriefs(signals_path=signals_path, debriefs_path=debriefs_path)
    assert len(pending) == 1
    assert pending[0]["strategy_id"] == "s2"


# ── adherence stats ──────────────────────────────────────────────────────────


def test_compute_adherence_stats_empty(tmp_path):
    from ui_v2.discipline_api import compute_adherence_stats

    path = tmp_path / "empty.jsonl"
    result = compute_adherence_stats(path=path)
    assert result["total"] == 0
    assert result["adherence_rate"] == 0.0


def test_compute_adherence_stats_mixed(tmp_path):
    from ui_v2.discipline_api import compute_adherence_stats

    path = tmp_path / "debriefs.jsonl"
    records = [
        {"strategy_id": "MGC_CME_REOPEN_E2_1", "adherence": "followed", "pnl_r": 2.0},
        {"strategy_id": "MGC_CME_REOPEN_E2_2", "adherence": "modified", "pnl_r": -1.0, "deviation_cost_dollars": 50},
        {"strategy_id": "MGC_CME_REOPEN_E2_3", "adherence": "followed", "pnl_r": 1.0},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    result = compute_adherence_stats(path=path)
    assert result["total"] == 3
    assert result["followed"] == 2
    assert result["adherence_rate"] == pytest.approx(2 / 3)
    assert result["avg_r_followed"] == pytest.approx(1.5)
    assert result["deviation_cost_dollars"] == 50


# ── cooling period ───────────────────────────────────────────────────────────


def test_trigger_cooling_sets_until(tmp_path):
    from ui_v2.discipline_api import is_cooling_active, trigger_cooling

    state = {}
    trigger_cooling(
        state,
        pnl_r=-1.0,
        consecutive_losses=2,
        session_pnl_r=-2.0,
        state_path=tmp_path / "state.jsonl",
    )
    assert "cooling_until" in state
    assert is_cooling_active(state) is True


def test_is_cooling_active_false_when_empty():
    from ui_v2.discipline_api import is_cooling_active

    assert is_cooling_active({}) is False


def test_is_cooling_active_false_when_expired():
    from ui_v2.discipline_api import is_cooling_active

    past = (datetime.now(UTC) - __import__("datetime").timedelta(seconds=200)).isoformat()
    state = {"cooling_until": past}
    assert is_cooling_active(state) is False


def test_cooling_remaining_seconds_zero_when_empty():
    from ui_v2.discipline_api import cooling_remaining_seconds

    assert cooling_remaining_seconds({}) == 0.0


def test_override_cooling_clears_state(tmp_path):
    from ui_v2.discipline_api import is_cooling_active, override_cooling, trigger_cooling

    state = {}
    trigger_cooling(
        state,
        pnl_r=-1.0,
        consecutive_losses=1,
        session_pnl_r=-1.0,
        state_path=tmp_path / "state.jsonl",
    )
    assert is_cooling_active(state) is True

    override_cooling(state, state_path=tmp_path / "state.jsonl")
    assert is_cooling_active(state) is False


# ── constants ────────────────────────────────────────────────────────────────


def test_adherence_values_present():
    from ui_v2.discipline_api import ADHERENCE_VALUES

    assert "followed" in ADHERENCE_VALUES
    assert "off_plan" in ADHERENCE_VALUES


def test_deviation_triggers_present():
    from ui_v2.discipline_api import DEVIATION_TRIGGERS

    assert "tilt_revenge_reentry" in DEVIATION_TRIGGERS
    assert len(DEVIATION_TRIGGERS) > 10


def test_cooling_seconds():
    from ui_v2.discipline_api import COOLING_SECONDS

    assert COOLING_SECONDS == 90


# ── coaching note ────────────────────────────────────────────────────────────


def test_load_coaching_note_missing_file(tmp_path):
    from ui_v2.discipline_api import load_coaching_note

    result = load_coaching_note(digests_path=tmp_path / "missing.jsonl")
    assert result is None


def test_load_coaching_note_returns_latest(tmp_path):
    from ui_v2.discipline_api import load_coaching_note

    path = tmp_path / "digests.jsonl"
    records = [
        {"coaching_note": "First note"},
        {"coaching_note": "Latest note"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    result = load_coaching_note(digests_path=path)
    assert result == "Latest note"


# ── latest letter ────────────────────────────────────────────────────────────


def test_get_latest_letter_none_when_empty(tmp_path):
    from ui_v2.discipline_api import get_latest_letter

    result = get_latest_letter("CME_REOPEN", path=tmp_path / "empty.jsonl")
    assert result is None


def test_get_latest_letter_finds_match(tmp_path):
    from ui_v2.discipline_api import get_latest_letter

    path = tmp_path / "debriefs.jsonl"
    records = [
        {"strategy_id": "MGC_CME_REOPEN_E2_1", "letter_to_future_self": "Stay patient", "ts": "2026-03-03T10:00:00"},
        {"strategy_id": "MGC_TOKYO_OPEN_E2_1", "letter_to_future_self": "Wrong session", "ts": "2026-03-03T11:00:00"},
        {"strategy_id": "MGC_CME_REOPEN_E2_2", "letter_to_future_self": "Trust the edge", "ts": "2026-03-04T10:00:00"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    result = get_latest_letter("CME_REOPEN", path=path)
    assert result is not None
    assert result["text"] == "Trust the edge"
