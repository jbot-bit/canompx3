"""Tests for discipline data layer — JSONL I/O, enums, pattern computation."""
import json
import pytest


def test_adherence_enum_values():
    from ui.discipline_data import ADHERENCE_VALUES
    assert set(ADHERENCE_VALUES) == {"followed", "modified", "overrode", "off_plan"}


def test_deviation_trigger_values():
    from ui.discipline_data import DEVIATION_TRIGGERS
    assert "narrative" in DEVIATION_TRIGGERS
    assert "chasing_loss" in DEVIATION_TRIGGERS
    assert len(DEVIATION_TRIGGERS) == 7


def test_append_debrief_creates_file(tmp_path):
    from ui.discipline_data import append_debrief
    f = tmp_path / "debriefs.jsonl"
    record = {
        "ts": "2026-03-06T23:15:00Z",
        "trading_day": "2026-03-06",
        "instrument": "MGC",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "adherence": "followed",
        "emotional_temp": 0.5,
    }
    append_debrief(record, path=f)
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["adherence"] == "followed"


def test_append_debrief_appends(tmp_path):
    from ui.discipline_data import append_debrief
    f = tmp_path / "debriefs.jsonl"
    for i in range(3):
        append_debrief({"ts": f"2026-03-0{i+1}T00:00:00Z", "adherence": "followed"}, path=f)
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 3


def test_load_debriefs_empty(tmp_path):
    from ui.discipline_data import load_debriefs
    f = tmp_path / "debriefs.jsonl"
    assert load_debriefs(path=f) == []


def test_load_debriefs_returns_records(tmp_path):
    from ui.discipline_data import load_debriefs, append_debrief
    f = tmp_path / "debriefs.jsonl"
    append_debrief({"ts": "2026-03-06T00:00:00Z", "strategy_id": "X"}, path=f)
    records = load_debriefs(path=f)
    assert len(records) == 1
    assert records[0]["strategy_id"] == "X"


def test_append_discipline_event(tmp_path):
    from ui.discipline_data import append_discipline_event
    f = tmp_path / "state.jsonl"
    append_discipline_event("cooling_triggered", {"tilt_score": 65}, path=f)
    lines = f.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event"] == "cooling_triggered"
    assert "ts" in parsed


def test_get_pending_debriefs_finds_unmatched(tmp_path):
    from ui.discipline_data import get_pending_debriefs
    signals_file = tmp_path / "signals.jsonl"
    debriefs_file = tmp_path / "debriefs.jsonl"
    signal = {
        "ts": "2026-03-06T23:15:00Z",
        "instrument": "MGC",
        "type": "SIGNAL_EXIT",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "price": 3245.50,
    }
    signals_file.write_text(json.dumps(signal) + "\n")
    pending = get_pending_debriefs(signals_path=signals_file, debriefs_path=debriefs_file)
    assert len(pending) == 1
    assert pending[0]["strategy_id"] == "MGC_CME_REOPEN_E2_CB1_G4_RR2.5"


def test_get_pending_debriefs_excludes_debriefed(tmp_path):
    from ui.discipline_data import get_pending_debriefs, append_debrief
    signals_file = tmp_path / "signals.jsonl"
    debriefs_file = tmp_path / "debriefs.jsonl"
    signal = {
        "ts": "2026-03-06T23:15:00Z",
        "instrument": "MGC",
        "type": "SIGNAL_EXIT",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "price": 3245.50,
    }
    signals_file.write_text(json.dumps(signal) + "\n")
    append_debrief({
        "ts": "2026-03-06T23:16:00Z",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "signal_exit_ts": "2026-03-06T23:15:00Z",
        "adherence": "followed",
    }, path=debriefs_file)
    pending = get_pending_debriefs(signals_path=signals_file, debriefs_path=debriefs_file)
    assert len(pending) == 0


def test_compute_adherence_stats(tmp_path):
    from ui.discipline_data import compute_adherence_stats, append_debrief
    f = tmp_path / "debriefs.jsonl"
    for adh in ["followed", "followed", "overrode"]:
        append_debrief({"adherence": adh, "pnl_r": 1.0 if adh == "followed" else -1.0,
                        "deviation_cost_dollars": 0 if adh == "followed" else 200,
                        "instrument": "MGC", "ts": "2026-03-06T00:00:00Z"}, path=f)
    stats = compute_adherence_stats(path=f)
    assert stats["total"] == 3
    assert stats["followed"] == 2
    assert stats["adherence_rate"] == pytest.approx(2/3)
    assert stats["deviation_cost_dollars"] == 200


def test_get_latest_letter(tmp_path):
    from ui.discipline_data import get_latest_letter, append_debrief
    f = tmp_path / "debriefs.jsonl"
    append_debrief({
        "ts": "2026-03-05T00:00:00Z",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "letter_to_future_self": "Stick to the plan.",
        "adherence": "overrode",
    }, path=f)
    append_debrief({
        "ts": "2026-03-06T00:00:00Z",
        "strategy_id": "MGC_CME_REOPEN_E2_CB1_G4_RR2.5",
        "letter_to_future_self": None,
        "adherence": "followed",
    }, path=f)
    letter = get_latest_letter(session="CME_REOPEN", path=f)
    assert letter is not None
    assert letter["text"] == "Stick to the plan."
