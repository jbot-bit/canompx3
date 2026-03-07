"""Tests for ui_v2.orb_tracker — ORB state tracking."""

from __future__ import annotations

from datetime import UTC, datetime, timezone

import pytest

from ui_v2.orb_tracker import OrbState, OrbTracker


@pytest.fixture
def tracker() -> OrbTracker:
    return OrbTracker()


def _bar(high: float, low: float, ts: datetime | None = None) -> dict:
    return {"high": high, "low": low, "ts": ts}


def test_start_orb_creates_state(tracker: OrbTracker) -> None:
    ts = datetime(2026, 3, 7, 9, 0, tzinfo=UTC)
    state = tracker.start_orb("CME_REOPEN", "MGC", 5, _bar(100.0, 98.0, ts))
    assert state.session == "CME_REOPEN"
    assert state.instrument == "MGC"
    assert state.high == 100.0
    assert state.low == 98.0
    assert state.size == pytest.approx(2.0)
    assert state.bars_elapsed == 1
    assert state.bars_total == 5
    assert state.started_at == ts


def test_update_bar_extends_range(tracker: OrbTracker) -> None:
    tracker.start_orb("CME_REOPEN", "MGC", 5, _bar(100.0, 98.0))
    state = tracker.update_bar("CME_REOPEN", "MGC", 102.0, 97.0)
    assert state is not None
    assert state.high == 102.0
    assert state.low == 97.0
    assert state.size == pytest.approx(5.0)


def test_update_bar_increments_elapsed(tracker: OrbTracker) -> None:
    tracker.start_orb("CME_REOPEN", "MGC", 5, _bar(100.0, 98.0))
    state = tracker.update_bar("CME_REOPEN", "MGC", 100.0, 98.0)
    assert state is not None
    assert state.bars_elapsed == 2
    state = tracker.update_bar("CME_REOPEN", "MGC", 100.0, 98.0)
    assert state is not None
    assert state.bars_elapsed == 3


def test_auto_complete_at_bars_total(tracker: OrbTracker) -> None:
    tracker.start_orb("CME_REOPEN", "MGC", 5, _bar(100.0, 98.0))
    # bars_elapsed starts at 1; need 4 more updates to reach bars_total=5
    for _ in range(3):
        tracker.update_bar("CME_REOPEN", "MGC", 100.0, 98.0)
    # 4th update hits bars_total → auto-complete removes from active
    result = tracker.update_bar("CME_REOPEN", "MGC", 100.0, 98.0)
    assert result is not None
    assert result.bars_elapsed == 5
    assert tracker.get_state("CME_REOPEN", "MGC") is None


def test_get_state_unknown_returns_none(tracker: OrbTracker) -> None:
    assert tracker.get_state("FAKE_SESSION", "FAKE") is None


def test_to_dict_serializable(tracker: OrbTracker) -> None:
    ts = datetime(2026, 3, 7, 9, 0, tzinfo=UTC)
    state = tracker.start_orb("CME_REOPEN", "MGC", 5, _bar(100.0, 98.0, ts))
    d = OrbTracker.to_dict(state)
    expected_keys = {
        "session",
        "instrument",
        "orb_minutes",
        "high",
        "low",
        "size",
        "bars_elapsed",
        "bars_total",
        "started_at",
        "qualifications",
    }
    assert set(d.keys()) == expected_keys
    assert d["started_at"] == ts.isoformat()
    assert isinstance(d["qualifications"], dict)


def test_update_unknown_returns_none(tracker: OrbTracker) -> None:
    assert tracker.update_bar("FAKE", "FAKE", 100.0, 98.0) is None
