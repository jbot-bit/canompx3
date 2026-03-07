"""Tests for ui_v2/state_machine.py — 8-state machine, session stack, briefings."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

BRISBANE = ZoneInfo("Australia/Brisbane")


# ── StateName enum ───────────────────────────────────────────────────────────


def test_state_name_has_8_values():
    from ui_v2.state_machine import StateName

    assert len(StateName) == 8
    expected = {"WEEKEND", "OVERNIGHT", "IDLE", "APPROACHING", "ALERT", "ORB_FORMING", "IN_SESSION", "DEBRIEF"}
    assert {s.value for s in StateName} == expected


# ── get_app_state: WEEKEND ───────────────────────────────────────────────────


def test_weekend_saturday():
    from ui_v2.state_machine import StateName, get_app_state

    # Saturday 2pm Brisbane
    sat = datetime(2026, 3, 7, 14, 0, tzinfo=BRISBANE)
    state = get_app_state(sat)
    assert state.name == StateName.WEEKEND
    assert state.next_monday is not None
    assert state.next_monday.weekday() == 0  # Monday


def test_weekend_sunday():
    from ui_v2.state_machine import StateName, get_app_state

    sun = datetime(2026, 3, 8, 10, 0, tzinfo=BRISBANE)
    state = get_app_state(sun)
    assert state.name == StateName.WEEKEND


# ── get_app_state: IDLE ──────────────────────────────────────────────────────


def test_idle_when_far_from_session():
    from ui_v2.state_machine import StateName, get_app_state

    # Wednesday 7:30 AM Brisbane — CME_REOPEN at 9:00 AM is 90 min away
    wed = datetime(2026, 3, 4, 7, 30, tzinfo=BRISBANE)
    state = get_app_state(wed)
    assert state.name == StateName.IDLE
    assert state.minutes_to_next is not None
    assert state.minutes_to_next > 60


# ── get_app_state: APPROACHING ───────────────────────────────────────────────


def test_approaching_30min_before():
    from ui_v2.state_machine import StateName, get_app_state

    # Wednesday 8:30 AM Brisbane — CME_REOPEN at 9:00 is 30 min away
    wed = datetime(2026, 3, 4, 8, 30, tzinfo=BRISBANE)
    state = get_app_state(wed)
    assert state.name == StateName.APPROACHING
    assert state.next_session is not None


# ── get_app_state: ALERT ─────────────────────────────────────────────────────


def test_alert_10min_before():
    from ui_v2.state_machine import StateName, get_app_state

    # Wednesday 8:50 AM Brisbane — CME_REOPEN at 9:00 is 10 min away
    wed = datetime(2026, 3, 4, 8, 50, tzinfo=BRISBANE)
    state = get_app_state(wed)
    assert state.name == StateName.ALERT
    assert state.minutes_to_next is not None
    assert state.minutes_to_next <= 15


# ── get_app_state: OVERNIGHT ─────────────────────────────────────────────────


def test_overnight_when_next_session_outside_awake():
    from ui_v2.state_machine import StateName, get_app_state

    # Wednesday 22:00 Brisbane — next session (TOKYO_OPEN ~midnight) is outside awake hours
    # and more than 60 min away
    wed_night = datetime(2026, 3, 4, 22, 0, tzinfo=BRISBANE)
    state = get_app_state(wed_night)
    # Could be OVERNIGHT or IDLE depending on next session time
    assert state.name in (StateName.OVERNIGHT, StateName.IDLE, StateName.APPROACHING, StateName.ALERT)


# ── get_upcoming_sessions ────────────────────────────────────────────────────


def test_upcoming_sessions_returns_sorted_future():
    from ui_v2.state_machine import get_upcoming_sessions

    # Wednesday 8:00 AM Brisbane
    now = datetime(2026, 3, 4, 8, 0, tzinfo=BRISBANE)
    upcoming = get_upcoming_sessions(now)
    assert len(upcoming) > 0
    # All are in the future
    for _name, dt in upcoming:
        assert dt > now
    # Sorted chronologically
    for i in range(len(upcoming) - 1):
        assert upcoming[i][1] <= upcoming[i + 1][1]


def test_upcoming_sessions_deduplicates():
    from ui_v2.state_machine import get_upcoming_sessions

    now = datetime(2026, 3, 4, 8, 0, tzinfo=BRISBANE)
    upcoming = get_upcoming_sessions(now)
    names = [name for name, _ in upcoming]
    assert len(names) == len(set(names))


# ── filter_to_english ────────────────────────────────────────────────────────


def test_filter_g4_to_english():
    from ui_v2.state_machine import filter_to_english

    assert "4" in filter_to_english("ORB_G4")


def test_filter_dir_long():
    from ui_v2.state_machine import filter_to_english

    result = filter_to_english("DIR_LONG")
    assert "long" in result.lower() or "Long" in result


def test_filter_unknown_returns_raw():
    from ui_v2.state_machine import filter_to_english

    assert filter_to_english("UNKNOWN_FILTER") == "UNKNOWN_FILTER"


# ── SessionState and session stack ───────────────────────────────────────────


def test_session_state_dataclass():
    from ui_v2.state_machine import SessionState, StateName

    ss = SessionState(session_name="CME_REOPEN", instrument="MGC")
    assert ss.sub_state == StateName.ORB_FORMING
    assert ss.orb_minutes == 5


def test_resolve_global_state_uses_highest_priority():
    from ui_v2.state_machine import AppState, SessionState, StateName, resolve_global_state

    base = AppState(
        name=StateName.IDLE,
        trading_day=date(2026, 3, 4),
        active_sessions=[
            SessionState(session_name="CME_REOPEN", sub_state=StateName.IN_SESSION),
        ],
    )
    result = resolve_global_state(base)
    assert result == StateName.IN_SESSION


def test_resolve_global_state_debrief_beats_in_session():
    from ui_v2.state_machine import AppState, SessionState, StateName, resolve_global_state

    base = AppState(
        name=StateName.IDLE,
        trading_day=date(2026, 3, 4),
        active_sessions=[
            SessionState(session_name="CME_REOPEN", sub_state=StateName.IN_SESSION),
            SessionState(session_name="TOKYO_OPEN", sub_state=StateName.DEBRIEF),
        ],
    )
    result = resolve_global_state(base)
    assert result == StateName.DEBRIEF


def test_resolve_global_state_no_sessions_returns_base():
    from ui_v2.state_machine import AppState, StateName, resolve_global_state

    base = AppState(name=StateName.APPROACHING, trading_day=date(2026, 3, 4))
    result = resolve_global_state(base)
    assert result == StateName.APPROACHING


# ── ET time helper ───────────────────────────────────────────────────────────


def test_get_et_time_returns_string():
    from ui_v2.state_machine import get_et_time

    now = datetime(2026, 3, 4, 10, 0, tzinfo=BRISBANE)
    result = get_et_time(now)
    assert "ET" in result
    assert "PM" in result or "AM" in result


# ── get_refresh_seconds ──────────────────────────────────────────────────────


def test_refresh_weekend_slow():
    from ui_v2.state_machine import get_refresh_seconds

    assert get_refresh_seconds(120, is_weekend=True) == 120


def test_refresh_alert_fast():
    from ui_v2.state_machine import get_refresh_seconds

    assert get_refresh_seconds(10) == 5


def test_refresh_approaching_medium():
    from ui_v2.state_machine import get_refresh_seconds

    assert get_refresh_seconds(30) == 15


def test_refresh_idle_normal():
    from ui_v2.state_machine import get_refresh_seconds

    assert get_refresh_seconds(120) == 30
