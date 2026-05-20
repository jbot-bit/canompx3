"""Backend test for the chart-cockpit ORB-window payload extension.

Covers `_orb_levels_for_instrument` after the 2026-05-19 patch that adds
canonical `pipeline.dst.orb_utc_window` projection. Pure read-only — writes
a synthetic bot_state.json under tmp_path, monkeypatches `STATE_FILE`, and
asserts the returned payload shape.

Three sessions exercised so the test fails if any inline UTC offset math is
ever (re-)introduced:
  - NYSE_OPEN   — Brisbane evening session, ORB rolls into next Brisbane
                  calendar day (handled by `orb_utc_window` hour<9 branch).
  - LONDON_METALS — Brisbane afternoon, same calendar day.
  - COMEX_SETTLE — early-morning UTC, next Brisbane calendar day.

Mutation probes:
  - trading_day key removed -> window fields null, no exception.
  - session_name absent from SESSION_CATALOG -> window fields null,
    warning logged.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pytest

import trading_app.live.bot_dashboard as dashboard
import trading_app.live.bot_state as bot_state
from pipeline.dst import orb_utc_window


def _write_state(tmp_path: Path, *, trading_day: str | None, session: str, orb_minutes: int = 5) -> Path:
    state = {
        "mode": "RUN_LIVE",
        "instrument": "MNQ",
        "lanes": {
            "MNQ_TEST": {
                "lane_key": "MNQ_TEST",
                "strategy_id": "MNQ_TEST",
                "instrument": "MNQ",
                "session_name": session,
                "session_time_brisbane": "00:30",
                "orb_minutes": orb_minutes,
                "orb_high": 18500.0,
                "orb_low": 18490.0,
                "orb_complete": True,
                "orb_break_direction": "LONG",
            }
        },
    }
    if trading_day is not None:
        state["trading_day"] = trading_day
    path = tmp_path / "bot_state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


@pytest.fixture()
def patched_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def _setup(*, trading_day: str | None, session: str, orb_minutes: int = 5) -> None:
        path = _write_state(tmp_path, trading_day=trading_day, session=session, orb_minutes=orb_minutes)
        monkeypatch.setattr(bot_state, "STATE_FILE", path)

    return _setup


@pytest.mark.parametrize(
    "session,trading_day,orb_minutes",
    [
        ("NYSE_OPEN", "2026-05-19", 5),
        ("LONDON_METALS", "2026-05-19", 5),
        ("COMEX_SETTLE", "2026-05-19", 15),
    ],
)
def test_window_matches_canonical_resolver(patched_state, session, trading_day, orb_minutes):
    patched_state(trading_day=trading_day, session=session, orb_minutes=orb_minutes)
    payload = dashboard._orb_levels_for_instrument("MNQ")
    expected_start, expected_end = orb_utc_window(date.fromisoformat(trading_day), session, orb_minutes)
    assert payload["orb_window_start_utc"] == int(expected_start.timestamp())
    assert payload["orb_window_end_utc"] == int(expected_end.timestamp())
    assert payload["orb_high"] == 18500.0
    assert payload["orb_low"] == 18490.0
    assert payload["orb_complete"] is True
    assert payload["orb_minutes"] == orb_minutes
    assert payload["session_name"] == session
    assert payload["session_time_brisbane"] == "00:30"
    assert payload["orb_break_direction"] == "LONG"
    # All declared keys present — frontend never sees missing keys.
    for k in dashboard._ORB_PAYLOAD_NULL_KEYS:
        assert k in payload


def test_missing_trading_day_yields_null_window(patched_state):
    # No trading_day -> window fields null, levels still rendered.
    patched_state(trading_day=None, session="NYSE_OPEN")
    payload = dashboard._orb_levels_for_instrument("MNQ")
    assert payload["orb_window_start_utc"] is None
    assert payload["orb_window_end_utc"] is None
    assert payload["orb_high"] == 18500.0
    assert payload["orb_low"] == 18490.0


def test_bad_trading_day_logs_and_yields_null_window(patched_state, caplog: pytest.LogCaptureFixture):
    patched_state(trading_day="not-a-date", session="NYSE_OPEN")
    with caplog.at_level(logging.WARNING, logger=dashboard.log.name):
        payload = dashboard._orb_levels_for_instrument("MNQ")
    assert payload["orb_window_start_utc"] is None
    assert payload["orb_window_end_utc"] is None
    assert any("bad trading_day" in rec.message for rec in caplog.records)


def test_unknown_session_logs_and_yields_null_window(patched_state, caplog: pytest.LogCaptureFixture):
    patched_state(trading_day="2026-05-19", session="NONEXISTENT_SESSION")
    with caplog.at_level(logging.WARNING, logger=dashboard.log.name):
        payload = dashboard._orb_levels_for_instrument("MNQ")
    assert payload["orb_window_start_utc"] is None
    assert payload["orb_window_end_utc"] is None
    assert any("rejected" in rec.message for rec in caplog.records)


def test_empty_state_returns_stable_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # bot_state.json missing -> read_state returns {} -> empty payload with all keys present.
    missing = tmp_path / "does_not_exist.json"
    monkeypatch.setattr(bot_state, "STATE_FILE", missing)
    payload = dashboard._orb_levels_for_instrument("MNQ")
    for k in dashboard._ORB_PAYLOAD_NULL_KEYS:
        assert k in payload
    assert payload["orb_high"] is None
    assert payload["orb_complete"] is False
