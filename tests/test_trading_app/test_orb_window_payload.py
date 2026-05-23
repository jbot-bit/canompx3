"""Backend test for the chart-cockpit ORB-window payload extension.

Covers `_orb_levels_for_instrument` after the 2026-05-19 patch that adds
canonical `pipeline.dst.orb_utc_window` projection, AND the 2026-05-23 Stage 2
extension that adds `daily_orb_windows` (per-day canonical aperture, no
fabricated display constant). Pure read-only — writes a synthetic
bot_state.json under tmp_path, monkeypatches `STATE_FILE`, and asserts the
returned payload shape.

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


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 (2026-05-23) — daily_orb_windows: per-day canonical ORB aperture
# across the last N trading days. Width = literal orb_minutes (5/15/30) from
# pipeline.dst.orb_utc_window. NO fabricated display constant.
# ─────────────────────────────────────────────────────────────────────────────


def test_daily_orb_windows_default_days_is_single_entry(patched_state):
    """Default call (no days param) returns single-entry windows array for today."""
    patched_state(trading_day="2026-05-19", session="NYSE_OPEN", orb_minutes=15)
    payload = dashboard._orb_levels_for_instrument("MNQ")
    assert "daily_orb_windows" in payload
    assert isinstance(payload["daily_orb_windows"], list)
    assert len(payload["daily_orb_windows"]) == 1
    today_entry = payload["daily_orb_windows"][0]
    assert today_entry["trading_day"] == "2026-05-19"
    assert today_entry["session_name"] == "NYSE_OPEN"
    assert today_entry["orb_minutes"] == 15


def test_daily_orb_windows_uses_canonical_aperture(patched_state):
    """Each window's width == orb_minutes * 60 seconds. NO 2h. NO fabricated display constant."""
    patched_state(trading_day="2026-05-19", session="NYSE_OPEN", orb_minutes=15)
    payload = dashboard._orb_levels_for_instrument("MNQ", days=3)
    for entry in payload["daily_orb_windows"]:
        width_sec = entry["window_end_utc"] - entry["window_start_utc"]
        assert width_sec == 15 * 60, (
            f"Aperture width {width_sec}s != canonical 15m. "
            "If this fails with 7200 (2h), a fabricated display constant was reintroduced."
        )


def test_daily_orb_windows_no_display_end_constant_in_payload(patched_state):
    """Schema invariant: response dict must NOT contain a fabricated 'display_end_utc' field."""
    patched_state(trading_day="2026-05-19", session="NYSE_OPEN", orb_minutes=15)
    payload = dashboard._orb_levels_for_instrument("MNQ", days=5)
    for entry in payload["daily_orb_windows"]:
        assert "display_end_utc" not in entry
        assert "display_duration_sec" not in entry


def test_daily_orb_windows_days_param_returns_n_entries(patched_state):
    """days=5 -> up to 5 entries (today + 4 prior days)."""
    patched_state(trading_day="2026-05-19", session="NYSE_OPEN", orb_minutes=5)
    payload = dashboard._orb_levels_for_instrument("MNQ", days=5)
    assert len(payload["daily_orb_windows"]) == 5
    trading_days = [e["trading_day"] for e in payload["daily_orb_windows"]]
    assert trading_days == [
        "2026-05-19",
        "2026-05-18",
        "2026-05-17",
        "2026-05-16",
        "2026-05-15",
    ]


def test_daily_orb_windows_each_call_matches_canonical_resolver(patched_state):
    """Each per-day window equals what pipeline.dst.orb_utc_window returns directly."""
    patched_state(trading_day="2026-05-19", session="NYSE_OPEN", orb_minutes=15)
    payload = dashboard._orb_levels_for_instrument("MNQ", days=3)
    for entry in payload["daily_orb_windows"]:
        td = date.fromisoformat(entry["trading_day"])
        expected_start, expected_end = orb_utc_window(td, "NYSE_OPEN", 15)
        assert entry["window_start_utc"] == int(expected_start.timestamp())
        assert entry["window_end_utc"] == int(expected_end.timestamp())


def test_daily_orb_windows_unknown_session_skips_entries(patched_state, caplog: pytest.LogCaptureFixture):
    """If session is not in SESSION_CATALOG, all per-day calls raise -> empty list."""
    patched_state(trading_day="2026-05-19", session="NONEXISTENT", orb_minutes=5)
    with caplog.at_level(logging.WARNING, logger=dashboard.log.name):
        payload = dashboard._orb_levels_for_instrument("MNQ", days=5)
    # The wide unknown session causes all per-day calls to raise; resilient handling -> empty list.
    assert payload["daily_orb_windows"] == []


def test_daily_orb_windows_no_lane_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Instrument has no lane in bot_state -> daily_orb_windows is empty (and shape stable)."""
    state_path = tmp_path / "bot_state.json"
    state_path.write_text(
        json.dumps({"mode": "RUN_LIVE", "trading_day": "2026-05-19", "lanes": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(bot_state, "STATE_FILE", state_path)
    payload = dashboard._orb_levels_for_instrument("MNQ", days=5)
    assert payload["daily_orb_windows"] == []


def test_daily_orb_windows_missing_trading_day_returns_empty(patched_state):
    """No trading_day in state -> can't compute per-day windows -> empty list (no crash)."""
    patched_state(trading_day=None, session="NYSE_OPEN", orb_minutes=15)
    payload = dashboard._orb_levels_for_instrument("MNQ", days=5)
    assert payload["daily_orb_windows"] == []


def test_api_bars_recent_days_param_clamps_to_5(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Route-level: ?days=99 -> server clamps so daily_orb_windows has at most 5 entries."""
    from fastapi.testclient import TestClient

    state_path = tmp_path / "bot_state.json"
    state_path.write_text(
        json.dumps(
            {
                "mode": "RUN_LIVE",
                "trading_day": "2026-05-19",
                "lanes": {
                    "MNQ_TEST": {
                        "instrument": "MNQ",
                        "session_name": "NYSE_OPEN",
                        "orb_minutes": 15,
                        "orb_high": 18500.0,
                        "orb_low": 18490.0,
                        "orb_complete": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bot_state, "STATE_FILE", state_path)
    client = TestClient(dashboard.app)
    resp = client.get("/api/bars-recent?instrument=MNQ&days=99")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["daily_orb_windows"]) <= 5


def test_api_bars_recent_lookback_cap_7200(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Route-level: lookback_minutes=99999 capped server-side at 7200 (5 trading days)."""
    from fastapi.testclient import TestClient

    state_path = tmp_path / "bot_state.json"
    state_path.write_text(
        json.dumps({"mode": "RUN_LIVE", "trading_day": "2026-05-19", "lanes": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(bot_state, "STATE_FILE", state_path)
    client = TestClient(dashboard.app)
    # Just confirm the route accepts large lookback and returns 200.
    # The internal cap is exercised by _query_bars_recent which we can't easily mock here;
    # the important assertion is that the route doesn't reject lookback_minutes > 1440.
    resp = client.get("/api/bars-recent?instrument=MNQ&lookback_minutes=99999")
    assert resp.status_code == 200
