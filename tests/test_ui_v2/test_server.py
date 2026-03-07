"""Tests for ui_v2/server.py — FastAPI TestClient endpoint tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ui_v2.server import app

client = TestClient(app)


# ── GET / ────────────────────────────────────────────────────────────────────


def test_serve_index_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "html" in response.headers.get("content-type", "").lower()


# ── GET /api/state ───────────────────────────────────────────────────────────


def test_get_state_returns_state():
    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "bris_time" in data
    assert "et_time" in data
    assert "refresh_seconds" in data
    assert "cooling_active" in data


def test_state_has_valid_state_name():
    response = client.get("/api/state")
    data = response.json()
    valid_states = {"WEEKEND", "OVERNIGHT", "IDLE", "APPROACHING", "ALERT", "ORB_FORMING", "IN_SESSION", "DEBRIEF"}
    assert data["name"] in valid_states


# ── GET /api/briefings ──────────────────────────────────────────────────────


def test_get_briefings_returns_list():
    response = client.get("/api/briefings")
    assert response.status_code == 200
    data = response.json()
    assert "briefings" in data
    assert isinstance(data["briefings"], list)


# ── GET /api/session-history/{name} ─────────────────────────────────────────


def test_session_history_returns_structure():
    response = client.get("/api/session-history/CME_REOPEN")
    assert response.status_code == 200
    data = response.json()
    assert data["session"] == "CME_REOPEN"
    assert "history" in data


def test_session_history_respects_limit():
    response = client.get("/api/session-history/CME_REOPEN?limit=5")
    assert response.status_code == 200


# ── GET /api/day-summary ────────────────────────────────────────────────────


def test_day_summary_returns_structure():
    response = client.get("/api/day-summary")
    assert response.status_code == 200
    data = response.json()
    assert "trading_day" in data
    assert "sessions" in data


# ── GET /api/rolling-pnl ────────────────────────────────────────────────────


def test_rolling_pnl_returns_structure():
    response = client.get("/api/rolling-pnl")
    assert response.status_code == 200
    data = response.json()
    assert "daily" in data
    assert "week_r" in data
    assert "month_r" in data


# ── GET /api/overnight-recap ────────────────────────────────────────────────


def test_overnight_recap_returns_structure():
    response = client.get("/api/overnight-recap")
    assert response.status_code == 200
    data = response.json()
    assert "trading_day" in data
    assert "recap" in data


# ── GET /api/fitness ─────────────────────────────────────────────────────────


def test_fitness_returns_structure():
    response = client.get("/api/fitness")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data


# ── GET /api/adherence-stats/{name} ─────────────────────────────────────────


def test_adherence_stats_returns_structure():
    response = client.get("/api/adherence-stats/CME_REOPEN")
    assert response.status_code == 200
    data = response.json()
    assert data["session"] == "CME_REOPEN"
    assert "stats" in data


# ── GET /api/debrief/pending ────────────────────────────────────────────────


def test_pending_debriefs():
    response = client.get("/api/debrief/pending")
    assert response.status_code == 200
    data = response.json()
    assert "pending" in data


# ── POST /api/debrief ───────────────────────────────────────────────────────


def test_submit_debrief_valid():
    payload = {
        "strategy_id": "MGC_CME_REOPEN_E2_1",
        "signal_exit_ts": "2026-03-04T10:00:00",
        "adherence": "followed",
        "pnl_r": 1.5,
    }
    response = client.post("/api/debrief", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_submit_debrief_invalid_adherence():
    payload = {
        "strategy_id": "test",
        "signal_exit_ts": "2026-03-04T10:00:00",
        "adherence": "invalid_value",
    }
    response = client.post("/api/debrief", json=payload)
    assert response.status_code == 400


def test_submit_debrief_triggers_cooling_on_loss():
    from ui_v2.server import _cooling_state

    _cooling_state.clear()
    payload = {
        "strategy_id": "MGC_CME_REOPEN_E2_1",
        "signal_exit_ts": "2026-03-04T10:00:00",
        "adherence": "followed",
        "pnl_r": -1.0,
    }
    response = client.post("/api/debrief", json=payload)
    assert response.status_code == 200
    assert response.json()["cooling_active"] is True


# ── POST /api/trade-log ─────────────────────────────────────────────────────


def test_log_trade_entry(tmp_path, monkeypatch):
    # Point SIGNALS_PATH to tmp
    monkeypatch.setattr("ui_v2.server.SIGNALS_PATH", tmp_path / "signals.jsonl")
    payload = {
        "instrument": "MGC",
        "direction": "long",
        "action": "entry",
        "price": 2450.5,
        "session": "CME_REOPEN",
    }
    response = client.post("/api/trade-log", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert (tmp_path / "signals.jsonl").exists()


# ── POST /api/commitment ────────────────────────────────────────────────────


def test_record_commitment():
    payload = {"items": {"chart_open": True, "order_ready": True, "risk_sized": False}}
    response = client.post("/api/commitment", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["items"]["chart_open"] is True
    assert data["items"]["risk_sized"] is False


# ── POST /api/cooling/override ───────────────────────────────────────────────


def test_cooling_override_when_active():
    # Trigger cooling first
    from ui_v2.discipline_api import trigger_cooling
    from ui_v2.server import _cooling_state

    _cooling_state.clear()
    trigger_cooling(_cooling_state, pnl_r=-1.0, consecutive_losses=1, session_pnl_r=-1.0)

    response = client.post("/api/cooling/override", json={})
    assert response.status_code == 200
    assert response.json()["cooling_active"] is False


def test_cooling_override_when_not_active():
    from ui_v2.server import _cooling_state

    _cooling_state.clear()
    response = client.post("/api/cooling/override", json={})
    assert response.status_code == 400


# ── POST /api/session/start and stop (placeholder) ──────────────────────────


def test_session_start_placeholder():
    response = client.post("/api/session/start")
    assert response.status_code == 200
    assert "not_implemented" in response.json()["status"]


def test_session_stop_placeholder():
    response = client.post("/api/session/stop")
    assert response.status_code == 200
    assert "not_implemented" in response.json()["status"]
