"""Tests for webhook server — auth, dedup, rate limiting."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest


def test_webhook_rejects_wrong_secret():
    """Requests with wrong secret get 403."""
    # Patch the module-level secret so lifespan won't raise
    with patch.dict("os.environ", {"WEBHOOK_SECRET": "correct-secret"}):
        import trading_app.live.webhook_server as ws

        ws.WEBHOOK_SECRET = "correct-secret"

        from fastapi.testclient import TestClient

        client = TestClient(ws.app, raise_server_exceptions=False)

        resp = client.post(
            "/trade",
            json={
                "instrument": "MGC",
                "direction": "long",
                "action": "entry",
                "qty": 1,
                "secret": "wrong-secret",
            },
        )
        assert resp.status_code == 403, f"Expected 403, got {resp.status_code}: {resp.text}"


def test_webhook_health_endpoint():
    """Health endpoint returns 200."""
    with patch.dict("os.environ", {"WEBHOOK_SECRET": "test-secret"}):
        import trading_app.live.webhook_server as ws

        ws.WEBHOOK_SECRET = "test-secret"

        from fastapi.testclient import TestClient

        client = TestClient(ws.app, raise_server_exceptions=False)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


def test_webhook_lifespan_blocks_empty_secret():
    """Server refuses to start when WEBHOOK_SECRET is empty."""
    with patch.dict("os.environ", {"WEBHOOK_SECRET": ""}):
        import trading_app.live.webhook_server as ws

        ws.WEBHOOK_SECRET = ""

        from fastapi.testclient import TestClient

        with pytest.raises(RuntimeError, match="WEBHOOK_SECRET env var is required"):
            with TestClient(ws.app):
                pass  # should never reach here


# ── Dedup tests ──────────────────────────────────────────────────────────────


def _make_client():
    """Set up a TestClient with auth and dedup enabled, cache cleared."""
    with patch.dict("os.environ", {"WEBHOOK_SECRET": "s3cret"}):
        import trading_app.live.webhook_server as ws

        ws.WEBHOOK_SECRET = "s3cret"
        ws.DEDUP_WINDOW = 10.0
        ws.MAX_OPEN_POSITIONS = 10  # high limit so dedup/rate tests aren't affected
        ws._DEDUP_CACHE.clear()
        ws._ORDER_TIMESTAMPS.clear()
        ws._OPEN_POSITIONS.clear()

        from fastapi.testclient import TestClient

        return TestClient(ws.app, raise_server_exceptions=False), ws


_ENTRY_PAYLOAD = {
    "instrument": "MGC",
    "direction": "long",
    "action": "entry",
    "qty": 1,
    "secret": "s3cret",
}


def test_dedup_blocks_duplicate_within_window():
    """Second identical request within 10s returns deduplicated, not a new order."""
    client, _ = _make_client()

    with (
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        r1 = client.post("/trade", json=_ENTRY_PAYLOAD)
        assert r1.status_code == 200
        assert r1.json()["status"] == "submitted"

        r2 = client.post("/trade", json=_ENTRY_PAYLOAD)
        assert r2.status_code == 200
        assert r2.json()["status"] == "deduplicated"


def test_dedup_allows_after_window_expires():
    """Request after dedup window expires is treated as new."""
    client, ws = _make_client()

    with (
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        r1 = client.post("/trade", json=_ENTRY_PAYLOAD)
        assert r1.json()["status"] == "submitted"

        # Expire the cache entry by backdating it
        for key in ws._DEDUP_CACHE:
            ts, resp = ws._DEDUP_CACHE[key]
            ws._DEDUP_CACHE[key] = (ts - 20.0, resp)  # 20s ago → well past 10s window

        r2 = client.post("/trade", json=_ENTRY_PAYLOAD)
        assert r2.json()["status"] == "submitted"


def test_dedup_different_key_not_blocked():
    """Entry then exit for same instrument are different keys — both go through."""
    client, _ = _make_client()

    exit_payload = {**_ENTRY_PAYLOAD, "action": "exit"}

    with (
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        r1 = client.post("/trade", json=_ENTRY_PAYLOAD)
        assert r1.json()["status"] == "submitted"

        r2 = client.post("/trade", json=exit_payload)
        assert r2.json()["status"] == "submitted"


# ── Position limit tests ────────────────────────────────────────────────


def test_entry_blocked_when_position_open():
    """Second entry for same instrument blocked when at position limit."""
    client, ws = _make_client()
    ws.MAX_OPEN_POSITIONS = 1
    ws._OPEN_POSITIONS.clear()
    ws._OPEN_POSITIONS["MGC"] = 1

    with (
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        resp = client.post("/trade", json=_ENTRY_PAYLOAD)
        assert resp.status_code == 429
        assert "position limit" in resp.json()["detail"].lower()


def test_exit_allowed_when_position_open():
    """Exit is never blocked by position limit."""
    client, ws = _make_client()
    ws._OPEN_POSITIONS.clear()
    ws._OPEN_POSITIONS["MGC"] = 1

    exit_payload = {**_ENTRY_PAYLOAD, "action": "exit"}

    with (
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        resp = client.post("/trade", json=exit_payload)
        assert resp.status_code == 200


# ── Instrument allowlist tests ─────────────────────────────────────────


def test_unknown_instrument_rejected():
    """Instruments not in ACTIVE_ORB_INSTRUMENTS get 400."""
    client, _ = _make_client()
    bad_payload = {**_ENTRY_PAYLOAD, "instrument": "FAKE"}

    resp = client.post("/trade", json=bad_payload)
    assert resp.status_code == 400
    assert "unknown instrument" in resp.json()["detail"].lower()


def test_known_instrument_accepted():
    """Active instruments pass the allowlist check."""
    client, _ = _make_client()
    # MGC is in ACTIVE_ORB_INSTRUMENTS
    with (
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        resp = client.post("/trade", json=_ENTRY_PAYLOAD)
        assert resp.status_code == 200


# ── Qty cap tests ──────────────────────────────────────────────────────


def test_qty_exceeds_max_rejected():
    """Orders with qty > MAX_ORDER_QTY get 400."""
    client, ws = _make_client()
    ws.MAX_ORDER_QTY = 5
    big_payload = {**_ENTRY_PAYLOAD, "qty": 10}

    resp = client.post("/trade", json=big_payload)
    assert resp.status_code == 400
    assert "exceeds max" in resp.json()["detail"].lower()


def test_qty_within_max_accepted():
    """Orders with qty <= MAX_ORDER_QTY pass."""
    client, ws = _make_client()
    ws.MAX_ORDER_QTY = 5
    ok_payload = {**_ENTRY_PAYLOAD, "qty": 3}

    with (
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        resp = client.post("/trade", json=ok_payload)
        assert resp.status_code == 200
