"""Tests for webhook server route logic and lifespan validation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException


def _load_ws(secret: str = "s3cret"):
    """Import webhook server with a test secret and reset mutable state."""
    with patch.dict("os.environ", {"WEBHOOK_SECRET": secret}):
        import trading_app.live.webhook_server as ws

    ws.WEBHOOK_SECRET = secret
    ws.DEDUP_WINDOW = 10.0
    ws.MAX_OPEN_POSITIONS = 10
    ws.MAX_ORDER_QTY = 5
    ws._DEDUP_CACHE.clear()
    ws._ORDER_TIMESTAMPS.clear()
    ws._OPEN_POSITIONS.clear()
    ws._contract_cache.clear()
    ws._broker_components = None
    ws._auth = None
    ws._account_id = None
    return ws


def _request_client():
    return SimpleNamespace(client="testclient")


class _ImmediateLoop:
    async def run_in_executor(self, _executor, func, *args):
        return func(*args)


async def _run_trade(ws, payload: dict):
    req = ws.TradeRequest(**payload)
    return await ws.trade(req, _request_client())


_ENTRY_PAYLOAD = {
    "instrument": "MGC",
    "direction": "long",
    "action": "entry",
    "qty": 1,
    "secret": "s3cret",
}


async def test_webhook_rejects_wrong_secret():
    """Requests with wrong secret get 403."""
    ws = _load_ws(secret="correct-secret")

    with pytest.raises(HTTPException) as exc:
        await _run_trade(
            ws,
            {
                "instrument": "MGC",
                "direction": "long",
                "action": "entry",
                "qty": 1,
                "secret": "wrong-secret",
            },
        )

    assert exc.value.status_code == 403
    assert "invalid webhook secret" in exc.value.detail.lower()


async def test_webhook_health_endpoint():
    """Health endpoint returns 200-equivalent payload."""
    ws = _load_ws(secret="test-secret")

    data = await ws.health()
    assert data["status"] == "ok"


async def test_webhook_lifespan_blocks_empty_secret():
    """Server refuses to start when WEBHOOK_SECRET is empty."""
    ws = _load_ws(secret="")
    ws.WEBHOOK_SECRET = ""

    async def _enter():
        async with ws.lifespan(ws.app):
            pass

    with pytest.raises(RuntimeError, match="WEBHOOK_SECRET env var is required"):
        await _enter()


async def test_dedup_blocks_duplicate_within_window():
    """Second identical request within 10s returns deduplicated, not a new order."""
    ws = _load_ws()

    with (
        patch("trading_app.live.webhook_server.asyncio.get_running_loop", return_value=_ImmediateLoop()),
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        r1 = await _run_trade(ws, _ENTRY_PAYLOAD)
        assert r1.status == "submitted"

        r2 = await _run_trade(ws, _ENTRY_PAYLOAD)
        assert r2.status == "deduplicated"


async def test_dedup_allows_after_window_expires():
    """Request after dedup window expires is treated as new."""
    ws = _load_ws()

    with (
        patch("trading_app.live.webhook_server.asyncio.get_running_loop", return_value=_ImmediateLoop()),
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        r1 = await _run_trade(ws, _ENTRY_PAYLOAD)
        assert r1.status == "submitted"

        for key in ws._DEDUP_CACHE:
            ts, resp = ws._DEDUP_CACHE[key]
            ws._DEDUP_CACHE[key] = (ts - 20.0, resp)

        r2 = await _run_trade(ws, _ENTRY_PAYLOAD)
        assert r2.status == "submitted"


async def test_dedup_different_key_not_blocked():
    """Entry then exit for same instrument are different keys — both go through."""
    ws = _load_ws()
    exit_payload = {**_ENTRY_PAYLOAD, "action": "exit"}

    with (
        patch("trading_app.live.webhook_server.asyncio.get_running_loop", return_value=_ImmediateLoop()),
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        r1 = await _run_trade(ws, _ENTRY_PAYLOAD)
        assert r1.status == "submitted"

        r2 = await _run_trade(ws, exit_payload)
        assert r2.status == "submitted"


async def test_entry_blocked_when_position_open():
    """Second entry for same instrument blocked when at position limit."""
    ws = _load_ws()
    ws.MAX_OPEN_POSITIONS = 1
    ws._OPEN_POSITIONS["MGC"] = 1

    with pytest.raises(HTTPException) as exc:
        await _run_trade(ws, _ENTRY_PAYLOAD)

    assert exc.value.status_code == 429
    assert "position limit" in exc.value.detail.lower()


async def test_exit_allowed_when_position_open():
    """Exit is never blocked by position limit."""
    ws = _load_ws()
    ws._OPEN_POSITIONS["MGC"] = 1
    exit_payload = {**_ENTRY_PAYLOAD, "action": "exit"}

    with (
        patch("trading_app.live.webhook_server.asyncio.get_running_loop", return_value=_ImmediateLoop()),
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        resp = await _run_trade(ws, exit_payload)
        assert resp.status == "submitted"


async def test_unknown_instrument_rejected():
    """Instruments not in ACTIVE_ORB_INSTRUMENTS get 400."""
    ws = _load_ws()

    with pytest.raises(HTTPException) as exc:
        await _run_trade(ws, {**_ENTRY_PAYLOAD, "instrument": "FAKE"})

    assert exc.value.status_code == 400
    assert "unknown instrument" in exc.value.detail.lower()


async def test_known_instrument_accepted():
    """Active instruments pass the allowlist check."""
    ws = _load_ws()

    with (
        patch("trading_app.live.webhook_server.asyncio.get_running_loop", return_value=_ImmediateLoop()),
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        resp = await _run_trade(ws, _ENTRY_PAYLOAD)
        assert resp.status == "submitted"


async def test_qty_exceeds_max_rejected():
    """Orders with qty > MAX_ORDER_QTY get 400."""
    ws = _load_ws()
    ws.MAX_ORDER_QTY = 5

    with pytest.raises(HTTPException) as exc:
        await _run_trade(ws, {**_ENTRY_PAYLOAD, "qty": 10})

    assert exc.value.status_code == 400
    assert "exceeds max" in exc.value.detail.lower()


async def test_qty_within_max_accepted():
    """Orders with qty <= MAX_ORDER_QTY pass."""
    ws = _load_ws()
    ws.MAX_ORDER_QTY = 5

    with (
        patch("trading_app.live.webhook_server.asyncio.get_running_loop", return_value=_ImmediateLoop()),
        patch("trading_app.live.webhook_server._place_order", return_value=12345),
        patch("trading_app.live.webhook_server._get_contract", return_value="MGCM5"),
    ):
        resp = await _run_trade(ws, {**_ENTRY_PAYLOAD, "qty": 3})
        assert resp.status == "submitted"
