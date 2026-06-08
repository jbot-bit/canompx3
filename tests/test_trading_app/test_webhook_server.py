"""Tests for webhook server route logic and lifespan validation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from trading_app.prop_profiles import AccountProfile


def _nq_mini_profile() -> AccountProfile:
    """Real AccountProfile that exercises the __post_init__ NQ-mini validator.

    SimpleNamespace bypasses the dataclass validator (cross-map keys, divisor
    int>=1, source-in-ASSET_CONFIGS, target-in-COST_SPECS). Use this helper
    so webhook tests probe the same construction path live profiles take.
    """
    return AccountProfile(
        profile_id="test_nq_mini",
        firm="test_firm",
        account_size=50_000,
        is_express_funded=True,
        execution_symbol_map={"MNQ": "NQ"},
        execution_qty_divisor={"MNQ": 4},
    )


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


async def test_profile_execution_map_applies_qty_cap_after_division():
    """Mapped broker qty, not strategy qty, should be checked against MAX_ORDER_QTY."""
    ws = _load_ws()
    ws.MAX_ORDER_QTY = 5
    profile = _nq_mini_profile()

    with (
        patch("trading_app.live.webhook_server.asyncio.get_running_loop", return_value=_ImmediateLoop()),
        patch("trading_app.live.webhook_server._get_account_profile", return_value=profile),
        patch("trading_app.live.webhook_server._get_contract", side_effect=lambda instrument: f"{instrument}M6"),
        patch("trading_app.live.webhook_server._place_order", return_value=12345) as place_order,
    ):
        resp = await _run_trade(ws, {**_ENTRY_PAYLOAD, "instrument": "MNQ", "qty": 20})

    assert resp.status == "submitted"
    place_order.assert_called_once()
    assert place_order.call_args.args[2] == 5


async def test_profile_execution_map_resolves_contract_and_divides_qty():
    """Configured webhook profile routes MNQ alerts to NQ broker contracts."""
    ws = _load_ws()
    ws.WEBHOOK_PROFILE_ID = "test_nq_mini"
    submitted = {}

    class Router:
        def __init__(self, **_kwargs):
            pass

        def build_order_spec(self, **kwargs):
            submitted.update(kwargs)
            return {"spec": kwargs}

        def submit(self, _spec):
            return {"order_id": 123}

    profile = _nq_mini_profile()

    with (
        patch("trading_app.live.webhook_server.asyncio.get_running_loop", return_value=_ImmediateLoop()),
        patch("trading_app.live.webhook_server._get_account_profile", return_value=profile),
        patch("trading_app.live.webhook_server._get_contract", side_effect=lambda instrument: f"{instrument}M6"),
        patch(
            "trading_app.live.webhook_server._get_broker",
            return_value={"router_class": Router},
        ),
        patch("trading_app.live.webhook_server._get_account_id", return_value=1),
        patch("trading_app.live.webhook_server._get_auth", return_value=object()),
    ):
        resp = await _run_trade(ws, {**_ENTRY_PAYLOAD, "instrument": "MNQ", "qty": 4})

    assert resp.contract == "NQM6"
    assert resp.qty == 1
    assert submitted["symbol"] == "NQM6"
    assert submitted["qty"] == 1


async def test_profile_execution_map_rejects_fractional_webhook_qty():
    """Webhook execution mapping fails closed when qty/divisor is not an integer lot."""
    ws = _load_ws()
    profile = _nq_mini_profile()

    with patch("trading_app.live.webhook_server._get_account_profile", return_value=profile):
        with pytest.raises(HTTPException) as exc:
            await _run_trade(ws, {**_ENTRY_PAYLOAD, "instrument": "MNQ", "qty": 3})

    assert exc.value.status_code == 400
    assert "not divisible" in exc.value.detail


async def test_concurrent_duplicate_entries_only_one_submits():
    """REGRESSION (capital-review Iter 6 fix): concurrent identical entry alerts
    must NOT both submit. The reserve-before-await fix (webhook_server.py step 5b)
    increments the position counter and plants an in-flight dedup placeholder
    SYNCHRONOUSLY — before the first `await` — so the second concurrent request
    hits the placeholder and is deduplicated. Exactly one order submits and the
    MAX_OPEN_POSITIONS=1 cap holds.

    This is the flipped form of the original TOCTOU characterization test, which
    pinned the unsafe behavior (both submit, counter==2). Operator-approved
    behavior change on the order-trigger path (AskUserQuestion, 2026-06-07).
    """
    ws = _load_ws()
    ws.MAX_OPEN_POSITIONS = 1
    ws.DEDUP_WINDOW = 10.0
    ws.WEBHOOK_PROFILE_ID = ""  # no profile mapping — plain MGC entry

    submit_calls = 0

    class _ConcurrentLoop:
        async def run_in_executor(self, _executor, func, *args):
            nonlocal submit_calls
            if func is ws._get_contract:
                return "MGCM5"
            submit_calls += 1
            # Yield control so the SECOND gathered coroutine runs its synchronous
            # guard section while this one is "awaiting" the order. The fix's
            # reserve-before-await means the second request hits the in-flight
            # placeholder and never reaches this executor — so we must NOT block
            # on a 2-party barrier here (only one party ever arrives → deadlock).
            await asyncio.sleep(0.01)
            return 12345

    with patch(
        "trading_app.live.webhook_server.asyncio.get_running_loop",
        return_value=_ConcurrentLoop(),
    ):
        results = await asyncio.gather(
            _run_trade(ws, _ENTRY_PAYLOAD),
            _run_trade(ws, _ENTRY_PAYLOAD),
        )

    # SAFE: only ONE order submitted; the second concurrent request deduplicated;
    # the position counter respects the cap=1.
    assert submit_calls == 1, "reserve-before-await must let only one concurrent entry submit"
    statuses = sorted(r.status for r in results)
    assert statuses == ["deduplicated_in_flight", "submitted"], (
        f"expected one submitted + one in-flight dedup, got {statuses}"
    )
    assert ws._OPEN_POSITIONS["MGC"] == 1, "position counter must respect cap=1"


async def test_failed_order_rolls_back_reservation():
    """REGRESSION (capital-review Iter 6 fix): if the awaited order placement
    fails AFTER the pre-await reservation, the position counter and dedup
    placeholder must roll back — so a legitimate retry isn't permanently blocked
    and the counter doesn't leak a phantom open position.
    """
    ws = _load_ws()
    ws.MAX_OPEN_POSITIONS = 1
    ws.DEDUP_WINDOW = 10.0
    ws.WEBHOOK_PROFILE_ID = ""

    class _FailingLoop:
        async def run_in_executor(self, _executor, func, *args):
            if func is ws._get_contract:
                return "MGCM5"
            raise RuntimeError("broker rejected order")

    with patch(
        "trading_app.live.webhook_server.asyncio.get_running_loop",
        return_value=_FailingLoop(),
    ):
        with pytest.raises(HTTPException) as exc:
            await _run_trade(ws, _ENTRY_PAYLOAD)
    assert exc.value.status_code == 500

    # Rolled back: counter back to 0, no lingering in-flight dedup placeholder.
    assert ws._OPEN_POSITIONS.get("MGC", 0) == 0, "counter must roll back after a failed order"
    assert ws._check_dedup(ws.TradeRequest(**_ENTRY_PAYLOAD)) is None, (
        "in-flight placeholder must be released so a retry is not blocked"
    )
