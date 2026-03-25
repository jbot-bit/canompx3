"""Tests for ProjectX order router 429 retry — IMPORTANT-2 and IMPORTANT-3.

IMPORTANT-2: cancel() retries on 429, raises RateLimitExhausted on exhaustion.
IMPORTANT-3: query_open_orders() and query_order_status() retry on 429,
             raise RateLimitExhausted (not return None/empty).
             verify_bracket_legs() propagates RateLimitExhausted.
"""

from unittest.mock import MagicMock, patch

import pytest


def _mock_resp(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status.return_value = None
    return resp


def _mock_429() -> MagicMock:
    return _mock_resp(status_code=429)


def _auth() -> MagicMock:
    auth = MagicMock()
    auth.headers.return_value = {"Authorization": "Bearer test"}
    return auth


def _router():
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    return ProjectXOrderRouter(account_id=123, auth=_auth(), tick_size=0.25)


# ---- IMPORTANT-2: cancel() 429 retry ----


@patch("trading_app.live.projectx.order_router.time.sleep")
@patch("trading_app.live.projectx.order_router.requests")
def test_cancel_retries_on_429_then_succeeds(mock_req, mock_sleep):
    """cancel() retries on 429 and succeeds on 3rd attempt."""
    mock_req.post.side_effect = [
        _mock_429(),
        _mock_429(),
        _mock_resp(200, {"success": True}),
    ]
    router = _router()
    router.cancel(order_id=42)  # should not raise
    assert mock_req.post.call_count == 3
    assert mock_sleep.call_count == 2


@patch("trading_app.live.projectx.order_router.time.sleep")
@patch("trading_app.live.projectx.order_router.requests")
def test_cancel_raises_rate_limit_exhausted_on_429(mock_req, mock_sleep):
    """cancel() raises RateLimitExhausted after all retries exhausted."""
    from trading_app.live.projectx.order_router import RateLimitExhausted

    mock_req.post.return_value = _mock_429()
    router = _router()
    with pytest.raises(RateLimitExhausted, match="429"):
        router.cancel(order_id=42)
    # 1 initial + 3 retries = 4 attempts
    assert mock_req.post.call_count == 4


# ---- IMPORTANT-3: query_order_status() 429 retry ----


@patch("trading_app.live.projectx.order_router.time.sleep")
@patch("trading_app.live.projectx.order_router.requests")
def test_query_order_status_retries_on_429(mock_req, mock_sleep):
    """query_order_status() retries on 429 and succeeds."""
    mock_req.get.side_effect = [
        _mock_429(),
        _mock_resp(200, {"status": 2, "filledPrice": 21000.0}),
    ]
    router = _router()
    result = router.query_order_status(order_id=99)
    assert result["status"] == "Filled"
    assert mock_req.get.call_count == 2


@patch("trading_app.live.projectx.order_router.time.sleep")
@patch("trading_app.live.projectx.order_router.requests")
def test_query_order_status_raises_on_429_exhaustion(mock_req, mock_sleep):
    """query_order_status() raises RateLimitExhausted — not silent return."""
    from trading_app.live.projectx.order_router import RateLimitExhausted

    mock_req.get.return_value = _mock_429()
    router = _router()
    with pytest.raises(RateLimitExhausted):
        router.query_order_status(order_id=99)


# ---- IMPORTANT-3: query_open_orders() 429 retry ----


@patch("trading_app.live.projectx.order_router.time.sleep")
@patch("trading_app.live.projectx.order_router.requests")
def test_query_open_orders_retries_on_429(mock_req, mock_sleep):
    """query_open_orders() retries on 429 and succeeds."""
    mock_req.post.side_effect = [
        _mock_429(),
        _mock_resp(200, {"orders": [{"id": 1, "type": 4}]}),
    ]
    router = _router()
    result = router.query_open_orders()
    assert len(result) == 1
    assert mock_req.post.call_count == 2


@patch("trading_app.live.projectx.order_router.time.sleep")
@patch("trading_app.live.projectx.order_router.requests")
def test_query_open_orders_raises_on_429_exhaustion(mock_req, mock_sleep):
    """query_open_orders() raises RateLimitExhausted — not empty list."""
    from trading_app.live.projectx.order_router import RateLimitExhausted

    mock_req.post.return_value = _mock_429()
    router = _router()
    with pytest.raises(RateLimitExhausted):
        router.query_open_orders()


# ---- IMPORTANT-3: verify_bracket_legs() propagates RateLimitExhausted ----


@patch("trading_app.live.projectx.order_router.time.sleep")
@patch("trading_app.live.projectx.order_router.requests")
def test_verify_bracket_legs_propagates_429(mock_req, mock_sleep):
    """verify_bracket_legs() must NOT return (None, None) on 429 — must raise."""
    from trading_app.live.projectx.order_router import RateLimitExhausted

    mock_req.post.return_value = _mock_429()
    router = _router()
    with pytest.raises(RateLimitExhausted):
        router.verify_bracket_legs(entry_order_id=100, contract_id="TEST")


@patch("trading_app.live.projectx.order_router.time.sleep")
@patch("trading_app.live.projectx.order_router.requests")
def test_cancel_bracket_orders_propagates_429(mock_req, mock_sleep):
    """cancel_bracket_orders() must NOT return 0 on 429 — must raise."""
    from trading_app.live.projectx.order_router import RateLimitExhausted

    mock_req.post.return_value = _mock_429()
    router = _router()
    with pytest.raises(RateLimitExhausted):
        router.cancel_bracket_orders(contract_id="TEST")


# ---- Non-429 errors still behave as before ----


@patch("trading_app.live.projectx.order_router.requests")
def test_verify_bracket_legs_returns_none_on_non_429_error(mock_req):
    """Non-429 errors in query_open_orders → (None, None) as before."""
    mock_req.post.side_effect = ConnectionError("network down")
    router = _router()
    sl, tp = router.verify_bracket_legs(entry_order_id=100, contract_id="TEST")
    assert sl is None
    assert tp is None
