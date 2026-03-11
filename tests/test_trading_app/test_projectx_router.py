"""Test ProjectX order routing — mocked HTTP."""

from unittest.mock import MagicMock, patch


def test_projectx_market_buy():
    mock_auth = MagicMock()
    mock_auth.headers.return_value = {"Authorization": "Bearer test"}

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "orderId": 9056,
        "success": True,
        "errorCode": 0,
        "errorMessage": None,
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_resp) as mock_post:
        from trading_app.live.projectx.order_router import ProjectXOrderRouter

        router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
        spec = router.build_order_spec(
            direction="long",
            entry_model="E1",
            entry_price=2950.0,
            symbol="CON.F.US.MGC.M26",
            qty=1,
        )
        result = router.submit(spec)
        assert result["order_id"] == 9056
        call_body = mock_post.call_args[1]["json"]
        assert call_body["accountId"] == 123
        assert call_body["type"] == 2  # Market
        assert call_body["side"] == 0  # Bid (buy)
        assert call_body["size"] == 1


def test_projectx_stop_sell():
    mock_auth = MagicMock()
    mock_auth.headers.return_value = {"Authorization": "Bearer test"}

    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
    spec = router.build_order_spec(
        direction="short",
        entry_model="E2",
        entry_price=2950.0,
        symbol="CON.F.US.MGC.M26",
        qty=1,
    )
    assert spec["type"] == 4  # Stop
    assert spec["side"] == 1  # Ask (sell)
    assert spec["stopPrice"] == 2950.0


def test_projectx_exit_reverses_direction():
    mock_auth = MagicMock()
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
    # Close a long = sell
    exit_spec = router.build_exit_spec("long", "CON.F.US.MGC.M26", qty=1)
    assert exit_spec["side"] == 1  # Ask (sell)
    assert exit_spec["type"] == 2  # Market

    # Close a short = buy
    exit_spec = router.build_exit_spec("short", "CON.F.US.MGC.M26", qty=1)
    assert exit_spec["side"] == 0  # Bid (buy)


def test_projectx_supports_brackets():
    mock_auth = MagicMock()
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
    assert router.supports_native_brackets() is True


def test_projectx_is_broker_router():
    mock_auth = MagicMock()
    from trading_app.live.broker_base import BrokerRouter
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
    assert isinstance(router, BrokerRouter)


def test_projectx_e3_blocked():
    import pytest

    mock_auth = MagicMock()
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=123, auth=mock_auth)
    with pytest.raises(ValueError, match="not supported"):
        router.build_order_spec("long", "E3", 2950.0, "CON.F.US.MGC.M26")


# ---- fill_price parsing (OR2) ----
# Validates the is-None guard introduced in iter 21 (OR1 fix).
# Key case: fill_price=0.0 must not fall through to fallback field.


def _px_mock_resp(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = data
    return resp


def _px_auth() -> MagicMock:
    auth = MagicMock()
    auth.headers.return_value = {"Authorization": "Bearer test"}
    return auth


def test_px_submit_fill_price_primary():
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    spec = {"accountId": 1, "contractId": "MGCM6", "type": 2, "side": 0, "size": 1}
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.post.return_value = _px_mock_resp({"success": True, "orderId": 99, "fillPrice": 2010.5})
        result = router.submit(spec)
    assert result["fill_price"] == 2010.5


def test_px_submit_fill_price_fallback():
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    spec = {"accountId": 1, "contractId": "MGCM6", "type": 2, "side": 0, "size": 1}
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.post.return_value = _px_mock_resp({"success": True, "orderId": 99, "averagePrice": 2010.5})
        result = router.submit(spec)
    assert result["fill_price"] == 2010.5


def test_px_submit_fill_price_none():
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    spec = {"accountId": 1, "contractId": "MGCM6", "type": 2, "side": 0, "size": 1}
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.post.return_value = _px_mock_resp({"success": True, "orderId": 99})
        result = router.submit(spec)
    assert result["fill_price"] is None


def test_px_submit_fill_price_zero_not_falsy():
    """fillPrice=0.0 must not fall through to averagePrice — guards the is-None fix."""
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    spec = {"accountId": 1, "contractId": "MGCM6", "type": 2, "side": 0, "size": 1}
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.post.return_value = _px_mock_resp(
            {"success": True, "orderId": 99, "fillPrice": 0.0, "averagePrice": 2010.5}
        )
        result = router.submit(spec)
    assert result["fill_price"] == 0.0


def test_px_query_fill_price_primary():
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.get.return_value = _px_mock_resp({"status": "Filled", "fillPrice": 2011.0})
        result = router.query_order_status(99)
    assert result["fill_price"] == 2011.0


def test_px_query_fill_price_fallback():
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.get.return_value = _px_mock_resp({"status": "Filled", "averagePrice": 2011.0})
        result = router.query_order_status(99)
    assert result["fill_price"] == 2011.0


def test_px_query_fill_price_zero_not_falsy():
    """fillPrice=0.0 must not fall through to averagePrice in query path."""
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    router = ProjectXOrderRouter(account_id=1, auth=_px_auth())
    with patch("trading_app.live.projectx.order_router.requests") as mock_req:
        mock_req.get.return_value = _px_mock_resp({"status": "Filled", "fillPrice": 0.0, "averagePrice": 2011.0})
        result = router.query_order_status(99)
    assert result["fill_price"] == 0.0
