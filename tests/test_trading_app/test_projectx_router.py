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
