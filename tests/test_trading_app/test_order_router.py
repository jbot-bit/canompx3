import pytest

from trading_app.live.tradovate.order_router import OrderSpec
from trading_app.live.tradovate.order_router import TradovateOrderRouter as OrderRouter


def test_e1_long_generates_market_buy():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = router.build_order_spec(
        direction="long",
        entry_model="E1",
        entry_price=2000.0,
        symbol="MGCM6",
        qty=1,
    )
    assert spec.order_type == "Market"
    assert spec.action == "Buy"
    assert spec.stop_price is None


def test_e1_short_generates_market_sell():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = router.build_order_spec(
        direction="short",
        entry_model="E1",
        entry_price=2000.0,
        symbol="MGCM6",
        qty=1,
    )
    assert spec.order_type == "Market"
    assert spec.action == "Sell"


def test_e2_long_generates_stop_buy():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = router.build_order_spec(
        direction="long",
        entry_model="E2",
        entry_price=2000.5,
        symbol="MGCM6",
        qty=1,
    )
    assert spec.order_type == "Stop"
    assert spec.action == "Buy"
    assert spec.stop_price == 2000.5


def test_e2_short_generates_stop_sell():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = router.build_order_spec(
        direction="short",
        entry_model="E2",
        entry_price=1999.5,
        symbol="MGCM6",
        qty=1,
    )
    assert spec.order_type == "Stop"
    assert spec.action == "Sell"
    assert spec.stop_price == 1999.5


def test_e3_raises_not_supported():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    with pytest.raises(ValueError, match="E3"):
        router.build_order_spec(
            direction="long",
            entry_model="E3",
            entry_price=2000.0,
            symbol="MGCM6",
            qty=1,
        )


def test_exit_long_generates_market_sell():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = router.build_exit_spec(direction="long", symbol="MGCM6", qty=1)
    assert spec.order_type == "Market"
    assert spec.action == "Sell"  # close a long by selling
    assert spec.stop_price is None


def test_exit_short_generates_market_buy():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = router.build_exit_spec(direction="short", symbol="MGCM6", qty=1)
    assert spec.order_type == "Market"
    assert spec.action == "Buy"  # close a short by buying
    assert spec.stop_price is None


def test_submit_without_auth_raises():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = OrderSpec(action="Buy", order_type="Market", symbol="MGCM6", qty=1, account_id=12345)
    with pytest.raises(RuntimeError, match="No auth"):
        router.submit(spec)


# ---- fill_price parsing (OR2) ----
# Validates the is-None guard introduced in iter 21 (OR1 fix).
# Key case: fill_price=0.0 must not fall through to fallback field.

from unittest.mock import MagicMock, patch


def _tv_mock_resp(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = data
    return resp


def _tv_auth() -> MagicMock:
    auth = MagicMock()
    auth.headers.return_value = {"Authorization": "Bearer test"}
    return auth


def test_submit_fill_price_from_avgpx():
    router = OrderRouter(account_id=1, auth=_tv_auth(), demo=True)
    spec = OrderSpec(action="Buy", order_type="Market", symbol="MGCM6", qty=1, account_id=1)
    with patch("trading_app.live.tradovate.order_router.requests") as mock_req:
        mock_req.post.return_value = _tv_mock_resp({"orderId": 42, "avgPx": 2010.5})
        result = router.submit(spec)
    assert result.fill_price == 2010.5


def test_submit_fill_price_from_fallback_fill_price():
    router = OrderRouter(account_id=1, auth=_tv_auth(), demo=True)
    spec = OrderSpec(action="Buy", order_type="Market", symbol="MGCM6", qty=1, account_id=1)
    with patch("trading_app.live.tradovate.order_router.requests") as mock_req:
        mock_req.post.return_value = _tv_mock_resp({"orderId": 42, "fillPrice": 2010.5})
        result = router.submit(spec)
    assert result.fill_price == 2010.5


def test_submit_fill_price_none_when_both_absent():
    router = OrderRouter(account_id=1, auth=_tv_auth(), demo=True)
    spec = OrderSpec(action="Buy", order_type="Market", symbol="MGCM6", qty=1, account_id=1)
    with patch("trading_app.live.tradovate.order_router.requests") as mock_req:
        mock_req.post.return_value = _tv_mock_resp({"orderId": 42})
        result = router.submit(spec)
    assert result.fill_price is None


def test_submit_fill_price_zero_not_falsy():
    """avgPx=0.0 must not fall through to fillPrice — guards the is-None fix."""
    router = OrderRouter(account_id=1, auth=_tv_auth(), demo=True)
    spec = OrderSpec(action="Buy", order_type="Market", symbol="MGCM6", qty=1, account_id=1)
    with patch("trading_app.live.tradovate.order_router.requests") as mock_req:
        mock_req.post.return_value = _tv_mock_resp({"orderId": 42, "avgPx": 0.0, "fillPrice": 2010.5})
        result = router.submit(spec)
    assert result.fill_price == 0.0


def test_query_order_fill_price_from_avgpx():
    router = OrderRouter(account_id=1, auth=_tv_auth(), demo=True)
    with patch("trading_app.live.tradovate.order_router.requests") as mock_req:
        mock_req.get.return_value = _tv_mock_resp({"ordStatus": "Filled", "avgPx": 2011.0})
        result = router.query_order_status(42)
    assert result["fill_price"] == 2011.0


def test_query_order_fill_price_from_fallback():
    router = OrderRouter(account_id=1, auth=_tv_auth(), demo=True)
    with patch("trading_app.live.tradovate.order_router.requests") as mock_req:
        mock_req.get.return_value = _tv_mock_resp({"ordStatus": "Filled", "fillPrice": 2011.0})
        result = router.query_order_status(42)
    assert result["fill_price"] == 2011.0


def test_query_order_fill_price_zero_not_falsy():
    """avgPx=0.0 must not fall through to fillPrice in query path."""
    router = OrderRouter(account_id=1, auth=_tv_auth(), demo=True)
    with patch("trading_app.live.tradovate.order_router.requests") as mock_req:
        mock_req.get.return_value = _tv_mock_resp({"ordStatus": "Filled", "avgPx": 0.0, "fillPrice": 2011.0})
        result = router.query_order_status(42)
    assert result["fill_price"] == 0.0
