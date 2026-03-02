import pytest

from trading_app.live.order_router import OrderRouter, OrderSpec


def test_e1_long_generates_market_buy():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = router.build_order_spec(
        direction="long", entry_model="E1",
        entry_price=2000.0, symbol="MGCM6", qty=1,
    )
    assert spec.order_type == "Market"
    assert spec.action == "Buy"
    assert spec.stop_price is None


def test_e1_short_generates_market_sell():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = router.build_order_spec(
        direction="short", entry_model="E1",
        entry_price=2000.0, symbol="MGCM6", qty=1,
    )
    assert spec.order_type == "Market"
    assert spec.action == "Sell"


def test_e2_long_generates_stop_buy():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = router.build_order_spec(
        direction="long", entry_model="E2",
        entry_price=2000.5, symbol="MGCM6", qty=1,
    )
    assert spec.order_type == "Stop"
    assert spec.action == "Buy"
    assert spec.stop_price == 2000.5


def test_e2_short_generates_stop_sell():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = router.build_order_spec(
        direction="short", entry_model="E2",
        entry_price=1999.5, symbol="MGCM6", qty=1,
    )
    assert spec.order_type == "Stop"
    assert spec.action == "Sell"
    assert spec.stop_price == 1999.5


def test_e3_raises_not_supported():
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    with pytest.raises(ValueError, match="E3"):
        router.build_order_spec(
            direction="long", entry_model="E3",
            entry_price=2000.0, symbol="MGCM6", qty=1,
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
    from trading_app.live.order_router import OrderSpec
    router = OrderRouter(account_id=12345, auth=None, demo=True)
    spec = OrderSpec(action="Buy", order_type="Market", symbol="MGCM6",
                     qty=1, account_id=12345)
    with pytest.raises(RuntimeError, match="No auth"):
        router.submit(spec)
