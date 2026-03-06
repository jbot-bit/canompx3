"""Test broker factory."""

import os
from unittest.mock import patch

import pytest

from trading_app.live.broker_factory import create_broker_components, get_broker_name


def test_factory_unknown_broker():
    with pytest.raises(ValueError, match="Unknown broker"):
        create_broker_components("nonexistent")


def test_factory_tradovate():
    components = create_broker_components("tradovate", demo=True)
    assert "auth" in components
    assert "feed_class" in components
    assert "router_class" in components
    assert "contracts_class" in components
    assert "positions_class" in components

    from trading_app.live.broker_base import BrokerAuth

    assert isinstance(components["auth"], BrokerAuth)


def test_factory_projectx():
    with patch.dict("os.environ", {"PROJECTX_USER": "test", "PROJECTX_API_KEY": "test"}):
        # ProjectXAuth doesn't call login in __init__, so no mock needed
        components = create_broker_components("projectx")
        assert "auth" in components

        from trading_app.live.broker_base import BrokerAuth

        assert isinstance(components["auth"], BrokerAuth)


def test_get_broker_name_default():
    old = os.environ.pop("BROKER", None)
    try:
        assert get_broker_name() == "projectx"
    finally:
        if old is not None:
            os.environ["BROKER"] = old


def test_get_broker_name_from_env():
    with patch.dict("os.environ", {"BROKER": "tradovate"}):
        assert get_broker_name() == "tradovate"


def test_get_broker_name_case_insensitive():
    with patch.dict("os.environ", {"BROKER": "ProjectX"}):
        assert get_broker_name() == "projectx"


def test_factory_returns_correct_classes_tradovate():
    components = create_broker_components("tradovate", demo=True)
    from trading_app.live.tradovate.auth import TradovateAuth
    from trading_app.live.tradovate.data_feed import TradovateDataFeed
    from trading_app.live.tradovate.order_router import TradovateOrderRouter

    assert isinstance(components["auth"], TradovateAuth)
    assert components["feed_class"] is TradovateDataFeed
    assert components["router_class"] is TradovateOrderRouter


def test_factory_returns_correct_classes_projectx():
    components = create_broker_components("projectx")
    from trading_app.live.projectx.auth import ProjectXAuth
    from trading_app.live.projectx.data_feed import ProjectXDataFeed
    from trading_app.live.projectx.order_router import ProjectXOrderRouter

    assert isinstance(components["auth"], ProjectXAuth)
    assert components["feed_class"] is ProjectXDataFeed
    assert components["router_class"] is ProjectXOrderRouter
