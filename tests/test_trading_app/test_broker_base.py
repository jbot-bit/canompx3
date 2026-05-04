"""Test broker abstraction layer."""

import pytest

from trading_app.live.broker_base import (
    BrokerAuth,
    BrokerContracts,
    BrokerFeed,
    BrokerPositions,
    BrokerRouter,
)


def test_abc_cannot_instantiate():
    """ABCs should not be directly instantiable."""
    with pytest.raises(TypeError):
        BrokerAuth()
    with pytest.raises(TypeError):
        BrokerFeed(auth=None, on_bar=None)
    with pytest.raises(TypeError):
        BrokerRouter(account_id=0, auth=None)
    with pytest.raises(TypeError):
        BrokerContracts(auth=None)
    with pytest.raises(TypeError):
        BrokerPositions(auth=None)
