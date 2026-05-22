"""Test broker abstraction layer."""

import pytest

from trading_app.live.broker_base import (
    BrokerAuth,
    BrokerContracts,
    BrokerFeed,
    BrokerPositions,
    BrokerRouter,
)
from trading_app.live.http_client import EquityReading


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


class _MinimalPositions(BrokerPositions):
    """Smallest BrokerPositions subclass — implements only the abstract method.

    Used to exercise the base-class defaults (``query_equity``,
    ``query_equity_with_age``, ``query_account_metadata``) without overriding
    them. Models the Rithmic / Tradovate-today coverage gap that Stage 5
    closed at the base-class layer.
    """

    def query_open(self, account_id: int) -> list[dict]:
        return []


def test_query_equity_with_age_default_returns_missing_reading():
    """Stage 5 base-class default closes the Stage 3 fail-open ducktype gap.

    Adapters that do not override ``query_equity_with_age`` MUST receive a
    typed ``EquityReading(value=None, age_s=0.0, source="missing")`` instead
    of being absent from the API. The orchestrator's broker-state-unknown SLA
    gate treats ``source="missing"`` identically to the old "no method
    present" branch — no kill switch fires (institutionally equivalent
    fail-open). Verifying the typed contract here pins that boundary.
    """
    positions = _MinimalPositions(auth=None)
    reading = positions.query_equity_with_age(account_id=12345)
    assert isinstance(reading, EquityReading)
    assert reading.value is None
    assert reading.age_s == 0.0
    assert reading.source == "missing"


def test_query_equity_default_returns_none():
    """``query_equity`` default returns None for adapters without an override."""
    positions = _MinimalPositions(auth=None)
    assert positions.query_equity(account_id=12345) is None


def test_query_account_metadata_default_returns_none():
    """``query_account_metadata`` default returns None for adapters without an override."""
    positions = _MinimalPositions(auth=None)
    assert positions.query_account_metadata(account_id=12345) is None
