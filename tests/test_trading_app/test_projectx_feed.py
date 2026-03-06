"""Test ProjectX data feed — verify quote parsing and bar creation."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from trading_app.live.bar_aggregator import Bar
from trading_app.live.broker_base import BrokerFeed
from trading_app.live.projectx.data_feed import ProjectXDataFeed


@pytest.fixture()
def feed():
    """Create a ProjectXDataFeed with mock auth and on_bar."""
    mock_auth = MagicMock()
    return ProjectXDataFeed(auth=mock_auth, on_bar=MagicMock())


class TestParseQuote:
    def test_last_price(self, feed):
        """parse_quote should extract lastPrice and volume."""
        price, vol = feed.parse_quote({"lastPrice": 2950.25, "volume": 100})
        assert price == 2950.25
        assert vol == 100

    def test_fallback_bid(self, feed):
        """Should fall back to bestBid if lastPrice missing."""
        price, vol = feed.parse_quote({"bestBid": 2949.50, "volume": 50})
        assert price == 2949.50
        assert vol == 50

    def test_fallback_ask(self, feed):
        """Should fall back to bestAsk if lastPrice and bestBid missing."""
        price, vol = feed.parse_quote({"bestAsk": 2951.00, "volume": 10})
        assert price == 2951.00
        assert vol == 10

    def test_no_price_raises(self, feed):
        """Should raise ValueError if no price fields present."""
        with pytest.raises(ValueError, match="No price"):
            feed.parse_quote({"volume": 100})

    def test_default_volume(self, feed):
        """Volume defaults to 1 when not present."""
        price, vol = feed.parse_quote({"lastPrice": 100.0})
        assert vol == 1

    def test_zero_volume_becomes_one(self, feed):
        """Volume of 0 or None should become 1."""
        _, vol = feed.parse_quote({"lastPrice": 100.0, "volume": 0})
        assert vol == 1


class TestFlush:
    def test_flush_empty_returns_none(self, feed):
        """flush() on empty aggregator returns None."""
        assert feed.flush("MGC") is None

    def test_flush_sets_symbol(self, feed):
        """flush() should set bar.symbol from argument or internal state."""
        from datetime import UTC, datetime

        # Feed a tick so aggregator has an open bar
        feed._agg.on_tick(2950.0, 1, datetime.now(UTC))
        bar = feed.flush("MGC")
        assert bar is not None
        assert bar.symbol == "MGC"

    def test_flush_uses_internal_symbol(self, feed):
        """flush() with no arg should use self._symbol."""
        from datetime import UTC, datetime

        feed._symbol = "MNQ"
        feed._agg.on_tick(18000.0, 1, datetime.now(UTC))
        bar = feed.flush()
        assert bar is not None
        assert bar.symbol == "MNQ"


class TestSyncCallbackQueueBridge:
    @pytest.mark.asyncio
    async def test_sync_callback_delivers_bars_via_queue(self):
        """Bars completed in sync callbacks must reach on_bar via the queue bridge."""
        auth = MagicMock()
        auth.get_token.return_value = "fake"
        delivered = []

        async def capture(bar):
            delivered.append(bar)

        feed = ProjectXDataFeed(auth=auth, on_bar=capture)
        feed._symbol = "12345"

        # Start the queue consumer
        consumer = asyncio.create_task(feed._drain_bar_queue())

        # Simulate a completed bar being pushed via sync callback path
        bar = Bar(
            ts_utc=datetime(2026, 3, 7, 14, 0, tzinfo=UTC),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=50,
            symbol="12345",
        )
        feed._bar_queue.put_nowait(bar)

        # Give consumer time to process
        await asyncio.sleep(0.05)
        consumer.cancel()

        assert len(delivered) == 1
        assert delivered[0].close == 100.5


class TestBrokerFeedABC:
    def test_is_broker_feed(self, feed):
        """ProjectXDataFeed must be a BrokerFeed."""
        assert isinstance(feed, BrokerFeed)

    def test_has_run_method(self, feed):
        """Must implement run()."""
        assert callable(feed.run)

    def test_has_flush_method(self, feed):
        """Must implement flush()."""
        assert callable(feed.flush)
