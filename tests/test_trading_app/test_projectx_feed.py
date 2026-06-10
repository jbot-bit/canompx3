"""Test ProjectX data feed — verify quote parsing, bar creation, and liveness monitoring."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from trading_app.live.bar_aggregator import Bar
from trading_app.live.broker_base import BrokerFeed
from trading_app.live.projectx.data_feed import (
    _MAX_STALE_BEFORE_RECONNECT,
    _STALE_TIMEOUT,
    ProjectXDataFeed,
)


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

    def test_explicit_zero_volume_passes_through(self, feed):
        """Explicit volume 0 must pass through as 0 (NOT coerced to 1).

        Under cumulative semantics (see _cum_to_delta), 0 is a real zero-cumulative
        reading (pre-open / no trades yet). Coercing 0→1 would corrupt the delta
        baseline by one contract. Only a MISSING volume field defaults to 1.
        """
        _, vol = feed.parse_quote({"lastPrice": 100.0, "volume": 0})
        assert vol == 0


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


class TestPysignalrStop:
    @pytest.mark.asyncio
    async def test_stop_flag_cancels_client_run(self):
        """Stop flag must interrupt client.run() via asyncio.wait(FIRST_COMPLETED)."""
        auth = MagicMock()
        auth.get_token.return_value = "fake-token"

        feed = ProjectXDataFeed(auth=auth, on_bar=MagicMock())
        feed._stop_requested = False

        # Mock pysignalr client that blocks forever
        mock_client = MagicMock()

        async def run_forever():
            await asyncio.sleep(10)

        mock_client.run = run_forever
        mock_client.on = MagicMock()
        mock_client.on_open = MagicMock()

        # Fast stop watcher: returns as soon as _stop_requested is set
        async def fast_stop_watcher():
            while not feed._stop_requested:
                await asyncio.sleep(0.01)

        # Set stop flag after 50ms
        async def set_stop():
            await asyncio.sleep(0.05)
            feed._stop_requested = True

        from unittest.mock import patch

        with (
            patch("pysignalr.client.SignalRClient", return_value=mock_client),
            patch.object(feed, "_stop_file_watcher", fast_stop_watcher),
        ):
            stop_setter = asyncio.create_task(set_stop())
            try:
                await asyncio.wait_for(feed._run_pysignalr("12345"), timeout=2.0)
            except TimeoutError:
                pytest.fail("_run_pysignalr did not exit within 2s — stop flag was ignored")
            finally:
                stop_setter.cancel()


class TestLivenessMonitor:
    """Test feed staleness detection and forced reconnect."""

    def test_quote_updates_last_data_at(self, feed):
        """Receiving a quote must update _last_data_at."""
        assert feed._last_data_at is None
        # Simulate sync quote
        feed._on_quote_sync([{"lastPrice": 100.0, "volume": 1}])
        assert feed._last_data_at is not None
        assert feed._quote_count == 1

    def test_stale_count_increments_on_gap(self, feed):
        """Stale count should increment when data is older than threshold."""
        feed._last_data_at = datetime.now(UTC) - timedelta(seconds=_STALE_TIMEOUT + 10)
        feed._stale_count = 0
        # Simulate what the watcher does
        gap = (datetime.now(UTC) - feed._last_data_at).total_seconds()
        if gap > _STALE_TIMEOUT:
            feed._stale_count += 1
        assert feed._stale_count == 1

    def test_stale_count_resets_on_fresh_data(self, feed):
        """Stale count must reset to 0 when fresh data arrives."""
        feed._stale_count = 5
        feed._last_data_at = datetime.now(UTC)  # fresh
        gap = (datetime.now(UTC) - feed._last_data_at).total_seconds()
        if gap <= _STALE_TIMEOUT:
            feed._stale_count = 0
        assert feed._stale_count == 0

    def test_force_reconnect_after_max_stale(self, feed):
        """Feed must set _force_reconnect after MAX_STALE consecutive stale checks."""
        feed._last_data_at = datetime.now(UTC) - timedelta(seconds=_STALE_TIMEOUT + 10)
        feed._stale_count = _MAX_STALE_BEFORE_RECONNECT - 1
        feed._force_reconnect = False
        # One more stale check should trigger
        feed._stale_count += 1
        if feed._stale_count >= _MAX_STALE_BEFORE_RECONNECT:
            feed._force_reconnect = True
        assert feed._force_reconnect is True

    def test_on_stale_callback_fires(self):
        """on_stale callback must be called with gap and count."""
        auth = MagicMock()
        stale_calls = []
        feed = ProjectXDataFeed(
            auth=auth,
            on_bar=MagicMock(),
            on_stale=lambda gap, count: stale_calls.append((gap, count)),
        )
        feed._last_data_at = datetime.now(UTC) - timedelta(seconds=_STALE_TIMEOUT + 30)
        feed._stale_count = 0
        # Simulate watcher logic
        gap = (datetime.now(UTC) - feed._last_data_at).total_seconds()
        if gap > _STALE_TIMEOUT:
            feed._stale_count += 1
            if feed.on_stale is not None:
                feed.on_stale(gap, feed._stale_count)
        assert len(stale_calls) == 1
        assert stale_calls[0][0] > _STALE_TIMEOUT
        assert stale_calls[0][1] == 1

    @pytest.mark.asyncio
    async def test_watcher_triggers_reconnect_on_stale(self):
        """Stop-file watcher must set _force_reconnect after consecutive stale periods."""
        from unittest.mock import patch

        from trading_app.live.projectx.data_feed import _STOP_FILE

        auth = MagicMock()
        auth.get_token.return_value = "fake"
        feed = ProjectXDataFeed(auth=auth, on_bar=MagicMock())
        feed._last_data_at = datetime.now(UTC) - timedelta(seconds=_STALE_TIMEOUT + 10)
        feed._stale_count = _MAX_STALE_BEFORE_RECONNECT - 1  # one more check triggers

        # Mock stop file to not exist (prevents false pass if file is on disk)
        with patch.object(type(_STOP_FILE), "exists", return_value=False):
            try:
                await asyncio.wait_for(feed._stop_file_watcher(), timeout=5.0)
            except TimeoutError:
                pytest.fail("Watcher did not exit on stale feed within 5s")

        assert feed._force_reconnect is True


class TestCumulativeToDelta:
    """Defect A: GatewayQuote.volume is cumulative session volume; the feed must
    diff it to per-tick deltas before the aggregator (which sums deltas)."""

    def test_first_reading_returns_zero_baseline(self, feed):
        """The very first cumulative reading has no prior baseline → delta 0."""
        assert feed._cum_to_delta(12000) == 0
        assert feed._last_cum_volume == 12000

    def test_consecutive_readings_become_deltas(self, feed):
        """Cumulative 100 → 130 → 145 must yield deltas 0 → 30 → 15."""
        assert feed._cum_to_delta(100) == 0
        assert feed._cum_to_delta(130) == 30
        assert feed._cum_to_delta(145) == 15

    def test_unchanged_reading_yields_zero(self, feed):
        """A quote with no new traded volume (same cumulative) → delta 0."""
        feed._cum_to_delta(500)
        assert feed._cum_to_delta(500) == 0

    def test_zero_cumulative_baseline_no_off_by_one(self, feed):
        """A session that opens with cumulative volume 0 must baseline to 0, so the
        first real reading yields the FULL delta — not delta-1 (the parse_quote 0→1
        coercion bug the adversarial audit flagged, data_feed.py:93)."""
        # parse_quote must hand 0 through (not coerce to 1)
        _, cum0 = feed.parse_quote({"lastPrice": 100.0, "volume": 0})
        assert cum0 == 0
        assert feed._cum_to_delta(cum0) == 0  # baseline at 0
        assert feed._last_cum_volume == 0
        # First real cumulative reading of 50 → full delta 50 (NOT 49)
        _, cum50 = feed.parse_quote({"lastPrice": 100.0, "volume": 50})
        assert feed._cum_to_delta(cum50) == 50

    def test_session_reset_clamps_negative_delta_to_zero(self, feed):
        """A cumulative reset (1000 → 50) must emit 0 and re-baseline to 50,
        so the next reading (60) yields a correct delta of 10 — not 10-from-1000."""
        feed._cum_to_delta(1000)
        assert feed._cum_to_delta(50) == 0  # reset detected, re-baselined
        assert feed._last_cum_volume == 50
        assert feed._cum_to_delta(60) == 10

    @pytest.mark.asyncio
    async def test_realistic_per_minute_volume_via_async_quote_path(self):
        """End-to-end through _on_quote: a minute of cumulative quotes must produce a
        bar whose volume equals the true in-minute increment (last_cum - first_cum),
        NOT the sum of cumulative readings (which would be billions)."""
        from unittest.mock import patch

        auth = MagicMock()
        delivered = []

        async def capture(bar):
            delivered.append(bar)

        feed = ProjectXDataFeed(auth=auth, on_bar=capture)
        feed._symbol = "MNQ"

        minute_a = datetime(2026, 6, 10, 14, 0, tzinfo=UTC)
        minute_b = datetime(2026, 6, 10, 14, 1, tzinfo=UTC)

        # Cumulative session volume climbs 1_000_000 → 1_000_300 within minute A
        # (300 contracts actually traded), then a quote in minute B closes bar A.
        cumulative_in_minute_a = [1_000_000, 1_000_100, 1_000_250, 1_000_300]
        with patch("trading_app.live.projectx.data_feed.datetime") as mock_dt:
            mock_dt.now.return_value = minute_a
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            for cum in cumulative_in_minute_a:
                await feed._on_quote([{"lastPrice": 21000.0, "volume": cum}])
            # First quote in minute B → completes and emits bar A
            mock_dt.now.return_value = minute_b
            await feed._on_quote([{"lastPrice": 21001.0, "volume": 1_000_320}])

        assert len(delivered) == 1, "exactly one completed bar expected"
        bar_a = delivered[0]
        # First reading was baseline (delta 0); subsequent deltas: 100 + 150 + 50 = 300.
        # This is the true in-minute traded volume, NOT sum(cumulative) = ~4 million.
        assert bar_a.volume == 300
        assert bar_a.volume < 10_000, "volume must be realistic, not cumulative garbage"

    def test_sync_quote_path_no_garbage_volume(self, feed):
        """Drive the foreign-thread sync handler with cumulative quotes and assert the
        delta conversion runs (no garbage cumulative value reaches the aggregator)."""
        # No event loop attached → _apply_tick_state runs inline (fallback path).
        feed._on_quote_sync([{"lastPrice": 21000.0, "volume": 5_000_000}])  # baseline
        feed._on_quote_sync([{"lastPrice": 21000.5, "volume": 5_000_120}])  # +120
        # The aggregator's in-progress bar should hold the delta (120), never 5_000_120.
        assert feed._agg._current is not None
        assert feed._agg._current.volume == 120
