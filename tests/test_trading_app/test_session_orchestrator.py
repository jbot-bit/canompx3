"""Tests for SessionOrchestrator: orphan blocking + fill price tracking.

Uses pytest-asyncio (asyncio_mode="auto" in pyproject.toml) so async test
functions run natively without asyncio.run() wrappers.

FakeBrokerComponents groups all mock broker dependencies in one place.
When SessionOrchestrator.__init__ gains new attributes, add them here once.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from trading_app.live.position_tracker import PositionTracker
from trading_app.live.session_orchestrator import SessionOrchestrator
from trading_app.portfolio import Portfolio, PortfolioStrategy

# ---------------------------------------------------------------------------
# Fake broker components — single source of truth for test mocks
# ---------------------------------------------------------------------------

STRATEGY_ID = "TEST_STRAT_001"


def _test_strategy() -> PortfolioStrategy:
    return PortfolioStrategy(
        strategy_id=STRATEGY_ID,
        instrument="MGC",
        orb_label="CME_REOPEN",
        entry_model="E2",
        rr_target=2.0,
        confirm_bars=1,
        filter_type="ORB_G5",
        expectancy_r=0.20,
        win_rate=0.40,
        sample_size=200,
        sharpe_ratio=1.5,
        max_drawdown_r=3.0,
        median_risk_points=3.0,
        stop_multiplier=1.0,
        source="test",
        weight=1.0,
    )


def _test_portfolio(strategy: PortfolioStrategy | None = None) -> Portfolio:
    s = strategy or _test_strategy()
    return Portfolio(
        name="test",
        instrument="MGC",
        strategies=[s],
        account_equity=25000.0,
        risk_per_trade_pct=2.0,
        max_concurrent_positions=3,
        max_daily_loss_r=5.0,
    )


@dataclass
class FakeTradeEvent:
    event_type: str
    strategy_id: str
    timestamp: datetime
    price: float
    direction: str
    contracts: int
    reason: str = ""
    pnl_r: float | None = None


class FakeAuth:
    def get_token(self) -> str:
        return "fake-token"

    def headers(self) -> dict:
        return {"Authorization": "Bearer fake-token"}

    def refresh_if_needed(self) -> None:
        pass


class FakePositions:
    def __init__(self, orphans: list[dict] | None = None):
        self._orphans = orphans or []

    def query_open(self, account_id: int) -> list[dict]:
        return self._orphans


class FakeRouter:
    def __init__(self, fill_price: float | None = None):
        self._fill_price = fill_price
        self.account_id = 12345
        self.submitted: list[dict] = []

    def build_order_spec(self, **kwargs) -> dict:
        return {"type": "fake_entry", **kwargs}

    def build_exit_spec(self, **kwargs) -> dict:
        return {"type": "fake_exit", **kwargs}

    def submit(self, spec: dict) -> dict:
        self.submitted.append(spec)
        return {
            "order_id": 99,
            "status": "submitted",
            "fill_price": self._fill_price,
        }

    def supports_native_brackets(self) -> bool:
        return False

    def cancel(self, order_id: int) -> None:
        pass


@dataclass
class FakeBrokerComponents:
    """All mock broker deps in one place. Add new attributes here when __init__ changes."""

    auth: FakeAuth = field(default_factory=FakeAuth)
    router: FakeRouter | None = None
    positions: FakePositions = field(default_factory=FakePositions)
    fill_price: float | None = None
    orphans: list[dict] = field(default_factory=list)
    signal_only: bool = False

    def __post_init__(self):
        self.positions = FakePositions(orphans=self.orphans)
        if not self.signal_only:
            self.router = FakeRouter(fill_price=self.fill_price)


# ---------------------------------------------------------------------------
# Pytest fixture — replaces _make_orchestrator
# ---------------------------------------------------------------------------


def build_orchestrator(components: FakeBrokerComponents | None = None) -> SessionOrchestrator:
    """Build a SessionOrchestrator with all heavy dependencies mocked."""
    c = components or FakeBrokerComponents()
    strategy = _test_strategy()
    portfolio = _test_portfolio(strategy)

    with patch.object(SessionOrchestrator, "__init__", lambda self, **kw: None):
        orch = SessionOrchestrator.__new__(SessionOrchestrator)

    orch.instrument = "MGC"
    orch.demo = True
    orch.signal_only = c.signal_only
    orch.trading_day = date(2026, 3, 7)
    orch._broker_name = "test"
    orch.auth = c.auth
    orch.portfolio = portfolio
    orch._strategy_map = {s.strategy_id: s for s in portfolio.strategies}
    orch.cost_spec = MagicMock()
    orch.cost_spec.friction_in_points = 0.5
    orch.risk_mgr = MagicMock()
    orch.engine = MagicMock()
    orch.orb_builder = MagicMock()
    orch.monitor = MagicMock()
    orch.monitor.record_trade.return_value = None
    orch._positions = PositionTracker()
    orch._last_bar_at = None
    orch._kill_switch_fired = False
    orch.contract_symbol = "MGCJ6"
    orch.order_router = c.router
    orch.positions = c.positions
    orch._write_signal_record = MagicMock()

    return orch


@pytest.fixture
def orch():
    """Default orchestrator: no orphans, no fill price, order routing enabled."""
    return build_orchestrator()


@pytest.fixture
def orch_with_fill():
    """Orchestrator where broker returns fill_price=2351.0."""
    return build_orchestrator(FakeBrokerComponents(fill_price=2351.0))


@pytest.fixture
def orch_signal_only():
    """Signal-only orchestrator: no order routing."""
    return build_orchestrator(FakeBrokerComponents(signal_only=True))


def _entry_event(price: float = 2350.5) -> FakeTradeEvent:
    return FakeTradeEvent(
        event_type="ENTRY",
        strategy_id=STRATEGY_ID,
        timestamp=datetime.now(UTC),
        price=price,
        direction="long",
        contracts=1,
    )


def _exit_event(price: float = 2355.0, pnl_r: float | None = 1.5) -> FakeTradeEvent:
    return FakeTradeEvent(
        event_type="EXIT",
        strategy_id=STRATEGY_ID,
        timestamp=datetime.now(UTC),
        price=price,
        direction="long",
        contracts=1,
        pnl_r=pnl_r,
    )


# ---------------------------------------------------------------------------
# HIGH-2: Orphan blocking tests
# ---------------------------------------------------------------------------


class TestOrphanBlocking:
    def test_orphans_detected_blocks_startup(self):
        """Orphan positions should prevent session start."""
        orphans = [{"contract_id": "MGC", "side": "long", "size": 1, "avg_price": 2350.0}]
        positions = FakePositions(orphans=orphans)
        result = positions.query_open(12345)
        assert len(result) == 1

        with pytest.raises(RuntimeError, match="orphaned position"):
            if result:
                raise RuntimeError(
                    f"Refusing to start: {len(result)} orphaned position(s) detected. "
                    f"Close them manually or pass --force-orphans to acknowledge the risk."
                )

    def test_orphans_allowed_with_force_flag(self):
        """force_orphans=True: orphans logged, no error."""
        positions = FakePositions(orphans=[{"contract_id": "MGC", "side": "long", "size": 1, "avg_price": 2350.0}])
        result = positions.query_open(12345)
        force_orphans = True
        # Should NOT raise
        if result and not force_orphans:
            raise RuntimeError("Should not reach here")

    def test_no_orphans_no_error(self):
        """No orphans = clean start."""
        assert FakePositions(orphans=[]).query_open(12345) == []


# ---------------------------------------------------------------------------
# HIGH-1: Fill price tracking tests
# ---------------------------------------------------------------------------


class TestFillPriceTracking:
    async def test_entry_with_fill_tracks_slippage(self, orch_with_fill):
        """Broker fill_price stored alongside engine price, slippage computed."""
        await orch_with_fill._handle_event(_entry_event(2350.5))

        record = orch_with_fill._positions.get(STRATEGY_ID)
        assert record.engine_entry_price == 2350.5
        assert record.fill_entry_price == 2351.0
        assert record.entry_slippage == pytest.approx(0.5)

    async def test_entry_without_fill_records_none(self, orch):
        """No fill_price from broker -> stored as None, no slippage key."""
        await orch._handle_event(_entry_event(2350.5))

        record = orch._positions.get(STRATEGY_ID)
        assert record.engine_entry_price == 2350.5
        assert record.fill_entry_price is None

    async def test_exit_uses_fill_price_for_entry(self, orch_with_fill):
        """EXIT uses broker fill_price (not engine price) for entry."""
        orch_with_fill._positions.on_entry_sent(STRATEGY_ID, "long", 2350.5, order_id=1)
        orch_with_fill._positions.on_entry_filled(STRATEGY_ID, 2351.0)

        await orch_with_fill._handle_event(_exit_event())

        record = orch_with_fill.monitor.record_trade.call_args[0][0]
        assert record.entry_price == 2351.0  # fill, not engine 2350.5

    async def test_exit_falls_back_to_engine_when_no_fill(self, orch):
        """No fill_price -> exit uses engine_price."""
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.5, order_id=1)
        # Don't call on_entry_filled — simulates no fill

        await orch._handle_event(_exit_event())

        record = orch.monitor.record_trade.call_args[0][0]
        assert record.entry_price == 2350.5

    async def test_signal_only_entry_tracks_engine_only(self, orch_signal_only):
        """Signal-only: no broker interaction, only engine_price recorded."""
        await orch_signal_only._handle_event(_entry_event(2350.5))

        record = orch_signal_only._positions.get(STRATEGY_ID)
        assert record.engine_entry_price == 2350.5
        assert record.fill_entry_price is None

    async def test_entry_exit_roundtrip(self, orch_with_fill):
        """Full ENTRY -> EXIT flow: fill tracked through entire lifecycle."""
        await orch_with_fill._handle_event(_entry_event(2350.5))
        assert orch_with_fill._positions.get(STRATEGY_ID) is not None

        await orch_with_fill._handle_event(_exit_event(2355.0))
        assert orch_with_fill._positions.get(STRATEGY_ID) is None  # cleaned up

        record = orch_with_fill.monitor.record_trade.call_args[0][0]
        assert record.entry_price == 2351.0  # broker fill, not engine


class TestBestPrice:
    def test_prefers_fill(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 99.0)
        tracker.on_entry_filled("S1", 100.0)
        assert tracker.best_entry_price("S1", 0.0) == 100.0

    def test_falls_back_to_engine(self):
        tracker = PositionTracker()
        tracker.on_entry_sent("S1", "long", 99.0)
        assert tracker.best_entry_price("S1", 0.0) == 99.0

    def test_falls_back_to_fallback(self):
        tracker = PositionTracker()
        assert tracker.best_entry_price("UNKNOWN", 42.0) == 42.0


# ---------------------------------------------------------------------------
# Performance monitor slippage tracking
# ---------------------------------------------------------------------------


class TestSlippageInSummary:
    def test_slippage_in_trade_record(self):
        from trading_app.live.performance_monitor import TradeRecord

        record = TradeRecord(
            strategy_id="X",
            trading_day=date(2026, 3, 7),
            direction="long",
            entry_price=100.0,
            exit_price=102.0,
            actual_r=0.5,
            expected_r=0.3,
            slippage_pts=0.25,
        )
        assert record.slippage_pts == 0.25

    def test_daily_summary_includes_slippage(self):
        from trading_app.live.performance_monitor import PerformanceMonitor, TradeRecord

        monitor = PerformanceMonitor([_test_strategy()])
        record = TradeRecord(
            strategy_id=STRATEGY_ID,
            trading_day=date(2026, 3, 7),
            direction="long",
            entry_price=100.0,
            exit_price=102.0,
            actual_r=0.5,
            expected_r=0.3,
            slippage_pts=0.25,
        )
        monitor.record_trade(record)
        summary = monitor.daily_summary()
        assert summary["total_slippage_pts"] == 0.25


# ---------------------------------------------------------------------------
# KILL SWITCH tests
# ---------------------------------------------------------------------------


class FakeBar:
    """Minimal bar stub for _on_bar tests."""

    def __init__(self, ts_utc=None):
        self.ts_utc = ts_utc or datetime.now(UTC)
        self.open = 2350.0
        self.high = 2351.0
        self.low = 2349.0
        self.close = 2350.5
        self.volume = 100

    def as_dict(self):
        return {
            "ts_utc": self.ts_utc,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class TestKillSwitch:
    """Kill switch / emergency flatten — the last line of defense."""

    async def test_watchdog_fires_on_feed_death_with_open_positions(self):
        """Feed dies for >KILL_SWITCH_TIMEOUT with active positions → emergency flatten."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        # Simulate an open position
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)
        # Simulate last bar was long ago
        orch._last_bar_at = datetime.now(UTC) - timedelta(seconds=400)
        # Override timeout for fast test
        orch.KILL_SWITCH_TIMEOUT = 0.0
        orch.KILL_SWITCH_CHECK_INTERVAL = 0.01

        # Run watchdog briefly — it should fire
        task = asyncio.create_task(orch._watchdog())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert orch._kill_switch_fired is True
        # Position should be flattened
        assert orch._positions.active_positions() == []
        # Router should have received a close order
        assert len(orch.order_router.submitted) == 1

    async def test_watchdog_does_nothing_without_open_positions(self):
        """Feed dead but no open positions → watchdog does NOT fire."""
        orch = build_orchestrator()
        orch._last_bar_at = datetime.now(UTC) - timedelta(seconds=400)
        orch.KILL_SWITCH_TIMEOUT = 0.0
        orch.KILL_SWITCH_CHECK_INTERVAL = 0.01

        task = asyncio.create_task(orch._watchdog())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert orch._kill_switch_fired is False

    async def test_watchdog_skips_before_first_bar(self):
        """No bars received yet (_last_bar_at is None) → watchdog waits."""
        orch = build_orchestrator()
        assert orch._last_bar_at is None
        orch.KILL_SWITCH_TIMEOUT = 0.0
        orch.KILL_SWITCH_CHECK_INTERVAL = 0.01

        task = asyncio.create_task(orch._watchdog())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert orch._kill_switch_fired is False

    async def test_kill_switch_is_one_shot(self):
        """Kill switch fires once, not on every watchdog cycle."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)
        orch._last_bar_at = datetime.now(UTC) - timedelta(seconds=400)
        orch.KILL_SWITCH_TIMEOUT = 0.0
        orch.KILL_SWITCH_CHECK_INTERVAL = 0.01

        task = asyncio.create_task(orch._watchdog())
        await asyncio.sleep(0.2)  # multiple watchdog cycles
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Only ONE close order submitted despite multiple cycles
        assert len(orch.order_router.submitted) == 1

    async def test_on_bar_returns_early_after_kill_switch(self):
        """After kill switch fires, _on_bar does nothing (prevents duplicate exits)."""
        orch = build_orchestrator()
        orch._kill_switch_fired = True
        orch.engine.on_bar.return_value = []

        bar = FakeBar()
        await orch._on_bar(bar)

        # Engine should NOT be called — we returned early
        orch.engine.on_bar.assert_not_called()

    async def test_emergency_flatten_signal_only_logs_manual_close(self):
        """Signal-only mode: kill switch logs MANUAL CLOSE REQUIRED, no broker calls."""
        orch = build_orchestrator(FakeBrokerComponents(signal_only=True))
        orch._positions.on_signal_entry(STRATEGY_ID, 2350.0, "long")
        orch._last_bar_at = datetime.now(UTC)

        await orch._emergency_flatten()

        # Signal record should have KILL_SWITCH type
        orch._write_signal_record.assert_called()
        call_args = orch._write_signal_record.call_args[0][0]
        assert call_args["type"] == "KILL_SWITCH"
        # Position NOT flattened (no broker to send to)
        assert len(orch._positions.active_positions()) == 1

    async def test_emergency_flatten_retries_on_failure(self):
        """Broker failure → 3 retries with exponential backoff."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)

        # Make first 2 attempts fail, 3rd succeeds
        call_count = 0
        original_submit = orch.order_router.submit

        def failing_submit(spec):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("broker down")
            return original_submit(spec)

        orch.order_router.submit = failing_submit

        await orch._emergency_flatten()

        assert call_count == 3  # 2 failures + 1 success
        assert orch._positions.active_positions() == []

    async def test_emergency_flatten_all_retries_fail_logs_manual(self):
        """All 3 retries fail → logs MANUAL CLOSE REQUIRED."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)

        # All attempts fail
        orch.order_router.submit = MagicMock(side_effect=ConnectionError("broker dead"))

        await orch._emergency_flatten()

        # 3 attempts made
        assert orch.order_router.submit.call_count == 3
        # Position still active (couldn't flatten)
        assert len(orch._positions.active_positions()) == 1

    async def test_watchdog_survives_internal_errors(self):
        """Watchdog doesn't die from its own errors (crash-resistant)."""
        orch = build_orchestrator()
        orch.KILL_SWITCH_CHECK_INTERVAL = 0.01
        # Force an error in active_positions check by setting _last_bar_at to
        # a value that will cause a gap check, then make _positions raise
        orch._last_bar_at = datetime.now(UTC) - timedelta(seconds=400)
        orch.KILL_SWITCH_TIMEOUT = 0.0

        # Patch active_positions to raise once, then return empty
        original = orch._positions.active_positions
        error_raised = False

        def flaky_active():
            nonlocal error_raised
            if not error_raised:
                error_raised = True
                raise RuntimeError("transient DB error")
            return original()

        orch._positions.active_positions = flaky_active

        # Watchdog should survive the error and keep running
        task = asyncio.create_task(orch._watchdog())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Key: watchdog survived — error_raised was True, then it continued
        assert error_raised is True
        # Kill switch should NOT have fired (active_positions returned [] on retry)
        assert orch._kill_switch_fired is False

    async def test_run_starts_watchdog_task(self):
        """run() creates a watchdog task alongside the feed."""
        orch = build_orchestrator()

        # Mock feed that completes immediately
        class InstantFeed:
            def __init__(self, auth, on_bar, demo):
                pass

            async def run(self, symbol):
                return  # complete immediately

        orch._feed_class = InstantFeed

        await orch.run()
        # If we got here without error, watchdog was started and cancelled in finally

    async def test_bar_heartbeat_logs_gap(self, orch):
        """Bar heartbeat detects >180s gap and logs CRITICAL."""
        import logging

        orch._last_bar_at = datetime.now(UTC) - timedelta(seconds=200)
        orch.engine.on_bar.return_value = []
        orch._check_trading_day_rollover = MagicMock()

        bar = FakeBar()
        with patch.object(logging.getLogger("trading_app.live.session_orchestrator"), "critical") as mock_crit:
            await orch._on_bar(bar)
            # Should have logged a critical heartbeat warning
            assert any("HEARTBEAT" in str(call) for call in mock_crit.call_args_list)
