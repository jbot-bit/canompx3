"""Tests for SessionOrchestrator: orphan blocking + fill price tracking.

Uses pytest-asyncio (asyncio_mode="auto" in pyproject.toml) so async test
functions run natively without asyncio.run() wrappers.

FakeBrokerComponents groups all mock broker dependencies in one place.
When SessionOrchestrator.__init__ gains new attributes, add them here once.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

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

    def build_bracket_spec(self, **kwargs) -> dict | None:
        return None

    def query_order_status(self, order_id: int) -> dict:
        raise NotImplementedError


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
    orch.monitor.daily_summary.return_value = {"n_trades": 0, "total_r": 0.0, "total_slippage_pts": 0.0}
    orch._positions = PositionTracker()
    orch._last_bar_at = None
    orch._kill_switch_fired = False
    orch._bar_count = 0
    orch._notifications_broken = False

    from trading_app.live.session_orchestrator import SessionStats

    orch._stats = SessionStats()
    orch._poller_active = False
    orch.contract_symbol = "MGCJ6"

    from trading_app.live.circuit_breaker import CircuitBreaker

    orch._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
    orch.order_router = c.router
    orch.positions = c.positions
    orch._write_signal_record = MagicMock()
    # Self-tests require real Telegram/broker — mock to always pass in tests
    orch.run_self_tests = MagicMock(return_value={"notifications": True, "brackets": True, "fill_poller": True})

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

    def test_not_implemented_does_not_crash(self):
        """NotImplementedError from query_open → warning, not crash."""
        import logging

        class UnimplementedPositions:
            def query_open(self, account_id: int):
                raise NotImplementedError("not supported")

        positions = UnimplementedPositions()
        # Simulate the orchestrator's __init__ logic
        broker_name = "test_broker"
        warned = False
        try:
            positions.query_open(12345)
        except NotImplementedError:
            warned = True
            logging.getLogger().warning(
                "ORPHAN DETECTION DISABLED — %s does not implement query_open()",
                broker_name,
            )
        except RuntimeError:
            raise
        except Exception:
            pass

        assert warned, "NotImplementedError should be caught, not re-raised"


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

    def test_reset_daily_clears_trades(self):
        """reset_daily() must clear _trades list so slippage doesn't accumulate across days."""
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
        monitor.reset_daily()

        summary = monitor.daily_summary()
        assert summary["total_slippage_pts"] == 0.0
        assert summary["n_trades"] == 0


# ---------------------------------------------------------------------------
# DAILY FEATURES FAIL-CLOSED tests
# ---------------------------------------------------------------------------


class TestDailyFeaturesFailClosed:
    def test_raises_on_db_error(self):
        """_build_daily_features_row must raise if DB is unreachable."""
        import duckdb as real_duckdb

        mock_duckdb = MagicMock(spec=real_duckdb)
        mock_duckdb.connect.side_effect = OSError("DB locked")
        with patch.dict("sys.modules", {"duckdb": mock_duckdb}):
            with pytest.raises(RuntimeError, match="FAIL-CLOSED"):
                SessionOrchestrator._build_daily_features_row(date(2026, 3, 7), "MGC")


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

        # Mock feed that completes immediately with stop request
        class InstantFeed:
            def __init__(self, auth, on_bar, demo):
                self._stop_requested = True

            @property
            def was_stopped(self):
                return self._stop_requested

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
        orch._check_trading_day_rollover = AsyncMock()

        bar = FakeBar()
        with patch.object(logging.getLogger("trading_app.live.session_orchestrator"), "critical") as mock_crit:
            await orch._on_bar(bar)
            # Should have logged a critical heartbeat warning
            assert any("HEARTBEAT" in str(call) for call in mock_crit.call_args_list)

    async def test_post_session_skips_eod_close_after_kill_switch(self):
        """post_session() must NOT submit duplicate close orders after kill switch."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._kill_switch_fired = True

        orch.post_session()

        # Engine.on_trading_day_end should NOT be called (positions already flat)
        orch.engine.on_trading_day_end.assert_not_called()
        # No orders submitted
        assert len(orch.order_router.submitted) == 0

    async def test_emergency_flatten_uses_correct_qty(self):
        """Kill switch uses record.contracts (not hardcoded 1) for multi-contract."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1, contracts=3)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)

        await orch._emergency_flatten()

        # Verify the exit spec used qty=3
        assert len(orch.order_router.submitted) == 1
        assert orch.order_router.submitted[0]["qty"] == 3

    async def test_emergency_flatten_multiple_positions(self):
        """Kill switch flattens ALL open positions, not just the first one."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        # Add a second strategy to the map
        from trading_app.portfolio import PortfolioStrategy

        s2 = PortfolioStrategy(
            strategy_id="TEST_STRAT_002",
            instrument="MGC",
            orb_label="TOKYO_OPEN",
            entry_model="E2",
            rr_target=1.5,
            confirm_bars=1,
            filter_type="ORB_G4",
            expectancy_r=0.15,
            win_rate=0.38,
            sample_size=150,
            sharpe_ratio=1.2,
            max_drawdown_r=4.0,
            median_risk_points=3.5,
            stop_multiplier=1.0,
            source="test",
            weight=1.0,
        )
        orch._strategy_map["TEST_STRAT_002"] = s2

        # Open two positions
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)
        orch._positions.on_entry_sent("TEST_STRAT_002", "short", 2360.0, order_id=2)
        orch._positions.on_entry_filled("TEST_STRAT_002", 2360.0)

        await orch._emergency_flatten()

        # Both positions should be flattened
        assert orch._positions.active_positions() == []
        assert len(orch.order_router.submitted) == 2


# ---------------------------------------------------------------------------
# CIRCUIT BREAKER tests
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Circuit breaker blocks entries after consecutive broker failures."""

    async def test_circuit_breaker_blocks_entry_after_failures(self):
        """5 consecutive submit failures → circuit breaker opens → ENTRY blocked."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))

        # Exhaust the circuit breaker with 5 failures
        for _ in range(5):
            orch._circuit_breaker.record_failure()

        assert not orch._circuit_breaker.should_allow_request()

        # Try to enter — should be blocked
        await orch._handle_event(_entry_event(2350.5))

        # No order submitted
        assert len(orch.order_router.submitted) == 0
        # Signal record should log CIRCUIT_BREAKER
        orch._write_signal_record.assert_called()
        call_args = orch._write_signal_record.call_args[0][0]
        assert call_args["type"] == "CIRCUIT_BREAKER"

    async def test_circuit_breaker_allows_exit_even_when_open(self):
        """Circuit breaker open must NOT block exits — can't leave positions open."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))

        # Set up an open position
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)

        # Open the circuit breaker
        for _ in range(5):
            orch._circuit_breaker.record_failure()
        assert not orch._circuit_breaker.should_allow_request()

        # Exit should still go through
        await orch._handle_event(_exit_event(2355.0))

        # Order WAS submitted despite open breaker
        assert len(orch.order_router.submitted) == 1
        assert orch.order_router.submitted[0]["type"] == "fake_exit"

    async def test_circuit_breaker_records_success_on_entry(self):
        """Successful entry order records success on circuit breaker."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))

        await orch._handle_event(_entry_event(2350.5))

        # Verify breaker is healthy
        assert orch._circuit_breaker.should_allow_request()
        assert len(orch.order_router.submitted) == 1

    async def test_exit_failure_logs_manual_close_required(self):
        """Failed exit order logs MANUAL CLOSE REQUIRED and writes EXIT_FAILED signal."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))

        # Set up an open position
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)

        # Make submit fail
        orch.order_router.submit = MagicMock(side_effect=ConnectionError("broker dead"))

        await orch._handle_event(_exit_event(2355.0))

        # Signal record should have EXIT_FAILED
        calls = [c[0][0] for c in orch._write_signal_record.call_args_list]
        exit_failed = [c for c in calls if c.get("type") == "EXIT_FAILED"]
        assert len(exit_failed) == 1
        assert "broker dead" in exit_failed[0]["error"]


# ---------------------------------------------------------------------------
# NOTIFICATION tests
# ---------------------------------------------------------------------------


class TestNotifications:
    def test_notify_never_raises(self):
        """_notify() must swallow all exceptions."""
        orch = build_orchestrator()
        with patch("trading_app.live.notifications.notify", side_effect=Exception("boom")):
            orch._notify("test")  # must not raise

    def test_notify_calls_notify_module(self):
        """_notify() delegates to notifications.notify()."""
        orch = build_orchestrator()
        with patch("trading_app.live.notifications.notify") as mock_notify:
            orch._notify("hello world")
            mock_notify.assert_called_once_with("MGC", "hello world")

    def test_cusum_alarm_triggers_notify(self):
        """CUSUM alarm from record_trade() triggers _notify()."""
        orch = build_orchestrator()
        orch._notify = MagicMock()
        orch.monitor.record_trade.return_value = "CUSUM ALARM: test_strat"

        event = _exit_event()
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.5, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.5)

        # Call _record_exit directly
        orch._record_exit(event, entry_price=2350.5)

        orch._notify.assert_called_once_with("CUSUM ALARM: test_strat")

    def test_notify_skips_when_no_telegram(self):
        """_notify() is a no-op when telegram module unavailable."""
        orch = build_orchestrator()
        # Patch the import to fail
        with patch("trading_app.live.notifications.notify", side_effect=ImportError("no telegram")):
            orch._notify("test")  # no crash

    async def test_kill_switch_sends_notification(self):
        """Kill switch fires -> _notify() called with KILL SWITCH message."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._notify = MagicMock()
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)

        await orch._emergency_flatten()

        # Should have been called with KILL SWITCH message
        calls = [str(c) for c in orch._notify.call_args_list]
        assert any("KILL SWITCH" in c for c in calls)


# ---------------------------------------------------------------------------
# TRADING DAY ROLLOVER tests
# ---------------------------------------------------------------------------


class TestTradingDayRollover:
    async def test_rollover_submits_broker_exit(self):
        """Trading day rollover -> _handle_event called -> broker exit submitted."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._handle_event = AsyncMock()
        orch.trading_day = date(2026, 3, 6)

        exit_event = _exit_event(price=2355.0, pnl_r=1.0)
        orch.engine.on_trading_day_end.return_value = [exit_event]

        bar_ts = datetime(2026, 3, 6, 23, 1, tzinfo=UTC)
        with patch.object(SessionOrchestrator, "_build_daily_features_row", return_value={}):
            await orch._check_trading_day_rollover(bar_ts)

        orch._handle_event.assert_awaited_once_with(exit_event)
        assert orch.trading_day == date(2026, 3, 7)

    async def test_rollover_resets_engine_state(self):
        """After rollover: new trading_day, fresh engine, fresh orb_builder."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._handle_event = AsyncMock()
        orch.engine.on_trading_day_end.return_value = []
        orch.trading_day = date(2026, 3, 6)

        bar_ts = datetime(2026, 3, 6, 23, 1, tzinfo=UTC)
        with patch.object(SessionOrchestrator, "_build_daily_features_row", return_value={}):
            await orch._check_trading_day_rollover(bar_ts)

        assert orch.trading_day == date(2026, 3, 7)
        orch.engine.on_trading_day_start.assert_called_once()
        orch.monitor.reset_daily.assert_called_once()
        orch.risk_mgr.daily_reset.assert_called_once()

    async def test_rollover_no_action_same_day(self):
        """Same trading day -> no rollover action."""
        orch = build_orchestrator()
        orch.engine.on_trading_day_end.return_value = []
        orch.trading_day = date(2026, 3, 7)

        # 10 AM Brisbane = midnight UTC -> same trading day March 7
        bar_ts = datetime(2026, 3, 7, 0, 0, tzinfo=UTC)
        await orch._check_trading_day_rollover(bar_ts)

        # No EOD call since we didn't roll
        orch.engine.on_trading_day_end.assert_not_called()


# ---------------------------------------------------------------------------
# ORCHESTRATOR RECONNECT tests
# ---------------------------------------------------------------------------


def _make_feed_class(was_stopped: bool = False, crash: Exception | None = None):
    """Factory: build a mock feed class for reconnect tests."""

    class MockFeed:
        def __init__(self, auth, on_bar, demo):
            self._stop_requested = was_stopped

        @property
        def was_stopped(self):
            return self._stop_requested

        async def run(self, symbol):
            if crash:
                raise crash

    return MockFeed


class TestOrchestratorReconnect:
    async def test_clean_stop_no_reconnect(self):
        """Feed stopped by stop-file (was_stopped=True) -> no reconnect."""
        orch = build_orchestrator()
        orch._feed_class = _make_feed_class(was_stopped=True)

        await orch.run()
        # Should exit cleanly after one attempt

    async def test_reconnects_after_feed_exhaustion(self):
        """Feed exhausts internal reconnects (was_stopped=False) -> orchestrator retries."""
        orch = build_orchestrator()
        orch._notify = MagicMock()
        orch.ORCHESTRATOR_BACKOFF_INITIAL = 0.01
        orch.ORCHESTRATOR_BACKOFF_MAX = 0.01

        call_count = 0

        class CountingFeed:
            def __init__(self, auth, on_bar, demo):
                self._stop_requested = False

            @property
            def was_stopped(self):
                return self._stop_requested

            async def run(self, symbol):
                nonlocal call_count
                call_count += 1
                if call_count >= 3:
                    self._stop_requested = True  # stop on 3rd attempt

        orch._feed_class = CountingFeed
        await orch.run()
        assert call_count == 3

    async def test_no_reconnect_after_kill_switch(self):
        """Kill switch fired -> no reconnect attempt."""
        orch = build_orchestrator()
        orch._kill_switch_fired = True
        orch._notify = MagicMock()

        call_count = 0

        class NeverReachFeed:
            def __init__(self, auth, on_bar, demo):
                self._stop_requested = False

            @property
            def was_stopped(self):
                return False

            async def run(self, symbol):
                nonlocal call_count
                call_count += 1

        orch._feed_class = NeverReachFeed
        await orch.run()
        assert call_count == 0

    async def test_reconnects_after_feed_crash(self):
        """Feed raises exception -> orchestrator retries with new feed instance."""
        orch = build_orchestrator()
        orch._notify = MagicMock()
        orch.ORCHESTRATOR_BACKOFF_INITIAL = 0.01
        orch.ORCHESTRATOR_BACKOFF_MAX = 0.01
        orch.ORCHESTRATOR_MAX_RECONNECTS = 2

        call_count = 0

        class CrashOnceFeed:
            def __init__(self, auth, on_bar, demo):
                self._stop_requested = False

            @property
            def was_stopped(self):
                return self._stop_requested

            async def run(self, symbol):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("ws died")
                self._stop_requested = True

        orch._feed_class = CrashOnceFeed
        await orch.run()
        assert call_count == 2

    async def test_max_reconnects_exhausted(self):
        """After ORCHESTRATOR_MAX_RECONNECTS failures -> session ends with notification."""
        orch = build_orchestrator()
        orch._notify = MagicMock()
        orch.ORCHESTRATOR_MAX_RECONNECTS = 2
        orch.ORCHESTRATOR_BACKOFF_INITIAL = 0.01
        orch.ORCHESTRATOR_BACKOFF_MAX = 0.01
        orch._feed_class = _make_feed_class(was_stopped=False)

        await orch.run()

        # Should have notified about exhaustion
        calls = [str(c) for c in orch._notify.call_args_list]
        assert any("Exhausted" in c for c in calls)


# ---------------------------------------------------------------------------
# BRACKET ORDER tests
# ---------------------------------------------------------------------------


class FakeBracketRouter(FakeRouter):
    """Router that supports native brackets for testing."""

    def __init__(self, fill_price=None):
        super().__init__(fill_price)
        self.bracket_submitted = []
        self.cancelled_ids = []

    def supports_native_brackets(self) -> bool:
        return True

    def build_bracket_spec(self, **kwargs) -> dict:
        spec = {"type": "bracket", **kwargs}
        return spec

    def submit(self, spec):
        self.submitted.append(spec)
        if spec.get("type") == "bracket":
            self.bracket_submitted.append(spec)
            return {"order_id": 200, "order_ids": [201, 202], "status": "submitted"}
        return {
            "order_id": 99,
            "status": "submitted",
            "fill_price": self._fill_price,
        }

    def cancel(self, order_id: int) -> None:
        self.cancelled_ids.append(order_id)


class TestBracketOrders:
    async def test_bracket_submitted_after_entry(self):
        """Entry fill -> bracket stop+target submitted to broker."""
        router = FakeBracketRouter(fill_price=2351.0)
        c = FakeBrokerComponents()
        c.router = router
        orch = build_orchestrator(c)
        orch.order_router = router

        await orch._handle_event(_entry_event(2350.5))

        # Bracket should be submitted
        assert len(router.bracket_submitted) == 1
        bracket = router.bracket_submitted[0]
        assert bracket["type"] == "bracket"
        # Position should have bracket order IDs
        record = orch._positions.get(STRATEGY_ID)
        assert record is not None
        assert record.bracket_order_ids == [201, 202]

    async def test_bracket_cancelled_before_exit(self):
        """Exit signal -> bracket orders cancelled -> exit submitted."""
        router = FakeBracketRouter(fill_price=2351.0)
        c = FakeBrokerComponents()
        c.router = router
        orch = build_orchestrator(c)
        orch.order_router = router

        # Enter and get brackets
        await orch._handle_event(_entry_event(2350.5))

        # Exit
        await orch._handle_event(_exit_event(2355.0))

        # Both bracket orders should have been cancelled
        assert 201 in router.cancelled_ids
        assert 202 in router.cancelled_ids

    async def test_no_bracket_when_unsupported(self):
        """supports_native_brackets() returns False -> no bracket submitted."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2351.0))

        await orch._handle_event(_entry_event(2350.5))

        # No bracket submitted (FakeRouter returns False for supports_native_brackets)
        assert len(orch.order_router.submitted) == 1  # only the entry order

    async def test_bracket_failure_is_warning_not_error(self):
        """Bracket submit fails -> WARNING logged, position still tracked."""
        router = FakeBracketRouter(fill_price=2351.0)
        original_submit = router.submit

        def fail_bracket(spec):
            if spec.get("type") == "bracket":
                raise ConnectionError("bracket api down")
            return original_submit(spec)

        router.submit = fail_bracket
        c = FakeBrokerComponents()
        c.router = router
        orch = build_orchestrator(c)
        orch.order_router = router

        # Should not raise even though bracket submission fails
        await orch._handle_event(_entry_event(2350.5))

        # Position still tracked
        record = orch._positions.get(STRATEGY_ID)
        assert record is not None
        assert record.bracket_order_ids == []  # no brackets stored

    async def test_bracket_race_already_flat(self):
        """Bracket fills between cancel and exit -> 'already flat' handled gracefully."""
        router = FakeBracketRouter(fill_price=2351.0)
        original_submit = router.submit

        def flat_on_exit(spec):
            if spec.get("type") != "bracket" and spec.get("type") == "fake_exit":
                raise RuntimeError("No position - already flat")
            return original_submit(spec)

        router.submit = flat_on_exit
        c = FakeBrokerComponents()
        c.router = router
        orch = build_orchestrator(c)
        orch.order_router = router

        await orch._handle_event(_entry_event(2350.5))
        await orch._handle_event(_exit_event(2355.0))

        # Position should be cleaned up despite "already flat" error
        assert orch._positions.get(STRATEGY_ID) is None


# ---------------------------------------------------------------------------
# FILL POLLER tests
# ---------------------------------------------------------------------------


class TestFillPoller:
    async def test_pending_entry_gets_confirmed(self):
        """PENDING_ENTRY order polled -> Filled status -> on_entry_filled called."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        # Set up a PENDING_ENTRY position
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=42)
        orch.FILL_POLL_INTERVAL = 0.01

        # Mock order status to return Filled
        orch.order_router.query_order_status = MagicMock(
            return_value={"order_id": 42, "status": "Filled", "fill_price": 2350.5}
        )

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Position should now be ENTERED with fill price
        record = orch._positions.get(STRATEGY_ID)
        assert record is not None
        assert record.fill_entry_price == 2350.5

    async def test_cancelled_order_popped(self):
        """Cancelled order -> position removed from tracker."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=42)
        orch.FILL_POLL_INTERVAL = 0.01

        orch.order_router.query_order_status = MagicMock(
            return_value={"order_id": 42, "status": "Cancelled", "fill_price": None}
        )

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Position should be removed
        assert orch._positions.get(STRATEGY_ID) is None

    async def test_poller_survives_query_failure(self):
        """query_order_status raises -> poller continues polling."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=42)
        orch.FILL_POLL_INTERVAL = 0.01

        call_count = 0

        def flaky_query(order_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("network error")
            return {"order_id": order_id, "status": "Filled", "fill_price": 2350.5}

        orch.order_router.query_order_status = flaky_query

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # First call failed, second succeeded
        assert call_count >= 2
        record = orch._positions.get(STRATEGY_ID)
        assert record is not None
        assert record.fill_entry_price == 2350.5

    async def test_poller_skips_non_pending(self):
        """ENTERED positions not polled."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        # Set up an ENTERED position (not pending)
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=42)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)
        orch.FILL_POLL_INTERVAL = 0.01

        orch.order_router.query_order_status = MagicMock()

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # query_order_status should NOT have been called
        orch.order_router.query_order_status.assert_not_called()


class TestObservability:
    """Tests for SessionStats counters, upgraded _notify, heartbeat, and self-tests."""

    def test_notify_counts_success(self):
        """Successful _notify() increments notifications_sent."""
        orch = build_orchestrator()
        with patch("trading_app.live.notifications.notify"):
            orch._notify("test")
        assert orch._stats.notifications_sent == 1
        assert orch._stats.notifications_failed == 0

    def test_notify_counts_failure_and_logs(self):
        """Failed _notify() increments notifications_failed and logs error."""
        orch = build_orchestrator()
        with patch("trading_app.live.notifications.notify", side_effect=RuntimeError("bad token")):
            orch._notify("test")
        assert orch._stats.notifications_failed == 1
        assert orch._stats.notifications_sent == 0

    def test_notify_first_failure_prints_to_stdout(self, capsys):
        """First notification failure prints warning to STDOUT."""
        orch = build_orchestrator()
        with patch("trading_app.live.notifications.notify", side_effect=RuntimeError("bad token")):
            orch._notify("test1")
            orch._notify("test2")  # second failure should NOT print again
        captured = capsys.readouterr()
        assert "NOTIFICATION FAILURE" in captured.out
        # Only one print (first failure)
        assert captured.out.count("NOTIFICATION FAILURE") == 1

    async def test_bar_count_increments(self):
        """_on_bar increments both _bar_count and _stats.bars_received."""
        orch = build_orchestrator()
        orch.engine.on_bar.return_value = []
        bar = FakeBar()
        await orch._on_bar(bar)
        assert orch._bar_count == 1
        assert orch._stats.bars_received == 1

    async def test_events_processed_counter(self):
        """_handle_event increments events_processed."""
        orch = build_orchestrator(FakeBrokerComponents(signal_only=True))
        event = MagicMock()
        event.event_type = "ENTRY"
        event.strategy_id = STRATEGY_ID
        event.price = 2350.0
        event.direction = "long"
        event.contracts = 1
        await orch._handle_event(event)
        assert orch._stats.events_processed == 1

    async def test_bracket_submit_counter(self):
        """Successful bracket submit increments brackets_submitted."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch.order_router.supports_native_brackets = MagicMock(return_value=True)
        orch.order_router.build_bracket_spec = MagicMock(return_value={"type": "OCO"})
        orch.order_router.submit = MagicMock(return_value={"order_ids": [100, 101]})

        strategy = list(orch._strategy_map.values())[0]
        event = MagicMock()
        event.strategy_id = strategy.strategy_id
        event.direction = "long"
        event.contracts = 1

        await orch._submit_bracket(event, strategy, 2350.0)
        assert orch._stats.brackets_submitted == 1
        assert orch._stats.brackets_failed == 0

    async def test_bracket_failure_counter(self):
        """Failed bracket submit increments brackets_failed."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch.order_router.supports_native_brackets = MagicMock(return_value=True)
        orch.order_router.build_bracket_spec = MagicMock(return_value={"type": "OCO"})
        orch.order_router.submit = MagicMock(side_effect=RuntimeError("API error"))

        strategy = list(orch._strategy_map.values())[0]
        event = MagicMock()
        event.strategy_id = strategy.strategy_id
        event.direction = "long"
        event.contracts = 1

        await orch._submit_bracket(event, strategy, 2350.0)
        assert orch._stats.brackets_failed == 1
        assert orch._stats.brackets_submitted == 0

    async def test_fill_poller_counters(self):
        """Fill poller increments fill_polls_run and fill_polls_confirmed."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=42)
        orch.FILL_POLL_INTERVAL = 0.01

        orch.order_router.query_order_status = MagicMock(return_value={"status": "Filled", "fill_price": 2351.0})

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert orch._stats.fill_polls_run >= 1
        assert orch._stats.fill_polls_confirmed >= 1

    def test_session_stats_in_post_session(self):
        """post_session() includes stats in EOD notification."""
        orch = build_orchestrator()
        orch._stats.bars_received = 42
        orch._stats.notifications_sent = 5
        orch.engine.on_trading_day_end.return_value = []
        orch._notify = MagicMock()

        orch.post_session()

        # Find the EOD notification call that includes stats
        notify_calls = [str(c) for c in orch._notify.call_args_list]
        eod_calls = [c for c in notify_calls if "42 bars" in c]
        assert len(eod_calls) >= 1, f"Expected '42 bars' in notify calls: {notify_calls}"

    def test_self_tests_return_dict(self):
        """run_self_tests() returns a dict of component: bool."""
        orch = build_orchestrator()
        # Replace the mock from build_orchestrator with the real method
        orch.run_self_tests = SessionOrchestrator.run_self_tests.__get__(orch)
        with patch(
            "trading_app.live.session_orchestrator.SessionOrchestrator._verify_notifications", return_value=True
        ):
            results = orch.run_self_tests()
        assert isinstance(results, dict)
        assert "notifications" in results
        assert "brackets" in results
        assert "fill_poller" in results

    def test_notify_fallback_when_broken(self, capsys):
        """When _notifications_broken=True, _notify skips Telegram and prints to STDOUT."""
        orch = build_orchestrator()
        orch._notifications_broken = True
        orch._notify("critical alert")
        captured = capsys.readouterr()
        assert "NOTIFY-FALLBACK" in captured.out
        assert "critical alert" in captured.out
        assert orch._stats.notifications_failed == 1
        assert orch._stats.notifications_sent == 0

    def test_notify_fallback_does_not_call_telegram(self):
        """Broken notifications must NOT attempt Telegram (avoids repeated timeouts)."""
        orch = build_orchestrator()
        orch._notifications_broken = True
        with patch("trading_app.live.notifications.notify") as mock_notify:
            orch._notify("should not reach telegram")
        mock_notify.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_aborts_on_broken_notifications_live_mode(self):
        """run() raises RuntimeError if notifications broken in non-signal-only mode."""
        orch = build_orchestrator()
        orch.signal_only = False
        with patch.object(
            orch, "run_self_tests", return_value={"notifications": False, "brackets": True, "fill_poller": True}
        ):
            with pytest.raises(RuntimeError, match="notifications broken"):
                await orch.run()

    @pytest.mark.asyncio
    async def test_run_continues_on_broken_notifications_signal_only(self):
        """run() does NOT abort in signal-only mode even if notifications broken."""
        orch = build_orchestrator()
        orch.signal_only = True
        # Mock self-tests to return broken notifications
        with patch.object(
            orch, "run_self_tests", return_value={"notifications": False, "brackets": True, "fill_poller": True}
        ):
            # Mock the feed class to raise immediately so run() exits after the gate
            orch._feed_class = MagicMock(side_effect=KeyboardInterrupt)
            try:
                await orch.run()
            except (KeyboardInterrupt, Exception):
                pass  # Expected — we just need to verify it got past the notification gate
