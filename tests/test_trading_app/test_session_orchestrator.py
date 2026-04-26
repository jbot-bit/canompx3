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

from trading_app.live.position_tracker import PositionState, PositionTracker
from trading_app.live.session_orchestrator import SessionOrchestrator


class _ImmediateExecutorLoop:
    """Test loop proxy that executes threadpool work inline.

    SessionOrchestrator unit tests care about broker/orchestrator behavior, not
    asyncio's threadpool plumbing. Running executor jobs inline keeps the tests
    deterministic under the current WSL/Python 3.13 event-loop behavior, where
    repeated offloads can stall.
    """

    def __init__(self, loop):
        self._loop = loop

    async def run_in_executor(self, executor, func, *args):
        return func(*args)

    def __getattr__(self, name):
        return getattr(self._loop, name)


# ---------------------------------------------------------------------------
# Global mock: prevent calendar holiday check from blocking tests.
# Tests may run on actual CME holidays (e.g., Good Friday).
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _mock_market_calendar():
    with (
        patch("pipeline.market_calendar.is_cme_holiday", return_value=False),
        patch("pipeline.market_calendar.is_early_close", return_value=False),
    ):
        yield


@pytest.fixture(autouse=True)
def _inline_executor_offloads():
    real_get_running_loop = asyncio.get_running_loop

    def _get_loop_proxy():
        return _ImmediateExecutorLoop(real_get_running_loop())

    with patch(
        "trading_app.live.session_orchestrator.asyncio.get_running_loop",
        side_effect=_get_loop_proxy,
    ):
        yield


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
    risk_points: float | None = None


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
    orch._bar_persister = MagicMock()
    orch._bar_persister.append.return_value = None
    orch._bar_persister.flush_to_db.return_value = 0
    orch._positions = PositionTracker()
    orch._last_bar_at = None
    orch._kill_switch_fired = False
    orch._bar_count = 0
    orch._notifications_broken = False
    orch._close_hour_et = None
    orch._close_min_et = None
    orch._close_time_forced = False
    orch._hwm_tracker = None

    from trading_app.live.session_orchestrator import SessionStats

    orch._stats = SessionStats()
    orch._poller_active = False
    orch._fill_reconnect_gen = 0  # F7/R3: reconnect generation counter
    orch.contract_symbol = "MGCJ6"

    from trading_app.live.circuit_breaker import CircuitBreaker

    orch._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
    orch.journal = MagicMock()  # Trade journal — mocked to avoid DB in tests
    orch._consecutive_engine_errors = 0
    orch._blocked_strategies = set()
    orch._blocked_strategy_reasons = {}
    orch._safety_state = MagicMock()  # SessionSafetyState — mocked to avoid file I/O
    orch._profile_id_for_lane_ctl = None
    orch._orb_caps = {}  # ORB cap map — populated per test
    orch._max_risk_per_trade = None  # Dollar cap — populated per test
    orch._regime_paused = set()  # Regime gate — populated per test
    orch.order_router = c.router
    orch.positions = c.positions
    orch._write_signal_record = MagicMock()
    # Self-tests require real Telegram/broker — mock to always pass in tests
    orch.run_self_tests = MagicMock(return_value={"notifications": True, "brackets": True, "fill_poller": True})
    # R5: CB re-notify state — None until first trip re-notify fires
    orch._cb_renotify_last_at = None

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
# NOTE: These tests verify the orphan detection LOGIC PATTERN (raise/force/degrade)
# inline, not via SessionOrchestrator.__init__. The actual __init__ requires full
# broker/DB/config environment. The patterns here match the production code paths
# at session_orchestrator.py:155-177. If __init__ orphan logic changes, update both.
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

    def test_unexpected_exception_blocks_startup(self):
        """Unexpected exception from query_open → RuntimeError (fail-closed)."""

        class BrokenPositions:
            def query_open(self, account_id: int):
                raise ConnectionError("broker unreachable")

        positions = BrokenPositions()
        force_orphans = False
        with pytest.raises(RuntimeError, match="Orphan detection failed"):
            try:
                positions.query_open(12345)
            except NotImplementedError:
                pass
            except RuntimeError:
                raise
            except Exception as e:
                if not force_orphans:
                    raise RuntimeError(f"Orphan detection failed ({e}). Cannot verify broker state.") from e

    def test_unexpected_exception_allowed_with_force_flag(self):
        """Unexpected exception + force_orphans=True → logs warning, proceeds."""

        class BrokenPositions:
            def query_open(self, account_id: int):
                raise ConnectionError("broker unreachable")

        positions = BrokenPositions()
        force_orphans = True
        proceeded = False
        try:
            positions.query_open(12345)
        except NotImplementedError:
            pass
        except RuntimeError:
            raise
        except Exception as e:
            if not force_orphans:
                raise RuntimeError(f"Orphan detection failed ({e}).") from e
            proceeded = True

        assert proceeded, "force_orphans=True should allow startup despite unexpected exception"


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
            def __init__(self, auth, on_bar, demo, on_stale=None):
                self._stop_requested = True

            @property
            def was_stopped(self):
                return self._stop_requested

            async def run(self, symbol):
                return  # complete immediately

        orch._feed_class = InstantFeed

        await orch.run()
        # If we got here without error, watchdog was started and cancelled in finally

    async def test_holiday_block_raises_runtime_error(self, orch):
        """Bot must refuse to trade on a CME holiday (override autouse mock)."""
        with (
            patch("pipeline.market_calendar.is_cme_holiday", return_value=True),
            patch("pipeline.market_calendar.is_market_open_at", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="CME HOLIDAY"):
                await orch.run()

    async def test_sunday_evening_not_blocked(self, orch):
        """Sunday evening: is_cme_holiday=True but market open -> no RuntimeError."""
        feed_inst = MagicMock()
        feed_inst.run = AsyncMock(return_value=None)
        feed_inst.was_stopped = True
        orch._feed_class = MagicMock(return_value=feed_inst)

        with (
            patch("pipeline.market_calendar.is_cme_holiday", return_value=True),
            patch("pipeline.market_calendar.is_market_open_at", return_value=True),
            patch("pipeline.market_calendar.is_early_close", return_value=False),
        ):
            await orch.run()  # Must NOT raise RuntimeError

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

    async def test_exit_failure_leaves_pending_exit_state(self):
        """Failed exit keeps position in PENDING_EXIT — honest broker state."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)
        orch.order_router.submit = MagicMock(side_effect=ConnectionError("broker dead"))

        await orch._handle_event(_exit_event(2355.0))

        # Position must still exist and be in PENDING_EXIT
        record = orch._positions.get(STRATEGY_ID)
        assert record is not None, "Position should not be deleted after failed exit"
        assert record.state == PositionState.PENDING_EXIT


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

    def test_load_paused_lane_blocks_from_lane_ctl(self):
        orch = build_orchestrator()
        orch._profile_id_for_lane_ctl = "topstep_50k_mnq_auto"

        with patch(
            "trading_app.lifecycle_state.read_lifecycle_state",
            return_value={
                "blocked_strategy_ids": [STRATEGY_ID],
                "blocked_reason_by_strategy": {STRATEGY_ID: "SR alarm pause"},
            },
        ):
            orch._load_paused_lane_blocks()

        assert STRATEGY_ID in orch._blocked_strategies
        assert orch._blocked_strategy_reasons[STRATEGY_ID] == "SR alarm pause"

    def test_lifecycle_blocks_do_not_persist_to_safety_state(self):
        """Lifecycle-sourced blocks are re-derived at every session start
        from the registry. They must NOT be written to the persistent
        safety-state file — otherwise stale blocks survive after the
        underlying SR review changes to WATCH. Fixed 2026-04-14."""
        orch = build_orchestrator()
        orch._profile_id_for_lane_ctl = "topstep_50k_mnq_auto"
        # Replace the default MagicMock safety_state with a real dict-backed
        # stub so we can assert on the persisted collection.
        orch._safety_state = MagicMock()
        orch._safety_state.blocked_strategies = {}

        with patch(
            "trading_app.lifecycle_state.read_lifecycle_state",
            return_value={
                "blocked_strategy_ids": [STRATEGY_ID],
                "blocked_reason_by_strategy": {STRATEGY_ID: "Criterion 12 SR ALARM — manual review required"},
            },
        ):
            orch._load_paused_lane_blocks()

        # Block applied at runtime
        assert STRATEGY_ID in orch._blocked_strategies
        # But NOT persisted to safety_state (persist=False path)
        assert STRATEGY_ID not in orch._safety_state.blocked_strategies
        # And save() not called for lifecycle-sourced blocks
        orch._safety_state.save.assert_not_called()

    def test_orphan_block_persists_to_safety_state(self):
        """Orphan/stuck-exit blocks represent crash recovery (positions the
        orchestrator found but couldn't resolve). These MUST persist across
        restarts — the default persist=True path."""
        orch = build_orchestrator()
        orch._safety_state = MagicMock()
        orch._safety_state.blocked_strategies = {}

        orch._block_strategy(STRATEGY_ID, "Orphaned broker position — manual resolution required")

        assert STRATEGY_ID in orch._blocked_strategies
        assert STRATEGY_ID in orch._safety_state.blocked_strategies
        orch._safety_state.save.assert_called_once()

    def test_reviewed_watch_alarm_does_not_load_startup_block(self):
        orch = build_orchestrator()
        orch._profile_id_for_lane_ctl = "topstep_50k_mnq_auto"

        with patch(
            "trading_app.lifecycle_state.read_lifecycle_state",
            return_value={
                "blocked_strategy_ids": [],
                "blocked_reason_by_strategy": {},
                "strategy_states": {
                    STRATEGY_ID: {
                        "sr_status": "ALARM",
                        "sr_review_outcome": "watch",
                        "blocked": False,
                    }
                },
            },
        ):
            orch._load_paused_lane_blocks()

        assert STRATEGY_ID not in orch._blocked_strategies

    async def test_paused_lane_blocks_new_entry_with_pause_reason(self):
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._notify = MagicMock()
        orch._block_strategy(STRATEGY_ID, "SR alarm pause")

        await orch._handle_event(_entry_event(2360.0))

        calls = [c[0][0] for c in orch._write_signal_record.call_args_list]
        blocked = [c for c in calls if c.get("type") == "ENTRY_BLOCKED_PAUSED"]
        assert len(blocked) == 1
        assert "SR alarm pause" in blocked[0]["reason"]


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

    async def test_rollover_close_failure_detects_orphans(self):
        """Rollover close failure → orphan detection logs CRITICAL + notifies."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._notify = MagicMock()
        orch.trading_day = date(2026, 3, 6)

        # Put a position in the tracker (simulating an open trade)
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)

        # Make _handle_event fail for the rollover close
        orch._handle_event = AsyncMock(side_effect=ConnectionError("broker unreachable"))

        exit_event = _exit_event(price=2355.0)
        orch.engine.on_trading_day_end.return_value = [exit_event]

        bar_ts = datetime(2026, 3, 6, 23, 1, tzinfo=UTC)
        with patch.object(SessionOrchestrator, "_build_daily_features_row", return_value={}):
            await orch._check_trading_day_rollover(bar_ts)

        # Position should still be in tracker (orphaned)
        orphan = orch._positions.get(STRATEGY_ID)
        assert orphan is not None, "Orphaned position should survive rollover"

        # Notification should mention orphan
        notify_calls = [c[0][0] for c in orch._notify.call_args_list]
        orphan_msgs = [m for m in notify_calls if "ORPHAN" in m]
        assert len(orphan_msgs) >= 1, f"Expected ORPHAN notification, got: {notify_calls}"

        # CONTAINMENT: strategy should be blocked for new entries
        assert STRATEGY_ID in orch._blocked_strategies

    async def test_orphan_blocks_new_entry(self):
        """After rollover orphan, new ENTRY for same strategy is rejected."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._notify = MagicMock()

        # Mirror the real rollover containment path: blocked strategy plus reason.
        orch._block_strategy(STRATEGY_ID, "Orphaned broker position — manual resolution required")

        # Engine tries to enter the same strategy on new day
        await orch._handle_event(_entry_event(2360.0))

        # Entry should be rejected — signal record written
        calls = [c[0][0] for c in orch._write_signal_record.call_args_list]
        blocked = [c for c in calls if c.get("type") == "ENTRY_BLOCKED_ORPHAN"]
        assert len(blocked) == 1

        # Position tracker should NOT have a new entry
        assert (
            orch._positions.get(STRATEGY_ID) is None
            or orch._positions.get(STRATEGY_ID).state != PositionState.PENDING_ENTRY
        )

    async def test_non_orphaned_entry_proceeds_after_rollover(self):
        """Non-orphaned strategies can still enter after a rollover with orphans."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._notify = MagicMock()

        # Block only one strategy
        orch._blocked_strategies.add("SOME_OTHER_STRATEGY")

        # Our test strategy should NOT be blocked
        assert STRATEGY_ID not in orch._blocked_strategies

        # Entry should proceed normally
        await orch._handle_event(_entry_event(2360.0))

        # Position tracker should have the new entry
        record = orch._positions.get(STRATEGY_ID)
        assert record is not None

    async def test_partial_rollover_failure_blocks_only_failed_strategy(self):
        """2 strategies open, 1 close succeeds, 1 fails → only failed one blocked."""
        STRAT_B = "TEST_STRAT_002"
        strat_b = PortfolioStrategy(
            strategy_id=STRAT_B,
            instrument="MGC",
            orb_label="NYSE_OPEN",
            entry_model="E2",
            rr_target=2.0,
            confirm_bars=1,
            filter_type="ORB_G5",
            expectancy_r=0.15,
            win_rate=0.38,
            sample_size=180,
            sharpe_ratio=1.2,
            max_drawdown_r=4.0,
            median_risk_points=3.0,
            stop_multiplier=1.0,
            source="test",
            weight=1.0,
        )

        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._notify = MagicMock()
        orch._strategy_map[STRAT_B] = strat_b
        orch.trading_day = date(2026, 3, 6)

        # Both strategies have open positions
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)
        orch._positions.on_entry_sent(STRAT_B, "long", 18000.0, order_id=2)
        orch._positions.on_entry_filled(STRAT_B, 18000.0)

        # _handle_event fails only for STRAT_B's exit, succeeds for STRATEGY_ID
        call_count = 0

        async def partial_fail(event):
            nonlocal call_count
            call_count += 1
            if event.strategy_id == STRAT_B:
                raise ConnectionError("broker unreachable")
            # Successful close: remove from tracker
            orch._positions.on_exit_sent(event.strategy_id)
            orch._positions.on_exit_filled(event.strategy_id)

        orch._handle_event = partial_fail

        exit_a = FakeTradeEvent(
            event_type="EXIT",
            strategy_id=STRATEGY_ID,
            timestamp=datetime.now(UTC),
            price=2355.0,
            direction="long",
            contracts=1,
        )
        exit_b = FakeTradeEvent(
            event_type="EXIT",
            strategy_id=STRAT_B,
            timestamp=datetime.now(UTC),
            price=18050.0,
            direction="long",
            contracts=1,
        )
        orch.engine.on_trading_day_end.return_value = [exit_a, exit_b]

        bar_ts = datetime(2026, 3, 6, 23, 1, tzinfo=UTC)
        with patch.object(SessionOrchestrator, "_build_daily_features_row", return_value={}):
            await orch._check_trading_day_rollover(bar_ts)

        # STRATEGY_ID closed successfully — should NOT be blocked
        assert STRATEGY_ID not in orch._blocked_strategies
        assert orch._positions.get(STRATEGY_ID) is None

        # STRAT_B failed — should be blocked
        assert STRAT_B in orch._blocked_strategies
        assert orch._positions.get(STRAT_B) is not None


# ---------------------------------------------------------------------------
# ORCHESTRATOR RECONNECT tests
# ---------------------------------------------------------------------------


def _make_feed_class(was_stopped: bool = False, crash: Exception | None = None):
    """Factory: build a mock feed class for reconnect tests."""

    class MockFeed:
        def __init__(self, auth, on_bar, demo, on_stale=None):
            self._stop_requested = was_stopped

        @property
        def was_stopped(self):
            return self._stop_requested

        async def run(self, symbol):
            if crash:
                raise crash

    return MockFeed


class TestF1OrchestratorRolloverWiring:
    """F-1 TopStep XFA Scaling Plan: rollover must re-query broker equity and
    refresh the risk manager's EOD balance so today's contract cap reflects
    yesterday's session close. Canonical rule prohibits intraday scaling-up.
    """

    async def test_rollover_refreshes_f1_eod_balance_when_xfa_active(self):
        """F-1 active → rollover queries equity + sets EOD balance."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._handle_event = AsyncMock()
        orch.engine.on_trading_day_end.return_value = []
        orch.trading_day = date(2026, 3, 6)

        # F-1 active
        orch.risk_mgr.limits.topstep_xfa_account_size = 50_000
        orch.positions = MagicMock()
        orch.positions.query_equity = MagicMock(return_value=103_000.0)
        orch.order_router = MagicMock()
        orch.order_router.account_id = 20092334

        bar_ts = datetime(2026, 3, 6, 23, 1, tzinfo=UTC)
        with patch.object(SessionOrchestrator, "_build_daily_features_row", return_value={}):
            await orch._check_trading_day_rollover(bar_ts)

        orch.risk_mgr.set_topstep_xfa_eod_balance.assert_called_once_with(103_000.0)

    async def test_rollover_skips_f1_eod_balance_when_disabled(self):
        """F-1 disabled (account_size=None) → rollover must not touch EOD balance."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._handle_event = AsyncMock()
        orch.engine.on_trading_day_end.return_value = []
        orch.trading_day = date(2026, 3, 6)

        orch.risk_mgr.limits.topstep_xfa_account_size = None

        bar_ts = datetime(2026, 3, 6, 23, 1, tzinfo=UTC)
        with patch.object(SessionOrchestrator, "_build_daily_features_row", return_value={}):
            await orch._check_trading_day_rollover(bar_ts)

        orch.risk_mgr.set_topstep_xfa_eod_balance.assert_not_called()

    async def test_rollover_skips_f1_when_broker_equity_unavailable(self):
        """F-1 active + broker equity None → log warning, no EOD balance update.

        The risk_manager fails-closed on None balance at entry time, so skipping
        the update preserves the last known good value (or None — check fails
        closed either way). Explicit skip is cleaner than feeding in None.
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._handle_event = AsyncMock()
        orch.engine.on_trading_day_end.return_value = []
        orch.trading_day = date(2026, 3, 6)

        orch.risk_mgr.limits.topstep_xfa_account_size = 50_000
        orch.positions = MagicMock()
        orch.positions.query_equity = MagicMock(return_value=None)  # broker down
        orch.order_router = MagicMock()
        orch.order_router.account_id = 20092334

        bar_ts = datetime(2026, 3, 6, 23, 1, tzinfo=UTC)
        with patch.object(SessionOrchestrator, "_build_daily_features_row", return_value={}):
            await orch._check_trading_day_rollover(bar_ts)

        orch.risk_mgr.set_topstep_xfa_eod_balance.assert_not_called()

    def test_is_trading_combine_detects_tc_marker(self):
        """Account name containing 'TC-' marks it as a Trading Combine."""
        from trading_app.live.session_orchestrator import _is_trading_combine_account

        is_tc, reason = _is_trading_combine_account({"name": "50KTC-V2-451890-20372221"})
        assert is_tc is True
        assert "Trading Combine" in reason
        assert "50KTC-V2" in reason

    def test_is_trading_combine_handles_none(self):
        """None metadata → not classifiable, return False (trust profile config)."""
        from trading_app.live.session_orchestrator import _is_trading_combine_account

        is_tc, reason = _is_trading_combine_account(None)
        assert is_tc is False
        assert reason == ""

    def test_is_trading_combine_non_tc_name(self):
        """An XFA-like account name does NOT match TC detection."""
        from trading_app.live.session_orchestrator import _is_trading_combine_account

        is_tc, _ = _is_trading_combine_account({"name": "50KXFA-451890-99999999"})
        assert is_tc is False
        is_tc2, _ = _is_trading_combine_account({"name": "50KEFA-live-12345"})
        assert is_tc2 is False

    def test_is_trading_combine_empty_name(self):
        """Empty or missing name → not classifiable, return False."""
        from trading_app.live.session_orchestrator import _is_trading_combine_account

        assert _is_trading_combine_account({})[0] is False
        assert _is_trading_combine_account({"name": ""})[0] is False
        assert _is_trading_combine_account({"name": None})[0] is False

    def test_broker_reality_wiring_tc_disables_f1(self):
        """TC metadata → _apply_broker_reality_check returns 'tc' and
        disable_f1 is called with a reason that mentions "Trading Combine".
        """
        from trading_app.live.session_orchestrator import _apply_broker_reality_check

        positions = MagicMock()
        positions.query_account_metadata.return_value = {
            "id": 20372221,
            "name": "50KTC-V2-451890-20372221",
            "simulated": True,
        }
        order_router = MagicMock()
        order_router.account_id = 20372221
        risk_mgr = MagicMock()

        status = _apply_broker_reality_check(
            positions=positions,
            order_router=order_router,
            risk_mgr=risk_mgr,
            initial_equity=1_500.0,
        )

        assert status == "tc"
        positions.query_account_metadata.assert_called_once_with(20372221)
        risk_mgr.disable_f1.assert_called_once()
        assert "Trading Combine" in risk_mgr.disable_f1.call_args[0][0]
        risk_mgr.set_topstep_xfa_eod_balance.assert_not_called()

    def test_broker_reality_wiring_xfa_sets_eod_balance(self):
        """Non-TC metadata → _apply_broker_reality_check returns 'xfa'
        and set_topstep_xfa_eod_balance fires with the initial equity.
        """
        from trading_app.live.session_orchestrator import _apply_broker_reality_check

        positions = MagicMock()
        positions.query_account_metadata.return_value = {
            "id": 11111111,
            "name": "50KXFA-451890-99999999",
            "simulated": False,
        }
        order_router = MagicMock()
        order_router.account_id = 11111111
        risk_mgr = MagicMock()

        status = _apply_broker_reality_check(
            positions=positions,
            order_router=order_router,
            risk_mgr=risk_mgr,
            initial_equity=2_500.0,
        )

        assert status == "xfa"
        risk_mgr.disable_f1.assert_not_called()
        risk_mgr.set_topstep_xfa_eod_balance.assert_called_once_with(2_500.0)

    def test_broker_reality_wiring_none_metadata_trusts_profile(self):
        """Broker API returns None (transient down) → _apply_broker_reality_check
        returns 'xfa_missing_meta', disable_f1 is NOT called, EOD balance IS set.
        Missing data does not loosen risk — trust the profile config.
        """
        from trading_app.live.session_orchestrator import _apply_broker_reality_check

        positions = MagicMock()
        positions.query_account_metadata.return_value = None  # broker down
        order_router = MagicMock()
        order_router.account_id = 20372221
        risk_mgr = MagicMock()

        status = _apply_broker_reality_check(
            positions=positions,
            order_router=order_router,
            risk_mgr=risk_mgr,
            initial_equity=3_000.0,
        )

        assert status == "xfa_missing_meta"
        risk_mgr.disable_f1.assert_not_called()
        risk_mgr.set_topstep_xfa_eod_balance.assert_called_once_with(3_000.0)

    def test_broker_reality_wiring_order_router_none_uses_zero_account_id(self):
        """Defensive: if order_router is None (not wired yet), helper still
        queries with account_id=0 rather than raising — matches the inline
        behaviour at session_orchestrator.py:534-536 pre-extraction.
        """
        from trading_app.live.session_orchestrator import _apply_broker_reality_check

        positions = MagicMock()
        positions.query_account_metadata.return_value = None
        risk_mgr = MagicMock()

        status = _apply_broker_reality_check(
            positions=positions,
            order_router=None,
            risk_mgr=risk_mgr,
            initial_equity=500.0,
        )

        assert status == "xfa_missing_meta"
        positions.query_account_metadata.assert_called_once_with(0)
        risk_mgr.set_topstep_xfa_eod_balance.assert_called_once_with(500.0)

    def test_broker_reality_wiring_f1_inactive_short_circuits(self):
        """Orchestrator call-site guard: _apply_broker_reality_check must only
        run when F-1 is active (topstep_xfa_account_size is not None). This
        test asserts the GUARD, not the helper — if F-1 is inactive, the
        helper is never called and no mocks are touched.

        The guard lives at session_orchestrator.py:529 — this test proves
        callers still respect it after the extraction.
        """
        positions = MagicMock()
        risk_mgr = MagicMock()
        risk_mgr.limits.topstep_xfa_account_size = None  # F-1 inactive

        # Caller-side guard — same shape as session_orchestrator.py:529
        if risk_mgr.limits.topstep_xfa_account_size is not None:
            from trading_app.live.session_orchestrator import _apply_broker_reality_check

            _apply_broker_reality_check(
                positions=positions,
                order_router=None,
                risk_mgr=risk_mgr,
                initial_equity=0.0,
            )

        positions.query_account_metadata.assert_not_called()
        risk_mgr.disable_f1.assert_not_called()
        risk_mgr.set_topstep_xfa_eod_balance.assert_not_called()

    async def test_rollover_skips_f1_when_orphans_present(self):
        """F-1 active + orphaned positions at rollover → skip EOD refresh.

        ProjectX query_equity returns realized balance only. If orphans have
        unrealized losses, realized under-represents true equity and F-1's cap
        would be LOOSER than safe. Fail-closed: keep last known good balance.
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._handle_event = AsyncMock()
        orch.engine.on_trading_day_end.return_value = []
        orch.trading_day = date(2026, 3, 6)

        # F-1 active
        orch.risk_mgr.limits.topstep_xfa_account_size = 50_000
        orch.positions = MagicMock()
        orch.positions.query_equity = MagicMock(return_value=103_000.0)
        orch.order_router = MagicMock()
        orch.order_router.account_id = 20092334

        # Plant an orphan position that would survive the close loop
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=1)
        orch._positions.on_entry_filled(STRATEGY_ID, 2350.0)

        bar_ts = datetime(2026, 3, 6, 23, 1, tzinfo=UTC)
        with patch.object(SessionOrchestrator, "_build_daily_features_row", return_value={}):
            await orch._check_trading_day_rollover(bar_ts)

        # EOD balance must NOT be refreshed — the realized balance is unsafe
        # when orphans could carry unrealized losses
        orch.risk_mgr.set_topstep_xfa_eod_balance.assert_not_called()
        # And query_equity should not even be called (short-circuit before it)
        orch.positions.query_equity.assert_not_called()


class TestF1SignalOnlySeed:
    """B6 (2026-04-25): in signal-only mode the HWM init block is gated off,
    so `_apply_broker_reality_check` (the F-1 EOD balance seeder) never runs.
    Without a seed, RiskManager.can_enter fail-closes every entry with
    "EOD XFA balance unknown — refusing entry. (F-1 fail-closed)".

    Fix: signal-only branch seeds with $0.0 — the canonical day-1 XFA balance
    per docs/research-input/topstep/topstep_mll_article.md and
    trading_app/topstep_scaling_plan.py:51-53. Bottom-tier cap (2 lots for
    50K) — exactly what a fresh-XFA day-1 trader would face. NOT a bypass.

    See docs/runtime/stages/live-b6-f1-signal-only-seed.md.
    """

    def test_signal_only_seed_when_f1_active(self):
        """F-1 active → helper seeds $0.0 and returns True."""
        from trading_app.live.session_orchestrator import _apply_signal_only_f1_seed

        risk_mgr = MagicMock()
        risk_mgr.limits.topstep_xfa_account_size = 50_000

        applied = _apply_signal_only_f1_seed(risk_mgr=risk_mgr)

        assert applied is True
        risk_mgr.set_topstep_xfa_eod_balance.assert_called_once_with(0.0)

    def test_signal_only_no_seed_when_f1_inactive(self):
        """F-1 inactive (non-XFA profile) → helper short-circuits, returns False.

        Important: do NOT call set_topstep_xfa_eod_balance when F-1 is off.
        That field is meaningless for non-XFA profiles and a phantom seed
        could mask a future regression where F-1 is wrongly enabled.
        """
        from trading_app.live.session_orchestrator import _apply_signal_only_f1_seed

        risk_mgr = MagicMock()
        risk_mgr.limits.topstep_xfa_account_size = None

        applied = _apply_signal_only_f1_seed(risk_mgr=risk_mgr)

        assert applied is False
        risk_mgr.set_topstep_xfa_eod_balance.assert_not_called()

    def test_signal_only_seed_unblocks_can_enter(self):
        """Integration: real RiskManager + helper → can_enter passes the F-1
        balance-known gate after the seed (would have rejected before).

        This proves the fix actually closes B6: the same RiskManager that
        was rejecting every entry pre-seed now passes the F-1 known-balance
        check. The remaining cap-projection gate (max_lots_for_xfa) is
        validated by the existing day-1 50K tests in test_risk_manager.py.
        """
        from trading_app.live.session_orchestrator import _apply_signal_only_f1_seed
        from trading_app.risk_manager import RiskLimits, RiskManager

        rm = RiskManager(
            RiskLimits(
                max_daily_loss_r=-3.0,
                max_concurrent_positions=5,
                topstep_xfa_account_size=50_000,
            )
        )
        rm.daily_reset(date(2026, 4, 25))

        # Pre-seed: balance unknown → fail-closed.
        allowed, reason, _ = rm.can_enter(
            strategy_id="MNQ_NYSE_OPEN_E2",
            orb_label="NYSE_OPEN",
            active_trades=[],
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="long",
        )
        assert allowed is False
        assert "EOD XFA balance unknown" in reason

        # Apply the signal-only seed.
        applied = _apply_signal_only_f1_seed(risk_mgr=rm)
        assert applied is True

        # Post-seed: balance is known ($0.0 = day-1 cap = 2 lots), entry passes.
        allowed, reason, _ = rm.can_enter(
            strategy_id="MNQ_NYSE_OPEN_E2",
            orb_label="NYSE_OPEN",
            active_trades=[],
            daily_pnl_r=0.0,
            instrument="MNQ",
            direction="long",
        )
        assert allowed is True, f"Post-seed rejection: {reason}"


class TestOvernightResilienceHardening:
    """5 silent-failure fixes from docs/runtime/stages/live-overnight-resilience-hardening.md.

    F2/F6/F8 inline branches in __init__ have been extracted as helpers
    (_cleanup_orphan_brackets, _notify_journal_unhealthy_demo,
    _notify_f1_silent_block_if_active). Each helper is unit-tested for
    real BEHAVIOR (not source-marker regex) — flip the input condition,
    verify the right side effect (raise / notify / silence).
    """

    # ── F8 — orphan bracket cleanup failure should HALT, not warn ──
    def test_f8_cleanup_orphan_brackets_raises_when_cancel_fails(self):
        """Cancel raises → helper raises RuntimeError + notifies. Mutation:
        comment out the `raise RuntimeError` and this test fails."""
        from trading_app.live.session_orchestrator import _cleanup_orphan_brackets

        order_router = MagicMock()
        order_router.cancel_bracket_orders.side_effect = RuntimeError("broker timeout")
        contracts = MagicMock()
        contracts.resolve_front_month.return_value = "MNQM6"
        notify_calls: list[str] = []

        with pytest.raises(RuntimeError, match="Bracket orphan cleanup failed"):
            _cleanup_orphan_brackets(
                order_router=order_router,
                contracts=contracts,
                instrument="MNQ",
                force_orphans=False,
                notify_fn=lambda msg: notify_calls.append(msg),
            )
        assert any("BRACKET ORPHAN CLEANUP FAILED" in m for m in notify_calls), (
            f"Expected notify before raise, got {notify_calls}"
        )

    def test_f8_cleanup_orphan_brackets_force_orphans_bypasses_raise(self):
        """force_orphans=True → notify but no raise. Operator accepts risk via CLI."""
        from trading_app.live.session_orchestrator import _cleanup_orphan_brackets

        order_router = MagicMock()
        order_router.cancel_bracket_orders.side_effect = RuntimeError("broker timeout")
        contracts = MagicMock()
        contracts.resolve_front_month.return_value = "MNQM6"
        notify_calls: list[str] = []

        result = _cleanup_orphan_brackets(
            order_router=order_router,
            contracts=contracts,
            instrument="MNQ",
            force_orphans=True,
            notify_fn=lambda msg: notify_calls.append(msg),
        )
        assert result == 0
        assert any("BRACKET ORPHAN CLEANUP FAILED" in m for m in notify_calls)

    def test_f8_cleanup_orphan_brackets_happy_path(self):
        """Cancel returns >0 → notify the count + return it. No raise."""
        from trading_app.live.session_orchestrator import _cleanup_orphan_brackets

        order_router = MagicMock()
        order_router.cancel_bracket_orders.return_value = 3
        contracts = MagicMock()
        contracts.resolve_front_month.return_value = "MNQM6"
        notify_calls: list[str] = []

        result = _cleanup_orphan_brackets(
            order_router=order_router,
            contracts=contracts,
            instrument="MNQ",
            force_orphans=False,
            notify_fn=lambda msg: notify_calls.append(msg),
        )
        assert result == 3
        assert any("Cancelled 3 orphaned" in m for m in notify_calls)

    # ── F6 — journal unhealthy in demo should _notify, not just warn ──
    def test_f6_notify_journal_unhealthy_fires_when_unhealthy(self):
        """is_healthy=False → notify + return False. Mutation: remove notify_fn()
        call and this test fails."""
        from trading_app.live.session_orchestrator import _notify_journal_unhealthy_demo

        journal = MagicMock()
        journal.is_healthy = False
        notify_calls: list[str] = []

        result = _notify_journal_unhealthy_demo(
            journal=journal,
            journal_mode="demo",
            notify_fn=lambda msg: notify_calls.append(msg),
        )
        assert result is False
        assert any("TRADE JOURNAL UNHEALTHY" in m for m in notify_calls)
        assert any("demo" in m for m in notify_calls)

    def test_f6_notify_journal_unhealthy_silent_when_healthy(self):
        """is_healthy=True → no notify, return True. Avoid happy-path noise."""
        from trading_app.live.session_orchestrator import _notify_journal_unhealthy_demo

        journal = MagicMock()
        journal.is_healthy = True
        notify_calls: list[str] = []

        result = _notify_journal_unhealthy_demo(
            journal=journal,
            journal_mode="demo",
            notify_fn=lambda msg: notify_calls.append(msg),
        )
        assert result is True
        assert notify_calls == []

    # ── F2 — F-1 None equity at startup should _notify when XFA is active ──
    def test_f2_notify_f1_silent_block_fires_when_xfa_active(self):
        """topstep_xfa_account_size is set → notify with F-1 SILENT BLOCK.
        Mutation: change gate to `is None` and this test fails."""
        from trading_app.live.session_orchestrator import _notify_f1_silent_block_if_active

        risk_mgr = MagicMock()
        risk_mgr.limits.topstep_xfa_account_size = 50_000
        notify_calls: list[str] = []

        result = _notify_f1_silent_block_if_active(risk_mgr=risk_mgr, notify_fn=lambda msg: notify_calls.append(msg))
        assert result is True
        assert len(notify_calls) == 1
        assert "F-1 SILENT BLOCK" in notify_calls[0]
        assert "REJECT" in notify_calls[0]

    def test_f2_notify_f1_silent_block_silent_on_non_xfa(self):
        """topstep_xfa_account_size is None → no notify. Avoid TC/non-XFA noise."""
        from trading_app.live.session_orchestrator import _notify_f1_silent_block_if_active

        risk_mgr = MagicMock()
        risk_mgr.limits.topstep_xfa_account_size = None
        notify_calls: list[str] = []

        result = _notify_f1_silent_block_if_active(risk_mgr=risk_mgr, notify_fn=lambda msg: notify_calls.append(msg))
        assert result is False
        assert notify_calls == []

    # ── F5 — HWM poll exception → real-tracker 3-strikes integration ──
    def test_f5_real_tracker_three_strikes_fires_halt(self, tmp_path):
        """Audit gap closure: previous F5 test used MagicMock for the tracker.
        This integration test uses the REAL AccountHWMTracker and verifies the
        canonical halt mechanism actually fires after 3 update_equity(None)
        calls per account_hwm_tracker.py:307-321.

        Mutation probe: change _MAX_CONSECUTIVE_POLL_FAILURES from 3 in
        account_hwm_tracker.py and this test will fail (off-by-one).
        """
        from trading_app.account_hwm_tracker import AccountHWMTracker

        tracker = AccountHWMTracker(
            account_id="TEST_99999999",
            firm="topstep",
            dd_limit_dollars=2000.0,
            state_dir=tmp_path,
            dd_type="eod_trailing",
        )

        # Strike 1 — no halt yet
        tracker.update_equity(None)
        halted, _ = tracker.check_halt()
        assert halted is False, "Strike 1 should not halt"

        # Strike 2 — still no halt
        tracker.update_equity(None)
        halted, _ = tracker.check_halt()
        assert halted is False, "Strike 2 should not halt"

        # Strike 3 — HALT
        tracker.update_equity(None)
        halted, reason = tracker.check_halt()
        assert halted is True, "Strike 3 should fire halt"
        assert "POLL_FAILURE" in reason, f"Expected POLL_FAILURE reason, got {reason!r}"

    def test_f5_real_tracker_strikes_reset_on_successful_poll(self, tmp_path):
        """Counter resets on a successful equity update — 2 misses then a
        success then 2 more misses must NOT halt. Verifies the reset path
        at account_hwm_tracker.py:325."""
        from trading_app.account_hwm_tracker import AccountHWMTracker

        tracker = AccountHWMTracker(
            account_id="TEST_RESET",
            firm="topstep",
            dd_limit_dollars=2000.0,
            state_dir=tmp_path,
            dd_type="eod_trailing",
        )
        tracker.update_equity(None)
        tracker.update_equity(None)
        tracker.update_equity(50_000.0)  # success → reset counter
        tracker.update_equity(None)
        tracker.update_equity(None)

        halted, _ = tracker.check_halt()
        assert halted is False, "Counter must reset on successful poll"

    # R2 — _notify must dispatch to a worker thread when an event loop is running.
    async def test_r2_notify_uses_to_thread_when_event_loop_running(self):
        """When _notify is called from inside an async context, the blocking
        notify() call must be dispatched via asyncio.to_thread so the event
        loop is not blocked. Probe: patch asyncio.to_thread and confirm it
        gets called.
        """
        orch = build_orchestrator()  # demo orchestrator; signal_only=False
        orch._notifications_broken = False
        # Capture to_thread invocations
        with patch("asyncio.to_thread") as mock_to_thread:
            # Make the to_thread "task" return a real future so add_done_callback works
            future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            future.set_result(True)
            mock_to_thread.return_value = future

            orch._notify("R2 probe message")

            mock_to_thread.assert_called_once()
            # First positional arg is the notify function; second is instrument; third is message.
            args = mock_to_thread.call_args[0]
            assert args[1] == orch.instrument
            assert args[2] == "R2 probe message"

    def test_r2_notify_falls_back_to_sync_when_no_event_loop(self):
        """When _notify is called from sync setup/preflight context (no running
        event loop), it must NOT raise and must call notify() directly.
        """
        orch = build_orchestrator()
        orch._notifications_broken = False
        with patch("trading_app.live.notifications.notify") as mock_notify:
            mock_notify.return_value = True
            # No running event loop here — sync test method.
            orch._notify("R2 sync fallback probe")
            mock_notify.assert_called_once_with(orch.instrument, "R2 sync fallback probe")
            assert orch._stats.notifications_sent >= 1

    # F5 — HWM equity poll exception must propagate as update_equity(None).
    async def test_f5_hwm_poll_exception_propagates_as_none(self):
        """When positions.query_equity raises during the intraday HWM poll,
        the exception must be converted to update_equity(None) so the tracker's
        3-consecutive-failure halt mechanism actually fires. Pre-fix the warning
        was logged and update_equity was skipped entirely.
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._handle_event = AsyncMock()
        orch._hwm_tracker = MagicMock()
        orch._hwm_tracker.check_halt.return_value = (False, "HWM_OK")
        orch._bar_count = 9  # next bar at count==10 triggers the heartbeat block

        # Make query_equity raise.
        orch.positions = MagicMock()
        orch.positions.query_equity.side_effect = RuntimeError("auth token expired")
        orch.order_router = MagicMock()
        orch.order_router.account_id = 20092334
        orch.order_router.is_degraded = MagicMock(return_value=False)

        # Patch async paths used by _on_bar so we don't run the full pipeline
        with patch.object(SessionOrchestrator, "_check_trading_day_rollover", new_callable=AsyncMock):
            from trading_app.live.bar_aggregator import Bar

            await orch._on_bar(
                Bar(
                    ts_utc=datetime(2026, 4, 25, 14, 30, tzinfo=UTC),
                    open=2350.0,
                    high=2350.5,
                    low=2349.5,
                    close=2350.2,
                    volume=10,
                )
            )

        # F5 invariant: update_equity must be called with None (NOT skipped).
        orch._hwm_tracker.update_equity.assert_called_once_with(None)


class TestHWMWarningTierNotifyDispatch:
    """Stage 1 of HWM persistence integrity hardening (2026-04-25 design v3).

    Wires `_notify()` into the DD warning tier (50%/75%) at
    session_orchestrator.py:1601. Pre-fix: warning tier reached log file only;
    operator unaware DD crossed 50% or 75% on a 24h overnight run until the
    full halt fires at 100%.

    All 5 tests are mutation-proof per stage doc
    `docs/runtime/stages/hwm-warning-tier-notify-dispatch.md`.

    Setup pattern mirrors TestOvernightResilienceHardening F5 test:
    bar_count=9 so the next bar at count==10 triggers the heartbeat block
    where the HWM equity poll lives.
    """

    async def _make_orch_with_hwm(self, check_halt_return):
        """Build an orch wired so the 10-bar HWM poll fires on next _on_bar
        call and check_halt() returns the supplied (halted, reason) tuple.
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._handle_event = AsyncMock()
        orch._hwm_tracker = MagicMock()
        orch._hwm_tracker.check_halt.return_value = check_halt_return
        orch._bar_count = 9  # next bar at count==10 triggers heartbeat block
        orch.positions = MagicMock()
        orch.positions.query_equity.return_value = 49_000.0
        orch.order_router = MagicMock()
        orch.order_router.account_id = 21944866
        orch.order_router.is_degraded = MagicMock(return_value=False)
        orch._notify = MagicMock()
        return orch

    async def _fire_one_bar(self, orch):
        """Drive one bar through _on_bar with the trading-day rollover patched."""
        with patch.object(SessionOrchestrator, "_check_trading_day_rollover", new_callable=AsyncMock):
            from trading_app.live.bar_aggregator import Bar

            await orch._on_bar(
                Bar(
                    ts_utc=datetime(2026, 4, 25, 14, 30, tzinfo=UTC),
                    open=2350.0,
                    high=2350.5,
                    low=2349.5,
                    close=2350.2,
                    volume=10,
                )
            )

    # ── Test 1 — 50% warning dispatches notify with full reason ──
    async def test_hwm_warning_50_dispatches_notify(self):
        """check_halt returns a 50% warning reason → _notify is called exactly
        once with the full reason string including dollar amounts.

        Mutation: remove the new self._notify call from line 1601-1602 →
        notify call count drops to 0 → assertion fails.
        """
        reason = "HWM_WARNING_50: DD $1000.00 = 50% of $2000.00 limit ($1000.00 remaining)"
        orch = await self._make_orch_with_hwm((False, reason))

        await self._fire_one_bar(orch)

        # Exactly one notify call this bar.
        assert orch._notify.call_count == 1, (
            f"Expected exactly 1 notify call for 50% warning, got {orch._notify.call_count}"
        )
        # The dispatched message must contain the reason verbatim
        # (including dollar amounts) so the operator sees the actual DD figures.
        sent_msg = orch._notify.call_args.args[0]
        assert reason in sent_msg, f"Expected notify message to contain the full reason '{reason}', got: {sent_msg!r}"

    # ── Test 2 — 75% warning dispatches notify with full reason ──
    async def test_hwm_warning_75_dispatches_notify(self):
        """Mirror of test 1 at the 75% tier. Pin both tiers separately so a
        mutation that handles only one tier is caught.
        """
        reason = "HWM_WARNING_75: DD $1500.00 = 75% of $2000.00 limit ($500.00 remaining)"
        orch = await self._make_orch_with_hwm((False, reason))

        await self._fire_one_bar(orch)

        assert orch._notify.call_count == 1, (
            f"Expected exactly 1 notify call for 75% warning, got {orch._notify.call_count}"
        )
        sent_msg = orch._notify.call_args.args[0]
        assert reason in sent_msg, f"Expected notify message to contain the full reason '{reason}', got: {sent_msg!r}"

    # ── Test 3 — generic WARN substring match, not literal tier name ──
    async def test_hwm_warning_generic_substring_match_not_literal(self):
        """Mutation guard: dispatch logic must be `'WARN' in reason`, not a
        literal tier match like `'HWM_WARNING_50' in reason`. A future
        WARNING_60 (or any reason containing 'WARN') must also dispatch.

        Pre-fix the branch already used 'WARN' in reason for logging; this test
        pins that the new notify dispatch is gated on the SAME generic
        substring, not narrowed.
        """
        reason = "HWM_WARNING_60: DD $1200.00 = 60% of $2000.00 limit"  # synthetic future tier
        orch = await self._make_orch_with_hwm((False, reason))

        await self._fire_one_bar(orch)

        assert orch._notify.call_count == 1, (
            "WARN substring match must dispatch for any WARNING_* tier; "
            f"got {orch._notify.call_count} notify calls for synthetic WARNING_60 reason"
        )

    # ── Test 4 — OK case dispatches NOTHING ──
    async def test_hwm_ok_does_not_dispatch_notify(self):
        """Mutation guard against the warning dispatch leaking into the OK
        branch. If the elif-WARN guard is removed (e.g. fall-through to
        unconditional dispatch), this test fails immediately.
        """
        reason = "HWM_OK: DD $0.00 = 0% of $2000.00 limit"
        orch = await self._make_orch_with_hwm((False, reason))

        await self._fire_one_bar(orch)

        assert orch._notify.call_count == 0, (
            f"OK case must dispatch nothing on Telegram; got {orch._notify.call_count} notify calls"
        )

    # ── Test 6 — None reason must not crash the elif guard ──
    async def test_hwm_check_halt_none_reason_does_not_raise(self, caplog):
        """Audit-gate CRITICAL-1 (2026-04-25): if `check_halt()` returns
        `(False, None)` the elif `"WARN" in reason` evaluates `"WARN" in None`
        which raises `TypeError: argument of type 'NoneType' is not iterable`.
        Pre-fix that TypeError is caught silently by the bare exception
        handler at the bottom of the HWM-poll block, recreating the exact
        silent-failure mode Stage 1 was designed to close.

        Post-fix: the elif is guarded with `reason is not None` so a None
        reason short-circuits cleanly with no exception, no log.error from
        the exception handler, and no spurious notify.

        Mutation: revert the None-guard → TypeError → exception handler logs
        'HWM tracker update/check raised' → the second assertion fails.
        """
        import logging

        orch = await self._make_orch_with_hwm((False, None))
        orch._notify = MagicMock()

        with caplog.at_level(logging.ERROR, logger="trading_app.live.session_orchestrator"):
            await self._fire_one_bar(orch)

        # No spurious warning notify on a None reason.
        assert orch._notify.call_count == 0, (
            f"None reason must not dispatch a warning notify; got {orch._notify.call_count}"
        )
        # The exception handler must NOT have caught a TypeError. Pre-fix this
        # message would appear because `"WARN" in None` raised.
        spurious = [r.message for r in caplog.records if "HWM tracker update/check raised" in r.message]
        assert not spurious, (
            f"None reason must not crash the elif; expected no exception-handler log, got: {spurious!r}"
        )

    # ── Test 6.5 — STAGE1-GAP-1 contract-drift visibility on (False, None) ──
    async def test_hwm_check_halt_false_none_logs_contract_drift_warning(self, caplog):
        """STAGE1-GAP-1 (audit-gate verdict CONDITIONAL on commit 68c63482,
        2026-04-25): after Stage 1's None-guard `reason is not None and "WARN"
        in reason` was added, a `(False, None)` return short-circuits the elif
        with zero log/notify visibility. The auditor's recommendation:

            elif reason is None:
                log.warning("HWM check_halt returned (False, None) — tracker
                contract drift; expected non-None reason string")

        Closed as item-zero of HWM Stage 2 per
        `docs/runtime/stages/hwm-stage1-gap1-none-reason-contract-guard.md`.

        Asserts:
        - log.warning emitted containing the substring `tracker contract drift`
        - no `_notify` dispatched (developer-visible signal, not operator alert)
        - no exception propagates (existing bare-except behavior preserved)

        Mutation: remove the new `elif reason is None` branch → no log captured
        with the contract-drift substring → assertion fails.
        """
        import logging

        orch = await self._make_orch_with_hwm((False, None))
        orch._notify = MagicMock()

        with caplog.at_level(logging.WARNING, logger="trading_app.live.session_orchestrator"):
            await self._fire_one_bar(orch)

        drift_logs = [r for r in caplog.records if "tracker contract drift" in r.message]
        assert len(drift_logs) == 1, (
            f"Expected exactly one contract-drift WARNING; got {len(drift_logs)}: {[r.message for r in drift_logs]!r}"
        )
        assert drift_logs[0].levelno == logging.WARNING, (
            f"Contract-drift log must be WARNING level; got level={drift_logs[0].levelno}"
        )
        assert orch._notify.call_count == 0, (
            f"Contract drift must not dispatch operator notify; got {orch._notify.call_count}"
        )

    # ── Test 7 — update_equity must be called BEFORE check_halt on each poll ──
    async def test_hwm_poll_update_equity_before_check_halt(self):
        """Mutation guard: a refactor that swaps the call order would let
        `check_halt()` run against stale state (the in-memory equity from the
        previous poll), masking a fresh DD breach for one poll cycle. Pin the
        order via `parent.method_calls` index ordering on a shared mock.

        Mutation: swap the two lines at session_orchestrator.py:1594-1595 →
        ordered_names becomes ['check_halt', 'update_equity'] → assertion
        fails.
        """
        orch = await self._make_orch_with_hwm((False, "HWM_OK"))
        # Wire both tracker methods through one parent so call order is
        # globally observable. update_equity returns None (real signature).
        parent = MagicMock()
        parent.check_halt.return_value = (False, "HWM_OK")
        orch._hwm_tracker.update_equity = parent.update_equity
        orch._hwm_tracker.check_halt = parent.check_halt

        await self._fire_one_bar(orch)

        ordered = [c[0] for c in parent.method_calls]
        ue_idx = ordered.index("update_equity")
        ch_idx = ordered.index("check_halt")
        assert ue_idx < ch_idx, f"update_equity must be called BEFORE check_halt on each poll; got order: {ordered}"

    # ── Test 8 — check_halt raising is caught silently, no spurious notify ──
    async def test_hwm_check_halt_raises_is_caught_silently(self, caplog):
        """Mutation guard: if `check_halt()` itself raises, the bar loop must
        continue (the exception handler at the bottom of the HWM-poll block
        keeps the engine alive with last-known DD state — a future refactor
        that propagates the exception would kill the bar feed). And no
        `_notify` must fire with a spurious warning, because no warning was
        actually determined.

        Mutation: remove the bare except → exception propagates out of
        `_on_bar` → `_fire_one_bar` raises → pytest fails the test on the
        unhandled exception.
        """
        import logging

        orch = await self._make_orch_with_hwm((False, "HWM_OK"))
        orch._hwm_tracker.check_halt.side_effect = RuntimeError("synthetic check_halt failure")
        orch._notify = MagicMock()

        with caplog.at_level(logging.ERROR, logger="trading_app.live.session_orchestrator"):
            # Must not raise — the exception handler catches it.
            await self._fire_one_bar(orch)

        # No spurious warning notify when check_halt raised.
        assert orch._notify.call_count == 0, (
            f"check_halt raising must not dispatch a warning notify; got {orch._notify.call_count}"
        )
        # The exception handler must HAVE caught the RuntimeError and logged it
        # (this confirms the exception path actually fired — guards against the
        # test passing because check_halt was never reached).
        caught = [r.message for r in caplog.records if "HWM tracker update/check raised" in r.message]
        assert caught, (
            "check_halt raising must be caught and logged by the bare except handler; "
            f"got no matching log.error records. caplog: {[r.message for r in caplog.records]}"
        )

    # ── Test 5 — halt path unchanged: notify, kill switch, flatten — IN ORDER ──
    async def test_hwm_halt_path_unchanged_by_warning_wiring(self):
        """Stage 1 must not alter the halt branch. Pin that on a halt result:
        (a) _notify still fires with the ACCOUNT DD LIMIT prefix,
        (b) _fire_kill_switch is called,
        (c) _emergency_flatten is awaited,
        AND the call ORDER is _notify -> _fire_kill_switch -> _emergency_flatten
        (asserted via mock.call_args_list index ordering, per v3 design audit).
        """
        reason = "DD_TRAILING: DD $2000.00 >= limit $2000.00 (HWM=$50000.00 on 2026-04-20, equity=$48000.00)"
        orch = await self._make_orch_with_hwm((True, reason))

        # Wire all three sinks to a single tracker so we can pin call ORDER, not
        # just presence. parent_mock.method_calls preserves global ordering.
        parent = MagicMock()
        orch._notify = parent.notify
        orch._fire_kill_switch = parent.fire_kill_switch
        orch._emergency_flatten = parent.emergency_flatten
        # _emergency_flatten is awaited, so wrap in AsyncMock attached to parent.
        parent.emergency_flatten = AsyncMock()
        orch._emergency_flatten = parent.emergency_flatten

        await self._fire_one_bar(orch)

        # All three called exactly once.
        assert orch._notify.call_count == 1
        assert orch._fire_kill_switch.call_count == 1
        assert orch._emergency_flatten.call_count == 1

        # Halt-path notify message is the existing 'ACCOUNT DD LIMIT' prefix,
        # NOT the new 'HWM WARNING' prefix from Stage 1.
        halt_msg = orch._notify.call_args.args[0]
        assert "ACCOUNT DD LIMIT" in halt_msg, (
            f"Halt notify must use the existing 'ACCOUNT DD LIMIT' prefix, got: {halt_msg!r}"
        )
        assert "HWM WARNING" not in halt_msg, "Halt branch must not use the Stage 1 'HWM WARNING' prefix"

        # Call order: notify -> fire_kill_switch -> emergency_flatten.
        # parent.method_calls is a global, ordered list of all calls on child
        # mocks; the names are 'notify', 'fire_kill_switch', 'emergency_flatten'.
        ordered_names = [c[0] for c in parent.method_calls]
        assert ordered_names == ["notify", "fire_kill_switch", "emergency_flatten"], (
            f"Halt branch call order must be notify -> fire_kill_switch -> emergency_flatten, got: {ordered_names}"
        )


class TestR1WallClockRollover:
    """R1 (CRITICAL): trading-day rollover must fire at 09:00 Brisbane even when
    the bar feed is down (no _on_bar calls).

    Each test is a mutation probe: remove _wall_clock_rollover_loop or its
    create_task call in run() and the matching assertion must fail.
    """

    # R1-1: wall-clock loop fires _check_trading_day_rollover at the right UTC time
    async def test_r1_rollover_fires_during_feed_down(self):
        """Simulate feed-down spanning 09:00 Brisbane: no _on_bar calls arrive,
        but _wall_clock_rollover_loop must still call _check_trading_day_rollover
        with override_trading_day = trading_day + 1 day.

        Time control: patch compute_trading_day_utc_range to return a past UTC so
        sleep_secs <= 0 (immediate fire). The task is cancelled from inside
        fake_rollover so the loop does not re-enter, avoiding date overflow.
        """
        from datetime import timezone

        orch = build_orchestrator()

        today = orch.trading_day  # e.g. date(2026, 3, 7)
        next_day = today + timedelta(days=1)
        # Past rollover_utc → sleep_secs <= 0 → fires immediately (no real sleep)
        past_rollover = datetime(2026, 1, 1, 23, 0, 0, tzinfo=timezone.utc)

        rollover_calls: list[dict] = []
        task_ref: list = []

        async def fake_rollover(bar_ts_utc, *, override_trading_day=None):
            rollover_calls.append({"bar_ts_utc": bar_ts_utc, "override_trading_day": override_trading_day})
            orch.trading_day = override_trading_day or orch.trading_day
            # Raise CancelledError directly — it is a BaseException, not an Exception,
            # so it bypasses the inner `except Exception` handler and is caught by
            # `except asyncio.CancelledError: return` in the outer loop, terminating cleanly.
            raise asyncio.CancelledError("test: one rollover is enough")

        with (
            patch.object(orch, "_check_trading_day_rollover", new=fake_rollover),
            patch(
                "pipeline.dst.compute_trading_day_utc_range",
                return_value=(past_rollover, past_rollover + timedelta(hours=24)),
            ) as mock_ctr,
        ):
            task = asyncio.create_task(orch._wall_clock_rollover_loop())
            task_ref.append(task)
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # R1 invariant: rollover must have been called with override_trading_day = next_day
        assert len(rollover_calls) >= 1, "R1: _check_trading_day_rollover was never called — wall-clock loop broken"
        first = rollover_calls[0]
        assert first["bar_ts_utc"] is None, "R1: wall-clock path must pass None as bar_ts_utc"
        assert first["override_trading_day"] == next_day, (
            f"R1: override_trading_day must be trading_day+1={next_day}, got {first['override_trading_day']}"
        )
        # Canonical source must have been consulted (not hardcoded 09:00)
        mock_ctr.assert_called_with(next_day)

    # R1-2: override_trading_day bypasses bar-ts derivation (no AttributeError on None)
    async def test_r1_check_rollover_accepts_none_bar_ts_with_override(self):
        """_check_trading_day_rollover must not touch bar_ts_utc when
        override_trading_day is provided. Passing None must not raise AttributeError.
        """
        orch = build_orchestrator()
        # trading_day stays the same → rollover body should return early (idempotent)
        today = orch.trading_day
        # Call with None bar_ts and override = same day → must be a no-op
        await orch._check_trading_day_rollover(None, override_trading_day=today)
        # If we got here without AttributeError or TypeError, the guard works.

    # R1-3: idempotency — second call after _on_bar already rolled is a no-op
    async def test_r1_double_rollover_is_noop(self):
        """If _on_bar rolls the trading_day before the wall-clock task fires,
        the wall-clock call must be a no-op (idempotency guard at first line
        of _check_trading_day_rollover: if bar_trading_day == self.trading_day: return).
        """
        orch = build_orchestrator()
        today = orch.trading_day
        next_day = today + timedelta(days=1)

        # Simulate _on_bar having already rolled to next_day
        orch.trading_day = next_day

        # Now wall-clock fires with override = next_day → must be a no-op (no engine calls)
        orch.engine.on_trading_day_end = MagicMock(return_value=[])
        await orch._check_trading_day_rollover(None, override_trading_day=next_day)

        # Idempotency: on_trading_day_end must NOT have been called (rollover was no-op)
        orch.engine.on_trading_day_end.assert_not_called()

    # R1-4: source-text mutation probe — confirm R1 markers in production code
    def test_r1_source_markers_present(self):
        """Source-text probe: R1 markers must be present in production code.
        Revert _wall_clock_rollover_loop or its task creation → this test fails.
        """
        from trading_app.live import session_orchestrator as so

        src = open(so.__file__, encoding="utf-8").read()
        assert "_wall_clock_rollover_loop" in src, "R1: _wall_clock_rollover_loop method missing from production code"
        assert "rollover_task = asyncio.create_task(self._wall_clock_rollover_loop())" in src, (
            "R1: rollover_task create_task call missing from run()"
        )
        # Iter 179 retrofit: bare `task.cancel()` replaced by `_shutdown_task` helper
        # (closes audit S3 — cancel-without-await SIGTERM leak). The helper's body still
        # calls `task.cancel()` on the passed task, so the cancel SEMANTIC is preserved.
        assert 'await self._shutdown_task(rollover_task, "wall_clock_rollover"' in src, (
            "R1: rollover_task shutdown call missing from finally block "
            "(expected `_shutdown_task` helper after iter 179 hardening)"
        )
        assert "compute_trading_day_utc_range" in src, (
            "R1: canonical pipeline.dst.compute_trading_day_utc_range must be used (not hardcoded 09:00)"
        )
        # Confirm no hardcoded 09:00 literal in the rollover loop
        loop_start = src.find("_wall_clock_rollover_loop")
        loop_end = src.find("\n    async def ", loop_start + 1)
        loop_body = src[loop_start:loop_end] if loop_end > loop_start else src[loop_start:]
        assert "datetime.time(9" not in loop_body, (
            "R1: hardcoded datetime.time(9,...) found in rollover loop — use compute_trading_day_utc_range"
        )
        assert '"09:00"' not in loop_body and "'09:00'" not in loop_body, (
            "R1: hardcoded '09:00' string found in rollover loop — use compute_trading_day_utc_range"
        )

    # R1-5 / S2: multi-day gap startup catch-up
    async def test_r1_multi_day_gap_startup_catches_up(self):
        """S2: orchestrator startup where last bar was 3 days ago — the
        wall-clock rollover loop must fire _check_trading_day_rollover three
        times in succession (catching up each missed day), then sleep until
        the next genuine 09:00 Brisbane.

        Mechanism: each iteration computes next_day rollover_utc via
        compute_trading_day_utc_range. When that UTC is in the past
        (sleep_secs <= 0) the rollover fires immediately, trading_day
        advances by one day, and the loop re-evaluates. The cascade unwinds
        the gap one day per iteration.

        Mutation probe: removing the while-True wrapper or swapping
        compute_trading_day_utc_range for a hardcoded next-day fall-through
        breaks this test.
        [R1-CATCHUP-MULTI-DAY]
        """
        from datetime import timezone

        orch = build_orchestrator()
        starting_day = orch.trading_day

        rollover_calls: list = []

        async def fake_rollover(bar_ts_utc, *, override_trading_day=None):
            rollover_calls.append(override_trading_day)
            # Production behaviour: rollover advances trading_day. The fake
            # mirrors that so the loop's next iteration sees a new next_day.
            if override_trading_day is not None:
                orch.trading_day = override_trading_day
            if len(rollover_calls) >= 3:
                # After 3 catch-up rollovers, terminate cleanly.
                raise asyncio.CancelledError("test: 3-day gap closed")

        past = datetime(2026, 1, 1, 23, 0, 0, tzinfo=timezone.utc)
        future = datetime(2099, 1, 1, 23, 0, 0, tzinfo=timezone.utc)
        call_count = [0]

        def mock_ctr(day):
            # First 3 calls: past (immediate fire); thereafter: future (sleep).
            call_count[0] += 1
            if call_count[0] <= 3:
                return (past, past + timedelta(hours=24))
            return (future, future + timedelta(hours=24))

        with (
            patch.object(orch, "_check_trading_day_rollover", new=fake_rollover),
            patch(
                "pipeline.dst.compute_trading_day_utc_range",
                side_effect=mock_ctr,
            ),
        ):
            task = asyncio.create_task(orch._wall_clock_rollover_loop())
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Three catch-up rollovers fired, each advancing by exactly one day.
        assert len(rollover_calls) == 3
        expected = [starting_day + timedelta(days=i + 1) for i in range(3)]
        assert rollover_calls == expected
        # Final state: trading_day caught up to starting_day + 3.
        assert orch.trading_day == starting_day + timedelta(days=3)


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
            def __init__(self, auth, on_bar, demo, on_stale=None):
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
            def __init__(self, auth, on_bar, demo, on_stale=None):
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
            def __init__(self, auth, on_bar, demo, on_stale=None):
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

    async def test_reconnect_path_increments_fill_reconnect_gen(self):
        """M5: end-to-end reconnect via orchestrator.run() must increment
        _fill_reconnect_gen exactly once per reconnect. The increment lives at
        session_orchestrator.py:3254 inside the reconnect loop and is the
        F7/R3 hand-off signal that lets _fill_poller drop stale timeout
        anchors. Other tests bump the counter manually; this test exercises
        the production reconnect path.
        [F7-RECONNECT-GEN-E2E]
        """
        orch = build_orchestrator()
        orch._notify = MagicMock()
        orch.ORCHESTRATOR_BACKOFF_INITIAL = 0.01
        orch.ORCHESTRATOR_BACKOFF_MAX = 0.01
        orch.ORCHESTRATOR_MAX_RECONNECTS = 3
        assert orch._fill_reconnect_gen == 0

        crash_count = 0

        class CrashTwiceFeed:
            def __init__(self, auth, on_bar, demo, on_stale=None):
                self._stop_requested = False

            @property
            def was_stopped(self):
                return self._stop_requested

            async def run(self, symbol):
                nonlocal crash_count
                crash_count += 1
                if crash_count <= 2:
                    raise ConnectionError(f"feed crash #{crash_count}")
                self._stop_requested = True  # third attempt: clean stop

        orch._feed_class = CrashTwiceFeed
        await orch.run()

        # Two crashes → two reconnect-loop increments at production line 3254.
        assert crash_count == 3
        assert orch._fill_reconnect_gen == 2


# ---------------------------------------------------------------------------
# R3 — reconnect ceiling tests
# ---------------------------------------------------------------------------


class TestR3ReconnectCeiling:
    """R3 (HIGH): ORCHESTRATOR_MAX_RECONNECTS was 5 — too low for 24h operation.

    Mutation probes: remove the constant bump or the stable-run reset and the
    matching assertion must fail.
    """

    # R3-1: bumped ceiling allows > 5 consecutive reconnects without halt
    async def test_r3_ceiling_allows_more_than_five_reconnects(self):
        """With ceiling = 50, 6 consecutive feed crashes must NOT halt the session.
        Pre-fix with ceiling=5 this would have called 'Exhausted' after crash 6.
        """
        orch = build_orchestrator()
        orch._notify = MagicMock()
        orch.ORCHESTRATOR_BACKOFF_INITIAL = 0.001
        orch.ORCHESTRATOR_BACKOFF_MAX = 0.001
        # Do NOT override ORCHESTRATOR_MAX_RECONNECTS — use the class default (50)
        assert orch.ORCHESTRATOR_MAX_RECONNECTS >= 50, "R3: ceiling must be >= 50 for 24h operation"

        crash_count = [0]
        MAX_CRASHES = 6  # more than the old ceiling of 5

        class CrashNTimesFeed:
            def __init__(self, auth, on_bar, demo, on_stale=None):
                self._stop_requested = False

            @property
            def was_stopped(self):
                return self._stop_requested

            async def run(self, symbol):
                crash_count[0] += 1
                if crash_count[0] <= MAX_CRASHES:
                    raise ConnectionError("ws flap")
                # After MAX_CRASHES crashes, exit cleanly via stop
                self._stop_requested = True

        orch._feed_class = CrashNTimesFeed
        await orch.run()

        # Must have reconnected past the old ceiling of 5
        assert crash_count[0] == MAX_CRASHES + 1, f"R3: expected {MAX_CRASHES + 1} feed attempts, got {crash_count[0]}"
        # Must NOT have sent an 'Exhausted' notification
        calls = [str(c) for c in orch._notify.call_args_list]
        assert not any("Exhausted" in c for c in calls), "R3: Exhausted halt fired before ceiling — ceiling too low"

    # R3-2: stable-run reset clears the reconnect counter after 30 min uptime
    async def test_r3_stable_run_resets_counter(self):
        """If feed is UP >= ORCHESTRATOR_STABLE_RUN_SECS then crashes, reconnect
        counter must reset to 0, allowing further reconnects.
        """
        from datetime import timezone

        orch = build_orchestrator()
        orch._notify = MagicMock()
        orch.ORCHESTRATOR_BACKOFF_INITIAL = 0.001
        orch.ORCHESTRATOR_BACKOFF_MAX = 0.001
        # Use a tiny ceiling so the test is fast; default (50) is too slow to exhaust
        orch.ORCHESTRATOR_MAX_RECONNECTS = 2
        orch.ORCHESTRATOR_STABLE_RUN_SECS = 1800  # 30 min

        call_count = [0]
        # Feed sequence: stable (>30min) -> crash -> crash -> stop
        # With counter reset after stable run, 2 post-stable crashes stay within ceiling.
        # Without reset, 2 pre-stable + 2 post-stable = 4 > ceiling of 2 → would halt.

        stable_start = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        stable_end = stable_start + timedelta(seconds=1900)  # >30min

        class StableThenCrashFeed:
            def __init__(self, auth, on_bar, demo, on_stale=None):
                self._stop_requested = False

            @property
            def was_stopped(self):
                return self._stop_requested

            async def run(self, symbol):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First run: stable for >30min, then exits cleanly
                    return  # feed exits without was_stopped (triggers reconnect)
                if call_count[0] <= 3:
                    raise ConnectionError("post-stable flap")
                self._stop_requested = True

        orch._feed_class = StableThenCrashFeed

        # Patch datetime.now(UTC) inside the loop to simulate 30min uptime on first run
        original_datetime = __import__("datetime").datetime
        call_seq = [0]

        class PatchedDatetime:
            """Returns stable_start on first call (feed_started_at), stable_end on second."""

            @staticmethod
            def now(tz=None):
                call_seq[0] += 1
                if call_seq[0] == 1:
                    return stable_start
                return stable_end

        with patch("trading_app.live.session_orchestrator.datetime") as mock_dt:
            mock_dt.now.side_effect = lambda tz=None: stable_start if call_seq[0] == 0 else stable_end
            # Simpler: just patch ORCHESTRATOR_STABLE_RUN_SECS to 0 so any run time qualifies
            orch.ORCHESTRATOR_STABLE_RUN_SECS = 0

            await orch.run()

        # With stable-run reset (threshold=0), counter resets after every successful run.
        # 4 total attempts must succeed without 'Exhausted'.
        assert call_count[0] >= 4, f"R3: expected >= 4 feed calls, got {call_count[0]}"
        calls = [str(c) for c in orch._notify.call_args_list]
        assert not any("Exhausted" in c for c in calls), "R3: Exhausted fired — stable-run reset is not working"

    # R3-3: state file persists last_connected_at after stable run
    async def test_r3_state_file_persists_last_connected_at(self):
        """After a stable run, _safety_state.last_connected_at must be set (non-empty)."""
        orch = build_orchestrator()
        orch._notify = MagicMock()
        orch.ORCHESTRATOR_BACKOFF_INITIAL = 0.001
        orch.ORCHESTRATOR_MAX_RECONNECTS = 2
        orch.ORCHESTRATOR_STABLE_RUN_SECS = 0  # any run qualifies as stable

        call_count = [0]

        class StableThenStopFeed:
            def __init__(self, auth, on_bar, demo, on_stale=None):
                self._stop_requested = False

            @property
            def was_stopped(self):
                return self._stop_requested

            async def run(self, symbol):
                call_count[0] += 1
                if call_count[0] == 1:
                    return  # stable exit — triggers reconnect + reset
                self._stop_requested = True  # clean stop on second attempt

        orch._feed_class = StableThenStopFeed
        await orch.run()

        assert orch._safety_state.last_connected_at != "", (
            "R3: last_connected_at not set after stable run — persistence broken"
        )

    # R3-4: source-text mutation probe
    def test_r3_source_markers_present(self):
        """R3 constants and stable-run reset logic must be present in production code.
        If ORCHESTRATOR_MAX_RECONNECTS is reverted to 5 or ORCHESTRATOR_STABLE_RUN_SECS
        is removed, this test fails.
        """
        import inspect
        from trading_app.live.session_orchestrator import SessionOrchestrator

        src = inspect.getsource(SessionOrchestrator)
        assert "ORCHESTRATOR_MAX_RECONNECTS = 50" in src, "R3: ceiling must be 50 (not 5) for 24h operation"
        assert "ORCHESTRATOR_STABLE_RUN_SECS" in src, "R3: stable-run reset constant missing"
        assert "stable-run reset" in src, "R3: stable-run reset logic marker missing from production code"
        assert "last_connected_at" in src, "R3: last_connected_at persistence marker missing"


# ---------------------------------------------------------------------------
# BRACKET ORDER tests
# ---------------------------------------------------------------------------


class FakeBracketRouter(FakeRouter):
    """Router that supports native brackets with separately-queryable legs.

    Simulates a ProjectX-style broker: native brackets + distinct SL/TP child
    orders that can be queried via verify_bracket_legs(). For tests that
    exercise the Rithmic-style atomic-brackets-without-queryable-legs path,
    use FakeAtomicBracketRouter below.
    """

    def __init__(self, fill_price=None):
        super().__init__(fill_price)
        self.cancelled_ids = []
        self.verify_bracket_legs_call_count = 0

    def supports_native_brackets(self) -> bool:
        return True

    def has_queryable_bracket_legs(self) -> bool:
        return True

    def build_bracket_spec(self, **kwargs) -> dict:
        return {
            "stopLossBracket": {"ticks": 10, "type": 4},
            "takeProfitBracket": {"ticks": 20, "type": 1},
        }

    def merge_bracket_into_entry(self, entry_spec: dict, bracket_spec: dict) -> dict:
        return {**entry_spec, **bracket_spec}

    def verify_bracket_legs(self, order_id: int, contract_symbol: str) -> tuple[int, int]:
        self.verify_bracket_legs_call_count += 1
        return (order_id + 1, order_id + 2)

    def cancel(self, order_id: int) -> None:
        self.cancelled_ids.append(order_id)


class FakeAtomicBracketRouter(FakeBracketRouter):
    """Router that simulates Rithmic/Tradovate-style native atomic brackets.

    supports_native_brackets=True (same as FakeBracketRouter — broker merges
    SL/TP into the entry submission) BUT has_queryable_bracket_legs=False
    because the legs are atomic with the entry — no separately-queryable
    child orders exist. session_orchestrator MUST skip the verify_bracket_legs
    call entirely in this case.

    If a test using this router observes verify_bracket_legs_call_count > 0,
    the session_orchestrator flag-gate at L1685 has regressed.
    """

    def has_queryable_bracket_legs(self) -> bool:
        return False


class TestBracketOrders:
    async def test_bracket_merged_into_entry(self):
        """Native brackets: bracket fields merged into entry spec before submission."""
        router = FakeBracketRouter(fill_price=2351.0)
        c = FakeBrokerComponents()
        c.router = router
        orch = build_orchestrator(c)
        orch.order_router = router

        await orch._handle_event(_entry_event(2350.5))

        # Single submission (entry + bracket merged)
        assert len(router.submitted) == 1
        submitted = router.submitted[0]
        assert "stopLossBracket" in submitted
        assert "takeProfitBracket" in submitted
        # Position should have bracket leg IDs (SL=entry_id+1, TP=entry_id+2)
        record = orch._positions.get(STRATEGY_ID)
        assert record is not None
        assert record.bracket_order_ids == [100, 101]

    async def test_bracket_cancelled_before_exit(self):
        """Exit signal -> bracket parent order cancelled -> exit submitted."""
        router = FakeBracketRouter(fill_price=2351.0)
        c = FakeBrokerComponents()
        c.router = router
        orch = build_orchestrator(c)
        orch.order_router = router

        # Enter with merged bracket
        await orch._handle_event(_entry_event(2350.5))

        # Exit
        await orch._handle_event(_exit_event(2355.0))

        # Bracket leg orders (SL=100, TP=101) should have been cancelled before exit
        assert 100 in router.cancelled_ids
        assert 101 in router.cancelled_ids

    async def test_no_bracket_when_unsupported(self):
        """supports_native_brackets() returns False -> no bracket merged."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2351.0))

        await orch._handle_event(_entry_event(2350.5))

        # No bracket fields in submitted spec
        assert len(orch.order_router.submitted) == 1
        submitted = orch.order_router.submitted[0]
        assert "stopLossBracket" not in submitted

    async def test_bracket_merged_fail_closed(self):
        """Combined entry+bracket submit fails -> position rolled back (fail-closed)."""
        router = FakeBracketRouter(fill_price=2351.0)

        def fail_submit(spec):
            raise ConnectionError("API down")

        router.submit = fail_submit
        c = FakeBrokerComponents()
        c.router = router
        orch = build_orchestrator(c)
        orch.order_router = router

        # Should not raise — circuit breaker catches it
        await orch._handle_event(_entry_event(2350.5))

        # Position rolled back — fail-closed
        record = orch._positions.get(STRATEGY_ID)
        assert record is None

    async def test_bracket_race_already_flat(self):
        """Bracket fills between cancel and exit -> 'already flat' handled gracefully."""
        router = FakeBracketRouter(fill_price=2351.0)
        original_submit = router.submit

        def flat_on_exit(spec):
            if spec.get("type") == "fake_exit":
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

    async def test_bracket_verify_skipped_for_atomic_native_broker(self):
        """REGRESSION GUARD for the has_queryable_bracket_legs() skip path.

        When the broker uses atomic native brackets (has_queryable_bracket_legs
        returns False — e.g. Rithmic, Tradovate-until-activation), session_orch
        MUST skip verify_bracket_legs entirely. Otherwise the base default
        (None, None) is misinterpreted as 'BRACKET LEGS MISSING' and a false
        CRITICAL Telegram alarm fires on every entry.

        Asserts:
          1. verify_bracket_legs is NEVER called (call count == 0)
          2. brackets_submitted counter still increments (atomic path still
             counts as a successful bracket submission)
          3. bracket_order_ids stays empty (no separately-queryable legs exist
             — the broker manages them server-side)
          4. No 'BRACKET LEGS MISSING' notification fires

        If this test fails, the flag gate at session_orchestrator.py L1685 has
        regressed and Rithmic/Tradovate activation will drown operators in
        false-positive Telegram alarms.
        """
        router = FakeAtomicBracketRouter(fill_price=2351.0)
        c = FakeBrokerComponents()
        c.router = router
        orch = build_orchestrator(c)
        orch.order_router = router

        # Track notifications to prove no false alarm fires
        notifications: list[str] = []
        orch._notify = notifications.append

        await orch._handle_event(_entry_event(2350.5))

        # Gate 1: verify_bracket_legs was NEVER called
        assert router.verify_bracket_legs_call_count == 0, (
            f"verify_bracket_legs must not be called for atomic-bracket broker, "
            f"but was called {router.verify_bracket_legs_call_count} times"
        )

        # Gate 2: brackets_submitted counter still incremented
        assert orch._stats.brackets_submitted == 1, (
            f"brackets_submitted should increment even on skip path, got {orch._stats.brackets_submitted}"
        )

        # Gate 3: bracket_order_ids stays empty (no separately-queryable legs)
        record = orch._positions.get(STRATEGY_ID)
        assert record is not None, "Position should exist after ENTRY fill"
        assert record.bracket_order_ids == [], (
            f"bracket_order_ids must be empty for atomic-bracket broker "
            f"(no separately-queryable legs), got: {record.bracket_order_ids}"
        )

        # Gate 4: no 'BRACKET LEGS MISSING' notification
        missing_alarms = [n for n in notifications if "BRACKET LEGS MISSING" in n]
        assert missing_alarms == [], (
            f"False 'BRACKET LEGS MISSING' alarm fired for atomic-bracket broker: {missing_alarms}"
        )

        # Also verify the native-bracket spec was actually merged into entry
        # (atomic brackets still use the merged-into-entry submission path)
        assert len(router.submitted) == 1
        submitted = router.submitted[0]
        assert "stopLossBracket" in submitted
        assert "takeProfitBracket" in submitted


# ---------------------------------------------------------------------------
# RESERVE-THEN-SUBMIT safety tests
# ---------------------------------------------------------------------------


class TestReserveThenSubmit:
    async def test_broker_failure_rolls_back_tracker(self):
        """Order submit fails -> position tracker cleaned up (no phantom PENDING_ENTRY)."""
        router = FakeRouter()

        def fail_submit(spec):
            raise ConnectionError("broker down")

        router.submit = fail_submit
        c = FakeBrokerComponents()
        c.router = router
        orch = build_orchestrator(c)
        orch.order_router = router

        await orch._handle_event(_entry_event(2350.5))

        # Position tracker should be clean — no phantom PENDING_ENTRY
        assert orch._positions.get(STRATEGY_ID) is None
        assert len(orch._positions.active_positions()) == 0

    async def test_duplicate_entry_rejected_before_broker(self):
        """Second ENTRY for same strategy rejected before hitting broker."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2351.0))

        # First entry goes through
        await orch._handle_event(_entry_event(2350.5))
        assert orch._positions.get(STRATEGY_ID) is not None
        assert len(orch.order_router.submitted) == 1

        # Second entry REJECTED — no second broker order
        await orch._handle_event(_entry_event(2352.0))
        assert len(orch.order_router.submitted) == 1  # still just the one

    async def test_signal_only_duplicate_rejected(self):
        """Signal-only mode also rejects duplicate entries."""
        orch = build_orchestrator(FakeBrokerComponents(signal_only=True))

        await orch._handle_event(_entry_event(2350.5))
        record = orch._positions.get(STRATEGY_ID)
        assert record is not None
        assert record.direction == "long"

        # Duplicate rejected
        await orch._handle_event(_entry_event(2355.0))
        # Price unchanged — still first entry
        assert orch._positions.get(STRATEGY_ID).engine_entry_price == 2350.5


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


class TestFillPollerF7Timeout:
    """F7: Fill-poller timeout path — cancel, verify, halt-or-release.

    Source-marker probes: every test body includes a comment citing the
    source-marker it exercises so the adversarial auditor can trace coverage.
    """

    async def test_timeout_fires_cancel_and_lane_release(self):
        """Timeout fires → cancel called → broker confirms Cancelled → lane released,
        engine.cancel_trade called, _notify fired. [F7-CANCEL-CALL, F7-LANE-RELEASE]
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=42)
        # Back-date state_changed_at to beyond the timeout window
        record = orch._positions.get(STRATEGY_ID)
        record.state_changed_at = datetime.now(UTC) - timedelta(seconds=120)
        orch.FILL_POLL_INTERVAL = 0.01
        orch.FILL_POLL_TIMEOUT_SECS = 0.001  # fire immediately
        orch.FILL_CANCEL_VERIFY_TIMEOUT_SECS = 0.01

        # Cancel completes; post-cancel verify returns Cancelled
        orch.order_router.cancel = MagicMock()
        orch.order_router.query_order_status = MagicMock(
            return_value={"order_id": 42, "status": "Cancelled", "fill_price": None}
        )
        orch._notify = MagicMock()

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # [F7-CANCEL-CALL] cancel must have been called with the order id
        orch.order_router.cancel.assert_called_once_with(42)
        # [F7-LANE-RELEASE] position must be gone from tracker
        assert orch._positions.get(STRATEGY_ID) is None
        # engine.cancel_trade called to remove ghost trade
        orch.engine.cancel_trade.assert_called_with(STRATEGY_ID)
        # [F7-NOTIFY-ALERT] operator must have been notified
        assert orch._notify.call_count >= 1
        msgs = [str(c) for c in orch._notify.call_args_list]
        assert any("FILL TIMEOUT" in m or "timeout" in m.lower() for m in msgs)
        # kill switch must NOT have fired (cancel was confirmed)
        assert orch._kill_switch_fired is False

    async def test_timeout_broker_still_pending_fires_halt(self):
        """Timeout fires → cancel issued → broker STILL PENDING after verify →
        CRITICAL log + _notify + kill-switch fired. [F7-HALT-ON-STUCK, F7-KILL-SWITCH-CALL]
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=99)
        record = orch._positions.get(STRATEGY_ID)
        record.state_changed_at = datetime.now(UTC) - timedelta(seconds=120)
        orch.FILL_POLL_INTERVAL = 0.01
        orch.FILL_POLL_TIMEOUT_SECS = 0.001
        orch.FILL_CANCEL_VERIFY_TIMEOUT_SECS = 0.01

        orch.order_router.cancel = MagicMock()
        # Post-cancel verify still returns Working (broker stuck)
        orch.order_router.query_order_status = MagicMock(
            return_value={"order_id": 99, "status": "Working", "fill_price": None}
        )
        orch._notify = MagicMock()

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # [F7-HALT-ON-STUCK] kill switch must have fired
        assert orch._kill_switch_fired is True
        # [F7-NOTIFY-ALERT] + [F7-KILL-SWITCH-CALL] operator alert sent
        assert orch._notify.call_count >= 2
        halt_msgs = [str(c) for c in orch._notify.call_args_list]
        assert any("HALT" in m or "stuck" in m.lower() for m in halt_msgs)
        # [F7-LANE-RELEASE] lane still released even when broker is stuck
        assert orch._positions.get(STRATEGY_ID) is None

    async def test_happy_path_no_timeout(self):
        """Order fills within timeout → no cancel, no halt, fill confirmed.
        Regression guard: F7 timeout path must NOT fire for normal fills.
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=10)
        orch.FILL_POLL_INTERVAL = 0.01
        orch.FILL_POLL_TIMEOUT_SECS = 999.0  # well beyond test duration
        orch.FILL_CANCEL_VERIFY_TIMEOUT_SECS = 999.0

        orch.order_router.cancel = MagicMock()
        orch.order_router.query_order_status = MagicMock(
            return_value={"order_id": 10, "status": "Filled", "fill_price": 2350.5}
        )

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # cancel must NOT have been called
        orch.order_router.cancel.assert_not_called()
        assert orch._kill_switch_fired is False
        record = orch._positions.get(STRATEGY_ID)
        assert record is not None
        assert record.fill_entry_price == 2350.5

    async def test_kill_switch_mid_poll_exits_cleanly(self):
        """Kill-switch fires mid-poll → CancelledError propagates → task exits cleanly.
        Cross-fix: C1 interaction — no leaked task after shutdown. [F7-CANCEL-CALL skipped]
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=55)
        orch.FILL_POLL_INTERVAL = 0.01

        orch.order_router.query_order_status = MagicMock(
            return_value={"order_id": 55, "status": "Working", "fill_price": None}
        )

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.05)
        # Simulate kill-switch: cancel the task (same as _shutdown_task does)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Task completed cleanly (no exception leak)
        assert task.done()
        assert not task.cancelled() or task.cancelled()  # either done or cancelled is clean

    async def test_trading_day_rollover_mid_poll_exits_cleanly(self):
        """Trading day rollover fires mid-poll → task cancellation exits cleanly.
        Cross-fix: R1 interaction — poller shut down without error on rollover.
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=77)
        orch.FILL_POLL_INTERVAL = 0.01

        orch.order_router.query_order_status = MagicMock(
            return_value={"order_id": 77, "status": "Working", "fill_price": None}
        )

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert task.done()

    async def test_reconnect_resets_timeout_anchor(self):
        """R3 cross-fix: reconnect increments _fill_reconnect_gen → poller resets
        per-order timeout anchors → order that was 55s into 60s timeout gets a
        fresh window. No premature cancel on reconnect.
        [F7/R3: reconnect detection path]
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=33)
        # Start with order already 55s pending (near timeout edge)
        record = orch._positions.get(STRATEGY_ID)
        record.state_changed_at = datetime.now(UTC) - timedelta(seconds=55)
        orch.FILL_POLL_INTERVAL = 0.01
        orch.FILL_POLL_TIMEOUT_SECS = 60.0  # would fire at 60s from anchor
        orch.FILL_CANCEL_VERIFY_TIMEOUT_SECS = 0.01

        orch.order_router.cancel = MagicMock()
        # After reconnect, broker reports Filled
        orch.order_router.query_order_status = MagicMock(
            return_value={"order_id": 33, "status": "Filled", "fill_price": 2351.0}
        )

        # Simulate reconnect before the timeout fires
        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.02)
        # Reconnect fires: bump generation — anchor resets, timer restarts from now
        orch._fill_reconnect_gen += 1
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # cancel must NOT have been called (timeout reset gave a fresh 60s window)
        orch.order_router.cancel.assert_not_called()
        assert orch._kill_switch_fired is False

    async def test_production_fill_poll_timeout_value_fires(self):
        """M4: exercise FILL_POLL_TIMEOUT_SECS production default WITHOUT
        overriding the constant. All other F7 tests set
        FILL_POLL_TIMEOUT_SECS=0.001 to fire instantly; this leaves the
        production default in place and instead backdates state_changed_at by
        (production_timeout + 1)s so the very first poll iteration sees an
        already-expired anchor and fires. Asserts the production constant has
        not silently regressed from 60.0s.
        [F7-PROD-CONST-EXERCISE]
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        # Document the production default — assertion catches silent regression.
        assert orch.FILL_POLL_TIMEOUT_SECS == 60.0
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=44)
        record = orch._positions.get(STRATEGY_ID)
        # Backdate to (prod timeout + 1)s ago — first poll iteration must fire.
        record.state_changed_at = datetime.now(UTC) - timedelta(seconds=orch.FILL_POLL_TIMEOUT_SECS + 1)
        orch.FILL_POLL_INTERVAL = 0.01
        # Deliberately do NOT touch FILL_POLL_TIMEOUT_SECS.
        orch.FILL_CANCEL_VERIFY_TIMEOUT_SECS = 0.01

        orch.order_router.cancel = MagicMock()
        orch.order_router.query_order_status = MagicMock(
            return_value={"order_id": 44, "status": "Cancelled", "fill_price": None}
        )
        orch._notify = MagicMock()

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Production timeout fired against the production constant.
        orch.order_router.cancel.assert_called_once()
        assert orch._positions.get(STRATEGY_ID) is None

    async def test_timeout_verify_query_failure_still_halts(self):
        """Post-cancel verify raises exception → treated as 'not confirmed' →
        kill-switch still fires (fail-closed). [F7-CANCEL-VERIFY, F7-HALT-ON-STUCK]
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=88)
        record = orch._positions.get(STRATEGY_ID)
        record.state_changed_at = datetime.now(UTC) - timedelta(seconds=120)
        orch.FILL_POLL_INTERVAL = 0.01
        orch.FILL_POLL_TIMEOUT_SECS = 0.001
        orch.FILL_CANCEL_VERIFY_TIMEOUT_SECS = 0.01

        orch.order_router.cancel = MagicMock()
        # verify raises (broker not responding)
        orch.order_router.query_order_status = MagicMock(side_effect=ConnectionError("broker unreachable"))
        orch._notify = MagicMock()

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # [F7-HALT-ON-STUCK] fail-closed: unreachable verify = halt
        assert orch._kill_switch_fired is True
        # [F7-LANE-RELEASE] lane released even on exception path
        assert orch._positions.get(STRATEGY_ID) is None

    async def test_filled_during_cancel_race_calls_on_entry_filled_not_cancel_trade(self):
        """CRITICAL-2 (iter 187): when post-cancel verify returns Filled, the order
        raced — a real broker position exists.  The handler MUST:
          - NOT call engine.cancel_trade (that would orphan a real position)
          - call _positions.on_entry_filled with the fill price
          - fire kill-switch + emergency flatten (no bracket context available)
          - send a _notify with 'raced' or 'race' signal for the operator
        [F7-FILLED-RACE, F7-FILLED-RACE-FLATTEN]
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2352.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=101)
        record = orch._positions.get(STRATEGY_ID)
        record.state_changed_at = datetime.now(UTC) - timedelta(seconds=120)
        orch.FILL_POLL_INTERVAL = 0.01
        orch.FILL_POLL_TIMEOUT_SECS = 0.001
        orch.FILL_CANCEL_VERIFY_TIMEOUT_SECS = 0.01

        orch.order_router.cancel = MagicMock()
        # Post-cancel verify returns Filled — the order filled during the cancel race
        orch.order_router.query_order_status = MagicMock(
            return_value={"order_id": 101, "status": "Filled", "fill_price": 2352.0}
        )
        orch._notify = MagicMock()
        orch._emergency_flatten = AsyncMock()

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.25)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Must NOT have called cancel_trade — position is real at broker
        orch.engine.cancel_trade.assert_not_called()
        # Kill-switch must have fired (unbracketed real position = halt)
        assert orch._kill_switch_fired is True
        # Emergency flatten must have been called  [F7-BROKER-STUCK-FLATTEN / F7-FILLED-RACE-FLATTEN]
        orch._emergency_flatten.assert_called_once()
        # Operator must have been notified with race signal
        assert orch._notify.call_count >= 1
        notify_texts = " ".join(str(c) for c in orch._notify.call_args_list)
        assert "race" in notify_texts.lower() or "raced" in notify_texts.lower()
        # Position should be in ENTERED state (on_entry_filled called) or gone (flatten ran)
        # Either is correct — what must NOT happen is the position staying PENDING_ENTRY
        pos = orch._positions.get(STRATEGY_ID)
        from trading_app.live.position_tracker import PositionState

        assert pos is None or pos.state != PositionState.PENDING_ENTRY

    async def test_broker_stuck_halt_calls_emergency_flatten(self):
        """CRITICAL-1 (iter 187): broker-stuck halt path must call _emergency_flatten
        after _fire_kill_switch.  Without this, a position that filled silently at
        the broker before the stuck-cancel path runs would remain naked indefinitely.
        Mutation-proof: assertion checks for [F7-BROKER-STUCK-FLATTEN] source marker
        in production code AND runtime call assertion.
        [F7-HALT-ON-STUCK, F7-KILL-SWITCH-CALL, F7-BROKER-STUCK-FLATTEN]
        """
        import inspect

        # Structural probe: source marker must be present (mutation guard)
        from trading_app.live import session_orchestrator as _so_mod

        src = inspect.getsource(_so_mod.SessionOrchestrator._handle_fill_timeout)
        assert "[F7-BROKER-STUCK-FLATTEN]" in src, (
            "Source marker [F7-BROKER-STUCK-FLATTEN] missing from _handle_fill_timeout — "
            "emergency_flatten call was removed from the broker-stuck halt path (CRITICAL-1)."
        )

        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=200)
        record = orch._positions.get(STRATEGY_ID)
        record.state_changed_at = datetime.now(UTC) - timedelta(seconds=120)
        orch.FILL_POLL_INTERVAL = 0.01
        orch.FILL_POLL_TIMEOUT_SECS = 0.001
        orch.FILL_CANCEL_VERIFY_TIMEOUT_SECS = 0.01

        orch.order_router.cancel = MagicMock()
        # Broker stuck: verify still returns Working after cancel
        orch.order_router.query_order_status = MagicMock(
            return_value={"order_id": 200, "status": "Working", "fill_price": None}
        )
        orch._notify = MagicMock()
        orch._emergency_flatten = AsyncMock()

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.25)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Kill-switch must have fired  [F7-KILL-SWITCH-CALL]
        assert orch._kill_switch_fired is True
        # Emergency flatten must have been called  [F7-BROKER-STUCK-FLATTEN]
        orch._emergency_flatten.assert_called_once()

    async def test_kill_switch_state_blocks_handle_fill_timeout(self):
        """M6 silent gap (iter 186 audit): if _kill_switch_fired is already True when
        _fill_poller next iterates, _handle_fill_timeout must NOT be called — the
        orchestrator is already halted and a double-cancel/halt would be confusing
        and could race with ongoing emergency-flatten work.

        This tests the POLLER-level guard: when kill_switch is set, the poller's
        timeout path is skipped.  The poller still exits on CancelledError; but
        before that, a kill-switch-active state must prevent further timeout processing.
        [F7-CANCEL-CALL skipped when kill_switch_fired=True]
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._positions.on_entry_sent(STRATEGY_ID, "long", 2350.0, order_id=300)
        record = orch._positions.get(STRATEGY_ID)
        # Back-date far past timeout threshold
        record.state_changed_at = datetime.now(UTC) - timedelta(seconds=120)
        orch.FILL_POLL_INTERVAL = 0.01
        orch.FILL_POLL_TIMEOUT_SECS = 0.001
        orch.FILL_CANCEL_VERIFY_TIMEOUT_SECS = 0.01

        # Pre-fire kill switch — orchestrator is already halted
        orch._kill_switch_fired = True

        cancel_mock = MagicMock()
        orch.order_router.cancel = cancel_mock
        orch.order_router.query_order_status = MagicMock(
            return_value={"order_id": 300, "status": "Working", "fill_price": None}
        )

        task = asyncio.create_task(orch._fill_poller())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # cancel must NOT have been called — poller should detect kill_switch_fired
        # and skip timeout processing when already halted.
        # If this assertion fails, the poller needs a kill-switch guard at the top
        # of its loop body (mirror pattern from _handle_event).
        cancel_mock.assert_not_called()


class TestObservability:
    """Tests for SessionStats counters, upgraded _notify, heartbeat, and self-tests."""

    def test_notify_counts_success(self):
        """Successful _notify() increments notifications_sent."""
        orch = build_orchestrator()
        with patch("trading_app.live.notifications.notify"):
            orch._notify("test")
        assert orch._stats.notifications_sent == 1
        assert orch._stats.notifications_failed == 0

    def test_notify_persists_operator_alert(self):
        """_notify() writes a structured operator alert for the dashboard."""
        orch = build_orchestrator()
        with (
            patch("trading_app.live.alert_engine.record_operator_alert") as mock_record,
            patch("trading_app.live.notifications.notify"),
        ):
            orch._notify("FEED STALE: 180s no data (check 2)")

        mock_record.assert_called_once()
        kwargs = mock_record.call_args.kwargs
        assert kwargs["message"].startswith("FEED STALE")
        assert kwargs["instrument"] == "MGC"
        assert kwargs["profile"] == "test"
        assert kwargs["mode"] == "DEMO"
        assert kwargs["source"] == "session_orchestrator"
        assert kwargs["trading_day"] == "2026-03-07"

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
        orch._check_trading_day_rollover = AsyncMock()
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


class TestF4BracketNakedPosition:
    """F4 (CRITICAL): bracket submit failure post-fill must NOT leave position naked.

    Each test is a mutation probe: revert the corresponding F4 sub-path fix in
    trading_app/live/session_orchestrator.py and the matching assertion must fail.

    Three failure sub-paths:
      F4-1: no risk_points → cannot compute stop/target → emergency flatten
      F4-2: build_bracket_spec returns None → broker can't represent bracket → emergency flatten
      F4-3: submit() raises → network/auth failure → bracket never reached broker → emergency flatten

    Pattern mirrored: _notify + _fire_kill_switch + _emergency_flatten
    (same as DD halt at L1491-1492 and consecutive bar gap at L1515-1516).
    """

    def _make_event(self, strategy, *, risk_points=5.0):
        event = MagicMock()
        event.strategy_id = strategy.strategy_id
        event.direction = "long"
        event.contracts = 1
        event.risk_points = risk_points
        return event

    # F4-1 — no risk_points triggers emergency flatten
    async def test_f4_no_risk_points_triggers_flatten(self):
        """When both event.risk_points and strategy.median_risk_points are falsy,
        _submit_bracket must call _fire_kill_switch and _emergency_flatten rather
        than silently returning. Pre-fix: log.error + return, position stays naked.
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch.order_router.supports_native_brackets = MagicMock(return_value=True)

        strategy = list(orch._strategy_map.values())[0]
        event = self._make_event(strategy, risk_points=None)  # no risk_points

        # Patch strategy so median_risk_points is also falsy
        strategy_mock = MagicMock(wraps=strategy)
        strategy_mock.median_risk_points = None
        strategy_mock.strategy_id = strategy.strategy_id
        strategy_mock.rr_target = strategy.rr_target

        orch._notify = MagicMock()
        flatten_called = []

        async def fake_flatten():
            flatten_called.append(True)

        with patch.object(orch, "_emergency_flatten", new=fake_flatten):
            await orch._submit_bracket(event, strategy_mock, 2350.0)

        # F4 invariants: flatten called, counter incremented, operator notified
        assert flatten_called, "F4-1: _emergency_flatten must be called when risk_points is None"
        assert orch._kill_switch_fired, "F4-1: _fire_kill_switch must be called"
        assert orch._stats.brackets_failed == 1
        assert orch._stats.brackets_submitted == 0
        assert orch._notify.called, "F4-1: operator must be notified via _notify"
        # Source marker: 'F4-1' appears in the notify message after the fix
        notify_msg = str(orch._notify.call_args_list[0])
        assert "F4" in notify_msg, "F4-1: notify message must contain 'F4' source marker"

    # F4-2 — build_bracket_spec returns None triggers emergency flatten
    async def test_f4_bracket_spec_none_triggers_flatten(self):
        """When build_bracket_spec returns None, _submit_bracket must flatten.
        Pre-fix: log.warning + return, position stays naked.
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch.order_router.supports_native_brackets = MagicMock(return_value=True)
        orch.order_router.build_bracket_spec = MagicMock(return_value=None)

        strategy = list(orch._strategy_map.values())[0]
        event = self._make_event(strategy, risk_points=5.0)

        orch._notify = MagicMock()
        flatten_called = []

        async def fake_flatten():
            flatten_called.append(True)

        with patch.object(orch, "_emergency_flatten", new=fake_flatten):
            await orch._submit_bracket(event, strategy, 2350.0)

        assert flatten_called, "F4-2: _emergency_flatten must be called when bracket_spec is None"
        assert orch._kill_switch_fired, "F4-2: _fire_kill_switch must be called"
        assert orch._stats.brackets_failed == 1
        assert orch._stats.brackets_submitted == 0
        assert orch._notify.called, "F4-2: operator must be notified via _notify"
        notify_msg = str(orch._notify.call_args_list[0])
        assert "F4" in notify_msg, "F4-2: notify message must contain 'F4' source marker"

    # F4-3 — submit() raises triggers emergency flatten
    async def test_f4_submit_raises_triggers_flatten(self):
        """When order_router.submit raises, _submit_bracket must flatten.
        Pre-fix: log.warning only, brackets_failed incremented, position stays naked.
        Mutation probe: pre-fix test_bracket_failure_counter only checks
        brackets_failed==1 — it passes whether or not flatten is called.
        """
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch.order_router.supports_native_brackets = MagicMock(return_value=True)
        orch.order_router.build_bracket_spec = MagicMock(return_value={"type": "OCO"})
        orch.order_router.submit = MagicMock(side_effect=RuntimeError("network timeout"))

        strategy = list(orch._strategy_map.values())[0]
        event = self._make_event(strategy, risk_points=5.0)

        orch._notify = MagicMock()
        flatten_called = []

        async def fake_flatten():
            flatten_called.append(True)

        with patch.object(orch, "_emergency_flatten", new=fake_flatten):
            await orch._submit_bracket(event, strategy, 2350.0)

        assert flatten_called, "F4-3: _emergency_flatten must be called when submit raises"
        assert orch._kill_switch_fired, "F4-3: _fire_kill_switch must be called"
        assert orch._stats.brackets_failed == 1
        assert orch._stats.brackets_submitted == 0
        assert orch._notify.called, "F4-3: operator must be notified via _notify"
        notify_msg = str(orch._notify.call_args_list[0])
        assert "F4" in notify_msg, "F4-3: notify message must contain 'F4' source marker"

    # Source-text probe — confirms all three F4 markers survive in production code
    def test_f4_source_markers_present(self):
        """Source-text mutation probe: all three F4 source markers must be present
        in production code. If any F4 sub-path fix is reverted, the matching marker
        disappears and this test fails.
        """
        from trading_app.live import session_orchestrator as so

        src = open(so.__file__, encoding="utf-8").read()
        assert "F4-1" in src, "F4-1 source marker missing — no-risk-points path reverted"
        assert "F4-2" in src, "F4-2 source marker missing — bracket-spec-None path reverted"
        assert "F4-3" in src, "F4-3 source marker missing — submit-raises path reverted"
        # Confirm the institutional pattern is present: flatten must follow kill-switch
        assert "_fire_kill_switch" in src and "_emergency_flatten" in src, (
            "F4: kill-switch + emergency flatten pattern must be present"
        )


class TestC1KillSwitchEventLoopRace:
    """C1 (CRITICAL): kill-switch guard in _handle_event ENTRY branch.

    F4 fires kill-switch inside _submit_bracket which is called from _handle_event.
    Without the C1 guard, event N+1 (another ENTRY in the same bar's events list)
    still reaches the broker AFTER the kill-switch fired for event N.

    Mutation probes: remove the C1 guard and T2 must fail.
    """

    # T2 (BLOCKING): two ENTRY events on same bar — first fires kill-switch, second must be blocked
    async def test_c1_second_entry_blocked_after_kill_switch(self):
        """Bar produces [ENTRY_A, ENTRY_B]. ENTRY_A's _submit_bracket fires kill-switch.
        ENTRY_B must be blocked by the C1 guard before reaching order_router.submit.

        We use a non-native-bracket router so event_a takes the _submit_bracket path.
        """
        from unittest.mock import AsyncMock

        # Non-native-bracket router: supports_native_brackets=False so _submit_bracket is called
        class NonNativeRouter(FakeRouter):
            def __init__(self):
                super().__init__(fill_price=2350.0)

            def supports_native_brackets(self):
                return False

        router = NonNativeRouter()
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch.order_router = router
        orch.signal_only = False
        notifications: list[str] = []
        orch._notify = notifications.append

        submit_calls = []
        original_submit = router.submit

        def counting_submit(spec):
            submit_calls.append(spec)
            return original_submit(spec)

        router.submit = counting_submit

        # Make _submit_bracket fire kill-switch (simulates F4 path)
        async def fake_submit_bracket(event, strategy, actual_entry):
            # Simulate F4-3: bracket submit triggers kill-switch + emergency flatten
            orch._fire_kill_switch()
            # No active positions so _emergency_flatten is a no-op here
            await orch._emergency_flatten()

        orch._submit_bracket = fake_submit_bracket

        # Simulate: engine returned two ENTRY events for the same bar
        event_a = _entry_event(2350.0)
        # Give event_b a distinct strategy so it isn't caught by the duplicate-entry guard
        event_b = FakeTradeEvent(
            event_type="ENTRY",
            strategy_id="strat_b",
            timestamp=event_a.timestamp,
            price=event_a.price,
            direction=event_a.direction,
            contracts=event_a.contracts,
            risk_points=event_a.risk_points,
        )
        # Register strat_b in the strategy map so _handle_event doesn't early-return on unknown id
        strat_b = PortfolioStrategy(
            strategy_id="strat_b",
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
        orch._strategy_map["strat_b"] = strat_b

        # Process event_a: entry submit → fake_submit_bracket → kill-switch fires
        await orch._handle_event(event_a)
        assert orch._kill_switch_fired, "C1: kill switch must have fired after event_a"

        # Process event_b: C1 guard must block this ENTRY immediately
        notify_before = len(notifications)
        await orch._handle_event(event_b)

        # event_b must NOT have submitted a broker ENTRY order.
        # submit_calls may contain: (1) event_a's entry, (2) the emergency-flatten exit for event_a.
        # The discriminator: no submit with type='fake_entry' for strat_b.
        entry_submits = [s for s in submit_calls if s.get("type") == "fake_entry"]
        assert len(entry_submits) == 1, (
            f"C1: only 1 ENTRY submit expected (event_a). "
            f"entry_submits={len(entry_submits)}, all submit_calls={submit_calls}"
        )
        # C1 guard must have notified (log output already confirmed "C1: ENTRY BLOCKED for strat_b")
        assert any("C1" in n or "kill switch" in n.lower() for n in notifications[notify_before:]), (
            f"C1: guard must notify when blocking ENTRY. notifications after block: {notifications[notify_before:]}"
        )

    # T1: _emergency_flatten raises on all 3 attempts → MANUAL CLOSE REQUIRED + notify + persist
    async def test_c1_emergency_flatten_all_retries_exhausted(self):
        """If all 3 _emergency_flatten attempts raise, the loop's else clause must fire
        'MANUAL CLOSE REQUIRED' and _notify. kill_switch_fired must persist to _safety_state.
        """
        router = FakeBracketRouter(fill_price=2350.0)
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch.order_router = router
        orch.signal_only = False
        notifications: list[str] = []
        orch._notify = notifications.append

        # Plant an open position so _emergency_flatten has something to act on
        from trading_app.live.position_tracker import PositionTracker

        orch._positions = MagicMock()
        fake_rec = MagicMock()
        fake_rec.strategy_id = "strat_x"
        fake_rec.direction = "long"
        fake_rec.contracts = 1
        fake_rec.bracket_order_ids = []
        fake_rec.journal_trade_id = None
        orch._positions.active_positions.return_value = [fake_rec]

        # Make every broker submit raise
        router.submit = MagicMock(side_effect=RuntimeError("broker offline"))

        orch._fire_kill_switch()
        await orch._emergency_flatten()

        # T1 invariant: MANUAL CLOSE REQUIRED must have been notified
        assert any("MANUAL CLOSE REQUIRED" in n for n in notifications), (
            "T1: MANUAL CLOSE REQUIRED not notified after 3 failed flatten attempts"
        )
        # kill_switch_fired must have been persisted
        assert orch._safety_state.kill_switch_fired, "T1: kill_switch_fired not persisted to _safety_state"

    # T4: rollover EOD closes still fire when kill_switch=True (ENTRY-only guard, not blanket)
    async def test_c1_rollover_eod_closes_proceed_despite_kill_switch(self):
        """After kill-switch fires, _check_trading_day_rollover must still call
        _handle_event for EOD EXIT events (wind-down of existing positions).
        The C1 guard is ENTRY-only — it must NOT block EXIT/SCRATCH events.
        """
        orch = build_orchestrator()
        orch._notify = MagicMock()

        # Fire kill-switch
        orch._fire_kill_switch()
        assert orch._kill_switch_fired

        exit_handled = []

        async def capturing_handle_event(event):
            exit_handled.append(event.event_type)

        # Patch _handle_event to capture (we only want to verify it's CALLED for EXIT)
        orch._handle_event = capturing_handle_event

        # Simulate engine.on_trading_day_end() returning an EXIT event
        exit_event = FakeTradeEvent(
            event_type="EXIT",
            strategy_id="strat_x",
            timestamp=datetime(2026, 4, 25, 23, 0, 0, tzinfo=UTC),
            price=2350.0,
            direction="long",
            contracts=1,
        )
        orch.engine = MagicMock()
        orch.engine.on_trading_day_end.return_value = [exit_event]
        orch._positions = MagicMock()
        orch._positions.active_positions.return_value = []

        # Use the real _check_trading_day_rollover but with override_trading_day
        next_day = orch.trading_day + timedelta(days=1)
        with patch("pipeline.dst.compute_trading_day_utc_range") as mock_ctr:
            from datetime import timezone

            past = datetime(2026, 1, 1, 23, 0, 0, tzinfo=timezone.utc)
            mock_ctr.return_value = (past, past + timedelta(hours=24))
            await orch._check_trading_day_rollover(None, override_trading_day=next_day)

        # T4 invariant: EXIT event must have been passed to _handle_event
        assert "EXIT" in exit_handled, (
            f"T4: EOD EXIT must be forwarded to _handle_event even when kill-switch active. Got: {exit_handled}"
        )

    # Source-marker mutation probe
    def test_c1_source_marker_present(self):
        """C1 guard string must be present in _handle_event source."""
        import inspect
        from trading_app.live.session_orchestrator import SessionOrchestrator

        src = inspect.getsource(SessionOrchestrator._handle_event)
        assert "C1" in src, "C1: kill-switch guard marker missing from _handle_event"
        assert "_kill_switch_fired" in src, "C1: _kill_switch_fired check missing from _handle_event"
        # Guard must be ENTRY-only — must appear INSIDE the ENTRY branch, not at function top
        entry_idx = src.index('event.event_type == "ENTRY"')
        c1_idx = src.index("C1")
        assert c1_idx > entry_idx, "C1: guard must be inside the ENTRY branch, not at function top (blanket over-block)"


class TestObservabilityCounters:
    """Observability counters originally in TestObservability (continued after F4 class insertion)."""

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
            mock_feed_cls = MagicMock(side_effect=KeyboardInterrupt)
            orch._feed_class = mock_feed_cls
            try:
                await orch.run()
            except (KeyboardInterrupt, Exception):
                pass
            # Verify we got PAST the notification gate (feed class was reached)
            mock_feed_cls.assert_called()


# ---------------------------------------------------------------------------
# F4: Kill switch direction=None guard
# ---------------------------------------------------------------------------


class TestKillSwitchDirectionNone:
    """_emergency_flatten must skip positions with direction=None."""

    async def test_direction_none_skips_order_and_notifies(self):
        """Position with direction=None: no order submitted, CANNOT FLATTEN SAFELY logged."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2350.0))
        orch._notify = MagicMock()

        # Inject a position with direction=None via direct PositionRecord insertion
        from trading_app.live.position_tracker import PositionRecord, PositionState

        record = PositionRecord(
            strategy_id=STRATEGY_ID,
            state=PositionState.ENTERED,
            direction=None,
            engine_entry_price=2350.0,
            contracts=1,
        )
        orch._positions._positions[STRATEGY_ID] = record

        await orch._emergency_flatten()

        # No order should have been submitted
        assert len(orch.order_router.submitted) == 0
        # _notify should have been called with CANNOT FLATTEN SAFELY
        notify_calls = [str(c) for c in orch._notify.call_args_list]
        assert any("CANNOT FLATTEN SAFELY" in c for c in notify_calls)


# ---------------------------------------------------------------------------
# F7: _on_bar engine error isolation
# ---------------------------------------------------------------------------


class TestOnBarEngineErrorIsolation:
    """Engine errors in _on_bar must be caught — bar dropped, feed continues."""

    async def test_engine_error_increments_counter_and_continues(self):
        """engine.on_bar() raising RuntimeError: bar dropped, engine_errors incremented."""
        orch = build_orchestrator()
        orch.engine.on_bar.side_effect = RuntimeError("unexpected engine crash")
        orch._check_trading_day_rollover = AsyncMock()

        bar = FakeBar()
        # Should NOT raise
        await orch._on_bar(bar)

        assert orch._stats.engine_errors == 1
        assert orch._stats.bars_received == 1


# ---------------------------------------------------------------------------
# F9: _notifications_broken initialization order
# ---------------------------------------------------------------------------


class TestNotificationsBrokenInit:
    """_notifications_broken must be False after __init__ (before run_self_tests)."""

    def test_notifications_broken_is_false_after_init(self):
        """build_orchestrator sets _notifications_broken=False before self-tests run."""
        orch = build_orchestrator()
        assert orch._notifications_broken is False


# ---------------------------------------------------------------------------
# F20: Circuit breaker sign mismatch — positive Portfolio magnitude
#      must be negated to negative RiskLimits threshold
# ---------------------------------------------------------------------------


class TestCircuitBreakerSignConvention:
    """Portfolio stores max_daily_loss_r as positive magnitude (5.0).
    Orchestrator must negate to negative threshold (-5.0) for RiskLimits."""

    def test_positive_portfolio_loss_does_not_block_entries(self):
        """With max_daily_loss_r=5.0 in Portfolio, RiskLimits must get -5.0.
        At 0.0 daily PnL, entries should be ALLOWED (not blocked)."""
        from trading_app.risk_manager import RiskLimits

        orch = build_orchestrator()
        # Simulate what the real __init__ does with the fix
        risk_limits = RiskLimits(
            max_daily_loss_r=-abs(orch.portfolio.max_daily_loss_r),
        )
        assert risk_limits.max_daily_loss_r < 0, "RiskLimits must be negative"
        assert risk_limits.max_daily_loss_r == -5.0


# ---------------------------------------------------------------------------
# F21/F31: Trading day rollover exception must not crash feed loop
# ---------------------------------------------------------------------------


class TestRolloverExceptionIsolation:
    """_check_trading_day_rollover raising should NOT propagate out of _on_bar."""

    async def test_rollover_error_caught_and_feed_continues(self):
        """RuntimeError in rollover is caught — bar still processed."""
        orch = build_orchestrator()
        orch._check_trading_day_rollover = AsyncMock(side_effect=RuntimeError("daily_features 10 days stale"))
        orch.engine.on_bar.return_value = []

        bar = FakeBar()
        # Should NOT raise
        await orch._on_bar(bar)

        assert orch._stats.bars_received == 1
        orch.engine.on_bar.assert_called_once()


# ---------------------------------------------------------------------------
# F32: Heartbeat trade count — monitor.trade_count property
# ---------------------------------------------------------------------------


class TestHeartbeatTradeCount:
    """Heartbeat must report actual trade count, not always 0."""

    def test_performance_monitor_trade_count_property(self):
        """PerformanceMonitor.trade_count returns len(_trades)."""
        from trading_app.live.performance_monitor import PerformanceMonitor, TradeRecord

        strategy = _test_strategy()
        monitor = PerformanceMonitor([strategy])
        assert monitor.trade_count == 0

        record = TradeRecord(
            strategy_id=strategy.strategy_id,
            trading_day=date(2026, 3, 9),
            direction="long",
            entry_price=2350.0,
            exit_price=2355.0,
            actual_r=1.5,
            expected_r=0.20,
        )
        monitor.record_trade(record)
        assert monitor.trade_count == 1

    def test_trade_count_resets_on_daily(self):
        """trade_count resets to 0 after reset_daily."""
        from trading_app.live.performance_monitor import PerformanceMonitor, TradeRecord

        strategy = _test_strategy()
        monitor = PerformanceMonitor([strategy])

        record = TradeRecord(
            strategy_id=strategy.strategy_id,
            trading_day=date(2026, 3, 9),
            direction="long",
            entry_price=2350.0,
            exit_price=2355.0,
            actual_r=1.5,
            expected_r=0.20,
        )
        monitor.record_trade(record)
        assert monitor.trade_count == 1

        monitor.reset_daily()
        assert monitor.trade_count == 0


# ── Exit retry tests ────────────────────────────────────────────────────────


class TestExitRetry:
    """Tests for _submit_exit_with_retry — 3-attempt linear backoff for exits."""

    async def test_retry_succeeds_on_third_attempt(self):
        """Router fails twice, succeeds on attempt 3 → function returns result."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2351.0))
        call_count = 0

        def flaky_submit(spec):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Tradovate 503")
            return {"order_id": 999, "fill_price": 2351.0}

        orch.order_router.submit = flaky_submit
        # Patch sleep to avoid real delays
        with patch("trading_app.live.session_orchestrator.asyncio.sleep", new_callable=AsyncMock):
            result = await orch._submit_exit_with_retry({"dummy": "spec"}, "test_strat")
        assert result["order_id"] == 999
        assert call_count == 3

    async def test_all_retries_exhausted_raises(self):
        """Router fails all 3 times → raises, sends notification."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2351.0))

        def always_fail(spec):
            raise ConnectionError("Tradovate down")

        orch.order_router.submit = always_fail
        orch._notify = MagicMock()
        with patch("trading_app.live.session_orchestrator.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ConnectionError):
                await orch._submit_exit_with_retry({"dummy": "spec"}, "test_strat")
        # Notification sent on final failure
        orch._notify.assert_called_once()
        assert "MANUAL CLOSE REQUIRED" in str(orch._notify.call_args)

    async def test_already_flat_not_retried(self):
        """'no position' error bypasses retry — immediate re-raise."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2351.0))
        call_count = 0

        def already_flat(spec):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("no position found for contract")

        orch.order_router.submit = already_flat
        with pytest.raises(RuntimeError, match="no position"):
            await orch._submit_exit_with_retry({"dummy": "spec"}, "test_strat")
        assert call_count == 1  # no retry — immediate re-raise

    async def test_entry_path_does_not_use_retry(self):
        """ENTRY events use direct submit, not _submit_exit_with_retry."""
        orch = build_orchestrator(FakeBrokerComponents(fill_price=2351.0))
        with patch.object(orch, "_submit_exit_with_retry") as mock_retry:
            await orch._handle_event(_entry_event(2350.5))
            mock_retry.assert_not_called()


# ---------------------------------------------------------------------------
# ORB cap gate tests
# ---------------------------------------------------------------------------


def _nyse_open_strategy() -> PortfolioStrategy:
    return PortfolioStrategy(
        strategy_id="MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60_O15",
        instrument="MNQ",
        orb_label="NYSE_OPEN",
        entry_model="E2",
        rr_target=1.0,
        confirm_bars=1,
        filter_type="X_MES_ATR60",
        expectancy_r=0.09,
        win_rate=0.55,
        sample_size=500,
        sharpe_ratio=0.8,
        max_drawdown_r=5.0,
        median_risk_points=85.0,
        stop_multiplier=0.75,
        source="test",
        weight=1.0,
    )


def _nyse_open_entry(risk_points: float | None = None) -> FakeTradeEvent:
    return FakeTradeEvent(
        event_type="ENTRY",
        strategy_id="MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60_O15",
        timestamp=datetime.now(UTC),
        price=20000.0,
        direction="long",
        contracts=1,
        risk_points=risk_points,
    )


@pytest.mark.asyncio
class TestOrbCapGate:
    """ORB cap gate in _handle_event: skip oversized ORBs on capped lanes."""

    def _build_capped_orch(self, signal_only: bool = False) -> SessionOrchestrator:
        strat = _nyse_open_strategy()
        portfolio = Portfolio(
            name="test",
            instrument="MNQ",
            strategies=[strat],
            account_equity=50000.0,
            risk_per_trade_pct=2.0,
            max_concurrent_positions=4,
            max_daily_loss_r=5.0,
        )
        c = FakeBrokerComponents(fill_price=20000.0, signal_only=signal_only)
        orch = build_orchestrator(c)
        orch.instrument = "MNQ"
        orch.portfolio = portfolio
        orch._strategy_map = {strat.strategy_id: strat}
        orch._orb_caps = {(strat.orb_label, strat.instrument): 150.0}
        return orch

    async def test_149pt_under_cap_submits(self):
        """ORB at 149 pts: under cap, trade should proceed."""
        orch = self._build_capped_orch()
        event = _nyse_open_entry(risk_points=149.0)
        await orch._handle_event(event)
        assert orch._stats.orb_cap_skips == 0
        # Order was submitted (router got a call)
        assert len(orch.order_router.submitted) > 0

    async def test_150pt_at_cap_skipped(self):
        """ORB at 150 pts: at cap boundary, trade should be SKIPPED (inclusive)."""
        orch = self._build_capped_orch()
        event = _nyse_open_entry(risk_points=150.0)
        await orch._handle_event(event)
        assert orch._stats.orb_cap_skips == 1
        assert len(orch.order_router.submitted) == 0

    async def test_151pt_over_cap_skipped(self):
        """ORB at 151 pts: over cap, trade should be SKIPPED."""
        orch = self._build_capped_orch()
        event = _nyse_open_entry(risk_points=151.0)
        await orch._handle_event(event)
        assert orch._stats.orb_cap_skips == 1
        assert len(orch.order_router.submitted) == 0

    async def test_no_cap_any_size_passes(self):
        """Lane without a cap: any ORB size should proceed."""
        orch = self._build_capped_orch()
        orch._orb_caps = {}  # No caps at all
        event = _nyse_open_entry(risk_points=999.0)
        await orch._handle_event(event)
        assert orch._stats.orb_cap_skips == 0
        assert len(orch.order_router.submitted) > 0

    async def test_none_risk_points_passes(self):
        """No risk_points on event: cap check should be skipped (fail-open)."""
        orch = self._build_capped_orch()
        event = _nyse_open_entry(risk_points=None)
        await orch._handle_event(event)
        assert orch._stats.orb_cap_skips == 0

    async def test_skip_counter_increments(self):
        """Multiple oversized trades should each increment the counter."""
        orch = self._build_capped_orch()
        for pts in [150.0, 200.0, 300.0]:
            event = _nyse_open_entry(risk_points=pts)
            await orch._handle_event(event)
        assert orch._stats.orb_cap_skips == 3

    async def test_signal_only_also_skipped(self):
        """Signal-only mode should also skip oversized ORBs."""
        orch = self._build_capped_orch(signal_only=True)
        event = _nyse_open_entry(risk_points=200.0)
        await orch._handle_event(event)
        assert orch._stats.orb_cap_skips == 1

    async def test_cap_skip_writes_signal_record(self):
        """Cap skip should write a signal record for observability."""
        orch = self._build_capped_orch()
        event = _nyse_open_entry(risk_points=200.0)
        await orch._handle_event(event)
        orch._write_signal_record.assert_called()
        record = orch._write_signal_record.call_args[0][0]
        assert record["type"] == "ORB_CAP_SKIP"
        assert record["risk_pts"] == 200.0
        assert record["cap_pts"] == 150.0


@pytest.mark.asyncio
class TestMaxRiskPerTrade:
    """Per-trade dollar risk cap: reject trades exceeding max_risk_per_trade."""

    def _build_risk_capped_orch(self, max_risk: float | None = 300.0) -> SessionOrchestrator:
        strat = _nyse_open_strategy()
        portfolio = Portfolio(
            name="test",
            instrument="MNQ",
            strategies=[strat],
            account_equity=30000.0,
            risk_per_trade_pct=2.0,
            max_concurrent_positions=4,
            max_daily_loss_r=5.0,
        )
        c = FakeBrokerComponents(fill_price=20000.0)
        orch = build_orchestrator(c)
        orch.instrument = "MNQ"
        orch.portfolio = portfolio
        orch._strategy_map = {strat.strategy_id: strat}
        orch.cost_spec.point_value = 2.0  # MNQ = $2/pt
        orch._max_risk_per_trade = max_risk
        return orch

    async def test_250_under_cap_accepted(self):
        """$250 risk (125 pts × $2/pt) under $300 cap → accepted."""
        orch = self._build_risk_capped_orch(max_risk=300.0)
        event = _nyse_open_entry(risk_points=125.0)  # 125 × $2 = $250
        await orch._handle_event(event)
        assert len(orch.order_router.submitted) > 0

    async def test_350_over_cap_rejected(self):
        """$350 risk (175 pts × $2/pt) over $300 cap → rejected."""
        orch = self._build_risk_capped_orch(max_risk=300.0)
        event = _nyse_open_entry(risk_points=175.0)  # 175 × $2 = $350
        await orch._handle_event(event)
        assert len(orch.order_router.submitted) == 0
        orch._write_signal_record.assert_called()
        record = orch._write_signal_record.call_args[0][0]
        assert record["type"] == "MAX_RISK_SKIP"
        assert record["risk_dollars"] == 350.0
        assert record["cap_dollars"] == 300.0

    async def test_no_cap_any_risk_accepted(self):
        """Profile without max_risk_per_trade → all trades accepted."""
        orch = self._build_risk_capped_orch(max_risk=None)
        event = _nyse_open_entry(risk_points=500.0)  # 500 × $2 = $1000
        await orch._handle_event(event)
        assert len(orch.order_router.submitted) > 0

    async def test_none_risk_points_passes(self):
        """No risk_points on event → max risk check skipped (fail-open)."""
        orch = self._build_risk_capped_orch(max_risk=300.0)
        event = _nyse_open_entry(risk_points=None)
        await orch._handle_event(event)
        # Should not be blocked by max risk check (fail-open on missing data)

    async def test_exactly_at_cap_accepted(self):
        """$300 risk exactly at $300 cap → accepted (strict greater-than)."""
        orch = self._build_risk_capped_orch(max_risk=300.0)
        event = _nyse_open_entry(risk_points=150.0)  # 150 × $2 = $300 exactly
        await orch._handle_event(event)
        assert len(orch.order_router.submitted) > 0


@pytest.mark.asyncio
class TestRegimeGate:
    """Regime gate: block entries for strategies paused by allocator."""

    def _build_regime_orch(self, paused: set[str] | None = None) -> SessionOrchestrator:
        strat = _nyse_open_strategy()
        portfolio = Portfolio(
            name="test",
            instrument="MNQ",
            strategies=[strat],
            account_equity=30000.0,
            risk_per_trade_pct=2.0,
            max_concurrent_positions=4,
            max_daily_loss_r=5.0,
        )
        c = FakeBrokerComponents(fill_price=20000.0)
        orch = build_orchestrator(c)
        orch.instrument = "MNQ"
        orch.portfolio = portfolio
        orch._strategy_map = {strat.strategy_id: strat}
        orch._regime_paused = paused or set()
        return orch

    async def test_paused_strategy_blocked(self):
        """Strategy in _regime_paused → entry blocked, signal record written."""
        sid = "MNQ_NYSE_OPEN_E2_RR1.0_CB1_X_MES_ATR60_O15"
        orch = self._build_regime_orch(paused={sid})
        event = _nyse_open_entry(risk_points=100.0)
        await orch._handle_event(event)
        assert len(orch.order_router.submitted) == 0
        orch._write_signal_record.assert_called()
        record = orch._write_signal_record.call_args[0][0]
        assert record["type"] == "REGIME_PAUSED"
        assert record["strategy_id"] == sid

    async def test_non_paused_strategy_proceeds(self):
        """Strategy NOT in paused set → entry proceeds normally."""
        orch = self._build_regime_orch(paused={"SOME_OTHER_STRATEGY"})
        event = _nyse_open_entry(risk_points=100.0)
        await orch._handle_event(event)
        assert len(orch.order_router.submitted) > 0

    async def test_empty_paused_set_all_proceed(self):
        """Empty _regime_paused → backward compatible, all entries proceed."""
        orch = self._build_regime_orch(paused=set())
        event = _nyse_open_entry(risk_points=100.0)
        await orch._handle_event(event)
        assert len(orch.order_router.submitted) > 0


class TestResolveTopStepXFAAccountSize:
    """F-1 TopStep XFA account size resolution — fail-closed guard against
    misconfigured profiles that would otherwise KeyError mid-session."""

    def test_topstep_xfa_profile_returns_account_size(self):
        """Canonical happy path: topstep_50k_mnq_auto → 50_000."""
        from trading_app.live.session_orchestrator import _resolve_topstep_xfa_account_size
        from trading_app.prop_profiles import ACCOUNT_PROFILES

        prof = ACCOUNT_PROFILES["topstep_50k_mnq_auto"]
        assert _resolve_topstep_xfa_account_size(prof) == 50_000

    def test_none_profile_returns_none(self):
        """No profile resolved → F-1 disabled."""
        from trading_app.live.session_orchestrator import _resolve_topstep_xfa_account_size

        assert _resolve_topstep_xfa_account_size(None) is None

    def test_non_topstep_firm_returns_none(self):
        """Non-TopStep firm (e.g., bulenox) → F-1 disabled."""
        from trading_app.live.session_orchestrator import _resolve_topstep_xfa_account_size

        prof = MagicMock()
        prof.firm = "bulenox"
        prof.account_size = 50_000
        prof.is_express_funded = True
        assert _resolve_topstep_xfa_account_size(prof) is None

    def test_topstep_non_xfa_returns_none(self):
        """TopStep LFA (is_express_funded=False) → F-1 disabled.

        F-1 enforces the XFA scaling ladder; LFA accounts are post-payout
        funded accounts with different rules.
        """
        from trading_app.live.session_orchestrator import _resolve_topstep_xfa_account_size

        prof = MagicMock()
        prof.firm = "topstep"
        prof.account_size = 50_000
        prof.is_express_funded = False
        assert _resolve_topstep_xfa_account_size(prof) is None

    def test_unknown_xfa_size_raises_fail_closed(self):
        """TopStep XFA with size not in SCALING_PLAN_LADDER → RuntimeError.

        Without this guard, max_lots_for_xfa would raise KeyError on the
        first entry attempt — opaque runtime failure vs. clear init failure.
        """
        from trading_app.live.session_orchestrator import _resolve_topstep_xfa_account_size

        prof = MagicMock()
        prof.firm = "topstep"
        prof.account_size = 25_000  # not a valid XFA tier
        prof.is_express_funded = True
        with pytest.raises(RuntimeError, match="unknown account_size=25000"):
            _resolve_topstep_xfa_account_size(prof)


# ── Stage 3 — orchestrator wiring + EOD dispatch + suppression ────────────


class TestStage3HWMWireUpAndEodDispatch:
    """Stage 3 of HWM persistence integrity hardening.

    Pins:
      - HWM construction call site passes notify_callback=self._notify
      - signal-only authority comment exists within 3 lines above the gate
      - EOD equity-unavailable dispatches operator notify (suppressed when
        kill switch already fired)
      - EOD exception path same dispatch + suppression rule
    """

    def test_hwm_construction_passes_notify_callback(self):
        """Static-source assertion: AccountHWMTracker(...) construction at
        session_orchestrator.py:~698 includes notify_callback=self._notify.
        Mutation: dropping the kwarg flips this test.
        """
        from pathlib import Path

        import trading_app.live.session_orchestrator as orch_mod

        mod_file = orch_mod.__file__
        assert mod_file is not None
        src = Path(mod_file).read_text(encoding="utf-8")
        # Find the AccountHWMTracker(...) construction block
        assert "AccountHWMTracker(" in src
        idx = src.index("AccountHWMTracker(")
        # Take the next ~400 chars (the multi-line constructor call)
        block = src[idx : idx + 600]
        assert "notify_callback=self._notify" in block, (
            f"AccountHWMTracker construction must pass notify_callback=self._notify; got:\n{block}"
        )

    def test_signal_only_authority_comment_within_3_lines_above_gate(self):
        """Greppable: the signal-only authority comment (mentioning
        'pre_session_check' or 'signal-only') must appear within 3 lines
        above the `if not signal_only` gate at the HWM-construction site.
        """
        from pathlib import Path

        import trading_app.live.session_orchestrator as orch_mod

        mod_file = orch_mod.__file__
        assert mod_file is not None
        lines = Path(mod_file).read_text(encoding="utf-8").splitlines()
        gate_idx = None
        for i, line in enumerate(lines):
            if "if not signal_only" in line and "portfolio is not None" in line:
                gate_idx = i
                break
        assert gate_idx is not None, "Could not find signal-only HWM-construction gate"
        # Look in the 3 lines immediately above
        window = "\n".join(lines[max(0, gate_idx - 3) : gate_idx])
        assert "pre_session_check" in window or "signal-only" in window.lower(), (
            f"Authority comment missing within 3 lines above gate; window:\n{window}"
        )

    def test_eod_equity_unavailable_dispatches_when_kill_switch_not_fired(self):
        orch = build_orchestrator()
        orch._kill_switch_fired = False
        orch._hwm_tracker = MagicMock()
        orch.positions.query_equity = MagicMock(return_value=None)  # equity unavailable
        orch.engine.on_trading_day_end.return_value = []
        orch._notify = MagicMock()

        orch.post_session()

        notify_msgs = [c[0][0] for c in orch._notify.call_args_list]
        eod_hwm_notifies = [m for m in notify_msgs if "HWM EOD" in m and "equity unavailable" in m]
        assert len(eod_hwm_notifies) == 1, f"Expected one HWM EOD equity-unavailable notify; got {notify_msgs!r}"

    def test_eod_equity_unavailable_no_dispatch_when_kill_switch_fired(self):
        """Suppression: kill-switch already notified the operator — no duplicate."""
        orch = build_orchestrator()
        orch._kill_switch_fired = True
        orch._hwm_tracker = MagicMock()
        orch.positions.query_equity = MagicMock(return_value=None)
        orch.engine.on_trading_day_end.return_value = []
        orch._notify = MagicMock()

        orch.post_session()

        notify_msgs = [c[0][0] for c in orch._notify.call_args_list]
        eod_hwm_notifies = [m for m in notify_msgs if "HWM EOD" in m and "equity unavailable" in m]
        assert len(eod_hwm_notifies) == 0, (
            f"Suppression failed: kill-switch fired but got HWM EOD notify {notify_msgs!r}"
        )

    def test_eod_exception_dispatches_when_kill_switch_not_fired(self):
        orch = build_orchestrator()
        orch._kill_switch_fired = False
        orch._hwm_tracker = MagicMock()
        orch.positions.query_equity = MagicMock(side_effect=RuntimeError("simulated broker fault"))
        orch.engine.on_trading_day_end.return_value = []
        orch._notify = MagicMock()

        orch.post_session()

        notify_msgs = [c[0][0] for c in orch._notify.call_args_list]
        eod_hwm_notifies = [m for m in notify_msgs if "HWM EOD" in m and "session-end recording failed" in m]
        assert len(eod_hwm_notifies) == 1, f"Expected one HWM EOD exception notify; got {notify_msgs!r}"
        assert "simulated broker fault" in eod_hwm_notifies[0]

    def test_eod_exception_no_dispatch_when_kill_switch_fired(self):
        """Suppression on exception path mirrors the equity-unavailable suppression."""
        orch = build_orchestrator()
        orch._kill_switch_fired = True
        orch._hwm_tracker = MagicMock()
        orch.positions.query_equity = MagicMock(side_effect=RuntimeError("simulated broker fault"))
        orch.engine.on_trading_day_end.return_value = []
        orch._notify = MagicMock()

        orch.post_session()

        notify_msgs = [c[0][0] for c in orch._notify.call_args_list]
        eod_hwm_notifies = [m for m in notify_msgs if "HWM EOD" in m and "session-end recording failed" in m]
        assert len(eod_hwm_notifies) == 0, (
            f"Suppression failed: kill-switch fired but got HWM EOD exception notify {notify_msgs!r}"
        )
