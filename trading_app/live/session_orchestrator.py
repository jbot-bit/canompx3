"""
Live trading session orchestrator.
DataFeed → BarAggregator → ExecutionEngine → OrderRouter → PerformanceMonitor.

VERIFIED API NOTES:
- build_live_portfolio() returns (Portfolio, notes) — unpack the tuple
- engine.on_bar(bar_dict) — bar_dict must have 'ts_utc' key, not 'ts_event'
- engine.on_trading_day_start(date) — call before first bar of day
- engine.on_trading_day_end() -> list[TradeEvent] — closes open positions at EOD
- TradeEvent.event_type: "ENTRY" or "EXIT" (not TradeState enum)
- TradeEvent.price: entry fill on ENTRY, exit fill on EXIT
- entry_model: look up from self._strategy_map[event.strategy_id].entry_model
"""

import asyncio
import json
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pipeline.calendar_filters import day_of_week, is_friday, is_nfp_day, is_opex_day
from pipeline.cost_model import CostSpec, get_cost_spec
from pipeline.daily_backfill import run_backfill_for_instrument
from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.execution_engine import ExecutionEngine
from trading_app.live.bar_aggregator import Bar
from trading_app.live.broker_factory import create_broker_components, get_broker_name
from trading_app.live.live_market_state import LiveORBBuilder
from trading_app.live.performance_monitor import PerformanceMonitor, TradeRecord
from trading_app.live.position_tracker import PositionTracker
from trading_app.live_config import build_live_portfolio
from trading_app.portfolio import PortfolioStrategy
from trading_app.risk_manager import RiskLimits, RiskManager

log = logging.getLogger(__name__)


class SessionOrchestrator:
    # JSONL file for UI signal display — written by this process, read by Streamlit
    SIGNALS_FILE = Path(__file__).parent.parent.parent / "live_signals.jsonl"

    def __init__(
        self,
        instrument: str,
        broker: str | None = None,
        demo: bool = True,
        account_id: int = 0,
        signal_only: bool = False,
        force_orphans: bool = False,
    ):
        self.instrument = instrument
        self.demo = demo
        self.signal_only = signal_only
        # Trading day = 09:00 Brisbane → 09:00 next day Brisbane.
        # If started before 09:00, we're still in yesterday's trading day.
        bris_now = datetime.now(ZoneInfo("Australia/Brisbane"))
        if bris_now.hour < 9:
            self.trading_day = (bris_now - timedelta(days=1)).date()
        else:
            self.trading_day = bris_now.date()

        # Create broker components via factory
        self._broker_name = broker or get_broker_name()
        components = create_broker_components(self._broker_name, demo=demo)
        self.auth = components["auth"]
        self._feed_class = components["feed_class"]
        contracts_cls = components["contracts_class"]
        self._positions_cls = components["positions_class"]

        # build_live_portfolio returns (Portfolio, list[str]) — unpack the tuple
        self.portfolio, notes = build_live_portfolio(db_path=GOLD_DB_PATH, instrument=instrument)
        for note in notes:
            log.info("live_config note: %s", note)

        if not self.portfolio.strategies:
            raise RuntimeError(f"No active strategies for {instrument}")

        # Strategy lookup map for resolving entry_model from strategy_id on TradeEvents
        self._strategy_map: dict[str, PortfolioStrategy] = {s.strategy_id: s for s in self.portfolio.strategies}

        # Execution stack
        self.cost_spec: CostSpec = get_cost_spec(instrument)
        cost = self.cost_spec
        risk_limits = RiskLimits(
            max_daily_loss_r=self.portfolio.max_daily_loss_r,
            max_concurrent_positions=self.portfolio.max_concurrent_positions,
        )
        self.risk_mgr = RiskManager(risk_limits)
        self.engine = ExecutionEngine(
            portfolio=self.portfolio,
            cost_spec=cost,
            risk_manager=self.risk_mgr,
            live_session_costs=True,
        )

        # Contract resolution (needed even in signal-only for front-month lookup)
        contracts = contracts_cls(auth=self.auth, demo=demo)

        # Order routing only needed when placing real/demo orders
        if signal_only:
            self.order_router = None
            self.positions = None
            log.info("Signal-only mode: order router skipped")
        else:
            if account_id == 0:
                account_id = contracts.resolve_account_id()
            router_cls = components["router_class"]
            self.order_router = router_cls(account_id=account_id, auth=self.auth, demo=demo)
            self.positions = self._positions_cls(auth=self.auth)

            # Position reconciliation on startup (M2.5 P0: crash recovery)
            try:
                orphans = self.positions.query_open(account_id)
                if orphans:
                    log.critical("ORPHANED POSITIONS DETECTED on session start: %s", orphans)
                    if not force_orphans:
                        raise RuntimeError(
                            f"Refusing to start: {len(orphans)} orphaned position(s) detected. "
                            f"Close them manually or pass --force-orphans to acknowledge the risk."
                        )
                    log.warning("--force-orphans: continuing with %d orphaned position(s)", len(orphans))
            except NotImplementedError:
                log.warning(
                    "ORPHAN DETECTION DISABLED — %s broker adapter does not implement query_open(). "
                    "You must manually verify no orphaned positions exist before trading.",
                    self._broker_name,
                )
            except RuntimeError:
                raise  # re-raise our own orphan-blocking error
            except Exception as e:
                log.warning("Position query failed on startup: %s", e)

        # Live infrastructure
        self.orb_builder = LiveORBBuilder(instrument, self.trading_day)
        # PerformanceMonitor takes list[PortfolioStrategy] (has strategy_id + expectancy_r)
        self.monitor = PerformanceMonitor(self.portfolio.strategies)

        # Position lifecycle tracker — replaces ad-hoc _entry_prices dict
        self._positions = PositionTracker()
        self._last_bar_at: datetime | None = None  # bar heartbeat
        self._kill_switch_fired = False  # one-shot emergency flatten

        # Circuit breaker: blocks order submission after 5 consecutive broker failures
        from trading_app.live.circuit_breaker import CircuitBreaker

        self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        # Resolve front-month contract symbol (needed even in signal-only for logging)
        self.contract_symbol = contracts.resolve_front_month(instrument)
        log.info(
            "Session ready: %s → %s (%s)",
            instrument,
            self.contract_symbol,
            "SIGNAL-ONLY" if signal_only else ("DEMO" if demo else "LIVE"),
        )

        # Write session-start marker to signals file
        self._write_signal_record(
            {
                "type": "SESSION_START",
                "instrument": instrument,
                "contract": self.contract_symbol,
                "mode": "signal_only" if signal_only else ("demo" if demo else "live"),
            }
        )

        # Build partial daily_features_row from what's available pre-session.
        # Without this, fail-closed filters (VOL_RV12_N20, DOW, calendar) silently
        # reject ALL trades because their required columns are missing.
        daily_row = self._build_daily_features_row(self.trading_day, instrument)

        # Signal engine that a new trading day is starting
        self.engine.on_trading_day_start(self.trading_day, daily_features_row=daily_row)

    @staticmethod
    def _build_daily_features_row(trading_day: date, instrument: str) -> dict:
        """Build a daily_features_row from DB + calendar for live execution.

        Without this, fail-closed filters (VOL_RV12_N20, DOW, break speed, calendar)
        silently reject ALL trades because their required columns are None.

        Populates:
          - Calendar (exact for today): is_nfp_day, is_opex_day, is_friday, day_of_week
          - From most recent DB row (yesterday's proxy): atr_20, atr_vel_regime,
            compression tiers, rel_vol_*, break_delay_min, etc.
          - Computed: median_atr_20 (rolling 252-day median for vol-scaling)

        The rel_vol_* and break_delay_min values are yesterday's — imperfect but
        vastly better than None (which silently kills every VOL/FAST strategy).
        orb_{label}_size is set by ExecutionEngine from live ORB, overriding the DB value.
        """
        row: dict = {}

        # Calendar flags — exact for today
        row["is_nfp_day"] = is_nfp_day(trading_day)
        row["is_opex_day"] = is_opex_day(trading_day)
        row["is_friday"] = is_friday(trading_day)
        row["day_of_week"] = day_of_week(trading_day)

        # Load ALL columns from the most recent daily_features row.
        # This gives filters yesterday's values as proxies for today.
        try:
            import duckdb

            con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
            try:
                configure_connection(con)
                df = con.execute(
                    """
                    SELECT * FROM daily_features
                    WHERE symbol = ? AND orb_minutes = 5
                      AND trading_day = (
                          SELECT MAX(trading_day) FROM daily_features
                          WHERE symbol = ? AND orb_minutes = 5
                      )
                """,
                    [instrument, instrument],
                ).fetchdf()
                if not df.empty:
                    latest = df.iloc[0].to_dict()
                    # Merge all DB columns (ATR, compression, rel_vol, break speed, etc.)
                    # Calendar flags from above override any DB values (exact for today).
                    for k, v in latest.items():
                        if k not in row:  # don't overwrite today's calendar flags
                            row[k] = v
                    # Staleness warning — daily_features should be from yesterday or today
                    latest_day = latest.get("trading_day")
                    if latest_day is not None:
                        gap = (trading_day - (latest_day.date() if hasattr(latest_day, "date") else latest_day)).days
                        if gap > 5:
                            raise RuntimeError(
                                f"FAIL-CLOSED: daily_features is {gap} days stale (latest: {latest_day}). "
                                f"Filters would silently reject all trades. Run: "
                                f"python pipeline/build_daily_features.py --instrument {instrument}"
                            )
                        elif gap > 3:
                            log.warning("daily_features data is %d days stale (latest: %s)", gap, latest_day)

                # median_atr_20 is NOT in daily_features — it's a rolling median computed
                # by paper_trader. Compute it here for live vol-scaling.
                median_result = con.execute(
                    """
                    SELECT MEDIAN(atr_20) FROM daily_features
                    WHERE symbol = ? AND orb_minutes = 5 AND atr_20 IS NOT NULL
                      AND trading_day < ? AND trading_day >= ? - INTERVAL '504 DAY'
                """,
                    [instrument, trading_day, trading_day],
                ).fetchone()
                if median_result and median_result[0] is not None:
                    row["median_atr_20"] = float(median_result[0])
            finally:
                con.close()
        except RuntimeError:
            raise  # re-raise our own fail-closed errors
        except Exception as e:
            raise RuntimeError(
                f"FAIL-CLOSED: Cannot load daily_features for {instrument}: {e}. "
                f"Filters would silently reject all trades. Fix the database or run: "
                f"python pipeline/build_daily_features.py --instrument {instrument}"
            ) from e

        log.info(
            "Daily features row: atr_20=%s, atr_vel=%s, nfp=%s, opex=%s, dow=%s",
            row.get("atr_20"),
            row.get("atr_vel_regime"),
            row.get("is_nfp_day"),
            row.get("is_opex_day"),
            row.get("day_of_week"),
        )
        return row

    def _write_signal_record(self, extra: dict) -> None:
        """Append a signal record to the JSONL file read by the Live Monitor UI."""
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "instrument": self.instrument,
            **extra,
        }
        try:
            with open(self.SIGNALS_FILE, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as e:
            log.warning("Could not write signal record: %s", e)

    def _check_trading_day_rollover(self, bar_ts_utc) -> None:
        """Detect 9:00 AM Brisbane boundary crossing and roll to new trading day.

        Without this, a session running past 9 AM Brisbane would continue using
        yesterday's ORB windows, calendar flags, daily P&L, and risk limits.
        """
        _bris = ZoneInfo("Australia/Brisbane")
        bris_time = bar_ts_utc.astimezone(_bris)
        if bris_time.hour < 9:
            bar_trading_day = (bris_time - timedelta(days=1)).date()
        else:
            bar_trading_day = bris_time.date()

        if bar_trading_day == self.trading_day:
            return

        log.info("Trading day rollover: %s -> %s", self.trading_day, bar_trading_day)

        # Close previous day's open positions
        eod_events = self.engine.on_trading_day_end()
        for event in eod_events:
            strategy = self._strategy_map.get(event.strategy_id)
            if strategy and event.event_type in ("EXIT", "SCRATCH"):
                entry_price = self._positions.best_entry_price(event.strategy_id, event.price)
                self._positions.pop(event.strategy_id)
                self._record_exit(event, entry_price)
                log.info("EOD rollover close: %s @ %.2f (pnl_r=%s)", event.strategy_id, event.price, event.pnl_r)

        # Start new trading day
        self.trading_day = bar_trading_day
        daily_row = self._build_daily_features_row(self.trading_day, self.instrument)
        self.engine.on_trading_day_start(self.trading_day, daily_features_row=daily_row)
        self.orb_builder = LiveORBBuilder(self.instrument, self.trading_day)
        self.monitor.reset_daily()
        self.risk_mgr.daily_reset(self.trading_day)
        log.info("New trading day started: %s", self.trading_day)

    async def _on_bar(self, bar: Bar) -> None:
        """Called for each completed 1-minute bar from DataFeed."""
        # Bar heartbeat monitoring
        now = datetime.now(UTC)
        if self._last_bar_at is not None:
            gap = (now - self._last_bar_at).total_seconds()
            if gap > 180:  # 3 minutes without a bar
                log.critical("BAR HEARTBEAT: %.0fs since last bar — feed may be dead", gap)
                stale = self._positions.stale_positions(timeout_seconds=300)
                if stale:
                    log.critical("STALE ORDERS: %s", [(s.strategy_id, s.state.value) for s in stale])
        self._last_bar_at = now

        # Kill switch fired = we already emergency-flattened at the broker.
        # Do NOT process further bars — engine doesn't know positions are closed,
        # so it would generate duplicate EXIT orders for already-flattened positions.
        if self._kill_switch_fired:
            return

        # Check if we've crossed the 9:00 AM Brisbane boundary
        self._check_trading_day_rollover(bar.ts_utc)

        self.orb_builder.on_bar(bar)

        # bar.as_dict() returns {ts_utc, open, high, low, close, volume}
        # — exactly what ExecutionEngine.on_bar() expects
        events = self.engine.on_bar(bar.as_dict())

        for event in events:
            await self._handle_event(event)

    def _compute_actual_r(self, entry_price: float, exit_price: float, direction: str, risk_pts: float) -> float:
        """Compute cost-adjusted R-multiple from entry/exit prices."""
        direction_sign = 1.0 if direction == "long" else -1.0
        gross_pts = direction_sign * (exit_price - entry_price)
        # Subtract transaction costs (spread + slippage + commission) in points
        net_pts = gross_pts - self.cost_spec.friction_in_points
        return net_pts / risk_pts if risk_pts > 0 else 0.0

    def _record_exit(self, event, entry_price: float, exit_fill_price: float | None = None) -> None:
        """Record a completed trade (EXIT or SCRATCH) in the performance monitor.

        Uses broker fill prices when available for more accurate P&L.
        Falls back to engine prices (event.price) when fills are unknown.
        """
        strategy = self._strategy_map[event.strategy_id]
        exit_price = exit_fill_price if exit_fill_price is not None else event.price

        # Use engine's authoritative pnl_r (session-adjusted costs) when available;
        # fall back to local computation only if not present.
        if event.pnl_r is not None:
            actual_r = event.pnl_r
        else:
            risk_pts = strategy.median_risk_points or 10.0
            actual_r = self._compute_actual_r(entry_price, exit_price, event.direction, risk_pts)

        # Compute total slippage (entry + exit) in points
        slippage_pts = 0.0
        if exit_fill_price is not None:
            slippage_pts += exit_fill_price - event.price

        record = TradeRecord(
            strategy_id=event.strategy_id,
            trading_day=self.trading_day,
            direction=event.direction,
            entry_price=entry_price,
            exit_price=exit_price,
            actual_r=actual_r,
            expected_r=strategy.expectancy_r,
            slippage_pts=slippage_pts,
        )
        alert = self.monitor.record_trade(record)
        if alert:
            log.warning(alert)

    async def _handle_event(self, event) -> None:
        """
        Handle a TradeEvent from ExecutionEngine.
        TradeEvent fields: event_type, strategy_id, timestamp, price, direction, contracts, reason
        event.price = entry fill on ENTRY events, exit fill on EXIT events
        There is NO entry_model, pnl_r, or expectancy_r on TradeEvent.

        HTTP order submission runs in a thread-pool executor so the async event loop
        (and the Tradovate heartbeat task) are never blocked.
        """
        strategy = self._strategy_map.get(event.strategy_id)
        if strategy is None:
            log.warning("Unknown strategy_id: %s", event.strategy_id)
            return

        if event.event_type == "ENTRY":
            if self.signal_only:
                self._positions.on_signal_entry(
                    event.strategy_id, event.price, event.direction, contracts=event.contracts
                )
                log.info(
                    "⚡ SIGNAL [%s]: %s %s @ %.2f  ← trade this manually on Tradovate/TradingView",
                    event.strategy_id,
                    event.direction.upper(),
                    self.contract_symbol,
                    event.price,
                )
                self._write_signal_record(
                    {
                        "type": "SIGNAL_ENTRY",
                        "strategy_id": event.strategy_id,
                        "contract": self.contract_symbol,
                        "direction": event.direction.upper(),
                        "price": event.price,
                        "contracts": event.contracts,
                    }
                )
                return

            if not self._circuit_breaker.should_allow_request():
                log.critical("CIRCUIT BREAKER OPEN — skipping ENTRY for %s", event.strategy_id)
                self._write_signal_record({"type": "CIRCUIT_BREAKER", "strategy_id": event.strategy_id})
                return

            spec = self.order_router.build_order_spec(
                direction=event.direction,
                entry_model=strategy.entry_model,
                entry_price=event.price,
                symbol=self.contract_symbol,
                qty=event.contracts,
            )
            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(None, self.order_router.submit, spec)
                self._circuit_breaker.record_success()
            except Exception as e:
                self._circuit_breaker.record_failure()
                log.error("ENTRY order failed for %s: %s", event.strategy_id, e)
                return
            order_id = result.get("order_id") if isinstance(result, dict) else result.order_id
            fill_price = result.get("fill_price") if isinstance(result, dict) else getattr(result, "fill_price", None)

            # Track entry via position tracker
            self._positions.on_entry_sent(
                event.strategy_id, event.direction, event.price, order_id=order_id, contracts=event.contracts
            )
            if fill_price is not None:
                self._positions.on_entry_filled(event.strategy_id, fill_price)
                slippage = fill_price - event.price
                log.info(
                    "ENTRY FILL: %s %s engine=%.2f fill=%.2f slip=%+.4f pts → orderId=%s",
                    event.strategy_id,
                    event.direction,
                    event.price,
                    fill_price,
                    slippage,
                    order_id,
                )
            else:
                log.info(
                    "ENTRY order: %s %s @ %.2f → orderId=%s (fill pending)",
                    event.strategy_id,
                    event.direction,
                    event.price,
                    order_id,
                )

            self._write_signal_record(
                {
                    "type": "ORDER_ENTRY",
                    "strategy_id": event.strategy_id,
                    "contract": self.contract_symbol,
                    "direction": event.direction.upper(),
                    "price": event.price,
                    "fill_price": fill_price,
                    "contracts": event.contracts,
                    "order_id": order_id,
                }
            )

        elif event.event_type in ("EXIT", "SCRATCH"):
            entry_price = self._positions.best_entry_price(event.strategy_id, event.price)
            if self._positions.get(event.strategy_id) is None:
                log.warning("EXIT for %s with no prior ENTRY — using engine exit price as fallback", event.strategy_id)

            if self.signal_only:
                log.info(
                    "⚡ EXIT SIGNAL [%s]: close %s @ %.2f  ← close manually",
                    event.strategy_id,
                    event.direction.upper(),
                    event.price,
                )
                self._record_exit(event, entry_price)
                self._positions.pop(event.strategy_id)
                self._write_signal_record(
                    {
                        "type": "SIGNAL_EXIT",
                        "strategy_id": event.strategy_id,
                        "contract": self.contract_symbol,
                        "direction": event.direction.upper(),
                        "price": event.price,
                    }
                )
                return

            # Submit close order and capture exit fill
            # NOTE: exits NEVER blocked by circuit breaker — can't leave positions open
            if not self._circuit_breaker.should_allow_request():
                log.critical(
                    "CIRCUIT BREAKER OPEN — EXIT for %s submitted anyway (can't leave positions open)",
                    event.strategy_id,
                )

            self._positions.on_exit_sent(event.strategy_id)
            exit_spec = self.order_router.build_exit_spec(
                direction=event.direction,
                symbol=self.contract_symbol,
                qty=event.contracts,
            )
            loop = asyncio.get_running_loop()
            try:
                result = await loop.run_in_executor(None, self.order_router.submit, exit_spec)
                self._circuit_breaker.record_success()
            except Exception as e:
                self._circuit_breaker.record_failure()
                log.critical("EXIT order FAILED for %s: %s — MANUAL CLOSE REQUIRED", event.strategy_id, e)
                self._write_signal_record({"type": "EXIT_FAILED", "strategy_id": event.strategy_id, "error": str(e)})
                return
            order_id = result.get("order_id") if isinstance(result, dict) else result.order_id
            exit_fill = result.get("fill_price") if isinstance(result, dict) else getattr(result, "fill_price", None)

            if exit_fill is not None:
                exit_slip = exit_fill - event.price
                log.info(
                    "EXIT FILL: %s %s engine=%.2f fill=%.2f slip=%+.4f pts → orderId=%s",
                    event.strategy_id,
                    event.direction,
                    event.price,
                    exit_fill,
                    exit_slip,
                    order_id,
                )
            else:
                log.info(
                    "%s close order: %s %s @ %.2f → orderId=%s",
                    event.event_type,
                    event.strategy_id,
                    event.direction,
                    event.price,
                    order_id,
                )

            self._positions.on_exit_filled(event.strategy_id, fill_price=exit_fill)
            self._record_exit(event, entry_price, exit_fill_price=exit_fill)
            self._write_signal_record(
                {
                    "type": f"ORDER_{event.event_type}",
                    "strategy_id": event.strategy_id,
                    "contract": self.contract_symbol,
                    "direction": event.direction.upper(),
                    "price": event.price,
                    "fill_price": exit_fill,
                }
            )

        elif event.event_type == "REJECT":
            log.info("REJECT: %s — %s", event.strategy_id, event.reason)
            self._write_signal_record(
                {
                    "type": "REJECT",
                    "strategy_id": event.strategy_id,
                    "reason": getattr(event, "reason", ""),
                }
            )

    # Kill switch: emergency flatten if feed dies with open positions.
    # 5 minutes of silence = assume feed is dead. This is the last line of defense.
    KILL_SWITCH_TIMEOUT = 300.0  # seconds without a bar before emergency flatten
    KILL_SWITCH_CHECK_INTERVAL = 30.0  # how often the watchdog checks

    async def _emergency_flatten(self) -> None:
        """Nuclear option: market-close every open position immediately.

        Runs when the feed is dead and we're blind with open exposure.
        Retries aggressively — the goal is to get flat no matter what.
        """
        active = self._positions.active_positions()
        if not active:
            return

        log.critical(
            "*** KILL SWITCH *** Feed dead for >%.0fs with %d open position(s). Emergency flatten ALL positions.",
            self.KILL_SWITCH_TIMEOUT,
            len(active),
        )
        self._write_signal_record(
            {
                "type": "KILL_SWITCH",
                "reason": "feed_dead",
                "positions": [r.strategy_id for r in active],
            }
        )

        if self.signal_only or self.order_router is None:
            log.critical(
                "*** MANUAL CLOSE REQUIRED *** Signal-only mode — cannot auto-flatten. Close these positions NOW: %s",
                [r.strategy_id for r in active],
            )
            return

        loop = asyncio.get_running_loop()
        for record in active:
            direction = record.direction or "long"
            for attempt in range(3):
                try:
                    self.auth.refresh_if_needed()
                    exit_spec = self.order_router.build_exit_spec(
                        direction=direction,
                        symbol=self.contract_symbol,
                        qty=record.contracts,
                    )
                    result = await loop.run_in_executor(None, self.order_router.submit, exit_spec)
                    order_id = result.get("order_id") if isinstance(result, dict) else result.order_id
                    log.critical(
                        "KILL SWITCH FLATTEN: %s %s → orderId=%s (attempt %d)",
                        record.strategy_id,
                        direction,
                        order_id,
                        attempt + 1,
                    )
                    self._positions.on_exit_filled(record.strategy_id)
                    break
                except Exception as e:
                    log.critical(
                        "KILL SWITCH FLATTEN FAILED: %s attempt %d/3 — %s",
                        record.strategy_id,
                        attempt + 1,
                        e,
                    )
                    await asyncio.sleep(2**attempt)
            else:
                log.critical(
                    "*** MANUAL CLOSE REQUIRED *** Failed to flatten %s after 3 attempts. "
                    "Close this position IMMEDIATELY on your broker platform.",
                    record.strategy_id,
                )

    async def _watchdog(self) -> None:
        """Independent watchdog task — fires kill switch if feed goes silent.

        Runs on its own asyncio schedule, not dependent on bar arrival.
        This is the fail-safe that protects against the feed dying silently.
        The watchdog MUST NOT die — it wraps everything in try/except so that
        a transient error doesn't kill the last line of defense.
        """
        while True:
            try:
                await asyncio.sleep(self.KILL_SWITCH_CHECK_INTERVAL)

                if self._kill_switch_fired:
                    continue  # already fired, don't spam

                if self._last_bar_at is None:
                    continue  # haven't received any bars yet

                gap = (datetime.now(UTC) - self._last_bar_at).total_seconds()
                if gap > self.KILL_SWITCH_TIMEOUT and self._positions.active_positions():
                    self._kill_switch_fired = True
                    await self._emergency_flatten()
            except asyncio.CancelledError:
                raise  # normal shutdown
            except Exception as e:
                log.error("Watchdog error (will retry): %s", e)

    async def run(self) -> None:
        feed = self._feed_class(self.auth, on_bar=self._on_bar, demo=self.demo)
        log.info("Starting live feed: %s (broker: %s)", self.contract_symbol, self._broker_name)
        watchdog = asyncio.create_task(self._watchdog())
        try:
            await feed.run(self.contract_symbol)
        finally:
            watchdog.cancel()

    def post_session(self) -> None:
        """EOD: close open positions, log summary, run incremental backfill.

        Called from a synchronous finally block after asyncio.run() completes,
        so we start a fresh event loop for the async close operations.
        Each event is wrapped individually so one failure doesn't abort the rest
        (CRIT-4: preventing open positions from being abandoned on error).
        """
        # If kill switch already fired, positions are already closed at the broker.
        # Skip EOD close to avoid duplicate close orders.
        if self._kill_switch_fired:
            log.info("Kill switch was activated — skipping EOD close (positions already flattened)")
            eod_events = []
        else:
            eod_events = self.engine.on_trading_day_end()

        async def _close_all() -> None:
            for event in eod_events:
                try:
                    await self._handle_event(event)
                except Exception as e:
                    log.error(
                        "EOD close failed for %s (%s) — position may remain open: %s",
                        event.strategy_id,
                        event.event_type,
                        e,
                    )

        if eod_events and not self.signal_only:
            # Pre-refresh auth token before close loop — if network is back,
            # this ensures we have a valid token. If still down, we log CRITICAL.
            auth_ok = False
            for attempt in range(3):
                try:
                    self.auth.get_token()
                    auth_ok = True
                    break
                except Exception as e:
                    log.critical("Auth refresh attempt %d/3 failed before EOD close: %s", attempt + 1, e)
                    import time

                    time.sleep(2**attempt)
            if not auth_ok:
                log.critical(
                    "*** MANUAL CLOSE REQUIRED *** Auth failed after 3 attempts. %d position(s) may remain open: %s",
                    len(eod_events),
                    [e.strategy_id for e in eod_events],
                )
            asyncio.run(_close_all())
        elif eod_events:
            asyncio.run(_close_all())

        summary = self.monitor.daily_summary()
        log.info("EOD summary: %s", summary)

        # EOD position reconciliation (M2.5 P0)
        if self.positions and not self.signal_only:
            try:
                account_id = self.order_router.account_id if self.order_router else 0
                remaining = self.positions.query_open(account_id)
                if remaining:
                    log.critical(
                        "EOD RECONCILIATION: %d positions still open after session end: %s", len(remaining), remaining
                    )
                else:
                    log.info("EOD reconciliation: no orphaned positions")
            except Exception as e:
                log.warning("EOD position reconciliation failed: %s", e)

        try:
            run_backfill_for_instrument(self.instrument)
        except Exception as e:
            log.error("EOD backfill failed: %s", e)
