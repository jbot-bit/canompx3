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
import logging
from datetime import date
from pathlib import Path

from trading_app.live.tradovate_auth import TradovateAuth
from trading_app.live.data_feed import DataFeed
from trading_app.live.bar_aggregator import Bar
from trading_app.live.live_market_state import LiveORBBuilder
from trading_app.live.order_router import OrderRouter
from trading_app.live.performance_monitor import PerformanceMonitor, TradeRecord
from trading_app.live.contract_resolver import resolve_front_month
from trading_app.execution_engine import ExecutionEngine
from trading_app.risk_manager import RiskManager, RiskLimits
from trading_app.live_config import build_live_portfolio
from trading_app.portfolio import PortfolioStrategy
from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from pipeline.daily_backfill import run_backfill_for_instrument

log = logging.getLogger(__name__)


class SessionOrchestrator:
    def __init__(self, instrument: str, demo: bool = True, account_id: int = 0):
        self.instrument = instrument
        self.demo = demo
        self.trading_day = date.today()

        self.auth = TradovateAuth(demo=demo)

        # build_live_portfolio returns (Portfolio, list[str]) — unpack the tuple
        self.portfolio, notes = build_live_portfolio(
            db_path=GOLD_DB_PATH, instrument=instrument
        )
        for note in notes:
            log.info("live_config note: %s", note)

        if not self.portfolio.strategies:
            raise RuntimeError(f"No active strategies for {instrument}")

        # Strategy lookup map for resolving entry_model from strategy_id on TradeEvents
        self._strategy_map: dict[str, PortfolioStrategy] = {
            s.strategy_id: s for s in self.portfolio.strategies
        }

        # Execution stack
        cost = get_cost_spec(instrument)
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

        # Live infrastructure
        self.orb_builder = LiveORBBuilder(instrument, self.trading_day)
        self.order_router = OrderRouter(account_id=account_id, auth=self.auth, demo=demo)
        # PerformanceMonitor takes list[PortfolioStrategy] (has strategy_id + expectancy_r)
        self.monitor = PerformanceMonitor(self.portfolio.strategies)

        # Entry price side-table: strategy_id → entry fill price (for pnl_r calculation on exit)
        self._entry_prices: dict[str, float] = {}

        # Resolve front-month contract symbol
        self.contract_symbol = resolve_front_month(instrument, self.auth, demo=demo)
        log.info("Session ready: %s → %s (%s)", instrument, self.contract_symbol,
                 "DEMO" if demo else "LIVE")

        # Signal engine that a new trading day is starting
        self.engine.on_trading_day_start(self.trading_day)

    def _on_bar(self, bar: Bar) -> None:
        """Called for each completed 1-minute bar from DataFeed."""
        self.orb_builder.on_bar(bar)

        # Bar dict must use key 'ts_utc' — that is what ExecutionEngine.on_bar() reads
        bar_dict = {
            "ts_utc": bar.ts_utc,   # NOT ts_event
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        # on_bar() takes ONE argument (bar dict), returns list[TradeEvent]
        events = self.engine.on_bar(bar_dict)

        for event in events:
            self._handle_event(event)

    def _handle_event(self, event) -> None:
        """
        Handle a TradeEvent from ExecutionEngine.
        TradeEvent fields: event_type, strategy_id, timestamp, price, direction, contracts, reason
        event.price = entry fill on ENTRY events, exit fill on EXIT events
        There is NO entry_model, pnl_r, or expectancy_r on TradeEvent.
        """
        strategy = self._strategy_map.get(event.strategy_id)
        if strategy is None:
            log.warning("Unknown strategy_id: %s", event.strategy_id)
            return

        if event.event_type == "ENTRY":
            # Store entry price for pnl calculation on exit
            self._entry_prices[event.strategy_id] = event.price

            # Look up entry_model from PortfolioStrategy (not TradeEvent)
            spec = self.order_router.build_order_spec(
                direction=event.direction,
                entry_model=strategy.entry_model,  # from PortfolioStrategy, not TradeEvent
                entry_price=event.price,           # TradeEvent.price on ENTRY = fill price
                symbol=self.contract_symbol,
                qty=event.contracts,
            )
            result = self.order_router.submit(spec)
            log.info("ENTRY order: %s %s @ %.2f → orderId=%d",
                     event.strategy_id, event.direction, event.price, result.order_id)

        elif event.event_type == "EXIT":
            entry_price = self._entry_prices.pop(event.strategy_id, event.price)
            exit_price = event.price  # TradeEvent.price on EXIT = fill price

            # Compute actual R from prices (engine doesn't put pnl_r on TradeEvent)
            risk_pts = strategy.median_risk_points or 10.0
            direction_sign = 1.0 if event.direction == "long" else -1.0
            gross_pts = direction_sign * (exit_price - entry_price)
            actual_r = gross_pts / risk_pts if risk_pts > 0 else 0.0

            record = TradeRecord(
                strategy_id=event.strategy_id,
                trading_day=self.trading_day,
                direction=event.direction,
                entry_price=entry_price,
                exit_price=exit_price,
                actual_r=actual_r,
                expected_r=strategy.expectancy_r,  # from PortfolioStrategy
            )
            alert = self.monitor.record_trade(record)
            if alert:
                log.warning(alert)
                # TODO Phase F: send to Slack / email

        elif event.event_type == "REJECT":
            log.info("REJECT: %s — %s", event.strategy_id, event.reason)

    async def run(self) -> None:
        feed = DataFeed(self.auth, on_bar=self._on_bar, demo=self.demo)
        log.info("Starting live feed: %s", self.contract_symbol)
        await feed.run(self.contract_symbol)

    def post_session(self) -> None:
        """EOD: close open positions, log summary, run incremental backfill."""
        # Close any positions still open at session end
        eod_events = self.engine.on_trading_day_end()
        for event in eod_events:
            self._handle_event(event)

        summary = self.monitor.daily_summary()
        log.info("EOD summary: %s", summary)
        try:
            run_backfill_for_instrument(self.instrument)
        except Exception as e:
            log.error("EOD backfill failed: %s", e)
