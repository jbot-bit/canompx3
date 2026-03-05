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
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from trading_app.live.tradovate_auth import TradovateAuth
from trading_app.live.data_feed import DataFeed
from trading_app.live.bar_aggregator import Bar
from trading_app.live.live_market_state import LiveORBBuilder
from trading_app.live.order_router import OrderRouter
from trading_app.live.performance_monitor import PerformanceMonitor, TradeRecord
from trading_app.live.contract_resolver import resolve_front_month, resolve_account_id
from trading_app.execution_engine import ExecutionEngine
from trading_app.risk_manager import RiskManager, RiskLimits
from trading_app.live_config import build_live_portfolio
from trading_app.portfolio import PortfolioStrategy
from pipeline.cost_model import get_cost_spec, CostSpec
from pipeline.paths import GOLD_DB_PATH
from pipeline.daily_backfill import run_backfill_for_instrument

log = logging.getLogger(__name__)


class SessionOrchestrator:
    # JSONL file for UI signal display — written by this process, read by Streamlit
    SIGNALS_FILE = Path(__file__).parent.parent.parent / "live_signals.jsonl"

    def __init__(self, instrument: str, demo: bool = True, account_id: int = 0,
                 signal_only: bool = False):
        self.instrument = instrument
        self.demo = demo
        self.signal_only = signal_only
        self.trading_day = date.today()

        # Auth is needed even in signal-only mode (for the market data WebSocket feed)
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

        # Order routing only needed when placing real/demo orders
        if signal_only:
            self.order_router = None
            log.info("Signal-only mode: order router skipped (no Tradovate account needed)")
        else:
            if account_id == 0:
                account_id = resolve_account_id(self.auth, demo=demo)
            self.order_router = OrderRouter(account_id=account_id, auth=self.auth, demo=demo)

        # Live infrastructure
        self.orb_builder = LiveORBBuilder(instrument, self.trading_day)
        # PerformanceMonitor takes list[PortfolioStrategy] (has strategy_id + expectancy_r)
        self.monitor = PerformanceMonitor(self.portfolio.strategies)

        # Entry price side-table: strategy_id → entry fill price (for pnl_r calculation on exit)
        self._entry_prices: dict[str, float] = {}

        # Resolve front-month contract symbol (needed even in signal-only for logging)
        self.contract_symbol = resolve_front_month(instrument, self.auth, demo=demo)
        log.info("Session ready: %s → %s (%s)", instrument, self.contract_symbol,
                 "SIGNAL-ONLY" if signal_only else ("DEMO" if demo else "LIVE"))

        # Write session-start marker to signals file
        self._write_signal_record({
            "type": "SESSION_START",
            "instrument": instrument,
            "contract": self.contract_symbol,
            "mode": "signal_only" if signal_only else ("demo" if demo else "live"),
        })

        # Signal engine that a new trading day is starting
        self.engine.on_trading_day_start(self.trading_day)

    def _write_signal_record(self, extra: dict) -> None:
        """Append a signal record to the JSONL file read by the Live Monitor UI."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "instrument": self.instrument,
            **extra,
        }
        try:
            with open(self.SIGNALS_FILE, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as e:
            log.warning("Could not write signal record: %s", e)

    def _on_bar(self, bar: Bar) -> None:
        """Called for each completed 1-minute bar from DataFeed."""
        self.orb_builder.on_bar(bar)

        # bar.as_dict() returns {ts_utc, open, high, low, close, volume}
        # — exactly what ExecutionEngine.on_bar() expects
        events = self.engine.on_bar(bar.as_dict())

        for event in events:
            self._handle_event(event)

    def _close_position(self, event) -> None:
        """Submit a closing market order to the broker for an EXIT or SCRATCH."""
        exit_spec = self.order_router.build_exit_spec(
            direction=event.direction,
            symbol=self.contract_symbol,
            qty=event.contracts,
        )
        result = self.order_router.submit(exit_spec)
        log.info("%s close order: %s %s @ %.2f → orderId=%d",
                 event.event_type, event.strategy_id, event.direction,
                 event.price, result.order_id)

    def _compute_actual_r(self, entry_price: float, exit_price: float,
                          direction: str, risk_pts: float) -> float:
        """Compute cost-adjusted R-multiple from entry/exit prices."""
        direction_sign = 1.0 if direction == "long" else -1.0
        gross_pts = direction_sign * (exit_price - entry_price)
        # Subtract transaction costs (spread + slippage + commission) in points
        net_pts = gross_pts - self.cost_spec.friction_in_points
        return net_pts / risk_pts if risk_pts > 0 else 0.0

    def _record_exit(self, event, entry_price: float) -> None:
        """Record a completed trade (EXIT or SCRATCH) in the performance monitor."""
        strategy = self._strategy_map[event.strategy_id]
        risk_pts = strategy.median_risk_points or 10.0
        actual_r = self._compute_actual_r(
            entry_price, event.price, event.direction, risk_pts
        )
        record = TradeRecord(
            strategy_id=event.strategy_id,
            trading_day=self.trading_day,
            direction=event.direction,
            entry_price=entry_price,
            exit_price=event.price,
            actual_r=actual_r,
            expected_r=strategy.expectancy_r,
        )
        alert = self.monitor.record_trade(record)
        if alert:
            log.warning(alert)

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
            self._entry_prices[event.strategy_id] = event.price

            if self.signal_only:
                log.info(
                    "⚡ SIGNAL [%s]: %s %s @ %.2f  ← trade this manually on Tradovate/TradingView",
                    event.strategy_id, event.direction.upper(), self.contract_symbol, event.price,
                )
                self._write_signal_record({
                    "type": "SIGNAL_ENTRY",
                    "strategy_id": event.strategy_id,
                    "contract": self.contract_symbol,
                    "direction": event.direction.upper(),
                    "price": event.price,
                    "contracts": event.contracts,
                })
                return

            spec = self.order_router.build_order_spec(
                direction=event.direction,
                entry_model=strategy.entry_model,
                entry_price=event.price,
                symbol=self.contract_symbol,
                qty=event.contracts,
            )
            result = self.order_router.submit(spec)
            log.info("ENTRY order: %s %s @ %.2f → orderId=%d",
                     event.strategy_id, event.direction, event.price, result.order_id)
            self._write_signal_record({
                "type": "ORDER_ENTRY",
                "strategy_id": event.strategy_id,
                "contract": self.contract_symbol,
                "direction": event.direction.upper(),
                "price": event.price,
                "contracts": event.contracts,
                "order_id": result.order_id,
            })

        elif event.event_type in ("EXIT", "SCRATCH"):
            entry_price = self._entry_prices.pop(event.strategy_id, event.price)

            if self.signal_only:
                log.info(
                    "⚡ EXIT SIGNAL [%s]: close %s @ %.2f  ← close manually",
                    event.strategy_id, event.direction.upper(), event.price,
                )
                self._record_exit(event, entry_price)
                self._write_signal_record({
                    "type": "SIGNAL_EXIT",
                    "strategy_id": event.strategy_id,
                    "contract": self.contract_symbol,
                    "direction": event.direction.upper(),
                    "price": event.price,
                })
                return

            self._close_position(event)
            self._record_exit(event, entry_price)
            self._write_signal_record({
                "type": f"ORDER_{event.event_type}",
                "strategy_id": event.strategy_id,
                "contract": self.contract_symbol,
                "direction": event.direction.upper(),
                "price": event.price,
            })

        elif event.event_type == "REJECT":
            log.info("REJECT: %s — %s", event.strategy_id, event.reason)
            self._write_signal_record({
                "type": "REJECT",
                "strategy_id": event.strategy_id,
                "reason": getattr(event, "reason", ""),
            })

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
