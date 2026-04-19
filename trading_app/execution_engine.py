"""
Execution engine for live/replay bar-by-bar strategy execution.

Processes 1-minute bars through a state machine:
  CONFIRMING -> ARMED -> ENTERED -> EXITED

Reuses config.py filters for ORB size filtering.
Optionally integrates RiskManager for position limits and circuit breaker.

Usage:
    engine = ExecutionEngine(portfolio, cost_spec)
    engine.on_trading_day_start(trading_day)
    for bar in bars:
        events = engine.on_bar(bar)
    engine.on_trading_day_end()
"""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import Enum

from pipeline.cost_model import CostSpec, get_session_cost_spec, to_r_multiple
from pipeline.dst import DYNAMIC_ORB_RESOLVERS, orb_utc_window
from pipeline.log import get_logger
from trading_app.calendar_overlay import CalendarAction, get_calendar_action
from trading_app.config import (
    ALL_FILTERS,
    EARLY_EXIT_MINUTES,
    HOLD_HOURS,
    IB_DURATION_MINUTES,
    SESSION_EXIT_MODE,
    is_e2_lookahead_filter,
)
from trading_app.portfolio import (
    Portfolio,
    PortfolioStrategy,
    compute_position_size_vol_scaled,
    compute_vol_scalar,
)

logger = get_logger(__name__)

# All sessions are now dynamic (DST-aware), resolved per-day via DYNAMIC_ORB_RESOLVERS.
# ORB_WINDOWS_UTC kept as empty dict for backward compatibility with existing imports.
ORB_WINDOWS_UTC: dict[str, tuple[int, int]] = {}


class TradeState(Enum):
    ARMED = "ARMED"
    CONFIRMING = "CONFIRMING"
    ENTERED = "ENTERED"
    EXITED = "EXITED"


@dataclass
class TradeEvent:
    """Abstract trade event (broker-agnostic)."""

    event_type: str  # "ENTRY", "EXIT", "SCRATCH", "REJECT"
    strategy_id: str
    timestamp: datetime
    price: float
    direction: str
    contracts: int
    reason: str
    pnl_r: float | None = None  # Populated on EXIT/SCRATCH events
    risk_points: float | None = None  # Actual risk at fill — used for broker brackets


@dataclass
class LiveORB:
    """ORB range being built or completed."""

    label: str
    window_start_utc: datetime
    window_end_utc: datetime
    orb_minutes: int = 5
    high: float | None = None
    low: float | None = None
    complete: bool = False
    break_dir: str | None = None
    break_ts: datetime | None = None
    e2_touched: bool = False  # True once E2 stop-market has triggered on first touch
    complete_ts: datetime | None = None  # Timestamp when ORB window closed

    @property
    def size(self) -> float | None:
        if self.high is not None and self.low is not None:
            return self.high - self.low
        return None


@dataclass
class LiveIB:
    """Initial Balance (first 120 minutes from 09:00 Brisbane = 23:00 UTC).

    Tracks IB high/low during formation, then detects first break after IB ends.
    Used by TOKYO_OPEN session for IB-conditional exit logic.
    """

    window_start_utc: datetime
    window_end_utc: datetime
    high: float | None = None
    low: float | None = None
    complete: bool = False
    break_dir: str | None = None
    break_ts: datetime | None = None

    def update(self, bar: dict) -> None:
        """Update IB range from a bar during formation period."""
        ts = bar["ts_utc"]
        if not self.complete and self.window_start_utc <= ts < self.window_end_utc:
            if self.high is None or bar["high"] > self.high:
                self.high = bar["high"]
            if self.low is None or bar["low"] < self.low:
                self.low = bar["low"]
        if not self.complete and ts >= self.window_end_utc:
            self.complete = True

    def check_break(self, bar: dict) -> str | None:
        """Check if bar breaks IB after formation. Returns 'long'/'short'/None."""
        if not self.complete or self.break_dir is not None:
            return None
        if self.high is None or self.low is None:
            return None
        if bar["close"] > self.high:
            self.break_dir = "long"
            self.break_ts = bar["ts_utc"]
            return "long"
        elif bar["close"] < self.low:
            self.break_dir = "short"
            self.break_ts = bar["ts_utc"]
            return "short"
        return None


@dataclass
class ActiveTrade:
    """A trade being tracked through its lifecycle."""

    strategy_id: str
    strategy: PortfolioStrategy
    orb_label: str
    orb_minutes: int  # Aperture this trade was armed on (5, 15, or 30)
    entry_model: str
    direction: str
    state: TradeState

    # Confirm tracking
    confirm_count: int = 0
    confirm_needed: int = 1
    bars_since_break: list = field(default_factory=list)
    armed_at_bar: int = -1  # Bar index when ARMED — skip fill on same bar

    # Entry details (populated when ENTERED)
    entry_ts: datetime | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    contracts: int = 1

    # IB-conditional exit mode (TOKYO_OPEN session)
    exit_mode: str = "fixed_target"  # "fixed_target" | "ib_pending" | "hold_7h"
    ib_alignment: str | None = None  # "aligned" | "opposed" | None

    # Timed early exit
    early_exit_checked: bool = False

    # Calendar overlay sizing (1.0 = full, 0.5 = half)
    size_multiplier: float = 1.0

    # Outcome tracking
    exit_ts: datetime | None = None
    exit_price: float | None = None
    pnl_r: float | None = None
    mae_points: float = 0.0
    mfe_points: float = 0.0


class ExecutionEngine:
    """
    Bar-by-bar execution engine.

    Processes 1-minute bars, detects ORB formations, confirm signals,
    and manages trade lifecycle through entry to exit.

    Optionally accepts a MarketState for context-aware strategy scoring.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        cost_spec: CostSpec,
        risk_manager=None,
        market_state=None,
        live_session_costs: bool = True,
        atr_velocity_overlay=None,
        e2_order_timeout: dict[tuple[str, str], float] | None = None,
    ):
        self.portfolio = portfolio
        self.cost_spec = cost_spec
        self.risk_manager = risk_manager  # Optional RiskManager for position limits
        self.market_state = market_state  # Optional MarketState for scoring
        self._live_session_costs = live_session_costs  # Use session-adjusted slippage
        self.atr_velocity_overlay = atr_velocity_overlay  # Contracting ATR skip overlay
        # E2 order timeout: {(instrument, session) -> minutes}. After ORB completes,
        # E2 stop-market trigger is skipped if elapsed > timeout. Implements break-speed
        # overlay without a daily_features filter lookup (which is look-ahead for E2).
        # None = no timeout (backward compatible). See config.E2_ORDER_TIMEOUT.
        # @research-source memory/break_speed_signal_retest.md
        self._e2_order_timeout = e2_order_timeout or {}

        # State
        self.trading_day: date | None = None
        self.orbs: dict[tuple[str, int], LiveORB] = {}  # (session_label, orb_minutes) → LiveORB
        self.ib: LiveIB | None = None
        self.active_trades: list[ActiveTrade] = []
        self.completed_trades: list[ActiveTrade] = []
        self.daily_pnl_r: float = 0.0
        self.daily_trade_count: int = 0
        self._bar_count: int = 0
        self._last_bar: dict | None = None  # Track last bar for scratch mark-to-market
        # Per-orb_minutes daily_features rows. Keyed by orb_minutes (5, 15, 30).
        # Paper trader / session orchestrator load one row per unique aperture
        # in the portfolio, so _arm_strategies can pick the correct ORB columns.
        self._daily_features_rows: dict[int, dict] = {}

    def _compute_contracts(self, risk_points: float, cost: CostSpec, max_contracts: int = 1) -> int:
        """Compute position size using vol-adjusted sizing from portfolio params.

        Uses account_equity and risk_per_trade_pct from self.portfolio.
        Applies Turtle-style vol scalar from ATR_20 / median_atr_20 (Carver Ch.9).
        Clamps result to max_contracts (from strategy or firm limits).
        Returns 0 if risk exceeds budget (caller should reject entry).
        """
        equity = self.portfolio.account_equity
        risk_pct = self.portfolio.risk_per_trade_pct
        if equity <= 0 or risk_points <= 0:
            return 0  # Fail-closed: invalid equity/risk → reject entry

        # Vol sizing uses 5m row — ATR is instrument-level, not aperture-specific
        row = self._daily_features_rows.get(5) or {}
        atr_20 = row.get("atr_20") or 0.0
        median_atr_20 = row.get("median_atr_20") or 0.0
        if atr_20 > 0 and median_atr_20 > 0:
            vol_scalar = compute_vol_scalar(atr_20, median_atr_20)
        else:
            vol_scalar = 1.0
            logger.warning(
                "vol_scalar=1.0 fallback for %s — daily_features missing or ATR null (atr_20=%s, median=%s)",
                self.portfolio.instrument,
                atr_20,
                median_atr_20,
            )

        contracts = compute_position_size_vol_scaled(
            equity,
            risk_pct,
            risk_points,
            cost,
            vol_scalar,
        )
        if contracts > max_contracts:
            logger.warning(
                "CONTRACTS CLAMPED: computed=%d > max=%d for %s — firm/strategy limit enforced",
                contracts,
                max_contracts,
                self.portfolio.instrument,
            )
            contracts = max_contracts
        return contracts

    def _total_pnl_r(self) -> float:
        """Realized + unrealized PnL in R for risk checks.

        Unrealized = mark-to-market of all ENTERED positions using last bar close.
        This ensures the circuit breaker accounts for open position losses,
        not just closed trade losses.
        """
        unrealized = 0.0
        if self._last_bar is not None:
            current_price = self._last_bar.get("close", 0)
            for t in self.active_trades:
                if t.state == TradeState.ENTERED and t.entry_price and t.stop_price:
                    risk = abs(t.entry_price - t.stop_price)
                    if risk > 0:
                        if t.direction == "long":
                            unrealized += (current_price - t.entry_price) / risk
                        else:
                            unrealized += (t.entry_price - current_price) / risk
        return self.daily_pnl_r + unrealized

    def mark_strategy_traded(self, strategy_id: str) -> None:
        """Mark a strategy as already traded today (crash recovery).

        Seeds completed_trades so _arm_strategies() won't re-arm it.
        Called by SessionOrchestrator on startup to load journal state.
        """
        strategy = next(
            (s for s in self.portfolio.strategies if s.strategy_id == strategy_id),
            None,
        )
        if strategy is None:
            logger.warning("mark_strategy_traded: '%s' not in portfolio — skipping", strategy_id)
            return
        trade = ActiveTrade(
            strategy_id=strategy_id,
            strategy=strategy,
            orb_label=strategy.orb_label,
            orb_minutes=strategy.orb_minutes,
            entry_model=strategy.entry_model,
            direction="unknown",
            state=TradeState.EXITED,
        )
        self.completed_trades.append(trade)
        logger.info("RESTART DEDUP: strategy '%s' marked as already traded today", strategy_id)

    def cancel_trade(self, strategy_id: str) -> bool:
        """R2-C3: Remove a trade from active_trades after broker cancellation.

        Called by the fill poller when an E2 stop-market order is cancelled/rejected
        by the broker. Without this, the engine holds a ghost trade that emits
        EXIT events for a position that never existed at the broker.

        Moves the trade to completed_trades (as EXITED) to prevent re-arming.
        Returns True if the trade was found and cancelled.
        """
        for trade in self.active_trades:
            if trade.strategy_id == strategy_id:
                trade.state = TradeState.EXITED
                self.active_trades.remove(trade)
                self.completed_trades.append(trade)
                logger.warning(
                    "TRADE CANCELLED: %s removed from active_trades (broker cancelled/rejected)",
                    strategy_id,
                )
                return True
        logger.debug("cancel_trade: %s not found in active_trades", strategy_id)
        return False

    def on_trading_day_start(
        self,
        trading_day: date,
        daily_features_row: dict | None = None,
        daily_features_rows: dict[int, dict] | None = None,
    ) -> None:
        """Reset state for a new trading day.

        Args:
            daily_features_row: Legacy single-row interface (assumed orb_minutes=5).
            daily_features_rows: Per-orb_minutes rows keyed by int (5, 15, 30).
                If both provided, daily_features_rows wins.
        """
        self.trading_day = trading_day
        self.orbs = {}
        self.active_trades = []
        self.completed_trades = []
        self.daily_pnl_r = 0.0
        self.daily_trade_count = 0
        self._bar_count = 0
        self._last_bar = None
        if daily_features_rows is not None:
            self._daily_features_rows = daily_features_rows
        elif daily_features_row is not None:
            self._daily_features_rows = {5: daily_features_row}
        else:
            self._daily_features_rows = {}

        # Initialize IB tracker (23:00 UTC to 01:00 UTC = 09:00-11:00 Brisbane)
        # 23 = Brisbane UTC+10 offset: 09:00 local - 10h = 23:00 UTC previous day.
        # Brisbane has no DST (Australia/Brisbane is fixed UTC+10), so this is stable.
        prev_day = trading_day - timedelta(days=1)
        ib_start = datetime(prev_day.year, prev_day.month, prev_day.day, 23, 0, tzinfo=UTC)
        ib_end = ib_start + timedelta(minutes=IB_DURATION_MINUTES)
        self.ib = LiveIB(window_start_utc=ib_start, window_end_utc=ib_end)

        # Initialize ORB windows for this trading day.
        # Multi-aperture: create one LiveORB per unique (session_label, orb_minutes)
        # combination needed by loaded strategies. A session can have 5m, 15m, and 30m
        # ORBs simultaneously — they share the same start time but end at different times.
        #
        # Canonical source (E2 canonical-window refactor 2026-04-07, Stage 3):
        # pipeline.dst.orb_utc_window is the single source of truth for ORB window
        # timing across backtest (outcome_builder), live engine (this file), and
        # feature builder (build_daily_features). Any divergence between these
        # paths is a lookahead/contamination risk per Chan Ch 1 p4 (the backtest
        # must equal live execution). Previously this file had an inline copy of
        # the Brisbane->UTC resolver logic which could (and did) drift — see
        # docs/postmortems/2026-04-07-e2-canonical-window-fix.md.

        # Collect unique (label, orb_minutes) pairs from portfolio
        needed_orbs: set[tuple[str, int]] = set()
        for s in self.portfolio.strategies:
            if s.orb_label in DYNAMIC_ORB_RESOLVERS:
                needed_orbs.add((s.orb_label, s.orb_minutes))

        # Canonical delegation — no inline resolver logic. The tzinfo is
        # normalised from ZoneInfo('UTC') to datetime.UTC for consistency with
        # the rest of this module (bar ts_utc, trade timestamps, LiveIB window).
        for label, orb_minutes in needed_orbs:
            start, end = orb_utc_window(trading_day, label, orb_minutes)
            start = start.replace(tzinfo=UTC)
            end = end.replace(tzinfo=UTC)
            self.orbs[(label, orb_minutes)] = LiveORB(
                label=label,
                window_start_utc=start,
                window_end_utc=end,
                orb_minutes=orb_minutes,
            )

    def on_bar(self, bar: dict) -> list[TradeEvent]:
        """
        Process one 1-minute bar.

        bar must have: ts_utc, open, high, low, close, volume
        Returns list of TradeEvent objects.
        """
        events = []
        ts = bar["ts_utc"]
        self._bar_count += 1
        self._last_bar = bar

        # Update market state timestamp if available
        if self.market_state is not None:
            self.market_state.current_ts = ts

        # Phase 1: Update ORB ranges
        for _label, orb in self.orbs.items():
            if not orb.complete and orb.window_start_utc <= ts < orb.window_end_utc:
                if orb.high is None or bar["high"] > orb.high:
                    orb.high = bar["high"]
                if orb.low is None or bar["low"] < orb.low:
                    orb.low = bar["low"]

            # Mark ORB as complete when window ends
            if not orb.complete and ts >= orb.window_end_utc:
                orb.complete = True
                orb.complete_ts = ts

        # Phase 1.5: E2 honest entry — first bar touching ORB after window end.
        # E2 stop-market fills on range touch, not close confirmation.
        # Must fire BEFORE Phase 2 (close-based break) so E2 enters on fakeout
        # bars that precede the confirmed break. Per Chan, "Algorithmic Trading"
        # (Wiley 2013) Ch 1 p4 (verified verbatim from
        # resources/Algorithmic_Trading_Chan.pdf PDF p22): "If your backtesting
        # and live trading programs are one and the same, and the only difference
        # between backtesting versus live trading is what kind of data you are
        # feeding into the program (historical data in the former, and live
        # market data in the latter), then there can be no look-ahead bias in
        # the program." This Phase 1.5 path is the live-engine half of that
        # invariant; backtest parity is enforced in outcome_builder via
        # pipeline.dst.orb_utc_window (E2 canonical-window refactor 2026-04-07).
        # (Prior comment cited Pardo Ch.4 — NOT verified; local Pardo PDF is
        # front matter only. Updated to verified Chan citation 2026-04-07.)
        for (_label, _om), orb in self.orbs.items():
            if orb.complete and not orb.e2_touched and orb.high is not None:
                # E2 order timeout: skip trigger if too much time elapsed since
                # ORB completion. Implements break-speed overlay via execution
                # timing (not daily_features lookup — that's look-ahead for E2).
                # Keyed by (instrument, session). Absent key = no timeout.
                # @research-source memory/break_speed_signal_retest.md
                _timeout_min = self._e2_order_timeout.get((self.portfolio.instrument, _label))
                if (
                    _timeout_min is not None
                    and orb.complete_ts is not None
                    and (ts - orb.complete_ts).total_seconds() / 60 > _timeout_min
                ):
                    orb.e2_touched = True  # prevent re-checking every bar
                    continue
                e2_dir = None
                if bar["high"] > orb.high:
                    e2_dir = "long"
                elif bar["low"] < orb.low:
                    e2_dir = "short"
                if e2_dir is not None:
                    orb.e2_touched = True
                    events.extend(self._arm_strategies(orb, bar, direction=e2_dir, entry_models=frozenset({"E2"})))

        # Phase 2: Detect breaks for complete ORBs (close-based — arms E1/E3).
        # E2 is already handled by Phase 1.5 (first touch). If E2 was armed
        # on an earlier bar, the duplicate-trade check in _arm_strategies
        # (L665-668) prevents double-arming.
        for _label, orb in self.orbs.items():
            if orb.complete and orb.break_dir is None and orb.high is not None:
                if bar["close"] > orb.high:
                    orb.break_dir = "long"
                    orb.break_ts = ts
                    events.extend(self._arm_strategies(orb, bar))
                elif bar["close"] < orb.low:
                    orb.break_dir = "short"
                    orb.break_ts = ts
                    events.extend(self._arm_strategies(orb, bar))

        # Phase 2.5: Update IB and check IB-conditional exits
        if self.ib is not None:
            self.ib.update(bar)
            ib_break = self.ib.check_break(bar)
            if ib_break is not None:
                events.extend(self._process_ib_break(ib_break, bar))

        # Phase 3: Process confirming trades
        events.extend(self._process_confirming(bar))

        # Phase 4: Check entered trades for exit
        events.extend(self._check_exits(bar))

        return events

    def on_trading_day_end(self) -> list[TradeEvent]:
        """Close all open positions as scratch at end of trading day."""
        events = []
        if not self._last_bar:
            # No bars processed — nothing to close (can't have ENTERED without bars)
            self.active_trades = []
            return events
        last_ts = self._last_bar["ts_utc"]
        last_close = self._last_bar["close"]

        for trade in list(self.active_trades):
            if trade.state == TradeState.ENTERED:
                trade.state = TradeState.EXITED
                trade.exit_ts = last_ts
                trade.exit_price = last_close

                # Mark-to-market PnL for scratch (not 0.0)
                if trade.entry_price is not None and trade.stop_price is not None:
                    risk_points = abs(trade.entry_price - trade.stop_price)
                    if risk_points > 0:
                        if trade.direction == "long":
                            mtm_points = last_close - trade.entry_price
                        else:
                            mtm_points = trade.entry_price - last_close
                        cost = self.cost_spec
                        if self._live_session_costs:
                            cost = get_session_cost_spec(
                                self.portfolio.instrument,
                                trade.orb_label,
                            )
                        trade.pnl_r = round(
                            to_r_multiple(cost, trade.entry_price, trade.stop_price, mtm_points),
                            4,
                        )
                    else:
                        trade.pnl_r = 0.0
                else:
                    trade.pnl_r = 0.0

                self.daily_pnl_r += trade.pnl_r
                self.completed_trades.append(trade)
                if self.risk_manager is not None:
                    self.risk_manager.on_trade_exit(trade.pnl_r)

                events.append(
                    TradeEvent(
                        event_type="SCRATCH",
                        strategy_id=trade.strategy_id,
                        timestamp=last_ts,
                        price=last_close,
                        direction=trade.direction,
                        contracts=trade.contracts,
                        reason="session_end",
                        pnl_r=trade.pnl_r,
                    )
                )
            elif trade.state in (TradeState.ARMED, TradeState.CONFIRMING):
                logger.debug(
                    "session_end: %s expired in %s state (never entered) — discarded",
                    trade.strategy_id,
                    trade.state.value,
                )
                trade.state = TradeState.EXITED
                self.completed_trades.append(trade)

        self.active_trades = []
        return events

    def get_active_trades(self) -> list[ActiveTrade]:
        """Return all currently active trades."""
        return [t for t in self.active_trades if t.state in (TradeState.CONFIRMING, TradeState.ENTERED)]

    def get_daily_summary(self) -> dict:
        """Return daily PnL summary."""
        wins = [t for t in self.completed_trades if t.pnl_r is not None and t.pnl_r > 0]
        losses = [t for t in self.completed_trades if t.pnl_r is not None and t.pnl_r < 0]
        scratches = [t for t in self.completed_trades if t.pnl_r is not None and t.pnl_r == 0]
        entered = [t for t in self.completed_trades if t.entry_price is not None]

        return {
            "trading_day": self.trading_day,
            "bars_processed": self._bar_count,
            "trades_entered": len(entered),
            "wins": len(wins),
            "losses": len(losses),
            "scratches": len(scratches),
            "daily_pnl_r": round(self.daily_pnl_r, 4),
        }

    # -----------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------

    def _arm_strategies(
        self,
        orb: LiveORB,
        bar: dict,
        direction: str | None = None,
        entry_models: frozenset[str] | None = None,
    ) -> list[TradeEvent]:
        """Arm matching strategies for confirmation or immediate entry.

        Args:
            direction: Trade direction. Defaults to orb.break_dir (close-based).
                       Phase 1.5 passes the touch direction for E2.
            entry_models: If provided, only arm strategies with these entry models.
                          Phase 1.5 passes frozenset({"E2"}). Phase 2 passes None
                          (no filter); E2 dedup is handled by the active_trades /
                          completed_trades guard at the bottom of this method.
        """
        resolved_dir = direction or orb.break_dir
        assert resolved_dir is not None
        direction = resolved_dir  # narrowed to str
        events = []
        for strategy in self.portfolio.strategies:
            if strategy.orb_label != orb.label:
                continue
            if strategy.orb_minutes != orb.orb_minutes:
                continue
            if entry_models and strategy.entry_model not in entry_models:
                continue

            # E2 filter exclusion: skip break-bar-derived filters (look-ahead
            # for stop-market entries that fire before the break bar closes).
            if strategy.entry_model == "E2" and is_e2_lookahead_filter(strategy.filter_type):
                continue

            # Check strategy filter (size, DOW, break speed, etc.)
            filt = ALL_FILTERS.get(strategy.filter_type)
            if filt is None:
                # Unknown filter type — fail-closed: do not arm.
                # Every filter_type in validated_setups MUST be in ALL_FILTERS.
                logger.error(
                    "Unknown filter_type '%s' for %s — skipping (fail-closed)",
                    strategy.filter_type,
                    strategy.strategy_id,
                )
                continue
            # Build full row: daily_features (correct orb_minutes) + ORB runtime data
            om_row = self._daily_features_rows.get(strategy.orb_minutes, self._daily_features_rows.get(5))
            row = {}
            if om_row is not None:
                row.update(om_row)
            row[f"orb_{orb.label}_size"] = orb.size
            if not filt.matches_row(row, orb.label):
                continue

            # Check calendar overlay (per-instrument×session rules)
            assert self.trading_day is not None  # set by on_trading_day_start before any bar processing
            action = get_calendar_action(strategy.instrument, orb.label, self.trading_day)
            if action == CalendarAction.SKIP:
                logger.info(
                    "Calendar SKIP: %s on %s (%s)",
                    strategy.strategy_id,
                    self.trading_day,
                    orb.label,
                )
                continue
            # size_multiplier: 0.5 for HALF_SIZE, 1.0 for NEUTRAL
            size_multiplier = action.value

            # Check ATR velocity overlay (Contracting×Neutral/Compressed skip)
            if (
                self.atr_velocity_overlay is not None
                and om_row is not None
                and not self.atr_velocity_overlay.matches_row(om_row, orb.label)
            ):
                continue

            # Check market state scoring (if available)
            if self.market_state is not None:
                from trading_app.scoring import MIN_SCORE_THRESHOLD

                score = self.market_state.strategy_scores.get(strategy.strategy_id)
                if score is not None and score < MIN_SCORE_THRESHOLD:
                    continue

            # ML meta-label block removed 2026-04-11 (ML V3 sprint Stage 4).
            # V1/V2/V3 all DEAD — filters are not interchangeable with ML
            # features (V3 validated this structurally: orb_volume_norm was
            # the top MDA feature in 2/3 trials but CPCV AUC stayed at 0.50).
            # See docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md.

            # Check if already have a trade for this strategy today
            if any(t.strategy_id == strategy.strategy_id for t in self.active_trades):
                continue
            if any(t.strategy_id == strategy.strategy_id for t in self.completed_trades):
                continue

            # E2 confirms on range touch (stop-market); E1/E3 confirm on close.
            if strategy.entry_model == "E2":
                confirmed = (direction == "long" and bar["high"] > orb.high) or (
                    direction == "short" and bar["low"] < orb.low
                )
            else:
                confirmed = (direction == "long" and bar["close"] > orb.high) or (
                    direction == "short" and bar["close"] < orb.low
                )

            trade = ActiveTrade(
                strategy_id=strategy.strategy_id,
                strategy=strategy,
                orb_label=orb.label,
                orb_minutes=orb.orb_minutes,
                entry_model=strategy.entry_model,
                direction=direction,
                state=TradeState.CONFIRMING,
                confirm_needed=strategy.confirm_bars,
                confirm_count=1 if confirmed else 0,
                bars_since_break=[bar],
                size_multiplier=size_multiplier,
            )

            # Check if immediately confirmed (confirm_bars=1 and break bar qualifies)
            if trade.confirm_count >= trade.confirm_needed:
                entry_events = self._try_entry(trade, orb, bar)
                events.extend(entry_events)
            else:
                self.active_trades.append(trade)

        return events

    def _process_confirming(self, bar: dict) -> list[TradeEvent]:
        """Process confirm bar counting for CONFIRMING trades."""
        events = []
        still_active = []
        new_trades = []  # Collect newly entered/armed trades separately

        for trade in self.active_trades:
            if trade.state != TradeState.CONFIRMING:
                still_active.append(trade)
                continue

            trade.bars_since_break.append(bar)
            orb = self.orbs[(trade.orb_label, trade.orb_minutes)]

            # Check if this bar closes in the break direction
            if trade.direction == "long":
                outside = bar["close"] > orb.high
            else:
                outside = bar["close"] < orb.low

            if outside:
                trade.confirm_count += 1
            else:
                trade.confirm_count = 0  # Reset on failure

            if trade.confirm_count >= trade.confirm_needed:
                entry_events = self._try_entry(trade, orb, bar, new_trades)
                events.extend(entry_events)
            else:
                still_active.append(trade)

        self.active_trades = still_active + new_trades
        return events

    def _try_entry(
        self, trade: ActiveTrade, orb: LiveORB, confirm_bar: dict, new_trades: list | None = None
    ) -> list[TradeEvent]:
        """Attempt to enter a trade after confirmation.

        new_trades: list to append newly active trades to (avoids mutating
        self.active_trades during iteration).
        """
        events = []
        if new_trades is None:
            new_trades = self.active_trades  # Fallback for _arm_strategies calls

        orb_high = orb.high
        orb_low = orb.low
        assert orb_high is not None and orb_low is not None  # ORB is complete when _try_entry is called
        stop_price = orb_low if trade.direction == "long" else orb_high

        # Apply tight stop (Option B): move stop closer by (1 - multiplier) * ORB range.
        # Uses orb_range (not entry-stop distance) — backtester uses risk_pts which includes
        # E2 slippage. Delta is ~1 tick per trade, structurally negligible (<0.2% of risk).
        sm = getattr(trade.strategy, "stop_multiplier", 1.0)
        if sm != 1.0:
            orb_range = orb_high - orb_low
            if trade.direction == "long":
                stop_price = orb_low + (1.0 - sm) * orb_range
            else:
                stop_price = orb_high - (1.0 - sm) * orb_range

        # Resolve entry price based on model
        if trade.entry_model == "E2":
            # E2: Stop-Market — fill at ORB edge + slippage ON the confirm bar.
            # The stop order triggers when the bar's range crosses the ORB level.
            # For longs: bar high must exceed orb_high; fill = orb_high + slippage.
            # For shorts: bar low must go below orb_low; fill = orb_low - slippage.
            from pipeline.cost_model import get_cost_spec as _get_cost_spec
            from trading_app.config import E2_SLIPPAGE_TICKS

            tick_size = _get_cost_spec(trade.strategy.instrument).tick_size
            slippage = E2_SLIPPAGE_TICKS * tick_size

            if trade.direction == "long":
                if confirm_bar["high"] <= orb_high:
                    return events  # Bar didn't cross ORB level — no fill
                entry_price = orb_high + slippage
            else:
                if confirm_bar["low"] >= orb_low:
                    return events  # Bar didn't cross ORB level — no fill
                entry_price = orb_low - slippage

            risk_points = abs(entry_price - stop_price)
            if risk_points <= 0:
                trade.state = TradeState.EXITED
                self.completed_trades.append(trade)
                events.append(
                    TradeEvent(
                        event_type="REJECT",
                        strategy_id=trade.strategy_id,
                        timestamp=confirm_bar["ts_utc"],
                        price=entry_price,
                        direction=trade.direction,
                        contracts=0,
                        reason="rejected: zero_risk_points",
                    )
                )
                return events

            # Position sizing (vol-adjusted, Carver Ch.9)
            cost = (
                get_session_cost_spec(
                    self.portfolio.instrument,
                    trade.orb_label,
                )
                if self._live_session_costs
                else self.cost_spec
            )
            trade.contracts = self._compute_contracts(risk_points, cost, trade.strategy.max_contracts)
            if trade.contracts == 0:
                trade.state = TradeState.EXITED
                self.completed_trades.append(trade)
                events.append(
                    TradeEvent(
                        event_type="REJECT",
                        strategy_id=trade.strategy_id,
                        timestamp=confirm_bar["ts_utc"],
                        price=entry_price,
                        direction=trade.direction,
                        contracts=0,
                        reason="sizing_rejected: risk_exceeds_budget",
                    )
                )
                return events

            # Risk manager check
            suggested_contract_factor = 1.0
            if self.risk_manager is not None:
                can_enter, reason, suggested_contract_factor = self.risk_manager.can_enter(
                    strategy_id=trade.strategy_id,
                    orb_label=trade.orb_label,
                    active_trades=self.active_trades,
                    daily_pnl_r=self._total_pnl_r(),
                    orb_minutes=trade.orb_minutes,
                    instrument=trade.strategy.instrument,  # F-2 hedging guard
                    direction=trade.direction,  # F-2 hedging guard
                )
                if not can_enter:
                    trade.state = TradeState.EXITED
                    self.completed_trades.append(trade)
                    events.append(
                        TradeEvent(
                            event_type="REJECT",
                            strategy_id=trade.strategy_id,
                            timestamp=confirm_bar["ts_utc"],
                            price=entry_price,
                            direction=trade.direction,
                            contracts=trade.contracts,
                            reason=f"risk_rejected: {reason}",
                        )
                    )
                    return events

            trade.contracts = max(1, int(trade.contracts * suggested_contract_factor))
            # Apply calendar overlay sizing (HALF_SIZE=0.5, NEUTRAL=1.0).
            # NOTE: For single-contract strategies, max(1, ...) floor means
            # HALF_SIZE is a no-op — effective only for multi-contract sizing.
            if trade.size_multiplier != 1.0:
                trade.contracts = max(1, int(trade.contracts * trade.size_multiplier))

            if trade.direction == "long":
                target_price = entry_price + risk_points * trade.strategy.rr_target
            else:
                target_price = entry_price - risk_points * trade.strategy.rr_target

            trade.entry_price = entry_price
            trade.entry_ts = confirm_bar["ts_utc"]
            trade.stop_price = stop_price
            trade.target_price = target_price
            trade.state = TradeState.ENTERED

            # Session exit mode (IB conditional, hold, etc.)
            session_mode = SESSION_EXIT_MODE.get(trade.orb_label, "fixed_target")
            if session_mode == "ib_conditional":
                if self.ib is not None and self.ib.break_dir is not None:
                    if self.ib.break_dir == trade.direction:
                        trade.exit_mode = "hold_7h"
                        trade.ib_alignment = "aligned"
                        trade.target_price = None
                    else:
                        trade.ib_alignment = "opposed"
                        trade.state = TradeState.EXITED
                        self.completed_trades.append(trade)
                        events.append(
                            TradeEvent(
                                event_type="REJECT",
                                strategy_id=trade.strategy_id,
                                timestamp=confirm_bar["ts_utc"],
                                price=entry_price,
                                direction=trade.direction,
                                contracts=trade.contracts,
                                reason="ib_already_opposed",
                            )
                        )
                        return events
                else:
                    trade.exit_mode = "ib_pending"

            self.daily_trade_count += 1
            if self.risk_manager is not None:
                self.risk_manager.on_trade_entry()

            new_trades.append(trade)
            events.append(
                TradeEvent(
                    event_type="ENTRY",
                    strategy_id=trade.strategy_id,
                    timestamp=confirm_bar["ts_utc"],
                    price=entry_price,
                    direction=trade.direction,
                    contracts=trade.contracts,
                    reason="stop_market_E2",
                    risk_points=risk_points,
                )
            )
            return events

        elif trade.entry_model == "E1":
            # E1: will enter on NEXT bar's open
            trade.stop_price = stop_price
            trade.state = TradeState.ARMED
            trade.armed_at_bar = self._bar_count
            new_trades.append(trade)
            return events

        elif trade.entry_model == "E3":
            # E3: will enter on retrace
            trade.stop_price = stop_price
            trade.entry_price = orb.high if trade.direction == "long" else orb.low
            trade.state = TradeState.ARMED
            trade.armed_at_bar = self._bar_count
            new_trades.append(trade)
            return events

        else:
            logger.error(
                "Unknown entry_model '%s' for %s — rejecting entry (fail-closed)",
                trade.entry_model,
                trade.strategy_id,
            )
            trade.state = TradeState.EXITED
            self.completed_trades.append(trade)
            events.append(
                TradeEvent(
                    event_type="REJECT",
                    strategy_id=trade.strategy_id,
                    timestamp=confirm_bar["ts_utc"],
                    price=confirm_bar["close"],
                    direction=trade.direction,
                    contracts=0,
                    reason=f"unknown_entry_model: {trade.entry_model}",
                )
            )
            return events

    def _process_ib_break(self, ib_break_dir: str, bar: dict) -> list[TradeEvent]:
        """Process IB break for trades in ib_pending mode."""
        events = []
        for trade in self.active_trades:
            if trade.state != TradeState.ENTERED or trade.exit_mode != "ib_pending":
                continue

            if ib_break_dir == trade.direction:
                # Aligned: unlock hold mode, remove fixed target
                trade.exit_mode = "hold_7h"
                trade.ib_alignment = "aligned"
                trade.target_price = None
            else:
                # Opposed: exit at market (bar close)
                trade.ib_alignment = "opposed"
                self._exit_trade(trade, bar, "ib_opposed", bar["close"], events)

        return events

    def _check_exits(self, bar: dict) -> list[TradeEvent]:
        """Check entered trades for target/stop hits."""
        events = []

        for trade in self.active_trades:
            # Handle ARMED trades (E1 waiting for next bar open, E3 waiting for retrace)
            # Skip if ARMED on this same bar — E1/E3 fill on NEXT bar, not the confirm bar
            if trade.state == TradeState.ARMED and trade.armed_at_bar == self._bar_count:
                continue
            if trade.state == TradeState.ARMED:
                if trade.entry_model == "E1":
                    # E1: enter at this bar's open
                    entry_price = bar["open"]
                    entry_ts = bar["ts_utc"]
                    stop_price = trade.stop_price
                    risk_points = abs(entry_price - stop_price)
                    if risk_points <= 0:
                        trade.state = TradeState.EXITED
                        self.completed_trades.append(trade)
                        events.append(
                            TradeEvent(
                                event_type="REJECT",
                                strategy_id=trade.strategy_id,
                                timestamp=entry_ts,
                                price=entry_price,
                                direction=trade.direction,
                                contracts=0,
                                reason="rejected: zero_risk_points",
                            )
                        )
                        continue

                    # Position sizing (vol-adjusted, Carver Ch.9)
                    cost = (
                        get_session_cost_spec(
                            self.portfolio.instrument,
                            trade.orb_label,
                        )
                        if self._live_session_costs
                        else self.cost_spec
                    )
                    trade.contracts = self._compute_contracts(risk_points, cost, trade.strategy.max_contracts)
                    if trade.contracts == 0:
                        trade.state = TradeState.EXITED
                        self.completed_trades.append(trade)
                        events.append(
                            TradeEvent(
                                event_type="REJECT",
                                strategy_id=trade.strategy_id,
                                timestamp=entry_ts,
                                price=entry_price,
                                direction=trade.direction,
                                contracts=0,
                                reason="sizing_rejected: risk_exceeds_budget",
                            )
                        )
                        continue

                    # Risk manager check BEFORE entry
                    # R2-H6: use _total_pnl_r() (realized + unrealized) for consistency
                    # with E2 path. Using realized-only would let E1/E3 enter through
                    # a daily loss gate that E2 would correctly block.
                    suggested_contract_factor = 1.0
                    if self.risk_manager is not None:
                        can_enter, reason, suggested_contract_factor = self.risk_manager.can_enter(
                            strategy_id=trade.strategy_id,
                            orb_label=trade.orb_label,
                            active_trades=self.active_trades,
                            daily_pnl_r=self._total_pnl_r(),
                            orb_minutes=trade.orb_minutes,
                            instrument=trade.strategy.instrument,  # F-2 hedging guard
                            direction=trade.direction,  # F-2 hedging guard
                        )
                        if not can_enter:
                            trade.state = TradeState.EXITED
                            self.completed_trades.append(trade)
                            events.append(
                                TradeEvent(
                                    event_type="REJECT",
                                    strategy_id=trade.strategy_id,
                                    timestamp=entry_ts,
                                    price=entry_price,
                                    direction=trade.direction,
                                    contracts=trade.contracts,
                                    reason=f"risk_rejected: {reason}",
                                )
                            )
                            continue

                    # Apply suggested contract factor
                    trade.contracts = max(1, int(trade.contracts * suggested_contract_factor))
                    # Apply calendar overlay sizing (HALF_SIZE=0.5, NEUTRAL=1.0)
                    if trade.size_multiplier != 1.0:
                        trade.contracts = max(1, int(trade.contracts * trade.size_multiplier))

                    if trade.direction == "long":
                        target_price = entry_price + risk_points * trade.strategy.rr_target
                    else:
                        target_price = entry_price - risk_points * trade.strategy.rr_target

                    trade.entry_price = entry_price
                    trade.entry_ts = entry_ts
                    trade.target_price = target_price
                    trade.state = TradeState.ENTERED
                    # Set exit mode based on session
                    session_mode = SESSION_EXIT_MODE.get(trade.orb_label, "fixed_target")
                    if session_mode == "ib_conditional":
                        if self.ib is not None and self.ib.break_dir is not None:
                            # IB already resolved before entry
                            if self.ib.break_dir == trade.direction:
                                trade.exit_mode = "hold_7h"
                                trade.ib_alignment = "aligned"
                                trade.target_price = None
                            else:
                                # IB already opposed — reject entry
                                trade.ib_alignment = "opposed"
                                trade.state = TradeState.EXITED
                                self.completed_trades.append(trade)
                                events.append(
                                    TradeEvent(
                                        event_type="REJECT",
                                        strategy_id=trade.strategy_id,
                                        timestamp=entry_ts,
                                        price=entry_price,
                                        direction=trade.direction,
                                        contracts=trade.contracts,
                                        reason="ib_already_opposed",
                                    )
                                )
                                continue
                        else:
                            trade.exit_mode = "ib_pending"
                    self.daily_trade_count += 1
                    if self.risk_manager is not None:
                        self.risk_manager.on_trade_entry()

                    events.append(
                        TradeEvent(
                            event_type="ENTRY",
                            strategy_id=trade.strategy_id,
                            timestamp=entry_ts,
                            price=entry_price,
                            direction=trade.direction,
                            contracts=trade.contracts,
                            reason="confirm_bars_met_E1",
                            risk_points=risk_points,
                        )
                    )
                    # Fall through to exit check — fill bar may hit stop/target
                    # (matches outcome_builder _check_fill_bar_exit behavior)

                elif trade.entry_model == "E3":
                    # E3: check if bar retraces to ORB level
                    orb = self.orbs[(trade.orb_label, trade.orb_minutes)]
                    orb_high = orb.high
                    orb_low = orb.low
                    assert orb_high is not None and orb_low is not None  # ORB complete for E3 retrace
                    trade_stop = trade.stop_price
                    assert trade_stop is not None  # set at arm time
                    if trade.direction == "long":
                        retrace = bar["low"] <= orb_high
                        # CRITICAL: Check stop BEFORE fill — if bar also breaches stop,
                        # the trade is dead. Can't enter a stopped-out position.
                        stop_hit = bar["low"] <= trade_stop
                        entry_price = orb_high
                    else:
                        retrace = bar["high"] >= orb_low
                        stop_hit = bar["high"] >= trade_stop
                        entry_price = orb_low

                    if stop_hit:
                        # Stop breached on or before retrace — no valid fill
                        trade.state = TradeState.EXITED
                        self.completed_trades.append(trade)
                        continue

                    if retrace:
                        stop_price = trade_stop
                        risk_points = abs(entry_price - stop_price)
                        if risk_points <= 0:
                            # Defensive dead code: stop_hit fires first when stop_price >= entry_price
                            # (for longs: stop_hit = bar.low <= stop_price; retrace = bar.low <= orb.high;
                            # zero-risk requires stop_price >= orb.high, so stop_hit always wins).
                            # Guard retained for safety in case of corrupt ORB data.
                            trade.state = TradeState.EXITED
                            self.completed_trades.append(trade)
                            events.append(
                                TradeEvent(
                                    event_type="REJECT",
                                    strategy_id=trade.strategy_id,
                                    timestamp=bar["ts_utc"],
                                    price=entry_price,
                                    direction=trade.direction,
                                    contracts=0,
                                    reason="rejected: zero_risk_points",
                                )
                            )
                            continue

                        # Position sizing (vol-adjusted, Carver Ch.9)
                        cost = (
                            get_session_cost_spec(
                                self.portfolio.instrument,
                                trade.orb_label,
                            )
                            if self._live_session_costs
                            else self.cost_spec
                        )
                        trade.contracts = self._compute_contracts(risk_points, cost, trade.strategy.max_contracts)
                        if trade.contracts == 0:
                            trade.state = TradeState.EXITED
                            self.completed_trades.append(trade)
                            events.append(
                                TradeEvent(
                                    event_type="REJECT",
                                    strategy_id=trade.strategy_id,
                                    timestamp=bar["ts_utc"],
                                    price=entry_price,
                                    direction=trade.direction,
                                    contracts=0,
                                    reason="sizing_rejected: risk_exceeds_budget",
                                )
                            )
                            continue

                        # Risk manager check BEFORE entry
                        # R2-H6: use _total_pnl_r() for consistency with E2 path
                        suggested_contract_factor = 1.0
                        if self.risk_manager is not None:
                            can_enter, reason, suggested_contract_factor = self.risk_manager.can_enter(
                                strategy_id=trade.strategy_id,
                                orb_label=trade.orb_label,
                                active_trades=self.active_trades,
                                daily_pnl_r=self._total_pnl_r(),
                                orb_minutes=trade.orb_minutes,
                                instrument=trade.strategy.instrument,  # F-2 hedging guard
                                direction=trade.direction,  # F-2 hedging guard
                            )
                            if not can_enter:
                                trade.state = TradeState.EXITED
                                self.completed_trades.append(trade)
                                events.append(
                                    TradeEvent(
                                        event_type="REJECT",
                                        strategy_id=trade.strategy_id,
                                        timestamp=bar["ts_utc"],
                                        price=entry_price,
                                        direction=trade.direction,
                                        contracts=trade.contracts,
                                        reason=f"risk_rejected: {reason}",
                                    )
                                )
                                continue

                        # Apply suggested contract factor
                        trade.contracts = max(1, int(trade.contracts * suggested_contract_factor))
                        # Apply calendar overlay sizing (HALF_SIZE=0.5, NEUTRAL=1.0)
                        if trade.size_multiplier != 1.0:
                            trade.contracts = max(1, int(trade.contracts * trade.size_multiplier))

                        if trade.direction == "long":
                            target_price = entry_price + risk_points * trade.strategy.rr_target
                        else:
                            target_price = entry_price - risk_points * trade.strategy.rr_target

                        trade.entry_price = entry_price
                        trade.entry_ts = bar["ts_utc"]
                        trade.target_price = target_price
                        trade.state = TradeState.ENTERED
                        # Set exit mode based on session
                        session_mode = SESSION_EXIT_MODE.get(trade.orb_label, "fixed_target")
                        if session_mode == "ib_conditional":
                            if self.ib is not None and self.ib.break_dir is not None:
                                if self.ib.break_dir == trade.direction:
                                    trade.exit_mode = "hold_7h"
                                    trade.ib_alignment = "aligned"
                                    trade.target_price = None
                                else:
                                    # IB already opposed — reject entry
                                    trade.ib_alignment = "opposed"
                                    trade.state = TradeState.EXITED
                                    self.completed_trades.append(trade)
                                    events.append(
                                        TradeEvent(
                                            event_type="REJECT",
                                            strategy_id=trade.strategy_id,
                                            timestamp=bar["ts_utc"],
                                            price=entry_price,
                                            direction=trade.direction,
                                            contracts=trade.contracts,
                                            reason="ib_already_opposed",
                                        )
                                    )
                                    continue
                            else:
                                trade.exit_mode = "ib_pending"
                        self.daily_trade_count += 1
                        if self.risk_manager is not None:
                            self.risk_manager.on_trade_entry()

                        events.append(
                            TradeEvent(
                                event_type="ENTRY",
                                strategy_id=trade.strategy_id,
                                timestamp=bar["ts_utc"],
                                price=entry_price,
                                direction=trade.direction,
                                contracts=trade.contracts,
                                reason="retrace_fill_E3",
                                risk_points=risk_points,
                            )
                        )
                        # Fall through to exit check — fill bar may hit target
                        # (E3 stop on fill bar is impossible: entry_rules rejects
                        # fills when stop breached on retrace bar)

            if trade.state != TradeState.ENTERED:
                continue

            # Narrow mutable attributes for Pyright (guaranteed set in ENTERED state)
            entry_px = trade.entry_price
            stop_px = trade.stop_price
            assert entry_px is not None and stop_px is not None

            # Check stop (always active)
            if trade.direction == "long":
                hit_stop = bar["low"] <= stop_px
                favorable = bar["high"] - entry_px
                adverse = entry_px - bar["low"]
            else:
                hit_stop = bar["high"] >= stop_px
                favorable = entry_px - bar["low"]
                adverse = bar["high"] - entry_px

            # Check target (only if target_price is set — hold_7h has no target)
            hit_target = False
            if trade.target_price is not None:
                if trade.direction == "long":
                    hit_target = bar["high"] >= trade.target_price
                else:
                    hit_target = bar["low"] <= trade.target_price

            # Track MAE/MFE
            if favorable > trade.mfe_points:
                trade.mfe_points = favorable
            if adverse > trade.mae_points:
                trade.mae_points = adverse

            # Hold-7h time cutoff: exit at bar close after HOLD_HOURS
            if trade.exit_mode == "hold_7h" and trade.entry_ts is not None:
                elapsed_hours = (bar["ts_utc"] - trade.entry_ts).total_seconds() / 3600.0
                if elapsed_hours >= HOLD_HOURS:
                    self._exit_trade(trade, bar, "hold_timeout", bar["close"], events)
                    continue

            # Timed early exit: kill losers at N minutes after fill
            # Skip for hold_7h — aligned trades should run, not get killed early
            threshold = EARLY_EXIT_MINUTES.get(trade.orb_label)
            if (
                threshold
                and not trade.early_exit_checked
                and trade.exit_mode != "hold_7h"
                and trade.entry_ts is not None
            ):
                elapsed = (bar["ts_utc"] - trade.entry_ts).total_seconds() / 60.0
                if elapsed >= threshold:
                    trade.early_exit_checked = True
                    if trade.direction == "long":
                        mtm = bar["close"] - entry_px
                    else:
                        mtm = entry_px - bar["close"]
                    if mtm < 0:
                        self._exit_trade(trade, bar, "early_exit", bar["close"], events)
                        continue

            if hit_target and hit_stop:
                # Ambiguous bar — conservative loss
                self._exit_trade(trade, bar, "loss", stop_px, events)
            elif hit_target:
                target_px = trade.target_price
                assert target_px is not None
                self._exit_trade(trade, bar, "win", target_px, events)
            elif hit_stop:
                self._exit_trade(trade, bar, "loss", stop_px, events)

        # Prune exited trades to prevent unbounded list growth and iteration bugs
        self.active_trades = [t for t in self.active_trades if t.state != TradeState.EXITED]

        return events

    def _exit_trade(
        self, trade: ActiveTrade, bar: dict, outcome: str, exit_price: float, events: list[TradeEvent]
    ) -> None:
        """Exit a trade and compute PnL."""
        trade.exit_ts = bar["ts_utc"]
        trade.exit_price = exit_price
        trade.state = TradeState.EXITED

        entry_px = trade.entry_price
        stop_px = trade.stop_price
        assert entry_px is not None and stop_px is not None  # trade is entered before exit
        risk_points = abs(entry_px - stop_px)

        # Use session-adjusted costs for live P&L (backtest uses flat cost_spec)
        cost = self.cost_spec
        if self._live_session_costs:
            cost = get_session_cost_spec(
                self.portfolio.instrument,
                trade.orb_label,
            )

        if outcome == "win":
            pnl_points = risk_points * trade.strategy.rr_target
            trade.pnl_r = round(
                to_r_multiple(cost, entry_px, stop_px, pnl_points),
                4,
            )
        elif outcome in ("early_exit", "hold_timeout", "ib_opposed"):
            # Mark-to-market exit at bar close
            if trade.direction == "long":
                mtm_points = exit_price - entry_px
            else:
                mtm_points = entry_px - exit_price
            trade.pnl_r = round(
                to_r_multiple(cost, entry_px, stop_px, mtm_points),
                4,
            )
        else:
            # Precise loss R-multiple under session-adjusted costs
            loss_points = -risk_points
            trade.pnl_r = round(
                to_r_multiple(cost, entry_px, stop_px, loss_points),
                4,
            )

        self.daily_pnl_r += trade.pnl_r
        self.completed_trades.append(trade)
        if self.risk_manager is not None:
            self.risk_manager.on_trade_exit(trade.pnl_r)

        # Build reason string
        if outcome == "early_exit":
            reason = "early_exit_timed"
        elif outcome == "hold_timeout":
            reason = "hold_timeout_7h"
        elif outcome == "ib_opposed":
            reason = "ib_opposed_kill"
        elif outcome == "win":
            reason = "win_target_hit"
        else:
            reason = "loss_stop_hit"

        events.append(
            TradeEvent(
                event_type="EXIT",
                strategy_id=trade.strategy_id,
                timestamp=bar["ts_utc"],
                price=exit_price,
                direction=trade.direction,
                contracts=trade.contracts,
                reason=reason,
                pnl_r=trade.pnl_r,
            )
        )
