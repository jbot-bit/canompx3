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

from pathlib import Path
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from pipeline.cost_model import CostSpec, to_r_multiple, get_session_cost_spec
from pipeline.dst import DYNAMIC_ORB_RESOLVERS
from trading_app.config import (
    ALL_FILTERS, EARLY_EXIT_MINUTES, SESSION_EXIT_MODE,
    IB_DURATION_MINUTES, HOLD_HOURS, ORB_DURATION_MINUTES,
    CalendarSkipFilter, CALENDAR_SKIP_NFP_OPEX,
)
from trading_app.portfolio import Portfolio, PortfolioStrategy

# =========================================================================
# ORB time windows (UTC offsets from Brisbane 09:00 trading day start)
# Brisbane = UTC+10, trading day starts 23:00 UTC prev day
# =========================================================================

ORB_WINDOWS_UTC = {
    # label: (start_hour_utc, start_minute)
    # Duration is session-specific — looked up from config.ORB_DURATION_MINUTES
    "0900": (23, 0),      # 09:00 Brisbane = 23:00 UTC  (5m ORB)
    "1000": (0, 0),       # 10:00 Brisbane = 00:00 UTC  (15m ORB — variable aperture)
    # 1100 OFF for breakout (shelved — 74% double-break, may revisit with fade/reversal)
    "1130": (1, 30),      # 11:30 Brisbane = 01:30 UTC  (5m ORB — HK/SG equity open)
    "1800": (8, 0),       # 18:00 Brisbane = 08:00 UTC  (5m ORB)
    "2300": (13, 0),      # 23:00 Brisbane = 13:00 UTC  (5m ORB)
    "0030": (14, 30),     # 00:30 Brisbane = 14:30 UTC  (5m ORB)
}

class TradeState(Enum):
    ARMED = "ARMED"
    CONFIRMING = "CONFIRMING"
    ENTERED = "ENTERED"
    EXITED = "EXITED"

@dataclass
class TradeEvent:
    """Abstract trade event (broker-agnostic)."""
    event_type: str         # "ENTRY", "EXIT", "SCRATCH", "REJECT"
    strategy_id: str
    timestamp: datetime
    price: float
    direction: str
    contracts: int
    reason: str

@dataclass
class LiveORB:
    """ORB range being built or completed."""
    label: str
    window_start_utc: datetime
    window_end_utc: datetime
    high: float | None = None
    low: float | None = None
    complete: bool = False
    break_dir: str | None = None
    break_ts: datetime | None = None

    @property
    def size(self) -> float | None:
        if self.high is not None and self.low is not None:
            return self.high - self.low
        return None

@dataclass
class LiveIB:
    """Initial Balance (first 120 minutes from 09:00 Brisbane = 23:00 UTC).

    Tracks IB high/low during formation, then detects first break after IB ends.
    Used by 1000 session for IB-conditional exit logic.
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

    # IB-conditional exit mode (1000 session)
    exit_mode: str = "fixed_target"   # "fixed_target" | "ib_pending" | "hold_7h"
    ib_alignment: str | None = None   # "aligned" | "opposed" | None

    # Timed early exit
    early_exit_checked: bool = False

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

    def __init__(self, portfolio: Portfolio, cost_spec: CostSpec,
                 risk_manager=None, market_state=None,
                 live_session_costs: bool = True,
                 calendar_overlay: CalendarSkipFilter | None = CALENDAR_SKIP_NFP_OPEX):
        self.portfolio = portfolio
        self.cost_spec = cost_spec
        self.risk_manager = risk_manager  # Optional RiskManager for position limits
        self.market_state = market_state  # Optional MarketState for scoring
        self._live_session_costs = live_session_costs  # Use session-adjusted slippage
        self.calendar_overlay = calendar_overlay  # NFP/OPEX skip overlay

        # State
        self.trading_day: date | None = None
        self.orbs: dict[str, LiveORB] = {}
        self.ib: LiveIB | None = None
        self.active_trades: list[ActiveTrade] = []
        self.completed_trades: list[ActiveTrade] = []
        self.daily_pnl_r: float = 0.0
        self.daily_trade_count: int = 0
        self._bar_count: int = 0
        self._last_bar: dict | None = None  # Track last bar for scratch mark-to-market
        self._daily_features_row: dict | None = None  # Calendar filter data

    def on_trading_day_start(self, trading_day: date,
                             daily_features_row: dict | None = None) -> None:
        """Reset state for a new trading day."""
        self.trading_day = trading_day
        self.orbs = {}
        self.active_trades = []
        self.completed_trades = []
        self.daily_pnl_r = 0.0
        self.daily_trade_count = 0
        self._bar_count = 0
        self._last_bar = None
        self._daily_features_row = daily_features_row

        # Initialize IB tracker (23:00 UTC to 01:00 UTC = 09:00-11:00 Brisbane)
        prev_day = trading_day - timedelta(days=1)
        ib_start = datetime(prev_day.year, prev_day.month, prev_day.day,
                            23, 0, tzinfo=timezone.utc)
        ib_end = ib_start + timedelta(minutes=IB_DURATION_MINUTES)
        self.ib = LiveIB(window_start_utc=ib_start, window_end_utc=ib_end)

        # Initialize ORB windows for this trading day
        # Trading day in UTC: starts at 23:00 UTC on prev calendar day
        # Duration is session-specific (variable aperture from config)

        # Fixed sessions: static UTC offsets from Brisbane time
        for label, (hour, minute) in ORB_WINDOWS_UTC.items():
            duration = ORB_DURATION_MINUTES.get(label, 5)
            if hour >= 23:
                base_date = prev_day
            else:
                base_date = trading_day
            start = datetime(base_date.year, base_date.month, base_date.day,
                             hour, minute, tzinfo=timezone.utc)
            end = start + timedelta(minutes=duration)
            self.orbs[label] = LiveORB(
                label=label,
                window_start_utc=start,
                window_end_utc=end,
            )

        # Dynamic sessions: DST-aware, resolved per-day
        from zoneinfo import ZoneInfo
        _brisbane = ZoneInfo("Australia/Brisbane")
        _utc = ZoneInfo("UTC")
        for label, resolver in DYNAMIC_ORB_RESOLVERS.items():
            # Only create ORB if we have strategies for this session
            if not any(s.orb_label == label for s in self.portfolio.strategies):
                continue
            bris_h, bris_m = resolver(trading_day)
            duration = ORB_DURATION_MINUTES.get(label, 5)
            # Determine Brisbane calendar date (before 09:00 = next day)
            if bris_h < 9:
                cal_date = trading_day + timedelta(days=1)
            else:
                cal_date = trading_day
            local_start = datetime(cal_date.year, cal_date.month, cal_date.day,
                                   bris_h, bris_m, 0, tzinfo=_brisbane)
            start = local_start.astimezone(_utc).replace(tzinfo=timezone.utc)
            end = start + timedelta(minutes=duration)
            self.orbs[label] = LiveORB(
                label=label,
                window_start_utc=start,
                window_end_utc=end,
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
        for label, orb in self.orbs.items():
            if not orb.complete and orb.window_start_utc <= ts < orb.window_end_utc:
                if orb.high is None or bar["high"] > orb.high:
                    orb.high = bar["high"]
                if orb.low is None or bar["low"] < orb.low:
                    orb.low = bar["low"]

            # Mark ORB as complete when window ends
            if not orb.complete and ts >= orb.window_end_utc:
                orb.complete = True

        # Phase 2: Detect breaks for complete ORBs
        for label, orb in self.orbs.items():
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
        last_ts = self._last_bar["ts_utc"] if self._last_bar else datetime.now(timezone.utc)
        last_close = self._last_bar["close"] if self._last_bar else 0.0

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
                                self.portfolio.instrument, trade.orb_label,
                            )
                        trade.pnl_r = round(
                            to_r_multiple(cost, trade.entry_price,
                                          trade.stop_price, mtm_points),
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

                events.append(TradeEvent(
                    event_type="SCRATCH",
                    strategy_id=trade.strategy_id,
                    timestamp=last_ts,
                    price=last_close,
                    direction=trade.direction,
                    contracts=trade.contracts,
                    reason="session_end",
                ))
            elif trade.state in (TradeState.ARMED, TradeState.CONFIRMING):
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

    def _arm_strategies(self, orb: LiveORB, bar: dict) -> list[TradeEvent]:
        """When an ORB breaks, arm matching strategies for confirmation."""
        events = []
        for strategy in self.portfolio.strategies:
            if strategy.orb_label != orb.label:
                continue

            # Check ORB size filter
            filt = ALL_FILTERS.get(strategy.filter_type)
            if filt is not None:
                row = {f"orb_{orb.label}_size": orb.size}
                if not filt.matches_row(row, orb.label):
                    continue

            # Check calendar overlay (NFP/OPEX skip)
            if (self.calendar_overlay is not None
                    and self._daily_features_row is not None
                    and not self.calendar_overlay.matches_row(
                        self._daily_features_row, orb.label)):
                continue

            # Check market state scoring (if available)
            if self.market_state is not None:
                from trading_app.scoring import MIN_SCORE_THRESHOLD
                score = self.market_state.strategy_scores.get(
                    strategy.strategy_id
                )
                if score is not None and score < MIN_SCORE_THRESHOLD:
                    continue

            # Check if already have a trade for this strategy today
            if any(t.strategy_id == strategy.strategy_id for t in self.active_trades):
                continue
            if any(t.strategy_id == strategy.strategy_id for t in self.completed_trades):
                continue

            trade = ActiveTrade(
                strategy_id=strategy.strategy_id,
                strategy=strategy,
                orb_label=orb.label,
                entry_model=strategy.entry_model,
                direction=orb.break_dir,
                state=TradeState.CONFIRMING,
                confirm_needed=strategy.confirm_bars,
                confirm_count=1 if bar["close"] > orb.high or bar["close"] < orb.low else 0,
                bars_since_break=[bar],
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
            orb = self.orbs[trade.orb_label]

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

    def _try_entry(self, trade: ActiveTrade, orb: LiveORB, confirm_bar: dict,
                   new_trades: list | None = None) -> list[TradeEvent]:
        """Attempt to enter a trade after confirmation.

        new_trades: list to append newly active trades to (avoids mutating
        self.active_trades during iteration).
        """
        events = []
        if new_trades is None:
            new_trades = self.active_trades  # Fallback for _arm_strategies calls

        stop_price = orb.low if trade.direction == "long" else orb.high

        # Resolve entry price based on model
        if trade.entry_model == "E1":
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
                        continue

                    # Risk manager check BEFORE entry
                    suggested_contract_factor = 1.0 # Default
                    if self.risk_manager is not None:
                        can_enter, reason, suggested_contract_factor = self.risk_manager.can_enter(
                            strategy_id=trade.strategy_id,
                            orb_label=trade.orb_label,
                            active_trades=self.active_trades,
                            daily_pnl_r=self.daily_pnl_r,
                        )
                        if not can_enter:
                            trade.state = TradeState.EXITED
                            self.completed_trades.append(trade)
                            events.append(TradeEvent(
                                event_type="REJECT",
                                strategy_id=trade.strategy_id,
                                timestamp=entry_ts,
                                price=entry_price,
                                direction=trade.direction,
                                contracts=trade.contracts,
                                reason=f"risk_rejected: {reason}",
                            ))
                            continue
                    
                    # Apply suggested contract factor
                    trade.contracts = max(1, int(trade.contracts * suggested_contract_factor))

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
                                events.append(TradeEvent(
                                    event_type="REJECT",
                                    strategy_id=trade.strategy_id,
                                    timestamp=entry_ts,
                                    price=entry_price,
                                    direction=trade.direction,
                                    contracts=trade.contracts,
                                    reason="ib_already_opposed",
                                ))
                                continue
                        else:
                            trade.exit_mode = "ib_pending"
                    self.daily_trade_count += 1
                    if self.risk_manager is not None:
                        self.risk_manager.on_trade_entry()

                    events.append(TradeEvent(
                        event_type="ENTRY",
                        strategy_id=trade.strategy_id,
                        timestamp=entry_ts,
                        price=entry_price,
                        direction=trade.direction,
                        contracts=trade.contracts,
                        reason="confirm_bars_met_E1",
                    ))
                    # Fall through to exit check — fill bar may hit stop/target
                    # (matches outcome_builder _check_fill_bar_exit behavior)

                elif trade.entry_model == "E3":
                    # E3: check if bar retraces to ORB level
                    orb = self.orbs[trade.orb_label]
                    if trade.direction == "long":
                        retrace = bar["low"] <= orb.high
                        # CRITICAL: Check stop BEFORE fill — if bar also breaches stop,
                        # the trade is dead. Can't enter a stopped-out position.
                        stop_hit = bar["low"] <= trade.stop_price
                        entry_price = orb.high
                    else:
                        retrace = bar["high"] >= orb.low
                        stop_hit = bar["high"] >= trade.stop_price
                        entry_price = orb.low

                    if stop_hit:
                        # Stop breached on or before retrace — no valid fill
                        trade.state = TradeState.EXITED
                        self.completed_trades.append(trade)
                        continue

                    if retrace:
                        stop_price = trade.stop_price
                        risk_points = abs(entry_price - stop_price)
                        if risk_points <= 0:
                            trade.state = TradeState.EXITED
                            self.completed_trades.append(trade)
                            continue

                        # Risk manager check BEFORE entry
                        suggested_contract_factor = 1.0 # Default
                        if self.risk_manager is not None:
                            can_enter, reason, suggested_contract_factor = self.risk_manager.can_enter(
                                strategy_id=trade.strategy_id,
                                orb_label=trade.orb_label,
                                active_trades=self.active_trades,
                                daily_pnl_r=self.daily_pnl_r,
                            )
                            if not can_enter:
                                trade.state = TradeState.EXITED
                                self.completed_trades.append(trade)
                                events.append(TradeEvent(
                                    event_type="REJECT",
                                    strategy_id=trade.strategy_id,
                                    timestamp=bar["ts_utc"],
                                    price=entry_price,
                                    direction=trade.direction,
                                    contracts=trade.contracts,
                                    reason=f"risk_rejected: {reason}",
                                ))
                                continue
                        
                        # Apply suggested contract factor
                        trade.contracts = max(1, int(trade.contracts * suggested_contract_factor))

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
                                    events.append(TradeEvent(
                                        event_type="REJECT",
                                        strategy_id=trade.strategy_id,
                                        timestamp=bar["ts_utc"],
                                        price=entry_price,
                                        direction=trade.direction,
                                        contracts=trade.contracts,
                                        reason="ib_already_opposed",
                                    ))
                                    continue
                            else:
                                trade.exit_mode = "ib_pending"
                        self.daily_trade_count += 1
                        if self.risk_manager is not None:
                            self.risk_manager.on_trade_entry()

                        events.append(TradeEvent(
                            event_type="ENTRY",
                            strategy_id=trade.strategy_id,
                            timestamp=bar["ts_utc"],
                            price=entry_price,
                            direction=trade.direction,
                            contracts=trade.contracts,
                            reason="retrace_fill_E3",
                        ))
                        # Fall through to exit check — fill bar may hit target
                        # (E3 stop on fill bar is impossible: entry_rules rejects
                        # fills when stop breached on retrace bar)

            if trade.state != TradeState.ENTERED:
                continue

            # Check stop (always active)
            if trade.direction == "long":
                hit_stop = bar["low"] <= trade.stop_price
                favorable = bar["high"] - trade.entry_price
                adverse = trade.entry_price - bar["low"]
            else:
                hit_stop = bar["high"] >= trade.stop_price
                favorable = trade.entry_price - bar["low"]
                adverse = bar["high"] - trade.entry_price

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
            if threshold and not trade.early_exit_checked and trade.exit_mode != "hold_7h":
                elapsed = (bar["ts_utc"] - trade.entry_ts).total_seconds() / 60.0
                if elapsed >= threshold:
                    trade.early_exit_checked = True
                    if trade.direction == "long":
                        mtm = bar["close"] - trade.entry_price
                    else:
                        mtm = trade.entry_price - bar["close"]
                    if mtm < 0:
                        self._exit_trade(trade, bar, "early_exit", bar["close"], events)
                        continue

            if hit_target and hit_stop:
                # Ambiguous bar — conservative loss
                self._exit_trade(trade, bar, "loss", trade.stop_price, events)
            elif hit_target:
                self._exit_trade(trade, bar, "win", trade.target_price, events)
            elif hit_stop:
                self._exit_trade(trade, bar, "loss", trade.stop_price, events)

        return events

    def _exit_trade(self, trade: ActiveTrade, bar: dict, outcome: str,
                    exit_price: float, events: list[TradeEvent]) -> None:
        """Exit a trade and compute PnL."""
        trade.exit_ts = bar["ts_utc"]
        trade.exit_price = exit_price
        trade.state = TradeState.EXITED

        risk_points = abs(trade.entry_price - trade.stop_price)

        # Use session-adjusted costs for live P&L (backtest uses flat cost_spec)
        cost = self.cost_spec
        if self._live_session_costs:
            cost = get_session_cost_spec(
                self.portfolio.instrument, trade.orb_label,
            )

        if outcome == "win":
            pnl_points = risk_points * trade.strategy.rr_target
            trade.pnl_r = round(
                to_r_multiple(cost, trade.entry_price, trade.stop_price, pnl_points),
                4,
            )
        elif outcome in ("early_exit", "hold_timeout", "ib_opposed"):
            # Mark-to-market exit at bar close
            if trade.direction == "long":
                mtm_points = exit_price - trade.entry_price
            else:
                mtm_points = trade.entry_price - exit_price
            trade.pnl_r = round(
                to_r_multiple(cost, trade.entry_price, trade.stop_price, mtm_points),
                4,
            )
        else:
            # Precise loss R-multiple under session-adjusted costs
            loss_points = -risk_points
            trade.pnl_r = round(
                to_r_multiple(cost, trade.entry_price, trade.stop_price, loss_points),
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

        events.append(TradeEvent(
            event_type="EXIT",
            strategy_id=trade.strategy_id,
            timestamp=bar["ts_utc"],
            price=exit_price,
            direction=trade.direction,
            contracts=trade.contracts,
            reason=reason,
        ))
