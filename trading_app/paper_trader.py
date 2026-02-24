"""
Historical replay and paper trading.

Feeds bars_1m through ExecutionEngine + RiskManager and produces a trade
journal. Two modes:
  1. Historical replay — compare to orb_outcomes for validation
  2. Walk-forward — train/test split for OOS robustness check

Usage:
    python trading_app/paper_trader.py --instrument MGC --start 2024-01-01 --end 2024-12-31
    python trading_app/paper_trader.py --instrument MGC --walk-forward --train-years 3 --test-years 1
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from pipeline.log import get_logger
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.stdout.reconfigure(line_buffering=True)

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec
from trading_app.portfolio import Portfolio, build_portfolio
from trading_app.execution_engine import ExecutionEngine
from trading_app.risk_manager import RiskManager, RiskLimits
from trading_app.config import ATR_VELOCITY_OVERLAY, CalendarSkipFilter

# Explicit mapping from execution engine exit reasons to journal outcomes.
# Fallback: split on "_" and take first token.
_EXIT_OUTCOME_MAP = {
    "win_target_hit": "win",
    "loss_stop_hit": "loss",
    "early_exit_timed": "early_exit",
    "hold_timeout_7h": "hold_timeout",
    "ib_opposed_kill": "ib_opposed",
    "session_end": "scratch",
}

# =========================================================================
# Data classes
# =========================================================================

@dataclass
class JournalEntry:
    """A single trade in the journal."""
    mode: str
    trading_day: date
    strategy_id: str
    entry_model: str
    direction: str
    entry_ts: datetime | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    contracts: int = 1
    exit_ts: datetime | None = None
    exit_price: float | None = None
    outcome: str | None = None
    pnl_r: float | None = None
    exit_mode: str | None = None
    ib_alignment: str | None = None
    risk_rejected: bool = False
    risk_reason: str = ""

@dataclass
class DaySummary:
    """Summary for one trading day."""
    trading_day: date
    bars_processed: int = 0
    trades_entered: int = 0
    wins: int = 0
    losses: int = 0
    scratches: int = 0
    daily_pnl_r: float = 0.0
    risk_rejections: int = 0

@dataclass
class ReplayResult:
    """Result from a historical replay run."""
    start_date: date
    end_date: date
    days_processed: int = 0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_scratches: int = 0
    total_pnl_r: float = 0.0
    total_risk_rejections: int = 0
    journal: list[JournalEntry] = field(default_factory=list)
    day_summaries: list[DaySummary] = field(default_factory=list)

# =========================================================================
# Trading day detection
# =========================================================================

def _get_trading_days(con, instrument: str, start_date: date, end_date: date) -> list[date]:
    """Get trading days from daily_features."""
    rows = con.execute(
        """SELECT DISTINCT trading_day FROM daily_features
           WHERE symbol = ? AND orb_minutes = 5
             AND trading_day >= ? AND trading_day <= ?
           ORDER BY trading_day""",
        [instrument, start_date, end_date],
    ).fetchall()
    return [r[0] for r in rows]


def _get_daily_features_row(con, instrument: str, trading_day: date) -> dict | None:
    """Fetch all daily_features columns for one day (orb_minutes=5).

    Returns a full dict so composite filters (DOW, break speed, break bar
    continues) get the columns they need without manual SELECT maintenance.
    """
    result = con.execute(
        """SELECT * FROM daily_features
           WHERE symbol = ? AND orb_minutes = 5 AND trading_day = ?
           LIMIT 1""",
        [instrument, trading_day],
    )
    row = result.fetchone()
    if row is None:
        return None
    columns = [desc[0] for desc in result.description]
    return dict(zip(columns, row))

def _get_bars_for_day(con, instrument: str, trading_day: date) -> list[dict]:
    """
    Fetch 1-minute bars for a trading day.

    Trading day in Brisbane starts at 23:00 UTC previous day.
    """
    prev_day = trading_day - timedelta(days=1)
    td_start = datetime(prev_day.year, prev_day.month, prev_day.day,
                        23, 0, tzinfo=timezone.utc)
    td_end = td_start + timedelta(hours=24)

    rows = con.execute(
        """SELECT ts_utc, open, high, low, close, volume
           FROM bars_1m
           WHERE symbol = ? AND ts_utc >= ? AND ts_utc < ?
           ORDER BY ts_utc""",
        [instrument, td_start, td_end],
    ).fetchall()

    return [
        {
            "ts_utc": r[0],
            "open": r[1],
            "high": r[2],
            "low": r[3],
            "close": r[4],
            "volume": r[5],
        }
        for r in rows
    ]

# =========================================================================
# Historical replay
# =========================================================================

def replay_historical(
    db_path: Path | None = None,
    portfolio: Portfolio | None = None,
    instrument: str = "MGC",
    start_date: date | None = None,
    end_date: date | None = None,
    risk_limits: RiskLimits | None = None,
    use_market_state: bool = False,
    live_session_costs: bool = False,
    max_correlation: float = 0.85,
    calendar_overlay: CalendarSkipFilter | None = None,
) -> ReplayResult:
    """
    Feed historical bars_1m through ExecutionEngine + RiskManager.

    When use_market_state=True, builds a MarketState per day for
    context-aware strategy scoring.

    Returns ReplayResult with journal entries and daily summaries.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH
    if risk_limits is None:
        risk_limits = RiskLimits()

    cost_spec = get_cost_spec(instrument)

    if portfolio is None:
        portfolio = build_portfolio(db_path=db_path, instrument=instrument,
                                    max_correlation=max_correlation,
                                    calendar_overlay=calendar_overlay)

    if not portfolio.strategies:
        return ReplayResult(
            start_date=start_date or date.min,
            end_date=end_date or date.max,
        )

    # Pre-build cascade table if using market state
    cascade_table = None
    if use_market_state:
        from trading_app.cascade_table import build_cascade_table
        cascade_table = build_cascade_table(db_path)

    risk_mgr = RiskManager(risk_limits, corr_lookup=portfolio.corr_lookup)

    # MarketState built per-day below; engine gets it on day start
    market_state = None
    engine = ExecutionEngine(portfolio, cost_spec, risk_manager=risk_mgr,
                             market_state=market_state,
                             live_session_costs=live_session_costs,
                             atr_velocity_overlay=ATR_VELOCITY_OVERLAY)

    with duckdb.connect(str(db_path), read_only=True) as con:
        if start_date is None or end_date is None:
            all_days = _get_trading_days(con, instrument, date(2000, 1, 1), date(2100, 1, 1))
            if not all_days:
                return ReplayResult(start_date=date.min, end_date=date.max)
            start_date = start_date or all_days[0]
            end_date = end_date or all_days[-1]

        trading_days = _get_trading_days(con, instrument, start_date, end_date)

        result = ReplayResult(start_date=start_date, end_date=end_date)

        for i, td in enumerate(trading_days):
            bars = _get_bars_for_day(con, instrument, td)
            if not bars:
                continue

            # Build market state for this day (if enabled)
            if use_market_state:
                from trading_app.market_state import MarketState
                market_state = MarketState.from_trading_day(
                    td, db_path, cascade_table=cascade_table,
                    visible_sessions=set(),  # nothing resolved yet
                    instrument=instrument,
                )
                market_state.score_strategies(portfolio.strategies)
                engine.market_state = market_state
            else:
                engine.market_state = None

            df_row = _get_daily_features_row(con, instrument, td)
            engine.on_trading_day_start(td, daily_features_row=df_row)
            risk_mgr.daily_reset(td)

            day_summary = DaySummary(trading_day=td)
            day_summary.bars_processed = len(bars)

            for bar in bars:
                events = engine.on_bar(bar)

                for event in events:
                    if event.event_type == "ENTRY":
                        # Engine already checked risk_manager — just record
                        entry = JournalEntry(
                            mode="replay",
                            trading_day=td,
                            strategy_id=event.strategy_id,
                            entry_model=_entry_model_from_strategy(event.strategy_id),
                            direction=event.direction,
                            entry_ts=event.timestamp,
                            entry_price=event.price,
                            contracts=event.contracts,
                        )
                        day_summary.trades_entered += 1
                        result.journal.append(entry)

                    elif event.event_type == "REJECT":
                        entry = JournalEntry(
                            mode="replay",
                            trading_day=td,
                            strategy_id=event.strategy_id,
                            entry_model=_entry_model_from_strategy(event.strategy_id),
                            direction=event.direction,
                            entry_ts=event.timestamp,
                            entry_price=event.price,
                            contracts=event.contracts,
                            risk_rejected=True,
                            risk_reason=event.reason,
                        )
                        day_summary.risk_rejections += 1
                        result.total_risk_rejections += 1
                        result.journal.append(entry)

                    elif event.event_type in ("EXIT", "SCRATCH"):
                        # Find matching journal entry and update
                        for je in reversed(result.journal):
                            if (je.strategy_id == event.strategy_id
                                    and je.trading_day == td
                                    and je.exit_ts is None
                                    and not je.risk_rejected):
                                je.exit_ts = event.timestamp
                                je.exit_price = event.price
                                je.outcome = _EXIT_OUTCOME_MAP.get(
                                    event.reason,
                                    event.reason.split("_")[0] if "_" in event.reason else event.reason
                                )
                                break

                        # Update PnL tracking from completed trade
                        trade = _find_completed_trade(engine, event.strategy_id)
                        if trade is None:
                            logger.warning(f"No completed trade for {event.event_type} event: "
                                           f"strategy={event.strategy_id}, day={td}")
                        if trade and trade.pnl_r is not None:
                            for je in reversed(result.journal):
                                if (je.strategy_id == event.strategy_id
                                        and je.trading_day == td
                                        and je.pnl_r is None
                                        and not je.risk_rejected):
                                    je.pnl_r = trade.pnl_r
                                    je.stop_price = trade.stop_price
                                    je.target_price = trade.target_price
                                    je.exit_mode = trade.exit_mode
                                    je.ib_alignment = trade.ib_alignment
                                    break

                            if trade.pnl_r > 0:
                                day_summary.wins += 1
                                result.total_wins += 1
                            elif trade.pnl_r < 0:
                                day_summary.losses += 1
                                result.total_losses += 1
                            else:
                                day_summary.scratches += 1
                                result.total_scratches += 1

                        # Reveal resolved ORB outcome in market state
                        if market_state is not None:
                            orb_label = _orb_from_strategy(event.strategy_id)
                            _reveal_outcome(market_state, orb_label, trade,
                                            cascade_table, portfolio)

            # End of day
            eod_events = engine.on_trading_day_end()
            for event in eod_events:
                if event.event_type == "SCRATCH":
                    day_summary.scratches += 1
                    result.total_scratches += 1
                    # Get actual mark-to-market PnL from engine's completed trade
                    trade = _find_completed_trade(engine, event.strategy_id)
                    if trade is None:
                        logger.warning(f"No completed trade for EOD SCRATCH: "
                                       f"strategy={event.strategy_id}, day={td}")
                    scratch_pnl = trade.pnl_r if (trade and trade.pnl_r is not None) else 0.0
                    for je in reversed(result.journal):
                        if (je.strategy_id == event.strategy_id
                                and je.trading_day == td
                                and je.exit_ts is None
                                and not je.risk_rejected):
                            je.outcome = "scratch"
                            je.exit_ts = event.timestamp
                            je.exit_price = event.price
                            je.pnl_r = scratch_pnl
                            break

            day_summary.daily_pnl_r = engine.daily_pnl_r
            result.day_summaries.append(day_summary)
            result.days_processed += 1
            result.total_pnl_r += engine.daily_pnl_r

            if (i + 1) % 50 == 0:
                logger.info(f"  Replayed {i + 1}/{len(trading_days)} days, "
                            f"{result.total_wins + result.total_losses + result.total_scratches} trades, "
                            f"PnL: {result.total_pnl_r:.2f}R")

        result.total_trades = result.total_wins + result.total_losses + result.total_scratches
        return result

# =========================================================================
# Helpers
# =========================================================================

def _orb_from_strategy(strategy_id: str) -> str:
    """Extract ORB label from strategy ID.

    Format: {INSTRUMENT}_{ORB_LABEL}_{ENTRY_MODEL}_RR{rr}_CB{cb}_{FILTER}
    ORB_LABEL can contain underscores (e.g., US_DATA_830, CME_REOPEN).
    Entry model is always E0, E1, or E3.
    """
    parts = strategy_id.split("_")
    # Find the entry model token (E0, E1, E3) — everything between
    # parts[0] (instrument) and that token is the ORB label.
    for i, p in enumerate(parts):
        if p in ("E0", "E1", "E3") and i > 1:
            return "_".join(parts[1:i])
    # Fallback: assume position 1 (legacy format)
    return parts[1] if len(parts) > 1 else ""

def _entry_model_from_strategy(strategy_id: str) -> str:
    """Extract entry model from strategy ID.

    Entry model is always E0, E1, or E3.
    """
    parts = strategy_id.split("_")
    for p in parts:
        if p in ("E0", "E1", "E3"):
            return p
    # Fallback: assume position 2 (legacy format)
    return parts[2] if len(parts) > 2 else ""

def _find_completed_trade(engine: ExecutionEngine, strategy_id: str):
    """Find the most recent completed trade for a strategy."""
    for trade in reversed(engine.completed_trades):
        if trade.strategy_id == strategy_id:
            return trade
    return None

def _reveal_outcome(market_state, orb_label: str, trade, cascade_table, portfolio) -> None:
    """Reveal a resolved ORB outcome in the market state and re-score.

    Called after a trade exits (TP/SL/EOD) so later sessions can see
    the outcome without lookahead.
    """
    orb = market_state.orbs.get(orb_label)
    if orb is None:
        return
    # Determine outcome from trade result
    if trade and trade.pnl_r is not None:
        if trade.pnl_r > 0:
            orb.outcome = "win"
        else:
            orb.outcome = "loss"
    # Re-derive cross-session signals and re-score
    market_state.update_signals(cascade_table)
    market_state.score_strategies(portfolio.strategies)

# =========================================================================
# CLI
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Historical replay and paper trading"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date")
    parser.add_argument("--end", type=date.fromisoformat, help="End date")
    parser.add_argument("--max-strategies", type=int, default=20, help="Max portfolio strategies")
    parser.add_argument("--max-daily-loss", type=float, default=-5.0, help="Max daily loss (R)")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent positions")
    parser.add_argument("--max-correlation", type=float, default=0.85, help="Max pairwise correlation (0-1)")
    parser.add_argument("--live-session-costs", action="store_true",
                        help="Use session-adjusted slippage (0900=1.3x, 2300=0.8x)")
    parser.add_argument("--calendar-filter", choices=["NFP", "OPEX", "NONE"],
                        default="NONE", help="Calendar overlay filter")
    args = parser.parse_args()

    # Convert CLI calendar filter to CalendarSkipFilter object
    calendar_overlay = None
    if args.calendar_filter == "NFP":
        calendar_overlay = CalendarSkipFilter(skip_nfp=True, skip_opex=False)
    elif args.calendar_filter == "OPEX":
        calendar_overlay = CalendarSkipFilter(skip_nfp=False, skip_opex=True)
    # "NONE" keeps calendar_overlay=None

    risk_limits = RiskLimits(
        max_daily_loss_r=args.max_daily_loss,
        max_concurrent_positions=args.max_concurrent,
    )

    result = replay_historical(
        instrument=args.instrument,
        start_date=args.start,
        end_date=args.end,
        risk_limits=risk_limits,
        live_session_costs=args.live_session_costs,
        max_correlation=args.max_correlation,
        calendar_overlay=calendar_overlay,
    )

    logger.info(f"\nReplay complete: {result.start_date} to {result.end_date}")
    logger.info(f"  Days: {result.days_processed}")
    logger.info(f"  Trades: {result.total_trades} (W:{result.total_wins} L:{result.total_losses} S:{result.total_scratches})")
    logger.info(f"  Total PnL: {result.total_pnl_r:.2f}R")
    logger.info(f"  Risk rejections: {result.total_risk_rejections}")

    if result.total_wins + result.total_losses > 0:
        wr = result.total_wins / (result.total_wins + result.total_losses)
        logger.info(f"  Win rate: {wr:.1%}")

if __name__ == "__main__":
    main()
