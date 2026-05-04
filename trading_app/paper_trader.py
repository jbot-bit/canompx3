"""
Historical replay and paper trading.

Feeds bars_1m through ExecutionEngine + RiskManager and produces a trade
journal. Two modes:
  1. Historical replay — compare to orb_outcomes for validation
  2. Walk-forward — train/test split for OOS robustness check

Usage:
    python -m trading_app.paper_trader --instrument MGC --start 2024-01-01 --end 2024-12-31
    python -m trading_app.paper_trader --instrument MGC --walk-forward --train-years 3 --test-years 1
"""

import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from pipeline.log import get_logger

logger = get_logger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

import duckdb

from pipeline.cost_model import get_cost_spec
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ATR_VELOCITY_OVERLAY, ENTRY_MODELS
from trading_app.execution_engine import ExecutionEngine
from trading_app.portfolio import Portfolio, build_multi_rr_portfolio, build_portfolio, build_raw_baseline_portfolio
from trading_app.risk_manager import RiskLimits, RiskManager

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

# Outcomes classified as losses in reporting. Single source of truth —
# used by _print_strategy_summary and _print_session_summary.
_LOSS_OUTCOMES = frozenset({"loss", "early_exit", "hold_timeout", "ib_opposed"})

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
    overlay_context: dict[str, dict] = field(default_factory=dict)
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
    ml_skips: int = 0


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
    total_ml_skips: int = 0
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


def _get_median_atr_20(con, instrument: str, trading_day: date, lookback_days: int = 252) -> float:
    """Rolling median ATR_20 over prior N trading days (Carver Ch.9 vol-targeting).

    Used by ExecutionEngine._compute_contracts() for Turtle-style vol scalar:
    scalar = median_atr / current_atr → sizes position inversely to vol.
    """
    result = con.execute(
        """SELECT MEDIAN(atr_20) FROM daily_features
           WHERE symbol = ? AND orb_minutes = 5 AND atr_20 IS NOT NULL
             AND trading_day < ? AND trading_day >= ? - INTERVAL '504 DAY'
        """,
        [instrument, trading_day, trading_day],
    ).fetchone()
    return float(result[0]) if result[0] is not None else 0.0


def _get_daily_features_row(con, instrument: str, trading_day: date, orb_minutes: int = 5) -> dict | None:
    """Fetch all daily_features columns for one day at a specific orb_minutes.

    Returns a full dict so composite filters (DOW, break speed, break bar
    continues) get the columns they need without manual SELECT maintenance.
    """
    result = con.execute(
        """SELECT * FROM daily_features
           WHERE symbol = ? AND orb_minutes = ? AND trading_day = ?
           LIMIT 1""",
        [instrument, orb_minutes, trading_day],
    )
    row = result.fetchone()
    if row is None:
        return None
    columns = [desc[0] for desc in result.description]
    return dict(zip(columns, row, strict=False))


def _inject_cross_asset_atrs_for_replay(con, row: dict, instrument: str, trading_day: date) -> None:
    """Inject cross-asset ATR percentiles into daily features row for replay.

    Mirrors session_orchestrator._build_daily_features_row() lines 330-348.
    Without this, CrossAssetATRFilter strategies silently reject every trade
    in replay (fail-closed: missing key → filter returns False).
    """
    from trading_app.config import ALL_FILTERS, CrossAssetATRFilter

    cross_sources = {f.source_instrument for f in ALL_FILTERS.values() if isinstance(f, CrossAssetATRFilter)}
    for source in cross_sources:
        if source == instrument:
            continue
        src_result = con.execute(
            """SELECT atr_20_pct FROM daily_features
               WHERE symbol = ? AND orb_minutes = 5 AND atr_20_pct IS NOT NULL
                 AND trading_day = ?
               LIMIT 1""",
            [source, trading_day],
        ).fetchone()
        if src_result and src_result[0] is not None:
            row[f"cross_atr_{source}_pct"] = float(src_result[0])


def _get_bars_for_day(con, instrument: str, trading_day: date) -> list[dict]:
    """
    Fetch 1-minute bars for a trading day.

    Trading day in Brisbane starts at 23:00 UTC previous day.
    """
    prev_day = trading_day - timedelta(days=1)
    td_start = datetime(prev_day.year, prev_day.month, prev_day.day, 23, 0, tzinfo=UTC)
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
    use_ml: bool = False,
    use_e2_timeout: bool = False,
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
        portfolio = build_portfolio(db_path=db_path, instrument=instrument, max_correlation=max_correlation)

    if not portfolio.strategies:
        return ReplayResult(
            start_date=start_date or date.min,
            end_date=end_date or date.max,
        )

    # Pre-build cascade table if using market state
    cascade_table = None
    if use_market_state:
        from trading_app.cascade_table import build_cascade_table

        cascade_table = build_cascade_table(db_path, instrument=instrument)

    risk_mgr = RiskManager(risk_limits, corr_lookup=portfolio.corr_lookup)

    # ML subsystem removed 2026-04-11 (ML V3 sprint Stage 4). V1/V2/V3 all
    # DEAD per docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md.
    # The use_ml parameter is retained for caller API stability but has no
    # effect; any caller passing use_ml=True gets a warning and is otherwise
    # treated identically to use_ml=False.
    if use_ml:
        logger.warning(
            "use_ml=True has no effect — ML subsystem was removed 2026-04-11. "
            "See docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md "
            "for the V1/V2/V3 DEAD verdict and Blueprint NO-GO registry entry."
        )

    # MarketState built per-day below; engine gets it on day start
    market_state = None
    # E2 order timeout: break-speed overlay via execution timing.
    # Enabled by default for validated (instrument, session) pairs.
    # Pass None to disable (raw baseline comparison).
    from trading_app.config import E2_ORDER_TIMEOUT

    _e2_timeout = E2_ORDER_TIMEOUT if use_e2_timeout else None

    engine = ExecutionEngine(
        portfolio,
        cost_spec,
        risk_manager=risk_mgr,
        market_state=market_state,
        live_session_costs=live_session_costs,
        atr_velocity_overlay=ATR_VELOCITY_OVERLAY,
        e2_order_timeout=_e2_timeout,
    )

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
                    td,
                    db_path,
                    cascade_table=cascade_table,
                    visible_sessions=set(),  # nothing resolved yet
                    instrument=instrument,
                )
                market_state.score_strategies(portfolio.strategies)
                engine.market_state = market_state
            else:
                engine.market_state = None

            # Load daily_features for each unique orb_minutes in the portfolio
            # so filters evaluate against the correct aperture's ORB columns.
            unique_om = {s.orb_minutes for s in portfolio.strategies}
            df_rows: dict[int, dict] = {}
            median_atr = _get_median_atr_20(con, instrument, td)
            for om in sorted(unique_om):
                r = _get_daily_features_row(con, instrument, td, orb_minutes=om)
                if r is not None:
                    r["median_atr_20"] = median_atr
                    _inject_cross_asset_atrs_for_replay(con, r, instrument, td)
                    df_rows[om] = r
            engine.on_trading_day_start(td, daily_features_rows=df_rows)
            risk_mgr.daily_reset(td)

            day_summary = DaySummary(trading_day=td)
            day_summary.bars_processed = len(bars)

            for bar in bars:
                events = engine.on_bar(bar)

                for event in events:
                    if event.event_type == "ENTRY":
                        # Engine already checked risk_manager — just record
                        trade = _find_active_trade(engine, event.strategy_id)
                        entry = JournalEntry(
                            mode="replay",
                            trading_day=td,
                            strategy_id=event.strategy_id,
                            entry_model=_entry_model_from_strategy(event.strategy_id),
                            direction=event.direction,
                            entry_ts=event.timestamp,
                            entry_price=event.price,
                            contracts=event.contracts,
                            overlay_context=trade.overlay_context if trade else {},
                        )
                        day_summary.trades_entered += 1
                        result.journal.append(entry)

                    elif event.event_type == "REJECT":
                        trade = _find_completed_trade(engine, event.strategy_id)
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
                            overlay_context=trade.overlay_context if trade else {},
                        )
                        day_summary.risk_rejections += 1
                        result.total_risk_rejections += 1
                        result.journal.append(entry)

                    elif event.event_type == "ML_SKIP":
                        day_summary.ml_skips += 1
                        result.total_ml_skips += 1
                        result.journal.append(
                            JournalEntry(
                                mode="replay",
                                trading_day=td,
                                strategy_id=event.strategy_id,
                                entry_model=_entry_model_from_strategy(event.strategy_id),
                                direction=event.direction,
                                outcome="ml_skip",
                                risk_rejected=True,
                            )
                        )

                    elif event.event_type in ("EXIT", "SCRATCH"):
                        # Find matching journal entry and update
                        for je in reversed(result.journal):
                            if (
                                je.strategy_id == event.strategy_id
                                and je.trading_day == td
                                and je.exit_ts is None
                                and not je.risk_rejected
                            ):
                                je.exit_ts = event.timestamp
                                je.exit_price = event.price
                                je.outcome = _EXIT_OUTCOME_MAP.get(
                                    event.reason, event.reason.split("_")[0] if "_" in event.reason else event.reason
                                )
                                break

                        # Update PnL tracking from completed trade
                        trade = _find_completed_trade(engine, event.strategy_id)
                        if trade is None:
                            logger.warning(
                                f"No completed trade for {event.event_type} event: "
                                f"strategy={event.strategy_id}, day={td}"
                            )
                        if trade and trade.pnl_r is not None:
                            scaled_pnl = trade.pnl_r * trade.size_multiplier
                            for je in reversed(result.journal):
                                if (
                                    je.strategy_id == event.strategy_id
                                    and je.trading_day == td
                                    and je.pnl_r is None
                                    and not je.risk_rejected
                                ):
                                    je.pnl_r = scaled_pnl
                                    je.stop_price = trade.stop_price
                                    je.target_price = trade.target_price
                                    je.exit_mode = trade.exit_mode
                                    je.ib_alignment = trade.ib_alignment
                                    break

                            if scaled_pnl > 0:
                                day_summary.wins += 1
                                result.total_wins += 1
                            elif scaled_pnl < 0:
                                day_summary.losses += 1
                                result.total_losses += 1
                            else:
                                day_summary.scratches += 1
                                result.total_scratches += 1

                        # Reveal resolved ORB outcome in market state
                        if market_state is not None:
                            orb_label = _orb_from_strategy(event.strategy_id)
                            _reveal_outcome(market_state, orb_label, trade, cascade_table, portfolio)

            # End of day
            eod_events = engine.on_trading_day_end()
            for event in eod_events:
                if event.event_type == "SCRATCH":
                    day_summary.scratches += 1
                    result.total_scratches += 1
                    # Get actual mark-to-market PnL from engine's completed trade
                    trade = _find_completed_trade(engine, event.strategy_id)
                    if trade is None:
                        logger.warning(f"No completed trade for EOD SCRATCH: strategy={event.strategy_id}, day={td}")
                    raw_pnl = trade.pnl_r if (trade and trade.pnl_r is not None) else 0.0
                    scratch_pnl = raw_pnl * (trade.size_multiplier if trade else 1.0)
                    for je in reversed(result.journal):
                        if (
                            je.strategy_id == event.strategy_id
                            and je.trading_day == td
                            and je.exit_ts is None
                            and not je.risk_rejected
                        ):
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
                logger.info(
                    f"  Replayed {i + 1}/{len(trading_days)} days, "
                    f"{result.total_wins + result.total_losses + result.total_scratches} trades, "
                    f"PnL: {result.total_pnl_r:.2f}R"
                )

        result.total_trades = result.total_wins + result.total_losses + result.total_scratches
        return result


# =========================================================================
# Helpers
# =========================================================================


def _orb_from_strategy(strategy_id: str) -> str:
    """Extract ORB label from strategy ID.

    Format: {INSTRUMENT}_{ORB_LABEL}_{ENTRY_MODEL}_RR{rr}_CB{cb}_{FILTER}
    ORB_LABEL can contain underscores (e.g., US_DATA_830, CME_REOPEN).
    Entry model is always E1, E2, or E3.
    """
    parts = strategy_id.split("_")
    # Find the entry model token (E1, E2, E3) — everything between
    # parts[0] (instrument) and that token is the ORB label.
    for i, p in enumerate(parts):
        if p in ENTRY_MODELS and i > 1:
            return "_".join(parts[1:i])
    # Fallback: assume position 1 (legacy format)
    return parts[1] if len(parts) > 1 else ""


def _entry_model_from_strategy(strategy_id: str) -> str:
    """Extract entry model from strategy ID."""
    parts = strategy_id.split("_")
    for p in parts:
        if p in ENTRY_MODELS:
            return p
    # Fallback: assume position 2 (legacy format)
    return parts[2] if len(parts) > 2 else ""


def _find_active_trade(engine: ExecutionEngine, strategy_id: str):
    """Find the most recent active trade for a strategy."""
    for trade in reversed(engine.active_trades):
        if trade.strategy_id == strategy_id:
            return trade
    return None


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
# CLI Output Helpers
# =========================================================================


def _print_header(result: ReplayResult, instrument: str) -> None:
    """Print replay header summary."""
    total_decided = result.total_wins + result.total_losses
    wr = (result.total_wins / total_decided * 100) if total_decided > 0 else 0.0
    print(f"\n{'=' * 72}")
    print(f"  PAPER TRADER REPLAY: {instrument}")
    print(f"  {result.start_date} to {result.end_date} ({result.days_processed} trading days)")
    print(f"{'=' * 72}")
    print(
        f"  Trades: {result.total_trades}  (W:{result.total_wins}  L:{result.total_losses}  S:{result.total_scratches})"
    )
    print(f"  Win Rate: {wr:.1f}%   Total PnL: {result.total_pnl_r:+.2f}R")
    if result.total_trades > 0:
        avg_pnl = result.total_pnl_r / result.total_trades
        print(f"  Avg PnL/Trade: {avg_pnl:+.3f}R")
    print(f"  Risk Rejections: {result.total_risk_rejections}")
    if result.total_ml_skips > 0:
        print(f"  ML Skips: {result.total_ml_skips}")


def _print_drawdown(result: ReplayResult) -> None:
    """Compute and print max drawdown from daily PnL."""
    if not result.day_summaries:
        return
    cum_pnl = 0.0
    high_water = 0.0
    max_dd = 0.0
    max_dd_date = result.start_date
    for ds in result.day_summaries:
        cum_pnl += ds.daily_pnl_r
        if cum_pnl > high_water:
            high_water = cum_pnl
        dd = cum_pnl - high_water
        if dd < max_dd:
            max_dd = dd
            max_dd_date = ds.trading_day
    if max_dd < 0:
        print(f"  Max Drawdown: {max_dd:+.2f}R (on {max_dd_date})")
    else:
        print("  Max Drawdown: 0.00R (none)")
    if high_water > 0:
        print(f"  High Water: {high_water:+.2f}R")


def _print_layer_summary(result: ReplayResult) -> None:
    """Print per-layer breakdown (O5 baseline vs O30 ML overlay).

    Only prints when both layers are present (multi-RR portfolio).
    Supports kill criteria monitoring from the multi-RR plan.
    """
    from collections import defaultdict

    layers: dict[str, dict] = defaultdict(lambda: {"w": 0, "l": 0, "s": 0, "pnl": 0.0, "ml_skip": 0})
    has_o30 = False
    for je in result.journal:
        if je.risk_rejected or je.outcome is None:
            continue
        # Detect layer from strategy_id: O30 strategies have _O30 suffix
        if "_O30" in je.strategy_id:
            layer = "Layer2 (O30 RR2.0 ML)"
            has_o30 = True
        else:
            layer = "Layer1 (O5 RR1.0 raw)"
        s = layers[layer]
        if je.pnl_r is not None:
            s["pnl"] += je.pnl_r
        if je.outcome == "win":
            s["w"] += 1
        elif je.outcome in _LOSS_OUTCOMES:
            s["l"] += 1
        elif je.outcome == "scratch":
            s["s"] += 1

    # Count ML skips for Layer 2
    for je in result.journal:
        if je.outcome == "ml_skip" and "_O30" in je.strategy_id:
            layers["Layer2 (O30 RR2.0 ML)"]["ml_skip"] += 1

    if not has_o30:
        return  # Not a multi-RR portfolio

    print(f"\n{'-' * 72}")
    print("  LAYER BREAKDOWN (kill criteria monitoring)")
    print(f"{'-' * 72}")
    header = f"  {'Layer':<30} {'Trd':>4} {'W':>3} {'L':>3} {'WR%':>5} {'PnL(R)':>8} {'ExpR':>7} {'Skip':>5}"
    print(header)
    print(f"  {'-' * 67}")

    for layer_name in ["Layer1 (O5 RR1.0 raw)", "Layer2 (O30 RR2.0 ML)"]:
        s = layers.get(layer_name, {"w": 0, "l": 0, "s": 0, "pnl": 0.0, "ml_skip": 0})
        total = s["w"] + s["l"] + s["s"]
        decided = s["w"] + s["l"]
        wr = (s["w"] / decided * 100) if decided > 0 else 0.0
        expr = s["pnl"] / total if total > 0 else 0.0
        skip_str = str(s["ml_skip"]) if s["ml_skip"] > 0 else "-"
        print(
            f"  {layer_name:<30} {total:>4} {s['w']:>3} {s['l']:>3} {wr:>5.1f} {s['pnl']:>+8.2f} {expr:>+7.3f} {skip_str:>5}"
        )


def _print_strategy_summary(result: ReplayResult) -> None:
    """Print per-strategy breakdown table."""
    # Aggregate by strategy_id (exclude risk-rejected entries)
    from collections import defaultdict

    stats: dict[str, dict] = defaultdict(lambda: {"w": 0, "l": 0, "s": 0, "pnl": 0.0})
    for je in result.journal:
        if je.risk_rejected:
            continue
        if je.outcome is None:
            continue
        s = stats[je.strategy_id]
        if je.pnl_r is not None:
            s["pnl"] += je.pnl_r
        if je.outcome == "win":
            s["w"] += 1
        elif je.outcome in _LOSS_OUTCOMES:
            s["l"] += 1
        elif je.outcome == "scratch":
            s["s"] += 1

    if not stats:
        return

    print(f"\n{'-' * 72}")
    print("  STRATEGY BREAKDOWN")
    print(f"{'-' * 72}")
    header = f"  {'Strategy':<45} {'Trd':>4} {'W':>3} {'L':>3} {'S':>3} {'WR%':>5} {'PnL(R)':>8}"
    print(header)
    print(f"  {'-' * 69}")

    # Sort by PnL descending
    for sid, s in sorted(stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        total = s["w"] + s["l"] + s["s"]
        decided = s["w"] + s["l"]
        wr = (s["w"] / decided * 100) if decided > 0 else 0.0
        # Truncate long strategy IDs
        name = sid if len(sid) <= 45 else sid[:42] + "..."
        print(f"  {name:<45} {total:>4} {s['w']:>3} {s['l']:>3} {s['s']:>3} {wr:>5.1f} {s['pnl']:>+8.2f}")


def _print_session_summary(result: ReplayResult) -> None:
    """Print per-session (ORB label) breakdown."""
    from collections import defaultdict

    stats: dict[str, dict] = defaultdict(lambda: {"w": 0, "l": 0, "s": 0, "pnl": 0.0})
    for je in result.journal:
        if je.risk_rejected or je.outcome is None:
            continue
        orb = _orb_from_strategy(je.strategy_id)
        s = stats[orb]
        if je.pnl_r is not None:
            s["pnl"] += je.pnl_r
        if je.outcome == "win":
            s["w"] += 1
        elif je.outcome in _LOSS_OUTCOMES:
            s["l"] += 1
        elif je.outcome == "scratch":
            s["s"] += 1

    if not stats:
        return

    print(f"\n{'-' * 72}")
    print("  SESSION BREAKDOWN")
    print(f"{'-' * 72}")
    header = f"  {'Session':<25} {'Trd':>4} {'W':>3} {'L':>3} {'S':>3} {'WR%':>5} {'PnL(R)':>8}"
    print(header)
    print(f"  {'-' * 53}")

    for orb, s in sorted(stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        total = s["w"] + s["l"] + s["s"]
        decided = s["w"] + s["l"]
        wr = (s["w"] / decided * 100) if decided > 0 else 0.0
        print(f"  {orb:<25} {total:>4} {s['w']:>3} {s['l']:>3} {s['s']:>3} {wr:>5.1f} {s['pnl']:>+8.2f}")


def _print_risk_rejections(result: ReplayResult) -> None:
    """Print risk rejection breakdown by reason."""
    from collections import Counter

    reasons: Counter[str] = Counter()
    for je in result.journal:
        if je.risk_rejected and je.risk_reason:
            reasons[je.risk_reason] += 1

    if not reasons:
        return

    print(f"\n{'-' * 72}")
    print(f"  RISK REJECTIONS: {sum(reasons.values())}")
    print(f"{'-' * 72}")
    for reason, count in reasons.most_common():
        print(f"    {reason}: {count}")


def _print_daily_equity(result: ReplayResult, quiet: bool = False) -> None:
    """Print daily equity curve (cumulative PnL)."""
    if not result.day_summaries or quiet:
        return

    print(f"\n{'-' * 72}")
    print("  DAILY EQUITY")
    print(f"{'-' * 72}")
    cum_pnl = 0.0
    for ds in result.day_summaries:
        cum_pnl += ds.daily_pnl_r
        trades = ds.wins + ds.losses + ds.scratches
        bar = "+" * int(max(0, cum_pnl)) + "-" * int(max(0, -cum_pnl))
        if not bar:
            bar = "."
        print(f"  {ds.trading_day}  {ds.daily_pnl_r:>+6.2f}R  cum:{cum_pnl:>+7.2f}R  ({trades}t)  {bar}")


def _export_csv(result: ReplayResult, output_path: str) -> None:
    """Export full journal to CSV."""
    import csv

    fieldnames = [
        "trading_day",
        "strategy_id",
        "entry_model",
        "direction",
        "entry_ts",
        "entry_price",
        "stop_price",
        "target_price",
        "contracts",
        "exit_ts",
        "exit_price",
        "outcome",
        "pnl_r",
        "exit_mode",
        "ib_alignment",
        "risk_rejected",
        "risk_reason",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for je in result.journal:
            writer.writerow(
                {
                    "trading_day": je.trading_day,
                    "strategy_id": je.strategy_id,
                    "entry_model": je.entry_model,
                    "direction": je.direction,
                    "entry_ts": je.entry_ts,
                    "entry_price": je.entry_price,
                    "stop_price": je.stop_price,
                    "target_price": je.target_price,
                    "contracts": je.contracts,
                    "exit_ts": je.exit_ts,
                    "exit_price": je.exit_price,
                    "outcome": je.outcome,
                    "pnl_r": je.pnl_r,
                    "exit_mode": je.exit_mode,
                    "ib_alignment": je.ib_alignment,
                    "risk_rejected": je.risk_rejected,
                    "risk_reason": je.risk_reason,
                }
            )
    print(f"\n  Journal exported: {output_path} ({len(result.journal)} rows)")


# =========================================================================
# CLI
# =========================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Historical replay and paper trading")
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date")
    parser.add_argument("--end", type=date.fromisoformat, help="End date")
    parser.add_argument("--max-strategies", type=int, default=20, help="Max portfolio strategies")
    parser.add_argument("--max-daily-loss", type=float, default=-5.0, help="Max daily loss (R)")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent positions")
    parser.add_argument("--max-correlation", type=float, default=0.85, help="Max pairwise correlation (0-1)")
    parser.add_argument(
        "--live-session-costs",
        action="store_true",
        help="Use session-adjusted slippage (CME_REOPEN=1.3x, US_DATA_830=0.8x)",
    )
    parser.add_argument(
        "--use-ml", action="store_true", help="Enable ML meta-label P(win) filtering (skip low-confidence trades)"
    )
    parser.add_argument("--output", type=str, default=None, help="Export journal to CSV file path")
    parser.add_argument("--quiet", action="store_true", help="Suppress daily equity lines")
    parser.add_argument(
        "--raw-baseline",
        action="store_true",
        default=False,
        help="Use raw baseline portfolio from orb_outcomes (no validated_setups needed)",
    )
    parser.add_argument("--rr-target", type=float, default=1.0, help="RR target for raw baseline (default 1.0)")
    parser.add_argument("--entry-model", type=str, default="E2", help="Entry model for raw baseline (default E2)")
    parser.add_argument(
        "--exclude-sessions",
        type=str,
        default="NYSE_CLOSE",
        help="Comma-separated sessions to exclude from raw baseline (default NYSE_CLOSE)",
    )
    parser.add_argument(
        "--stop-multiplier",
        type=float,
        default=1.0,
        help="Stop multiplier (1.0=standard, 0.75=tight prop stop)",
    )
    parser.add_argument("--orb-minutes", type=int, default=5, help="ORB aperture (5, 15, or 30)")
    parser.add_argument(
        "--multi-rr",
        action="store_true",
        default=False,
        help="Multi-RR portfolio: O5 RR1.0 raw baseline + O30 RR2.0 ML-filtered overlay. Implies --use-ml.",
    )
    parser.add_argument(
        "--e2-timeout",
        action="store_true",
        default=False,
        help="Enable E2 order timeout (break-speed overlay). MNQ NYSE_OPEN/CLOSE: skip E2 trigger after 5min.",
    )
    args = parser.parse_args()

    # --multi-rr implies --use-ml (Layer 2 needs ML predictor)
    if args.multi_rr:
        args.use_ml = True

    risk_limits = RiskLimits(
        max_daily_loss_r=args.max_daily_loss,
        max_concurrent_positions=args.max_concurrent,
        max_per_session_positions=2,
    )

    portfolio = None
    if args.multi_rr:
        portfolio = build_multi_rr_portfolio(
            instrument=args.instrument,
            stop_multiplier=args.stop_multiplier,
            max_concurrent_positions=args.max_concurrent,
            max_daily_loss_r=args.max_daily_loss,
        )
        logger.info(
            "Multi-RR: %d strategies (%d Layer1 O5 + %d Layer2 O30)",
            len(portfolio.strategies),
            sum(1 for s in portfolio.strategies if s.orb_minutes == 5),
            sum(1 for s in portfolio.strategies if s.orb_minutes == 30),
        )
    elif args.raw_baseline:
        exclude = {s.strip() for s in args.exclude_sessions.split(",") if s.strip()}
        portfolio = build_raw_baseline_portfolio(
            instrument=args.instrument,
            rr_target=args.rr_target,
            entry_model=args.entry_model,
            orb_minutes=args.orb_minutes,
            exclude_sessions=exclude,
            stop_multiplier=args.stop_multiplier,
            max_concurrent_positions=args.max_concurrent,
            max_daily_loss_r=args.max_daily_loss,
        )
        logger.info("Raw baseline: %d strategies loaded", len(portfolio.strategies))

    result = replay_historical(
        instrument=args.instrument,
        portfolio=portfolio,
        start_date=args.start,
        end_date=args.end,
        risk_limits=risk_limits,
        live_session_costs=args.live_session_costs,
        max_correlation=args.max_correlation,
        use_ml=args.use_ml,
        use_e2_timeout=args.e2_timeout,
    )

    # Fix 6: No-data warning
    if result.days_processed == 0:
        logger.warning(
            "No trading days found for %s between %s and %s. "
            "Check that bars_1m and daily_features have data in this range.",
            args.instrument,
            args.start,
            args.end,
        )
        return

    # Rich output
    _print_header(result, args.instrument)
    _print_drawdown(result)
    _print_layer_summary(result)
    _print_strategy_summary(result)
    _print_session_summary(result)
    _print_risk_rejections(result)
    _print_daily_equity(result, quiet=args.quiet)

    # CSV export
    if args.output:
        _export_csv(result, args.output)

    print(f"\n{'=' * 72}")


if __name__ == "__main__":
    main()
