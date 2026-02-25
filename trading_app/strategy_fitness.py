"""
Strategy fitness assessment: rolling regime fitness + decay monitoring.

3-layer framework:
  1. Structural edge (full-history validation — from validated_setups)
  2. Rolling regime fitness (18-month trailing window)
  3. Decay monitoring (rolling trade-window Sharpe tracking)

Read-only consumer of existing tables. No new DB tables.
Answers: "Is this strategy still working in the current regime?"

Usage:
    python trading_app/strategy_fitness.py --instrument MGC
    python trading_app/strategy_fitness.py --strategy-id MGC_CME_REOPEN_E1_RR2.5_CB2_ORB_G4
    python trading_app/strategy_fitness.py --instrument MGC --rolling-months 12
    python trading_app/strategy_fitness.py --instrument MGC --format json
"""

import sys
import json
from pathlib import Path
from datetime import date, timezone
from dataclasses import dataclass, asdict

from pipeline.log import get_logger
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.init_db import ORB_LABELS
from pipeline.dst import DST_AFFECTED_SESSIONS, is_winter_for_session
from trading_app.config import ALL_FILTERS, VolumeFilter, EXCLUDED_FROM_FITNESS
from trading_app.strategy_discovery import compute_metrics

# Whitelist for SQL column interpolation safety
_VALID_ORB_LABELS = set(ORB_LABELS)

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

# =========================================================================
# Data classes
# =========================================================================

@dataclass(frozen=True)
class FitnessScore:
    """Fitness assessment for a single strategy."""
    strategy_id: str
    # Layer 1: Structural (from validated_setups)
    full_period_exp_r: float
    full_period_sharpe: float | None
    full_period_sample: int
    # Layer 2: Rolling regime (trailing window)
    rolling_exp_r: float | None
    rolling_sharpe: float | None
    rolling_win_rate: float | None
    rolling_sample: int
    rolling_window_months: int
    # Layer 3: Decay monitoring
    recent_sharpe_30: float | None
    recent_sharpe_60: float | None
    sharpe_delta_30: float | None
    sharpe_delta_60: float | None
    # Classification
    fitness_status: str
    fitness_notes: str

@dataclass(frozen=True)
class FitnessReport:
    """Fitness report for all strategies in a portfolio."""
    as_of_date: date
    scores: list[FitnessScore]
    summary: dict

# =========================================================================
# Classification
# =========================================================================

MIN_ROLLING_FIT = 15
MIN_ROLLING_WATCH = 10

def classify_fitness(
    rolling_exp_r: float | None,
    rolling_sample: int,
    recent_sharpe_30: float | None,
) -> tuple[str, str]:
    """
    Classify strategy fitness status.

    FIT:   rolling_exp_r > 0 AND recent_sharpe_30 > -0.1 AND rolling_sample >= 15
    WATCH: rolling_exp_r > 0 AND recent_sharpe_30 <= -0.1 (declining but positive)
           OR rolling_sample between 10-14 (thin data)
    DECAY: rolling_exp_r <= 0 (negative expectancy in recent window)
    STALE: rolling_sample < 10 (not enough recent trades to assess)
    """
    if rolling_sample < MIN_ROLLING_WATCH:
        return "STALE", f"Only {rolling_sample} trades in rolling window (need >= {MIN_ROLLING_WATCH})"

    if rolling_exp_r is None:
        return "STALE", "Rolling expectancy unavailable"

    if rolling_exp_r <= 0:
        return "DECAY", f"Negative rolling ExpR: {rolling_exp_r:.4f}"

    # rolling_exp_r > 0
    if rolling_sample < MIN_ROLLING_FIT:
        return "WATCH", f"Thin data: {rolling_sample} trades (need >= {MIN_ROLLING_FIT} for FIT)"

    if recent_sharpe_30 is not None and recent_sharpe_30 <= -0.1:
        return "WATCH", f"Declining Sharpe: recent_30={recent_sharpe_30:.4f}"

    return "FIT", "Positive rolling ExpR with stable recent Sharpe"

# =========================================================================
# Core metric computation
# =========================================================================

def _recent_trade_sharpe(outcomes: list[dict], n_trades: int) -> float | None:
    """Sharpe of last N trades (sorted by trading_day). None if < n_trades."""
    traded = [o for o in outcomes if o.get("outcome") in ("win", "loss")]
    traded.sort(key=lambda o: o["trading_day"])

    if len(traded) < n_trades:
        return None

    recent = traded[-n_trades:]
    r_values = [o["pnl_r"] for o in recent]

    if len(r_values) < 2:
        return None

    mean_r = sum(r_values) / len(r_values)
    variance = sum((r - mean_r) ** 2 for r in r_values) / (len(r_values) - 1)
    std_r = variance ** 0.5

    if std_r <= 0:
        return None

    return mean_r / std_r

def _rolling_window_start(as_of: date, months: int) -> date:
    """Compute start date for rolling window (as_of - N months)."""
    year = as_of.year
    month = as_of.month - months
    while month <= 0:
        year -= 1
        month += 12
    # Clamp day to valid range for target month
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    day = min(as_of.day, max_day)
    return date(year, month, day)

# =========================================================================
# Data loading
# =========================================================================

def _load_strategy_params(con, strategy_id: str) -> dict | None:
    """Load strategy parameters from validated_setups."""
    row = con.execute(
        """SELECT strategy_id, instrument, orb_label, orb_minutes,
                  entry_model, rr_target, confirm_bars, filter_type,
                  sample_size, win_rate, expectancy_r, sharpe_ratio,
                  max_drawdown_r
           FROM validated_setups
           WHERE strategy_id = ?""",
        [strategy_id],
    ).fetchone()

    if row is None:
        return None

    cols = [desc[0] for desc in con.description]
    return dict(zip(cols, row))


def _enrich_relative_volumes(con, feat_dicts, instrument, orb_label, lookback_days):
    """Compute rel_vol_{orb_label} for daily_features row dicts using bars_1m.

    Mirrors strategy_discovery._compute_relative_volumes() for a single orb_label.
    Modifies feat_dicts in-place. Fail-closed: missing data = no rel_vol key.
    """
    import statistics

    col = f"orb_{orb_label}_break_ts"
    unique_minutes = set()
    for row in feat_dicts:
        break_ts = row.get(col)
        if break_ts is not None and hasattr(break_ts, "hour"):
            utc_ts = break_ts.astimezone(timezone) if break_ts.tzinfo else break_ts
            unique_minutes.add(utc_ts.hour * 60 + utc_ts.minute)

    if not unique_minutes:
        return

    def _minute_key(ts):
        utc = ts.astimezone(timezone) if ts.tzinfo else ts
        return (utc.year, utc.month, utc.day, utc.hour, utc.minute)

    # Load historical volumes for each unique minute-of-day
    minute_history = {}
    for mod in sorted(unique_minutes):
        h, m = divmod(mod, 60)
        rows = con.execute(
            """SELECT ts_utc, volume FROM bars_1m
               WHERE symbol = ?
               AND EXTRACT(HOUR FROM (ts_utc AT TIME ZONE 'UTC')) = ?
               AND EXTRACT(MINUTE FROM (ts_utc AT TIME ZONE 'UTC')) = ?
               ORDER BY ts_utc""",
            [instrument, h, m],
        ).fetchall()
        minute_history[mod] = [(_minute_key(ts), vol) for ts, vol in rows]

    # Compute relative volume for each row
    for row in feat_dicts:
        break_ts = row.get(col)
        if break_ts is None:
            continue

        break_key = _minute_key(break_ts)
        utc_ts = break_ts.astimezone(timezone) if break_ts.tzinfo else break_ts
        mod = utc_ts.hour * 60 + utc_ts.minute
        history = minute_history.get(mod, [])
        if not history:
            continue

        idx = None
        for j, (k, _) in enumerate(history):
            if k == break_key:
                idx = j
                break
        if idx is None:
            continue

        break_vol = history[idx][1]
        if break_vol is None or break_vol == 0:
            continue

        start_idx = max(0, idx - lookback_days)
        prior_vols = [v for _, v in history[start_idx:idx] if v > 0]
        if not prior_vols:
            continue

        baseline = statistics.median(prior_vols)
        if baseline <= 0:
            continue

        row[f"rel_vol_{orb_label}"] = break_vol / baseline


def _load_strategy_outcomes(
    con,
    instrument: str,
    orb_label: str,
    orb_minutes: int,
    entry_model: str,
    rr_target: float,
    confirm_bars: int,
    filter_type: str,
    start_date: date | None = None,
    end_date: date | None = None,
    dst_regime: str | None = None,
) -> list[dict]:
    """
    Load filtered outcomes for a strategy from orb_outcomes + daily_features.

    Applies the strategy's filter_type to ensure only eligible days are included.
    Reuses filter logic from config.ALL_FILTERS.

    Args:
        dst_regime: If 'winter' or 'summer', restrict DST-affected sessions to
            that regime only. Ignored for clean sessions (1000/1100 etc.).
    """
    # Load outcomes
    params = [instrument, orb_minutes, orb_label, entry_model, rr_target, confirm_bars]
    where = [
        "symbol = ?", "orb_minutes = ?", "orb_label = ?",
        "entry_model = ?", "rr_target = ?", "confirm_bars = ?",
        "outcome IS NOT NULL",
    ]
    if start_date:
        where.append("trading_day >= ?")
        params.append(start_date)
    if end_date:
        where.append("trading_day <= ?")
        params.append(end_date)

    rows = con.execute(
        f"""SELECT trading_day, outcome, pnl_r, mae_r, mfe_r,
                   entry_price, stop_price
            FROM orb_outcomes
            WHERE {' AND '.join(where)}
            ORDER BY trading_day""",
        params,
    ).fetchall()
    cols = [desc[0] for desc in con.description]
    all_outcomes = [dict(zip(cols, r)) for r in rows]

    if not all_outcomes:
        return []

    def _apply_dst(outcomes: list[dict]) -> list[dict]:
        """Apply DST regime filter if requested for a DST-affected session."""
        if dst_regime is None or orb_label not in DST_AFFECTED_SESSIONS:
            return outcomes
        want_winter = (dst_regime == "winter")
        return [
            o for o in outcomes
            if is_winter_for_session(o["trading_day"], orb_label) == want_winter
        ]

    # Apply filter: load daily_features and check eligibility
    filt = ALL_FILTERS.get(filter_type)
    if filt is None:
        return _apply_dst(all_outcomes)  # unknown filter = pass-through

    # For NO_FILTER, skip the expensive daily_features check
    if filter_type == "NO_FILTER":
        return _apply_dst(all_outcomes)

    # Validate orb_label before f-string SQL interpolation
    if orb_label not in _VALID_ORB_LABELS:
        raise ValueError(f"Invalid orb_label '{orb_label}' for SQL column lookup")

    # Load daily features for the relevant date range.
    # FIX (F-10): Select ALL columns that any filter.matches_row() may need,
    # not just orb_size + day_of_week. Composite filters (DOW, break quality,
    # ATR velocity) need break_delay_min, break_bar_continues, compression_tier,
    # atr_vel_regime, break_dir etc. Using SELECT * is safe here — daily_features
    # has a bounded column set and we're already loading the full date range.
    feat_params = [instrument, orb_minutes]
    feat_where = ["symbol = ?", "orb_minutes = ?"]
    if start_date:
        feat_where.append("trading_day >= ?")
        feat_params.append(start_date)
    if end_date:
        feat_where.append("trading_day <= ?")
        feat_params.append(end_date)

    feat_rows = con.execute(
        f"""SELECT *
            FROM daily_features
            WHERE {' AND '.join(feat_where)}""",
        feat_params,
    ).fetchall()
    feat_cols = [desc[0] for desc in con.description]
    feat_dicts = [dict(zip(feat_cols, r)) for r in feat_rows]

    # VolumeFilter needs rel_vol enrichment from bars_1m
    if isinstance(filt, VolumeFilter):
        _enrich_relative_volumes(con, feat_dicts, instrument, orb_label, filt.lookback_days)

    eligible_days = set()
    for row_dict in feat_dicts:
        if filt.matches_row(row_dict, orb_label):
            eligible_days.add(row_dict["trading_day"])

    filtered = [o for o in all_outcomes if o["trading_day"] in eligible_days]
    return _apply_dst(filtered)

# =========================================================================
# Fitness computation
# =========================================================================

def _compute_fitness_with_con(
    con,
    strategy_id: str,
    as_of_date: date,
    rolling_months: int = 18,
    min_rolling_trades: int = 15,
) -> FitnessScore:
    """Compute fitness for a single strategy using an existing connection."""
    rolling_start = _rolling_window_start(as_of_date, rolling_months)

    # Load strategy parameters
    params = _load_strategy_params(con, strategy_id)
    if params is None:
        raise ValueError(f"Strategy '{strategy_id}' not found in validated_setups")

    # Layer 1: Full-period stats (from validated_setups)
    full_exp_r = params.get("expectancy_r", 0.0) or 0.0
    full_sharpe = params.get("sharpe_ratio")
    full_sample = params.get("sample_size", 0) or 0

    # FIX: Pass end_date=as_of_date to prevent lookahead leak.
    # Without this, _recent_trade_sharpe would include future outcomes
    # when running with a past --as-of date.
    all_outcomes = _load_strategy_outcomes(
        con,
        instrument=params["instrument"],
        orb_label=params["orb_label"],
        orb_minutes=params["orb_minutes"],
        entry_model=params["entry_model"],
        rr_target=params["rr_target"],
        confirm_bars=params["confirm_bars"],
        filter_type=params["filter_type"],
        end_date=as_of_date,
    )

    # Layer 2: Rolling regime metrics
    rolling_outcomes = [
        o for o in all_outcomes
        if rolling_start <= o["trading_day"] <= as_of_date
    ]
    rolling_metrics = compute_metrics(rolling_outcomes)

    # Raw rolling values for classification (before nulling)
    raw_rolling_exp_r = rolling_metrics["expectancy_r"]
    rolling_sharpe = rolling_metrics["sharpe_ratio"]
    rolling_wr = rolling_metrics["win_rate"]
    rolling_sample = rolling_metrics["sample_size"]

    # Layer 3: Decay monitoring
    recent_30 = _recent_trade_sharpe(all_outcomes, 30)
    recent_60 = _recent_trade_sharpe(all_outcomes, 60)

    delta_30 = None
    delta_60 = None
    if recent_30 is not None and full_sharpe is not None:
        delta_30 = recent_30 - full_sharpe
    if recent_60 is not None and full_sharpe is not None:
        delta_60 = recent_60 - full_sharpe

    # FIX: Classify using raw rolling_exp_r (before nulling).
    # Previously, nulling rolling_exp_r for thin data (10-14 trades)
    # caused classify_fitness to return STALE instead of WATCH.
    status, notes = classify_fitness(raw_rolling_exp_r, rolling_sample, recent_30)

    # Null out display values if below threshold (after classification)
    rolling_exp_r = raw_rolling_exp_r
    if rolling_sample < min_rolling_trades:
        rolling_exp_r = None
        rolling_sharpe = None
        rolling_wr = None

    return FitnessScore(
        strategy_id=strategy_id,
        full_period_exp_r=full_exp_r,
        full_period_sharpe=full_sharpe,
        full_period_sample=full_sample,
        rolling_exp_r=rolling_exp_r,
        rolling_sharpe=rolling_sharpe,
        rolling_win_rate=rolling_wr,
        rolling_sample=rolling_sample,
        rolling_window_months=rolling_months,
        recent_sharpe_30=recent_30,
        recent_sharpe_60=recent_60,
        sharpe_delta_30=delta_30,
        sharpe_delta_60=delta_60,
        fitness_status=status,
        fitness_notes=notes,
    )

def compute_fitness(
    strategy_id: str,
    db_path: Path | None = None,
    as_of_date: date | None = None,
    rolling_months: int = 18,
    min_rolling_trades: int = 15,
) -> FitnessScore:
    """Compute fitness for a single strategy. Read-only DB access."""
    if db_path is None:
        db_path = GOLD_DB_PATH
    if as_of_date is None:
        as_of_date = date.today()

    with duckdb.connect(str(db_path), read_only=True) as con:
        return _compute_fitness_with_con(
            con, strategy_id, as_of_date, rolling_months, min_rolling_trades,
        )
def compute_portfolio_fitness(
    db_path: Path | None = None,
    instrument: str = "MGC",
    as_of_date: date | None = None,
    rolling_months: int = 18,
) -> FitnessReport:
    """Compute fitness for all validated strategies. Returns FitnessReport."""
    if db_path is None:
        db_path = GOLD_DB_PATH
    if as_of_date is None:
        as_of_date = date.today()

    with duckdb.connect(str(db_path), read_only=True) as con:
        # Exclude sessions with no confirmed edge (see config.EXCLUDED_FROM_FITNESS)
        exclusion_clause = " AND ".join(
            f"orb_label != '{s}'" for s in sorted(EXCLUDED_FROM_FITNESS)
        )
        rows = con.execute(
            f"""SELECT strategy_id FROM validated_setups
               WHERE instrument = ? AND LOWER(status) = 'active'
                 AND {exclusion_clause}
               ORDER BY strategy_id""",
            [instrument],
        ).fetchall()
        strategy_ids = [r[0] for r in rows]

        # Reuse single connection for all strategies (was N+1 before)
        scores = []
        for sid in strategy_ids:
            try:
                score = _compute_fitness_with_con(
                    con, sid, as_of_date, rolling_months,
                )
                scores.append(score)
            except Exception as e:
                logger.warning(f"  WARN: Failed to compute fitness for {sid}: {e}")
    summary = {"fit": 0, "watch": 0, "decay": 0, "stale": 0}
    for s in scores:
        key = s.fitness_status.lower()
        summary[key] = summary.get(key, 0) + 1

    return FitnessReport(
        as_of_date=as_of_date,
        scores=scores,
        summary=summary,
    )

# =========================================================================
# CLI
# =========================================================================

def _format_table(report: FitnessReport) -> str:
    """Format fitness report as a text table."""
    lines = []
    lines.append(f"Strategy Fitness Report (as of {report.as_of_date})")
    lines.append(f"{'='*80}")
    lines.append(
        f"{'Strategy':<45} {'Status':<6} {'RollExpR':>8} {'RollN':>5} "
        f"{'Sh30':>6} {'D30':>6}"
    )
    lines.append(f"{'-'*80}")

    for s in sorted(report.scores, key=lambda x: x.fitness_status):
        roll_exp = f"{s.rolling_exp_r:.4f}" if s.rolling_exp_r is not None else "N/A"
        sh30 = f"{s.recent_sharpe_30:.3f}" if s.recent_sharpe_30 is not None else "N/A"
        d30 = f"{s.sharpe_delta_30:+.3f}" if s.sharpe_delta_30 is not None else "N/A"
        lines.append(
            f"{s.strategy_id:<45} {s.fitness_status:<6} {roll_exp:>8} "
            f"{s.rolling_sample:>5} {sh30:>6} {d30:>6}"
        )

    lines.append(f"{'-'*80}")
    lines.append(f"Summary: {report.summary}")
    return "\n".join(lines)

def _format_json(report: FitnessReport) -> str:
    """Format fitness report as JSON."""
    data = {
        "as_of_date": report.as_of_date.isoformat(),
        "summary": report.summary,
        "scores": [asdict(s) for s in report.scores],
    }
    return json.dumps(data, indent=2, default=str)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Strategy fitness assessment: rolling regime + decay monitoring"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--strategy-id", default=None, help="Single strategy ID")
    parser.add_argument("--rolling-months", type=int, default=18, help="Rolling window months")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None, help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    args = parser.parse_args()

    if args.strategy_id:
        score = compute_fitness(
            args.strategy_id,
            as_of_date=args.as_of,
            rolling_months=args.rolling_months,
        )
        if args.format == "json":
            logger.info(json.dumps(asdict(score), indent=2, default=str))
        else:
            logger.info(f"Strategy: {score.strategy_id}")
            logger.info(f"  Status: {score.fitness_status}")
            logger.info(f"  Full-period: ExpR={score.full_period_exp_r:.4f}, Sharpe={score.full_period_sharpe}, N={score.full_period_sample}")
            logger.info(f"  Rolling ({score.rolling_window_months}mo): ExpR={score.rolling_exp_r}, Sharpe={score.rolling_sharpe}, WR={score.rolling_win_rate}, N={score.rolling_sample}")
            logger.info(f"  Recent Sharpe: 30-trade={score.recent_sharpe_30}, 60-trade={score.recent_sharpe_60}")
            logger.info(f"  Sharpe Delta: 30={score.sharpe_delta_30}, 60={score.sharpe_delta_60}")
            logger.info(f"  Notes: {score.fitness_notes}")
    else:
        logger.info(f"Computing fitness for all validated {args.instrument} strategies...")
        report = compute_portfolio_fitness(
            instrument=args.instrument,
            as_of_date=args.as_of,
            rolling_months=args.rolling_months,
        )
        if args.format == "json":
            logger.info(_format_json(report))
        else:
            logger.info(_format_table(report))

if __name__ == "__main__":
    main()
