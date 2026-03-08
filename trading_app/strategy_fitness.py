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

import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date
from pathlib import Path

from pipeline.log import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import duckdb

from pipeline.dst import DST_AFFECTED_SESSIONS, is_winter_for_session
from pipeline.init_db import ORB_LABELS
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, EXCLUDED_FROM_FITNESS, VolumeFilter, apply_tight_stop
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
    std_r = variance**0.5

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
                  COALESCE(stop_multiplier, 1.0) as stop_multiplier,
                  sample_size, win_rate, expectancy_r, sharpe_ratio,
                  max_drawdown_r
           FROM validated_setups
           WHERE strategy_id = ?""",
        [strategy_id],
    ).fetchone()

    if row is None:
        return None

    cols = [desc[0] for desc in con.description]
    return dict(zip(cols, row, strict=False))


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
            utc_ts = break_ts.astimezone(UTC) if break_ts.tzinfo is not None else break_ts
            unique_minutes.add(utc_ts.hour * 60 + utc_ts.minute)

    if not unique_minutes:
        return

    def _minute_key(ts):
        utc = ts.astimezone(UTC) if ts.tzinfo is not None else ts
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
        utc_ts = break_ts.astimezone(UTC) if break_ts.tzinfo is not None else break_ts
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
            that regime only. Ignored for clean sessions (TOKYO_OPEN/SINGAPORE_OPEN etc.).
    """
    # Load outcomes
    params = [instrument, orb_minutes, orb_label, entry_model, rr_target, confirm_bars]
    where = [
        "symbol = ?",
        "orb_minutes = ?",
        "orb_label = ?",
        "entry_model = ?",
        "rr_target = ?",
        "confirm_bars = ?",
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
            WHERE {" AND ".join(where)}
            ORDER BY trading_day""",
        params,
    ).fetchall()
    cols = [desc[0] for desc in con.description]
    all_outcomes = [dict(zip(cols, r, strict=False)) for r in rows]

    if not all_outcomes:
        return []

    def _apply_dst(outcomes: list[dict]) -> list[dict]:
        """Apply DST regime filter if requested for a DST-affected session."""
        if dst_regime is None or orb_label not in DST_AFFECTED_SESSIONS:
            return outcomes
        want_winter = dst_regime == "winter"
        return [o for o in outcomes if is_winter_for_session(o["trading_day"], orb_label) == want_winter]

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
            WHERE {" AND ".join(feat_where)}""",
        feat_params,
    ).fetchall()
    feat_cols = [desc[0] for desc in con.description]
    feat_dicts = [dict(zip(feat_cols, r, strict=False)) for r in feat_rows]

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
# Bulk-load helpers (used by compute_portfolio_fitness only)
# =========================================================================


def _bulk_load_all_outcomes(con, instrument: str, end_date: date | None = None) -> dict:
    """Load ALL outcomes for an instrument, indexed by strategy key tuple.

    Returns: {(orb_label, orb_minutes, entry_model, rr_target, confirm_bars): [outcome_dicts]}
    Used by compute_portfolio_fitness() bulk path only.
    """
    params = [instrument]
    where = ["symbol = ?", "outcome IS NOT NULL"]
    if end_date:
        where.append("trading_day <= ?")
        params.append(end_date)

    rows = con.execute(
        f"""SELECT orb_label, orb_minutes, entry_model, rr_target, confirm_bars,
                   trading_day, outcome, pnl_r, mae_r, mfe_r, entry_price, stop_price
            FROM orb_outcomes
            WHERE {" AND ".join(where)}
            ORDER BY trading_day""",
        params,
    ).fetchall()
    cols = [desc[0] for desc in con.description]

    index = defaultdict(list)
    for row in rows:
        d = dict(zip(cols, row, strict=False))
        key = (d["orb_label"], d["orb_minutes"], d["entry_model"], d["rr_target"], d["confirm_bars"])
        index[key].append(d)
    return dict(index)


def _bulk_load_all_features(con, instrument: str) -> dict:
    """Load ALL daily_features for an instrument, indexed by (trading_day, orb_minutes).

    Returns: {(trading_day, orb_minutes): feature_dict}
    Used by compute_portfolio_fitness() bulk path only.
    """
    rows = con.execute(
        "SELECT * FROM daily_features WHERE symbol = ?",
        [instrument],
    ).fetchall()
    cols = [desc[0] for desc in con.description]

    index = {}
    for row in rows:
        d = dict(zip(cols, row, strict=False))
        index[(d["trading_day"], d["orb_minutes"])] = d
    return index


# =========================================================================
# Fitness computation (cached path for bulk portfolio)
# =========================================================================


def _compute_fitness_from_cache(
    con,
    strategy_id: str,
    params: dict,
    outcome_cache: dict,
    feature_cache: dict,
    as_of_date: date,
    rolling_months: int = 18,
    min_rolling_trades: int = 15,
) -> FitnessScore:
    """Compute fitness using pre-loaded outcome and feature caches.

    Same logic as _compute_fitness_with_con but avoids per-strategy DB queries.
    """
    rolling_start = _rolling_window_start(as_of_date, rolling_months)

    # Layer 1: Full-period stats (from validated_setups params)
    full_exp_r = params.get("expectancy_r", 0.0) or 0.0
    full_sharpe = params.get("sharpe_ratio")
    full_sample = params.get("sample_size", 0) or 0

    # Get outcomes from cache
    key = (
        params["orb_label"],
        params["orb_minutes"],
        params["entry_model"],
        params["rr_target"],
        params["confirm_bars"],
    )
    all_outcomes_raw = outcome_cache.get(key, [])
    if not all_outcomes_raw:
        logger.debug("Cache miss for %s (key=%s) — 0 outcomes", strategy_id, key)

    # end_date filter already applied during bulk load, but filter for as_of_date safety
    all_outcomes = [o for o in all_outcomes_raw if o["trading_day"] <= as_of_date]

    # Apply filter (same logic as _load_strategy_outcomes)
    filter_type = params["filter_type"]
    filt = ALL_FILTERS.get(filter_type)
    if filt is not None and filter_type != "NO_FILTER":
        orb_label = params["orb_label"]
        orb_minutes = params["orb_minutes"]
        # Get eligible days from feature cache
        eligible_days = set()
        for (td, om), feat_dict in feature_cache.items():
            if om == orb_minutes and filt.matches_row(feat_dict, orb_label):
                eligible_days.add(td)
        all_outcomes = [o for o in all_outcomes if o["trading_day"] in eligible_days]

    # Apply tight stop simulation for S075 strategies
    sm = params.get("stop_multiplier", 1.0)
    if sm != 1.0:
        from pipeline.cost_model import get_cost_spec

        cost_spec = get_cost_spec(params["instrument"])
        all_outcomes = apply_tight_stop(all_outcomes, sm, cost_spec)

    # Layer 2: Rolling regime metrics
    rolling_outcomes = [o for o in all_outcomes if rolling_start <= o["trading_day"] <= as_of_date]
    rolling_metrics = compute_metrics(rolling_outcomes)

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

    status, notes = classify_fitness(raw_rolling_exp_r, rolling_sample, recent_30)

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


# =========================================================================
# Fitness computation (per-strategy path)
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

    # Apply tight stop simulation for S075 strategies
    sm = params.get("stop_multiplier", 1.0)
    if sm != 1.0:
        from pipeline.cost_model import get_cost_spec

        cost_spec = get_cost_spec(params["instrument"])
        all_outcomes = apply_tight_stop(all_outcomes, sm, cost_spec)

    # Layer 2: Rolling regime metrics
    rolling_outcomes = [o for o in all_outcomes if rolling_start <= o["trading_day"] <= as_of_date]
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
            con,
            strategy_id,
            as_of_date,
            rolling_months,
            min_rolling_trades,
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
        exclusion_clause = " AND ".join(f"orb_label != '{s}'" for s in sorted(EXCLUDED_FROM_FITNESS))
        rows = con.execute(
            f"""SELECT strategy_id, instrument, orb_label, orb_minutes,
                       entry_model, rr_target, confirm_bars, filter_type,
                       COALESCE(stop_multiplier, 1.0) as stop_multiplier,
                       sample_size, win_rate, expectancy_r, sharpe_ratio,
                       max_drawdown_r
               FROM validated_setups
               WHERE instrument = ? AND LOWER(status) = 'active'
                 AND {exclusion_clause}
               ORDER BY strategy_id""",
            [instrument],
        ).fetchall()
        param_cols = [desc[0] for desc in con.description]
        all_params = [dict(zip(param_cols, r, strict=False)) for r in rows]

        if not all_params:
            return FitnessReport(
                as_of_date=as_of_date, scores=[], summary={"fit": 0, "watch": 0, "decay": 0, "stale": 0}
            )

        # Bulk-load ALL outcomes and features for this instrument (2 queries total)
        outcome_cache = _bulk_load_all_outcomes(con, instrument, end_date=as_of_date)
        feature_cache = _bulk_load_all_features(con, instrument)

        # Compute fitness from cache (no per-strategy queries)
        scores = []
        skipped = 0
        for p in all_params:
            try:
                filt = ALL_FILTERS.get(p["filter_type"])
                if isinstance(filt, VolumeFilter):
                    # VolumeFilter needs bars_1m enrichment — use per-strategy path
                    score = _compute_fitness_with_con(con, p["strategy_id"], as_of_date, rolling_months)
                else:
                    score = _compute_fitness_from_cache(
                        con,
                        p["strategy_id"],
                        p,
                        outcome_cache,
                        feature_cache,
                        as_of_date,
                        rolling_months,
                    )
                scores.append(score)
            except (ValueError, duckdb.Error, KeyError):
                logger.exception("Failed to compute fitness for %s", p["strategy_id"])
                skipped += 1
        if skipped:
            logger.warning("Skipped %d/%d strategies due to errors", skipped, len(all_params))

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
# Decay diagnostics: regime shift vs overfit
# =========================================================================


@dataclass(frozen=True)
class DecayDiagnosis:
    """Diagnosis of why a strategy is DECAY or WATCH."""

    strategy_id: str
    fitness_status: str
    family_hash: str | None
    family_size: int
    family_robustness: str | None
    # Sibling fitness counts
    siblings_fit: int
    siblings_watch: int
    siblings_decay: int
    siblings_stale: int
    # Diagnosis
    diagnosis: str  # REGIME_SHIFT | OVERFIT | FRAGMENTED | SINGLETON | NO_FAMILY
    diagnosis_notes: str


def diagnose_decay(
    con,
    strategy_id: str,
    as_of_date: date,
    rolling_months: int = 18,
) -> DecayDiagnosis:
    """
    For a DECAY or WATCH strategy, check its edge family siblings to
    distinguish regime shift from overfit.

    REGIME_SHIFT: >= 50% of family siblings are also DECAY/WATCH
    OVERFIT:      < 50% siblings decaying — this strategy is alone
    FRAGMENTED:   Mixed (some FIT, some DECAY) — brittle edge
    SINGLETON:    Family has only 1 member — no peers to compare
    NO_FAMILY:    Strategy has no family_hash assigned
    """
    # Compute this strategy's actual fitness status
    try:
        own_fitness = _compute_fitness_with_con(con, strategy_id, as_of_date, rolling_months)
        actual_status = own_fitness.fitness_status
    except (ValueError, duckdb.Error) as e:
        logger.warning("Could not compute fitness for %s: %s", strategy_id, e)
        actual_status = "UNKNOWN"

    # Get strategy's family info
    row = con.execute(
        """
        SELECT vs.family_hash, vs.status,
               ef.member_count, ef.robustness_status, ef.cv_expectancy
        FROM validated_setups vs
        LEFT JOIN edge_families ef ON vs.family_hash = ef.family_hash
        WHERE vs.strategy_id = ?
    """,
        [strategy_id],
    ).fetchone()

    if row is None:
        return DecayDiagnosis(
            strategy_id=strategy_id,
            fitness_status=actual_status,
            family_hash=None,
            family_size=0,
            family_robustness=None,
            siblings_fit=0,
            siblings_watch=0,
            siblings_decay=0,
            siblings_stale=0,
            diagnosis="NO_FAMILY",
            diagnosis_notes="Strategy not found",
        )

    family_hash = row[0]
    member_count = row[2] or 0
    robustness = row[3]

    if family_hash is None:
        return DecayDiagnosis(
            strategy_id=strategy_id,
            fitness_status=actual_status,
            family_hash=None,
            family_size=0,
            family_robustness=None,
            siblings_fit=0,
            siblings_watch=0,
            siblings_decay=0,
            siblings_stale=0,
            diagnosis="NO_FAMILY",
            diagnosis_notes="No family_hash assigned — run build_edge_families",
        )

    if member_count <= 1:
        return DecayDiagnosis(
            strategy_id=strategy_id,
            fitness_status=actual_status,
            family_hash=family_hash,
            family_size=1,
            family_robustness=robustness,
            siblings_fit=0,
            siblings_watch=0,
            siblings_decay=0,
            siblings_stale=0,
            diagnosis="SINGLETON",
            diagnosis_notes="Single-member family — no peers to compare",
        )

    # Get all siblings (same family, excluding self)
    siblings = con.execute(
        """
        SELECT strategy_id FROM validated_setups
        WHERE family_hash = ?
          AND strategy_id != ?
          AND LOWER(status) = 'active'
    """,
        [family_hash, strategy_id],
    ).fetchall()

    sibling_ids = [r[0] for r in siblings]

    # Compute fitness for each sibling
    counts = {"FIT": 0, "WATCH": 0, "DECAY": 0, "STALE": 0}
    for sid in sibling_ids:
        try:
            score = _compute_fitness_with_con(con, sid, as_of_date, rolling_months)
            key = score.fitness_status
            counts[key] = counts.get(key, 0) + 1
        except (ValueError, duckdb.Error) as e:
            logger.debug("Sibling %s fitness failed: %s", sid, e)
            counts["STALE"] += 1

    total_assessed = counts["FIT"] + counts["WATCH"] + counts["DECAY"]
    if total_assessed == 0:
        diagnosis = "SINGLETON"
        notes = f"All {len(sibling_ids)} siblings STALE — effectively isolated"
    else:
        decay_frac = (counts["DECAY"] + counts["WATCH"]) / total_assessed
        if decay_frac >= 0.50:
            diagnosis = "REGIME_SHIFT"
            notes = f"{counts['DECAY']}D + {counts['WATCH']}W of {total_assessed} assessed siblings also declining"
        elif counts["DECAY"] == 0 and counts["WATCH"] == 0:
            diagnosis = "OVERFIT"
            notes = f"All {counts['FIT']} assessed siblings are FIT — this strategy's decay is isolated"
        else:
            # Some siblings decaying but < 50%
            diagnosis = "FRAGMENTED"
            notes = (
                f"{counts['FIT']}F / {counts['WATCH']}W / {counts['DECAY']}D — "
                f"mixed signals, edge may be parameter-sensitive"
            )

    return DecayDiagnosis(
        strategy_id=strategy_id,
        fitness_status=actual_status,
        family_hash=family_hash,
        family_size=member_count,
        family_robustness=robustness,
        siblings_fit=counts["FIT"],
        siblings_watch=counts["WATCH"],
        siblings_decay=counts["DECAY"],
        siblings_stale=counts["STALE"],
        diagnosis=diagnosis,
        diagnosis_notes=notes,
    )


def diagnose_portfolio_decay(
    db_path: Path | None = None,
    instrument: str | None = None,
    as_of_date: date | None = None,
    rolling_months: int = 18,
) -> list[DecayDiagnosis]:
    """
    For all DECAY/WATCH strategies in the portfolio, diagnose whether
    the cause is regime shift or overfit.

    Returns list of DecayDiagnosis sorted by diagnosis type.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH
    if as_of_date is None:
        as_of_date = date.today()

    instruments = [instrument] if instrument else None

    with duckdb.connect(str(db_path), read_only=True) as con:
        # Get all active strategies
        where = ["LOWER(status) = 'active'"]
        params = []
        if instruments:
            where.append("instrument = ?")
            params.append(instruments[0])

        exclusion_clause = " AND ".join(f"orb_label != '{s}'" for s in sorted(EXCLUDED_FROM_FITNESS))
        where.append(exclusion_clause)

        rows = con.execute(
            f"""SELECT strategy_id FROM validated_setups
               WHERE {" AND ".join(where)}
               ORDER BY strategy_id""",
            params,
        ).fetchall()
        strategy_ids = [r[0] for r in rows]

        # First pass: find all DECAY/WATCH strategies
        decay_ids = []
        for sid in strategy_ids:
            try:
                score = _compute_fitness_with_con(con, sid, as_of_date, rolling_months)
                if score.fitness_status in ("DECAY", "WATCH"):
                    decay_ids.append(sid)
            except (ValueError, duckdb.Error) as e:
                logger.debug("Fitness computation failed for %s: %s", sid, e)

        # Second pass: diagnose each decaying strategy
        diagnoses = []
        # Cache: avoid re-diagnosing strategies in the same family
        diagnosed_families: dict[str, DecayDiagnosis] = {}
        for sid in decay_ids:
            # Check if we already diagnosed this family
            fh_row = con.execute(
                "SELECT family_hash FROM validated_setups WHERE strategy_id = ?",
                [sid],
            ).fetchone()
            fh = fh_row[0] if fh_row else None

            if fh and fh in diagnosed_families:
                # Reuse family diagnosis with different strategy_id
                cached = diagnosed_families[fh]
                # Get this strategy's actual fitness status
                try:
                    own_score = _compute_fitness_with_con(con, sid, as_of_date, rolling_months)
                    own_status = own_score.fitness_status
                except (ValueError, duckdb.Error):
                    own_status = "UNKNOWN"
                diag = DecayDiagnosis(
                    strategy_id=sid,
                    fitness_status=own_status,
                    family_hash=cached.family_hash,
                    family_size=cached.family_size,
                    family_robustness=cached.family_robustness,
                    siblings_fit=cached.siblings_fit,
                    siblings_watch=cached.siblings_watch,
                    siblings_decay=cached.siblings_decay,
                    siblings_stale=cached.siblings_stale,
                    diagnosis=cached.diagnosis,
                    diagnosis_notes=cached.diagnosis_notes,
                )
            else:
                diag = diagnose_decay(con, sid, as_of_date, rolling_months)
                if fh:
                    diagnosed_families[fh] = diag

            diagnoses.append(diag)

    # Sort: REGIME_SHIFT first, then OVERFIT, then others
    order = {"REGIME_SHIFT": 0, "OVERFIT": 1, "FRAGMENTED": 2, "SINGLETON": 3, "NO_FAMILY": 4}
    diagnoses.sort(key=lambda d: (order.get(d.diagnosis, 5), d.strategy_id))
    return diagnoses


# =========================================================================
# CLI
# =========================================================================


def _format_table(report: FitnessReport) -> str:
    """Format fitness report as a text table."""
    lines = []
    lines.append(f"Strategy Fitness Report (as of {report.as_of_date})")
    lines.append(f"{'=' * 80}")
    lines.append(f"{'Strategy':<45} {'Status':<6} {'RollExpR':>8} {'RollN':>5} {'Sh30':>6} {'D30':>6}")
    lines.append(f"{'-' * 80}")

    for s in sorted(report.scores, key=lambda x: x.fitness_status):
        roll_exp = f"{s.rolling_exp_r:.4f}" if s.rolling_exp_r is not None else "N/A"
        sh30 = f"{s.recent_sharpe_30:.3f}" if s.recent_sharpe_30 is not None else "N/A"
        d30 = f"{s.sharpe_delta_30:+.3f}" if s.sharpe_delta_30 is not None else "N/A"
        lines.append(
            f"{s.strategy_id:<45} {s.fitness_status:<6} {roll_exp:>8} {s.rolling_sample:>5} {sh30:>6} {d30:>6}"
        )

    lines.append(f"{'-' * 80}")
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

    parser = argparse.ArgumentParser(description="Strategy fitness assessment: rolling regime + decay monitoring")
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--strategy-id", default=None, help="Single strategy ID")
    parser.add_argument("--rolling-months", type=int, default=18, help="Rolling window months")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None, help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    parser.add_argument("--diagnose", action="store_true", help="Run decay diagnostics (regime shift vs overfit)")
    args = parser.parse_args()

    if args.diagnose:
        logger.info(f"Diagnosing DECAY/WATCH strategies for {args.instrument}...")
        diagnoses = diagnose_portfolio_decay(
            instrument=args.instrument,
            as_of_date=args.as_of,
            rolling_months=args.rolling_months,
        )
        if not diagnoses:
            logger.info("No DECAY or WATCH strategies found.")
        elif args.format == "json":
            logger.info(json.dumps([asdict(d) for d in diagnoses], indent=2, default=str))
        else:
            logger.info(f"{'Strategy':<50} {'Diagnosis':<16} {'Family':>6} {'F/W/D/S':>10}")
            logger.info("-" * 86)
            for d in diagnoses:
                fam = f"N={d.family_size}" if d.family_hash else "none"
                counts = f"{d.siblings_fit}/{d.siblings_watch}/{d.siblings_decay}/{d.siblings_stale}"
                logger.info(f"{d.strategy_id:<50} {d.diagnosis:<16} {fam:>6} {counts:>10}")
                logger.info(f"  {d.diagnosis_notes}")
            counts = {}
            for d in diagnoses:
                counts[d.diagnosis] = counts.get(d.diagnosis, 0) + 1
            logger.info(f"\nSummary: {dict(counts)}")
    elif args.strategy_id:
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
            logger.info(
                f"  Full-period: ExpR={score.full_period_exp_r:.4f}, Sharpe={score.full_period_sharpe}, N={score.full_period_sample}"
            )
            logger.info(
                f"  Rolling ({score.rolling_window_months}mo): ExpR={score.rolling_exp_r}, Sharpe={score.rolling_sharpe}, WR={score.rolling_win_rate}, N={score.rolling_sample}"
            )
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
