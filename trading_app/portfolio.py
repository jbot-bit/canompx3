"""
Portfolio construction from validated strategies.

Selects a diversified subset of validated_setups, computes position sizes,
and estimates capital requirements.

Usage:
    python trading_app/portfolio.py --instrument MGC
    python trading_app/portfolio.py --instrument MGC --max-strategies 10 --risk-pct 2.0
    python trading_app/portfolio.py --instrument MGC --account-equity 25000
"""

import json
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from pipeline.log import get_logger

logger = get_logger(__name__)

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

import duckdb
import numpy as np
import pandas as pd

from pipeline.build_daily_features import COMPRESSION_SESSIONS
from pipeline.cost_model import CostSpec, get_cost_spec
from pipeline.init_db import ORB_LABELS
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import (
    ALL_FILTERS,
    ATRVelocityFilter,
    CalendarSkipFilter,
    apply_tight_stop,
    classify_strategy,
    get_excluded_sessions,
)
from trading_app.validated_shelf import (
    deployable_validated_relation,
    validated_setups_has_deployment_scope,
)


def _get_table_names(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Return set of table names in the main schema."""
    rows = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
    return {r[0] for r in rows}


# =========================================================================
# Data classes
# =========================================================================


@dataclass(frozen=True)
class PortfolioStrategy:
    """A single strategy selected for the portfolio."""

    strategy_id: str
    instrument: str
    orb_label: str
    entry_model: str
    rr_target: float
    confirm_bars: int
    filter_type: str
    expectancy_r: float
    win_rate: float
    sample_size: int
    sharpe_ratio: float | None
    max_drawdown_r: float | None
    median_risk_points: float | None
    median_risk_dollars: float | None = None
    avg_risk_dollars: float | None = None
    avg_win_dollars: float | None = None
    avg_loss_dollars: float | None = None
    orb_minutes: int = 5  # ORB aperture (5, 15, or 30). Default 5 for backward compat.
    stop_multiplier: float = 1.0  # 1.0 = standard stop, 0.75 = tight stop at 75% of ORB range
    source: str = "baseline"  # "baseline", "nested", or "rolling"
    weight: float = 1.0
    max_contracts: int = 1

    @property
    def classification(self) -> str:
        """CORE / REGIME / INVALID per FIX5 rules."""
        return classify_strategy(self.sample_size)


@dataclass
class Portfolio:
    """A collection of strategies with risk parameters."""

    name: str
    instrument: str
    strategies: list[PortfolioStrategy]
    account_equity: float
    risk_per_trade_pct: float
    max_concurrent_positions: int
    max_daily_loss_r: float
    max_per_orb_positions: int = 1
    corr_lookup: dict[tuple[str, str], float] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize portfolio to JSON."""
        data = {
            "name": self.name,
            "instrument": self.instrument,
            "account_equity": self.account_equity,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "max_concurrent_positions": self.max_concurrent_positions,
            "max_daily_loss_r": self.max_daily_loss_r,
            "max_per_orb_positions": self.max_per_orb_positions,
            "strategy_count": len(self.strategies),
            "strategies": [asdict(s) for s in self.strategies],
        }
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "Portfolio":
        """Deserialize portfolio from JSON."""
        data = json.loads(json_str)
        strategies = [PortfolioStrategy(**s) for s in data.pop("strategies")]
        data.pop("strategy_count", None)
        return cls(strategies=strategies, **data)

    def summary(self) -> dict:
        """Return portfolio summary statistics."""
        if not self.strategies:
            return {"strategy_count": 0}

        orb_dist = {}
        em_dist = {}
        for s in self.strategies:
            orb_dist[s.orb_label] = orb_dist.get(s.orb_label, 0) + 1
            em_dist[s.entry_model] = em_dist.get(s.entry_model, 0) + 1

        exp_values = [s.expectancy_r for s in self.strategies]
        wr_values = [s.win_rate for s in self.strategies]

        return {
            "strategy_count": len(self.strategies),
            "orb_distribution": orb_dist,
            "entry_model_distribution": em_dist,
            "avg_expectancy_r": sum(exp_values) / len(exp_values),
            "avg_win_rate": sum(wr_values) / len(wr_values),
            "min_expectancy_r": min(exp_values),
            "max_expectancy_r": max(exp_values),
        }


# =========================================================================
# Position sizing
# =========================================================================


def compute_position_size(
    account_equity: float,
    risk_per_trade_pct: float,
    risk_points: float,
    cost_spec: CostSpec,
) -> int:
    """
    Compute number of contracts for a trade.

    PositionSize = (AccountEquity * Risk%) / (risk_points * PointValue)
    From CANONICAL_LOGIC.txt Section 3.

    Returns 0 if risk per contract exceeds risk budget.
    """
    if risk_points <= 0:
        return 0

    risk_dollars = risk_points * cost_spec.point_value
    available_risk = account_equity * (risk_per_trade_pct / 100.0)

    if risk_dollars <= 0:
        return 0

    contracts = available_risk / risk_dollars
    if contracts < 1.0:
        return 0  # Risk per contract exceeds budget — don't trade
    return int(contracts)


def compute_position_size_prop(
    max_drawdown: float,
    risk_per_trade_pct: float,
    risk_points: float,
    cost_spec: CostSpec,
) -> int:
    """
    Prop firm position sizing.

    True_Risk% = RiskAmount / MaxDrawdown (NOT / AccountBalance)
    From CANONICAL_LOGIC.txt Section 3.
    """
    if risk_points <= 0 or max_drawdown <= 0:
        return 0

    risk_dollars = risk_points * cost_spec.point_value
    available_risk = max_drawdown * (risk_per_trade_pct / 100.0)

    if risk_dollars <= 0:
        return 0

    contracts = available_risk / risk_dollars
    if contracts < 1.0:
        return 0  # Risk per contract exceeds budget — don't trade
    return int(contracts)


def compute_vol_scalar(
    atr_20: float,
    median_atr_20: float,
    max_scalar: float = 1.5,
    min_scalar: float = 0.5,
) -> float:
    """
    Turtle-style volatility scalar for position sizing.

    scalar = median_atr / current_atr
    - ATR above median -> scalar < 1.0 (smaller position)
    - ATR below median -> scalar > 1.0 (larger position)
    - Clamped to [min_scalar, max_scalar] to prevent extremes.

    Pass the result to compute_position_size_vol_scaled().
    """
    if atr_20 <= 0 or median_atr_20 <= 0:
        return 1.0
    raw = median_atr_20 / atr_20
    return max(min_scalar, min(raw, max_scalar))


def compute_position_size_vol_scaled(
    account_equity: float,
    risk_per_trade_pct: float,
    risk_points: float,
    cost_spec: CostSpec,
    vol_scalar: float = 1.0,
) -> int:
    """
    Position sizing with volatility normalization.

    Same as compute_position_size but applies vol_scalar to available risk.
    High ATR -> vol_scalar < 1 -> fewer contracts.
    Low ATR  -> vol_scalar > 1 -> more contracts (up to cap).
    """
    if risk_points <= 0 or vol_scalar <= 0:
        return 0

    risk_dollars = risk_points * cost_spec.point_value
    available_risk = account_equity * (risk_per_trade_pct / 100.0) * vol_scalar

    if risk_dollars <= 0:
        return 0

    contracts = available_risk / risk_dollars
    if contracts < 1.0:
        return 0
    return int(contracts)


# =========================================================================
# Portfolio construction
# =========================================================================


def load_validated_strategies(
    db_path: Path,
    instrument: str,
    min_expectancy_r: float = 0.10,
    include_nested: bool = False,
    include_rolling: bool = False,
    rolling_train_months: int = 12,
    family_heads_only: bool = False,
) -> list[dict]:
    """Load validated strategies from DB, filtered by minimum ExpR.

    When include_nested=True, also loads nested_validated strategies and
    marks each with source='baseline' or source='nested'.
    When include_rolling=True, also loads rolling-validated strategies
    (source='rolling') from the rolling portfolio aggregator.
    When family_heads_only=True, restricts to family head strategies from
    edge_families table. Falls back to unfiltered if edge_families doesn't exist.
    """
    with duckdb.connect(str(db_path), read_only=True) as con:
        shelf_relation = deployable_validated_relation(con, alias="vs")
        # Resolve family head filter (post-filter in Python to avoid SQL interpolation)
        head_ids = None
        if family_heads_only:
            from trading_app.db_manager import get_family_head_ids, has_edge_families

            if has_edge_families(con):
                head_ids = get_family_head_ids(con, instrument)

        # Load baseline strategies, enforcing locked RR from family_rr_locks
        # Per-instrument session exclusions (see config.EXCLUDED_FROM_FITNESS)
        excluded_sessions = sorted(get_excluded_sessions(instrument))
        baseline_rows = con.execute(
            f"""
            SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.entry_model,
                   vs.rr_target, vs.confirm_bars, vs.filter_type,
                   vs.expectancy_r, vs.win_rate, vs.sample_size,
                   vs.sharpe_ratio, vs.max_drawdown_r,
                   es.median_risk_points,
                   vs.median_risk_dollars, vs.avg_risk_dollars,
                   vs.avg_win_dollars, vs.avg_loss_dollars,
                   vs.orb_minutes,
                   COALESCE(vs.stop_multiplier, 1.0) as stop_multiplier,
                   'baseline' as source
            FROM {shelf_relation}
            LEFT JOIN experimental_strategies es
              ON vs.strategy_id = es.strategy_id
            LEFT JOIN family_rr_locks frl
              ON vs.instrument = frl.instrument
              AND vs.orb_label = frl.orb_label
              AND vs.filter_type = frl.filter_type
              AND vs.entry_model = frl.entry_model
              AND vs.orb_minutes = frl.orb_minutes
              AND vs.confirm_bars = frl.confirm_bars
            WHERE vs.instrument = ?
              AND vs.expectancy_r >= ?
              AND vs.orb_label NOT IN (SELECT UNNEST(?::VARCHAR[]))
              AND (frl.locked_rr IS NULL OR vs.rr_target = frl.locked_rr)
            ORDER BY vs.expectancy_r DESC
        """,
            [instrument, min_expectancy_r, excluded_sessions],
        ).fetchall()

        cols = [desc[0] for desc in con.description]
        results = [dict(zip(cols, row, strict=False)) for row in baseline_rows]

        # Load nested strategies if requested
        if include_nested:
            table_names = _get_table_names(con)

            if "nested_validated" in table_names:
                nested_rows = con.execute(
                    """
                    SELECT nv.strategy_id, nv.instrument, nv.orb_label, nv.entry_model,
                           nv.rr_target, nv.confirm_bars, nv.filter_type,
                           nv.expectancy_r, nv.win_rate, nv.sample_size,
                           nv.sharpe_ratio, nv.max_drawdown_r,
                           ns.median_risk_points,
                           NULL as median_risk_dollars, NULL as avg_risk_dollars,
                           NULL as avg_win_dollars, NULL as avg_loss_dollars,
                           nv.orb_minutes,
                           CAST(1.0 AS DOUBLE) as stop_multiplier,
                           'nested' as source
                    FROM nested_validated nv
                    LEFT JOIN nested_strategies ns
                      ON nv.strategy_id = ns.strategy_id
                    LEFT JOIN family_rr_locks frl
                      ON nv.instrument = frl.instrument
                      AND nv.orb_label = frl.orb_label
                      AND nv.filter_type = frl.filter_type
                      AND nv.entry_model = frl.entry_model
                      AND nv.orb_minutes = frl.orb_minutes
                      AND nv.confirm_bars = frl.confirm_bars
                    WHERE nv.instrument = ?
                      AND LOWER(nv.status) = 'active'
                      AND nv.expectancy_r >= ?
                      AND nv.orb_label NOT IN (SELECT UNNEST(?::VARCHAR[]))
                      AND (frl.locked_rr IS NULL OR nv.rr_target = frl.locked_rr)
                    ORDER BY nv.expectancy_r DESC
                """,
                    [instrument, min_expectancy_r, excluded_sessions],
                ).fetchall()

                nested_cols = [desc[0] for desc in con.description]
                assert len(nested_cols) == len(cols), (
                    f"Nested column count ({len(nested_cols)}) != baseline ({len(cols)})"
                )

                # Append nested results
                for row in nested_rows:
                    results.append(dict(zip(cols, row, strict=False)))

        # Load rolling-validated strategies if requested
        if include_rolling:
            from trading_app.rolling_portfolio import load_rolling_validated_strategies

            rolling_results = load_rolling_validated_strategies(
                db_path,
                instrument,
                rolling_train_months,
                min_expectancy_r=min_expectancy_r,
            )
            # Deduplicate: don't add rolling strategies already in baseline/nested
            existing_ids = {r["strategy_id"] for r in results}
            for r in rolling_results:
                if r["strategy_id"] not in existing_ids:
                    results.append(r)

        # Apply family head filter (Python-side to avoid SQL interpolation)
        if head_ids is not None:
            results = [r for r in results if r["strategy_id"] in head_ids]

        return results


def diversify_strategies(
    candidates: list[dict],
    max_strategies: int,
    max_per_orb: int = 5,
    max_per_entry_model: int | None = None,
    corr_matrix: pd.DataFrame | None = None,
    max_correlation: float = 0.85,
) -> list[dict]:
    """
    Select a diversified subset of strategies.

    Selection priority:
    1. Highest ExpR first
    2. Enforce max per ORB label
    3. Enforce max per entry model (if set)
    4. Reject candidates too correlated (>= max_correlation) with any already-selected strategy
    5. Stop at max_strategies
    """
    selected = []
    orb_counts = {}
    em_counts = {}

    for s in candidates:
        if len(selected) >= max_strategies:
            break

        orb = s["orb_label"]
        em = s["entry_model"]

        # Check ORB diversification limit
        if orb_counts.get(orb, 0) >= max_per_orb:
            continue

        # Check entry model diversification limit
        if max_per_entry_model is not None:
            if em_counts.get(em, 0) >= max_per_entry_model:
                continue

        # Check correlation with already-selected strategies
        if corr_matrix is not None:
            sid = s["strategy_id"]
            if sid in corr_matrix.columns:
                too_correlated = any(
                    sel["strategy_id"] in corr_matrix.columns
                    and pd.notna(corr_matrix.loc[sid, sel["strategy_id"]])
                    and abs(corr_matrix.loc[sid, sel["strategy_id"]]) >= max_correlation
                    for sel in selected
                )
                if too_correlated:
                    continue

        selected.append(s)
        orb_counts[orb] = orb_counts.get(orb, 0) + 1
        em_counts[em] = em_counts.get(em, 0) + 1

    return selected


def build_portfolio(
    db_path: Path | None = None,
    instrument: str = "MGC",
    name: str = "default",
    max_strategies: int = 20,
    min_expectancy_r: float = 0.10,
    max_per_orb: int = 5,
    account_equity: float = 25000.0,
    risk_per_trade_pct: float = 2.0,
    max_concurrent_positions: int = 3,
    max_daily_loss_r: float = 5.0,
    include_nested: bool = False,
    include_rolling: bool = False,
    rolling_train_months: int = 12,
    max_correlation: float = 0.85,
    family_heads_only: bool = False,
    calendar_overlay: CalendarSkipFilter | None = None,
    atr_velocity_overlay: ATRVelocityFilter | None = None,
) -> Portfolio:
    """
    Build a diversified portfolio from validated strategies.

    Args:
        include_nested: If True, also includes nested_validated strategies.
        include_rolling: If True, also includes rolling-validated strategies.
        rolling_train_months: Training window size for rolling evaluation.

    Returns Portfolio with selected strategies and risk parameters.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    # Load and filter candidates
    candidates = load_validated_strategies(
        db_path,
        instrument,
        min_expectancy_r,
        include_nested=include_nested,
        include_rolling=include_rolling,
        rolling_train_months=rolling_train_months,
        family_heads_only=family_heads_only,
    )

    if not candidates:
        return Portfolio(
            name=name,
            instrument=instrument,
            strategies=[],
            account_equity=account_equity,
            risk_per_trade_pct=risk_per_trade_pct,
            max_concurrent_positions=max_concurrent_positions,
            max_daily_loss_r=max_daily_loss_r,
        )

    # Compute correlation matrix for diversification filtering
    corr = None
    if len(candidates) > 1:
        strategy_ids = [c["strategy_id"] for c in candidates]
        corr = correlation_matrix(
            db_path,
            strategy_ids,
            min_overlap_days=100,
            calendar_overlay=calendar_overlay,
            atr_velocity_overlay=atr_velocity_overlay,
        )
        if corr.empty:
            corr = None

    # Diversify selection
    selected = diversify_strategies(
        candidates,
        max_strategies,
        max_per_orb,
        corr_matrix=corr,
        max_correlation=max_correlation,
    )

    # Convert to PortfolioStrategy objects
    strategies = []
    for s in selected:
        strategies.append(
            PortfolioStrategy(
                strategy_id=s["strategy_id"],
                instrument=s["instrument"],
                orb_label=s["orb_label"],
                entry_model=s["entry_model"],
                rr_target=s["rr_target"],
                confirm_bars=s["confirm_bars"],
                filter_type=s["filter_type"],
                expectancy_r=s["expectancy_r"],
                win_rate=s["win_rate"],
                sample_size=s["sample_size"],
                sharpe_ratio=s.get("sharpe_ratio"),
                max_drawdown_r=s.get("max_drawdown_r"),
                median_risk_points=s.get("median_risk_points"),
                median_risk_dollars=s.get("median_risk_dollars"),
                avg_risk_dollars=s.get("avg_risk_dollars"),
                avg_win_dollars=s.get("avg_win_dollars"),
                avg_loss_dollars=s.get("avg_loss_dollars"),
                orb_minutes=int(s.get("orb_minutes", 5)),
                stop_multiplier=s.get("stop_multiplier", 1.0),
                source=s.get("source", "baseline"),
            )
        )

    # Classification-aware weight adjustment (FIX5)
    # CORE (>=100): weight=1.0 (default), REGIME (30-99): weight=0.5, INVALID (<30): excluded
    classified = []
    for strat in strategies:
        cls = strat.classification
        if cls == "INVALID":
            logger.info("Excluding INVALID strategy %s (sample_size=%d)", strat.strategy_id, strat.sample_size)
            continue
        if cls == "REGIME":
            strat = replace(strat, weight=0.5)
            logger.info(
                "REGIME strategy %s: weight reduced to 0.5 (sample_size=%d)", strat.strategy_id, strat.sample_size
            )
        classified.append(strat)
    strategies = classified

    # Build corr_lookup for RiskManager (flat dict from selected strategies only)
    lookup: dict[tuple[str, str], float] = {}
    if corr is not None:
        selected_ids = {s["strategy_id"] for s in selected}
        for sid_a in selected_ids:
            for sid_b in selected_ids:
                if sid_a >= sid_b:
                    continue
                if sid_a in corr.columns and sid_b in corr.columns:
                    val = corr.loc[sid_a, sid_b]
                    if pd.notna(val):
                        lookup[(sid_a, sid_b)] = float(val)

    return Portfolio(
        name=name,
        instrument=instrument,
        strategies=strategies,
        account_equity=account_equity,
        risk_per_trade_pct=risk_per_trade_pct,
        max_concurrent_positions=max_concurrent_positions,
        max_daily_loss_r=max_daily_loss_r,
        corr_lookup=lookup,
    )


def build_raw_baseline_portfolio(
    db_path: Path | None = None,
    instrument: str = "MNQ",
    rr_target: float = 1.0,
    entry_model: str = "E2",
    confirm_bars: int = 1,
    orb_minutes: int = 5,
    exclude_sessions: set[str] | None = None,
    account_equity: float = 25000.0,
    risk_per_trade_pct: float = 2.0,
    max_concurrent_positions: int = 3,
    max_daily_loss_r: float = 5.0,
    stop_multiplier: float = 1.0,
    include_negative: bool = False,
) -> Portfolio:
    """Build a portfolio directly from orb_outcomes — no validated_setups required.

    One PortfolioStrategy per session with positive expectancy. Used for raw
    baseline paper trading (no ML, no filters).

    When include_negative=True, includes sessions with negative baselines too.
    Use this for ML replay comparison where ML IS the filter.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH
    if exclude_sessions is None:
        exclude_sessions = {"NYSE_CLOSE"}

    with duckdb.connect(str(db_path), read_only=True) as con:
        rows = con.execute(
            """
            SELECT
                orb_label,
                COUNT(*) AS n,
                AVG(pnl_r) AS expr,
                COUNT(CASE WHEN pnl_r > 0 THEN 1 END) * 1.0 / COUNT(*) AS wr,
                STDDEV_POP(pnl_r) AS sd,
                MEDIAN(risk_dollars) AS med_risk_dollars
            FROM orb_outcomes
            WHERE symbol = ? AND entry_model = ? AND rr_target = ?
              AND confirm_bars = ? AND orb_minutes = ?
            GROUP BY orb_label
            ORDER BY expr DESC
            """,
            [instrument, entry_model, rr_target, confirm_bars, orb_minutes],
        ).fetchall()

    cost_spec = get_cost_spec(instrument)
    strategies: list[PortfolioStrategy] = []

    for orb_label, n, expr, wr, sd, med_risk_dollars in rows:
        if orb_label in exclude_sessions:
            logger.info("Raw baseline: excluding %s (in exclude_sessions)", orb_label)
            continue
        if not include_negative and (expr is None or expr <= 0):
            logger.info("Raw baseline: excluding %s (ExpR=%.4f <= 0)", orb_label, expr or 0)
            continue

        sharpe = float(expr / sd) if sd and sd > 0 else None
        med_risk_pts = float(med_risk_dollars / cost_spec.point_value) if med_risk_dollars else None
        sid = f"{instrument}_{orb_label}_{entry_model}_RR{rr_target}_CB{confirm_bars}_NO_FILTER"

        strategies.append(
            PortfolioStrategy(
                strategy_id=sid,
                instrument=instrument,
                orb_label=orb_label,
                entry_model=entry_model,
                rr_target=rr_target,
                confirm_bars=confirm_bars,
                filter_type="NO_FILTER",
                expectancy_r=float(expr),
                win_rate=float(wr),
                sample_size=int(n),
                sharpe_ratio=sharpe,
                max_drawdown_r=None,
                median_risk_points=med_risk_pts,
                median_risk_dollars=float(med_risk_dollars) if med_risk_dollars else None,
                orb_minutes=orb_minutes,
                stop_multiplier=stop_multiplier,
                source="raw_baseline",
            )
        )

    logger.info(
        "Raw baseline portfolio: %d sessions (excluded %d)",
        len(strategies),
        len(exclude_sessions),
    )
    return Portfolio(
        name=f"raw_baseline_{instrument}_{entry_model}_RR{rr_target}",
        instrument=instrument,
        strategies=strategies,
        account_equity=account_equity,
        risk_per_trade_pct=risk_per_trade_pct,
        max_concurrent_positions=max_concurrent_positions,
        max_daily_loss_r=max_daily_loss_r,
    )


def build_profile_portfolio(
    profile_id: str,
    db_path: Path | None = None,
    account_equity: float | None = None,
    risk_per_trade_pct: float = 2.0,
    max_concurrent_positions: int = 3,
    max_daily_loss_r: float = 5.0,
    instrument: str | None = None,
) -> Portfolio:
    """Build a portfolio from a prop_profiles.py account profile.

    Reads daily_lanes from the profile, looks up each strategy_id in
    validated_setups, and builds a Portfolio with the exact validated params.

    Args:
        instrument: If provided, include only lanes matching this instrument.
            Use for multi-instrument profiles that need one orchestrator per instrument.
            DD budget check is skipped for subsets (validated on full profile in tests).

    Raises RuntimeError if any lane's strategy_id is missing from validated_setups.
    """
    from trading_app.prop_profiles import ACCOUNT_PROFILES, get_account_tier

    if profile_id not in ACCOUNT_PROFILES:
        raise ValueError(f"Unknown profile '{profile_id}'. Valid: {sorted(ACCOUNT_PROFILES.keys())}")

    from trading_app.prop_profiles import effective_daily_lanes

    profile = ACCOUNT_PROFILES[profile_id]
    all_lanes = effective_daily_lanes(profile)
    if not all_lanes:
        raise RuntimeError(
            f"Profile '{profile_id}' has no daily_lanes defined and no allocation JSON. "
            f"Cannot build portfolio without explicit strategy lanes."
        )

    if db_path is None:
        db_path = GOLD_DB_PATH

    # Always fetch tier (needed for DD budget check even if equity is specified)
    tier = get_account_tier(profile.firm, profile.account_size)
    if account_equity is None:
        account_equity = float(profile.account_size)

    # Filter lanes by instrument if specified (multi-instrument profiles)
    lanes = all_lanes
    if instrument is not None:
        lanes = tuple(lane for lane in lanes if lane.instrument == instrument)
        if not lanes:
            raise RuntimeError(
                f"Profile '{profile_id}' has no lanes for instrument '{instrument}'. "
                f"Available: {sorted({lane.instrument for lane in all_lanes})}"
            )
    else:
        # No filter — all lanes must be same instrument
        instruments = {lane.instrument for lane in lanes}
        if len(instruments) != 1:
            raise RuntimeError(
                f"Profile '{profile_id}' has mixed instruments: {instruments}. "
                f"Pass instrument= to select one, or use MultiInstrumentRunner."
            )
        instrument = instruments.pop()

    strategies: list[PortfolioStrategy] = []
    cost_spec = get_cost_spec(instrument)

    _skip_dd_check = instrument is not None and len(lanes) < len(all_lanes)

    with duckdb.connect(str(db_path), read_only=True) as con:
        has_deployment_scope = validated_setups_has_deployment_scope(con)
        deployment_scope_sql = "COALESCE(deployment_scope, 'deployable')" if has_deployment_scope else "'deployable'"
        for lane in lanes:
            row = con.execute(
                f"""
                SELECT strategy_id, instrument, orb_label, entry_model,
                       rr_target, confirm_bars, filter_type, orb_minutes,
                       sample_size, win_rate, expectancy_r, sharpe_ratio,
                       max_drawdown_r, median_risk_dollars, avg_risk_dollars,
                       avg_win_dollars, avg_loss_dollars, stop_multiplier,
                       status, {deployment_scope_sql} AS deployment_scope, fdr_significant
                FROM validated_setups
                WHERE strategy_id = ?
                """,
                [lane.strategy_id],
            ).fetchone()

            if row is None:
                raise RuntimeError(
                    f"BLOCKER: Strategy '{lane.strategy_id}' from profile "
                    f"'{profile_id}' NOT FOUND in validated_setups. "
                    f"Run strategy validation first."
                )

            (
                sid,
                inst,
                orb_label,
                entry_model,
                rr_target,
                confirm_bars,
                filter_type,
                orb_minutes,
                sample_size,
                win_rate,
                expectancy_r,
                sharpe_ratio,
                max_drawdown_r,
                median_risk_dollars,
                avg_risk_dollars,
                avg_win_dollars,
                avg_loss_dollars,
                stop_mult,
                status,
                deployment_scope,
                fdr_sig,
            ) = row

            if status == "RETIRED":
                raise RuntimeError(
                    f"BLOCKER: Strategy '{sid}' is RETIRED in validated_setups. "
                    f"Remove from profile '{profile_id}' or re-validate."
                )
            if deployment_scope and str(deployment_scope).lower() != "deployable":
                raise RuntimeError(
                    f"BLOCKER: Strategy '{sid}' is marked deployment_scope={deployment_scope!r}. "
                    f"Profiles may only reference deployable shelf rows."
                )

            if not fdr_sig:
                logger.warning(
                    "WARNING: Strategy '%s' is NOT FDR-significant (fdr_significant=False). "
                    "Proceeding but this strategy has weak statistical evidence.",
                    sid,
                )

            med_risk_pts = float(median_risk_dollars / cost_spec.point_value) if median_risk_dollars else None

            # Use profile's stop multiplier if lane doesn't override
            effective_stop_mult = (
                lane.planned_stop_multiplier if lane.planned_stop_multiplier is not None else profile.stop_multiplier
            )

            strategies.append(
                PortfolioStrategy(
                    strategy_id=sid,
                    instrument=inst,
                    orb_label=orb_label,
                    entry_model=entry_model,
                    rr_target=float(rr_target),
                    confirm_bars=int(confirm_bars),
                    filter_type=filter_type,
                    expectancy_r=float(expectancy_r),
                    win_rate=float(win_rate),
                    sample_size=int(sample_size),
                    sharpe_ratio=float(sharpe_ratio) if sharpe_ratio else None,
                    max_drawdown_r=float(max_drawdown_r) if max_drawdown_r else None,
                    median_risk_points=med_risk_pts,
                    median_risk_dollars=float(median_risk_dollars) if median_risk_dollars else None,
                    avg_risk_dollars=float(avg_risk_dollars) if avg_risk_dollars else None,
                    avg_win_dollars=float(avg_win_dollars) if avg_win_dollars else None,
                    avg_loss_dollars=float(avg_loss_dollars) if avg_loss_dollars else None,
                    orb_minutes=int(orb_minutes),
                    stop_multiplier=float(effective_stop_mult),
                    source="profile",
                    weight=1.0,
                    max_contracts=1,
                )
            )
            logger.info(
                "Profile lane: %s | %s %s RR%.1f O%d | N=%d WR=%.0f%% ExpR=%.3f | stop=%.2fx",
                sid,
                orb_label,
                filter_type,
                rr_target,
                orb_minutes,
                sample_size,
                win_rate * 100,
                expectancy_r,
                effective_stop_mult,
            )

    if not strategies:
        raise RuntimeError(f"Profile '{profile_id}' resolved 0 strategies")

    # Validate session names against SESSION_CATALOG
    from pipeline.dst import SESSION_CATALOG

    valid_sessions = set(SESSION_CATALOG.keys())
    for s in strategies:
        if s.orb_label not in valid_sessions:
            raise RuntimeError(
                f"Unknown session '{s.orb_label}' in profile '{profile_id}' "
                f"strategy '{s.strategy_id}'. "
                f"Valid sessions: {sorted(valid_sessions)}. "
                f"Check for typos in prop_profiles.py daily_lanes."
            )

    # --- DD budget pre-flight check (fail-closed) ---
    # Skip for instrument-filtered subsets — DD is validated on the full profile
    # by validate_dd_budget() in tests. Per-instrument subset check is meaningless
    # (the total DD is what matters, not per-instrument slices).
    if not _skip_dd_check:
        from trading_app.prop_portfolio import _compute_dd_per_contract
        from trading_app.prop_profiles import get_firm_spec

        firm_spec = get_firm_spec(profile.firm)
        dd_type = firm_spec.dd_type

        dd_breakdown: list[str] = []
        total_dd = 0.0
        for s in strategies:
            lane_dd = _compute_dd_per_contract(s.stop_multiplier, dd_type)
            total_dd += lane_dd
            dd_breakdown.append(f"  {s.strategy_id}: stop={s.stop_multiplier:.2f}x -> ${lane_dd:,.0f}")

        dd_pct = total_dd / tier.max_dd * 100 if tier.max_dd > 0 else 0
        if total_dd > tier.max_dd:
            raise RuntimeError(
                f"DD BUDGET EXCEEDED: ${total_dd:,.0f} / ${tier.max_dd:,.0f} ({dd_pct:.0f}%) "
                f"for profile '{profile_id}' ({profile.firm} ${profile.account_size:,}).\n"
                + "\n".join(dd_breakdown)
                + "\nReduce lanes or increase account tier. "
                "select_for_profile() would reject lanes 3+."
            )
        logger.info(
            "DD budget: $%.0f / $%.0f (%.0f%%) — %d lanes for %s",
            total_dd,
            tier.max_dd,
            dd_pct,
            len(strategies),
            profile_id,
        )

    logger.info(
        "Profile portfolio '%s': %d strategies for %s (stop=%.2fx)",
        profile_id,
        len(strategies),
        instrument,
        profile.stop_multiplier,
    )
    return Portfolio(
        name=f"profile_{profile_id}",
        instrument=instrument,
        strategies=strategies,
        account_equity=account_equity,
        risk_per_trade_pct=risk_per_trade_pct,
        max_concurrent_positions=max_concurrent_positions,
        max_daily_loss_r=max_daily_loss_r,
    )


# Sessions with bootstrap-verified ML skill at RR2.0 O30 (BH FDR q=0.05, N=7).
# Do NOT add sessions here without bootstrap evidence.
ML_OVERLAY_SESSIONS = {"NYSE_OPEN", "US_DATA_1000", "US_DATA_830"}


def build_multi_rr_portfolio(
    db_path: Path | None = None,
    instrument: str = "MNQ",
    account_equity: float = 25000.0,
    risk_per_trade_pct: float = 2.0,
    max_concurrent_positions: int = 3,
    max_daily_loss_r: float = 5.0,
    stop_multiplier: float = 1.0,
) -> Portfolio:
    """Build a two-layer multi-RR portfolio from orb_outcomes.

    Layer 1: O5 RR1.0 raw baseline (11 sessions, no filter).
    Layer 2: O30 RR2.0 overlay (3 bootstrap-verified sessions).

    NOTE 2026-04-11: The ML overlay that previously filtered Layer 2 was
    removed in the ML V3 sprint Stage 4 — V1/V2/V3 all DEAD per
    docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md.
    Layer 2 now runs unfiltered. If Layer 2 performance degrades without
    ML, reconsider whether the bootstrap-verified session list is still
    valid or whether Layer 2 should be retired.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    # Layer 1: O5 RR1.0 raw baseline (all sessions except NYSE_CLOSE)
    layer1 = build_raw_baseline_portfolio(
        db_path=db_path,
        instrument=instrument,
        rr_target=1.0,
        orb_minutes=5,
        exclude_sessions={"NYSE_CLOSE"},
        account_equity=account_equity,
        risk_per_trade_pct=risk_per_trade_pct,
        max_concurrent_positions=max_concurrent_positions,
        max_daily_loss_r=max_daily_loss_r,
        stop_multiplier=stop_multiplier,
    )

    # Layer 2: O30 RR2.0 ML-filtered overlay (3 sessions only)
    layer2 = build_raw_baseline_portfolio(
        db_path=db_path,
        instrument=instrument,
        rr_target=2.0,
        orb_minutes=30,
        exclude_sessions={"NYSE_CLOSE"},
        include_negative=True,  # ML filter provides the edge
        account_equity=account_equity,
        risk_per_trade_pct=risk_per_trade_pct,
        max_concurrent_positions=max_concurrent_positions,
        max_daily_loss_r=max_daily_loss_r,
        stop_multiplier=stop_multiplier,
    )

    # Filter Layer 2 to only ML_OVERLAY_SESSIONS
    layer2_strategies = [s for s in layer2.strategies if s.orb_label in ML_OVERLAY_SESSIONS]

    # Re-tag Layer 2 strategy IDs to include aperture (avoid future confusion)
    retagged = []
    for s in layer2_strategies:
        new_sid = f"{s.strategy_id}_O{s.orb_minutes}"
        retagged.append(replace(s, strategy_id=new_sid, source="ml_overlay"))
    layer2_strategies = retagged

    all_strategies = list(layer1.strategies) + layer2_strategies
    logger.info(
        "Multi-RR portfolio: %d Layer1 (O5 RR1.0) + %d Layer2 (O30 RR2.0 ML)",
        len(layer1.strategies),
        len(layer2_strategies),
    )

    return Portfolio(
        name=f"multi_rr_{instrument}",
        instrument=instrument,
        strategies=all_strategies,
        account_equity=account_equity,
        risk_per_trade_pct=risk_per_trade_pct,
        max_concurrent_positions=max_concurrent_positions,
        max_daily_loss_r=max_daily_loss_r,
    )


# Minimum overlapping non-NaN days required for a meaningful correlation.
# Pairs below this threshold get NaN correlation (insufficient evidence).
MIN_OVERLAP_DAYS = 200


def build_strategy_daily_series(
    db_path: Path,
    strategy_ids: list[str],
    calendar_overlay: CalendarSkipFilter | None = None,
    atr_velocity_overlay: ATRVelocityFilter | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build per-strategy daily R-series on a shared calendar.

    Supports both baseline (from orb_outcomes) and nested (from nested_outcomes)
    strategies. Queries both validated_setups and nested_validated to determine
    source. Handles mixed orb_minutes by loading daily_features per group.

    For each strategy:
      - Trade day with outcome -> actual pnl_r
      - Eligible day (filter passes) but no trade -> 0.0
      - Ineligible day (filter fails OR calendar skip) -> NaN

    When calendar_overlay is provided, days matching the calendar filter
    (e.g. NFP, OPEX) are marked ineligible to match live execution behavior.

    Returns:
      (series_df, stats): DataFrame indexed by trading_day with one column
      per strategy_id, and dict of {strategy_id: {eligible, traded, padded}}.
    """
    if not strategy_ids:
        return pd.DataFrame(), {}

    with duckdb.connect(str(db_path), read_only=True) as con:
        placeholders = ", ".join(["?"] * len(strategy_ids))

        # Step 1: Load strategy parameters from baseline, nested, and rolling
        baseline_strats = con.execute(
            f"""
            SELECT strategy_id, instrument, orb_label, orb_minutes,
                   entry_model, rr_target, confirm_bars, filter_type,
                   'baseline' as source
            FROM validated_setups
            WHERE strategy_id IN ({placeholders})
        """,
            strategy_ids,
        ).fetchdf()

        # Check if nested/rolling tables exist
        tables = _get_table_names(con)
        nested_strats = pd.DataFrame()
        if "nested_validated" in tables:
            nested_strats = con.execute(
                f"""
                SELECT strategy_id, instrument, orb_label, orb_minutes,
                       entry_model, rr_target, confirm_bars, filter_type,
                       'nested' as source
                FROM nested_validated
                WHERE strategy_id IN ({placeholders})
            """,
                strategy_ids,
            ).fetchdf()

        # Rolling strategies: pick from regime_validated (deduplicate by strategy_id)
        rolling_strats = pd.DataFrame()
        if "regime_validated" in tables:
            rolling_strats = con.execute(
                f"""
                SELECT DISTINCT ON (strategy_id)
                       strategy_id, instrument, orb_label, orb_minutes,
                       entry_model, rr_target, confirm_bars, filter_type,
                       'rolling' as source
                FROM regime_validated
                WHERE strategy_id IN ({placeholders})
                  AND run_label LIKE 'rolling_%'
                ORDER BY strategy_id, run_label DESC
            """,
                strategy_ids,
            ).fetchdf()

        # Combine all sources (deduplicate: baseline wins over rolling)
        strats = pd.concat([baseline_strats, nested_strats, rolling_strats], ignore_index=True)
        strats = strats.drop_duplicates(subset=["strategy_id"], keep="first")
        if strats.empty:
            return pd.DataFrame(), {}

        instrument = strats.iloc[0]["instrument"]

        # Step 2: Load shared calendar per orb_minutes group
        unique_om = sorted(strats["orb_minutes"].unique())
        df_rows_by_om = {}
        all_days = None
        for om in unique_om:
            om_int = int(om)
            # Build dynamic column list from ORB_LABELS
            _size_cols = ", ".join(f"orb_{lbl}_size" for lbl in ORB_LABELS)
            # Compression tier only exists for momentum sessions (canonical: pipeline.build_daily_features.COMPRESSION_SESSIONS)
            _comp_cols = ", ".join(f"orb_{lbl}_compression_tier" for lbl in COMPRESSION_SESSIONS)
            df_om = con.execute(
                f"""
                SELECT trading_day, day_of_week,
                       {_size_cols},
                       is_nfp_day, is_opex_day, is_friday,
                       atr_vel_regime,
                       {_comp_cols}
                FROM daily_features
                WHERE symbol = ? AND orb_minutes = ?
                ORDER BY trading_day
            """,
                [instrument, om_int],
            ).fetchdf()
            if df_om.empty:
                continue
            df_rows_by_om[om_int] = df_om
            if all_days is None:
                all_days = pd.DatetimeIndex(pd.to_datetime(df_om["trading_day"]))

        if all_days is None:
            return pd.DataFrame(), {}

        # Step 3: Load outcomes from both orb_outcomes (baseline) and nested_outcomes (nested)
        outcomes_baseline = con.execute(
            f"""
            SELECT vs.strategy_id, oo.trading_day, oo.pnl_r,
                   oo.mae_r, oo.entry_price, oo.stop_price,
                   COALESCE(vs.stop_multiplier, 1.0) as stop_multiplier
            FROM validated_setups vs
            JOIN orb_outcomes oo
              ON oo.symbol = vs.instrument
              AND oo.orb_label = vs.orb_label
              AND oo.orb_minutes = vs.orb_minutes
              AND oo.entry_model = vs.entry_model
              AND oo.rr_target = vs.rr_target
              AND oo.confirm_bars = vs.confirm_bars
            WHERE vs.strategy_id IN ({placeholders})
              AND oo.pnl_r IS NOT NULL
        """,
            strategy_ids,
        ).fetchdf()

        # Apply tight stop simulation for S075 strategies
        if not outcomes_baseline.empty and "stop_multiplier" in outcomes_baseline.columns:
            sm_strategies = outcomes_baseline[outcomes_baseline["stop_multiplier"] != 1.0]
            if not sm_strategies.empty:
                for sid in sm_strategies["strategy_id"].unique():  # type: ignore[union-attr]
                    mask = outcomes_baseline["strategy_id"] == sid
                    sm = outcomes_baseline.loc[mask, "stop_multiplier"].iloc[0]
                    instr = strats.loc[strats["strategy_id"] == sid, "instrument"].iloc[0]
                    cost_spec = get_cost_spec(instr)
                    rows_as_dicts = outcomes_baseline.loc[mask].to_dict("records")
                    adjusted = apply_tight_stop(rows_as_dicts, sm, cost_spec)
                    outcomes_baseline.loc[mask, "pnl_r"] = [r["pnl_r"] for r in adjusted]
            outcomes_baseline = outcomes_baseline[["strategy_id", "trading_day", "pnl_r"]]

        outcomes_nested = pd.DataFrame()
        if "nested_outcomes" in tables and "nested_validated" in tables:
            outcomes_nested = con.execute(
                f"""
                SELECT nv.strategy_id, no.trading_day, no.pnl_r
                FROM nested_validated nv
                JOIN nested_outcomes no
                  ON no.symbol = nv.instrument
                  AND no.orb_label = nv.orb_label
                  AND no.orb_minutes = nv.orb_minutes
                  AND no.entry_model = nv.entry_model
                  AND no.rr_target = nv.rr_target
                  AND no.confirm_bars = nv.confirm_bars
                  AND no.entry_resolution = nv.entry_resolution
                WHERE nv.strategy_id IN ({placeholders})
                  AND no.pnl_r IS NOT NULL
            """,
                strategy_ids,
            ).fetchdf()

        # Combine outcomes
        outcomes = pd.concat([outcomes_baseline, outcomes_nested], ignore_index=True)

        # Step 4: Build per-strategy series
        series_dict = {}
        stats = {}

        for _, strat in strats.iterrows():
            sid = str(strat["strategy_id"])
            ftype = str(strat["filter_type"])
            orb_label = str(strat["orb_label"])
            strat_om = int(strat["orb_minutes"])

            # Look up daily_features for this strategy's orb_minutes
            df_rows = df_rows_by_om.get(strat_om)
            if df_rows is None:
                continue

            # Determine eligible days from daily_features using filter
            filt = ALL_FILTERS.get(ftype)
            if filt is None:
                continue

            if ftype == "NO_FILTER":
                eligible_mask = np.ones(len(df_rows), dtype=bool)
            else:
                # Pass the full row dict so CompositeFilter (e.g. size + DOW)
                # can access all columns (orb_size, day_of_week, etc.).
                # Convert NaN to None to preserve fail-closed filter semantics.
                eligible_mask = df_rows.apply(
                    lambda row, filt=filt, orb_label=orb_label: filt.matches_row(
                        {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()},
                        orb_label,
                    ),
                    axis=1,
                ).values

            # Apply calendar overlay (NFP/OPEX skip) — marks calendar-event
            # days as ineligible, matching live ExecutionEngine behavior.
            if calendar_overlay is not None:
                for idx in range(len(df_rows)):
                    if eligible_mask[idx]:
                        row_dict = {
                            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                            for k, v in df_rows.iloc[idx].items()
                        }
                        if not calendar_overlay.matches_row(row_dict, orb_label):
                            eligible_mask[idx] = False

            # Apply ATR velocity overlay (Contracting×Neutral/Compressed skip).
            if atr_velocity_overlay is not None:
                for idx in range(len(df_rows)):
                    if eligible_mask[idx]:
                        row_dict = {
                            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                            for k, v in df_rows.iloc[idx].items()
                        }
                        if not atr_velocity_overlay.matches_row(row_dict, orb_label):
                            eligible_mask[idx] = False

            # Start with NaN (ineligible), set eligible days to 0.0
            series = pd.Series(np.nan, index=all_days, name=sid)
            series.iloc[eligible_mask] = 0.0

            # Overlay actual trade returns — ONLY on eligible days (0.0).
            # orb_outcomes has break-days regardless of filter, so we must
            # guard against overwriting NaN (ineligible) with a real pnl_r.
            strat_outcomes = outcomes[outcomes["strategy_id"] == sid]
            trade_days_set = set()
            overlays_skipped_ineligible = 0
            overlays_skipped_missing = 0
            for _, oc in strat_outcomes.iterrows():  # type: ignore[union-attr]
                td = pd.Timestamp(oc["trading_day"])  # type: ignore[arg-type]
                if td not in series.index:
                    overlays_skipped_missing += 1
                elif series.loc[td] == 0.0:
                    series.loc[td] = oc["pnl_r"]
                    trade_days_set.add(td)
                else:
                    # Day is NaN (ineligible for this filter) — skip
                    overlays_skipped_ineligible += 1

            series_dict[sid] = series

            eligible_count = int(eligible_mask.sum())
            traded_count = len(trade_days_set)
            stats[sid] = {
                "eligible_days": eligible_count,
                "traded_days": traded_count,
                "padded_zero_days": eligible_count - traded_count,
                "overlays_skipped_ineligible": overlays_skipped_ineligible,
                "overlays_skipped_missing": overlays_skipped_missing,
            }

        if not series_dict:
            return pd.DataFrame(), {}

        result = pd.DataFrame(series_dict, index=all_days)
        result.index.name = "trading_day"
        return result, stats


def correlation_matrix(
    db_path: Path,
    strategy_ids: list[str],
    min_overlap_days: int = MIN_OVERLAP_DAYS,
    calendar_overlay: CalendarSkipFilter | None = None,
    atr_velocity_overlay: ATRVelocityFilter | None = None,
) -> pd.DataFrame:
    """
    Compute daily R-series correlation between strategies.

    Uses a shared daily calendar with 0.0 for eligible-but-no-trade days.
    Pairs with fewer than min_overlap_days overlapping non-NaN days get
    NaN correlation (insufficient data to compute meaningful correlation).

    Returns symmetric correlation matrix (diagonal = 1.0).
    """
    series_df, stats = build_strategy_daily_series(
        db_path,
        strategy_ids,
        calendar_overlay=calendar_overlay,
        atr_velocity_overlay=atr_velocity_overlay,
    )

    if series_df.empty:
        return pd.DataFrame()

    # Compute pairwise correlation with overlap guard
    cols: list[str] = list(series_df.columns)
    n = len(cols)
    corr = pd.DataFrame(np.nan, index=cols, columns=cols)  # type: ignore[call-overload]

    for i in range(n):
        corr.iloc[i, i] = 1.0
        for j in range(i + 1, n):
            a = series_df[cols[i]]
            b = series_df[cols[j]]
            # Count overlapping non-NaN days
            overlap = a.notna() & b.notna()
            overlap_count = int(overlap.sum())
            if overlap_count >= min_overlap_days:
                r = a[overlap].corr(b[overlap])  # type: ignore[union-attr,arg-type]
                corr.iloc[i, j] = r
                corr.iloc[j, i] = r
            # else: stays NaN (insufficient overlap)

    return corr


# =========================================================================
# Capital estimation
# =========================================================================


def estimate_daily_capital(
    portfolio: Portfolio,
    cost_spec: CostSpec,
) -> dict:
    """
    Estimate daily capital requirements.

    Returns dict with margin estimates, max concurrent risk, worst-case draw.
    """
    if not portfolio.strategies:
        return {
            "estimated_daily_trades": 0,
            "max_concurrent_risk_dollars": 0,
            "worst_case_daily_loss_dollars": 0,
        }

    # Estimate daily trade frequency
    # Each strategy trades when its ORB breaks + filter passes + confirm bars met
    # @research-source: empirical range 0.3-0.5 trades/strategy/day from backtests.
    # 0.4 is midpoint; used only for capital estimation (informational).
    # @revalidated-for: E1/E2 event-based (2026-03-09)
    est_trades_per_day = len(portfolio.strategies) * 0.4

    # Max concurrent risk = max_concurrent * max_risk_per_trade
    # Prefer stored dollar values; fall back to computing from points
    risk_dollars_list = [
        s.median_risk_dollars
        for s in portfolio.strategies
        if s.median_risk_dollars is not None and s.median_risk_dollars > 0
    ]
    if risk_dollars_list:
        risk_per_trade_dollars = sum(risk_dollars_list) / len(risk_dollars_list)
        # Back-compute avg_risk_points for reporting
        avg_risk_points = risk_per_trade_dollars / cost_spec.point_value
    else:
        # Fallback: compute from points (legacy path)
        risk_points_list = [
            s.median_risk_points
            for s in portfolio.strategies
            if s.median_risk_points is not None and s.median_risk_points > 0
        ]
        avg_risk_points = (
            sum(risk_points_list) / len(risk_points_list) if risk_points_list else cost_spec.min_risk_floor_points
        )
        risk_per_trade_dollars = avg_risk_points * cost_spec.point_value
    max_concurrent_risk = portfolio.max_concurrent_positions * risk_per_trade_dollars

    # Worst case daily loss
    worst_case = portfolio.max_daily_loss_r * risk_per_trade_dollars

    return {
        "estimated_daily_trades": round(est_trades_per_day, 1),
        "avg_risk_points": round(avg_risk_points, 2),
        "risk_per_trade_dollars": round(risk_per_trade_dollars, 2),
        "max_concurrent_risk_dollars": round(max_concurrent_risk, 2),
        "worst_case_daily_loss_dollars": round(worst_case, 2),
    }


# =========================================================================
# Fitness-weighted portfolio
# =========================================================================

# Weight multipliers by fitness status
FITNESS_WEIGHTS = {
    "FIT": 1.0,
    "WATCH": 0.5,
    "DECAY": 0.0,
    "STALE": 0.0,
}


def fitness_weighted_portfolio(portfolio: Portfolio, fitness_report) -> Portfolio:
    """
    Adjust portfolio weights based on fitness scores.

    FIT: weight=1.0, WATCH: weight=0.5, DECAY: weight=0.0, STALE: weight=0.0

    Classification weight (CORE=1.0, REGIME=0.5) is preserved as a cap:
    final weight = min(fitness_weight, classification_weight).
    A REGIME strategy with FIT status gets 0.5, not 1.0.

    Returns new Portfolio with adjusted weights. Does NOT modify input.
    fitness_report must have a .scores list with .strategy_id and .fitness_status.
    """
    score_map = {s.strategy_id: s.fitness_status for s in fitness_report.scores}

    adjusted = []
    for strat in portfolio.strategies:
        status = score_map.get(strat.strategy_id, "STALE")
        fitness_weight = FITNESS_WEIGHTS.get(status, 0.0)
        # Classification weight caps fitness weight (REGIME=0.5 cap)
        new_weight = min(fitness_weight, strat.weight)
        adjusted.append(replace(strat, weight=new_weight))

    return Portfolio(
        name=portfolio.name,
        instrument=portfolio.instrument,
        strategies=adjusted,
        account_equity=portfolio.account_equity,
        risk_per_trade_pct=portfolio.risk_per_trade_pct,
        max_concurrent_positions=portfolio.max_concurrent_positions,
        max_daily_loss_r=portfolio.max_daily_loss_r,
        max_per_orb_positions=portfolio.max_per_orb_positions,
        corr_lookup=portfolio.corr_lookup,
    )


# =========================================================================
# CLI
# =========================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build a diversified strategy portfolio from validated setups")
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--max-strategies", type=int, default=20, help="Max strategies")
    parser.add_argument("--min-expr", type=float, default=0.10, help="Min ExpR threshold")
    parser.add_argument("--max-per-orb", type=int, default=5, help="Max strategies per ORB")
    parser.add_argument("--account-equity", type=float, default=25000.0, help="Account equity ($)")
    parser.add_argument("--risk-pct", type=float, default=2.0, help="Risk per trade in percent")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent positions")
    parser.add_argument("--max-daily-loss", type=float, default=5.0, help="Max daily loss (R)")
    parser.add_argument("--include-nested", action="store_true", help="Include nested ORB strategies")
    parser.add_argument("--include-rolling", action="store_true", help="Include rolling-validated strategies")
    parser.add_argument("--rolling-train-months", type=int, default=12, help="Rolling training window months")
    parser.add_argument("--max-correlation", type=float, default=0.85, help="Max pairwise correlation (0-1)")
    parser.add_argument(
        "--family-heads-only", action="store_true", help="Only include family head strategies (1 per edge family)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    portfolio = build_portfolio(
        instrument=args.instrument,
        max_strategies=args.max_strategies,
        min_expectancy_r=args.min_expr,
        max_per_orb=args.max_per_orb,
        account_equity=args.account_equity,
        risk_per_trade_pct=args.risk_pct,
        max_concurrent_positions=args.max_concurrent,
        max_daily_loss_r=args.max_daily_loss,
        max_correlation=args.max_correlation,
        include_nested=args.include_nested,
        include_rolling=args.include_rolling,
        rolling_train_months=args.rolling_train_months,
        family_heads_only=args.family_heads_only,
    )

    summary = portfolio.summary()
    cost_spec = get_cost_spec(args.instrument)
    capital = estimate_daily_capital(portfolio, cost_spec)

    logger.info(f"Portfolio: {portfolio.name}")
    logger.info(f"  Strategies: {summary.get('strategy_count', 0)}")
    if summary.get("strategy_count", 0) > 0:
        logger.info(f"  ORB distribution: {summary['orb_distribution']}")
        logger.info(f"  Entry model distribution: {summary['entry_model_distribution']}")
        logger.info(f"  Avg ExpR: {summary['avg_expectancy_r']:.3f}")
        logger.info(f"  Avg WR: {summary['avg_win_rate']:.1%}")
        logger.info(f"  Capital estimates: {capital}")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(portfolio.to_json())
        logger.info(f"  Written to {output_path}")


if __name__ == "__main__":
    main()
