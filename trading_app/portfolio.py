"""
Portfolio construction from validated strategies.

Selects a diversified subset of validated_setups, computes position sizes,
and estimates capital requirements.

Usage:
    python trading_app/portfolio.py --instrument MGC
    python trading_app/portfolio.py --instrument MGC --max-strategies 10 --risk-pct 2.0
    python trading_app/portfolio.py --instrument MGC --account-equity 25000
"""

import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(line_buffering=True)

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from pipeline.cost_model import get_cost_spec, CostSpec
from trading_app.config import ALL_FILTERS, classify_strategy


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
        return json.dumps(data, indent=2)

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


# =========================================================================
# Portfolio construction
# =========================================================================

def load_validated_strategies(
    db_path: Path,
    instrument: str,
    min_expectancy_r: float = 0.10,
) -> list[dict]:
    """Load validated strategies from DB, filtered by minimum ExpR."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT vs.strategy_id, vs.instrument, vs.orb_label, vs.entry_model,
                   vs.rr_target, vs.confirm_bars, vs.filter_type,
                   vs.expectancy_r, vs.win_rate, vs.sample_size,
                   vs.sharpe_ratio, vs.max_drawdown_r,
                   es.median_risk_points
            FROM validated_setups vs
            LEFT JOIN experimental_strategies es
              ON vs.strategy_id = es.strategy_id
            WHERE vs.instrument = ?
              AND vs.status = 'ACTIVE'
              AND vs.expectancy_r >= ?
            ORDER BY vs.expectancy_r DESC
        """, [instrument, min_expectancy_r]).fetchall()

        cols = [desc[0] for desc in con.description]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        con.close()


def diversify_strategies(
    candidates: list[dict],
    max_strategies: int,
    max_per_orb: int = 5,
    max_per_entry_model: int | None = None,
) -> list[dict]:
    """
    Select a diversified subset of strategies.

    Selection priority:
    1. Highest ExpR first
    2. Enforce max per ORB label
    3. Enforce max per entry model (if set)
    4. Stop at max_strategies
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
) -> Portfolio:
    """
    Build a diversified portfolio from validated strategies.

    Returns Portfolio with selected strategies and risk parameters.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    # Load and filter candidates
    candidates = load_validated_strategies(db_path, instrument, min_expectancy_r)

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

    # Diversify selection
    selected = diversify_strategies(candidates, max_strategies, max_per_orb)

    # Convert to PortfolioStrategy objects
    strategies = []
    for s in selected:
        strategies.append(PortfolioStrategy(
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
        ))

    return Portfolio(
        name=name,
        instrument=instrument,
        strategies=strategies,
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
) -> tuple[pd.DataFrame, dict]:
    """
    Build per-strategy daily R-series on a shared calendar.

    For each strategy:
      - Trade day with outcome -> actual pnl_r
      - Eligible day (filter passes) but no trade -> 0.0
      - Ineligible day (filter fails) -> NaN

    Returns:
      (series_df, stats): DataFrame indexed by trading_day with one column
      per strategy_id, and dict of {strategy_id: {eligible, traded, padded}}.
    """
    if not strategy_ids:
        return pd.DataFrame(), {}

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        placeholders = ", ".join(["?"] * len(strategy_ids))

        # Step 1: Load strategy parameters
        strats = con.execute(f"""
            SELECT strategy_id, instrument, orb_label, orb_minutes,
                   entry_model, rr_target, confirm_bars, filter_type
            FROM validated_setups
            WHERE strategy_id IN ({placeholders})
        """, strategy_ids).fetchdf()

        if strats.empty:
            return pd.DataFrame(), {}

        instrument = strats.iloc[0]["instrument"]
        orb_minutes = int(strats.iloc[0]["orb_minutes"])

        # Step 2: Load shared calendar (all daily_features rows for instrument)
        df_rows = con.execute("""
            SELECT trading_day, orb_0900_size, orb_1000_size, orb_1100_size,
                   orb_1800_size, orb_2300_size, orb_0030_size
            FROM daily_features
            WHERE symbol = ? AND orb_minutes = ?
            ORDER BY trading_day
        """, [instrument, orb_minutes]).fetchdf()

        if df_rows.empty:
            return pd.DataFrame(), {}

        all_days = pd.DatetimeIndex(pd.to_datetime(df_rows["trading_day"]))

        # Step 3: Load all relevant outcomes in one query
        outcomes = con.execute(f"""
            SELECT vs.strategy_id, oo.trading_day, oo.pnl_r
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
        """, strategy_ids).fetchdf()

        # Step 4: Build per-strategy series
        series_dict = {}
        stats = {}

        for _, strat in strats.iterrows():
            sid = strat["strategy_id"]
            ftype = strat["filter_type"]
            orb_label = strat["orb_label"]

            # Determine eligible days from daily_features using filter
            filt = ALL_FILTERS.get(ftype)
            if filt is None:
                continue

            size_col = f"orb_{orb_label}_size"
            if ftype == "NO_FILTER":
                eligible_mask = np.ones(len(df_rows), dtype=bool)
            else:
                sizes = df_rows[size_col].values
                eligible_mask = np.array([
                    filt.matches_row({size_col: s}, orb_label)
                    if s is not None and not (isinstance(s, float) and np.isnan(s))
                    else False
                    for s in sizes
                ])

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
            for _, oc in strat_outcomes.iterrows():
                td = pd.Timestamp(oc["trading_day"])
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

    finally:
        con.close()


def correlation_matrix(
    db_path: Path,
    strategy_ids: list[str],
    min_overlap_days: int = MIN_OVERLAP_DAYS,
) -> pd.DataFrame:
    """
    Compute daily R-series correlation between strategies.

    Uses a shared daily calendar with 0.0 for eligible-but-no-trade days.
    Pairs with fewer than min_overlap_days overlapping non-NaN days get
    NaN correlation (insufficient data to compute meaningful correlation).

    Returns symmetric correlation matrix (diagonal = 1.0).
    """
    series_df, stats = build_strategy_daily_series(db_path, strategy_ids)

    if series_df.empty:
        return pd.DataFrame()

    # Compute pairwise correlation with overlap guard
    cols = list(series_df.columns)
    n = len(cols)
    corr = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i in range(n):
        corr.iloc[i, i] = 1.0
        for j in range(i + 1, n):
            a = series_df[cols[i]]
            b = series_df[cols[j]]
            # Count overlapping non-NaN days
            overlap = a.notna() & b.notna()
            overlap_count = int(overlap.sum())
            if overlap_count >= min_overlap_days:
                r = a[overlap].corr(b[overlap])
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
    # Rough estimate: ~0.3-0.5 trades per strategy per day
    est_trades_per_day = len(portfolio.strategies) * 0.4

    # Max concurrent risk = max_concurrent * max_risk_per_trade
    risk_points_list = [
        s.median_risk_points for s in portfolio.strategies
        if s.median_risk_points is not None and s.median_risk_points > 0
    ]
    avg_risk_points = (
        sum(risk_points_list) / len(risk_points_list)
        if risk_points_list
        else cost_spec.min_risk_floor_points
    )
    risk_per_trade_dollars = avg_risk_points * cost_spec.point_value + cost_spec.total_friction
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
# CLI
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a diversified strategy portfolio from validated setups"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--max-strategies", type=int, default=20, help="Max strategies")
    parser.add_argument("--min-expr", type=float, default=0.10, help="Min ExpR threshold")
    parser.add_argument("--max-per-orb", type=int, default=5, help="Max strategies per ORB")
    parser.add_argument("--account-equity", type=float, default=25000.0, help="Account equity ($)")
    parser.add_argument("--risk-pct", type=float, default=2.0, help="Risk per trade in percent")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent positions")
    parser.add_argument("--max-daily-loss", type=float, default=5.0, help="Max daily loss (R)")
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
    )

    summary = portfolio.summary()
    cost_spec = get_cost_spec(args.instrument)
    capital = estimate_daily_capital(portfolio, cost_spec)

    print(f"Portfolio: {portfolio.name}")
    print(f"  Strategies: {summary.get('strategy_count', 0)}")
    if summary.get("strategy_count", 0) > 0:
        print(f"  ORB distribution: {summary['orb_distribution']}")
        print(f"  Entry model distribution: {summary['entry_model_distribution']}")
        print(f"  Avg ExpR: {summary['avg_expectancy_r']:.3f}")
        print(f"  Avg WR: {summary['avg_win_rate']:.1%}")
        print(f"  Capital estimates: {capital}")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(portfolio.to_json())
        print(f"  Written to {output_path}")


if __name__ == "__main__":
    main()
