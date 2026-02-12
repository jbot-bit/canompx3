"""
Declarative live portfolio configuration.

Defines exactly what to trade based on rolling evaluation results:
- CORE tier: always-on strategies (STABLE families from rolling eval)
- REGIME tier: conditionally-gated strategies (fitness-checked before trading)

Usage:
    python -m trading_app.live_config --db-path C:/db/gold.db
    python -m trading_app.live_config --db-path C:/db/gold.db --output live_portfolio.json
"""

import sys
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(line_buffering=True)

import duckdb

from pipeline.paths import GOLD_DB_PATH
from trading_app.portfolio import Portfolio, PortfolioStrategy
from trading_app.rolling_portfolio import (
    load_rolling_validated_strategies,
    STABLE_THRESHOLD,
    DEFAULT_LOOKBACK_WINDOWS,
)
from trading_app.strategy_fitness import compute_fitness


# =========================================================================
# Live portfolio specification
# =========================================================================

@dataclass(frozen=True)
class LiveStrategySpec:
    """Declarative specification for a live strategy family."""
    family_id: str          # e.g. "1000_E2_ORB_G2"
    tier: str               # "core" or "regime"
    orb_label: str
    entry_model: str
    filter_type: str
    regime_gate: str | None  # None = always-on, "high_vol" = fitness must be FIT


# The live portfolio: what we actually trade.
#
# TIER 1 (CORE): Always on. Both full-period validated AND present in
#   rolling eval windows. 1000 session, G3 filter -- the smallest ORB size
#   that passes full-period validation at 1000. G3 = ORB >= 3 points.
#
# TIER 2 (REGIME): Gated by strategy_fitness. Full-period validated but
#   TRANSITIONING in rolling eval (score 0.42-0.54). Excellent in high-vol
#   regimes (2025+), negative in low-vol. Only trade when fitness = FIT.
#   This is the volatility regime switch -- turns off when market lacks
#   expansion energy, turns on when ORB sizes indicate trending conditions.
LIVE_PORTFOLIO = [
    # CORE: 1000 session, G3 filter (smallest validated filter for 1000)
    LiveStrategySpec("1000_E1_ORB_G3", "core", "1000", "E1", "ORB_G3", None),
    # REGIME: 0900 session, G4 filter (high-vol overlay)
    LiveStrategySpec("0900_E1_ORB_G4", "regime", "0900", "E1", "ORB_G4", "high_vol"),
]


# =========================================================================
# Portfolio builder
# =========================================================================

def _load_best_regime_variant(
    db_path: Path,
    instrument: str,
    orb_label: str,
    entry_model: str,
    filter_type: str,
) -> dict | None:
    """Load the best RR/CB variant from validated_setups for a regime family."""
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
              AND vs.orb_label = ?
              AND vs.entry_model = ?
              AND vs.filter_type = ?
              AND LOWER(vs.status) = 'active'
            ORDER BY vs.sharpe_ratio DESC NULLS LAST
            LIMIT 1
        """, [instrument, orb_label, entry_model, filter_type]).fetchall()

        if not rows:
            return None

        cols = [desc[0] for desc in con.description]
        return dict(zip(cols, rows[0]))
    finally:
        con.close()


def build_live_portfolio(
    db_path: Path | None = None,
    instrument: str = "MGC",
    rolling_train_months: int = 12,
    account_equity: float = 25000.0,
    risk_per_trade_pct: float = 2.0,
) -> tuple[Portfolio, list[str]]:
    """
    Build the live portfolio from LIVE_PORTFOLIO spec.

    Core tier: loads best variant from rolling eval (most recent window).
    Regime tier: loads best variant from validated_setups, then checks
    strategy_fitness -- weight=0.0 if not FIT.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    strategies = []
    notes = []

    # --- CORE tier: try rolling validated first, fall back to validated_setups ---
    core_specs = [s for s in LIVE_PORTFOLIO if s.tier == "core"]
    if core_specs:
        rolling_strats = load_rolling_validated_strategies(
            db_path, instrument, rolling_train_months,
            min_weighted_score=STABLE_THRESHOLD,
            min_expectancy_r=0.05,
            lookback_windows=DEFAULT_LOOKBACK_WINDOWS,
        )

        for spec in core_specs:
            # Try rolling eval first
            match = None
            source = "rolling"
            for rs in rolling_strats:
                if (rs["orb_label"] == spec.orb_label
                        and rs["entry_model"] == spec.entry_model
                        and rs["filter_type"] == spec.filter_type):
                    match = rs
                    break

            # Fall back to validated_setups (best Sharpe variant)
            if match is None:
                match = _load_best_regime_variant(
                    db_path, instrument,
                    spec.orb_label, spec.entry_model, spec.filter_type,
                )
                source = "baseline"

            if match is None:
                notes.append(f"WARN: {spec.family_id} -- no variant found")
                continue

            strategies.append(PortfolioStrategy(
                strategy_id=match["strategy_id"],
                instrument=match["instrument"],
                orb_label=match["orb_label"],
                entry_model=match["entry_model"],
                rr_target=match["rr_target"],
                confirm_bars=match["confirm_bars"],
                filter_type=match["filter_type"],
                expectancy_r=match["expectancy_r"],
                win_rate=match["win_rate"],
                sample_size=match["sample_size"],
                sharpe_ratio=match.get("sharpe_ratio"),
                max_drawdown_r=match.get("max_drawdown_r"),
                median_risk_points=match.get("median_risk_points"),
                source=source,
                weight=1.0,
            ))
            notes.append(
                f"CORE: {spec.family_id} -> {match['strategy_id']} "
                f"(ExpR={match['expectancy_r']:+.3f}, source={source}, weight=1.0)"
            )

    # --- REGIME tier: from validated_setups + fitness gate ---
    regime_specs = [s for s in LIVE_PORTFOLIO if s.tier == "regime"]
    for spec in regime_specs:
        variant = _load_best_regime_variant(
            db_path, instrument,
            spec.orb_label, spec.entry_model, spec.filter_type,
        )

        if variant is None:
            notes.append(f"WARN: {spec.family_id} -- no validated variant found")
            continue

        # Check fitness gate
        weight = 1.0
        fitness_note = "no gate"
        if spec.regime_gate == "high_vol":
            try:
                fitness = compute_fitness(
                    variant["strategy_id"], db_path=db_path,
                )
                if fitness.fitness_status != "FIT":
                    weight = 0.0
                    fitness_note = f"GATED OFF ({fitness.fitness_status}: {fitness.fitness_notes})"
                else:
                    fitness_note = "FIT -- gate OPEN"
            except (ValueError, duckdb.Error) as e:
                weight = 0.0
                fitness_note = f"GATED OFF (fitness error: {e})"

        strategies.append(PortfolioStrategy(
            strategy_id=variant["strategy_id"],
            instrument=variant["instrument"],
            orb_label=variant["orb_label"],
            entry_model=variant["entry_model"],
            rr_target=variant["rr_target"],
            confirm_bars=variant["confirm_bars"],
            filter_type=variant["filter_type"],
            expectancy_r=variant["expectancy_r"],
            win_rate=variant["win_rate"],
            sample_size=variant["sample_size"],
            sharpe_ratio=variant.get("sharpe_ratio"),
            max_drawdown_r=variant.get("max_drawdown_r"),
            median_risk_points=variant.get("median_risk_points"),
            source="baseline",
            weight=weight,
        ))
        notes.append(
            f"REGIME: {spec.family_id} -> {variant['strategy_id']} "
            f"(ExpR={variant['expectancy_r']:+.3f}, weight={weight}, {fitness_note})"
        )

    return Portfolio(
        name="live",
        instrument=instrument,
        strategies=strategies,
        account_equity=account_equity,
        risk_per_trade_pct=risk_per_trade_pct,
        max_concurrent_positions=3,
        max_daily_loss_r=5.0,
    ), notes


# =========================================================================
# CLI
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build live portfolio from declarative config"
    )
    parser.add_argument("--db-path", type=Path, default=None,
                        help="Path to gold.db")
    parser.add_argument("--instrument", default="MGC")
    parser.add_argument("--rolling-train-months", type=int, default=12)
    parser.add_argument("--output", type=str, default=None,
                        help="Write portfolio JSON to this path")
    args = parser.parse_args()

    db_path = args.db_path or GOLD_DB_PATH

    print("Building live portfolio...")
    print(f"  DB: {db_path}")
    print(f"  Rolling window: {args.rolling_train_months}m")
    print()

    portfolio, notes = build_live_portfolio(
        db_path=db_path,
        instrument=args.instrument,
        rolling_train_months=args.rolling_train_months,
    )

    # Print strategy details
    print(f"{'='*70}")
    print("LIVE PORTFOLIO")
    print(f"{'='*70}")
    for note in notes:
        print(f"  {note}")
    print()

    active = [s for s in portfolio.strategies if s.weight > 0]
    gated = [s for s in portfolio.strategies if s.weight == 0]

    print(f"Active strategies: {len(active)}")
    for s in active:
        print(f"  {s.strategy_id}: {s.source}, weight={s.weight}, "
              f"ExpR={s.expectancy_r:+.3f}, WR={s.win_rate:.1%}, N={s.sample_size}")

    if gated:
        print(f"\nGated OFF (weight=0): {len(gated)}")
        for s in gated:
            print(f"  {s.strategy_id}: regime gate closed")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(portfolio.to_json())
        print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
