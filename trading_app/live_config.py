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
    family_id: str          # e.g. "1000_E1_ORB_G4"
    tier: str               # "core", "regime", or "hot"
    orb_label: str
    entry_model: str
    filter_type: str
    regime_gate: str | None  # None = always-on, "high_vol" = fitness must be FIT
                             # "rolling" = must be STABLE in recent rolling eval

# Lookback for HOT tier rolling stability check (recent months only).
HOT_LOOKBACK_WINDOWS = 10

# Minimum rolling stability score for HOT tier to be active.
HOT_MIN_STABILITY = 0.6

# The live portfolio: what we actually trade.
#
# TIER 1 (CORE): Always on. Full-period validated (75%+ years positive,
#   ROBUST edge family = 5+ parameter-stable members).
#   Specs are instrument-agnostic: run with --instrument MGC/MNQ/MES.
#   If a spec has no match for an instrument, it emits WARN and skips.
#
# TIER 2 (HOT): Rolling-eval gated. Must be STABLE (>=0.6) in recent
#   rolling windows. NOTE: gated off until rolling_portfolio.py is
#   re-run after edge families rebuild (Feb 2026).
#
# TIER 3 (REGIME): Fitness-gated. Full-period validated but regime-dependent.
#   Only trade when strategy_fitness = FIT.
#
# EXIT MODES (see config.py SESSION_EXIT_MODE):
#   0900 = fixed_target
#   1000 = ib_conditional (IB aligned=hold 7h, opposed=kill at market)
#   1800/2300/0030 = fixed_target
#
# Updated 2026-02-21: Added E0 entry model (validated Feb 2026).
#   E0 = limit fill at ORB edge on the confirm bar itself.
#   Top ROBUST CORE families by instrument (head_sharpe_ann):
#     MNQ: 0900 E0 ORB_G5 (Sharpe 2.94, N=305), 1000 E0 ORB_G5 (Sharpe 2.84, N=406)
#     MES: 1000 E0 ORB_G5_L12 (Sharpe 1.54), 1000 E0 ORB_G4_L12 (Sharpe 1.46)
#     MGC: 1000 E0 ORB_G4 (Sharpe 1.35, N=118), 1800 E0 ORB_G4_NOMON (Sharpe 0.43)
LIVE_PORTFOLIO = [
    # --- CORE: always on, full-period validated ROBUST families ---

    # 0900 session (MGC + MNQ dominant; MES secondary)
    # E0 validated Feb 2026 — limit at ORB edge, best fill price
    LiveStrategySpec("0900_E0_ORB_G5", "core", "0900", "E0", "ORB_G5", None),
    LiveStrategySpec("0900_E0_ORB_G4", "core", "0900", "E0", "ORB_G4", None),
    # E1 kept as fallback for instruments where E0 doesn't reach ROBUST threshold
    LiveStrategySpec("0900_E1_ORB_G5", "core", "0900", "E1", "ORB_G5", None),

    # 1000 session (universal — positive for MGC, MNQ, MES)
    LiveStrategySpec("1000_E0_ORB_G5", "core", "1000", "E0", "ORB_G5", None),
    LiveStrategySpec("1000_E0_ORB_G4", "core", "1000", "E0", "ORB_G4", None),
    LiveStrategySpec("1000_E1_ORB_G5", "core", "1000", "E1", "ORB_G5", None),

    # 1800 session (MGC-specific; MNQ marginal)
    LiveStrategySpec("1800_E0_ORB_G4_NOMON", "core", "1800", "E0", "ORB_G4_NOMON", None),

    # --- HOT: rolling-eval gated ---
    # Auto-activates when stability >= HOT_MIN_STABILITY after rolling_portfolio.py re-run.
    # All HOT entries are currently gated off ("family not found in rolling results").
    LiveStrategySpec("0900_E0_ORB_G4_NOFRI", "hot", "0900", "E0", "ORB_G4_NOFRI", "rolling"),
    LiveStrategySpec("1000_E0_ORB_G5_L12", "hot", "1000", "E0", "ORB_G5_L12", "rolling"),

    # --- REGIME: fitness-gated ---
    # (none currently active — add as fitness monitoring matures)
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

def _load_best_experimental_variant(
    db_path: Path,
    instrument: str,
    orb_label: str,
    entry_model: str,
    filter_type: str,
) -> dict | None:
    """Load the best RR/CB variant from experimental_strategies.

    Used for HOT tier families that haven't passed full-period validation
    but are STABLE in recent rolling windows.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = con.execute("""
            SELECT strategy_id, instrument, orb_label, entry_model,
                   rr_target, confirm_bars, filter_type,
                   expectancy_r, win_rate, sample_size,
                   sharpe_ratio, max_drawdown_r,
                   median_risk_points
            FROM experimental_strategies
            WHERE instrument = ?
              AND orb_label = ?
              AND entry_model = ?
              AND filter_type = ?
              AND expectancy_r > 0
            ORDER BY sharpe_ratio DESC NULLS LAST
            LIMIT 1
        """, [instrument, orb_label, entry_model, filter_type]).fetchall()

        if not rows:
            return None

        cols = [desc[0] for desc in con.description]
        return dict(zip(cols, rows[0]))
    finally:
        con.close()

def _check_rolling_stability(
    db_path: Path,
    instrument: str,
    orb_label: str,
    entry_model: str,
    filter_type: str,
    train_months: int = 12,
    lookback_windows: int = HOT_LOOKBACK_WINDOWS,
    min_stability: float = HOT_MIN_STABILITY,
) -> tuple[float, str]:
    """Check rolling stability for a family over recent windows.

    Returns (stability_score, note_string).
    """
    from trading_app.rolling_portfolio import (
        load_all_rolling_run_labels,
        load_rolling_results,
        load_rolling_degraded_counts,
        aggregate_rolling_performance,
        make_family_id,
    )

    all_labels = load_all_rolling_run_labels(
        db_path, train_months, instrument, lookback_windows
    )
    if not all_labels:
        return 0.0, "no rolling windows found"

    validated = load_rolling_results(
        db_path, train_months, instrument, run_labels=all_labels
    )
    degraded = load_rolling_degraded_counts(
        db_path, train_months, instrument, run_labels=all_labels
    )
    families = aggregate_rolling_performance(validated, all_labels, degraded)

    target_fid = make_family_id(orb_label, entry_model, filter_type)
    for fam in families:
        if fam.family_id == target_fid:
            if fam.weighted_stability >= min_stability:
                return fam.weighted_stability, (
                    f"STABLE ({fam.weighted_stability:.3f}, "
                    f"{fam.windows_passed}/{fam.windows_total} windows)"
                )
            else:
                return fam.weighted_stability, (
                    f"NOT STABLE ({fam.weighted_stability:.3f}, "
                    f"{fam.windows_passed}/{fam.windows_total} windows)"
                )

    return 0.0, "family not found in rolling results"

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
    _VALID_INSTRUMENTS = {"MGC", "MNQ", "MES", "M2K"}
    if instrument not in _VALID_INSTRUMENTS:
        raise ValueError(f"Unknown instrument '{instrument}'. Valid: {sorted(_VALID_INSTRUMENTS)}")

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

    # --- HOT tier: from experimental_strategies + rolling stability gate ---
    hot_specs = [s for s in LIVE_PORTFOLIO if s.tier == "hot"]
    for spec in hot_specs:
        variant = _load_best_experimental_variant(
            db_path, instrument,
            spec.orb_label, spec.entry_model, spec.filter_type,
        )

        if variant is None:
            notes.append(f"WARN: {spec.family_id} -- no experimental variant with positive ExpR")
            continue

        # Check rolling stability gate
        weight = 1.0
        stability_note = "no gate"
        if spec.regime_gate == "rolling":
            stability_score, stability_note = _check_rolling_stability(
                db_path, instrument,
                spec.orb_label, spec.entry_model, spec.filter_type,
                train_months=rolling_train_months,
            )
            if stability_score < HOT_MIN_STABILITY:
                weight = 0.0
                stability_note = f"GATED OFF ({stability_note})"
            else:
                stability_note = f"ACTIVE ({stability_note})"

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
            source="hot_rolling",
            weight=weight,
        ))
        notes.append(
            f"HOT: {spec.family_id} -> {variant['strategy_id']} "
            f"(ExpR={variant['expectancy_r']:+.3f}, weight={weight}, {stability_note})"
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
    parser.add_argument("--instrument", default="MGC",
                        choices=["MGC", "MNQ", "MES", "M2K"])
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
