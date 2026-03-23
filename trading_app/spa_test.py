"""
Hansen's Superior Predictive Ability (SPA) test — institutional-grade strategy validation.

Tests the null hypothesis: "No strategy has superior predictive ability over the benchmark."
Uses the stationary bootstrap (Politis & Romano 1994) with 10,000 replications to build
the null distribution, preserving temporal dependence in the return series.

This is the test the academic literature prescribes for "is my best strategy real?"
It's complementary to the full-pipeline null test (which tests "can my pipeline fake edge?").

References:
    Hansen, P.R. (2005) "A Test for Superior Predictive Ability."
    Journal of Business & Economic Statistics, 23(4), 365-380.

    White, H. (2000) "A Reality Check for Data Snooping."
    Econometrica, 68(5), 1097-1126.

    Sullivan, R., Timmermann, A., White, H. (1999) "Data-Snooping,
    Technical Trading Rule Performance, and the Bootstrap."
    Journal of Finance, 54(5), 1647-1691.

Usage:
    python -m trading_app.spa_test                        # All instruments, 10K reps
    python -m trading_app.spa_test --instrument MGC       # Single instrument
    python -m trading_app.spa_test --reps 1000            # Quick test
    python -m trading_app.spa_test --block-size 20        # Custom block size

Output: p-values (lower/consistent/upper) per instrument. If consistent p-value < 0.05,
reject the null — at least one strategy has statistically significant edge.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH

log = logging.getLogger(__name__)


def build_daily_pnl_matrix(
    db_path: Path,
    instrument: str,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Build the T x K daily P&L matrix for SPA.

    For each strategy: P&L on trade days, 0 on non-trade days.
    Benchmark: zero (no trading = zero cost, zero return).

    Returns:
        benchmark_losses: shape (T,) — all zeros (negated: doing nothing costs nothing)
        model_losses: shape (T, K) — negated daily P&L per strategy
        strategy_ids: list of K strategy IDs (column order)
        trading_days: DatetimeIndex of T trading days

    Convention: SPA uses LOSSES (higher = worse). We negate pnl_r so
    positive loss = negative P&L = bad performance.
    """
    from pipeline.db_config import configure_connection

    con = duckdb.connect(str(db_path), read_only=True)
    configure_connection(con)

    try:
        # Get all active strategies for this instrument
        strategies = con.execute(
            """SELECT strategy_id, orb_label, entry_model, rr_target,
                      confirm_bars, filter_type, orb_minutes
               FROM validated_setups
               WHERE instrument = ?
               AND (status IS NULL OR status NOT IN ('RETIRED', 'PURGED'))
               ORDER BY strategy_id""",
            [instrument],
        ).fetchall()

        if not strategies:
            raise ValueError(f"No active strategies for {instrument}")

        # Get the full trading day range for this instrument
        day_range = con.execute(
            """SELECT DISTINCT trading_day FROM orb_outcomes
               WHERE symbol = ? ORDER BY trading_day""",
            [instrument],
        ).fetchall()
        all_days = pd.DatetimeIndex([r[0] for r in day_range])
        # day_to_idx keyed by datetime.date (matches DuckDB DATE return type)
        day_to_idx = {d.date() if hasattr(d, "date") else d: i for i, d in enumerate(all_days)}
        T = len(all_days)

        log.info(f"  {instrument}: {T} trading days, {len(strategies)} strategies")

        # Build the matrix: T rows x K columns
        K = len(strategies)
        # pnl_matrix[t, k] = pnl_r for strategy k on day t (0 if no trade)
        pnl_matrix = np.zeros((T, K), dtype=np.float64)
        strategy_ids = []

        for k, strat in enumerate(strategies):
            sid = strat[0]
            orb_label = strat[1]
            entry_model = strat[2]
            rr_target = strat[3]
            confirm_bars = strat[4]
            filter_type = strat[5]
            orb_minutes = strat[6]
            strategy_ids.append(sid)

            # Get per-day P&L for this strategy
            # JOIN daily_features for filter application (canonical pattern)
            rows = con.execute(
                """SELECT o.trading_day, o.pnl_r
                   FROM orb_outcomes o
                   JOIN daily_features df
                     ON o.trading_day = df.trading_day
                     AND df.symbol = o.symbol
                     AND df.orb_minutes = o.orb_minutes
                   WHERE o.symbol = ?
                     AND o.orb_label = ?
                     AND o.entry_model = ?
                     AND o.rr_target = ?
                     AND o.confirm_bars = ?
                     AND o.orb_minutes = ?
                     AND o.outcome IN ('win', 'loss')""",
                [instrument, orb_label, entry_model, rr_target, confirm_bars, orb_minutes],
            ).fetchall()

            for day, pnl in rows:
                idx = day_to_idx.get(day)
                if idx is not None:
                    pnl_matrix[idx, k] = pnl

        # SPA convention: losses (higher = worse)
        # Benchmark: zero (no trading)
        benchmark_losses = np.zeros(T, dtype=np.float64)
        model_losses = -pnl_matrix  # negate: positive pnl → negative loss

        return benchmark_losses, model_losses, strategy_ids, all_days

    finally:
        con.close()


def run_spa_test(
    db_path: Path,
    instrument: str,
    reps: int = 10000,
    block_size: int | None = None,
    seed: int = 42,
) -> dict:
    """Run Hansen's SPA test for one instrument.

    Returns dict with:
        instrument, n_strategies, n_days,
        p_lower, p_consistent, p_upper,
        best_strategy_id, best_mean_pnl,
        block_size_used, reps, elapsed_s
    """
    from arch.bootstrap import SPA

    t0 = time.time()

    benchmark_losses, model_losses, strategy_ids, trading_days = build_daily_pnl_matrix(db_path, instrument)

    T, K = model_losses.shape
    if block_size is None:
        block_size = max(int(np.sqrt(T)), 1)

    log.info(f"  Running SPA: T={T}, K={K}, reps={reps}, block_size={block_size}")

    # Run the test
    spa = SPA(
        benchmark_losses,
        model_losses,
        block_size=block_size,
        reps=reps,
        bootstrap="stationary",
        studentize=True,
        seed=seed,
    )
    spa.compute()

    elapsed = time.time() - t0

    # Find best strategy (highest mean daily P&L = lowest mean loss)
    mean_losses = model_losses.mean(axis=0)
    best_idx = int(np.argmin(mean_losses))
    best_sid = strategy_ids[best_idx]
    best_mean_pnl = -float(mean_losses[best_idx])  # un-negate for reporting

    # Which strategies are significantly better than benchmark?
    # better_models returns indices of models with p < threshold
    spa_error = None
    try:
        better_005 = spa.better_models(pvalue=0.05, pvalue_type="consistent")
        n_better = len(better_005) if better_005 is not None else 0
    except Exception as exc:
        n_better = 0
        spa_error = f"{type(exc).__name__}: {exc}"

    return {
        "instrument": instrument,
        "n_strategies": K,
        "n_days": T,
        "p_lower": float(spa.pvalues["lower"]),
        "p_consistent": float(spa.pvalues["consistent"]),
        "p_upper": float(spa.pvalues["upper"]),
        "best_strategy_id": best_sid,
        "best_mean_daily_pnl_r": round(best_mean_pnl, 6),
        "n_better_005": n_better,
        "spa_error": spa_error,
        "block_size_used": block_size,
        "reps": reps,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Hansen's SPA test — is any strategy better than not trading?")
    parser.add_argument("--instrument", type=str, default=None, help="Single instrument (default: all active)")
    parser.add_argument("--reps", type=int, default=10000, help="Bootstrap replications (default: 10000)")
    parser.add_argument("--block-size", type=int, default=None, help="Bootstrap block size (default: sqrt(T))")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--db-path", type=Path, default=None, help="Path to gold.db")
    args = parser.parse_args()

    db_path = args.db_path or GOLD_DB_PATH

    from pipeline.asset_configs import get_active_instruments

    instruments = [args.instrument] if args.instrument else get_active_instruments()

    print("=" * 70)
    print("HANSEN'S SUPERIOR PREDICTIVE ABILITY (SPA) TEST")
    print("=" * 70)
    print(f"  Replications: {args.reps}")
    print(f"  Bootstrap:    stationary (Politis-Romano)")
    print(f"  Block size:   {args.block_size or 'sqrt(T) (auto)'}")
    print(f"  Benchmark:    zero (no trading)")
    print(f"  H0:           No strategy is superior to not trading")
    print(f"  DB:           {db_path}")
    print()

    results = []
    for instrument in instruments:
        print(f"--- {instrument} ---")
        try:
            result = run_spa_test(
                db_path,
                instrument,
                reps=args.reps,
                block_size=args.block_size,
                seed=args.seed,
            )
            results.append(result)

            # Report
            p = result["p_consistent"]
            verdict = "REJECT H0 (edge exists)" if p < 0.05 else "FAIL TO REJECT (no evidence of edge)"
            print(f"  p-value (consistent): {p:.4f} -> {verdict}")
            print(f"  p-value (lower):      {result['p_lower']:.4f}")
            print(f"  p-value (upper):      {result['p_upper']:.4f}")
            print(f"  Best strategy:        {result['best_strategy_id']}")
            print(f"  Best mean daily P&L:  {result['best_mean_daily_pnl_r']:+.6f} R")
            print(f"  Strategies beating benchmark (p<0.05): {result['n_better_005']}")
            print(f"  ({result['n_strategies']} strategies, {result['n_days']} days, {result['elapsed_s']:.0f}s)")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    # Summary
    if results:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for r in results:
            p = r["p_consistent"]
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            print(
                f"  {r['instrument']:4s}: p={p:.4f}{sig:3s} | "
                f"best={r['best_mean_daily_pnl_r']:+.6f}R/day | "
                f"{r['n_better_005']} strategies significant | "
                f"{r['elapsed_s']:.0f}s"
            )
        print()
        print("  *** p<0.01  ** p<0.05  * p<0.10")
        print("  H0: no strategy is superior to not trading")
        print("  Consistent p-value (Hansen 2005 recommended)")


if __name__ == "__main__":
    main()
