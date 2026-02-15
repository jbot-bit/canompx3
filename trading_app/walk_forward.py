"""
Walk-forward out-of-sample evaluation for validated strategies.

Splits the trading day timeline into (train, test) folds and evaluates
strategy metrics on TEST days only. No peeking across folds.

Usage:
    python trading_app/walk_forward.py --instrument MGC --walk-forward --train-years 3 --test-years 1
    python trading_app/walk_forward.py --instrument MGC --walk-forward --train-years 3 --test-years 1 --min-oos-trades 50
"""

import sys
import csv
import json
from pathlib import Path
from datetime import date
from dataclasses import dataclass, field, asdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.paths import GOLD_DB_PATH
from pipeline.init_db import ORB_LABELS
from trading_app.config import ALL_FILTERS, VolumeFilter
from trading_app.strategy_discovery import compute_metrics

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)


@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""
    fold_idx: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_days: int
    test_days: int
    trade_count: int
    win_rate: float | None
    expectancy_r: float | None
    sharpe_ratio: float | None
    max_drawdown_r: float | None


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward evaluation for a single strategy."""
    strategy_id: str
    filter_type: str
    folds: list[FoldResult] = field(default_factory=list)
    oos_trade_count: int = 0
    oos_win_rate: float | None = None
    oos_expectancy_r: float | None = None
    oos_sharpe_ratio: float | None = None
    oos_max_drawdown_r: float | None = None


def build_folds(
    trading_days: list[date],
    train_years: int = 3,
    test_years: int = 1,
) -> list[tuple[list[date], list[date]]]:
    """
    Partition trading days into (train, test) fold pairs.

    Folds slide forward by test_years. A test fold is only included if
    it has at least 100 trading days (roughly half a year).

    Returns list of (train_days, test_days) tuples.
    """
    if not trading_days:
        return []

    sorted_days = sorted(trading_days)
    min_year = sorted_days[0].year
    max_year = sorted_days[-1].year

    folds = []
    test_start_year = min_year + train_years

    while test_start_year + test_years - 1 <= max_year:
        train_start_year = test_start_year - train_years
        test_end_year = test_start_year + test_years - 1

        train = [d for d in sorted_days if train_start_year <= d.year < test_start_year]
        test = [d for d in sorted_days if test_start_year <= d.year <= test_end_year]

        # Only include folds with meaningful test data (>= 100 days)
        if len(test) >= 100:
            folds.append((train, test))

        test_start_year += test_years

    return folds


def evaluate_fold(
    outcomes: list[dict],
    eligible_days: set[date],
    test_days: set[date],
) -> dict:
    """
    Compute metrics for a strategy on test-fold days only.

    Filters outcomes to test days that are also eligible, then runs
    compute_metrics on the filtered set.
    """
    test_outcomes = [
        o for o in outcomes
        if o["trading_day"] in test_days and o["trading_day"] in eligible_days
    ]
    return compute_metrics(test_outcomes)


def walk_forward_eval(
    db_path: Path | None = None,
    instrument: str = "MGC",
    orb_minutes: int = 5,
    train_years: int = 3,
    test_years: int = 1,
    min_oos_trades: int = 100,
) -> list[WalkForwardResult]:
    """
    Run walk-forward OOS evaluation for all validated strategies.

    Volume-filter strategies (VOL_*) are excluded because their rolling
    baseline could leak across fold boundaries.

    Returns list of WalkForwardResult, one per strategy.
    """
    if db_path is None:
        db_path = GOLD_DB_PATH

    con = duckdb.connect(str(db_path), read_only=True)

    try:
        # Load all trading days
        all_days = [
            r[0] for r in con.execute(
                "SELECT DISTINCT trading_day FROM daily_features "
                "WHERE symbol = ? AND orb_minutes = ? ORDER BY trading_day",
                [instrument, orb_minutes],
            ).fetchall()
        ]
        print(f"Loaded {len(all_days)} trading days ({all_days[0]} to {all_days[-1]})")

        # Build folds
        folds = build_folds(all_days, train_years, test_years)
        print(f"Built {len(folds)} walk-forward folds (train={train_years}y, test={test_years}y)")
        for i, (train, test) in enumerate(folds):
            print(f"  Fold {i+1}: Train {train[0]}-{train[-1]} ({len(train)}d), "
                  f"Test {test[0]}-{test[-1]} ({len(test)}d)")

        if not folds:
            print("ERROR: No valid folds. Need more data or smaller windows.")
            return []

        # Load daily_features for eligibility
        features = con.execute(
            "SELECT * FROM daily_features WHERE symbol = ? AND orb_minutes = ? ORDER BY trading_day",
            [instrument, orb_minutes],
        ).fetchall()
        feat_cols = [desc[0] for desc in con.description]
        feat_dicts = [dict(zip(feat_cols, r)) for r in features]

        # Build filter -> eligible days (excluding volume filters)
        non_vol_filters = {k: v for k, v in ALL_FILTERS.items() if not isinstance(v, VolumeFilter)}
        filter_day_sets = {}
        for filter_key, strategy_filter in non_vol_filters.items():
            for orb_label in ORB_LABELS:
                days = set()
                for row in feat_dicts:
                    if row.get(f"orb_{orb_label}_break_dir") is None:
                        continue
                    if strategy_filter.matches_row(row, orb_label):
                        days.add(row["trading_day"])
                filter_day_sets[(filter_key, orb_label)] = days

        # Load validated strategies (excluding volume filter strategies)
        strategies = con.execute(
            "SELECT strategy_id, instrument, orb_label, orb_minutes, "
            "rr_target, confirm_bars, entry_model, filter_type "
            "FROM validated_setups WHERE instrument = ? AND status = 'active'",
            [instrument],
        ).fetchall()
        strat_cols = ["strategy_id", "instrument", "orb_label", "orb_minutes",
                      "rr_target", "confirm_bars", "entry_model", "filter_type"]
        strat_dicts = [dict(zip(strat_cols, r)) for r in strategies]

        # Filter out volume strategies
        vol_skipped = [s for s in strat_dicts if s["filter_type"].startswith("VOL_")]
        strat_dicts = [s for s in strat_dicts if not s["filter_type"].startswith("VOL_")]
        if vol_skipped:
            print(f"Skipping {len(vol_skipped)} volume-filter strategies (fold-boundary leakage)")
        print(f"Evaluating {len(strat_dicts)} strategies across {len(folds)} folds")

        # Load all outcomes (bulk)
        all_outcomes = con.execute(
            "SELECT trading_day, symbol, orb_label, orb_minutes, rr_target, "
            "confirm_bars, entry_model, outcome, pnl_r, mae_r, mfe_r, "
            "entry_price, stop_price "
            "FROM orb_outcomes WHERE symbol = ? AND orb_minutes = ? "
            "AND outcome IS NOT NULL ORDER BY trading_day",
            [instrument, orb_minutes],
        ).fetchall()
        outcome_cols = ["trading_day", "symbol", "orb_label", "orb_minutes",
                        "rr_target", "confirm_bars", "entry_model", "outcome",
                        "pnl_r", "mae_r", "mfe_r", "entry_price", "stop_price"]

        # Index outcomes by (orb_label, entry_model, rr_target, confirm_bars)
        from collections import defaultdict
        outcome_index = defaultdict(list)
        for r in all_outcomes:
            rd = dict(zip(outcome_cols, r))
            key = (rd["orb_label"], rd["entry_model"], rd["rr_target"], rd["confirm_bars"])
            outcome_index[key].append(rd)

        # Evaluate each strategy
        results = []
        for si, strat in enumerate(strat_dicts):
            sid = strat["strategy_id"]
            orb = strat["orb_label"]
            em = strat["entry_model"]
            rr = strat["rr_target"]
            cb = strat["confirm_bars"]
            ft = strat["filter_type"]

            outcome_key = (orb, em, rr, cb)
            strat_outcomes = outcome_index.get(outcome_key, [])
            eligible = filter_day_sets.get((ft, orb), set())

            wf_result = WalkForwardResult(strategy_id=sid, filter_type=ft)
            all_oos_outcomes = []

            for fi, (train_days, test_days) in enumerate(folds):
                test_set = set(test_days)
                train_set = set(train_days)

                # Assertion: no leakage
                assert not (train_set & test_set), "LEAKAGE: train/test overlap!"

                metrics = evaluate_fold(strat_outcomes, eligible, test_set)

                fold_r = FoldResult(
                    fold_idx=fi + 1,
                    train_start=train_days[0],
                    train_end=train_days[-1],
                    test_start=test_days[0],
                    test_end=test_days[-1],
                    train_days=len(train_days),
                    test_days=len(test_days),
                    trade_count=metrics["sample_size"],
                    win_rate=metrics["win_rate"],
                    expectancy_r=metrics["expectancy_r"],
                    sharpe_ratio=metrics["sharpe_ratio"],
                    max_drawdown_r=metrics["max_drawdown_r"],
                )
                wf_result.folds.append(fold_r)

                # Collect OOS outcomes for aggregation
                oos = [
                    o for o in strat_outcomes
                    if o["trading_day"] in test_set and o["trading_day"] in eligible
                ]
                all_oos_outcomes.extend(oos)

            # Aggregate OOS metrics
            agg = compute_metrics(all_oos_outcomes)
            wf_result.oos_trade_count = agg["sample_size"]
            wf_result.oos_win_rate = agg["win_rate"]
            wf_result.oos_expectancy_r = agg["expectancy_r"]
            wf_result.oos_sharpe_ratio = agg["sharpe_ratio"]
            wf_result.oos_max_drawdown_r = agg["max_drawdown_r"]

            results.append(wf_result)

            if (si + 1) % 50 == 0:
                print(f"  Evaluated {si + 1}/{len(strat_dicts)} strategies")

        print(f"Done: {len(results)} strategies evaluated")

        # Summary
        passed = [r for r in results if r.oos_trade_count >= min_oos_trades
                  and r.oos_expectancy_r is not None and r.oos_expectancy_r > 0]
        print(f"OOS positive ExpR with >= {min_oos_trades} trades: {len(passed)}/{len(results)}")

        return results

    finally:
        con.close()


def write_artifacts(results: list[WalkForwardResult], output_dir: Path):
    """Write walk-forward results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_path = output_dir / "walk_forward_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy_id", "filter_type", "num_folds",
            "oos_trade_count", "oos_win_rate", "oos_expectancy_r",
            "oos_sharpe_ratio", "oos_max_drawdown_r",
        ])
        for r in results:
            writer.writerow([
                r.strategy_id, r.filter_type, len(r.folds),
                r.oos_trade_count, r.oos_win_rate, r.oos_expectancy_r,
                r.oos_sharpe_ratio, r.oos_max_drawdown_r,
            ])
    print(f"Written: {summary_path}")

    # Per-fold CSV
    folds_path = output_dir / "walk_forward_folds.csv"
    with open(folds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy_id", "fold_idx", "train_start", "train_end",
            "test_start", "test_end", "train_days", "test_days",
            "trade_count", "win_rate", "expectancy_r",
            "sharpe_ratio", "max_drawdown_r",
        ])
        for r in results:
            for fold in r.folds:
                writer.writerow([
                    r.strategy_id, fold.fold_idx, fold.train_start, fold.train_end,
                    fold.test_start, fold.test_end, fold.train_days, fold.test_days,
                    fold.trade_count, fold.win_rate, fold.expectancy_r,
                    fold.sharpe_ratio, fold.max_drawdown_r,
                ])
    print(f"Written: {folds_path}")

    # JSON (machine-readable)
    json_path = output_dir / "walk_forward_results.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    print(f"Written: {json_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk-forward OOS evaluation for validated strategies"
    )
    parser.add_argument("--instrument", default="MGC", help="Instrument symbol")
    parser.add_argument("--walk-forward", action="store_true", required=True,
                        help="Run walk-forward evaluation")
    parser.add_argument("--train-years", type=int, default=3,
                        help="Training window in years (default: 3)")
    parser.add_argument("--test-years", type=int, default=1,
                        help="Test window in years (default: 1)")
    parser.add_argument("--min-oos-trades", type=int, default=100,
                        help="Min OOS trades to consider strategy valid (default: 100)")
    parser.add_argument("--orb-minutes", type=int, default=5,
                        help="ORB duration in minutes (default: 5)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: artifacts/walk_forward/)")
    args = parser.parse_args()

    results = walk_forward_eval(
        instrument=args.instrument,
        orb_minutes=args.orb_minutes,
        train_years=args.train_years,
        test_years=args.test_years,
        min_oos_trades=args.min_oos_trades,
    )

    if results:
        output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "artifacts" / "walk_forward"
        write_artifacts(results, output_dir)


if __name__ == "__main__":
    main()
