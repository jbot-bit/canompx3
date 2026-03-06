"""Sweep ML thresholds on paper trader for a specific instrument.

Tests what threshold maximizes PnL for VALIDATED strategies
(different from the all-outcomes population used for training).

Run: python scripts/tools/ml_threshold_sweep.py --instrument MNQ
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trading_app.ml.config import MODEL_DIR
from trading_app.paper_trader import replay_historical


def main():
    parser = argparse.ArgumentParser(description="ML Threshold Sweep on Paper Trader")
    parser.add_argument("--instrument", type=str, default="MNQ")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--min-thresh", type=float, default=0.40)
    parser.add_argument("--max-thresh", type=float, default=0.60)
    parser.add_argument("--step", type=float, default=0.02)
    args = parser.parse_args()

    # First run baseline (no ML)
    print(f"{'=' * 80}")
    print(f"  ML THRESHOLD SWEEP — {args.instrument} ({args.start} to {args.end})")
    print(f"{'=' * 80}")

    print("\nRunning baseline (no ML)...")
    from datetime import date

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    baseline = replay_historical(instrument=args.instrument, start_date=start, end_date=end, use_ml=False)
    print(
        f"Baseline: {baseline.total_trades} trades, WR={baseline.total_wins / max(baseline.total_trades, 1):.1%}, "
        f"PnL={baseline.total_pnl_r:+.2f}R"
    )

    # Load model to override threshold
    model_path = MODEL_DIR / f"meta_label_{args.instrument}.joblib"
    if not model_path.exists():
        print(f"ERROR: No model at {model_path}")
        return

    bundle = joblib.load(model_path)
    original_threshold = bundle["optimal_threshold"]
    print(f"Model original threshold: {original_threshold:.2f}")

    # Sweep thresholds
    results = []
    thresholds = np.arange(args.min_thresh, args.max_thresh + args.step, args.step)

    try:
        for t in thresholds:
            # Override threshold in the model file
            bundle["optimal_threshold"] = float(t)
            joblib.dump(bundle, model_path)

            result = replay_historical(instrument=args.instrument, start_date=start, end_date=end, use_ml=True)
            results.append(
                {
                    "threshold": t,
                    "trades": result.total_trades,
                    "wins": result.total_wins,
                    "losses": result.total_losses,
                    "wr": result.total_wins / max(result.total_trades, 1),
                    "pnl_r": result.total_pnl_r,
                    "ml_skips": result.total_ml_skips,
                }
            )
            print(
                f"  t={t:.2f}: {result.total_trades} trades, WR={result.total_wins / max(result.total_trades, 1):.1%}, "
                f"PnL={result.total_pnl_r:+.2f}R, ML skips={result.total_ml_skips}"
            )
    finally:
        # Restore original threshold even on crash/interrupt
        bundle["optimal_threshold"] = original_threshold
        joblib.dump(bundle, model_path)

    # Summary
    print(f"\n{'=' * 80}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n{'Thresh':>7} {'Trades':>7} {'Wins':>6} {'WR':>7} {'PnL(R)':>9} {'MLSkip':>7} {'vs Base':>9}")
    print(f"{'-' * 7} {'-' * 7} {'-' * 6} {'-' * 7} {'-' * 9} {'-' * 7} {'-' * 9}")

    best_pnl = -999
    best_t = 0
    for r in results:
        delta = r["pnl_r"] - baseline.total_pnl_r
        marker = ""
        if abs(r["threshold"] - original_threshold) < 0.005:
            marker = " <-- TRAINED"
        if r["pnl_r"] > best_pnl:
            best_pnl = r["pnl_r"]
            best_t = r["threshold"]
        print(
            f"{r['threshold']:>7.2f} {r['trades']:>7} {r['wins']:>6} {r['wr']:>6.1%} "
            f"{r['pnl_r']:>+9.2f} {r['ml_skips']:>7} {delta:>+9.2f}{marker}"
        )

    print(f"\nBaseline (no ML): {baseline.total_trades} trades, PnL={baseline.total_pnl_r:+.2f}R")
    print(f"Best threshold: {best_t:.2f} (PnL={best_pnl:+.2f}R, delta={best_pnl - baseline.total_pnl_r:+.2f}R)")
    print(f"Trained threshold: {original_threshold:.2f}")

    if best_pnl < baseline.total_pnl_r:
        print("\n** NO threshold beats baseline! ML adds no value for validated strategies. **")
    else:
        print(f"\n** Best ML threshold {best_t:.2f} beats baseline by {best_pnl - baseline.total_pnl_r:+.2f}R **")


if __name__ == "__main__":
    main()
