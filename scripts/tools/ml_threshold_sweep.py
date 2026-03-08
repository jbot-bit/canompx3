"""Sweep ML thresholds on paper trader for a specific instrument.

Tests what threshold maximizes PnL for VALIDATED strategies
(different from the all-outcomes population used for training).

Run: python scripts/tools/ml_threshold_sweep.py --instrument MNQ
"""

import argparse
import shutil
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

    # Load model to override threshold (hybrid -> legacy fallback)
    hybrid_path = MODEL_DIR / f"meta_label_{args.instrument}_hybrid.joblib"
    legacy_path = MODEL_DIR / f"meta_label_{args.instrument}.joblib"
    model_path = hybrid_path if hybrid_path.exists() else legacy_path
    if not model_path.exists():
        print(f"ERROR: No model at {model_path}")
        return

    bundle = joblib.load(model_path)
    is_hybrid = "sessions" in bundle

    if is_hybrid:
        # Collect original per-session thresholds for restore
        original_thresholds = {}
        is_per_aperture = bundle.get("bundle_format") == "per_aperture"
        for session, val in bundle.get("sessions", {}).items():
            if is_per_aperture:
                original_thresholds[session] = {
                    ak: info.get("optimal_threshold")
                    for ak, info in val.items()
                    if isinstance(info, dict) and info.get("model") is not None
                }
            else:
                if isinstance(val, dict) and val.get("model") is not None:
                    original_thresholds[session] = val.get("optimal_threshold")
        # Use median of original thresholds for display
        all_orig = [
            v
            for v in (
                original_thresholds.values()
                if not is_per_aperture
                else [t for d in original_thresholds.values() for t in d.values() if t is not None]
            )
            if v is not None
        ]
        original_threshold = float(np.median(all_orig)) if all_orig else 0.5
    else:
        original_threshold = bundle["optimal_threshold"]

    print(f"Model original threshold: {original_threshold:.2f} ({'hybrid' if is_hybrid else 'legacy'})")

    # Sweep thresholds
    results = []
    thresholds = np.arange(args.min_thresh, args.max_thresh + args.step, args.step)

    # Safety: backup original model file before mutating thresholds.
    # If anything goes wrong during sweep, we can restore from backup.
    backup_path = model_path.with_suffix(".joblib.bak")
    shutil.copy2(model_path, backup_path)

    try:
        for t in thresholds:
            # Override threshold(s) in the model file
            if is_hybrid:
                for _session, val in bundle.get("sessions", {}).items():
                    if is_per_aperture:
                        for _ak, info in val.items():
                            if isinstance(info, dict) and info.get("model") is not None:
                                info["optimal_threshold"] = float(t)
                    else:
                        if isinstance(val, dict) and val.get("model") is not None:
                            val["optimal_threshold"] = float(t)
            else:
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
        # Restore original model from backup (atomic: copy is always clean)
        if backup_path.exists():
            shutil.copy2(backup_path, model_path)
            backup_path.unlink()
        else:
            # Fallback: reconstruct from in-memory thresholds
            if is_hybrid:
                for session, val in bundle.get("sessions", {}).items():
                    if is_per_aperture:
                        for ak, info in val.items():
                            if isinstance(info, dict) and info.get("model") is not None:
                                orig = original_thresholds.get(session, {}).get(ak)
                                if orig is not None:
                                    info["optimal_threshold"] = orig
                    else:
                        if isinstance(val, dict) and val.get("model") is not None:
                            orig = original_thresholds.get(session)
                            if orig is not None:
                                val["optimal_threshold"] = orig
            else:
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
