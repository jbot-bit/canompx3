"""Before/after quantification of meta-label impact.

Loads a trained model, generates predictions on full dataset,
and compares baseline vs filtered performance across multiple cuts.

Usage:
    python -m trading_app.ml.evaluate --instrument MGC
    python -m trading_app.ml.evaluate --all
"""

from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd

from pipeline.paths import GOLD_DB_PATH
from trading_app.ml.config import ACTIVE_INSTRUMENTS, MODEL_DIR
from trading_app.ml.features import load_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _sharpe(pnl: pd.Series, annual_factor: float = 252) -> float:
    """Annualized Sharpe ratio."""
    if pnl.std() == 0 or len(pnl) < 2:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(annual_factor))


def _max_drawdown_r(pnl: pd.Series) -> float:
    """Maximum drawdown in R-units."""
    cumulative = pnl.cumsum()
    peak = cumulative.cummax()
    dd = cumulative - peak
    return float(dd.min()) if len(dd) > 0 else 0.0


def _fill_missing_features(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Align X to expected feature columns, using 0.0 for one-hot and -999.0 for numeric."""
    missing = set(feature_names) - set(X.columns)
    for col in missing:
        # One-hot encoded columns (contain category prefix) should be 0, not -999
        is_onehot = any(col.startswith(f"{cat}_") for cat in [
            "orb_label", "entry_model", "prev_day_direction",
            "gap_type", "atr_vel_regime", "break_dir",
        ])
        X[col] = 0.0 if is_onehot else -999.0
    return X[feature_names]


def evaluate_instrument(instrument: str, db_path: str) -> dict:
    """Full before/after evaluation for one instrument.

    WARNING: Metrics blend in-sample (80% train) and out-of-sample (20% test)
    data. For unbiased assessment, use evaluate_validated.py which tests only
    on validated strategies, or check the OOS-only section below.
    """
    model_path = MODEL_DIR / f"meta_label_{instrument}.joblib"
    if not model_path.exists():
        logger.error(f"No trained model for {instrument}. Run meta_label.py first.")
        return {"status": "no_model"}

    # Load model
    bundle = joblib.load(model_path)
    rf = bundle["model"]
    threshold = bundle["optimal_threshold"]
    feature_names = bundle["feature_names"]
    n_train = bundle.get("n_train", 0)

    # Load data
    X, y, meta = load_feature_matrix(db_path, instrument)

    # Align features
    X = _fill_missing_features(X, feature_names)

    # Predict
    y_prob = rf.predict_proba(X)[:, 1]
    meta = meta.copy()
    meta["p_win"] = y_prob
    meta["take"] = y_prob >= threshold
    meta["is_oos"] = False
    if n_train > 0 and n_train < len(meta):
        meta.iloc[n_train:, meta.columns.get_loc("is_oos")] = True

    pnl_all = meta["pnl_r"]
    pnl_kept = meta.loc[meta["take"], "pnl_r"]

    results = {
        "instrument": instrument,
        "threshold": threshold,
        "n_total": len(meta),
        "n_kept": meta["take"].sum(),
        "n_skipped": (~meta["take"]).sum(),
        "skip_pct": float(1 - meta["take"].mean()),
        "baseline": {
            "avg_r": float(pnl_all.mean()),
            "total_r": float(pnl_all.sum()),
            "sharpe": _sharpe(pnl_all),
            "max_dd": _max_drawdown_r(pnl_all),
            "wr": float((pnl_all > 0).mean()),
        },
        "filtered": {
            "avg_r": float(pnl_kept.mean()),
            "total_r": float(pnl_kept.sum()),
            "sharpe": _sharpe(pnl_kept),
            "max_dd": _max_drawdown_r(pnl_kept),
            "wr": float((pnl_kept > 0).mean()),
        },
    }

    # Skipped trades analysis
    pnl_skipped = meta.loc[~meta["take"], "pnl_r"]
    results["skipped"] = {
        "avg_r": float(pnl_skipped.mean()) if len(pnl_skipped) > 0 else 0,
        "wr": float((pnl_skipped > 0).mean()) if len(pnl_skipped) > 0 else 0,
    }

    # Per-session breakdown
    sessions = []
    for session in sorted(meta["orb_label"].unique()):
        sm = meta[meta["orb_label"] == session]
        if len(sm) < 30:
            continue
        kept = sm[sm["take"]]
        skipped = sm[~sm["take"]]
        sessions.append({
            "session": session,
            "n_total": len(sm),
            "n_kept": len(kept),
            "skip_pct": round(1 - len(kept) / len(sm), 3),
            "base_avgR": round(sm["pnl_r"].mean(), 4),
            "filt_avgR": round(kept["pnl_r"].mean(), 4) if len(kept) > 0 else 0,
            "skip_avgR": round(skipped["pnl_r"].mean(), 4) if len(skipped) > 0 else 0,
            "lift": round(kept["pnl_r"].mean() - sm["pnl_r"].mean(), 4) if len(kept) > 0 else 0,
        })
    results["sessions"] = sessions

    # Probability calibration check (lift curve)
    meta_sorted = meta.sort_values("p_win")
    n_quintiles = 5
    quintile_size = len(meta_sorted) // n_quintiles
    calibration = []
    for q in range(n_quintiles):
        start = q * quintile_size
        end = start + quintile_size if q < n_quintiles - 1 else len(meta_sorted)
        chunk = meta_sorted.iloc[start:end]
        calibration.append({
            "quintile": q + 1,
            "p_win_range": f"{chunk['p_win'].min():.3f}-{chunk['p_win'].max():.3f}",
            "actual_wr": round((chunk["pnl_r"] > 0).mean(), 3),
            "avg_r": round(chunk["pnl_r"].mean(), 4),
            "n": len(chunk),
        })
    results["calibration"] = calibration

    # OOS-only metrics (unbiased — model never saw this data)
    oos = meta[meta["is_oos"]]
    if len(oos) > 0:
        oos_kept = oos[oos["take"]]
        oos_skipped = oos[~oos["take"]]
        results["oos"] = {
            "n_total": len(oos),
            "n_kept": len(oos_kept),
            "n_skipped": len(oos_skipped),
            "baseline": {
                "avg_r": float(oos["pnl_r"].mean()),
                "sharpe": _sharpe(oos["pnl_r"]),
                "wr": float((oos["pnl_r"] > 0).mean()),
            },
            "filtered": {
                "avg_r": float(oos_kept["pnl_r"].mean()) if len(oos_kept) > 0 else 0.0,
                "sharpe": _sharpe(oos_kept["pnl_r"]) if len(oos_kept) > 0 else 0.0,
                "wr": float((oos_kept["pnl_r"] > 0).mean()) if len(oos_kept) > 0 else 0.0,
            },
            "skipped_avg_r": float(oos_skipped["pnl_r"].mean()) if len(oos_skipped) > 0 else 0.0,
        }

    return results


def print_evaluation(results: dict) -> None:
    """Print formatted evaluation report."""
    if results.get("status") == "no_model":
        return

    inst = results["instrument"]
    b = results["baseline"]
    f = results["filtered"]
    sk = results["skipped"]

    print(f"\n{'=' * 75}")
    print(f"  META-LABEL EVALUATION — {inst}")
    print(f"  Threshold: {results['threshold']:.2f} | "
          f"Skip: {results['skip_pct']:.1%} ({results['n_skipped']:,d} of {results['n_total']:,d})")
    print(f"{'=' * 75}")

    # Before/After comparison
    print(f"\n  {'METRIC':<18} {'BASELINE':>12} {'FILTERED':>12} {'SKIPPED':>12} {'DELTA':>12}")
    print(f"  {'-' * 18} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")
    print(f"  {'Trades':<18} {results['n_total']:>12,d} {results['n_kept']:>12,d} "
          f"{results['n_skipped']:>12,d}")
    print(f"  {'Avg R':<18} {b['avg_r']:>+12.4f} {f['avg_r']:>+12.4f} "
          f"{sk['avg_r']:>+12.4f} {f['avg_r'] - b['avg_r']:>+12.4f}")
    print(f"  {'Total R':<18} {b['total_r']:>12.1f} {f['total_r']:>12.1f} "
          f"{'':>12} {f['total_r'] - b['total_r']:>+12.1f}")
    print(f"  {'Sharpe':<18} {b['sharpe']:>12.3f} {f['sharpe']:>12.3f} "
          f"{'':>12} {f['sharpe'] - b['sharpe']:>+12.3f}")
    print(f"  {'Win Rate':<18} {b['wr']:>11.1%} {f['wr']:>11.1%} "
          f"{sk['wr']:>11.1%} {f['wr'] - b['wr']:>+11.1%}")
    print(f"  {'Max DD (R)':<18} {b['max_dd']:>12.1f} {f['max_dd']:>12.1f}")

    # Calibration (lift curve)
    print("\n  PROBABILITY CALIBRATION (should increase monotonically)")
    print(f"  {'Q':<4} {'P(win) range':<18} {'Actual WR':>10} {'Avg R':>10} {'N':>8}")
    print(f"  {'-' * 4} {'-' * 18} {'-' * 10} {'-' * 10} {'-' * 8}")
    for c in results["calibration"]:
        marker = " <-- skip" if c["quintile"] <= 2 and c["avg_r"] < 0 else ""
        print(f"  Q{c['quintile']:<3d} {c['p_win_range']:<18} {c['actual_wr']:>9.1%} "
              f"{c['avg_r']:>+10.4f} {c['n']:>8,d}{marker}")

    # Session breakdown
    if results["sessions"]:
        print("\n  PER-SESSION IMPACT (sorted by lift)")
        print(f"  {'SESSION':<20} {'N':>6} {'KEPT':>6} {'SKIP%':>7} "
              f"{'BASE':>8} {'FILT':>8} {'SKIP':>8} {'LIFT':>8}")
        print(f"  {'-' * 20} {'-' * 6} {'-' * 6} {'-' * 7} "
              f"{'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
        for s in sorted(results["sessions"], key=lambda x: -x["lift"]):
            print(f"  {s['session']:<20} {s['n_total']:>6d} {s['n_kept']:>6d} "
                  f"{s['skip_pct']:>6.1%} {s['base_avgR']:>+8.4f} {s['filt_avgR']:>+8.4f} "
                  f"{s['skip_avgR']:>+8.4f} {s['lift']:>+8.4f}")

    # OOS-only section (unbiased)
    if "oos" in results:
        oos = results["oos"]
        ob = oos["baseline"]
        of = oos["filtered"]
        print("\n  OUT-OF-SAMPLE ONLY (model never saw this data)")
        print(f"  {'METRIC':<18} {'BASELINE':>12} {'FILTERED':>12} {'DELTA':>12}")
        print(f"  {'-' * 18} {'-' * 12} {'-' * 12} {'-' * 12}")
        print(f"  {'Trades':<18} {oos['n_total']:>12,d} {oos['n_kept']:>12,d} "
              f"{oos['n_kept'] - oos['n_total']:>+12,d}")
        print(f"  {'Avg R':<18} {ob['avg_r']:>+12.4f} {of['avg_r']:>+12.4f} "
              f"{of['avg_r'] - ob['avg_r']:>+12.4f}")
        print(f"  {'Sharpe':<18} {ob['sharpe']:>12.3f} {of['sharpe']:>12.3f} "
              f"{of['sharpe'] - ob['sharpe']:>+12.3f}")
        print(f"  {'Win Rate':<18} {ob['wr']:>11.1%} {of['wr']:>11.1%} "
              f"{of['wr'] - ob['wr']:>+11.1%}")

    print(f"\n{'=' * 75}\n")


def main():
    parser = argparse.ArgumentParser(description="Meta-Label Evaluation")
    parser.add_argument("--instrument", type=str, help="Single instrument")
    parser.add_argument("--all", action="store_true", help="All instruments")
    parser.add_argument("--db-path", type=str, default=str(GOLD_DB_PATH))
    args = parser.parse_args()

    instruments = ACTIVE_INSTRUMENTS if args.all else [args.instrument or "MGC"]
    for inst in instruments:
        results = evaluate_instrument(inst, args.db_path)
        print_evaluation(results)


if __name__ == "__main__":
    main()
