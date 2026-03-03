"""Meta-label classifier: train, validate (CPCV), threshold-optimize, predict.

This is the core module. Architecture per de Prado:
  Primary model = ORB rules → direction (LONG/SHORT)
  Meta-label    = RF → P(win) → skip/take decision

Usage:
    python -m trading_app.ml.meta_label --instrument MGC
    python -m trading_app.ml.meta_label --all
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from pipeline.paths import GOLD_DB_PATH
from trading_app.ml.config import (
    ACTIVE_INSTRUMENTS,
    MIN_SAMPLES_TRAIN,
    MODEL_DIR,
    RF_PARAMS,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
)
from trading_app.ml.cpcv import cpcv_score
from trading_app.ml.features import load_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _optimize_threshold(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    pnl_r: np.ndarray,
) -> dict:
    """Find P(win) threshold that maximizes Sharpe improvement.

    Optimizes on PROFIT (total R), not accuracy.
    Returns dict with optimal threshold and performance at that threshold.

    WARNING: Threshold is optimized on the same holdout used for OOS reporting.
    This means OOS metrics at the selected threshold are biased (optimistic).
    Only CPCV AUC is truly unbiased. Phase 2 fix: 3-way split or CPCV-derived
    threshold. See docs/specs/STATISTICAL_HARDENING.md FIX 10.
    """
    # Filter out NaN pnl_r before computing metrics
    valid = ~np.isnan(pnl_r)
    y_prob = y_prob[valid]
    y_true = y_true[valid]
    pnl_r = pnl_r[valid]

    baseline_total_r = float(pnl_r.sum())
    baseline_avg_r = float(pnl_r.mean())
    baseline_n = len(pnl_r)
    # Per-trade Sharpe (no annualization — data is per-trade, not per-day)
    baseline_sharpe = float(pnl_r.mean() / pnl_r.std()) if pnl_r.std() > 0 else 0.0
    baseline_wr = float((pnl_r > 0).mean())

    # Minimum kept trades: at least 10% of baseline or 200, whichever is larger
    min_kept = max(200, int(baseline_n * 0.10))

    best = {"threshold": 0.5, "sharpe_improvement": -999.0}
    results = []

    for t in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
        mask = y_prob >= t
        n_kept = mask.sum()
        if n_kept < min_kept:
            continue

        kept_pnl = pnl_r[mask]
        avg_r = kept_pnl.mean()
        total_r = kept_pnl.sum()
        sharpe = kept_pnl.mean() / kept_pnl.std() if kept_pnl.std() > 0 else 0
        skip_pct = 1 - n_kept / baseline_n
        wr = (kept_pnl > 0).mean()

        improvement = sharpe - baseline_sharpe

        row = {
            "threshold": round(t, 2),
            "n_kept": n_kept,
            "skip_pct": round(skip_pct, 3),
            "avg_r": round(avg_r, 4),
            "total_r": round(total_r, 2),
            "sharpe": round(sharpe, 3),
            "wr": round(wr, 3),
            "sharpe_improvement": round(improvement, 3),
            "avg_r_improvement": round(avg_r - baseline_avg_r, 4),
        }
        results.append(row)

        if improvement > best["sharpe_improvement"]:
            best = row

    return {
        "optimal": best,
        "baseline": {
            "n": baseline_n,
            "avg_r": round(baseline_avg_r, 4),
            "total_r": round(baseline_total_r, 2),
            "sharpe": round(baseline_sharpe, 3),
            "wr": round(baseline_wr, 3),
        },
        "sweep": results,
    }


def train_meta_label(
    instrument: str,
    db_path: str,
    *,
    run_cpcv: bool = True,
    max_cpcv_splits: int | None = 20,
    save_model: bool = True,
) -> dict:
    """Train a meta-label classifier for one instrument.

    Steps:
      1. Load feature matrix (all outcomes for instrument)
      2. CPCV validation (45 splits or capped)
      3. Train final model on 80% time-ordered data
      4. Threshold optimization on 20% holdout
      5. Save model + report results

    BIAS NOTE: The 20% holdout serves double duty — threshold optimization
    AND OOS evaluation. This means OOS metrics at the selected threshold
    are slightly optimistic. The CPCV AUC (step 2) is the only fully
    unbiased performance estimate. See STATISTICAL_HARDENING.md FIX 10.

    Returns:
        dict with cpcv_results, threshold_results, model_path
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"  META-LABEL TRAINING: {instrument}")
    logger.info(f"{'=' * 60}")

    # --- Load data ---
    X, y, meta = load_feature_matrix(db_path, instrument)

    if len(X) < MIN_SAMPLES_TRAIN:
        logger.warning(f"Insufficient samples for {instrument}: {len(X)} < {MIN_SAMPLES_TRAIN}")
        return {"status": "insufficient_data", "n_samples": len(X)}

    logger.info(f"Samples: {len(X):,d} | Features: {X.shape[1]} | Win rate: {y.mean():.1%}")

    # --- CPCV Validation ---
    cpcv_results = None
    if run_cpcv:
        logger.info("Running CPCV validation...")
        cpcv_results = cpcv_score(
            RandomForestClassifier,
            RF_PARAMS,
            X, y,
            meta["trading_day"],
            max_splits=max_cpcv_splits,
        )
        logger.info(f"CPCV AUC: {cpcv_results['auc_mean']:.4f} +/- {cpcv_results['auc_std']:.4f} "
                     f"({cpcv_results['n_splits']} splits)")

        if cpcv_results["auc_mean"] < 0.505:
            logger.warning(f"CPCV AUC {cpcv_results['auc_mean']:.4f} is barely above random. "
                          f"Meta-label may not add value for {instrument}.")

    # --- Train final model (80/20 time split) ---
    n_train = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]
    pnl_test = meta["pnl_r"].iloc[n_train:].values
    meta_test = meta.iloc[n_train:].copy()

    logger.info(f"Training final model: {n_train:,d} train / {len(X) - n_train:,d} test")

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    oos_auc = roc_auc_score(y_test, y_prob)
    logger.info(f"Final model OOS AUC: {oos_auc:.4f}")

    # --- Threshold optimization ---
    logger.info("Optimizing threshold on profit metric...")
    threshold_results = _optimize_threshold(y_prob, y_test.values, pnl_test)

    opt = threshold_results["optimal"]
    base = threshold_results["baseline"]
    logger.info(f"Baseline: avgR={base['avg_r']:.4f} | Sharpe={base['sharpe']:.3f} | N={base['n']:,d}")
    logger.info(f"Optimal:  t={opt['threshold']:.2f} | avgR={opt['avg_r']:.4f} | "
                f"Sharpe={opt['sharpe']:.3f} | skip={opt['skip_pct']:.1%} | N={opt['n_kept']:,d}")
    logger.info(f"Improvement: avgR +{opt['avg_r_improvement']:.4f} | Sharpe +{opt['sharpe_improvement']:.3f}")

    # --- Save model ---
    model_path = None
    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"meta_label_{instrument}.joblib"
        # Config hash for reproducibility tracking
        config_str = f"{RF_PARAMS}|{THRESHOLD_MIN}|{THRESHOLD_MAX}|{THRESHOLD_STEP}"
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

        joblib.dump({
            "model": rf,
            "feature_names": list(X.columns),
            "instrument": instrument,
            "n_train": n_train,
            "oos_auc": oos_auc,
            "optimal_threshold": opt["threshold"],
            "cpcv_auc": cpcv_results["auc_mean"] if cpcv_results else None,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_date_range": (
                str(meta["trading_day"].min()),
                str(meta["trading_day"].max()),
            ),
            "config_hash": config_hash,
        }, model_path)
        logger.info(f"Model saved: {model_path}")

    # --- Per-session breakdown ---
    meta_test["y_prob"] = y_prob
    meta_test["y_true"] = y_test.values
    meta_test["pnl_r"] = pnl_test

    session_breakdown = []
    for session in sorted(meta_test["orb_label"].unique()):
        smask = meta_test["orb_label"] == session
        if smask.sum() < 30:
            continue
        s_pnl = meta_test.loc[smask, "pnl_r"]
        s_prob = meta_test.loc[smask, "y_prob"]

        # Apply optimal threshold
        kept = s_prob >= opt["threshold"]
        if kept.sum() < 10:
            continue

        base_avg = s_pnl.mean()
        filt_avg = s_pnl[kept].mean()
        session_breakdown.append({
            "session": session,
            "n_total": smask.sum(),
            "n_kept": kept.sum(),
            "skip_pct": round(1 - kept.sum() / smask.sum(), 3),
            "base_avgR": round(base_avg, 4),
            "filt_avgR": round(filt_avg, 4),
            "lift": round(filt_avg - base_avg, 4),
        })

    return {
        "status": "trained",
        "instrument": instrument,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "oos_auc": oos_auc,
        "cpcv": cpcv_results,
        "threshold": threshold_results,
        "session_breakdown": session_breakdown,
        "model_path": str(model_path) if model_path else None,
    }


def print_results(results: dict) -> None:
    """Print formatted training results."""
    if results["status"] != "trained":
        print(f"  {results.get('instrument', '?')}: {results['status']}")
        return

    inst = results["instrument"]
    print(f"\n{'=' * 70}")
    print(f"  META-LABEL RESULTS — {inst}")
    print(f"{'=' * 70}")

    # Summary
    base = results["threshold"]["baseline"]
    opt = results["threshold"]["optimal"]
    print(f"  Samples: {results['n_samples']:,d} | Features: {results['n_features']}")
    if results["cpcv"]:
        c = results["cpcv"]
        print(f"  CPCV AUC: {c['auc_mean']:.4f} +/- {c['auc_std']:.4f} ({c['n_splits']} splits)")
    print(f"  OOS AUC:  {results['oos_auc']:.4f}")
    print()

    # Before/After
    print(f"  {'METRIC':<20} {'BASELINE':>12} {'FILTERED':>12} {'CHANGE':>12}")
    print(f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12}")
    print(f"  {'Trades':<20} {base['n']:>12,d} {opt['n_kept']:>12,d} {opt['n_kept'] - base['n']:>+12,d}")
    print(f"  {'Skip %':<20} {'0.0%':>12} {opt['skip_pct']:>11.1%}")
    print(f"  {'Avg R':<20} {base['avg_r']:>12.4f} {opt['avg_r']:>12.4f} {opt['avg_r_improvement']:>+12.4f}")
    print(f"  {'Sharpe':<20} {base['sharpe']:>12.3f} {opt['sharpe']:>12.3f} {opt['sharpe_improvement']:>+12.3f}")
    print(f"  {'Win Rate':<20} {'':>12} {opt['wr']:>11.1%}")
    print(f"  {'Threshold':<20} {'':>12} {opt['threshold']:>12.2f}")
    print()

    # Per-session breakdown
    if results["session_breakdown"]:
        print(f"  {'SESSION':<20} {'N':>6} {'KEPT':>6} {'SKIP%':>7} {'BASE':>8} {'FILT':>8} {'LIFT':>8}")
        print(f"  {'-' * 20} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 8} {'-' * 8} {'-' * 8}")
        for s in sorted(results["session_breakdown"], key=lambda x: -x["lift"]):
            print(f"  {s['session']:<20} {s['n_total']:>6d} {s['n_kept']:>6d} "
                  f"{s['skip_pct']:>6.1%} {s['base_avgR']:>+8.4f} {s['filt_avgR']:>+8.4f} "
                  f"{s['lift']:>+8.4f}")

    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Meta-Label Classifier Training")
    parser.add_argument("--instrument", type=str, help="Single instrument")
    parser.add_argument("--all", action="store_true", help="All active instruments")
    parser.add_argument("--no-cpcv", action="store_true", help="Skip CPCV (faster)")
    parser.add_argument("--max-cpcv-splits", type=int, default=20,
                        help="Max CPCV splits (default 20 of 45)")
    parser.add_argument("--db-path", type=str, default=str(GOLD_DB_PATH))
    args = parser.parse_args()

    instruments = ACTIVE_INSTRUMENTS if args.all else [args.instrument or "MGC"]

    for inst in instruments:
        results = train_meta_label(
            inst, args.db_path,
            run_cpcv=not args.no_cpcv,
            max_cpcv_splits=args.max_cpcv_splits,
        )
        print_results(results)


if __name__ == "__main__":
    main()
