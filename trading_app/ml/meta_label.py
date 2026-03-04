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

import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from pipeline.paths import GOLD_DB_PATH
from trading_app.ml.config import (
    ACTIVE_INSTRUMENTS,
    CROSS_SESSION_FEATURES,
    LEVEL_PROXIMITY_FEATURES,
    MAX_EARLY_SESSION_INDEX,
    MIN_SAMPLES_TRAIN,
    MIN_SESSION_SAMPLES,
    MODEL_DIR,
    RF_PARAMS,
    SESSION_CHRONOLOGICAL_ORDER,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
    compute_config_hash,
)
from trading_app.ml.cpcv import cpcv_score
from trading_app.ml.features import (
    apply_e6_filter,
    load_feature_matrix,
    load_validated_feature_matrix,
)

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

    Called on the VALIDATION set (middle 20% of 3-way split).
    Honest OOS evaluation happens separately on the frozen test set.
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


def _optimize_threshold_profit(
    y_prob: np.ndarray,
    pnl_r: np.ndarray,
    *,
    min_kept: int = 50,
) -> tuple[float | None, float]:
    """Find threshold that maximizes total R improvement over baseline.

    Simpler than _optimize_threshold — used for per-session models where
    sample sizes are smaller and Sharpe is too noisy.

    Called on the VALIDATION set (middle 20%) — never on the test set.
    Honest OOS evaluation happens separately on the frozen test set.

    Args:
        min_kept: Minimum trades to keep at a threshold. Caller should set
            this to max(50, int(n_val * 0.15)) to prevent overfitting
            to a tiny subset of val trades.

    Returns:
        (best_threshold, best_delta_r) — None if no threshold beats baseline.
    """
    valid = ~np.isnan(pnl_r)
    y_prob = y_prob[valid]
    pnl_r = pnl_r[valid]

    base_r = float(pnl_r.sum())
    best_delta = 0.0  # Must beat baseline
    best_t = None

    for t in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
        kept = y_prob >= t
        if kept.sum() < min_kept:
            continue
        filt_r = float(pnl_r[kept].sum())
        delta = filt_r - base_r
        if delta > best_delta:
            best_delta = delta
            best_t = round(t, 2)

    return best_t, best_delta


def _get_session_features(
    X_e6: pd.DataFrame,
    session: str,
) -> pd.DataFrame:
    """Get features appropriate for a specific session.

    Early sessions (CME_REOPEN, TOKYO_OPEN) drop cross-session features
    because they have near-constant values (0 or 1 prior sessions).
    TOKYO_OPEN keeps level proximity features (1 prior session provides
    meaningful level data).
    """
    session_idx = (
        SESSION_CHRONOLOGICAL_ORDER.index(session)
        if session in SESSION_CHRONOLOGICAL_ORDER
        else -1
    )

    if session_idx <= MAX_EARLY_SESSION_INDEX:
        # Early session: drop cross-session counts
        drop_cols = [c for c in CROSS_SESSION_FEATURES if c in X_e6.columns]
        if session_idx == 0:
            # CME_REOPEN: also drop level proximity (no prior sessions)
            drop_cols += [c for c in LEVEL_PROXIMITY_FEATURES if c in X_e6.columns]
        X_session = X_e6.drop(columns=drop_cols, errors="ignore")
    else:
        X_session = X_e6

    return X_session


def train_per_session_meta_label(
    instrument: str,
    db_path: str,
    *,
    save_model: bool = True,
    run_cpcv: bool = True,
) -> dict:
    """Train hybrid per-session meta-label models for one instrument.

    Architecture per de Prado (meta-labeling + honest evaluation):
      1. Load validated feature matrix, apply E6 noise filter
      2. 3-way time split: train (60%) / val (20%) / test (20%)
         - Train: RF fitting + optional per-session CPCV
         - Val: threshold optimization (find best P(win) cutoff)
         - Test: honest OOS evaluation (NEVER touched by optimization)
      3. For each session with >= MIN_SESSION_SAMPLES:
         - Adjust features for early sessions (drop cross-session)
         - Train RF with adaptive min_samples_leaf
         - CPCV within training data (5 groups, 10 splits) for unbiased AUC
         - Optimize threshold on VAL set (total R delta)
         - Evaluate honestly on TEST set (frozen, no optimization)
         - Sessions where no threshold helps >> NO_MODEL
      4. Save all session models in one bundle

    The 3-way split ensures reported delta_R values are honest out-of-sample.
    Prior implementation used 80/20 with threshold optimization on the holdout,
    which made OOS metrics optimistically biased (optimizing 36 thresholds on
    the same data used for reporting = multiple testing on the holdout).

    @research-source: ml_hybrid_experiment.py (Mar 4 2026)
    @revalidated-for: E1, E2
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"  PER-SESSION META-LABEL: {instrument}")
    logger.info(f"{'=' * 60}")

    # --- Load validated data + E6 filter ---
    X_all, y_all, meta_all = load_validated_feature_matrix(db_path, instrument)
    X_e6 = apply_e6_filter(X_all)

    logger.info(f"Total: {len(X_e6):,d} samples, {X_e6.shape[1]} E6 features")

    # --- 3-way time split: 60% train / 20% val / 20% test ---
    n_total = len(X_e6)
    n_train_end = int(n_total * 0.60)
    n_val_end = int(n_total * 0.80)
    pnl_r = meta_all["pnl_r"].values

    logger.info(
        f"  Split: train={n_train_end:,d} / val={n_val_end - n_train_end:,d} / "
        f"test={n_total - n_val_end:,d}"
    )

    sessions = sorted(meta_all["orb_label"].unique())
    session_results: dict[str, dict] = {}
    rf_base_params = {k: v for k, v in RF_PARAMS.items() if k != "min_samples_leaf"}

    for session in sessions:
        smask = (meta_all["orb_label"] == session).values
        n_session = smask.sum()
        session_indices = np.where(smask)[0]

        # 3-way split per session using global index boundaries
        train_idx = session_indices[session_indices < n_train_end]
        val_idx = session_indices[
            (session_indices >= n_train_end) & (session_indices < n_val_end)
        ]
        test_idx = session_indices[session_indices >= n_val_end]

        # Need enough data in ALL three splits
        if n_session < MIN_SESSION_SAMPLES or len(val_idx) < 20 or len(test_idx) < 20:
            reason = (
                f"N={n_session}" if n_session < MIN_SESSION_SAMPLES
                else f"N_val={len(val_idx)},N_test={len(test_idx)}"
            )
            logger.info(f"  {session:<20} >> NO_MODEL ({reason} < threshold)")
            session_results[session] = {"model_type": "NONE", "reason": reason}
            continue

        # Get session-appropriate features
        X_session = _get_session_features(X_e6, session)
        feature_names = list(X_session.columns)

        # Adaptive leaf size: bigger datasets get bigger leaves
        leaf_size = max(20, min(100, len(train_idx) // 20))

        # --- Optional CPCV within training data ---
        cpcv_auc = None
        if run_cpcv and len(train_idx) >= 200:
            try:
                cpcv_results = cpcv_score(
                    RandomForestClassifier,
                    {**rf_base_params, "min_samples_leaf": leaf_size},
                    X_session.iloc[train_idx],
                    y_all.iloc[train_idx],
                    meta_all["trading_day"].iloc[train_idx],
                    n_groups=5,      # C(5,2) = 10 splits (feasible for per-session)
                    k_test=2,
                    max_splits=10,
                )
                cpcv_auc = cpcv_results["auc_mean"]
            except Exception:
                logger.debug(f"  {session}: CPCV failed, continuing without")

        # --- Train RF on training data (first 60%) ---
        rf = RandomForestClassifier(**rf_base_params, min_samples_leaf=leaf_size)
        rf.fit(X_session.iloc[train_idx], y_all.iloc[train_idx])

        # --- Optimize threshold on VALIDATION set (middle 20%) ---
        val_prob = rf.predict_proba(X_session.iloc[val_idx])[:, 1]
        val_pnl = pnl_r[val_idx]
        val_min_kept = max(50, int(len(val_idx) * 0.15))
        best_t, best_delta = _optimize_threshold_profit(
            val_prob, val_pnl, min_kept=val_min_kept,
        )

        # --- Evaluate honestly on TEST set (final 20%) ---
        test_prob = rf.predict_proba(X_session.iloc[test_idx])[:, 1]
        test_pnl = pnl_r[test_idx]
        y_test = y_all.iloc[test_idx].values
        try:
            test_auc = roc_auc_score(y_test, test_prob)
        except ValueError:
            test_auc = 0.5  # Single class in test

        if best_t is not None:
            # Apply val-optimized threshold to frozen test set
            kept_test = test_prob >= best_t
            n_kept_test = int(kept_test.sum())
            if n_kept_test < 10:
                # Threshold too aggressive for test set — reject
                logger.info(
                    f"  {session:<20} >> NO_MODEL (threshold {best_t:.2f} keeps "
                    f"only {n_kept_test} test trades)"
                )
                session_results[session] = {
                    "model_type": "NONE",
                    "reason": f"threshold_too_aggressive_test_n={n_kept_test}",
                    "test_auc": round(test_auc, 4),
                    "cpcv_auc": round(cpcv_auc, 4) if cpcv_auc else None,
                }
                continue

            honest_base_r = float(test_pnl.sum())
            honest_filt_r = float(test_pnl[kept_test].sum())
            honest_delta_r = honest_filt_r - honest_base_r
            skip_pct = 1 - n_kept_test / len(test_idx)

            cpcv_str = f"{cpcv_auc:.3f}" if cpcv_auc else "  --"
            logger.info(
                f"  {session:<20} >> ML t={best_t:.2f} "
                f"CPCV={cpcv_str} TestAUC={test_auc:.3f} "
                f"ValDelta={best_delta:+.1f} HonestDelta={honest_delta_r:+.1f} "
                f"Skip={skip_pct:.0%} leaf={leaf_size}"
            )
            session_results[session] = {
                "model_type": "SESSION",
                "model": rf,
                "feature_names": feature_names,
                "threshold": best_t,
                "cpcv_auc": round(cpcv_auc, 4) if cpcv_auc else None,
                "test_auc": round(test_auc, 4),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "val_delta_r": round(best_delta, 2),
                "honest_delta_r": round(honest_delta_r, 2),
                "honest_base_r": round(honest_base_r, 2),
                "skip_pct": round(skip_pct, 3),
                "leaf_size": leaf_size,
            }
        else:
            logger.info(
                f"  {session:<20} >> NO_MODEL (no threshold beats baseline, "
                f"TestAUC={test_auc:.3f})"
            )
            session_results[session] = {
                "model_type": "NONE",
                "reason": "no_positive_threshold",
                "test_auc": round(test_auc, 4),
                "cpcv_auc": round(cpcv_auc, 4) if cpcv_auc else None,
            }

    # --- Summary ---
    n_ml = sum(1 for s in session_results.values() if s["model_type"] == "SESSION")
    n_none = sum(1 for s in session_results.values() if s["model_type"] == "NONE")
    total_val_delta = sum(
        s.get("val_delta_r", 0) for s in session_results.values()
        if s["model_type"] == "SESSION"
    )
    total_honest_delta = sum(
        s.get("honest_delta_r", 0) for s in session_results.values()
        if s["model_type"] == "SESSION"
    )

    logger.info(
        f"\n  SUMMARY: {n_ml} ML sessions, {n_none} NO_MODEL"
        f"\n  Val delta (optimized):  {total_val_delta:+.1f}R"
        f"\n  Honest delta (frozen):  {total_honest_delta:+.1f}R"
    )

    # --- Save bundle ---
    model_path = None
    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"meta_label_{instrument}_hybrid.joblib"
        config_hash = compute_config_hash()

        bundle = {
            "model_type": "hybrid_per_session",
            "instrument": instrument,
            "sessions": {},
            "config_hash": config_hash,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_date_range": (
                str(meta_all["trading_day"].min()),
                str(meta_all["trading_day"].max()),
            ),
            "split_ratios": "60/20/20",
            "n_total_samples": n_total,
            "n_ml_sessions": n_ml,
            "n_none_sessions": n_none,
            "total_val_delta_r": round(total_val_delta, 2),
            "total_honest_delta_r": round(total_honest_delta, 2),
        }

        for session, info in session_results.items():
            if info["model_type"] == "SESSION":
                bundle["sessions"][session] = {
                    "model": info["model"],
                    "feature_names": info["feature_names"],
                    "optimal_threshold": info["threshold"],
                    "cpcv_auc": info.get("cpcv_auc"),
                    "test_auc": info["test_auc"],
                    "n_train": info["n_train"],
                    "n_val": info["n_val"],
                    "n_test": info["n_test"],
                    "val_delta_r": info["val_delta_r"],
                    "honest_delta_r": info["honest_delta_r"],
                    "skip_pct": info["skip_pct"],
                    "leaf_size": info["leaf_size"],
                }
            else:
                bundle["sessions"][session] = {
                    "model": None,
                    "model_type": "NONE",
                    "reason": info.get("reason", "insufficient_data"),
                }

        joblib.dump(bundle, model_path)
        logger.info(f"  Hybrid model saved: {model_path}")

    return {
        "status": "trained",
        "instrument": instrument,
        "model_type": "hybrid_per_session",
        "n_samples": n_total,
        "n_ml_sessions": n_ml,
        "n_none_sessions": n_none,
        "total_val_delta_r": round(total_val_delta, 2),
        "total_honest_delta_r": round(total_honest_delta, 2),
        "sessions": {
            s: {
                "model_type": info["model_type"],
                "threshold": info.get("threshold"),
                "cpcv_auc": info.get("cpcv_auc"),
                "test_auc": info.get("test_auc"),
                "val_delta_r": info.get("val_delta_r"),
                "honest_delta_r": info.get("honest_delta_r"),
                "skip_pct": info.get("skip_pct"),
            }
            for s, info in session_results.items()
        },
        "model_path": str(model_path) if model_path else None,
    }


def print_per_session_results(results: dict) -> None:
    """Print formatted per-session training results."""
    if results["status"] != "trained":
        print(f"  {results.get('instrument', '?')}: {results['status']}")
        return

    inst = results["instrument"]
    print(f"\n{'=' * 70}")
    print(f"  HYBRID PER-SESSION RESULTS -- {inst} (3-way split: 60/20/20)")
    print(f"{'=' * 70}")
    print(f"  Samples: {results['n_samples']:,d}")
    print(f"  ML sessions: {results['n_ml_sessions']} | "
          f"NO_MODEL sessions: {results['n_none_sessions']}")
    print(f"  Val delta (optimized):  {results['total_val_delta_r']:+.1f}R")
    print(f"  Honest delta (frozen):  {results['total_honest_delta_r']:+.1f}R")
    print()

    print(f"  {'SESSION':<22} {'TYPE':>5} {'THRESH':>6} "
          f"{'CPCV':>6} {'T_AUC':>6} {'VAL_dR':>7} {'OOS_dR':>7} {'SKIP%':>6}")
    print(f"  {'-' * 22} {'-' * 5} {'-' * 6} "
          f"{'-' * 6} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 6}")

    for session in SESSION_CHRONOLOGICAL_ORDER:
        if session not in results["sessions"]:
            continue
        info = results["sessions"][session]
        if info["model_type"] == "SESSION":
            cpcv_str = f"{info['cpcv_auc']:.3f}" if info.get("cpcv_auc") else "  --"
            print(
                f"  {session:<22} {'ML':>5} {info['threshold']:>6.2f} "
                f"{cpcv_str:>6} {info['test_auc']:>6.3f} "
                f"{info['val_delta_r']:>+7.1f} {info['honest_delta_r']:>+7.1f} "
                f"{info['skip_pct']:>5.1%}"
            )
        else:
            print(f"  {session:<22} {'NONE':>5} {'--':>6} "
                  f"{'--':>6} {'--':>6} {'--':>7} {'--':>7} {'--':>6}")

    # Any sessions not in SESSION_CHRONOLOGICAL_ORDER
    for session, info in sorted(results["sessions"].items()):
        if session not in SESSION_CHRONOLOGICAL_ORDER:
            print(f"  {session:<22} {'NONE':>5} {'--':>6} "
                  f"{'--':>6} {'--':>6} {'--':>7} {'--':>7} {'--':>6}")

    print(f"{'=' * 70}\n")


def train_meta_label(
    instrument: str,
    db_path: str,
    *,
    run_cpcv: bool = True,
    max_cpcv_splits: int | None = 20,
    save_model: bool = True,
    validated_only: bool = False,
) -> dict:
    """Train a meta-label classifier for one instrument.

    Steps:
      1. Load feature matrix (all outcomes OR validated-only)
      2. CPCV validation on training data (45 splits or capped)
      3. 3-way time split: train (60%) / val (20%) / test (20%)
      4. Train final model on 60% training data
      5. Threshold optimization on 20% validation set
      6. Honest OOS evaluation on 20% test set (frozen)
      7. Save model + report results

    Args:
        validated_only: If True, train only on outcomes matching validated
            strategy combos with filter_type eligibility applied.

    Returns:
        dict with cpcv_results, threshold_results, honest_oos, model_path
    """
    mode = "VALIDATED-ONLY" if validated_only else "ALL OUTCOMES"
    logger.info(f"{'=' * 60}")
    logger.info(f"  META-LABEL TRAINING: {instrument} ({mode})")
    logger.info(f"{'=' * 60}")

    # --- Load data ---
    if validated_only:
        X, y, meta = load_validated_feature_matrix(db_path, instrument)
    else:
        X, y, meta = load_feature_matrix(db_path, instrument)

    if len(X) < MIN_SAMPLES_TRAIN:
        logger.warning(f"Insufficient samples for {instrument}: {len(X)} < {MIN_SAMPLES_TRAIN}")
        return {"status": "insufficient_data", "n_samples": len(X)}

    logger.info(f"Samples: {len(X):,d} | Features: {X.shape[1]} | Win rate: {y.mean():.1%}")

    # --- 3-way time split: 60% train / 20% val / 20% test ---
    n_total = len(X)
    n_train_end = int(n_total * 0.60)
    n_val_end = int(n_total * 0.80)

    X_train, y_train = X.iloc[:n_train_end], y.iloc[:n_train_end]
    X_val, y_val = X.iloc[n_train_end:n_val_end], y.iloc[n_train_end:n_val_end]
    X_test, y_test = X.iloc[n_val_end:], y.iloc[n_val_end:]
    pnl_val = meta["pnl_r"].iloc[n_train_end:n_val_end].values
    pnl_test = meta["pnl_r"].iloc[n_val_end:].values
    meta_test = meta.iloc[n_val_end:].copy()

    logger.info(
        f"Split: train={n_train_end:,d} / val={n_val_end - n_train_end:,d} / "
        f"test={n_total - n_val_end:,d}"
    )

    # --- CPCV Validation (on training data only) ---
    cpcv_results = None
    if run_cpcv:
        logger.info("Running CPCV validation on training data...")
        cpcv_results = cpcv_score(
            RandomForestClassifier,
            RF_PARAMS,
            X_train, y_train,
            meta["trading_day"].iloc[:n_train_end],
            max_splits=max_cpcv_splits,
        )
        logger.info(f"CPCV AUC: {cpcv_results['auc_mean']:.4f} +/- {cpcv_results['auc_std']:.4f} "
                     f"({cpcv_results['n_splits']} splits)")

        if cpcv_results["auc_mean"] < 0.505:
            logger.warning(f"CPCV AUC {cpcv_results['auc_mean']:.4f} is barely above random. "
                          f"Meta-label may not add value for {instrument}.")

    # --- Train final model on training data (first 60%) ---
    logger.info(f"Training final model: {n_train_end:,d} train")

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    # --- Threshold optimization on VALIDATION set (middle 20%) ---
    logger.info("Optimizing threshold on validation set...")
    val_prob = rf.predict_proba(X_val)[:, 1]
    threshold_results = _optimize_threshold(val_prob, y_val.values, pnl_val)

    opt = threshold_results["optimal"]
    base = threshold_results["baseline"]
    logger.info(f"Val baseline: avgR={base['avg_r']:.4f} | Sharpe={base['sharpe']:.3f} | N={base['n']:,d}")
    logger.info(f"Val optimal:  t={opt['threshold']:.2f} | avgR={opt['avg_r']:.4f} | "
                f"Sharpe={opt['sharpe']:.3f} | skip={opt['skip_pct']:.1%} | N={opt['n_kept']:,d}")

    # --- Honest OOS evaluation on TEST set (final 20%) ---
    test_prob = rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_prob)

    # Apply val-optimized threshold to frozen test set
    test_kept = test_prob >= opt["threshold"]
    test_base_r = float(pnl_test.sum())
    test_filt_r = float(pnl_test[test_kept].sum()) if test_kept.sum() > 0 else 0.0
    test_delta_r = test_filt_r - test_base_r
    test_skip_pct = 1 - test_kept.sum() / len(pnl_test) if len(pnl_test) > 0 else 0.0

    logger.info(f"Honest OOS: AUC={test_auc:.4f} | BaseR={test_base_r:+.1f} | "
                f"Delta={test_delta_r:+.1f} | Skip={test_skip_pct:.1%}")

    # --- Save model ---
    model_path = None
    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"meta_label_{instrument}.joblib"
        config_hash = compute_config_hash()

        joblib.dump({
            "model": rf,
            "feature_names": list(X.columns),
            "instrument": instrument,
            "n_train": n_train_end,
            "split_ratios": "60/20/20",
            "oos_auc": test_auc,
            "optimal_threshold": opt["threshold"],
            "cpcv_auc": cpcv_results["auc_mean"] if cpcv_results else None,
            "honest_delta_r": round(test_delta_r, 2),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_date_range": (
                str(meta["trading_day"].min()),
                str(meta["trading_day"].max()),
            ),
            "config_hash": config_hash,
            "validated_only": validated_only,
            "n_total_samples": n_total,
        }, model_path)
        logger.info(f"Model saved: {model_path}")

    # --- Per-session breakdown (on honest test set) ---
    meta_test["y_prob"] = test_prob
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
        "n_samples": n_total,
        "n_features": X.shape[1],
        "oos_auc": test_auc,
        "honest_delta_r": round(test_delta_r, 2),
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
    print(f"  META-LABEL RESULTS — {inst} (3-way split: 60/20/20)")
    print(f"{'=' * 70}")

    # Summary
    base = results["threshold"]["baseline"]
    opt = results["threshold"]["optimal"]
    print(f"  Samples: {results['n_samples']:,d} | Features: {results['n_features']}")
    if results["cpcv"]:
        c = results["cpcv"]
        print(f"  CPCV AUC (train): {c['auc_mean']:.4f} +/- {c['auc_std']:.4f} ({c['n_splits']} splits)")
    print(f"  Honest OOS AUC:   {results['oos_auc']:.4f}")
    print(f"  Honest OOS delta: {results['honest_delta_r']:+.1f}R")
    print()

    # Before/After (validation set — used for threshold optimization)
    print("  VALIDATION SET (threshold optimization):")
    print(f"  {'METRIC':<20} {'BASELINE':>12} {'FILTERED':>12} {'CHANGE':>12}")
    print(f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 12}")
    print(f"  {'Trades':<20} {base['n']:>12,d} {opt['n_kept']:>12,d} {opt['n_kept'] - base['n']:>+12,d}")
    print(f"  {'Skip %':<20} {'0.0%':>12} {opt['skip_pct']:>11.1%}")
    print(f"  {'Avg R':<20} {base['avg_r']:>12.4f} {opt['avg_r']:>12.4f} {opt['avg_r_improvement']:>+12.4f}")
    print(f"  {'Sharpe':<20} {base['sharpe']:>12.3f} {opt['sharpe']:>12.3f} {opt['sharpe_improvement']:>+12.3f}")
    print(f"  {'Win Rate':<20} {'':>12} {opt['wr']:>11.1%}")
    print(f"  {'Threshold':<20} {'':>12} {opt['threshold']:>12.2f}")
    print()

    # Per-session breakdown (honest test set)
    if results["session_breakdown"]:
        print("  HONEST OOS (test set — never touched by optimization):")
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
    parser.add_argument("--validated-only", action="store_true",
                        help="Train only on validated strategy outcomes (recommended)")
    parser.add_argument("--per-session", action="store_true",
                        help="Train hybrid per-session models (recommended, implies --validated-only)")
    args = parser.parse_args()

    instruments = ACTIVE_INSTRUMENTS if args.all else [args.instrument or "MGC"]

    for inst in instruments:
        if args.per_session:
            results = train_per_session_meta_label(inst, args.db_path)
            print_per_session_results(results)
        else:
            results = train_meta_label(
                inst, args.db_path,
                run_cpcv=not args.no_cpcv,
                max_cpcv_splits=args.max_cpcv_splits,
                validated_only=args.validated_only,
            )
            print_results(results)


if __name__ == "__main__":
    main()
