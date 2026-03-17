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
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
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
    load_single_config_feature_matrix,
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

    Multiple-testing note: sweeps thresholds from THRESHOLD_MIN to
    THRESHOLD_MAX in steps of THRESHOLD_STEP (~36 candidates). Selecting
    best-of-N on a finite validation set introduces mild selection bias —
    the chosen threshold may not be optimal out-of-sample. Mitigation: 4
    OOS quality gates applied on the frozen test set gate downstream model
    acceptance (delta_r >= 0, CPCV AUC >= 0.50, test AUC >= 0.52,
    skip_pct <= 0.85). The test set is NEVER used during optimization.
    See train_per_session_meta_label for gate implementation.

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
    session_idx = SESSION_CHRONOLOGICAL_ORDER.index(session) if session in SESSION_CHRONOLOGICAL_ORDER else -1

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
    single_config: bool = False,
    rr_target: float | None = None,
    min_session_samples: int | None = None,
    config_selection: str = "max_samples",
    skip_filter: bool = False,
    per_aperture: bool = False,
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
    mode_str = "SINGLE-CONFIG" if single_config else "MULTI-CONFIG"
    rr_str = f" RR={rr_target}" if rr_target is not None else ""
    sel_str = f" sel={config_selection}" if single_config else ""
    filt_str = " UNFILTERED" if skip_filter else ""
    aperture_str = " PER-APERTURE" if per_aperture else ""
    logger.info(f"{'=' * 60}")
    logger.info(f"  PER-SESSION META-LABEL: {instrument} ({mode_str}{rr_str}{sel_str}{filt_str}{aperture_str})")
    logger.info(f"{'=' * 60}")

    # --- Load data + E6 filter ---
    if single_config:
        X_all, y_all, meta_all = load_single_config_feature_matrix(
            db_path,
            instrument,
            rr_target=rr_target,
            config_selection=config_selection,
            skip_filter=skip_filter,
            per_aperture=per_aperture,
            apply_rr_lock=False,
        )
    else:
        X_all, y_all, meta_all = load_validated_feature_matrix(db_path, instrument)
    X_e6 = apply_e6_filter(X_all)

    logger.info(f"Total: {len(X_e6):,d} samples, {X_e6.shape[1]} E6 features")

    # --- 3-way time split: 60% train / 20% val / 20% test ---
    n_total = len(X_e6)
    n_train_end = int(n_total * 0.60)
    n_val_end = int(n_total * 0.80)
    pnl_r = meta_all["pnl_r"].values

    logger.info(f"  Split: train={n_train_end:,d} / val={n_val_end - n_train_end:,d} / test={n_total - n_val_end:,d}")

    sessions = sorted(meta_all["orb_label"].unique())
    session_results: dict[str, dict] = {}
    rf_base_params = {k: v for k, v in RF_PARAMS.items() if k != "min_samples_leaf"}

    # Per-aperture: build list of (session, aperture) training units.
    # Flat mode: one unit per session. Per-aperture: one per (session, aperture).
    if per_aperture:
        all_apertures = sorted(meta_all["orb_minutes"].unique())
    else:
        all_apertures = [None]  # sentinel: no aperture sub-loop

    for session in sessions:
        smask_session = (meta_all["orb_label"] == session).values

        if per_aperture:
            session_results[session] = {}  # nested dict for apertures

        for aperture in all_apertures:
            if per_aperture:
                # Further filter to this aperture within the session
                amask = smask_session & (meta_all["orb_minutes"] == aperture).values
                log_prefix = f"  {session:<20} O{aperture:<3}"
            else:
                amask = smask_session
                log_prefix = f"  {session:<20}"

            n_session = amask.sum()
            session_indices = np.where(amask)[0]

            # Helper: store result in flat or nested dict
            _ak = f"O{aperture}" if per_aperture else None

            def _store(result: dict, _s=session, _a=_ak) -> None:
                if _a is not None:
                    session_results[_s][_a] = result
                else:
                    session_results[_s] = result

            if n_session == 0:
                if per_aperture:
                    _store({"model_type": "NONE", "reason": "no_data_for_aperture"})
                continue

            # Capture training aperture/RR BEFORE constant column drop removes them.
            training_aperture = int(meta_all["orb_minutes"].iloc[session_indices[0]])
            training_rr = float(meta_all["rr_target"].iloc[session_indices[0]])

            # 3-way split per session using global index boundaries
            train_idx = session_indices[session_indices < n_train_end]
            val_idx = session_indices[(session_indices >= n_train_end) & (session_indices < n_val_end)]
            test_idx = session_indices[session_indices >= n_val_end]

            # Effective threshold: lower for single-config (independent samples)
            effective_min = (
                min_session_samples
                if min_session_samples is not None
                else (200 if single_config else MIN_SESSION_SAMPLES)
            )

            # Need enough data in ALL three splits
            if n_session < effective_min or len(val_idx) < 20 or len(test_idx) < 20:
                reason = (
                    f"N={n_session}<{effective_min}"
                    if n_session < effective_min
                    else f"N_val={len(val_idx)},N_test={len(test_idx)}"
                )
                logger.info(f"{log_prefix} >> NO_MODEL ({reason} < threshold)")
                _store({"model_type": "NONE", "reason": reason})
                continue

            # Get session-appropriate features
            X_session = _get_session_features(X_e6, session)

            # Drop constant columns within this session (e.g., entry_model one-hots
            # are constant per session — they waste a feature slot with zero info gain)
            session_data = X_session.iloc[session_indices]
            const_cols = [c for c in X_session.columns if session_data[c].nunique() <= 1]
            if const_cols:
                X_session = X_session.drop(columns=const_cols)
                preview = const_cols[:5]
                extra = f" (+{len(const_cols) - 5} more)" if len(const_cols) > 5 else ""
                logger.info(f"{log_prefix}    dropped {len(const_cols)} constant cols: {preview}{extra}")

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
                        n_groups=5,  # C(5,2) = 10 splits (feasible for per-session)
                        k_test=2,
                        max_splits=10,
                    )
                    cpcv_auc = cpcv_results["auc_mean"]
                except Exception:
                    logger.warning(f"  {session}: CPCV failed, continuing without CPCV validation", exc_info=True)

            # --- Train RF on training data (first 60%) ---
            rf = RandomForestClassifier(**rf_base_params, min_samples_leaf=leaf_size)
            rf.fit(X_session.iloc[train_idx], y_all.iloc[train_idx])

            # --- Feature importance audit trail (top 10 by MDI) ---
            importances = rf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            top_feats = [(feature_names[i], importances[i]) for i in top_idx]
            feat_str = ", ".join(f"{n}={v:.1%}" for n, v in top_feats)
            logger.info(f"{log_prefix}    top10: {feat_str}")

            # --- Optimize threshold on VALIDATION set (middle 20%) ---
            # Threshold search is rank-based (which trades to keep/skip).
            # It MUST operate on raw RF probabilities, not calibrated ones.
            # Calibration is a separate concern for P(win) interpretation only.
            val_prob_raw = rf.predict_proba(X_session.iloc[val_idx])[:, 1]

            val_pnl = pnl_r[val_idx]
            val_min_kept = max(50, int(len(val_idx) * 0.15))
            best_t, best_delta = _optimize_threshold_profit(
                val_prob_raw,
                val_pnl,
                min_kept=val_min_kept,
            )

            # --- Probability calibration (isotonic regression) ---
            # Fit on val set: maps raw RF probabilities to calibrated P(win).
            # Monotone transform — preserves ranking, makes probabilities meaningful.
            # Used ONLY at prediction time for display/Kelly sizing, never for
            # threshold search (which is rank-based and uses raw probabilities).
            y_val = y_all.iloc[val_idx].values
            calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            calibrator.fit(val_prob_raw, y_val)

            # --- Evaluate honestly on TEST set (final 20%) ---
            # Use RAW probabilities for threshold comparison and AUC
            # (consistent with threshold optimized on raw val probs).
            test_prob_raw = rf.predict_proba(X_session.iloc[test_idx])[:, 1]
            test_pnl = pnl_r[test_idx]
            y_test = y_all.iloc[test_idx].values
            try:
                test_auc = roc_auc_score(y_test, test_prob_raw)
            except ValueError:
                test_auc = 0.5  # Single class in test

            if best_t is not None:
                # Apply val-optimized threshold to frozen test set (raw probs)
                kept_test = test_prob_raw >= best_t
                n_kept_test = int(kept_test.sum())

                # Compute honest OOS delta BEFORE gates — needed for total_full_delta_r
                # (White 2000: always report ALL sessions, not just survivors)
                honest_base_r = float(test_pnl.sum())
                honest_filt_r = float(test_pnl[kept_test].sum())
                honest_delta_r = honest_filt_r - honest_base_r
                skip_pct = 1 - n_kept_test / len(test_idx)

                if n_kept_test < 10:
                    # Threshold too aggressive for test set — reject
                    logger.info(
                        f"{log_prefix} >> NO_MODEL (threshold {best_t:.2f} keeps only {n_kept_test} test trades)"
                    )
                    _store(
                        {
                            "model_type": "NONE",
                            "reason": f"threshold_too_aggressive_test_n={n_kept_test}",
                            "test_auc": round(test_auc, 4),
                            "cpcv_auc": round(cpcv_auc, 4) if cpcv_auc else None,
                            "honest_delta_r": round(honest_delta_r, 2),
                        }
                    )
                    continue

                # --- OOS quality gates ---
                # Gate 1: OOS must be positive (ML must not hurt)
                if honest_delta_r < 0:
                    logger.info(
                        f"{log_prefix} >> NO_MODEL (OOS negative: {honest_delta_r:+.1f}R, TestAUC={test_auc:.3f})"
                    )
                    _store(
                        {
                            "model_type": "NONE",
                            "reason": f"oos_negative_{honest_delta_r:+.1f}R",
                            "test_auc": round(test_auc, 4),
                            "cpcv_auc": round(cpcv_auc, 4) if cpcv_auc else None,
                            "honest_delta_r": round(honest_delta_r, 2),
                        }
                    )
                    continue

                # Gate 2: CPCV must be >= 0.50 (model must not be worse than random
                # in cross-validation on training data). Only applies when CPCV ran.
                if cpcv_auc is not None and cpcv_auc < 0.50:
                    logger.info(f"{log_prefix} >> NO_MODEL (CPCV={cpcv_auc:.3f} < 0.50, below random in CV)")
                    _store(
                        {
                            "model_type": "NONE",
                            "reason": f"cpcv_below_random_{cpcv_auc:.3f}",
                            "test_auc": round(test_auc, 4),
                            "cpcv_auc": round(cpcv_auc, 4),
                            "honest_delta_r": round(honest_delta_r, 2),
                        }
                    )
                    continue

                # Gate 3: AUC must be clearly above random (0.52 minimum)
                if test_auc < 0.52:
                    logger.info(f"{log_prefix} >> NO_MODEL (AUC={test_auc:.3f} < 0.52, near-random discrimination)")
                    _store(
                        {
                            "model_type": "NONE",
                            "reason": f"auc_too_low_{test_auc:.3f}",
                            "test_auc": round(test_auc, 4),
                            "cpcv_auc": round(cpcv_auc, 4) if cpcv_auc else None,
                            "honest_delta_r": round(honest_delta_r, 2),
                        }
                    )
                    continue

                # Gate 4: Skip rate must be < 85% (avoid "just don't trade" models)
                if skip_pct > 0.85:
                    logger.info(f"{log_prefix} >> NO_MODEL (skip={skip_pct:.0%} > 85%, avoidance not discrimination)")
                    _store(
                        {
                            "model_type": "NONE",
                            "reason": f"skip_too_high_{skip_pct:.0%}",
                            "test_auc": round(test_auc, 4),
                            "cpcv_auc": round(cpcv_auc, 4) if cpcv_auc else None,
                            "honest_delta_r": round(honest_delta_r, 2),
                        }
                    )
                    continue

                cpcv_str = f"{cpcv_auc:.3f}" if cpcv_auc else "  --"
                logger.info(
                    f"{log_prefix} >> ML t={best_t:.2f} "
                    f"CPCV={cpcv_str} TestAUC={test_auc:.3f} "
                    f"ValDelta={best_delta:+.1f} HonestDelta={honest_delta_r:+.1f} "
                    f"Skip={skip_pct:.0%} leaf={leaf_size}"
                )
                _store(
                    {
                        "model_type": "SESSION",
                        "model": rf,
                        "calibrator": calibrator,
                        "feature_names": feature_names,
                        "threshold": best_t,
                        "training_aperture": training_aperture,
                        "training_rr": training_rr,
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
                        "top_features": top_feats,
                    }
                )
            else:
                logger.info(f"{log_prefix} >> NO_MODEL (no threshold beats baseline, TestAUC={test_auc:.3f})")
                _store(
                    {
                        "model_type": "NONE",
                        "reason": "no_positive_threshold",
                        "test_auc": round(test_auc, 4),
                        "cpcv_auc": round(cpcv_auc, 4) if cpcv_auc else None,
                        "honest_delta_r": 0.0,
                    }
                )

    # --- Summary ---
    # Flatten results for counting (per-aperture has nested dicts)
    def _iter_results():
        for session, val in session_results.items():
            if per_aperture:
                for ak, info in val.items():
                    yield session, ak, info
            else:
                yield session, None, val

    all_results = list(_iter_results())
    n_ml = sum(1 for _, _, r in all_results if r["model_type"] == "SESSION")
    n_none = sum(1 for _, _, r in all_results if r["model_type"] == "NONE")
    total_val_delta = sum(r.get("val_delta_r", 0) for _, _, r in all_results if r["model_type"] == "SESSION")
    total_honest_delta = sum(r.get("honest_delta_r", 0) for _, _, r in all_results if r["model_type"] == "SESSION")

    # White (2000) / HLZ (2016): report ALL trained sessions, not just survivors.
    # Sessions with test_auc trained an RF — include their honest_delta_r even if
    # gates rejected them. Sessions without test_auc (no data / insufficient) use 0.
    total_full_delta = sum(r.get("honest_delta_r", 0) for _, _, r in all_results if r.get("test_auc") is not None)
    selection_uplift = total_honest_delta - total_full_delta

    unit_label = "models" if per_aperture else "ML sessions"
    logger.info(
        f"\n  SUMMARY: {n_ml} {unit_label}, {n_none} NO_MODEL"
        f"\n  Val delta (optimized):  {total_val_delta:+.1f}R"
        f"\n  Honest delta (deployed):{total_honest_delta:+.1f}R"
        f"\n  Full delta (all trained):{total_full_delta:+.1f}R"
        f"\n  Selection uplift:       {selection_uplift:+.1f}R"
    )
    if total_full_delta != 0 and abs(selection_uplift) > 0.5 * abs(total_full_delta):
        logger.warning(
            f"  ⚠ Selection uplift ({selection_uplift:+.1f}R) is >{50}% of full delta — "
            f"reported honest_delta_r is cherry-picked upward (White 2000)"
        )

    # --- Save bundle ---
    model_path = None
    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # Production path: always _hybrid.joblib (predict_live.py expects this).
        # Both single_config and multi-config per-session models are "hybrid"
        # from the predictor's perspective (dict of per-session sub-models).
        model_path = MODEL_DIR / f"meta_label_{instrument}_hybrid.joblib"
        config_hash = compute_config_hash()

        bundle = {
            "model_type": "single_config_per_session" if single_config else "hybrid_per_session",
            "bundle_format": "per_aperture" if per_aperture else "flat",
            "instrument": instrument,
            "sessions": {},
            "config_hash": config_hash,
            "trained_at": datetime.now(UTC).isoformat(),
            "data_date_range": (
                str(meta_all["trading_day"].min()),
                str(meta_all["trading_day"].max()),
            ),
            "single_config": single_config,
            "rr_target_lock": rr_target,
            "split_ratios": "60/20/20",
            "n_total_samples": n_total,
            "n_ml_sessions": n_ml,
            "n_none_sessions": n_none,
            "total_val_delta_r": round(total_val_delta, 2),
            "total_honest_delta_r": round(total_honest_delta, 2),
            "total_full_delta_r": round(total_full_delta, 2),
        }

        def _to_bundle_entry(info: dict) -> dict:
            """Convert a session result dict to bundle format."""
            if info["model_type"] == "SESSION":
                return {
                    "model": info["model"],
                    "calibrator": info.get("calibrator"),
                    "feature_names": info["feature_names"],
                    "optimal_threshold": info["threshold"],
                    "training_aperture": info["training_aperture"],
                    "training_rr": info["training_rr"],
                    "cpcv_auc": info.get("cpcv_auc"),
                    "test_auc": info["test_auc"],
                    "n_train": info["n_train"],
                    "n_val": info["n_val"],
                    "n_test": info["n_test"],
                    "val_delta_r": info["val_delta_r"],
                    "honest_delta_r": info["honest_delta_r"],
                    "skip_pct": info["skip_pct"],
                    "leaf_size": info["leaf_size"],
                    "top_features": info.get("top_features"),
                }
            entry = {
                "model": None,
                "model_type": "NONE",
                "reason": info.get("reason", "insufficient_data"),
            }
            if "honest_delta_r" in info:
                entry["honest_delta_r"] = info["honest_delta_r"]
            return entry

        for session, val in session_results.items():
            if per_aperture:
                # Nested: sessions[session][aperture_key] = {...}
                bundle["sessions"][session] = {ak: _to_bundle_entry(info) for ak, info in val.items()}
            else:
                # Flat: sessions[session] = {...}
                bundle["sessions"][session] = _to_bundle_entry(val)

        # --- Feature importance stability check (compare with previous model) ---
        if model_path.exists():
            try:
                old_bundle = joblib.load(model_path)
                old_sessions = old_bundle.get("sessions", {})
                old_is_per_aperture = old_bundle.get("bundle_format") == "per_aperture"

                def _check_stability(label: str, new_info: dict, old_info: dict) -> None:
                    if new_info.get("top_features") is None:
                        return
                    old_top = old_info.get("top_features")
                    if old_top is None:
                        return
                    old_names = [f[0] for f in old_top[:10]]
                    new_names = [f[0] for f in new_info["top_features"][:10]]
                    drifted = []
                    for i, name in enumerate(new_names):
                        if name in old_names:
                            old_rank = old_names.index(name)
                            shift = abs(i - old_rank)
                            if shift > 3:
                                drifted.append(f"{name} #{old_rank + 1}->{i + 1}")
                        else:
                            drifted.append(f"{name} NEW(#{i + 1})")
                    if drifted:
                        logger.warning(f"  {label} FEATURE DRIFT: {', '.join(drifted)}")
                    else:
                        logger.info(f"  {label} feature importance stable")

                for sess, new_val in bundle["sessions"].items():
                    old_val = old_sessions.get(sess, {})
                    if per_aperture:
                        for ak, new_info in new_val.items():
                            old_info = old_val.get(ak, {}) if old_is_per_aperture else {}
                            _check_stability(f"{sess} {ak}", new_info, old_info)
                    else:
                        _check_stability(sess, new_val, old_val)
            except Exception:
                logger.warning(
                    "Could not load previous model for stability check",
                    exc_info=True,
                )

        joblib.dump(bundle, model_path)
        logger.info(f"  Hybrid model saved: {model_path}")

    # Build return-dict sessions (summary info only — no model objects)
    def _to_return_entry(info: dict) -> dict:
        if info["model_type"] == "SESSION":
            return {
                "model_type": "SESSION",
                "threshold": info["threshold"],
                "cpcv_auc": info.get("cpcv_auc"),
                "test_auc": info["test_auc"],
                "val_delta_r": info["val_delta_r"],
                "honest_delta_r": info["honest_delta_r"],
                "skip_pct": info["skip_pct"],
            }
        entry = {"model_type": "NONE", "reason": info.get("reason", "insufficient_data")}
        if "honest_delta_r" in info:
            entry["honest_delta_r"] = info["honest_delta_r"]
        return entry

    return_sessions: dict = {}
    for session, val in session_results.items():
        if per_aperture:
            return_sessions[session] = {ak: _to_return_entry(info) for ak, info in val.items()}
        else:
            return_sessions[session] = _to_return_entry(val)

    return {
        "status": "trained",
        "instrument": instrument,
        "model_type": "hybrid_per_session",
        "n_samples": n_total,
        "n_ml_sessions": n_ml,
        "n_none_sessions": n_none,
        "total_val_delta_r": round(total_val_delta, 2),
        "total_honest_delta_r": round(total_honest_delta, 2),
        "total_full_delta_r": round(total_full_delta, 2),
        "per_aperture": per_aperture,
        "sessions": return_sessions,
        "model_path": str(model_path) if model_path else None,
    }


def print_per_session_results(results: dict) -> None:
    """Print formatted per-session training results."""
    if results["status"] != "trained":
        print(f"  {results.get('instrument', '?')}: {results['status']}")
        return

    inst = results["instrument"]
    is_per_aperture = results.get("per_aperture", False)
    mode_label = "PER-APERTURE" if is_per_aperture else "PER-SESSION"

    print(f"\n{'=' * 70}")
    print(f"  HYBRID {mode_label} RESULTS -- {inst} (3-way split: 60/20/20)")
    print(f"{'=' * 70}")
    print(f"  Samples: {results['n_samples']:,d}")
    unit = "models" if is_per_aperture else "ML sessions"
    print(f"  {unit}: {results['n_ml_sessions']} | NO_MODEL: {results['n_none_sessions']}")
    print(f"  Val delta (optimized):  {results['total_val_delta_r']:+.1f}R")
    print(f"  Honest delta (deployed):{results['total_honest_delta_r']:+.1f}R")
    full_delta = results.get("total_full_delta_r")
    if full_delta is not None:
        uplift = results["total_honest_delta_r"] - full_delta
        print(f"  Full delta (all trained):{full_delta:+.1f}R  (selection uplift: {uplift:+.1f}R)")
    print()

    def _print_row(label: str, info: dict) -> None:
        if info["model_type"] == "SESSION":
            cpcv_str = f"{info['cpcv_auc']:.3f}" if info.get("cpcv_auc") else "  --"
            print(
                f"  {label:<26} {'ML':>5} {info['threshold']:>6.2f} "
                f"{cpcv_str:>6} {info['test_auc']:>6.3f} "
                f"{info['val_delta_r']:>+7.1f} {info['honest_delta_r']:>+7.1f} "
                f"{info['skip_pct']:>5.1%}"
            )
        else:
            print(f"  {label:<26} {'NONE':>5} {'--':>6} {'--':>6} {'--':>6} {'--':>7} {'--':>7} {'--':>6}")

    print(
        f"  {'SESSION':<26} {'TYPE':>5} {'THRESH':>6} {'CPCV':>6} {'T_AUC':>6} {'VAL_dR':>7} {'OOS_dR':>7} {'SKIP%':>6}"
    )
    print(f"  {'-' * 26} {'-' * 5} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 6}")

    all_sessions = list(SESSION_CHRONOLOGICAL_ORDER) + sorted(
        s for s in results["sessions"] if s not in SESSION_CHRONOLOGICAL_ORDER
    )
    for session in all_sessions:
        if session not in results["sessions"]:
            continue
        val = results["sessions"][session]
        if is_per_aperture:
            for ak in sorted(val.keys()):
                _print_row(f"{session} {ak}", val[ak])
        else:
            _print_row(session, val)

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

    logger.info(f"Split: train={n_train_end:,d} / val={n_val_end - n_train_end:,d} / test={n_total - n_val_end:,d}")

    # --- CPCV Validation (on training data only) ---
    cpcv_results = None
    if run_cpcv:
        logger.info("Running CPCV validation on training data...")
        cpcv_results = cpcv_score(
            RandomForestClassifier,
            RF_PARAMS,
            X_train,
            y_train,
            meta["trading_day"].iloc[:n_train_end],
            max_splits=max_cpcv_splits,
        )
        logger.info(
            f"CPCV AUC: {cpcv_results['auc_mean']:.4f} +/- {cpcv_results['auc_std']:.4f} "
            f"({cpcv_results['n_splits']} splits)"
        )

        if cpcv_results["auc_mean"] < 0.505:
            logger.warning(
                f"CPCV AUC {cpcv_results['auc_mean']:.4f} is barely above random. "
                f"Meta-label may not add value for {instrument}."
            )

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
    logger.info(
        f"Val optimal:  t={opt['threshold']:.2f} | avgR={opt['avg_r']:.4f} | "
        f"Sharpe={opt['sharpe']:.3f} | skip={opt['skip_pct']:.1%} | N={opt['n_kept']:,d}"
    )

    # --- Honest OOS evaluation on TEST set (final 20%) ---
    test_prob = rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_prob)

    # Apply val-optimized threshold to frozen test set
    test_kept = test_prob >= opt["threshold"]
    test_base_r = float(pnl_test.sum())
    test_filt_r = float(pnl_test[test_kept].sum()) if test_kept.sum() > 0 else 0.0
    test_delta_r = test_filt_r - test_base_r
    test_skip_pct = 1 - test_kept.sum() / len(pnl_test) if len(pnl_test) > 0 else 0.0

    logger.info(
        f"Honest OOS: AUC={test_auc:.4f} | BaseR={test_base_r:+.1f} | "
        f"Delta={test_delta_r:+.1f} | Skip={test_skip_pct:.1%}"
    )

    # --- Save model ---
    model_path = None
    if save_model:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / f"meta_label_{instrument}.joblib"
        config_hash = compute_config_hash()

        joblib.dump(
            {
                "model": rf,
                "feature_names": list(X.columns),
                "instrument": instrument,
                "n_train": n_train_end,
                "split_ratios": "60/20/20",
                "oos_auc": test_auc,
                "optimal_threshold": opt["threshold"],
                "cpcv_auc": cpcv_results["auc_mean"] if cpcv_results else None,
                "honest_delta_r": round(test_delta_r, 2),
                "trained_at": datetime.now(UTC).isoformat(),
                "data_date_range": (
                    str(meta["trading_day"].min()),
                    str(meta["trading_day"].max()),
                ),
                "config_hash": config_hash,
                "validated_only": validated_only,
                "n_total_samples": n_total,
            },
            model_path,
        )
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
        session_breakdown.append(
            {
                "session": session,
                "n_total": smask.sum(),
                "n_kept": kept.sum(),
                "skip_pct": round(1 - kept.sum() / smask.sum(), 3),
                "base_avgR": round(base_avg, 4),
                "filt_avgR": round(filt_avg, 4),
                "lift": round(filt_avg - base_avg, 4),
            }
        )

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
            print(
                f"  {s['session']:<20} {s['n_total']:>6d} {s['n_kept']:>6d} "
                f"{s['skip_pct']:>6.1%} {s['base_avgR']:>+8.4f} {s['filt_avgR']:>+8.4f} "
                f"{s['lift']:>+8.4f}"
            )

    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Meta-Label Classifier Training")
    parser.add_argument("--instrument", type=str, help="Single instrument")
    parser.add_argument("--all", action="store_true", help="All active instruments")
    parser.add_argument("--no-cpcv", action="store_true", help="Skip CPCV (faster)")
    parser.add_argument("--max-cpcv-splits", type=int, default=20, help="Max CPCV splits (default 20 of 45)")
    parser.add_argument("--db-path", type=str, default=str(GOLD_DB_PATH))
    parser.add_argument(
        "--validated-only", action="store_true", help="Train only on validated strategy outcomes (recommended)"
    )
    parser.add_argument(
        "--per-session",
        action="store_true",
        help="Train hybrid per-session models (recommended, implies --validated-only)",
    )
    parser.add_argument(
        "--single-config", action="store_true", help="Single config per session: 1 row per (day, session), clean labels"
    )
    parser.add_argument(
        "--rr-target",
        type=float,
        default=None,
        help="Lock to a specific RR target (e.g. 2.0). Default: best per session",
    )
    parser.add_argument("--sweep-rr", action="store_true", help="Sweep all validated RR targets and compare honest OOS")
    parser.add_argument(
        "--config-selection",
        type=str,
        default="max_samples",
        choices=["max_samples", "best_sharpe"],
        help="How to pick one config per session: max_samples (most data, "
        "recommended) or best_sharpe (highest Sharpe, fewest samples)",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip filter eligibility — ML trains on ALL break days and "
        "learns to discriminate via features (orb_size, atr, etc.)",
    )
    parser.add_argument(
        "--per-aperture",
        action="store_true",
        help="Train separate model per (session, aperture). Fixes aperture covariate shift. Requires --single-config.",
    )
    args = parser.parse_args()

    if args.per_aperture and not args.single_config:
        parser.error("--per-aperture requires --single-config")

    instruments = ACTIVE_INSTRUMENTS if args.all else [args.instrument or "MGC"]

    for inst in instruments:
        if args.sweep_rr:
            # Sweep all validated RR targets for this instrument
            import duckdb as _ddb

            from pipeline.db_config import configure_connection as _cc

            _con = _ddb.connect(args.db_path, read_only=True)
            _cc(_con)
            try:
                rr_vals = (
                    _con.execute(
                        "SELECT DISTINCT rr_target FROM validated_setups "
                        "WHERE instrument = ? AND status = 'active' ORDER BY rr_target",
                        [inst],
                    )
                    .fetchdf()["rr_target"]
                    .tolist()
                )
            finally:
                _con.close()

            sweep_results = {}
            for rr in rr_vals:
                logger.info(f"\n{'#' * 60}")
                logger.info(f"  SWEEP: {inst} RR={rr}")
                logger.info(f"{'#' * 60}")
                try:
                    results = train_per_session_meta_label(
                        inst,
                        args.db_path,
                        single_config=True,
                        rr_target=rr,
                        save_model=False,  # Don't save during sweep
                        run_cpcv=not args.no_cpcv,
                        config_selection=args.config_selection,
                        skip_filter=args.skip_filter,
                        per_aperture=args.per_aperture,
                    )
                    sweep_results[rr] = results
                    print_per_session_results(results)
                except Exception as e:
                    logger.warning(f"  RR={rr}: {e}")
                    sweep_results[rr] = {"status": "error", "error": str(e)}

            # Print sweep comparison
            print(f"\n{'=' * 70}")
            print(f"  SWEEP COMPARISON — {inst}")
            print(f"{'=' * 70}")
            print(f"  {'RR':>4} {'N':>7} {'ML_SESS':>8} {'VAL_dR':>8} {'OOS_dR':>8} {'FULL_dR':>8} {'STATUS':>8}")
            print(
                f"  {'----':>4} {'-------':>7} {'--------':>8} {'--------':>8} {'--------':>8} {'--------':>8} {'--------':>8}"
            )
            for rr in sorted(sweep_results.keys()):
                r = sweep_results[rr]
                if r.get("status") == "trained":
                    full_dr = r.get("total_full_delta_r")
                    full_str = f"{full_dr:>+8.1f}" if full_dr is not None else f"{'--':>8}"
                    print(
                        f"  {rr:>4.1f} {r['n_samples']:>7,d} "
                        f"{r['n_ml_sessions']:>8d} "
                        f"{r['total_val_delta_r']:>+8.1f} "
                        f"{r['total_honest_delta_r']:>+8.1f} "
                        f"{full_str} "
                        f"{'OK':>8}"
                    )
                else:
                    print(f"  {rr:>4.1f} {'--':>7} {'--':>8} {'--':>8} {'--':>8} {'--':>8} {'SKIP':>8}")
            print(f"{'=' * 70}\n")

        elif args.per_session or args.single_config:
            results = train_per_session_meta_label(
                inst,
                args.db_path,
                single_config=args.single_config,
                rr_target=args.rr_target,
                run_cpcv=not args.no_cpcv,
                config_selection=args.config_selection,
                skip_filter=args.skip_filter,
                per_aperture=args.per_aperture,
            )
            print_per_session_results(results)
        else:
            results = train_meta_label(
                inst,
                args.db_path,
                run_cpcv=not args.no_cpcv,
                max_cpcv_splits=args.max_cpcv_splits,
                validated_only=args.validated_only,
            )
            print_results(results)


if __name__ == "__main__":
    main()
