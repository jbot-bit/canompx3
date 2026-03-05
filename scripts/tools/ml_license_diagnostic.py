"""ML License Diagnostic: understand WHY the model works (or doesn't) per instrument.

Three-phase analysis:
  Phase 1: Feature importance per instrument × session — what is the RF ACTUALLY learning?
  Phase 2: Honest out-of-time test — train pre-2025, freeze threshold, test 2025+
  Phase 3: Per-instrument signal strength — AUC distribution, feature ablation

Output: structured findings that tell us WHERE signal lives and WHERE it's noise.

Usage:
    python scripts/tools/ml_license_diagnostic.py --db-path C:/db/gold.db
    python scripts/tools/ml_license_diagnostic.py --db-path C:/db/gold.db --instrument MNQ
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.paths import GOLD_DB_PATH
from trading_app.ml.config import (
    ACTIVE_INSTRUMENTS,
    CROSS_SESSION_FEATURES,
    LEVEL_PROXIMITY_FEATURES,
    MAX_EARLY_SESSION_INDEX,
    MIN_SESSION_SAMPLES,
    RF_PARAMS,
    SESSION_CHRONOLOGICAL_ORDER,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
)
from trading_app.ml.features import (
    apply_e6_filter,
    load_validated_feature_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_session_features(X_e6: pd.DataFrame, session: str) -> list[str]:
    """Get feature list appropriate for this session (drop cross-session for early sessions)."""
    session_idx = SESSION_CHRONOLOGICAL_ORDER.index(session) if session in SESSION_CHRONOLOGICAL_ORDER else 99
    drop_cols = set()
    if session_idx <= MAX_EARLY_SESSION_INDEX:
        drop_cols.update(CROSS_SESSION_FEATURES)
        if session_idx == 0:
            drop_cols.update(LEVEL_PROXIMITY_FEATURES)
    return [c for c in X_e6.columns if c not in drop_cols]


def phase1_feature_importance(X_e6, y, meta, instrument: str):
    """Phase 1: What features matter for each session?"""
    print(f"\n{'=' * 80}")
    print(f"  PHASE 1: FEATURE IMPORTANCE — {instrument}")
    print(f"{'=' * 80}")

    sessions = meta["orb_label"].unique()
    all_importances = {}

    for session in sorted(sessions):
        mask = meta["orb_label"] == session
        n = mask.sum()
        if n < MIN_SESSION_SAMPLES:
            continue

        X_sess = X_e6.loc[mask]
        y_sess = y[mask]
        feature_names = _get_session_features(X_e6, session)
        X_feat = X_sess[feature_names]

        # Train RF on full session data (this is for importance analysis, not prediction)
        rf = RandomForestClassifier(**{**RF_PARAMS, "n_estimators": 200})
        rf.fit(X_feat, y_sess)

        importances = dict(zip(feature_names, rf.feature_importances_))
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        all_importances[session] = sorted_imp

        print(f"\n  {session} (N={n:,d}, win_rate={y_sess.mean():.1%})")
        print(f"  {'Feature':<35} {'Importance':>10}  {'Cum%':>6}")
        print(f"  {'-' * 55}")
        cum = 0
        for feat, imp in sorted_imp[:15]:
            cum += imp
            print(f"  {feat:<35} {imp:>10.4f}  {cum:>5.1%}")

    # Cross-instrument feature frequency analysis
    print(f"\n  TOP FEATURES ACROSS ALL SESSIONS:")
    print(f"  {'Feature':<35} {'Avg Importance':>14}  {'#Sessions':>9}")
    print(f"  {'-' * 62}")

    feature_scores = {}
    for session, sorted_imp in all_importances.items():
        for feat, imp in sorted_imp:
            if feat not in feature_scores:
                feature_scores[feat] = []
            feature_scores[feat].append(imp)

    avg_scores = {f: np.mean(scores) for f, scores in feature_scores.items()}
    for feat, avg in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:20]:
        n_sess = len(feature_scores[feat])
        print(f"  {feat:<35} {avg:>14.4f}  {n_sess:>9}")

    return all_importances


def phase2_honest_oos(X_e6, y, meta, instrument: str):
    """Phase 2: Honest out-of-time test.

    Train: all data before 2025
    Validation: last 20% of training data (for threshold selection)
    Test: 2025+ data (FROZEN threshold — no peeking)
    """
    print(f"\n{'=' * 80}")
    print(f"  PHASE 2: HONEST OUT-OF-TIME TEST — {instrument}")
    print(f"{'=' * 80}")

    trading_days = meta["trading_day"]
    pnl_r = meta["pnl_r"].values

    # Split: pre-2025 = train+val, 2025+ = test
    cutoff = pd.Timestamp("2025-01-01")
    train_val_mask = trading_days < cutoff
    test_mask = trading_days >= cutoff

    n_train_val = train_val_mask.sum()
    n_test = test_mask.sum()

    if n_test < 50:
        print(f"  SKIP: Only {n_test} test samples after 2025-01-01")
        return None

    print(f"  Train+Val: {n_train_val:,d} samples (pre-2025)")
    print(f"  Test:      {n_test:,d} samples (2025+)")

    # Within train+val, use last 20% for validation (threshold selection)
    train_val_indices = np.where(train_val_mask)[0]
    n_tv = len(train_val_indices)
    n_train = int(n_tv * 0.8)
    train_idx = train_val_indices[:n_train]
    val_idx = train_val_indices[n_train:]
    test_idx = np.where(test_mask)[0]

    sessions = meta["orb_label"].unique()
    results = {}

    print(f"\n  {'Session':<22} {'N_train':>7} {'N_val':>6} {'N_test':>6} "
          f"{'AUC_val':>7} {'AUC_test':>8} {'Thresh':>6} "
          f"{'BaseR':>7} {'DeltaR':>7} {'Skip%':>6} {'VERDICT':>8}")
    print(f"  {'-' * 100}")

    total_base_r = 0.0
    total_filtered_r = 0.0
    total_test_n = 0

    for session in sorted(sessions):
        sess_mask = (meta["orb_label"] == session).values
        sess_train = np.intersect1d(train_idx, np.where(sess_mask)[0])
        sess_val = np.intersect1d(val_idx, np.where(sess_mask)[0])
        sess_test = np.intersect1d(test_idx, np.where(sess_mask)[0])

        if len(sess_train) < MIN_SESSION_SAMPLES or len(sess_test) < 20:
            continue

        feature_names = _get_session_features(X_e6, session)
        X_train = X_e6.iloc[sess_train][feature_names]
        X_val = X_e6.iloc[sess_val][feature_names]
        X_test = X_e6.iloc[sess_test][feature_names]
        y_train = y.iloc[sess_train] if hasattr(y, 'iloc') else y[sess_train]
        y_val = y.iloc[sess_val] if hasattr(y, 'iloc') else y[sess_val]
        y_test = y.iloc[sess_test] if hasattr(y, 'iloc') else y[sess_test]

        leaf_size = max(20, min(100, len(sess_train) // 20))
        rf = RandomForestClassifier(**{**RF_PARAMS, "min_samples_leaf": leaf_size})
        rf.fit(X_train, y_train)

        # Val AUC + threshold selection
        val_prob = rf.predict_proba(X_val)[:, 1]
        val_pnl = pnl_r[sess_val]
        try:
            auc_val = roc_auc_score(y_val, val_prob)
        except ValueError:
            continue

        # Find best threshold on VALIDATION set only
        base_r_val = float(val_pnl.sum())
        best_t = None
        best_delta_val = 0.0
        for t in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP):
            kept = val_prob >= t
            if kept.sum() < 20:
                continue
            delta = float(val_pnl[kept].sum()) - base_r_val
            if delta > best_delta_val:
                best_delta_val = delta
                best_t = round(t, 2)

        # Test with FROZEN threshold — no optimization on test set
        test_prob = rf.predict_proba(X_test)[:, 1]
        test_pnl = pnl_r[sess_test]
        try:
            auc_test = roc_auc_score(y_test, test_prob)
        except ValueError:
            auc_test = 0.5

        base_r_test = float(test_pnl.sum())
        total_base_r += base_r_test
        total_test_n += len(sess_test)

        if best_t is not None:
            kept_test = test_prob >= best_t
            filtered_r = float(test_pnl[kept_test].sum())
            delta_r = filtered_r - base_r_test
            skip_pct = 1 - kept_test.sum() / len(sess_test)
            total_filtered_r += filtered_r
        else:
            # No threshold found on val — take all trades
            delta_r = 0.0
            skip_pct = 0.0
            filtered_r = base_r_test
            total_filtered_r += filtered_r
            best_t = 0.0

        # Verdict
        if delta_r > 5 and auc_test > 0.55:
            verdict = "REAL"
        elif delta_r > 0 and auc_test > 0.52:
            verdict = "WEAK"
        elif delta_r <= 0:
            verdict = "NO-GO"
        else:
            verdict = "NOISE"

        results[session] = {
            "auc_val": auc_val, "auc_test": auc_test,
            "threshold": best_t, "delta_r": delta_r,
            "base_r": base_r_test, "n_test": len(sess_test),
            "verdict": verdict,
        }

        print(f"  {session:<22} {len(sess_train):>7,d} {len(sess_val):>6,d} {len(sess_test):>6,d} "
              f"{auc_val:>7.3f} {auc_test:>8.3f} {best_t:>6.2f} "
              f"{base_r_test:>+7.1f} {delta_r:>+7.1f} {skip_pct:>5.0%} "
              f"{verdict:>8}")

    total_delta = total_filtered_r - total_base_r
    print(f"\n  TOTAL: BaseR={total_base_r:+.1f}  FilteredR={total_filtered_r:+.1f}  "
          f"Delta={total_delta:+.1f}  N_test={total_test_n:,d}")

    # Summary verdict
    real_sessions = [s for s, r in results.items() if r["verdict"] == "REAL"]
    weak_sessions = [s for s, r in results.items() if r["verdict"] == "WEAK"]
    nogo_sessions = [s for s, r in results.items() if r["verdict"] == "NO-GO"]

    print(f"\n  VERDICTS:")
    print(f"    REAL signal: {real_sessions or 'NONE'}")
    print(f"    WEAK signal: {weak_sessions or 'NONE'}")
    print(f"    NO-GO:       {nogo_sessions or 'NONE'}")

    if total_delta > 0:
        print(f"\n  OVERALL: ML ADDS VALUE (+{total_delta:.1f}R on honest OOS)")
    else:
        print(f"\n  OVERALL: ML DOES NOT ADD VALUE ({total_delta:+.1f}R on honest OOS)")

    return results


def phase3_signal_diagnosis(X_e6, y, meta, instrument: str, oos_results: dict | None):
    """Phase 3: What KIND of signal does the RF find?"""
    print(f"\n{'=' * 80}")
    print(f"  PHASE 3: SIGNAL DIAGNOSIS — {instrument}")
    print(f"{'=' * 80}")

    # For each session with signal: what feature groups matter?
    feature_groups = {
        "volatility": ["atr_20", "atr_vel_ratio", "garch_atr_ratio", "garch_forecast_vol",
                        "prev_day_range", "overnight_range"],
        "orb_quality": ["orb_size", "orb_volume", "orb_break_bar_volume",
                         "orb_break_delay_min", "orb_break_bar_continues"],
        "direction": ["rsi_14_at_CME_REOPEN", "gap_open_points", "orb_break_dir_LONG",
                       "orb_break_dir_SHORT", "entry_model_E1", "entry_model_E2"],
        "calendar": ["day_of_week", "is_friday", "is_monday"],
        "cross_session": list(CROSS_SESSION_FEATURES) + list(LEVEL_PROXIMITY_FEATURES),
        "trade_config": ["rr_target", "confirm_bars", "orb_minutes"],
    }

    sessions = meta["orb_label"].unique()
    for session in sorted(sessions):
        mask = (meta["orb_label"] == session).values
        n = mask.sum()
        if n < MIN_SESSION_SAMPLES:
            continue

        # Check OOS verdict if available
        verdict = ""
        if oos_results and session in oos_results:
            verdict = f" [{oos_results[session]['verdict']}]"

        feature_names = _get_session_features(X_e6, session)
        X_sess = X_e6.loc[mask, feature_names]
        y_sess = y[mask] if not hasattr(y, 'iloc') else y.iloc[np.where(mask)[0]]

        rf = RandomForestClassifier(**{**RF_PARAMS, "n_estimators": 200})
        rf.fit(X_sess, y_sess)

        importances = dict(zip(feature_names, rf.feature_importances_))

        # Group importances
        group_scores = {}
        for group_name, group_features in feature_groups.items():
            score = sum(importances.get(f, 0) for f in group_features)
            group_scores[group_name] = score

        total_imp = sum(group_scores.values())
        remaining = 1.0 - total_imp if total_imp < 1.0 else 0.0

        print(f"\n  {session}{verdict} (N={n:,d})")
        for group, score in sorted(group_scores.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(score * 50)
            print(f"    {group:<16} {score:>6.1%}  {bar}")
        if remaining > 0.01:
            bar = "#" * int(remaining * 50)
            print(f"    {'other':<16} {remaining:>6.1%}  {bar}")

    # Per-instrument summary
    print(f"\n  INSTRUMENT DIAGNOSIS:")
    total_n = len(y)
    win_rate = y.mean() if hasattr(y, 'mean') else np.mean(y)
    n_sessions = len([s for s in sessions if (meta["orb_label"] == s).sum() >= MIN_SESSION_SAMPLES])
    print(f"    Total samples: {total_n:,d}")
    print(f"    Win rate: {win_rate:.1%}")
    print(f"    Sessions with enough data: {n_sessions}")

    if oos_results:
        real = sum(1 for r in oos_results.values() if r["verdict"] == "REAL")
        total_delta = sum(r["delta_r"] for r in oos_results.values())
        print(f"    Sessions with REAL signal: {real}/{len(oos_results)}")
        print(f"    Total honest OOS delta: {total_delta:+.1f}R")


def main():
    parser = argparse.ArgumentParser(description="ML License Diagnostic")
    parser.add_argument("--db-path", type=str, default=str(GOLD_DB_PATH))
    parser.add_argument("--instrument", type=str, help="Single instrument (default: all)")
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else list(ACTIVE_INSTRUMENTS)

    grand_summary = {}

    for instrument in instruments:
        print(f"\n{'#' * 80}")
        print(f"  ML LICENSE DIAGNOSTIC: {instrument}")
        print(f"{'#' * 80}")

        X_all, y_all, meta_all = load_validated_feature_matrix(args.db_path, instrument)
        X_e6 = apply_e6_filter(X_all)

        logger.info(f"{instrument}: {len(X_e6):,d} samples, {X_e6.shape[1]} features, "
                     f"win_rate={y_all.mean():.1%}")

        # Phase 1: Feature importance
        importances = phase1_feature_importance(X_e6, y_all, meta_all, instrument)

        # Phase 2: Honest OOS
        oos_results = phase2_honest_oos(X_e6, y_all, meta_all, instrument)

        # Phase 3: Signal diagnosis
        phase3_signal_diagnosis(X_e6, y_all, meta_all, instrument, oos_results)

        grand_summary[instrument] = oos_results

    # Grand summary
    print(f"\n{'#' * 80}")
    print(f"  GRAND SUMMARY: ML LICENSE STATUS")
    print(f"{'#' * 80}")

    print(f"\n  {'Instrument':<12} {'REAL':>6} {'WEAK':>6} {'NO-GO':>6} {'Total Delta':>12} {'Verdict':>10}")
    print(f"  {'-' * 58}")

    for inst, results in grand_summary.items():
        if results is None:
            print(f"  {inst:<12} {'N/A':>6} {'N/A':>6} {'N/A':>6} {'N/A':>12} {'SKIP':>10}")
            continue
        real = sum(1 for r in results.values() if r["verdict"] == "REAL")
        weak = sum(1 for r in results.values() if r["verdict"] == "WEAK")
        nogo = sum(1 for r in results.values() if r["verdict"] == "NO-GO")
        total_delta = sum(r["delta_r"] for r in results.values())

        if total_delta > 20 and real >= 2:
            verdict = "LICENSE"
        elif total_delta > 0 and (real + weak) >= 1:
            verdict = "LEARNER"
        else:
            verdict = "NO-GO"

        print(f"  {inst:<12} {real:>6} {weak:>6} {nogo:>6} {total_delta:>+12.1f}R {verdict:>10}")

    print()


if __name__ == "__main__":
    main()
