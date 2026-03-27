"""Bootstrap permutation test for ML meta-label survivors.

Tests whether ML honest_delta_r is genuinely better than chance by shuffling
labels and re-running the full train+threshold pipeline N times.

This is the test that caught the negative-baseline threshold artifact (p=0.35).
Must be applied to ALL ML survivors before trusting any result.

Usage:
    python scripts/tools/ml_bootstrap_test.py
"""

import logging
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from pipeline.paths import GOLD_DB_PATH
from trading_app.ml.config import (
    RF_PARAMS,
)
from trading_app.ml.features import apply_e6_filter, load_single_config_feature_matrix
from trading_app.ml.meta_label import _get_session_features, _optimize_threshold_profit

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "ml_bootstrap_results.log"),
    ],
)
log = logging.getLogger(__name__)

# V1 survivors (historical reference only — replaced by V2 config selection)
_V1_SURVIVORS = [
    ("NYSE_OPEN", 30, 2.0, 33.5),
    ("US_DATA_1000", 30, 2.0, 38.5),
    ("US_DATA_1000", 15, 2.0, 10.6),
    ("US_DATA_830", 30, 2.0, 12.5),
    ("NYSE_OPEN", None, 2.0, 3.3),
    ("CME_PRECLOSE", None, 2.0, 2.2),
    ("CME_PRECLOSE", None, 1.5, 5.4),
]


def load_selected_configs() -> list[tuple[str, int | None, float, float]]:
    """Load V2 selected configs from retrain results JSON.

    Returns list of (session, aperture_or_None, rr_target, honest_delta_r).
    Raises FileNotFoundError if retrain hasn't been run yet.
    """
    import json

    json_path = Path(__file__).resolve().parent.parent.parent / "logs" / "ml_v2_retrain_results.json"
    if not json_path.exists():
        raise FileNotFoundError(f"V2 retrain results not found at {json_path}. Run ml_v2_retrain_all.py first.")
    with open(json_path) as f:
        data = json.load(f)

    configs = data.get("selected_configs", [])
    if not configs:
        log.warning("V2 retrain produced 0 selected configs — ML DEAD before bootstrap")
        return []

    return [(c["session"], c["aperture"], c["rr"], c["honest_delta_r"]) for c in configs]


# Active survivors: loaded from V2 retrain results at runtime
SURVIVORS: list[tuple[str, int | None, float, float]] = []

N_PERMUTATIONS = 5000  # 5000 per Phipson & Smyth 2010: reliable p-values need 1000+ perms


def bootstrap_one(
    session: str,
    aperture: int | None,
    rr_target: float,
    real_delta: float,
    n_perms: int = N_PERMUTATIONS,
) -> dict:
    """Run bootstrap permutation test for one (session, aperture, RR) combo.

    Shuffles win/loss labels N times, trains RF + optimizes threshold each time.
    Compares real honest_delta to null distribution.
    """
    log.info(f"\n{'=' * 60}")
    ap_str = f" O{aperture}" if aperture else ""
    log.info(f"  BOOTSTRAP: {session}{ap_str} RR{rr_target} (real delta={real_delta:+.1f}R)")
    log.info(f"  Permutations: {n_perms}")
    log.info(f"{'=' * 60}")

    # Load data — same pipeline as training
    X_all, y_all, meta_all = load_single_config_feature_matrix(
        str(GOLD_DB_PATH),
        "MNQ",
        rr_target=rr_target,
        config_selection="max_samples",
        skip_filter=True,
        per_aperture=(aperture is not None),
        apply_rr_lock=False,
        bypass_validated=True,  # V2: match training data source (full universe)
    )
    X_e6 = apply_e6_filter(X_all)

    # V2 methodology: select only expert-prior features (same as training)
    from trading_app.ml.config import ML_CORE_FEATURES

    available_core = [f for f in ML_CORE_FEATURES if f in X_e6.columns]
    X_e6 = X_e6[available_core]
    log.info(f"  V2 core features: {len(available_core)} ({available_core})")

    pnl_r = meta_all["pnl_r"].values

    # Filter to this session (and aperture if per-aperture)
    smask = (meta_all["orb_label"] == session).values
    if aperture is not None:
        smask = smask & (meta_all["orb_minutes"] == aperture).values

    session_indices = np.where(smask)[0]
    n_session = len(session_indices)

    if n_session == 0:
        log.warning(f"  No data for {session}{ap_str}")
        return {"session": session, "aperture": aperture, "rr": rr_target, "error": "no_data"}

    # 3-way split (same boundaries as training)
    n_total = len(X_e6)
    n_train_end = int(n_total * 0.60)
    n_val_end = int(n_total * 0.80)

    train_idx = session_indices[session_indices < n_train_end]
    val_idx = session_indices[(session_indices >= n_train_end) & (session_indices < n_val_end)]
    test_idx = session_indices[session_indices >= n_val_end]

    if len(train_idx) < 50 or len(val_idx) < 20 or len(test_idx) < 20:
        log.warning(f"  Insufficient data: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
        return {"session": session, "aperture": aperture, "rr": rr_target, "error": "insufficient_splits"}

    log.info(f"  Data: {n_session} total, train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Get session-appropriate features
    X_session = _get_session_features(X_e6, session)
    # Fix C sync: detect constants on train split only (matches meta_label.py)
    session_data = X_session.iloc[train_idx]
    const_cols = [c for c in X_session.columns if session_data[c].nunique() <= 1]
    if const_cols:
        X_session = X_session.drop(columns=const_cols)

    # RF params
    rf_base_params = {k: v for k, v in RF_PARAMS.items() if k not in ("min_samples_leaf", "random_state")}
    leaf_size = max(20, min(100, len(train_idx) // 20))

    # --- Compute real delta (verify it matches sweep) ---
    rf_real = RandomForestClassifier(**rf_base_params, min_samples_leaf=leaf_size, random_state=42)
    rf_real.fit(X_session.iloc[train_idx], y_all.iloc[train_idx])

    val_prob = rf_real.predict_proba(X_session.iloc[val_idx])[:, 1]
    val_min_kept = max(50, int(len(val_idx) * 0.15))
    best_t, _ = _optimize_threshold_profit(val_prob, pnl_r[val_idx], min_kept=val_min_kept)

    test_pnl = pnl_r[test_idx]
    if best_t is not None:
        test_prob = rf_real.predict_proba(X_session.iloc[test_idx])[:, 1]
        kept = test_prob >= best_t
        verified_delta = float(test_pnl[kept].sum()) - float(test_pnl.sum())
    else:
        verified_delta = 0.0

    log.info(f"  Verified real delta: {verified_delta:+.1f}R (sweep reported: {real_delta:+.1f}R)")

    # --- Permutation loop ---
    null_deltas = []
    start = time.time()

    for i in range(n_perms):
        # Shuffle train labels ONLY (val/test labels stay real for honest eval)
        shuffled_y_train = y_all.iloc[train_idx].values.copy()
        np.random.shuffle(shuffled_y_train)

        # Train RF on shuffled labels
        rf_null = RandomForestClassifier(
            **rf_base_params,
            min_samples_leaf=leaf_size,
            random_state=i,  # Different seed per permutation
        )
        rf_null.fit(X_session.iloc[train_idx], shuffled_y_train)

        # Optimize threshold on val set (with REAL pnl_r — we're testing discrimination, not luck)
        null_val_prob = rf_null.predict_proba(X_session.iloc[val_idx])[:, 1]
        null_t, _ = _optimize_threshold_profit(null_val_prob, pnl_r[val_idx], min_kept=val_min_kept)

        if null_t is not None:
            null_test_prob = rf_null.predict_proba(X_session.iloc[test_idx])[:, 1]
            null_kept = null_test_prob >= null_t
            null_delta = float(test_pnl[null_kept].sum()) - float(test_pnl.sum())
        else:
            null_delta = 0.0

        null_deltas.append(null_delta)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n_perms - i - 1) / rate
            log.info(f"    {i + 1}/{n_perms} permutations ({elapsed:.0f}s, ETA {eta:.0f}s)")

    null_deltas = np.array(null_deltas)
    # Phipson & Smyth 2010: (B+1)/(m+1) prevents anti-conservative p=0.0000
    p_value = float((null_deltas >= verified_delta).sum() + 1) / (n_perms + 1)
    elapsed_total = time.time() - start

    log.info(f"\n  RESULT: {session}{ap_str} RR{rr_target}")
    log.info(f"  Real delta:   {verified_delta:+.1f}R")
    log.info(f"  Null mean:    {null_deltas.mean():+.1f}R")
    log.info(f"  Null median:  {np.median(null_deltas):+.1f}R")
    log.info(f"  Null std:     {null_deltas.std():.1f}R")
    log.info(f"  Null max:     {null_deltas.max():+.1f}R")
    log.info(f"  p-value:      {p_value:.4f}")
    log.info(f"  Elapsed:      {elapsed_total:.0f}s")

    if p_value < 0.05:
        log.info(f"  >>> PASS — ML has genuine skill (p={p_value:.4f} < 0.05)")
    elif p_value < 0.10:
        log.info(f"  >>> MARGINAL — suggestive but not significant (p={p_value:.4f})")
    else:
        log.info(f"  >>> FAIL — no better than chance (p={p_value:.4f} >= 0.10)")

    return {
        "session": session,
        "aperture": aperture,
        "rr": rr_target,
        "verified_delta": round(verified_delta, 2),
        "null_mean": round(float(null_deltas.mean()), 2),
        "null_std": round(float(null_deltas.std()), 2),
        "null_max": round(float(null_deltas.max()), 2),
        "p_value": round(p_value, 4),
        "n_perms": n_perms,
        "elapsed_s": round(elapsed_total, 1),
    }


def main():
    log.info("ML Bootstrap Permutation Test (V2)")

    # Load selected configs from V2 retrain
    try:
        survivors = load_selected_configs()
    except FileNotFoundError as e:
        log.error(str(e))
        return

    if not survivors:
        log.info("0 configs to bootstrap — ML DEAD (pre-registration: 0 = NO-GO)")
        return

    log.info(f"Testing {len(survivors)} V2 selected configs")
    log.info(f"Permutations per test: {N_PERMUTATIONS}")

    all_results = []
    for session, aperture, rr, delta in survivors:
        try:
            result = bootstrap_one(session, aperture, rr, delta)
        except Exception as exc:
            log.error(f"  bootstrap_one({session}, {aperture}, {rr}) CRASHED: {exc}", exc_info=True)
            result = {"session": session, "aperture": aperture, "rr": rr, "error": str(exc)}
        all_results.append(result)

    # Final summary
    log.info(f"\n{'=' * 70}")
    log.info("  BOOTSTRAP SUMMARY")
    log.info(f"{'=' * 70}")
    log.info(f"  {'Session':<20} {'Ap':>3} {'RR':>4} {'Delta':>7} {'NullMu':>7} {'p':>7} {'Verdict':>10}")
    log.info(f"  {'-' * 20} {'---':>3} {'----':>4} {'-------':>7} {'-------':>7} {'-------':>7} {'----------':>10}")

    for r in all_results:
        if "error" in r:
            log.info(f"  {r['session']:<20} {'--':>3} {r['rr']:>4.1f} {'ERROR':>7} {'':>7} {'':>7} {r['error']:>10}")
            continue
        ap = f"O{r['aperture']}" if r["aperture"] else "--"
        p = r["p_value"]
        verdict = "PASS" if p < 0.05 else ("MARGINAL" if p < 0.10 else "FAIL")
        log.info(
            f"  {r['session']:<20} {ap:>3} {r['rr']:>4.1f} "
            f"{r['verified_delta']:>+7.1f} {r['null_mean']:>+7.1f} "
            f"{r['p_value']:>7.4f} {verdict:>10}"
        )

    n_pass = sum(1 for r in all_results if r.get("p_value", 1) < 0.05)
    n_total = len([r for r in all_results if "error" not in r])
    log.info(f"\n  {n_pass}/{n_total} passed bootstrap (p < 0.05, unadjusted)")

    # --- BH FDR correction (V2 methodology) ---
    # Session is the discovery unit. K=12 for promotion, K=108 as footnote.
    # Reuses the canonical BH implementation from strategy_validator.py.
    from trading_app.strategy_validator import benjamini_hochberg

    valid_results = [r for r in all_results if "error" not in r and "p_value" in r]
    session_pvalues = [(r["session"], r["p_value"]) for r in valid_results]

    if session_pvalues:
        log.info(f"\n{'=' * 70}")
        log.info("  BH FDR CORRECTION")
        log.info(f"{'=' * 70}")

        # K=12: session-level family (promotion decision)
        fdr_k12 = benjamini_hochberg(session_pvalues, alpha=0.05, total_tests=12)
        # K=108: global grid (conservative footnote per RESEARCH_RULES.md)
        fdr_k108 = benjamini_hochberg(session_pvalues, alpha=0.05, total_tests=108)

        log.info(f"  {'Session':<22} {'raw_p':>7} {'BH_K12':>8} {'K12':>5} {'BH_K108':>8} {'K108':>5}")
        log.info(f"  {'-' * 22} {'-' * 7} {'-' * 8} {'-' * 5} {'-' * 8} {'-' * 5}")

        for sid, _ in sorted(session_pvalues, key=lambda x: x[1]):
            k12 = fdr_k12.get(sid, {})
            k108 = fdr_k108.get(sid, {})
            sig12 = "SIG" if k12.get("fdr_significant") else "n.s."
            sig108 = "SIG" if k108.get("fdr_significant") else "n.s."
            log.info(
                f"  {sid:<22} {k12.get('raw_p', 0):>7.4f} "
                f"{k12.get('adjusted_p', 1):>8.4f} {sig12:>5} "
                f"{k108.get('adjusted_p', 1):>8.4f} {sig108:>5}"
            )

        n_survivors_k12 = sum(1 for v in fdr_k12.values() if v["fdr_significant"])
        n_survivors_k108 = sum(1 for v in fdr_k108.values() if v["fdr_significant"])
        log.info("\n  BH FDR SURVIVORS:")
        log.info(f"    K=12  (promotion):   {n_survivors_k12}/12")
        log.info(f"    K=108 (conservative): {n_survivors_k108}/12")

        if n_survivors_k12 >= 2:
            log.info("  VERDICT: ML ALIVE — proceed to Phase 2")
        elif n_survivors_k12 == 1:
            log.info("  VERDICT: ML CONDITIONAL — 1 survivor, investigate only, no Phase 2")
        else:
            log.info("  VERDICT: ML DEAD — 0 BH survivors at K=12. Add to NO-GO registry.")

    log.info(f"\n  Full results: {LOG_DIR / 'ml_bootstrap_results.log'}")


if __name__ == "__main__":
    main()
