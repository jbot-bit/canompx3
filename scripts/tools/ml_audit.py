"""ML discrimination audit: bulk analysis of model predictions vs outcomes.

Uses load_single_config_feature_matrix for fast bulk analysis.
Run: python scripts/tools/ml_audit.py [--instrument MNQ] [--all]
"""

import argparse

import joblib
import numpy as np

from trading_app.ml.config import ACTIVE_INSTRUMENTS, MODEL_DIR
from trading_app.ml.features import load_single_config_feature_matrix
from trading_app.ml.predict_live import LiveMLPredictor


def audit_instrument(instrument: str, db_path: str) -> dict:
    """Run full ML audit for one instrument."""
    # Hybrid → legacy fallback (matching predict_live.py pattern)
    hybrid_path = MODEL_DIR / f"meta_label_{instrument}_hybrid.joblib"
    legacy_path = MODEL_DIR / f"meta_label_{instrument}.joblib"
    path = hybrid_path if hybrid_path.exists() else legacy_path
    if not path.exists():
        print(f"\n{instrument}: NO MODEL FOUND")
        return {}

    bundle = joblib.load(path)

    # For hybrid models, use the first session with a model for audit
    if "sessions" in bundle:
        model = None
        threshold = 0.5
        feature_names = []
        is_per_aperture = bundle.get("bundle_format") == "per_aperture"
        for _session, info in bundle.get("sessions", {}).items():
            if is_per_aperture:
                for _ak, sub_info in info.items():
                    if isinstance(sub_info, dict) and sub_info.get("model") is not None:
                        model = sub_info["model"]
                        threshold = float(sub_info["optimal_threshold"])
                        feature_names = sub_info["feature_names"]
                        break
                if model is not None:
                    break
            else:
                if isinstance(info, dict) and info.get("model") is not None:
                    model = info["model"]
                    threshold = float(info["optimal_threshold"])
                    feature_names = info["feature_names"]
                    break
        if model is None:
            print(f"\n{instrument}: HYBRID MODEL HAS NO TRAINED SESSIONS")
            return {}
        importances = model.feature_importances_
    else:
        model = bundle["model"]
        threshold = float(bundle["optimal_threshold"])
        feature_names = bundle["feature_names"]
        importances = model.feature_importances_

    print(f"\n{'=' * 70}")
    print(f"  ML AUDIT — {instrument}")
    print(f"{'=' * 70}")
    print(
        f"Model: threshold={threshold}, OOS AUC={bundle.get('oos_auc', 0):.4f}, "
        f"CPCV AUC={bundle.get('cpcv_auc', 0):.4f}"
    )
    print(f"Features: {len(feature_names)}, n_train={bundle.get('n_train', '?'):,}")
    print(f"Data range: {bundle.get('data_date_range', '?')}")

    # --- Feature Importance ---
    idx = np.argsort(importances)[::-1]
    rr_idx = feature_names.index("rr_target") if "rr_target" in feature_names else -1
    rr_imp = importances[rr_idx] if rr_idx >= 0 else 0

    print("\n--- FEATURE IMPORTANCE ---")
    print(f"rr_target dominance: {rr_imp:.1%} of total importance")
    sp = ["rr_target", "confirm_bars", "orb_minutes"]
    sp_imp = sum(importances[feature_names.index(f)] for f in sp if f in feature_names)
    print(f"Strategy params total: {sp_imp:.1%} | Market features: {1 - sp_imp:.1%}")
    print("\nTop 10 features:")
    for rank, i in enumerate(idx[:10], 1):
        print(f"  {rank:>2}. {feature_names[i]:<35} {importances[i]:.4f}")

    # --- Load 2025 OOS data ---
    # Load ALL outcomes (training used <= 2025-02-03, so 2025 data is partially IS)
    # Use the last 20% holdout logic from training to get true OOS
    X, y, meta = load_single_config_feature_matrix(db_path, instrument, bypass_validated=True)

    # Split same as training: last 20% is OOS
    n_train = int(len(X) * 0.8)
    X_oos = X.iloc[n_train:]
    y_oos = y.iloc[n_train:]
    meta_oos = meta.iloc[n_train:].copy()
    pnl_oos = meta_oos["pnl_r"].values

    print(f"\nOOS holdout: {len(X_oos):,} trades ({meta_oos['trading_day'].min()} to {meta_oos['trading_day'].max()})")

    # --- Align features and predict ---
    X_aligned = LiveMLPredictor._align_features(X_oos, feature_names)
    y_prob = model.predict_proba(X_aligned)[:, 1]

    meta_oos = meta_oos.copy()
    meta_oos["p_win"] = y_prob
    meta_oos["y_true"] = y_oos.values

    # === CORE ANALYSIS: Taken vs Skipped ===
    take_mask = y_prob >= threshold
    skip_mask = ~take_mask

    taken_pnl = pnl_oos[take_mask]
    skipped_pnl = pnl_oos[skip_mask]

    print(f"\n--- TAKE vs SKIP at threshold={threshold:.2f} ---")
    print(f"{'Metric':<25} {'ALL':>12} {'TAKEN':>12} {'SKIPPED':>12}")
    print(f"{'-' * 25} {'-' * 12} {'-' * 12} {'-' * 12}")
    print(f"{'Count':<25} {len(pnl_oos):>12,} {len(taken_pnl):>12,} {len(skipped_pnl):>12,}")
    print(
        f"{'Win Rate':<25} {(pnl_oos > 0).mean():>11.1%} "
        f"{(taken_pnl > 0).mean():>11.1%} {(skipped_pnl > 0).mean():>11.1%}"
    )
    print(f"{'Avg R':<25} {pnl_oos.mean():>+12.4f} {taken_pnl.mean():>+12.4f} {skipped_pnl.mean():>+12.4f}")
    print(f"{'Total R':<25} {pnl_oos.sum():>+12.2f} {taken_pnl.sum():>+12.2f} {skipped_pnl.sum():>+12.2f}")
    sh_all = pnl_oos.mean() / pnl_oos.std() if pnl_oos.std() > 0 else 0
    sh_take = taken_pnl.mean() / taken_pnl.std() if taken_pnl.std() > 0 else 0
    sh_skip = skipped_pnl.mean() / skipped_pnl.std() if skipped_pnl.std() > 0 else 0
    print(f"{'Sharpe':<25} {sh_all:>12.3f} {sh_take:>12.3f} {sh_skip:>12.3f}")

    # Skip quality
    print("\n--- SKIP QUALITY ---")
    sw = (skipped_pnl > 0).sum()
    sl = (skipped_pnl <= 0).sum()
    print(f"Skipped winners: {sw} ({sw / len(skipped_pnl) * 100:.1f}%)")
    print(f"Skipped losers: {sl} ({sl / len(skipped_pnl) * 100:.1f}%)")
    print(f"Net R of skips: {skipped_pnl.sum():+.2f}R")
    if skipped_pnl.sum() < 0:
        print("** Model IS correctly skipping net-losing trades **")
    else:
        print("** WARNING: Model is skipping net-WINNING trades! **")

    # === RR TARGET TAUTOLOGY CHECK ===
    print("\n--- RR TARGET TAUTOLOGY CHECK ---")
    print("rr_target is the dominant feature. Is the model just learning")
    print("'low RR = higher win rate' (which is tautological)?")
    rrs = sorted(meta_oos["rr_target"].unique())
    print(f"\n{'RR':>5} {'N':>7} {'P(win)':>8} {'ActWR':>7} {'AvgR':>8} {'TotR':>9} {'TakeN':>7} {'SkipN':>7}")
    for rr in rrs:
        mask = meta_oos["rr_target"].values == rr
        n = mask.sum()
        if n < 10:
            continue
        pw = y_prob[mask].mean()
        wr = (pnl_oos[mask] > 0).mean()
        avg_r = pnl_oos[mask].mean()
        tot_r = pnl_oos[mask].sum()
        t_n = (take_mask & mask).sum()
        s_n = (skip_mask & mask).sum()
        print(f"{rr:>5.1f} {n:>7} {pw:>8.3f} {wr:>6.1%} {avg_r:>+8.4f} {tot_r:>+9.2f} {t_n:>7} {s_n:>7}")

    # === PER-SESSION ANALYSIS ===
    print("\n--- PER-SESSION SKIP ANALYSIS ---")
    sessions = sorted(meta_oos["orb_label"].unique())
    print(f"{'Session':<20} {'TakeN':>6} {'TakeAvgR':>9} {'SkipN':>6} {'SkipAvgR':>9} {'SkipNet':>9} {'Verdict':>8}")
    print(f"{'-' * 20} {'-' * 6} {'-' * 9} {'-' * 6} {'-' * 9} {'-' * 9} {'-' * 8}")
    for s in sessions:
        s_mask = meta_oos["orb_label"].values == s
        t_mask_s = take_mask & s_mask
        sk_mask_s = skip_mask & s_mask
        t_n = t_mask_s.sum()
        sk_n = sk_mask_s.sum()
        if t_n + sk_n < 10:
            continue
        t_avg = pnl_oos[t_mask_s].mean() if t_n > 0 else 0
        sk_avg = pnl_oos[sk_mask_s].mean() if sk_n > 0 else 0
        sk_net = pnl_oos[sk_mask_s].sum() if sk_n > 0 else 0
        verdict = "GOOD" if sk_net <= 0 else "BAD"
        print(f"{s:<20} {t_n:>6} {t_avg:>+9.4f} {sk_n:>6} {sk_avg:>+9.4f} {sk_net:>+9.2f} {verdict:>8}")

    # === P(win) CALIBRATION ===
    print("\n--- P(win) CALIBRATION ---")
    print("Does P(win) = 0.60 actually mean 60% win rate?")
    bins = np.arange(0.30, 0.70, 0.05)
    print(f"{'P(win) bin':<12} {'N':>6} {'ActualWR':>9} {'AvgR':>9} {'TotalR':>9}")
    for i in range(len(bins) - 1):
        bmask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        n = bmask.sum()
        if n < 5:
            continue
        wr = (pnl_oos[bmask] > 0).mean()
        avg_r = pnl_oos[bmask].mean()
        total_r = pnl_oos[bmask].sum()
        print(f"[{bins[i]:.2f}-{bins[i + 1]:.2f}) {n:>6} {wr:>8.1%} {avg_r:>+9.4f} {total_r:>+9.2f}")

    # === P(win) DISTRIBUTION ===
    print("\n--- P(win) DISTRIBUTION ---")
    print(f"Mean={y_prob.mean():.3f}  Std={y_prob.std():.3f}  Min={y_prob.min():.3f}  Max={y_prob.max():.3f}")
    hist_bins = np.arange(0.30, 0.70, 0.05)
    for i in range(len(hist_bins) - 1):
        bmask = (y_prob >= hist_bins[i]) & (y_prob < hist_bins[i + 1])
        n = bmask.sum()
        bar = "#" * (n // 50)
        marker = " <-- THRESHOLD" if hist_bins[i] <= threshold < hist_bins[i + 1] else ""
        print(f"  [{hist_bins[i]:.2f}-{hist_bins[i + 1]:.2f}) {n:>6} {bar}{marker}")

    # === THRESHOLD SWEEP ===
    print("\n--- THRESHOLD SWEEP ---")
    print(
        f"{'Thresh':>7} {'TakeN':>7} {'SkipN':>7} {'TakeAvgR':>9} "
        f"{'TakeTotR':>9} {'TakeWR':>7} {'SkipAvgR':>9} {'SkipTotR':>9}"
    )
    best_tot_r = -999
    best_t = 0.5
    for t in np.arange(0.50, 0.66, 0.01):
        t_m = y_prob >= t
        s_m = ~t_m
        t_n = t_m.sum()
        s_n = s_m.sum()
        if t_n < 10:
            continue
        t_avg = pnl_oos[t_m].mean()
        t_tot = pnl_oos[t_m].sum()
        t_wr = (pnl_oos[t_m] > 0).mean()
        s_avg = pnl_oos[s_m].mean() if s_n > 0 else 0
        s_tot = pnl_oos[s_m].sum() if s_n > 0 else 0
        marker = " <-- CURRENT" if abs(t - threshold) < 0.005 else ""
        if t_tot > best_tot_r:
            best_tot_r = t_tot
            best_t = t
        print(
            f"{t:>7.2f} {t_n:>7} {s_n:>7} {t_avg:>+9.4f} {t_tot:>+9.2f} "
            f"{t_wr:>6.1%} {s_avg:>+9.4f} {s_tot:>+9.2f}{marker}"
        )

    print(f"\nBest threshold for max PnL: {best_t:.2f} (total R={best_tot_r:+.2f})")
    print(f"Current threshold: {threshold:.2f} (total R={taken_pnl.sum():+.2f})")
    print(f"Baseline (no filter): total R={pnl_oos.sum():+.2f}")

    # === CONDITIONAL RR ANALYSIS ===
    # The real question: within each RR level, does the model discriminate?
    print("\n--- WITHIN-RR DISCRIMINATION ---")
    print("Within each RR level, does P(win) predict outcomes?")
    print(
        f"{'RR':>5} {'TopQ_N':>7} {'TopQ_WR':>8} {'TopQ_AvgR':>10} "
        f"{'BotQ_N':>7} {'BotQ_WR':>8} {'BotQ_AvgR':>10} {'Delta':>8}"
    )
    for rr in rrs:
        rr_mask = meta_oos["rr_target"].values == rr
        rr_probs = y_prob[rr_mask]
        rr_pnl = pnl_oos[rr_mask]
        if len(rr_probs) < 40:
            continue
        median_p = np.median(rr_probs)
        top_q = rr_probs >= median_p
        bot_q = rr_probs < median_p
        if top_q.sum() < 10 or bot_q.sum() < 10:
            continue
        top_wr = (rr_pnl[top_q] > 0).mean()
        bot_wr = (rr_pnl[bot_q] > 0).mean()
        top_avg = rr_pnl[top_q].mean()
        bot_avg = rr_pnl[bot_q].mean()
        delta = top_avg - bot_avg
        print(
            f"{rr:>5.1f} {top_q.sum():>7} {top_wr:>7.1%} {top_avg:>+10.4f} "
            f"{bot_q.sum():>7} {bot_wr:>7.1%} {bot_avg:>+10.4f} {delta:>+8.4f}"
        )

    # === WITHOUT RR: does market-only signal work? ===
    print("\n--- MARKET-ONLY SIGNAL (RR=1.5 subset) ---")
    print("Controlling for RR=1.5 (most common), does P(win) variation predict?")
    rr15_mask = meta_oos["rr_target"].values == 1.5
    if rr15_mask.sum() > 100:
        rr15_probs = y_prob[rr15_mask]
        rr15_pnl = pnl_oos[rr15_mask]
        rr15_won = rr15_pnl > 0
        # Tercile split
        p33 = np.percentile(rr15_probs, 33)
        p67 = np.percentile(rr15_probs, 67)
        lo = rr15_probs < p33
        mid = (rr15_probs >= p33) & (rr15_probs < p67)
        hi = rr15_probs >= p67
        print(f"  P(win) tercile thresholds: p33={p33:.3f}, p67={p67:.3f}")
        for label, mask in [("Bottom 33%", lo), ("Middle 33%", mid), ("Top 33%", hi)]:
            n = mask.sum()
            if n < 10:
                continue
            wr = rr15_won[mask].mean()
            avg_r = rr15_pnl[mask].mean()
            tot_r = rr15_pnl[mask].sum()
            print(f"  {label:<12} N={n:>5}  WR={wr:.1%}  AvgR={avg_r:+.4f}  TotR={tot_r:+.2f}")

    print(f"\n{'=' * 70}")
    print("Audit complete.")
    return {
        "instrument": instrument,
        "threshold": threshold,
        "oos_n": len(pnl_oos),
        "baseline_total_r": float(pnl_oos.sum()),
        "taken_total_r": float(taken_pnl.sum()),
        "skipped_total_r": float(skipped_pnl.sum()),
        "best_threshold": best_t,
        "best_total_r": best_tot_r,
        "rr_dominance": float(rr_imp),
    }


def main():
    parser = argparse.ArgumentParser(description="ML Discrimination Audit")
    parser.add_argument("--instrument", type=str, default="MNQ")
    parser.add_argument("--all", action="store_true", help="Audit all instruments")
    parser.add_argument("--db-path", type=str, default="gold.db")
    args = parser.parse_args()

    instruments = list(ACTIVE_INSTRUMENTS) if args.all else [args.instrument]

    results = {}
    for inst in instruments:
        r = audit_instrument(inst, args.db_path)
        if r:
            results[inst] = r

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print("  CROSS-INSTRUMENT SUMMARY")
        print(f"{'=' * 70}")
        print(
            f"{'Inst':<6} {'Thresh':>7} {'OOS_N':>7} {'Base_R':>8} {'Take_R':>8} "
            f"{'Skip_R':>8} {'Best_T':>7} {'Best_R':>8} {'RR_Dom':>7}"
        )
        for inst, r in results.items():
            print(
                f"{inst:<6} {r['threshold']:>7.2f} {r['oos_n']:>7,} "
                f"{r['baseline_total_r']:>+8.1f} {r['taken_total_r']:>+8.1f} "
                f"{r['skipped_total_r']:>+8.1f} {r['best_threshold']:>7.2f} "
                f"{r['best_total_r']:>+8.1f} {r['rr_dominance']:>6.1%}"
            )


if __name__ == "__main__":
    main()
