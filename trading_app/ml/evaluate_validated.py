"""Evaluate meta-label impact on VALIDATED strategies only.

This is the real test: does the meta-label improve strategies that are
already BH FDR validated + walk-forward tested?

Loads validated_setups from gold.db, filters outcomes to only include
validated strategy parameters, then applies meta-label predictions.

Usage:
    python -m trading_app.ml.evaluate_validated --instrument MGC
    python -m trading_app.ml.evaluate_validated --all
"""

from __future__ import annotations

import argparse
import logging

import duckdb
import joblib
import pandas as pd

from pipeline.db_config import configure_connection
from pipeline.paths import GOLD_DB_PATH
from trading_app.ml.config import ACTIVE_INSTRUMENTS, MODEL_DIR
from trading_app.ml.features import load_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_validated_params(db_path: str, instrument: str) -> pd.DataFrame:
    """Get parameter combinations from validated_setups for this instrument."""
    con = duckdb.connect(db_path, read_only=True)
    configure_connection(con)
    try:
        df = con.execute("""
            SELECT DISTINCT
                orb_label, entry_model, rr_target, confirm_bars,
                filter_type, orb_minutes
            FROM validated_setups
            WHERE instrument = $1
              AND status = 'active'
        """, [instrument]).fetchdf()
    finally:
        con.close()

    logger.info(f"Found {len(df)} validated parameter combos for {instrument}")
    return df


def _sharpe(pnl: pd.Series) -> float:
    """Per-trade Sharpe ratio (no annualization — data is per-trade, not per-day)."""
    if pnl.std() == 0 or len(pnl) < 2:
        return 0.0
    return float(pnl.mean() / pnl.std())


def evaluate_validated(instrument: str, db_path: str) -> None:
    """Evaluate meta-label on validated strategies only."""
    model_path = MODEL_DIR / f"meta_label_{instrument}.joblib"
    if not model_path.exists():
        logger.error(f"No trained model for {instrument}")
        return

    bundle = joblib.load(model_path)
    rf = bundle["model"]
    threshold = bundle["optimal_threshold"]
    feature_names = bundle["feature_names"]

    # Load full feature matrix
    X, y, meta = load_feature_matrix(db_path, instrument)

    # Align features (0.0 for one-hot, -999.0 for numeric)
    from trading_app.ml.evaluate import _fill_missing_features
    X = _fill_missing_features(X, feature_names)

    # Predict
    y_prob = rf.predict_proba(X)[:, 1]
    meta = meta.copy()
    meta["p_win"] = y_prob

    # Get validated strategy params
    validated = _get_validated_params(db_path, instrument)

    if validated.empty:
        logger.warning(f"No validated strategies for {instrument}")
        return

    # Filter to only outcomes matching validated strategy parameters
    # rr_target in meta, rr_target in validated
    meta["rr_target_rounded"] = meta["rr_target"].round(1)
    validated["rr_target_rounded"] = validated["rr_target"].round(1)

    # Create a set of validated (orb_label, entry_model, rr_target, confirm_bars, orb_minutes)
    validated_keys = set()
    for _, row in validated.iterrows():
        key = (row["orb_label"], row["entry_model"], round(row["rr_target"], 1),
               int(row["confirm_bars"]), int(row["orb_minutes"]))
        validated_keys.add(key)

    # Filter meta to validated combos
    meta["combo_key"] = list(zip(
        meta["orb_label"],
        meta["entry_model"],
        meta["rr_target_rounded"],
        meta["confirm_bars"].astype(int),
        meta["orb_minutes"].astype(int),
    ))
    validated_mask = meta["combo_key"].isin(validated_keys)
    meta_val = meta[validated_mask].copy()

    logger.info(f"Validated-strategy outcomes: {len(meta_val):,d} "
                f"(of {len(meta):,d} total, {len(meta_val)/len(meta):.1%})")

    if len(meta_val) < 100:
        logger.warning(f"Too few validated outcomes ({len(meta_val)})")
        return

    # Apply threshold
    pnl_all = meta_val["pnl_r"]
    take_mask = meta_val["p_win"] >= threshold
    pnl_kept = meta_val.loc[take_mask, "pnl_r"]
    pnl_skipped = meta_val.loc[~take_mask, "pnl_r"]

    # Results
    print(f"\n{'=' * 75}")
    print(f"  VALIDATED-STRATEGY META-LABEL EVALUATION — {instrument}")
    print(f"  {len(validated)} validated combos | threshold {threshold:.2f}")
    print(f"{'=' * 75}")

    print(f"\n  {'METRIC':<18} {'BASELINE':>12} {'FILTERED':>12} {'SKIPPED':>12} {'DELTA':>12}")
    print(f"  {'-' * 18} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")
    print(f"  {'Trades':<18} {len(meta_val):>12,d} {take_mask.sum():>12,d} "
          f"{(~take_mask).sum():>12,d}")
    print(f"  {'Avg R':<18} {pnl_all.mean():>+12.4f} {pnl_kept.mean():>+12.4f} "
          f"{pnl_skipped.mean():>+12.4f} {pnl_kept.mean() - pnl_all.mean():>+12.4f}")
    print(f"  {'Total R':<18} {pnl_all.sum():>12.1f} {pnl_kept.sum():>12.1f} "
          f"{pnl_skipped.sum():>12.1f} {pnl_kept.sum() - pnl_all.sum():>+12.1f}")
    print(f"  {'Sharpe':<18} {_sharpe(pnl_all):>12.3f} {_sharpe(pnl_kept):>12.3f} "
          f"{_sharpe(pnl_skipped):>12.3f} {_sharpe(pnl_kept) - _sharpe(pnl_all):>+12.3f}")
    print(f"  {'Win Rate':<18} {(pnl_all > 0).mean():>11.1%} {(pnl_kept > 0).mean():>11.1%} "
          f"{(pnl_skipped > 0).mean():>11.1%} {(pnl_kept > 0).mean() - (pnl_all > 0).mean():>+11.1%}")
    print(f"  {'Skip %':<18} {'':>12} {1 - take_mask.mean():>11.1%}")

    # Per-session for validated strategies
    print("\n  PER-SESSION (validated strategies only)")
    print(f"  {'SESSION':<20} {'N':>6} {'KEPT':>6} {'SKIP%':>7} "
          f"{'BASE':>8} {'FILT':>8} {'SKIP':>8} {'LIFT':>8}")
    print(f"  {'-' * 20} {'-' * 6} {'-' * 6} {'-' * 7} "
          f"{'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    for session in sorted(meta_val["orb_label"].unique()):
        sm = meta_val[meta_val["orb_label"] == session]
        if len(sm) < 30:
            continue
        kept = sm[sm["p_win"] >= threshold]
        skipped = sm[sm["p_win"] < threshold]
        if len(kept) < 10:
            continue

        lift = kept["pnl_r"].mean() - sm["pnl_r"].mean()
        print(f"  {session:<20} {len(sm):>6d} {len(kept):>6d} "
              f"{1 - len(kept)/len(sm):>6.1%} "
              f"{sm['pnl_r'].mean():>+8.4f} {kept['pnl_r'].mean():>+8.4f} "
              f"{skipped['pnl_r'].mean():>+8.4f} {lift:>+8.4f}")

    # Calibration within validated strategies
    meta_val_sorted = meta_val.sort_values("p_win")
    quintile_size = len(meta_val_sorted) // 5
    print("\n  CALIBRATION (validated strategies, should increase)")
    print(f"  {'Q':<4} {'P(win) range':<18} {'WR':>8} {'avgR':>10} {'N':>8}")
    print(f"  {'-' * 4} {'-' * 18} {'-' * 8} {'-' * 10} {'-' * 8}")
    for q in range(5):
        s = q * quintile_size
        e = s + quintile_size if q < 4 else len(meta_val_sorted)
        chunk = meta_val_sorted.iloc[s:e]
        print(f"  Q{q+1:<3d} {chunk['p_win'].min():.3f}-{chunk['p_win'].max():.3f}"
              f"     {(chunk['pnl_r'] > 0).mean():>7.1%} "
              f"{chunk['pnl_r'].mean():>+10.4f} {len(chunk):>8,d}")

    print(f"\n{'=' * 75}\n")


def main():
    parser = argparse.ArgumentParser(description="Validated-Strategy Meta-Label Evaluation")
    parser.add_argument("--instrument", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--db-path", type=str, default=str(GOLD_DB_PATH))
    args = parser.parse_args()

    instruments = ACTIVE_INSTRUMENTS if args.all else [args.instrument or "MGC"]
    for inst in instruments:
        evaluate_validated(inst, args.db_path)


if __name__ == "__main__":
    main()
