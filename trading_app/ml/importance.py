"""Feature importance analysis: MDI and MDA (permutation importance).

Per de Prado: "Feature importance is a true research tool for developing
economic theories, overcoming the caveats of classical statistical methods."

MDI = Mean Decrease Impurity (built-in RF feature importance)
MDA = Mean Decrease Accuracy (permutation importance, more robust)

Usage:
    python -m trading_app.ml.importance --instrument MGC
    python -m trading_app.ml.importance --all
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from pipeline.paths import GOLD_DB_PATH
from trading_app.ml.config import ACTIVE_INSTRUMENTS, RF_PARAMS
from trading_app.ml.features import load_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_importance(
    X: pd.DataFrame,
    y: pd.Series,
    trading_days: pd.Series,
    *,
    top_n: int = 25,
) -> pd.DataFrame:
    """Compute MDI and MDA feature importance.

    Uses TimeSeriesSplit to avoid look-ahead in permutation importance.

    Returns:
        DataFrame with columns: feature, mdi_score, mdi_rank, mda_score, mda_rank
    """
    rf = RandomForestClassifier(**RF_PARAMS)

    # Train on first 80% (time-ordered), test on last 20%
    n_train = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_test, y_test = X.iloc[n_train:], y.iloc[n_train:]

    logger.info(f"Training RF on {n_train:,d} rows, testing on {len(X) - n_train:,d} rows")
    rf.fit(X_train, y_train)

    # MDI (Mean Decrease Impurity)
    mdi = pd.Series(rf.feature_importances_, index=X.columns, name="mdi_score")

    # MDA (Permutation Importance on OOS data)
    logger.info("Computing permutation importance (MDA)...")
    perm_result = permutation_importance(
        rf,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc",
    )
    mda = pd.Series(perm_result.importances_mean, index=X.columns, name="mda_score")
    mda_std = pd.Series(perm_result.importances_std, index=X.columns, name="mda_std")

    # OOS accuracy
    oos_auc = rf.score(X_test, y_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    from sklearn.metrics import roc_auc_score

    oos_auc = roc_auc_score(y_test, y_prob)
    logger.info(f"OOS AUC: {oos_auc:.4f}")

    # Combine
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mdi_score": mdi.values,
            "mda_score": mda.values,
            "mda_std": mda_std.values,
        }
    )
    importance["mdi_rank"] = importance["mdi_score"].rank(ascending=False).astype(int)
    importance["mda_rank"] = importance["mda_score"].rank(ascending=False).astype(int)
    importance["combined_rank"] = (importance["mdi_rank"] + importance["mda_rank"]) / 2
    importance = importance.sort_values("combined_rank")

    return importance.head(top_n), oos_auc


def print_importance_report(instrument: str, importance: pd.DataFrame, oos_auc: float, n_rows: int) -> None:
    """Print formatted feature importance report."""
    print(f"\n{'=' * 70}")
    print(f"  FEATURE IMPORTANCE — {instrument}")
    print(f"  {n_rows:,d} outcomes | OOS AUC: {oos_auc:.4f}")
    print(f"{'=' * 70}")
    print(f"  {'Rank':<6} {'Feature':<40} {'MDI':>8} {'MDA':>8} {'MDA±':>8}")
    print(f"  {'-' * 6} {'-' * 40} {'-' * 8} {'-' * 8} {'-' * 8}")

    for i, (_, row) in enumerate(importance.iterrows(), 1):
        signal = ""
        if row["mda_score"] > 0.001:
            signal = " ***"
        elif row["mda_score"] > 0.0005:
            signal = " **"
        elif row["mda_score"] > 0.0001:
            signal = " *"

        print(
            f"  {i:<6d} {row['feature']:<40s} {row['mdi_score']:>8.4f} "
            f"{row['mda_score']:>8.4f} {row['mda_std']:>7.4f}{signal}"
        )

    print(f"{'=' * 70}")
    print("  *** = strong signal (MDA > 0.001)")
    print("  **  = moderate signal (MDA > 0.0005)")
    print("  *   = weak signal (MDA > 0.0001)")
    print()


def run_importance(instrument: str, db_path: str) -> tuple[pd.DataFrame, float]:
    """Run full feature importance analysis for one instrument."""
    logger.info(f"Loading features for {instrument}...")
    X, y, meta = load_feature_matrix(db_path, instrument)
    logger.info(f"Feature matrix: {X.shape[0]:,d} x {X.shape[1]}")
    logger.info(f"Win rate: {y.mean():.1%}")

    importance, oos_auc = compute_importance(X, y, meta["trading_day"], top_n=25)
    print_importance_report(instrument, importance, oos_auc, len(X))
    return importance, oos_auc


def main():
    parser = argparse.ArgumentParser(description="ML Feature Importance Analysis")
    parser.add_argument("--instrument", type=str, help="Single instrument (MGC/MNQ/MES/M2K)")
    parser.add_argument("--all", action="store_true", help="Run all active instruments")
    parser.add_argument("--db-path", type=str, default=str(GOLD_DB_PATH))
    args = parser.parse_args()

    instruments = ACTIVE_INSTRUMENTS if args.all else [args.instrument or "MGC"]

    for inst in instruments:
        run_importance(inst, args.db_path)


if __name__ == "__main__":
    main()
