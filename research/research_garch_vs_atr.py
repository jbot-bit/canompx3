#!/usr/bin/env python3
"""
GARCH vs ATR Velocity — Head-to-Head Comparison

Compare skip-filter precision and recall:
- ATR velocity filter: skip when atr_vel_regime='Contracting' AND compression_tier != 'Expanded'
- GARCH filter: skip when garch_atr_ratio < threshold (grid search over thresholds)

For each instrument × session, report:
- Precision: % of skipped days that were actually losers
- Recall: % of all losers that the filter caught
- Net R-improvement: avg R with filter - avg R without
- BH FDR on multi-threshold grid

Output: research/output/garch_vs_atr_results.csv

Usage:
    python research/research_garch_vs_atr.py
    python research/research_garch_vs_atr.py --instrument MGC MES
"""
import argparse
import logging
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

from pipeline.paths import GOLD_DB_PATH


def load_trades(con, instrument: str) -> pd.DataFrame:
    """Load orb_outcomes joined with daily_features GARCH/ATR columns."""
    df = con.execute("""
        SELECT
            o.trading_day, o.symbol, o.orb_label AS session_label,
            o.rr_target, o.pnl_r,
            d.atr_vel_regime, d.atr_vel_ratio,
            d.garch_forecast_vol, d.garch_atr_ratio, d.atr_20,
            d.orb_0900_compression_tier,
            d.orb_1000_compression_tier,
            d.orb_1800_compression_tier
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
            AND o.symbol = d.symbol
            AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = ?
            AND d.orb_minutes = 5
            AND o.pnl_r IS NOT NULL
    """, [instrument]).df()
    return df


def atr_vel_would_skip(row) -> bool:
    """Replicate ATRVelocityFilter logic."""
    session = row["session_label"]
    if session not in ("0900", "1000"):
        return False
    if row["atr_vel_regime"] != "Contracting":
        return False
    tier_col = f"orb_{session}_compression_tier"
    tier = row.get(tier_col)
    if tier is None:
        return False
    return tier != "Expanded"


def evaluate_filter(df: pd.DataFrame, skip_mask: pd.Series, label: str) -> dict:
    """Compute precision, recall, net R for a boolean skip mask."""
    total_n = len(df)
    skipped = int(skip_mask.sum())
    kept = total_n - skipped

    if skipped == 0 or kept == 0:
        return {"label": label, "total": total_n, "skipped": 0, "kept": kept,
                "precision": None, "recall": None, "net_r": None,
                "avg_r_all": None, "avg_r_kept": None}

    losers = df["pnl_r"] < 0
    skipped_losers = int((skip_mask & losers).sum())
    total_losers = int(losers.sum())

    precision = skipped_losers / skipped if skipped > 0 else None
    recall = skipped_losers / total_losers if total_losers > 0 else None

    avg_r_all = float(df["pnl_r"].mean())
    avg_r_kept = float(df.loc[~skip_mask, "pnl_r"].mean())
    net_r = avg_r_kept - avg_r_all

    # T-test: is kept subset significantly better than full set?
    kept_vals = df.loc[~skip_mask, "pnl_r"]
    if len(kept_vals) > 1:
        t_stat, p_val = scipy_stats.ttest_1samp(kept_vals, avg_r_all)
    else:
        t_stat, p_val = None, None

    return {
        "label": label,
        "total": total_n,
        "skipped": skipped,
        "kept": kept,
        "precision": round(precision, 4) if precision else None,
        "recall": round(recall, 4) if recall else None,
        "avg_r_all": round(avg_r_all, 4),
        "avg_r_kept": round(avg_r_kept, 4),
        "net_r": round(net_r, 4),
        "t_stat": round(t_stat, 3) if t_stat else None,
        "p_val": round(p_val, 4) if p_val else None,
    }


def run_comparison(instrument: str, db_path: str):
    """Run full GARCH vs ATR comparison for one instrument."""
    con = duckdb.connect(db_path, read_only=True)
    df = load_trades(con, instrument)
    con.close()

    n_garch = int(df["garch_forecast_vol"].notna().sum())
    logger.info(f"{instrument}: {len(df):,} trades loaded, {n_garch:,} have GARCH forecast")

    df_garch = df[df["garch_forecast_vol"].notna()].copy()
    if len(df_garch) < 100:
        logger.warning(f"{instrument}: Only {len(df_garch)} trades with GARCH — skipping")
        return []

    results = []
    sessions = sorted(df_garch["session_label"].unique())

    for session in sessions:
        sdf = df_garch[df_garch["session_label"] == session].copy()
        if len(sdf) < 50:
            continue

        # ATR velocity filter baseline
        atr_skip = sdf.apply(atr_vel_would_skip, axis=1)
        atr_result = evaluate_filter(sdf, atr_skip, f"ATR_VEL")
        results.append({"instrument": instrument, "session": session, **atr_result})

        # GARCH threshold grid (ratio < threshold → skip)
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            garch_skip = sdf["garch_atr_ratio"] < threshold
            garch_result = evaluate_filter(sdf, garch_skip, f"GARCH_LT_{threshold}")
            results.append({"instrument": instrument, "session": session, **garch_result})

    return results


def apply_bh_fdr(results_df: pd.DataFrame, q: float = 0.10):
    """Apply Benjamini-Hochberg FDR correction to p-values."""
    valid = results_df["p_val"].notna()
    if valid.sum() == 0:
        results_df["p_bh"] = None
        results_df["bh_sig"] = False
        return results_df

    pvals = results_df.loc[valid, "p_val"].values
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(1, n + 1)
    p_bh = pvals * n / ranks
    p_bh = np.minimum.accumulate(p_bh[np.argsort(-ranks)])[np.argsort(ranks)]
    p_bh = np.clip(p_bh, 0, 1)

    results_df.loc[valid, "p_bh"] = np.round(p_bh, 4)
    results_df["bh_sig"] = results_df["p_bh"].notna() & (results_df["p_bh"] < q)
    return results_df


def main():
    parser = argparse.ArgumentParser(description="GARCH vs ATR head-to-head")
    parser.add_argument("--instrument", nargs="+", default=["MGC", "MNQ", "MES", "M2K"])
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH))
    args = parser.parse_args()

    all_results = []
    for inst in args.instrument:
        all_results.extend(run_comparison(inst, args.db_path))

    if not all_results:
        logger.warning("No results to report")
        return

    results_df = pd.DataFrame(all_results)
    results_df = apply_bh_fdr(results_df)

    # Print summary
    print("\n" + "=" * 100)
    print("GARCH vs ATR Velocity — Head-to-Head Results")
    print("=" * 100)
    display_cols = ["instrument", "session", "label", "total", "skipped", "kept",
                    "precision", "recall", "avg_r_all", "avg_r_kept", "net_r",
                    "p_val", "p_bh", "bh_sig"]
    for inst in args.instrument:
        inst_df = results_df[results_df["instrument"] == inst]
        if inst_df.empty:
            continue
        print(f"\n--- {inst} ---")
        print(inst_df[display_cols].to_string(index=False))

    # Summary: best GARCH threshold per instrument/session
    print("\n" + "=" * 100)
    print("BEST GARCH THRESHOLD PER SESSION (by net_r)")
    print("=" * 100)
    garch_only = results_df[results_df["label"].str.startswith("GARCH")]
    for (inst, sess), group in garch_only.groupby(["instrument", "session"]):
        best = group.loc[group["net_r"].idxmax()] if group["net_r"].notna().any() else None
        if best is not None and best["net_r"] > 0:
            atr_row = results_df[(results_df["instrument"] == inst) &
                                 (results_df["session"] == sess) &
                                 (results_df["label"] == "ATR_VEL")]
            atr_net = atr_row["net_r"].values[0] if len(atr_row) > 0 else None
            winner = "GARCH" if (atr_net is None or best["net_r"] > atr_net) else "ATR"
            print(f"  {inst} {sess}: {best['label']} net_r={best['net_r']:+.4f} "
                  f"(ATR: {atr_net if atr_net else 'N/A'}) -> {winner} wins")

    # Save
    out_dir = Path("research/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "garch_vs_atr_results.csv", index=False)
    logger.info(f"\nSaved to {out_dir / 'garch_vs_atr_results.csv'}")


if __name__ == "__main__":
    main()
