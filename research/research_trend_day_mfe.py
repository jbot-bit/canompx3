"""research/research_trend_day_mfe.py

Phase 1 Research: Unicorn/Trend-Day MFE Discovery

Computes TRUE session MFE from bars_1m (uncapped by target/stop),
quantifies the gap vs stored mfe_r, and tests predictors of outsized moves.

@research-source: docs/plans/2026-03-06-trend-day-mfe-impl.md
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import warnings

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.cost_model import COST_SPECS, CostSpec, pnl_points_to_r
from pipeline.paths import GOLD_DB_PATH
from pipeline.dst import SESSION_CATALOG
from pipeline.build_daily_features import compute_trading_day_utc_range
from research.lib.stats import bh_fdr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTIVE_INSTRUMENTS = sorted(ACTIVE_ORB_INSTRUMENTS)
SESSIONS = sorted(SESSION_CATALOG.keys())
ORB_MINUTES = [5, 15, 30]
RR_TARGETS = [1.5, 2.0, 2.5, 3.0]
DB_PATH = GOLD_DB_PATH
MIN_TRADES = 30


# ---------------------------------------------------------------------------
# TRUE MFE computation
# ---------------------------------------------------------------------------
def compute_true_session_mfe(
    bars_1m: pd.DataFrame,  # pre-filtered to entry_ts < ts_utc <= trading_day_end
    entry_price: float,
    stop_price: float,
    break_dir: int,  # 1=long, -1=short
    cost_spec: CostSpec,
) -> dict:
    """Compute uncapped MFE/MAE from entry to session end using 1m bars."""
    if bars_1m.empty:
        return {
            "true_mfe_r": None,
            "true_mae_r": None,
            "session_close_r": None,
            "time_to_mfe_min": None,
            "bars_after_entry": 0,
        }

    highs = bars_1m["high"].values
    lows = bars_1m["low"].values

    if break_dir == 1:  # long
        favorable_pts = highs - entry_price
        adverse_pts = entry_price - lows
    else:  # short
        favorable_pts = entry_price - lows
        adverse_pts = highs - entry_price

    max_fav = max(float(np.max(favorable_pts)), 0.0)
    max_adv = max(float(np.max(adverse_pts)), 0.0)

    true_mfe_r = pnl_points_to_r(cost_spec, entry_price, stop_price, max_fav)
    true_mae_r = pnl_points_to_r(cost_spec, entry_price, stop_price, max_adv)

    # Session close R (signed)
    last_close = float(bars_1m["close"].iloc[-1])
    if break_dir == 1:
        close_pts = last_close - entry_price
    else:
        close_pts = entry_price - last_close
    session_close_r = pnl_points_to_r(cost_spec, entry_price, stop_price, close_pts)

    mfe_bar_idx = int(np.argmax(favorable_pts))
    time_to_mfe_min = mfe_bar_idx  # bars are 1-minute
    bars_after_entry = len(bars_1m)

    return {
        "true_mfe_r": round(true_mfe_r, 4),
        "true_mae_r": round(true_mae_r, 4),
        "session_close_r": round(session_close_r, 4),
        "time_to_mfe_min": time_to_mfe_min,
        "bars_after_entry": bars_after_entry,
    }


def load_bars_1m_for_day(
    con: duckdb.DuckDBPyConnection, instrument: str, trading_day
) -> pd.DataFrame:
    """Load all 1m bars for a trading day's UTC range."""
    start_utc, end_utc = compute_trading_day_utc_range(trading_day)
    sql = """
        SELECT ts_utc, high, low, close
        FROM bars_1m
        WHERE symbol = ?
          AND ts_utc >= ?
          AND ts_utc < ?
        ORDER BY ts_utc
    """
    return con.execute(sql, [instrument, start_utc, end_utc]).fetchdf()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_outcomes(con: duckdb.DuckDBPyConnection, instrument: str,
                  limit: int | None = None) -> pd.DataFrame:
    """Load E1/E2 outcomes for a single instrument.

    Returns DataFrame with entry details needed for TRUE MFE computation.
    """
    sql = """
        SELECT o.trading_day, o.symbol, o.orb_label, o.orb_minutes,
               o.entry_model, o.rr_target, o.confirm_bars,
               o.entry_ts, o.entry_price, o.stop_price, o.target_price,
               o.pnl_r, o.mfe_r AS capped_mfe_r, o.outcome,
               o.exit_ts,
               CASE WHEN o.entry_price > o.stop_price THEN 1 ELSE -1 END AS break_dir
        FROM orb_outcomes o
        WHERE o.symbol = ?
          AND o.entry_model IN ('E1', 'E2')
          AND o.outcome IS NOT NULL
          AND o.entry_ts IS NOT NULL
        ORDER BY o.trading_day, o.orb_label
    """
    if limit is not None:
        sql += f"\n        LIMIT {int(limit)}"

    return con.execute(sql, [instrument]).fetchdf()


# ---------------------------------------------------------------------------
# Task 3: Gap quantification
# ---------------------------------------------------------------------------
def analyze_mfe_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify gap between capped and TRUE MFE per combo."""
    valid = df.dropna(subset=["true_mfe_r", "capped_mfe_r"]).copy()
    if len(valid) == 0:
        return pd.DataFrame()

    valid["gap_r"] = valid["true_mfe_r"] - valid["capped_mfe_r"]

    rows = []
    for (orb_label, orb_minutes, rr_target), grp in valid.groupby(
        ["orb_label", "orb_minutes", "rr_target"]
    ):
        n = len(grp)
        if n < MIN_TRADES:
            continue

        gap = grp["gap_r"]
        true_mfe = grp["true_mfe_r"]
        capped_mfe = grp["capped_mfe_r"]

        t_stat, p_value = stats.ttest_1samp(gap, 0)

        rows.append({
            "orb_label": orb_label,
            "orb_minutes": orb_minutes,
            "rr_target": rr_target,
            "n_trades": n,
            "mean_gap": round(float(gap.mean()), 4),
            "median_gap": round(float(gap.median()), 4),
            "p90_gap": round(float(gap.quantile(0.90)), 4),
            "p95_gap": round(float(gap.quantile(0.95)), 4),
            "p99_gap": round(float(gap.quantile(0.99)), 4),
            "unicorn_pct": round(
                float((true_mfe > 3 * rr_target).sum()) / n * 100, 2
            ),
            "mega_unicorn_pct": round(
                float((true_mfe > 5 * rr_target).sum()) / n * 100, 2
            ),
            "mean_true_mfe": round(float(true_mfe.mean()), 4),
            "mean_capped_mfe": round(float(capped_mfe.mean()), 4),
            "t_stat": round(float(t_stat), 3),
            "p_value": float(p_value),
        })

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    summary.sort_values("unicorn_pct", ascending=False, inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary


def print_gap_table(summary: pd.DataFrame, instrument: str) -> None:
    """Print formatted MFE gap analysis table."""
    header = f" {instrument}: MFE Gap Analysis "
    print(f"\n  {'=' * 25}{header}{'=' * 25}")
    print(
        f"  {'Session':25s}| {'O':>2s} | {'RR':>3s} | {'N':>5s} "
        f"| {'MeanGap':>7s} | {'P90':>6s} | {'P95':>6s} "
        f"| {'Uni%':>5s} | {'Mega%':>5s} | {'p-value':>7s}"
    )
    print(f"  {'-' * 95}")

    for _, r in summary.iterrows():
        p_str = "<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
        print(
            f"  {r['orb_label']:25s}| {int(r['orb_minutes']):>2d} "
            f"| {r['rr_target']:>3.1f} | {int(r['n_trades']):>5d} "
            f"| {r['mean_gap']:>+6.2f}R | {r['p90_gap']:>+5.1f}R "
            f"| {r['p95_gap']:>+5.1f}R | {r['unicorn_pct']:>4.1f}% "
            f"| {r['mega_unicorn_pct']:>4.1f}% | {p_str:>7s}"
        )
    print()


# ---------------------------------------------------------------------------
# Task 4: Predictor analysis with BH FDR
# ---------------------------------------------------------------------------
PREDICTOR_COLS = [
    "atr_20",
    "gap_open_points",
    "overnight_range",
    "prev_day_range",
    "garch_atr_ratio",
    "atr_vel_ratio",
    "orb_size",              # session-specific ORB size, extracted per-row
    "overnight_expansion",   # overnight_range / atr_20 (derived)
]


def load_predictor_features(
    con: duckdb.DuckDBPyConnection, instrument: str
) -> pd.DataFrame:
    """Load daily_features predictor columns for merging with outcomes.

    CRITICAL: daily_features has 3 rows per (trading_day, symbol) — one per
    orb_minutes.  We return trading_day + orb_minutes as join keys so the
    caller can merge on both columns (triple-join rule).
    """
    # Build list of orb_{SESSION}_size columns to fetch
    orb_size_cols = [f"orb_{s}_size" for s in SESSIONS]
    orb_size_sql = ", ".join(orb_size_cols)

    sql = f"""
        SELECT trading_day, orb_minutes,
               atr_20, gap_open_points, overnight_range,
               prev_day_range, prev_day_direction,
               garch_atr_ratio, atr_vel_ratio,
               {orb_size_sql}
        FROM daily_features
        WHERE symbol = ?
    """
    return con.execute(sql, [instrument]).fetchdf()


def _extract_orb_size(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, extract the session-specific ORB size into 'orb_size'."""
    df["orb_size"] = np.nan
    for session in SESSIONS:
        col = f"orb_{session}_size"
        if col in df.columns:
            mask = df["orb_label"] == session
            df.loc[mask, "orb_size"] = df.loc[mask, col]
    return df


def _cohen_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size (pooled std)."""
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return float("nan")
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return float("nan")
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def test_unicorn_predictors(
    df: pd.DataFrame,
    instrument: str,
) -> pd.DataFrame:
    """Test which predictors distinguish unicorn trades (true_mfe_r > 3*rr_target).

    For each (orb_label, orb_minutes, rr_target) combo with N >= MIN_TRADES:
    - Binary label: unicorn = (true_mfe_r > 3 * rr_target)
    - Welch's t-test, point-biserial correlation, Cohen's d per predictor
    - Returns DataFrame of all test results (p-values for BH FDR later)
    """
    rows = []
    valid = df.dropna(subset=["true_mfe_r"]).copy()

    for (orb_label, orb_minutes, rr_target), grp in valid.groupby(
        ["orb_label", "orb_minutes", "rr_target"]
    ):
        n = len(grp)
        if n < MIN_TRADES:
            continue

        unicorn_flag = (grp["true_mfe_r"] > 3 * rr_target).astype(int)
        n_unicorn = int(unicorn_flag.sum())
        n_non = n - n_unicorn

        # Need both groups to have >= 5 members
        if n_unicorn < 5 or n_non < 5:
            continue

        for pred in PREDICTOR_COLS:
            if pred not in grp.columns:
                continue

            pred_vals = grp[pred].values.astype(float)
            mask_valid = ~np.isnan(pred_vals)
            if mask_valid.sum() < MIN_TRADES:
                continue

            pred_clean = pred_vals[mask_valid]
            flag_clean = unicorn_flag.values[mask_valid]

            uni_vals = pred_clean[flag_clean == 1]
            non_vals = pred_clean[flag_clean == 0]

            if len(uni_vals) < 5 or len(non_vals) < 5:
                continue

            # Welch's t-test
            t_stat, p_value = stats.ttest_ind(uni_vals, non_vals, equal_var=False)

            # Point-biserial correlation
            corr, corr_p = stats.pointbiserialr(flag_clean, pred_clean)

            # Cohen's d (unicorn - non-unicorn)
            cohen = _cohen_d(uni_vals, non_vals)

            rows.append({
                "orb_label": orb_label,
                "orb_minutes": int(orb_minutes),
                "rr_target": float(rr_target),
                "predictor": pred,
                "n": int(mask_valid.sum()),
                "n_unicorn": int(len(uni_vals)),
                "mean_unicorn": round(float(np.mean(uni_vals)), 4),
                "mean_non_unicorn": round(float(np.mean(non_vals)), 4),
                "t_stat": round(float(t_stat), 3),
                "p_value": float(p_value),
                "cohen_d": round(float(cohen), 4),
                "correlation": round(float(corr), 4),
                "test_type": "welch_t",
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def test_overnight_expansion(df: pd.DataFrame) -> pd.DataFrame:
    """Test overnight_expansion tertiles as unicorn predictor.

    For each (orb_label, orb_minutes, rr_target) combo:
    - Split overnight_expansion into tertiles (low/mid/high)
    - Chi-square test on unicorn rates across tertiles
    - ANOVA on true_mfe_r across tertiles
    """
    rows = []
    valid = df.dropna(subset=["true_mfe_r", "overnight_expansion"]).copy()

    for (orb_label, orb_minutes, rr_target), grp in valid.groupby(
        ["orb_label", "orb_minutes", "rr_target"]
    ):
        n = len(grp)
        if n < MIN_TRADES:
            continue

        # Create tertiles
        try:
            grp = grp.copy()
            grp["oe_tertile"] = pd.qcut(
                grp["overnight_expansion"], q=3, labels=["low", "mid", "high"]
            )
        except ValueError:
            # Not enough unique values for tertiles
            continue

        unicorn_flag = (grp["true_mfe_r"] > 3 * rr_target).astype(int)

        tertile_groups = grp.groupby("oe_tertile", observed=True)
        if len(tertile_groups) < 3:
            continue

        # Chi-square: unicorn rate by tertile
        contingency = pd.crosstab(grp["oe_tertile"], unicorn_flag)
        if contingency.shape[1] < 2:
            # All unicorn or all non-unicorn
            continue

        try:
            chi2, chi2_p, _, _ = stats.chi2_contingency(contingency)
        except ValueError:
            continue

        # ANOVA: true_mfe_r by tertile
        tertile_mfe = [g["true_mfe_r"].values for _, g in tertile_groups]
        if any(len(t) < 5 for t in tertile_mfe):
            continue
        f_stat, anova_p = stats.f_oneway(*tertile_mfe)

        # Unicorn rates per tertile
        uni_rates = {}
        for label in ["low", "mid", "high"]:
            t_grp = grp[grp["oe_tertile"] == label]
            if len(t_grp) > 0:
                uni_rates[label] = round(
                    float((t_grp["true_mfe_r"] > 3 * rr_target).mean()) * 100, 2
                )
            else:
                uni_rates[label] = 0.0

        rows.append({
            "orb_label": orb_label,
            "orb_minutes": int(orb_minutes),
            "rr_target": float(rr_target),
            "predictor": "overnight_expansion_tertile",
            "n": n,
            "n_unicorn": int(unicorn_flag.sum()),
            "chi2": round(float(chi2), 3),
            "p_value": float(chi2_p),  # chi-square p for BH FDR
            "f_stat": round(float(f_stat), 3),
            "anova_p": float(anova_p),
            "uni_rate_low": uni_rates["low"],
            "uni_rate_mid": uni_rates["mid"],
            "uni_rate_high": uni_rates["high"],
            "test_type": "chi2_anova",
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def apply_bh_fdr_to_results(
    predictor_results: pd.DataFrame,
    overnight_results: pd.DataFrame,
    q: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply BH FDR across ALL tests from both result sets.

    Returns updated DataFrames with 'fdr_survives' column.
    """
    # Collect all p-values into one array with source tracking
    all_p = []
    sources = []  # (source_df_name, row_index)

    if len(predictor_results) > 0:
        for idx, row in predictor_results.iterrows():
            all_p.append(row["p_value"])
            sources.append(("predictor", idx))

    if len(overnight_results) > 0:
        for idx, row in overnight_results.iterrows():
            all_p.append(row["p_value"])
            sources.append(("overnight", idx))

    if not all_p:
        return predictor_results, overnight_results

    survivors = bh_fdr(all_p, q=q)

    # Mark survivors
    if len(predictor_results) > 0:
        predictor_results = predictor_results.copy()
        predictor_results["fdr_survives"] = False
    if len(overnight_results) > 0:
        overnight_results = overnight_results.copy()
        overnight_results["fdr_survives"] = False

    for global_idx in survivors:
        source_name, row_idx = sources[global_idx]
        if source_name == "predictor":
            predictor_results.loc[row_idx, "fdr_survives"] = True
        else:
            overnight_results.loc[row_idx, "fdr_survives"] = True

    return predictor_results, overnight_results


def print_predictor_table(
    predictor_results: pd.DataFrame,
    overnight_results: pd.DataFrame,
    instrument: str,
) -> None:
    """Print formatted predictor analysis table."""
    header = f" {instrument}: Unicorn Predictor Analysis "
    print(f"\n  {'=' * 25}{header}{'=' * 25}")

    # --- Welch t-test results ---
    if len(predictor_results) > 0:
        # Show FDR survivors and near-misses (p < 0.10 before FDR)
        show = predictor_results[predictor_results["p_value"] < 0.10].copy()
        show = show.sort_values("p_value")

        if len(show) > 0:
            print(
                f"\n  {'Session':20s}| {'O':>2s} | {'RR':>3s} "
                f"| {'Predictor':22s}| {'Cohen d':>8s} | {'p-value':>8s} | {'FDR':>3s}"
            )
            print(f"  {'-' * 78}")

            for _, r in show.iterrows():
                p_str = (
                    "<0.001" if r["p_value"] < 0.001
                    else f"{r['p_value']:.4f}"
                )
                fdr_str = "YES" if r.get("fdr_survives", False) else "NO"
                print(
                    f"  {r['orb_label']:20s}| {r['orb_minutes']:>2d} "
                    f"| {r['rr_target']:>3.1f} "
                    f"| {r['predictor']:22s}| {r['cohen_d']:>+7.3f} "
                    f"| {p_str:>8s} | {fdr_str:>3s}"
                )
        else:
            print("\n  No predictors with p < 0.10 (before FDR).")

        n_total = len(predictor_results)
        n_fdr = int(predictor_results["fdr_survives"].sum()) if "fdr_survives" in predictor_results.columns else 0
        print(f"\n  Welch t-tests: {n_total} total, {n_fdr} BH FDR survivors (q=0.05)")

    # --- Overnight expansion results ---
    if len(overnight_results) > 0:
        show_oe = overnight_results[overnight_results["p_value"] < 0.10].copy()
        show_oe = show_oe.sort_values("p_value")

        if len(show_oe) > 0:
            print(f"\n  Overnight Expansion Tertile Analysis:")
            print(
                f"  {'Session':20s}| {'O':>2s} | {'RR':>3s} "
                f"| {'chi2':>6s} | {'p-val':>7s} | {'FDR':>3s} "
                f"| {'Uni%Lo':>6s} | {'Uni%Mi':>6s} | {'Uni%Hi':>6s}"
            )
            print(f"  {'-' * 90}")

            for _, r in show_oe.iterrows():
                p_str = (
                    "<0.001" if r["p_value"] < 0.001
                    else f"{r['p_value']:.4f}"
                )
                fdr_str = "YES" if r.get("fdr_survives", False) else "NO"
                print(
                    f"  {r['orb_label']:20s}| {r['orb_minutes']:>2d} "
                    f"| {r['rr_target']:>3.1f} "
                    f"| {r['chi2']:>6.2f} | {p_str:>7s} | {fdr_str:>3s} "
                    f"| {r['uni_rate_low']:>5.1f}% | {r['uni_rate_mid']:>5.1f}% "
                    f"| {r['uni_rate_high']:>5.1f}%"
                )

        n_oe_total = len(overnight_results)
        n_oe_fdr = int(overnight_results["fdr_survives"].sum()) if "fdr_survives" in overnight_results.columns else 0
        print(f"\n  Overnight expansion tests: {n_oe_total} total, {n_oe_fdr} BH FDR survivors (q=0.05)")

    print()


def print_unicorn_summary(summary: pd.DataFrame) -> None:
    """Print top unicorn producers."""
    top = summary[summary["n_trades"] >= MIN_TRADES].nlargest(10, "unicorn_pct")
    if len(top) == 0:
        return
    print(f"  TOP UNICORN PRODUCERS (>3xRR, minimum {MIN_TRADES} trades):")
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        print(
            f"  {rank:>2d}. {r['orb_label']} O{int(r['orb_minutes'])} "
            f"RR{r['rr_target']:.1f}: {r['unicorn_pct']:.1f}% unicorn rate "
            f"(N={int(r['n_trades']):,})"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trend-Day MFE Discovery — TRUE session MFE from bars_1m"
    )
    parser.add_argument(
        "--instrument", type=str, default=None,
        help="Single instrument (default: all active)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print row counts per session and exit",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap rows per instrument (for testing)",
    )
    parser.add_argument(
        "--predictors", action="store_true",
        help="Enable predictor analysis (Task 3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="research/output/",
        help="Output directory for results",
    )
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else ACTIVE_INSTRUMENTS

    print(f"=== Trend-Day MFE Research ===")
    print(f"DB: {DB_PATH}")
    print(f"Instruments: {instruments}")
    print()

    con = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        for instrument in instruments:
            t0 = time.time()
            df = load_outcomes(con, instrument, limit=args.limit)
            elapsed = time.time() - t0
            print(f"--- {instrument}: {len(df):,} outcomes loaded ({elapsed:.1f}s) ---")

            # Row counts per session
            if len(df) > 0:
                session_counts = (
                    df.groupby("orb_label")
                    .size()
                    .sort_values(ascending=False)
                )
                for session, count in session_counts.items():
                    print(f"  {session:25s} {count:>7,}")
            print()

            if args.dry_run:
                continue

            # ----- Task 2: TRUE MFE computation from bars_1m -----
            if len(df) == 0:
                print(f"  No outcomes for {instrument}, skipping.\n")
                continue

            cost_spec = COST_SPECS[instrument]

            # Group by trading_day for efficient bar loading
            grouped = df.groupby("trading_day")
            n_days = len(grouped)
            print(f"  Computing TRUE MFE: {len(df):,} outcomes across {n_days:,} trading days...")

            t_mfe = time.time()
            true_mfe_results = []
            rows_processed = 0

            for day_idx, (td, day_df) in enumerate(grouped):
                # Load bars once per trading day
                day_bars = load_bars_1m_for_day(con, instrument, td)
                _, td_end = compute_trading_day_utc_range(td)
                td_end_ts = pd.Timestamp(td_end)

                # Process all outcomes for this day
                for row_idx, row in day_df.iterrows():
                    entry_ts = pd.Timestamp(row["entry_ts"])

                    # Filter to post-entry bars up to session end
                    post_entry = day_bars[
                        (day_bars["ts_utc"] > entry_ts) & (day_bars["ts_utc"] <= td_end_ts)
                    ]

                    result = compute_true_session_mfe(
                        bars_1m=post_entry,
                        entry_price=row["entry_price"],
                        stop_price=row["stop_price"],
                        break_dir=row["break_dir"],
                        cost_spec=cost_spec,
                    )
                    result["idx"] = row_idx
                    true_mfe_results.append(result)

                rows_processed += len(day_df)
                if (day_idx + 1) % 200 == 0 or (day_idx + 1) == n_days:
                    elapsed_mfe = time.time() - t_mfe
                    pct = rows_processed / len(df) * 100
                    print(
                        f"    [{day_idx + 1:,}/{n_days:,} days] "
                        f"{rows_processed:,}/{len(df):,} rows ({pct:.0f}%) "
                        f"— {elapsed_mfe:.1f}s"
                    )

            # Merge results back
            mfe_df = pd.DataFrame(true_mfe_results).set_index("idx")
            for col in mfe_df.columns:
                df[col] = mfe_df[col]

            elapsed_mfe = time.time() - t_mfe
            print(f"  TRUE MFE complete: {elapsed_mfe:.1f}s total\n")

            # Verification: compare capped vs true MFE for first 10 rows
            valid = df.dropna(subset=["true_mfe_r", "capped_mfe_r"]).head(10)
            if len(valid) > 0:
                print("  Verification (first 10 rows): capped_mfe_r vs true_mfe_r")
                print(f"  {'capped':>10s}  {'true':>10s}  {'gap':>10s}  {'ok':>4s}")
                for _, r in valid.iterrows():
                    gap = r["true_mfe_r"] - r["capped_mfe_r"]
                    ok = "YES" if r["true_mfe_r"] >= r["capped_mfe_r"] - 0.001 else "NO"
                    print(
                        f"  {r['capped_mfe_r']:10.4f}  {r['true_mfe_r']:10.4f}  "
                        f"{gap:10.4f}  {ok:>4s}"
                    )

                # Aggregate check
                check_df = df.dropna(subset=["true_mfe_r", "capped_mfe_r"])
                n_valid = len(check_df)
                n_ok = int(
                    (check_df["true_mfe_r"] >= check_df["capped_mfe_r"] - 0.001).sum()
                )
                print(f"\n  Aggregate: {n_ok:,}/{n_valid:,} rows have true_mfe_r >= capped_mfe_r")
                print()

            # ----- Task 3: Gap quantification -----
            gap_summary = analyze_mfe_gap(df)
            if len(gap_summary) > 0:
                print_gap_table(gap_summary, instrument)
                print_unicorn_summary(gap_summary)

            # ----- Task 4: Predictor analysis with BH FDR -----
            if args.predictors:
                print(f"  Loading predictor features for {instrument}...")
                features = load_predictor_features(con, instrument)
                print(f"    {len(features):,} feature rows loaded")

                # Merge on (trading_day, orb_minutes) — triple-join rule
                df = df.merge(
                    features,
                    on=["trading_day", "orb_minutes"],
                    how="left",
                )

                # Extract per-row ORB size from session-specific columns
                df = _extract_orb_size(df)

                # Derive overnight_expansion
                df["overnight_expansion"] = np.where(
                    (df["atr_20"].notna()) & (df["atr_20"] > 0),
                    df["overnight_range"] / df["atr_20"],
                    np.nan,
                )

                # Run predictor tests (suppress scipy precision warnings for small N)
                print(f"  Running unicorn predictor tests...")
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Precision loss occurred",
                        category=RuntimeWarning,
                    )
                    predictor_results = test_unicorn_predictors(df, instrument)
                    overnight_results = test_overnight_expansion(df)

                n_pred = len(predictor_results)
                n_oe = len(overnight_results)
                print(
                    f"    {n_pred} predictor tests, "
                    f"{n_oe} overnight expansion tests"
                )

                # Apply BH FDR across ALL tests
                predictor_results, overnight_results = apply_bh_fdr_to_results(
                    predictor_results, overnight_results, q=0.05
                )

                n_fdr = 0
                if len(predictor_results) > 0 and "fdr_survives" in predictor_results.columns:
                    n_fdr += int(predictor_results["fdr_survives"].sum())
                if len(overnight_results) > 0 and "fdr_survives" in overnight_results.columns:
                    n_fdr += int(overnight_results["fdr_survives"].sum())
                print(f"    BH FDR survivors (q=0.05): {n_fdr}/{n_pred + n_oe}")

                print_predictor_table(predictor_results, overnight_results, instrument)

    finally:
        con.close()

    if args.dry_run:
        print("Dry run complete.")
        return

    print("Done.")


if __name__ == "__main__":
    main()
