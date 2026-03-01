#!/usr/bin/env python3
"""
research_vol_regime_switching.py — Volatility Regime-Dependent Parameter Switching

HYPOTHESIS:
    H0: Optimal ORB breakout parameters (RR target, filter stringency) are
        independent of the current volatility regime.
    H1: Different vol regimes have systematically different optimal parameters,
        and regime-adaptive switching improves risk-adjusted returns.

MECHANISM:
    - High-vol: larger ORBs, friction is a smaller fraction of risk, higher
      RR targets are reachable.
    - Low-vol: smaller ORBs, friction eats more, lower RR targets are more
      realistic, stricter G-filters may be needed.
    - Extends existing ATR contraction AVOID signal from binary to continuous
      regime adaptation.

REGIME DEFINITION:
    Primary variable: atr_20 from daily_features (20-day ATR SMA).
    Classification: Expanding-window percentile rank with shift(1) (no look-ahead).
    Minimum 60-day warmup. Terciles: LOW (<33rd pctile), MID (33-67), HIGH (>67).

PARAMETER GRID (tight for tractability):
    - 4 instruments x active sessions (pre-scanned for validated strategies)
    - Entry models: E1, E2
    - RR targets: 1.5, 2.0, 2.5, 3.0
    - Filters: G4, G6
    - CB: 1 only

STATISTICAL TESTS:
    1. Per-regime one-sample t-test vs zero
    2. Kruskal-Wallis across regimes for same parameter set
    3. Optimal parameter shift: does best RR/filter differ across regimes?
    4. Adaptive vs static: paired t-test on daily pnl_r
    5. BH FDR at q=0.05 across all tests
    6. Year-by-year stability for survivors

NO LOOK-AHEAD:
    - atr_pct_rank uses expanding window with shift(1) on atr_20.
    - vol_regime[i] depends only on atr_20[0:i-1].
    - Spot-check assertion included in script.

OUTPUT:
    - Console summary
    - research/output/vol_regime_switching_summary.md
    - research/output/vol_regime_switching_results.csv
"""

import sys
import os
from pathlib import Path

# Windows line buffering + encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, get_enabled_sessions
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ORB_DURATION_MINUTES

DB_PATH = str(GOLD_DB_PATH)
OUTPUT_DIR = PROJECT_ROOT / "research" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Parameter grid ---------------------------------------------------------
ENTRY_MODELS = ["E1", "E2"]
RR_TARGETS = [1.5, 2.0, 2.5, 3.0]
FILTER_THRESHOLDS = {"G4": 4.0, "G6": 6.0}
CB = 1  # CB1 only

# -- Statistical helpers ----------------------------------------------------

def ttest_1s(arr, mu=0.0):
    """One-sample t-test vs mu. Returns (n, mean, wr, t, p)."""
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 10:
        return len(a), float("nan"), float("nan"), float("nan"), float("nan")
    t_stat, p_val = stats.ttest_1samp(a, mu)
    return len(a), float(a.mean()), float((a > 0).mean()), float(t_stat), float(p_val)


def bh_fdr(p_values, q=0.05):
    """Benjamini-Hochberg FDR correction. Returns set of rejected indices."""
    n = len(p_values)
    if n == 0:
        return set()
    ranked = sorted(enumerate(p_values), key=lambda x: x[1])
    thresholds = [q * (k + 1) / n for k in range(n)]
    max_k = -1
    for k, (_, p) in enumerate(ranked):
        if not np.isnan(p) and p <= thresholds[k]:
            max_k = k
    rejected = set()
    if max_k >= 0:
        for k, (idx, _) in enumerate(ranked[:max_k + 1]):
            rejected.add(idx)
    return rejected


def sharpe(arr):
    """Annualized Sharpe ratio (daily R-multiples, ~252 trading days)."""
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 10 or np.std(a, ddof=1) == 0:
        return float("nan")
    return float(a.mean() / np.std(a, ddof=1) * np.sqrt(252))


def sample_label(n):
    if n < 30:
        return "INVALID"
    if n < 100:
        return "REGIME"
    if n < 200:
        return "PRELIMINARY"
    if n < 500:
        return "CORE"
    return "HIGH-CONFIDENCE"


# -- Pre-scan validated sessions --------------------------------------------

def get_validated_sessions(con):
    """Return dict of {instrument: [session_list]} from validated_setups."""
    try:
        rows = con.execute("""
            SELECT DISTINCT instrument, orb_label
            FROM validated_setups
            WHERE entry_model IN ('E1', 'E2')
        """).fetchall()
    except Exception:
        # Table might not exist; fall back to enabled sessions
        print("  WARNING: validated_setups not found, using all enabled sessions")
        result = {}
        for inst in ACTIVE_ORB_INSTRUMENTS:
            result[inst] = get_enabled_sessions(inst)
        return result

    result = {}
    for inst, session in rows:
        if inst in ACTIVE_ORB_INSTRUMENTS:
            result.setdefault(inst, set()).add(session)

    # Convert to sorted lists
    return {k: sorted(v) for k, v in result.items()}


# -- Data loading ----------------------------------------------------------

def load_data(con, instrument, session):
    """Load daily_features + orb_outcomes for (instrument, session).

    Returns DataFrame with trading_day, atr_20, orb_size, rr_target,
    confirm_bars, entry_model, pnl_r, outcome, year.
    """
    size_col = f"orb_{session}_size"
    orb_minutes = ORB_DURATION_MINUTES.get(session, 5)

    sql = f"""
    SELECT
        d.trading_day,
        d.atr_20,
        d.{size_col} AS orb_size,
        o.rr_target,
        o.confirm_bars,
        o.entry_model,
        o.pnl_r,
        o.outcome,
        EXTRACT(YEAR FROM d.trading_day) AS year
    FROM daily_features d
    JOIN orb_outcomes o
        ON o.trading_day = d.trading_day
        AND o.symbol = d.symbol
        AND o.orb_minutes = d.orb_minutes
    WHERE d.symbol = '{instrument}'
      AND d.orb_minutes = {orb_minutes}
      AND d.atr_20 IS NOT NULL
      AND d.{size_col} IS NOT NULL
      AND o.orb_label = '{session}'
      AND o.confirm_bars = {CB}
      AND o.entry_model IN ('E1', 'E2')
      AND o.pnl_r IS NOT NULL
      AND o.outcome IN ('win', 'loss')
    ORDER BY d.trading_day
    """
    return con.execute(sql).fetchdf()


def compute_vol_regime_for_instrument(con, instrument):
    """Compute vol_regime for ALL trading days of an instrument.

    Uses expanding-window percentile rank on atr_20 from daily_features
    (orb_minutes=5, one row per day). shift(1) ensures no look-ahead.
    60-day minimum warmup.

    Returns DataFrame with (trading_day, atr_20, atr_pct_rank, vol_regime).
    """
    sql = f"""
    SELECT DISTINCT trading_day, atr_20
    FROM daily_features
    WHERE symbol = '{instrument}'
      AND orb_minutes = 5
      AND atr_20 IS NOT NULL
    ORDER BY trading_day
    """
    days = con.execute(sql).fetchdf()
    if days.empty:
        days["atr_pct_rank"] = np.nan
        days["vol_regime"] = np.nan
        return days

    days = days.sort_values("trading_day").reset_index(drop=True)
    days["atr_pct_rank"] = (
        days["atr_20"]
        .expanding(min_periods=60)
        .rank(pct=True)
        .shift(1)
    )
    days["vol_regime"] = pd.cut(
        days["atr_pct_rank"],
        bins=[0, 0.333, 0.667, 1.001],
        labels=["LOW", "MID", "HIGH"],
        right=False,
    )
    return days[["trading_day", "atr_20", "atr_pct_rank", "vol_regime"]]


def merge_vol_regime(df, vol_regime_df):
    """Merge pre-computed vol_regime onto outcome data."""
    if df.empty:
        df["atr_pct_rank"] = np.nan
        df["vol_regime"] = np.nan
        return df
    return df.merge(
        vol_regime_df[["trading_day", "atr_pct_rank", "vol_regime"]],
        on="trading_day",
        how="left",
    )


def verify_no_lookahead(vol_regime_df):
    """Spot-check: vol_regime[i] must depend only on atr_20 before day i.

    Recompute rank on first half of data; it should exactly match the
    full-data computation (since expanding rank only uses prior data).
    """
    days = vol_regime_df.dropna(subset=["atr_pct_rank"]).reset_index(drop=True)
    if len(days) < 10:
        return True

    # Recompute on first half only -- should match stored rank
    half = len(days) // 2
    recomputed = (
        vol_regime_df["atr_20"].iloc[:half]
        .expanding(min_periods=60)
        .rank(pct=True)
        .shift(1)
    )

    stored = vol_regime_df["atr_pct_rank"].iloc[:half]
    both_valid = stored.notna() & recomputed.notna()
    if both_valid.sum() < 5:
        return True

    max_diff = (stored[both_valid] - recomputed[both_valid]).abs().max()
    if max_diff > 1e-10:
        print(f"  LOOKAHEAD WARNING: max rank diff = {max_diff:.6f}")
        return False
    return True


# -- Main analysis ---------------------------------------------------------

def run():
    print()
    print("=" * 78)
    print("VOLATILITY REGIME-DEPENDENT PARAMETER SWITCHING RESEARCH")
    print("=" * 78)

    con = duckdb.connect(DB_PATH, read_only=True)

    # Pre-scan validated sessions
    validated = get_validated_sessions(con)
    print(f"\nInstruments: {ACTIVE_ORB_INSTRUMENTS}")
    print(f"Validated sessions per instrument:")
    for inst, sessions in sorted(validated.items()):
        print(f"  {inst}: {sessions}")

    all_records = []      # Per-regime t-test results
    kw_records = []       # Kruskal-Wallis results
    adaptive_records = [] # Adaptive vs static comparison
    report_lines = []

    def report(line=""):
        print(line)
        report_lines.append(line)

    report(f"\nParameter grid: RR={RR_TARGETS}, Filters={list(FILTER_THRESHOLDS.keys())}, "
           f"Entry={ENTRY_MODELS}, CB={CB}")
    report(f"BH FDR q=0.05 (stricter due to high comparison count)")
    report(f"Vol regime: expanding-window ATR percentile rank, terciles (LOW/MID/HIGH)")
    report(f"Warmup: 60 days minimum per instrument")

    combos_tested = 0
    combos_skipped_n = 0

    # -- Main loop: instrument x session ------------------------------------
    # Pre-compute vol regime once per instrument (on ALL trading days)
    vol_regimes = {}
    for instrument in ACTIVE_ORB_INSTRUMENTS:
        vr = compute_vol_regime_for_instrument(con, instrument)
        if not verify_no_lookahead(vr):
            report(f"\n  FATAL: Lookahead detected in vol regime for {instrument}")
            continue
        vol_regimes[instrument] = vr
        report(f"\n  {instrument}: vol regime computed on {len(vr)} trading days, "
               f"lookahead check PASSED")

    for instrument in ACTIVE_ORB_INSTRUMENTS:
        sessions = validated.get(instrument, [])
        if not sessions:
            report(f"\n  {instrument}: no validated sessions, skipping")
            continue
        if instrument not in vol_regimes:
            report(f"\n  {instrument}: skipped (lookahead check failed)")
            continue

        vr_df = vol_regimes[instrument]

        for session in sessions:
            report(f"\n{'-' * 78}")
            report(f"  {instrument} / {session}")
            report(f"{'-' * 78}")

            df_raw = load_data(con, instrument, session)
            if df_raw.empty:
                report(f"  NO DATA for {instrument} {session}")
                continue

            df_raw = merge_vol_regime(df_raw, vr_df)

            # Drop warmup days (no vol_regime)
            df = df_raw.dropna(subset=["vol_regime"]).copy()
            if len(df) < 30:
                report(f"  Insufficient data after warmup: N={len(df)}")
                continue

            # Regime distribution
            unique_days = df[["trading_day", "vol_regime"]].drop_duplicates("trading_day")
            for regime in ["LOW", "MID", "HIGH"]:
                n_days = (unique_days["vol_regime"] == regime).sum()
                report(f"  {regime}: {n_days} trading days "
                       f"({100 * n_days / len(unique_days):.0f}%)")

            # -- Per-regime analysis for each parameter combo -----------
            for entry_model in ENTRY_MODELS:
                for rr in RR_TARGETS:
                    for filt_name, filt_min in FILTER_THRESHOLDS.items():
                        combo_label = f"{instrument}_{session}_{entry_model}_RR{rr}_{filt_name}"

                        # Filter data
                        mask = (
                            (df["entry_model"] == entry_model)
                            & (df["rr_target"] == rr)
                            & (df["orb_size"] >= filt_min)
                        )
                        sub = df[mask]

                        if len(sub) < 30:
                            combos_skipped_n += 1
                            continue

                        combos_tested += 1

                        # Overall stats
                        n_all, avg_all, wr_all, t_all, p_all = ttest_1s(sub["pnl_r"])
                        sh_all = sharpe(sub["pnl_r"])

                        # Per-regime stats
                        regime_data = {}
                        regime_pnl_arrays = {}
                        for regime in ["LOW", "MID", "HIGH"]:
                            r_sub = sub[sub["vol_regime"] == regime]["pnl_r"].values
                            n, avg, wr, t, p = ttest_1s(r_sub)
                            sh = sharpe(r_sub)
                            regime_data[regime] = {
                                "n": n, "avg_r": avg, "wr": wr, "t": t, "p": p, "sharpe": sh
                            }
                            regime_pnl_arrays[regime] = r_sub[~np.isnan(r_sub)]

                            all_records.append({
                                "instrument": instrument,
                                "session": session,
                                "entry_model": entry_model,
                                "rr": rr,
                                "filter": filt_name,
                                "regime": regime,
                                "n": n,
                                "avg_r": avg,
                                "wr": wr,
                                "t_stat": t,
                                "p_value": p,
                                "sharpe": sh,
                                "combo_label": combo_label,
                            })

                        # Kruskal-Wallis across regimes
                        valid_groups = [a for a in regime_pnl_arrays.values() if len(a) >= 10]
                        kw_p = float("nan")
                        if len(valid_groups) == 3:
                            try:
                                kw_stat, kw_p = stats.kruskal(*valid_groups)
                            except Exception:
                                kw_p = float("nan")

                        kw_records.append({
                            "combo_label": combo_label,
                            "instrument": instrument,
                            "session": session,
                            "entry_model": entry_model,
                            "rr": rr,
                            "filter": filt_name,
                            "n_total": n_all,
                            "avg_r_all": avg_all,
                            "sharpe_all": sh_all,
                            "kw_p": kw_p,
                            "low_avg": regime_data["LOW"]["avg_r"],
                            "mid_avg": regime_data["MID"]["avg_r"],
                            "high_avg": regime_data["HIGH"]["avg_r"],
                            "low_n": regime_data["LOW"]["n"],
                            "mid_n": regime_data["MID"]["n"],
                            "high_n": regime_data["HIGH"]["n"],
                            "low_sharpe": regime_data["LOW"]["sharpe"],
                            "mid_sharpe": regime_data["MID"]["sharpe"],
                            "high_sharpe": regime_data["HIGH"]["sharpe"],
                        })

            # -- Adaptive vs static comparison per (instrument, session, entry_model) --
            for entry_model in ENTRY_MODELS:
                em_df = df[df["entry_model"] == entry_model].copy()
                if len(em_df) < 60:
                    continue

                # Find best static parameter set (best Sharpe across all regimes)
                best_static_sharpe = -999
                best_static_rr = None
                best_static_filt = None

                for rr in RR_TARGETS:
                    for filt_name, filt_min in FILTER_THRESHOLDS.items():
                        mask = (em_df["rr_target"] == rr) & (em_df["orb_size"] >= filt_min)
                        sub = em_df[mask]
                        if len(sub) < 30:
                            continue
                        sh = sharpe(sub["pnl_r"])
                        if not np.isnan(sh) and sh > best_static_sharpe:
                            best_static_sharpe = sh
                            best_static_rr = rr
                            best_static_filt = filt_name

                if best_static_rr is None:
                    continue

                # Find best parameter per regime
                best_per_regime = {}
                for regime in ["LOW", "MID", "HIGH"]:
                    r_df = em_df[em_df["vol_regime"] == regime]
                    best_sh = -999
                    best_rr = None
                    best_filt = None
                    for rr in RR_TARGETS:
                        for filt_name, filt_min in FILTER_THRESHOLDS.items():
                            mask = (r_df["rr_target"] == rr) & (r_df["orb_size"] >= filt_min)
                            sub = r_df[mask]
                            if len(sub) < 10:  # Relaxed for per-regime
                                continue
                            sh = sharpe(sub["pnl_r"])
                            if not np.isnan(sh) and sh > best_sh:
                                best_sh = sh
                                best_rr = rr
                                best_filt = filt_name
                    best_per_regime[regime] = {
                        "rr": best_rr, "filter": best_filt, "sharpe": best_sh
                    }

                # Check if regimes picked different parameters
                regime_params = set()
                for regime, bp in best_per_regime.items():
                    if bp["rr"] is not None:
                        regime_params.add((bp["rr"], bp["filter"]))

                params_differ = len(regime_params) > 1

                # Compute adaptive pnl_r series: for each day, use the parameter
                # set optimal for that day's regime
                adaptive_pnl = []
                static_pnl = []

                for _, day_rows in em_df.groupby("trading_day"):
                    regime = day_rows["vol_regime"].iloc[0]
                    if pd.isna(regime):
                        continue
                    bp = best_per_regime.get(regime, {})
                    if bp.get("rr") is None:
                        continue

                    # Adaptive: use regime-best params
                    adapt_mask = (
                        (day_rows["rr_target"] == bp["rr"])
                        & (day_rows["orb_size"] >= FILTER_THRESHOLDS.get(bp["filter"], 0))
                    )
                    adapt_rows = day_rows[adapt_mask]
                    if not adapt_rows.empty:
                        adaptive_pnl.append(adapt_rows["pnl_r"].iloc[0])
                    else:
                        adaptive_pnl.append(0.0)  # No trade = 0 R

                    # Static: use global best
                    stat_mask = (
                        (day_rows["rr_target"] == best_static_rr)
                        & (day_rows["orb_size"] >= FILTER_THRESHOLDS.get(best_static_filt, 0))
                    )
                    stat_rows = day_rows[stat_mask]
                    if not stat_rows.empty:
                        static_pnl.append(stat_rows["pnl_r"].iloc[0])
                    else:
                        static_pnl.append(0.0)

                adaptive_arr = np.array(adaptive_pnl)
                static_arr = np.array(static_pnl)

                # Paired t-test: adaptive - static
                diff = adaptive_arr - static_arr
                non_zero_diff = diff[diff != 0]
                paired_t = float("nan")
                paired_p = float("nan")
                if len(non_zero_diff) >= 10:
                    paired_t, paired_p = stats.ttest_1samp(non_zero_diff, 0)
                    paired_t = float(paired_t)
                    paired_p = float(paired_p)

                adaptive_records.append({
                    "instrument": instrument,
                    "session": session,
                    "entry_model": entry_model,
                    "best_static": f"RR{best_static_rr}_{best_static_filt}",
                    "static_sharpe": best_static_sharpe,
                    "params_differ": params_differ,
                    "regime_params": {r: f"RR{bp['rr']}_{bp['filter']}"
                                      for r, bp in best_per_regime.items()
                                      if bp['rr'] is not None},
                    "adaptive_avg_r": float(np.nanmean(adaptive_arr)),
                    "static_avg_r": float(np.nanmean(static_arr)),
                    "delta_avg_r": float(np.nanmean(adaptive_arr) - np.nanmean(static_arr)),
                    "adaptive_sharpe": sharpe(adaptive_arr),
                    "static_sharpe_daily": sharpe(static_arr),
                    "paired_t": paired_t,
                    "paired_p": paired_p,
                    "n_days": len(adaptive_arr),
                })

    con.close()

    # -- BH FDR on per-regime t-tests --------------------------------------
    report(f"\n{'=' * 78}")
    report(f"RESULTS SUMMARY")
    report(f"{'=' * 78}")
    report(f"\nCombos tested: {combos_tested}")
    report(f"Combos skipped (N<30): {combos_skipped_n}")

    # BH FDR on one-sample t-tests (per-regime)
    valid_records = [(i, r) for i, r in enumerate(all_records)
                     if not np.isnan(r["p_value"]) and r["n"] >= 30]
    p_vals = [r["p_value"] for _, r in valid_records]
    rejected = bh_fdr(p_vals, q=0.05)

    report(f"\nPer-regime t-tests: {len(valid_records)} valid tests")
    report(f"BH FDR survivors (q=0.05): {len(rejected)}")

    # Mark survivors
    for local_i, (orig_i, r) in enumerate(valid_records):
        all_records[orig_i]["bh_survives"] = local_i in rejected

    # Show BH survivors
    if rejected:
        report(f"\n  BH FDR SURVIVORS (per-regime t-test vs zero):")
        report(f"  {'Combo':<45} {'Regime':<6} {'N':>5} {'avgR':>7} {'WR':>6} "
               f"{'Sharpe':>7} {'p':>8}")
        report(f"  {'-' * 90}")
        survivors_list = [(local_i, valid_records[local_i]) for local_i in sorted(rejected)]
        survivors_list.sort(key=lambda x: x[1][1]["p_value"])
        for local_i, (orig_i, r) in survivors_list:
            report(f"  {r['combo_label']:<45} {r['regime']:<6} "
                   f"{r['n']:>5} {r['avg_r']:>+7.3f} {r['wr']:>6.1%} "
                   f"{r['sharpe']:>7.2f} {r['p_value']:>8.4f}")
    else:
        report(f"\n  NO BH FDR survivors at q=0.05")

    # -- Kruskal-Wallis results --------------------------------------------
    report(f"\n{'-' * 78}")
    report(f"KRUSKAL-WALLIS: Do regimes differ for the same parameter set?")
    report(f"{'-' * 78}")

    kw_valid = [r for r in kw_records if not np.isnan(r["kw_p"])]
    kw_p_vals = [r["kw_p"] for r in kw_valid]
    kw_rejected = bh_fdr(kw_p_vals, q=0.05)

    report(f"  Valid KW tests: {len(kw_valid)}")
    report(f"  BH FDR survivors (q=0.05): {len(kw_rejected)}")

    if kw_rejected:
        report(f"\n  {'Combo':<45} {'KW_p':>8} {'LOW':>10} {'MID':>10} {'HIGH':>10}")
        report(f"  {'-' * 90}")
        kw_survivors = [(i, kw_valid[i]) for i in sorted(kw_rejected)]
        kw_survivors.sort(key=lambda x: x[1]["kw_p"])
        for _, r in kw_survivors:
            low_s = f"{r['low_avg']:+.3f}({r['low_n']})" if not np.isnan(r['low_avg']) else "N/A"
            mid_s = f"{r['mid_avg']:+.3f}({r['mid_n']})" if not np.isnan(r['mid_avg']) else "N/A"
            high_s = f"{r['high_avg']:+.3f}({r['high_n']})" if not np.isnan(r['high_avg']) else "N/A"
            report(f"  {r['combo_label']:<45} {r['kw_p']:>8.4f} "
                   f"{low_s:>10} {mid_s:>10} {high_s:>10}")
    else:
        report(f"\n  NO regime differences survive BH FDR at q=0.05")
        report(f"  Regimes do NOT significantly differ for any tested parameter set.")

    # -- Best parameter per regime (do they differ?) -----------------------
    report(f"\n{'-' * 78}")
    report(f"ADAPTIVE vs STATIC: Does regime-switching improve performance?")
    report(f"{'-' * 78}")

    for ar in adaptive_records:
        report(f"\n  {ar['instrument']} {ar['session']} {ar['entry_model']}:")
        report(f"    Best static: {ar['best_static']} (Sharpe={ar['static_sharpe']:.2f})")
        report(f"    Parameters differ across regimes: {ar['params_differ']}")
        if ar['params_differ']:
            for regime, params in sorted(ar['regime_params'].items()):
                report(f"      {regime}: {params}")
        report(f"    Adaptive avgR: {ar['adaptive_avg_r']:+.4f}")
        report(f"    Static avgR:   {ar['static_avg_r']:+.4f}")
        report(f"    Delta:         {ar['delta_avg_r']:+.4f}")
        report(f"    Adaptive Sharpe: {ar['adaptive_sharpe']:.2f}")
        report(f"    Static Sharpe:   {ar['static_sharpe_daily']:.2f}")
        report(f"    Paired t-test: t={ar['paired_t']:.2f}, p={ar['paired_p']:.4f} "
               f"(N_days={ar['n_days']})")

    # BH FDR on adaptive vs static paired tests
    adapt_valid = [(i, r) for i, r in enumerate(adaptive_records) if not np.isnan(r["paired_p"])]
    adapt_p_vals = [r["paired_p"] for _, r in adapt_valid]
    adapt_rejected = bh_fdr(adapt_p_vals, q=0.05)

    report(f"\n  Adaptive vs Static BH FDR (q=0.05): {len(adapt_rejected)} survivors "
           f"out of {len(adapt_valid)} tests")
    if adapt_rejected:
        for local_i in sorted(adapt_rejected):
            _, r = adapt_valid[local_i]
            report(f"    {r['instrument']} {r['session']} {r['entry_model']}: "
                   f"delta={r['delta_avg_r']:+.4f}, p={r['paired_p']:.4f}")
    else:
        report(f"  Regime-adaptive switching does NOT significantly improve over best static params.")

    # -- Year-by-year stability for per-regime BH survivors ----------------
    if rejected:
        report(f"\n{'-' * 78}")
        report(f"YEAR-BY-YEAR STABILITY: BH survivors")
        report(f"{'-' * 78}")

        con = duckdb.connect(DB_PATH, read_only=True)
        for local_i in sorted(rejected):
            _, r = valid_records[local_i]
            inst = r["instrument"]
            session = r["session"]
            em = r["entry_model"]
            rr = r["rr"]
            filt_name = r["filter"]
            regime = r["regime"]
            filt_min = FILTER_THRESHOLDS[filt_name]

            df_raw = load_data(con, inst, session)
            if df_raw.empty:
                continue
            vr_for_inst = compute_vol_regime_for_instrument(con, inst)
            df_raw = merge_vol_regime(df_raw, vr_for_inst)
            df_clean = df_raw.dropna(subset=["vol_regime"])

            mask = (
                (df_clean["entry_model"] == em)
                & (df_clean["rr_target"] == rr)
                & (df_clean["orb_size"] >= filt_min)
                & (df_clean["vol_regime"] == regime)
            )
            sub = df_clean[mask]

            report(f"\n  {r['combo_label']} / {regime}:")
            years = sorted(sub["year"].dropna().unique())
            pos_years = 0
            total_years = 0
            for yr in years:
                yr_sub = sub[sub["year"] == yr]["pnl_r"].values
                yr_sub = yr_sub[~np.isnan(yr_sub)]
                if len(yr_sub) < 3:
                    continue
                total_years += 1
                avg = np.mean(yr_sub)
                if avg > 0:
                    pos_years += 1
                marker = "+" if avg > 0 else "-"
                report(f"    {int(yr)}: N={len(yr_sub):>3}, avgR={avg:>+7.3f} [{marker}]")
            pct = pos_years / total_years if total_years else 0
            report(f"    Positive years: {pos_years}/{total_years} ({pct:.0%})")

        con.close()

    # -- Honest summary ----------------------------------------------------
    report(f"\n{'=' * 78}")
    report(f"HONEST SUMMARY")
    report(f"{'=' * 78}")

    n_regime_ttest_survivors = len(rejected)
    n_kw_survivors = len(kw_rejected)
    n_adapt_survivors = len(adapt_rejected)

    if n_regime_ttest_survivors == 0 and n_kw_survivors == 0 and n_adapt_survivors == 0:
        report(f"\nVERDICT: NO-GO")
        report(f"  Volatility regime-dependent parameter switching does NOT produce")
        report(f"  statistically significant improvements over static parameters.")
        report(f"  The existing binary ATR contraction AVOID signal is sufficient.")
        report(f"")
        report(f"  Key findings:")
        report(f"  - Per-regime t-tests: 0/{len(valid_records)} survive BH FDR at q=0.05")
        report(f"  - Kruskal-Wallis: 0/{len(kw_valid)} regime differences survive BH FDR")
        report(f"  - Adaptive vs static: 0/{len(adapt_valid)} paired tests survive BH FDR")
    elif n_adapt_survivors == 0:
        report(f"\nVERDICT: STATISTICAL OBSERVATION (not actionable)")
        report(f"  Some per-regime differences exist but adaptive switching does NOT")
        report(f"  produce significant improvement over static best parameters.")
        report(f"  The existing binary ATR contraction AVOID signal remains sufficient.")
        report(f"")
        report(f"  Per-regime BH survivors: {n_regime_ttest_survivors}")
        report(f"  KW regime differences: {n_kw_survivors}")
        report(f"  Adaptive improvement: 0 (NOT significant)")
    else:
        report(f"\nVERDICT: PROMISING HYPOTHESIS (requires OOS validation)")
        report(f"  Regime-adaptive parameter switching shows significant improvement.")
        report(f"  HOWEVER: this is in-sample only. OOS validation required before live use.")
        report(f"")
        report(f"  Per-regime BH survivors: {n_regime_ttest_survivors}")
        report(f"  KW regime differences: {n_kw_survivors}")
        report(f"  Adaptive improvement: {n_adapt_survivors} significant")

    report(f"\nMANDATORY DISCLOSURES:")
    report(f"  Instruments: {ACTIVE_ORB_INSTRUMENTS}")
    report(f"  Parameter grid: RR={RR_TARGETS}, Filters={list(FILTER_THRESHOLDS.keys())}, "
           f"Entry={ENTRY_MODELS}, CB={CB}")
    report(f"  Total effective tests: {len(valid_records)} (per-regime) + {len(kw_valid)} (KW) "
           f"+ {len(adapt_valid)} (adaptive)")
    report(f"  BH FDR: q=0.05")
    report(f"  Vol regime: expanding-window percentile rank on atr_20, shift(1), 60-day warmup")
    report(f"  IS/OOS: In-sample only -- NO holdout OOS performed")
    report(f"  What kills it: MGC regime shift (gold tripled), MNQ only ~2 years, "
           f"regime proxy for ORB size")

    # -- Save outputs ------------------------------------------------------
    # CSV of all per-regime results
    results_df = pd.DataFrame(all_records)
    csv_path = OUTPUT_DIR / "vol_regime_switching_results.csv"
    results_df.to_csv(csv_path, index=False)
    report(f"\nResults CSV: {csv_path}")

    # Summary markdown
    md_path = OUTPUT_DIR / "vol_regime_switching_summary.md"
    with open(md_path, "w") as f:
        f.write("# Volatility Regime-Dependent Parameter Switching\n\n")
        f.write(f"**Date:** 2026-03-01\n")
        f.write(f"**Script:** `research/research_vol_regime_switching.py`\n\n")
        for line in report_lines:
            f.write(line + "\n")
    report(f"Summary MD: {md_path}")
    report(f"\nDone.")


if __name__ == "__main__":
    run()
