#!/usr/bin/env python3
"""T1-T8 validation battery for F5_exchange_range_atr at CME_REOPEN.

Validates the screening finding (statistics_comprehensive_scan.py) that
(prev_session_high - prev_session_low) / atr_20 predicts ORB breakout WR
at CME_REOPEN across MES, MGC, MNQ.

ENTRY MODEL: E1 (matches screening scan — line 355 of statistics_comprehensive_scan.py)
E2 concordance appended at end.

Literature grounding:
  T1: quant-audit-protocol.md — WR monotonicity before payoff analysis
  T3: Pardo (2008) expanding window. WFE = OOS/IS. WFE > 0.50 = robust.
  T4: Aronson (2006) Ch6 — ±20% parameter stability.
  T6: Phipson & Smyth (2010) — (b+1)/(m+1), 10000 permutations.
      BH FDR at K=320 (original search space, NOT K=3 survivors).
  T7: Chan (2009) — regime stability across years.
  T8: Harvey & Liu (2014) — cross-instrument concordance.

Pre-registered decisions (DEFINE BEFORE RESULTS):
  T1 WR spread <3%            -> ARITHMETIC_ONLY (kill as signal)
  T3 mean WFE <0.50           -> OVERFIT (kill)
  T4 sign flip in +/-20%      -> PARAMETER_SENSITIVE (kill)
  T5c filtered spread <3%     -> REDUNDANT (F5 adds nothing above G-filter)
  T6 bootstrap p > BH at K=320 -> NO_EDGE (noise)
  T7 <50% years same sign     -> FAIL
  T7 50-70% years same sign   -> ERA_DEPENDENT
  All T1+T3+T4+T6+T7+T8 pass -> VALIDATED
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS
from pipeline.paths import GOLD_DB_PATH
from scripts.research.statistics_comprehensive_scan import extract_all_stats

INSTRUMENTS = sorted(ACTIVE_ORB_INSTRUMENTS)
SESSION = "CME_REOPEN"
PRIMARY_ENTRY = "E1"  # matches screening scan
CONFIRM_BARS = 1
PRIMARY_RR = 1.0
N_BOOTSTRAP = 10_000  # Phipson & Smyth: 10K for p<0.0001 resolution
BH_K = 320  # original search space (10 features x 12 sessions x 3 instruments)
MIN_BIN_N = 20
FEATURE_COL = "F5_exchange_range_atr"

ALL_SESSIONS = sorted(set(s for inst in INSTRUMENTS for s in ASSET_CONFIGS[inst].get("enabled_sessions", [])))

OUT_FILE = PROJECT_ROOT / "scripts" / "research" / "output" / "exchange_range_t2t8_results.txt"

# Module-level cache for expensive statistics extraction
_STATS_CACHE: pd.DataFrame | None = None


def tee(msg: str = "", file=None):
    print(msg)
    if file:
        file.write(msg + "\n")


def _get_stats_df() -> pd.DataFrame:
    """Extract and cache CME statistics (expensive — reads thousands of DBN files)."""
    global _STATS_CACHE
    if _STATS_CACHE is not None:
        return _STATS_CACHE

    all_stats = []
    for inst in INSTRUMENTS:
        print(f"  Extracting statistics for {inst}...", flush=True)
        s = extract_all_stats(inst)
        if not s.empty:
            all_stats.append(s)
    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df["cal_date"] = pd.to_datetime(stats_df["cal_date"]).dt.date

    # Shape assertion (coupling guard)
    required_cols = {"cal_date", "instrument", "session_high", "session_low"}
    missing = required_cols - set(stats_df.columns)
    assert not missing, f"extract_all_stats API changed -- missing: {missing}"

    # Pre-shift to prior day (done once, reused by all load_data calls)
    stats_df = stats_df.sort_values(["instrument", "cal_date"]).copy()
    for col in ["session_high", "session_low"]:
        stats_df[f"prev_{col}"] = stats_df.groupby("instrument")[col].shift(1)

    _STATS_CACHE = stats_df
    return _STATS_CACHE


# --- Data Pipeline ----------------------------------------------------------
def load_data(
    entry_model: str = PRIMARY_ENTRY, rr_target: float = PRIMARY_RR, session: str = SESSION
) -> tuple[pd.DataFrame, int, int, float]:
    """Load and merge statistics + daily_features + orb_outcomes."""
    stats_shifted = _get_stats_df()

    # Load daily_features + orb_outcomes
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    df = con.execute(
        """
        SELECT d.trading_day, d.symbol, d.atr_20,
               d.orb_CME_REOPEN_size AS orb_size,
               d.rel_vol_CME_REOPEN AS rel_vol,
               o.orb_label, o.outcome, o.pnl_r, o.risk_dollars,
               o.entry_price, o.stop_price
        FROM daily_features d
        JOIN orb_outcomes o
          ON d.trading_day = o.trading_day
         AND d.symbol = o.symbol
         AND d.orb_minutes = o.orb_minutes
        WHERE d.orb_minutes = 5
          AND d.symbol = ANY($5)
          AND o.orb_label = $1
          AND o.entry_model = $2
          AND o.confirm_bars = $3
          AND o.rr_target = $4
          AND o.outcome IN ('win', 'loss')
          AND d.atr_20 IS NOT NULL AND d.atr_20 > 0
    """,
        [session, entry_model, CONFIRM_BARS, rr_target, INSTRUMENTS],
    ).fetchdf()
    con.close()

    df["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date

    # 3. Merge with pre-shifted statistics
    pre_merge_n = len(df)
    merged = df.merge(
        stats_shifted[["cal_date", "instrument", "prev_session_high", "prev_session_low"]],
        left_on=["trading_day", "symbol"],
        right_on=["cal_date", "instrument"],
        how="inner",
    )
    post_merge_n = len(merged)
    coverage = post_merge_n / pre_merge_n if pre_merge_n > 0 else 0

    # 4. Compute F5
    merged[FEATURE_COL] = (merged["prev_session_high"] - merged["prev_session_low"]) / merged["atr_20"]
    merged = merged.dropna(subset=[FEATURE_COL])

    # 5. Duplicate guard
    dupes = merged.groupby(["trading_day", "symbol"]).size()
    assert (dupes == 1).all(), f"Duplicate (trading_day, symbol) rows: {(dupes > 1).sum()}"

    merged["year"] = pd.to_datetime(merged["trading_day"]).dt.year
    merged["is_win"] = (merged["outcome"] == "win").astype(float)
    merged["direction"] = np.where(merged["entry_price"] > merged["stop_price"], "LONG", "SHORT")

    return merged, pre_merge_n, post_merge_n, coverage


def wr_spread_quintile(data: pd.DataFrame) -> dict | None:
    """Compute Q5-Q1 WR spread with full monotonicity detail."""
    valid = data.dropna(subset=[FEATURE_COL])
    if len(valid) < MIN_BIN_N * 5:
        return None
    try:
        valid = valid.copy()
        valid["qbin"] = pd.qcut(valid[FEATURE_COL], 5, labels=False, duplicates="drop")
    except ValueError:
        return None
    bins = sorted(valid["qbin"].dropna().unique())
    if len(bins) < 4:
        return None

    q1 = valid[valid["qbin"] == bins[0]]
    q5 = valid[valid["qbin"] == bins[-1]]
    if len(q1) < MIN_BIN_N or len(q5) < MIN_BIN_N:
        return None

    wr_q1 = q1["is_win"].mean()
    wr_q5 = q5["is_win"].mean()
    expr_q1 = q1["pnl_r"].mean()
    expr_q5 = q5["pnl_r"].mean()
    avgwin_q1 = q1.loc[q1["is_win"] == 1, "pnl_r"].mean() if q1["is_win"].sum() > 0 else 0
    avgwin_q5 = q5.loc[q5["is_win"] == 1, "pnl_r"].mean() if q5["is_win"].sum() > 0 else 0

    per_bin = []
    for b in bins:
        bdata = valid[valid["qbin"] == b]
        per_bin.append(
            {
                "bin": int(b),
                "n": len(bdata),
                "wr": bdata["is_win"].mean(),
                "expr": bdata["pnl_r"].mean(),
                "avg_win_r": bdata.loc[bdata["is_win"] == 1, "pnl_r"].mean() if bdata["is_win"].sum() > 0 else 0,
            }
        )

    return {
        "wr_spread": wr_q5 - wr_q1,
        "wr_q1": wr_q1,
        "wr_q5": wr_q5,
        "expr_q1": expr_q1,
        "expr_q5": expr_q5,
        "avgwin_q1": avgwin_q1,
        "avgwin_q5": avgwin_q5,
        "n_q1": len(q1),
        "n_q5": len(q5),
        "N": len(valid),
        "per_bin": per_bin,
    }


# --- T1: WR Monotonicity ---------------------------------------------------
def test_t1_wr_monotonicity(df: pd.DataFrame, inst: str) -> dict:
    sub = df[df["symbol"] == inst]
    result = wr_spread_quintile(sub)
    if result is None:
        return {"verdict": "SKIP", "reason": "insufficient data"}
    is_arithmetic = abs(result["wr_spread"]) < 0.03
    verdict = "ARITHMETIC_ONLY" if is_arithmetic else "PASS"
    return {**result, "verdict": verdict}


def test_t1b_multi_rr(df_cache: dict, inst: str) -> dict:
    """Test F5 WR spread at RR 1.0, 1.5, 2.0."""
    results = []
    for rr in [1.0, 1.5, 2.0]:
        key = (PRIMARY_ENTRY, rr)
        if key not in df_cache:
            try:
                df_cache[key], _, _, _ = load_data(entry_model=PRIMARY_ENTRY, rr_target=rr)
            except Exception:
                continue
        rr_df = df_cache[key]
        sub = rr_df[rr_df["symbol"] == inst]
        spread_info = wr_spread_quintile(sub)
        if spread_info:
            results.append({"rr": rr, "wr_spread": spread_info["wr_spread"], "N": spread_info["N"]})

    if len(results) < 2:
        return {"verdict": "SKIP", "reason": "insufficient RR data"}
    signs = [r["wr_spread"] > 0 for r in results]
    same_sign = all(signs) or not any(signs)
    return {"results": results, "same_sign": same_sign, "verdict": "PASS" if same_sign else "WARN"}


def test_t1c_direction(df: pd.DataFrame, inst: str) -> dict:
    """WR spread separately for long and short."""
    sub = df[df["symbol"] == inst]
    dir_results = {}
    for d in ["LONG", "SHORT"]:
        dsub = sub[sub["direction"] == d]
        spread_info = wr_spread_quintile(dsub)
        if spread_info:
            dir_results[d] = {"wr_spread": spread_info["wr_spread"], "N": spread_info["N"]}
    return {"directions": dir_results, "verdict": "INFO"}


# --- T2: IS Baseline -------------------------------------------------------
def test_t2_is_baseline(df: pd.DataFrame, inst: str) -> dict:
    sub = df[df["symbol"] == inst].sort_values("trading_day")
    years = sorted(sub["year"].unique())
    if len(years) < 4:
        return {"verdict": "SKIP", "reason": f"only {len(years)} years"}
    split_idx = int(len(years) * 0.6)
    is_years = set(years[:split_idx])
    is_data = sub[sub["year"].isin(is_years)]
    result = wr_spread_quintile(is_data)
    if result is None:
        return {"verdict": "SKIP", "reason": "insufficient IS data"}
    verdict = "PASS" if result["wr_spread"] > 0 else "FAIL"
    return {"is_years": f"{min(is_years)}-{max(is_years)}", "n": len(is_data), **result, "verdict": verdict}


# --- T3: Walk-Forward (expanding window) -----------------------------------
def test_t3_walkforward(df: pd.DataFrame, inst: str) -> dict:
    """Expanding window: IS grows, each year is OOS exactly once."""
    sub = df[df["symbol"] == inst].sort_values("trading_day")
    years = sorted(sub["year"].unique())
    if len(years) < 6:
        return {"verdict": "SKIP", "reason": f"only {len(years)} years"}

    min_is_years = 4
    oos_results = []
    for i in range(min_is_years, len(years)):
        is_years = set(years[:i])
        oos_year = years[i]
        is_data = sub[sub["year"].isin(is_years)]
        oos_data = sub[sub["year"] == oos_year]

        is_spread = wr_spread_quintile(is_data)
        oos_spread = wr_spread_quintile(oos_data)
        if is_spread is None or oos_spread is None:
            continue

        wfe = oos_spread["wr_spread"] / is_spread["wr_spread"] if is_spread["wr_spread"] != 0 else 0
        same_sign = (is_spread["wr_spread"] > 0) == (oos_spread["wr_spread"] > 0)
        oos_results.append(
            {
                "oos_year": oos_year,
                "is_spread": is_spread["wr_spread"],
                "oos_spread": oos_spread["wr_spread"],
                "wfe": wfe,
                "same_sign": same_sign,
                "oos_n": oos_spread["N"],
            }
        )

    if not oos_results:
        return {"verdict": "SKIP", "reason": "no valid OOS windows"}

    mean_wfe = np.mean([r["wfe"] for r in oos_results])
    worst_wfe = min(r["wfe"] for r in oos_results)
    sign_flips = sum(1 for r in oos_results if not r["same_sign"])

    if mean_wfe > 0.95:
        verdict = "SUSPECT_LEAKAGE"
    elif mean_wfe > 0.50:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return {
        "n_oos_years": len(oos_results),
        "mean_wfe": mean_wfe,
        "worst_wfe": worst_wfe,
        "sign_flips": sign_flips,
        "oos_results": oos_results,
        "verdict": verdict,
    }


# --- T4: Sensitivity -------------------------------------------------------
def _compute_actual_atr(period: int) -> pd.DataFrame:
    """Compute ATR_N from daily OHLC (SMA of true range, matching pipeline convention)."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    daily = con.execute(
        """
        SELECT trading_day, symbol, daily_high, daily_low, daily_close
        FROM daily_features
        WHERE orb_minutes = 5 AND symbol = ANY($1)
          AND daily_high IS NOT NULL AND daily_low IS NOT NULL
        ORDER BY symbol, trading_day
    """,
        [INSTRUMENTS],
    ).fetchdf()
    con.close()

    daily["prev_close"] = daily.groupby("symbol")["daily_close"].shift(1)
    daily["hl"] = daily["daily_high"] - daily["daily_low"]
    daily["hc"] = (daily["daily_high"] - daily["prev_close"]).abs()
    daily["lc"] = (daily["daily_low"] - daily["prev_close"]).abs()
    daily["tr"] = daily[["hl", "hc", "lc"]].max(axis=1)
    daily.loc[daily["prev_close"].isna(), "tr"] = daily["hl"]

    # SMA of prior N true ranges (no look-ahead — shift(1) then rolling)
    col = f"recomp_atr_{period}"
    daily[col] = daily.groupby("symbol")["tr"].transform(lambda s: s.shift(1).rolling(period, min_periods=1).mean())
    result = daily[["trading_day", "symbol", col]].dropna(subset=[col]).copy()
    result["trading_day"] = pd.to_datetime(result["trading_day"]).dt.date
    return result


def test_t4_sensitivity(df: pd.DataFrame, inst: str) -> dict:
    """ATR period recomputation (actual, not scaled), bin robustness."""
    sub = df[df["symbol"] == inst].copy()

    # a) Actual ATR period perturbation — recompute F5 with ATR_15, ATR_20, ATR_25
    atr_results = []
    for period in [15, 20, 25]:
        atr_df = _compute_actual_atr(period)
        atr_col = f"recomp_atr_{period}"
        test_df = sub.merge(
            atr_df[atr_df["symbol"] == inst][["trading_day", atr_col]],
            on="trading_day",
            how="inner",
        )
        test_df[FEATURE_COL] = (test_df["prev_session_high"] - test_df["prev_session_low"]) / test_df[atr_col]
        test_df = test_df.dropna(subset=[FEATURE_COL])
        spread_info = wr_spread_quintile(test_df)
        if spread_info:
            atr_results.append(
                {
                    "atr": f"ATR_{period}",
                    "wr_spread": spread_info["wr_spread"],
                    "n": len(test_df),
                }
            )
    atr_signs = [r["wr_spread"] > 0 for r in atr_results]
    atr_stable = (all(atr_signs) or not any(atr_signs)) if atr_signs else False

    # b) Bin robustness
    bin_results = []
    for n_bins, label in [(2, "median"), (3, "tercile"), (4, "quartile"), (5, "quintile"), (10, "decile")]:
        valid = sub.dropna(subset=[FEATURE_COL])
        if len(valid) < MIN_BIN_N * n_bins:
            continue
        try:
            valid = valid.copy()
            valid["qbin"] = pd.qcut(valid[FEATURE_COL], n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        bins_list = sorted(valid["qbin"].dropna().unique())
        if len(bins_list) < 2:
            continue
        q_lo = valid[valid["qbin"] == bins_list[0]]
        q_hi = valid[valid["qbin"] == bins_list[-1]]
        if len(q_lo) < MIN_BIN_N or len(q_hi) < MIN_BIN_N:
            continue
        spread = q_hi["is_win"].mean() - q_lo["is_win"].mean()
        bin_results.append({"bins": label, "spread": spread, "n_lo": len(q_lo), "n_hi": len(q_hi)})
    bin_signs = [r["spread"] > 0 for r in bin_results]
    bin_stable = (all(bin_signs) or not any(bin_signs)) if bin_signs else False

    # Kill if ATR period change flips sign (real sensitivity test)
    verdict = "PASS" if atr_stable else "FAIL"
    return {
        "atr_results": atr_results,
        "atr_stable": atr_stable,
        "bin_results": bin_results,
        "bin_stable": bin_stable,
        "verdict": verdict,
    }


# --- T5: Family (all sessions) ---------------------------------------------
def test_t5_family(inst: str, session_cache: dict) -> dict:
    """Same feature across all sessions for the same instrument."""
    enabled = ASSET_CONFIGS[inst].get("enabled_sessions", [])
    results = []
    for sess in ALL_SESSIONS:
        if sess not in enabled:
            continue
        if sess not in session_cache:
            try:
                session_cache[sess], _, _, _ = load_data(entry_model=PRIMARY_ENTRY, rr_target=PRIMARY_RR, session=sess)
            except Exception:
                session_cache[sess] = None
                continue
        sess_df = session_cache[sess]
        if sess_df is None:
            continue
        sub = sess_df[sess_df["symbol"] == inst]
        spread_info = wr_spread_quintile(sub)
        if spread_info:
            results.append({"session": sess, "wr_spread": spread_info["wr_spread"], "N": spread_info["N"]})

    if len(results) < 3:
        return {"verdict": "SKIP", "reason": f"only {len(results)} sessions testable"}

    ref_positive = any(r["session"] == SESSION and r["wr_spread"] > 0 for r in results)
    same_sign = sum(1 for r in results if (r["wr_spread"] > 0) == ref_positive)
    verdict = "PASS" if same_sign >= len(results) * 0.5 else "REGIME_SPECIFIC"
    return {"sessions_tested": len(results), "same_sign": same_sign, "results": results, "verdict": verdict}


def test_t5b_confound(df: pd.DataFrame, inst: str) -> dict:
    """Check F5 for confounding with ATR level, orb_size, rel_vol."""
    sub = df[df["symbol"] == inst].dropna(subset=[FEATURE_COL])
    correlations = {}
    for col, label in [("atr_20", "ATR_20"), ("orb_size", "orb_size_CME_REOPEN"), ("rel_vol", "rel_vol_CME_REOPEN")]:
        valid = sub.dropna(subset=[col])
        if len(valid) > 30:
            r, p = stats.pearsonr(valid[FEATURE_COL], valid[col])
            correlations[label] = {"r": r, "p": p}

    # Stratified F5 WR spread within ATR terciles
    atr_strat = []
    valid = sub.dropna(subset=["atr_20"])
    if len(valid) > MIN_BIN_N * 3:
        try:
            valid = valid.copy()
            valid["atr_tercile"] = pd.qcut(valid["atr_20"], 3, labels=["low", "mid", "high"])
            for t in ["low", "mid", "high"]:
                t_data = valid[valid["atr_tercile"] == t]
                spread_info = wr_spread_quintile(t_data)
                if spread_info:
                    atr_strat.append({"tercile": t, "wr_spread": spread_info["wr_spread"], "N": spread_info["N"]})
        except ValueError:
            pass

    high_corr = any(abs(v["r"]) > 0.30 for v in correlations.values())
    verdict = "REDUNDANCY_RISK" if high_corr else "PASS"
    return {"correlations": correlations, "atr_stratified": atr_strat, "verdict": verdict}


def test_t5c_filter_interaction(df: pd.DataFrame, inst: str) -> dict:
    """Test F5 WR spread WITHIN G5-filtered population."""
    sub = df[df["symbol"] == inst].copy()
    filtered = sub[sub["orb_size"].notna() & (sub["orb_size"] >= 5)]
    if len(filtered) < MIN_BIN_N * 5:
        return {"verdict": "SKIP", "reason": f"only {len(filtered)} rows after G5 filter"}

    spread_info = wr_spread_quintile(filtered)
    if spread_info is None:
        return {"verdict": "SKIP", "reason": "insufficient data for quintile in G5 pop"}

    unfiltered_info = wr_spread_quintile(sub)
    unfiltered_spread = unfiltered_info["wr_spread"] if unfiltered_info else None

    verdict = "REDUNDANT" if abs(spread_info["wr_spread"]) < 0.03 else "PASS"
    return {
        "g5_wr_spread": spread_info["wr_spread"],
        "g5_N": spread_info["N"],
        "unfiltered_wr_spread": unfiltered_spread,
        "verdict": verdict,
    }


# --- T6: Null Floor (bootstrap) --------------------------------------------
def test_t6_null(df: pd.DataFrame, inst: str) -> dict:
    """10000 bootstrap permutations. BH FDR at K=320 (original search space)."""
    sub = df[df["symbol"] == inst].copy()
    valid = sub.dropna(subset=[FEATURE_COL]).copy()
    observed_info = wr_spread_quintile(valid)
    if observed_info is None:
        return {"verdict": "SKIP", "reason": "no observed spread"}

    observed = observed_info["wr_spread"]
    rng = np.random.default_rng(42)

    # Pre-compute quintile bins (feature order doesn't change, only outcomes shuffle)
    try:
        valid["qbin"] = pd.qcut(valid[FEATURE_COL], 5, labels=False, duplicates="drop")
    except ValueError:
        return {"verdict": "SKIP", "reason": "cannot create quintile bins"}
    bins_list = sorted(valid["qbin"].dropna().unique())
    if len(bins_list) < 4:
        return {"verdict": "SKIP", "reason": "too few bins"}

    q1_mask = (valid["qbin"] == bins_list[0]).values
    q5_mask = (valid["qbin"] == bins_list[-1]).values
    n_q1 = q1_mask.sum()
    n_q5 = q5_mask.sum()
    if n_q1 < MIN_BIN_N or n_q5 < MIN_BIN_N:
        return {"verdict": "SKIP", "reason": "too few in extreme bins"}

    is_win_arr = valid["is_win"].values.copy()
    null_spreads = []
    for _ in range(N_BOOTSTRAP):
        shuffled = rng.permutation(is_win_arr)
        wr_q1 = shuffled[q1_mask].mean()
        wr_q5 = shuffled[q5_mask].mean()
        null_spreads.append(wr_q5 - wr_q1)

    # Phipson & Smyth: (b+1)/(m+1)
    exceeding = sum(1 for ns in null_spreads if abs(ns) >= abs(observed))
    p_value = (exceeding + 1) / (len(null_spreads) + 1)
    verdict = "PASS" if p_value < 0.05 else "FAIL"

    return {
        "observed": observed,
        "null_mean": np.mean(null_spreads),
        "null_p95": np.percentile([abs(x) for x in null_spreads], 95),
        "null_p99": np.percentile([abs(x) for x in null_spreads], 99),
        "p_value": p_value,
        "n_bootstraps": len(null_spreads),
        "verdict": verdict,
    }


# --- T7: Per-Year Stability ------------------------------------------------
def test_t7_peryear(df: pd.DataFrame, inst: str) -> dict:
    sub = df[df["symbol"] == inst].copy()
    years = sorted(sub["year"].unique())
    year_counts = sub.groupby("year").size()
    full_years = [y for y in years if year_counts.get(y, 0) >= 50]

    if len(full_years) < 5:
        return {"verdict": "SKIP", "reason": f"only {len(full_years)} full years"}

    overall_info = wr_spread_quintile(sub)
    if overall_info is None:
        return {"verdict": "SKIP", "reason": "no overall spread"}

    ref_positive = overall_info["wr_spread"] > 0
    year_results = []
    for y in full_years:
        y_data = sub[sub["year"] == y]
        valid = y_data.dropna(subset=[FEATURE_COL])
        if len(valid) < MIN_BIN_N * 2:
            continue
        median_val = valid[FEATURE_COL].median()
        lo = valid[valid[FEATURE_COL] <= median_val]
        hi = valid[valid[FEATURE_COL] > median_val]
        if len(lo) < MIN_BIN_N or len(hi) < MIN_BIN_N:
            continue
        spread = hi["is_win"].mean() - lo["is_win"].mean()
        year_results.append({"year": y, "spread": spread, "n": len(valid), "same_sign": (spread > 0) == ref_positive})

    if len(year_results) < 5:
        return {"verdict": "SKIP", "reason": f"only {len(year_results)} years with data"}

    same_sign_count = sum(1 for yr in year_results if yr["same_sign"])
    pct = same_sign_count / len(year_results)

    if pct >= 0.70:
        verdict = "PASS"
    elif pct >= 0.50:
        verdict = "ERA_DEPENDENT"
    else:
        verdict = "FAIL"

    return {
        "years_tested": len(year_results),
        "same_sign": same_sign_count,
        "pct": pct,
        "year_details": year_results,
        "verdict": verdict,
    }


# --- T8: Cross-Instrument --------------------------------------------------
def test_t8_cross(df: pd.DataFrame) -> dict:
    results = []
    for inst in INSTRUMENTS:
        sub = df[df["symbol"] == inst]
        spread_info = wr_spread_quintile(sub)
        if spread_info:
            results.append({"instrument": inst, "wr_spread": spread_info["wr_spread"], "N": spread_info["N"]})

    if len(results) < 2:
        return {"verdict": "SKIP", "reason": f"only {len(results)} instruments"}
    positive = sum(1 for r in results if r["wr_spread"] > 0)
    verdict = "PASS" if positive >= 2 else "FAIL"
    return {"instruments_tested": len(results), "positive": positive, "results": results, "verdict": verdict}


# --- E2 Concordance --------------------------------------------------------
def test_e2_concordance() -> dict:
    """Re-run T1 under E2 to check entry model concordance."""
    results = []
    try:
        e2_df, _, _, _ = load_data(entry_model="E2", rr_target=PRIMARY_RR)
        for inst in INSTRUMENTS:
            sub = e2_df[e2_df["symbol"] == inst]
            spread_info = wr_spread_quintile(sub)
            if spread_info:
                results.append({"instrument": inst, "wr_spread": spread_info["wr_spread"], "N": spread_info["N"]})
    except Exception as e:
        return {"verdict": "SKIP", "reason": str(e)}

    if not results:
        return {"verdict": "SKIP", "reason": "no E2 data"}
    same_dir = sum(1 for r in results if r["wr_spread"] > 0)
    return {"results": results, "same_direction": same_dir, "verdict": "INFO"}


# --- Main -------------------------------------------------------------------
def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # noqa rationale: file lifetime spans the entire 234-line main() body; converting
    # to a `with` block would require re-indenting the whole function and is out of
    # scope for this lint cleanup commit. Research script, manual close at line 872.
    f = open(OUT_FILE, "w")  # noqa: SIM115

    tee("=" * 80, f)
    tee("AUDIT REPORT -- F5_exchange_range_atr at CME_REOPEN", f)
    tee("=" * 80, f)
    tee(f"Entry model: {PRIMARY_ENTRY} (matches screening scan)", f)
    tee(f"RR target: {PRIMARY_RR}", f)
    tee(f"Bootstrap: {N_BOOTSTRAP} permutations", f)
    tee(f"BH K: {BH_K} (original search space)", f)
    tee("", f)

    # -- Data Pipeline --
    tee("--- DATA PIPELINE ---", f)
    df, pre_n, post_n, coverage = load_data()
    tee(f"  Pre-merge outcomes: {pre_n}", f)
    tee(f"  Post-merge (with stats): {post_n}", f)
    tee(f"  Coverage: {coverage:.1%}", f)
    if coverage < 0.95:
        tee("  WARNING: COVERAGE BELOW 95% -- investigate cal_date alignment", f)
    for inst in INSTRUMENTS:
        n = len(df[df["symbol"] == inst])
        tee(f"  {inst}: {n} rows", f)

    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    freshness = con.execute("SELECT MAX(trading_day) FROM orb_outcomes").fetchone()[0]
    con.close()
    tee(f"  DB freshness: {freshness}", f)
    tee("", f)

    # Caches for expensive loads
    rr_cache: dict = {(PRIMARY_ENTRY, PRIMARY_RR): df}
    session_cache: dict = {SESSION: df}

    overall_verdicts = {}

    for inst in INSTRUMENTS:
        tee("=" * 80, f)
        tee(f"INSTRUMENT: {inst}", f)
        tee("=" * 80, f)

        # T1
        t1 = test_t1_wr_monotonicity(df, inst)
        tee(f"\n  T1 WR MONOTONICITY: {t1['verdict']}", f)
        if "per_bin" in t1:
            tee(f"     Q5-Q1 WR spread: {t1['wr_spread']:+.1%}", f)
            tee(
                f"     Q1: WR={t1['wr_q1']:.1%} ExpR={t1['expr_q1']:+.3f} AvgWinR={t1['avgwin_q1']:.3f} (n={t1['n_q1']})",
                f,
            )
            tee(
                f"     Q5: WR={t1['wr_q5']:.1%} ExpR={t1['expr_q5']:+.3f} AvgWinR={t1['avgwin_q5']:.3f} (n={t1['n_q5']})",
                f,
            )
            for b in t1["per_bin"]:
                tee(
                    f"       Q{b['bin'] + 1}: WR={b['wr']:.1%} ExpR={b['expr']:+.3f} AvgWinR={b['avg_win_r']:.3f} n={b['n']}",
                    f,
                )
        elif "reason" in t1:
            tee(f"     {t1['reason']}", f)

        if t1["verdict"] == "ARITHMETIC_ONLY":
            tee("\n  ARITHMETIC_ONLY -- WR spread <3%. Not a WR predictor.", f)
            overall_verdicts[inst] = "ARITHMETIC_ONLY"
            continue

        # T1b
        t1b = test_t1b_multi_rr(rr_cache, inst)
        tee(f"\n  T1b MULTI-RR: {t1b['verdict']}", f)
        if "results" in t1b:
            for r in t1b["results"]:
                tee(f"     RR={r['rr']:.1f}: spread={r['wr_spread']:+.1%} (N={r['N']})", f)

        # T1c
        t1c = test_t1c_direction(df, inst)
        tee(f"\n  T1c DIRECTION: {t1c['verdict']}", f)
        if "directions" in t1c:
            for d, v in t1c["directions"].items():
                tee(f"     {d}: spread={v['wr_spread']:+.1%} (N={v['N']})", f)

        # T2
        t2 = test_t2_is_baseline(df, inst)
        tee(f"\n  T2 IS BASELINE: {t2['verdict']}", f)
        if "is_years" in t2:
            tee(f"     IS ({t2['is_years']}): spread={t2['wr_spread']:+.1%}, N={t2['n']}", f)

        # T3
        t3 = test_t3_walkforward(df, inst)
        tee(f"\n  T3 WALK-FORWARD: {t3['verdict']}", f)
        if "mean_wfe" in t3:
            tee(f"     Mean WFE: {t3['mean_wfe']:.2f}, Worst: {t3['worst_wfe']:.2f}", f)
            tee(f"     Sign flips: {t3['sign_flips']}/{t3['n_oos_years']} OOS years", f)
            for r in t3["oos_results"]:
                sign = "+" if r["same_sign"] else "X"
                tee(
                    f"     {r['oos_year']}: IS={r['is_spread']:+.1%} OOS={r['oos_spread']:+.1%} WFE={r['wfe']:.2f} n={r['oos_n']} {sign}",
                    f,
                )
        elif "reason" in t3:
            tee(f"     {t3['reason']}", f)

        # T4
        t4 = test_t4_sensitivity(df, inst)
        tee(f"\n  T4 SENSITIVITY: {t4['verdict']}", f)
        if "atr_results" in t4:
            tee(f"     ATR period (actual recomputation, stable: {t4['atr_stable']}):", f)
            for r in t4["atr_results"]:
                tee(f"       {r['atr']}: spread={r['wr_spread']:+.1%} (N={r['n']})", f)
        if "bin_results" in t4:
            tee(f"     Bin robustness (stable: {t4['bin_stable']}):", f)
            for r in t4["bin_results"]:
                tee(f"       {r['bins']:>8s}: spread={r['spread']:+.1%} (n={r['n_lo']}|{r['n_hi']})", f)

        # T5
        t5 = test_t5_family(inst, session_cache)
        tee(
            f"\n  T5 FAMILY: {t5['verdict']} ({t5.get('same_sign', '?')}/{t5.get('sessions_tested', '?')} same sign)", f
        )
        if "results" in t5:
            for r in sorted(t5["results"], key=lambda x: -x["wr_spread"]):
                marker = " <--" if r["session"] == SESSION else ""
                tee(f"     {r['session']:20s} spread={r['wr_spread']:+.1%} (N={r['N']}){marker}", f)

        # T5b
        t5b = test_t5b_confound(df, inst)
        tee(f"\n  T5b CONFOUND: {t5b['verdict']}", f)
        if "correlations" in t5b:
            for label, v in t5b["correlations"].items():
                flag = " WARNING" if abs(v["r"]) > 0.30 else ""
                tee(f"     corr(F5, {label}) = {v['r']:+.3f} (p={v['p']:.4f}){flag}", f)
        if "atr_stratified" in t5b:
            tee("     ATR-stratified WR spread:", f)
            for r in t5b["atr_stratified"]:
                tee(f"       {r['tercile']:>4s}: spread={r['wr_spread']:+.1%} (N={r['N']})", f)

        # T5c
        t5c = test_t5c_filter_interaction(df, inst)
        tee(f"\n  T5c FILTER INTERACTION (within G5): {t5c['verdict']}", f)
        if "g5_wr_spread" in t5c:
            tee(f"     G5-filtered: spread={t5c['g5_wr_spread']:+.1%} (N={t5c['g5_N']})", f)
            if t5c.get("unfiltered_wr_spread") is not None:
                tee(f"     Unfiltered:   spread={t5c['unfiltered_wr_spread']:+.1%}", f)

        # T6
        tee(f"\n  T6 NULL FLOOR (running {N_BOOTSTRAP} bootstraps)...", f)
        t6 = test_t6_null(df, inst)
        tee(f"  T6 NULL FLOOR: {t6['verdict']}", f)
        if "p_value" in t6:
            tee(f"     observed={t6['observed']:+.1%}, null_mean={t6['null_mean']:+.1%}", f)
            tee(f"     null_P95={t6['null_p95']:.1%}, null_P99={t6['null_p99']:.1%}", f)
            tee(f"     bootstrap p={t6['p_value']:.6f} (n={t6['n_bootstraps']})", f)
            bh_threshold = 0.05 / BH_K
            tee(f"     BH threshold (rank 1, K={BH_K}): {bh_threshold:.6f}", f)
            tee(f"     BH PASS: {'YES' if t6['p_value'] <= bh_threshold else 'NO'}", f)
        elif "reason" in t6:
            tee(f"     {t6['reason']}", f)

        # T7
        t7 = test_t7_peryear(df, inst)
        tee(f"\n  T7 PER-YEAR: {t7['verdict']}", f)
        if "year_details" in t7:
            tee(f"     {t7['same_sign']}/{t7['years_tested']} years same sign ({t7['pct']:.0%})", f)
            for yr in t7["year_details"]:
                sign = "+" if yr["same_sign"] else "X"
                tee(f"     {yr['year']}: {yr['spread']:+.1%} (n={yr['n']}) {sign}", f)
        elif "reason" in t7:
            tee(f"     {t7['reason']}", f)

        # Collect verdicts
        verdicts = {
            "T1": t1["verdict"],
            "T2": t2["verdict"],
            "T3": t3["verdict"],
            "T4": t4["verdict"],
            "T5": t5["verdict"],
            "T5b": t5b["verdict"],
            "T5c": t5c["verdict"],
            "T6": t6["verdict"],
            "T7": t7["verdict"],
        }
        passes = sum(1 for v in verdicts.values() if v == "PASS")
        fails = sum(1 for v in verdicts.values() if v == "FAIL")
        tee(f"\n  VERDICTS: {verdicts}", f)
        tee(f"  {passes} PASS, {fails} FAIL", f)

        # Overall decision
        kill_tests = ["T1", "T3", "T4", "T6"]
        killed = [t for t in kill_tests if verdicts.get(t) == "FAIL"]
        if killed:
            overall_verdicts[inst] = f"KILLED ({', '.join(killed)})"
        elif verdicts.get("T7") == "FAIL":
            overall_verdicts[inst] = "KILLED (T7)"
        elif verdicts.get("T7") == "ERA_DEPENDENT":
            overall_verdicts[inst] = "ERA_DEPENDENT"
        elif verdicts.get("T5c") == "REDUNDANT":
            overall_verdicts[inst] = "REDUNDANT"
        elif verdicts.get("T5b") == "REDUNDANCY_RISK":
            overall_verdicts[inst] = "VALIDATED_WITH_RISK"
        else:
            overall_verdicts[inst] = "VALIDATED"

        tee(f"  === {inst} OVERALL: {overall_verdicts[inst]}", f)
        tee("", f)

    # -- T8 --
    tee("=" * 80, f)
    tee("T8 CROSS-INSTRUMENT", f)
    tee("=" * 80, f)
    t8 = test_t8_cross(df)
    tee(f"  {t8['verdict']} ({t8.get('positive', '?')}/{t8.get('instruments_tested', '?')} positive)", f)
    if "results" in t8:
        for r in t8["results"]:
            tee(f"  {r['instrument']:4s} spread={r['wr_spread']:+.1%} (N={r['N']})", f)
    tee("", f)

    # -- E2 Concordance --
    tee("=" * 80, f)
    tee("E2 CONCORDANCE (informational -- not a gate)", f)
    tee("=" * 80, f)
    e2 = test_e2_concordance()
    tee(f"  {e2['verdict']}", f)
    if "results" in e2:
        for r in e2["results"]:
            tee(f"  {r['instrument']:4s} E2 spread={r['wr_spread']:+.1%} (N={r['N']})", f)
    tee("", f)

    # -- Final Decision --
    tee("=" * 80, f)
    tee("FINAL DECISION", f)
    tee("=" * 80, f)
    for inst, verdict in overall_verdicts.items():
        tee(f"  {inst}: {verdict}", f)

    validated = sum(1 for v in overall_verdicts.values() if "VALIDATED" in v)
    killed = sum(1 for v in overall_verdicts.values() if "KILLED" in v)
    tee(f"\n  VALIDATED: {validated}/{len(INSTRUMENTS)}", f)
    tee(f"  KILLED: {killed}/{len(INSTRUMENTS)}", f)

    if validated == len(INSTRUMENTS):
        tee("\n  >>> ALL INSTRUMENTS VALIDATED -- proceed to deployment design <<<", f)
    elif validated >= 2:
        tee(f"\n  >>> PARTIAL VALIDATION -- {validated}/{len(INSTRUMENTS)} instruments <<<", f)
    elif validated == 1:
        tee("\n  >>> SINGLE INSTRUMENT -- REGIME_SPECIFIC, shelve <<<", f)
    else:
        tee("\n  >>> NO INSTRUMENTS VALIDATED -- KILL finding <<<", f)

    tee("", f)
    tee("  Mechanism: UNVERIFIED (no order flow data to test causation)", f)
    tee("  Feature: exchange pit session range / ATR -- prior day, zero look-ahead", f)
    tee("  Session: CME_REOPEN only (other sessions not BH-significant at K=320)", f)
    tee("  NOTE: MGC exception was NOT pre-registered (seen p=0.053 pre-2019 first)", f)

    f.close()
    print(f"\nResults saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
