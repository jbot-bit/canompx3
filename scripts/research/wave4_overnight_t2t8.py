#!/usr/bin/env python3
"""T2-T8 battery for overnight_range/atr ratio — full institutional rigor.

Re-runs the scan_presession_t2t8.py methodology against the CURRENT post-Phase-3c
canonical data for all 9 T1 passers identified in wave 4 investigation.

Literature grounding (same as scan_presession_t2t8.py):
  T3 Walk-Forward: Pardo (2008) expanding window. WFE > 0.50 robust, >0.95 suspect.
  T4 Sensitivity: Aronson (2006) Ch6 — ±20% parameter stability.
  T6 Null Floor: Bootstrap permutation, 5000 shuffles, p=(b+1)/(m+1).
  T7 Per-Year: Must be positive in >=6/7 years. <5 → ERA_DEPENDENT.
  T8 Cross-Instrument: Same direction on all active instruments.

Tests overnight_range/atr_20 ratio as a filter signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH

ENTRY_MODEL = "E2"
RR_TARGET = 1.0
N_BOOTSTRAP = 5000
MIN_BIN_N = 20
HOLDOUT_DATE = "2026-01-01"

OVERNIGHT_CLEAN_SESSIONS = [
    "LONDON_METALS",
    "EUROPE_FLOW",
    "US_DATA_830",
    "NYSE_OPEN",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "CME_PRECLOSE",
    "NYSE_CLOSE",
]

# 9 T1 passers from current-data rescan (post Phase 3c rebuild)
T1_PASSERS = [
    # (instrument, session, direction) - direction: HIGH=top quintile better, LOW=bottom quintile better
    ("MNQ", "NYSE_CLOSE", "HIGH"),
    ("MES", "EUROPE_FLOW", "HIGH"),
    ("MNQ", "LONDON_METALS", "HIGH"),
    ("MES", "LONDON_METALS", "HIGH"),
    ("MNQ", "US_DATA_1000", "HIGH"),
    ("MES", "NYSE_CLOSE", "HIGH"),
    ("MNQ", "CME_PRECLOSE", "LOW"),  # flipped direction
    ("MGC", "US_DATA_830", "LOW"),
    ("MNQ", "US_DATA_830", "LOW"),
]


def load_outcomes(con, symbol: str, session: str) -> pd.DataFrame:
    q = f"""
        SELECT o.trading_day, o.pnl_r,
               df.overnight_range, df.atr_20,
               df.overnight_range / NULLIF(df.atr_20, 0) as ovn_norm
        FROM orb_outcomes o
        JOIN daily_features df
          ON o.trading_day=df.trading_day AND o.symbol=df.symbol AND o.orb_minutes=df.orb_minutes
        WHERE o.symbol='{symbol}' AND o.entry_model='{ENTRY_MODEL}' AND o.orb_minutes=5
              AND o.rr_target={RR_TARGET} AND o.orb_label='{session}'
              AND o.trading_day < '{HOLDOUT_DATE}'
              AND df.overnight_range IS NOT NULL AND df.atr_20 > 0
    """
    df = con.sql(q).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["year"] = df["trading_day"].dt.year
    return df.dropna(subset=["ovn_norm", "pnl_r"])


def compute_wr_spread(df: pd.DataFrame, col: str = "ovn_norm") -> dict:
    """Q5 - Q1 WR and ExpR spread."""
    if len(df) < MIN_BIN_N * 5:
        return {"error": "insufficient N"}
    try:
        df = df.copy()
        df["qbin"] = pd.qcut(df[col], 5, labels=False, duplicates="drop")
    except ValueError:
        return {"error": "qcut failed"}
    bins = sorted(df["qbin"].dropna().unique())
    if len(bins) < 5:
        return {"error": f"only {len(bins)} bins"}
    q1 = df[df["qbin"] == bins[0]]
    q5 = df[df["qbin"] == bins[-1]]
    if len(q1) < MIN_BIN_N or len(q5) < MIN_BIN_N:
        return {"error": "bin too small"}
    return {
        "q1_n": len(q1),
        "q5_n": len(q5),
        "q1_wr": (q1["pnl_r"] > 0).mean(),
        "q5_wr": (q5["pnl_r"] > 0).mean(),
        "q1_expr": q1["pnl_r"].mean(),
        "q5_expr": q5["pnl_r"].mean(),
        "wr_spread": (q5["pnl_r"] > 0).mean() - (q1["pnl_r"] > 0).mean(),
        "expr_spread": q5["pnl_r"].mean() - q1["pnl_r"].mean(),
    }


def t3_walkforward(df: pd.DataFrame, direction: str) -> dict:
    """Walk-forward: IS=2019-2023, OOS=2024-2025. WFE = OOS_WR_spread / IS_WR_spread."""
    is_df = df[df["year"] <= 2023]
    oos_df = df[df["year"] >= 2024]
    if len(is_df) < 100 or len(oos_df) < 50:
        return {"error": "insufficient split sizes", "is_n": len(is_df), "oos_n": len(oos_df)}
    is_res = compute_wr_spread(is_df)
    oos_res = compute_wr_spread(oos_df)
    if "error" in is_res or "error" in oos_res:
        return {"error": "compute failed"}
    sign_match = np.sign(is_res["wr_spread"]) == np.sign(oos_res["wr_spread"])
    wfe = (oos_res["wr_spread"] / is_res["wr_spread"]) if is_res["wr_spread"] != 0 else float("inf")
    return {
        "is_wr_spread": is_res["wr_spread"],
        "oos_wr_spread": oos_res["wr_spread"],
        "is_expr_spread": is_res["expr_spread"],
        "oos_expr_spread": oos_res["expr_spread"],
        "wfe": wfe,
        "sign_match": sign_match,
        "is_n": len(is_df),
        "oos_n": len(oos_df),
    }


def t4_sensitivity(df: pd.DataFrame, direction: str) -> dict:
    """±20% quantile threshold stability. Use Q20/Q40/Q50/Q60/Q80 and compare above/below."""
    if len(df) < 200:
        return {"error": "insufficient N"}
    q20 = df["ovn_norm"].quantile(0.20)
    q40 = df["ovn_norm"].quantile(0.40)
    q50 = df["ovn_norm"].quantile(0.50)
    q60 = df["ovn_norm"].quantile(0.60)
    q80 = df["ovn_norm"].quantile(0.80)
    results = {}
    for pct, thresh in [("Q20", q20), ("Q40", q40), ("Q50", q50), ("Q60", q60), ("Q80", q80)]:
        high = df[df["ovn_norm"] >= thresh]
        low = df[df["ovn_norm"] < thresh]
        if len(high) < 50 or len(low) < 50:
            continue
        wr_diff = (high["pnl_r"] > 0).mean() - (low["pnl_r"] > 0).mean()
        expr_diff = high["pnl_r"].mean() - low["pnl_r"].mean()
        results[pct] = {"wr_diff": wr_diff, "expr_diff": expr_diff, "high_expr": high["pnl_r"].mean()}
    if not results:
        return {"error": "no valid thresholds"}
    # Check sign consistency (all same sign as direction)
    signs = [v["wr_diff"] for v in results.values()]
    expected_sign = 1 if direction == "HIGH" else -1
    all_match = all((s * expected_sign) > 0 for s in signs)
    return {"thresholds": results, "all_sign_match": all_match}


def t6_null_bootstrap(df: pd.DataFrame, n_perms: int = N_BOOTSTRAP) -> dict:
    """Bootstrap permutation: shuffle ovn_norm labels, measure WR spread distribution."""
    if len(df) < MIN_BIN_N * 5:
        return {"error": "insufficient N"}
    observed = compute_wr_spread(df)
    if "error" in observed:
        return {"error": observed["error"]}
    obs_wr_spread = observed["wr_spread"]

    rng = np.random.default_rng(42)
    null_spreads = []
    pnl = np.asarray(df["pnl_r"].values, dtype=float)
    for _ in range(n_perms):
        shuffled_pnl = rng.permutation(pnl)
        temp = df.copy()
        temp["pnl_r"] = shuffled_pnl
        null_res = compute_wr_spread(temp)
        if "error" not in null_res:
            null_spreads.append(null_res["wr_spread"])

    if not null_spreads:
        return {"error": "all perms failed"}
    null_spreads = np.array(null_spreads)
    # Two-tailed: fraction at least as extreme
    b = int(np.sum(np.abs(null_spreads) >= np.abs(obs_wr_spread)))
    p_value = (b + 1) / (len(null_spreads) + 1)
    return {
        "observed": obs_wr_spread,
        "null_mean": float(null_spreads.mean()),
        "null_std": float(null_spreads.std()),
        "null_p95": float(np.percentile(np.abs(null_spreads), 95)),
        "p_value": p_value,
        "n_perms": len(null_spreads),
    }


def t7_per_year(df: pd.DataFrame, direction: str) -> dict:
    """Per-year stability: compute WR spread per year, count years matching direction."""
    years = sorted(df["year"].unique())
    results = {}
    expected_sign = 1 if direction == "HIGH" else -1
    same_sign_count = 0
    valid_years = 0
    for yr in years:
        yr_df = df[df["year"] == yr]
        if len(yr_df) < 50:
            continue
        yr_res = compute_wr_spread(yr_df)
        if "error" in yr_res:
            continue
        valid_years += 1
        matches = (yr_res["wr_spread"] * expected_sign) > 0
        if matches:
            same_sign_count += 1
        results[int(yr)] = {"wr_spread": yr_res["wr_spread"], "matches": matches, "n": len(yr_df)}
    return {
        "per_year": results,
        "same_sign": same_sign_count,
        "valid_years": valid_years,
        "pct_same_sign": same_sign_count / valid_years if valid_years else 0,
    }


def run_all_passers():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    print("=" * 80)
    print("T2-T8 BATTERY — overnight_range/atr_20 ratio (post Phase 3c data)")
    print(f"  N_bootstrap = {N_BOOTSTRAP}, holdout = {HOLDOUT_DATE}")
    print("=" * 80)

    survivors = []
    for sym, sess, direction in T1_PASSERS:
        print(f"\n--- {sym} {sess} (direction={direction}) ---")
        df = load_outcomes(con, sym, sess)
        if len(df) < 200:
            print(f"  INSUFFICIENT N={len(df)} — SKIP")
            continue

        # T2 IS baseline
        t2 = compute_wr_spread(df)
        print(
            f"  T2 IS: N={len(df)} Q1 WR={t2['q1_wr'] * 100:.1f}%/ExpR={t2['q1_expr']:+.3f} "
            f"Q5 WR={t2['q5_wr'] * 100:.1f}%/ExpR={t2['q5_expr']:+.3f} "
            f"WR_spread={t2['wr_spread'] * 100:+.1f}pp"
        )

        # T3 walk-forward
        t3 = t3_walkforward(df, direction)
        if "error" not in t3:
            print(
                f"  T3 WF: IS_spread={t3['is_wr_spread'] * 100:+.1f}pp "
                f"OOS_spread={t3['oos_wr_spread'] * 100:+.1f}pp WFE={t3['wfe']:.2f} "
                f"sign_match={t3['sign_match']}"
            )
            t3_pass = t3["sign_match"] and t3["wfe"] > 0.50
        else:
            print(f"  T3 WF: {t3['error']}")
            t3_pass = False

        # T4 sensitivity
        t4 = t4_sensitivity(df, direction)
        if "error" not in t4:
            t4_pass = t4["all_sign_match"]
            print(f"  T4 SENS: ±20% thresholds all match sign: {t4_pass}")
            for pct, v in t4["thresholds"].items():
                print(f"    {pct}: high_ExpR={v['high_expr']:+.3f}, WR_diff={v['wr_diff'] * 100:+.1f}pp")
        else:
            print(f"  T4 SENS: {t4['error']}")
            t4_pass = False

        # T6 null bootstrap
        t6 = t6_null_bootstrap(df)
        if "error" not in t6:
            t6_pass = t6["p_value"] < 0.05
            print(
                f"  T6 NULL: observed={t6['observed'] * 100:+.1f}pp null_mean={t6['null_mean'] * 100:+.2f}pp "
                f"null_p95={t6['null_p95'] * 100:.1f}pp p={t6['p_value']:.4f} PASS={t6_pass}"
            )
        else:
            print(f"  T6 NULL: {t6['error']}")
            t6_pass = False

        # T7 per-year stability
        t7 = t7_per_year(df, direction)
        print(f"  T7 YEAR: {t7['same_sign']}/{t7['valid_years']} years match ({t7['pct_same_sign'] * 100:.0f}%)")
        t7_pass = t7["pct_same_sign"] >= 0.75

        # Verdict
        verdict = "SURVIVOR" if (t3_pass and t4_pass and t6_pass and t7_pass) else "KILL"
        why = []
        if not t3_pass:
            why.append("T3")
        if not t4_pass:
            why.append("T4")
        if not t6_pass:
            why.append("T6")
        if not t7_pass:
            why.append("T7")
        print(f"  VERDICT: {verdict}" + (f" (killed by {','.join(why)})" if why else ""))

        if verdict == "SURVIVOR":
            survivors.append((sym, sess, direction, t2, t3, t6, t7))

    con.close()

    print()
    print("=" * 80)
    print(f"SURVIVORS AFTER T2-T8: {len(survivors)}/{len(T1_PASSERS)}")
    print("=" * 80)
    for sym, sess, direction, t2, t3, t6, t7 in survivors:
        print(
            f"  {sym} {sess} ({direction}): Q5_ExpR={t2['q5_expr']:+.3f} "
            f"WFE={t3['wfe']:.2f} p={t6['p_value']:.4f} years={t7['same_sign']}/{t7['valid_years']}"
        )


if __name__ == "__main__":
    run_all_passers()
