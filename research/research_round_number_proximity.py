#!/usr/bin/env python3
"""
Round-Number Proximity — Feature Build + T0-T8 Audit

Hypothesis: Price anti-clusters at round numbers during US sessions.
Round numbers = momentum accelerators at high-liquidity events.

Features computed:
  - distance_to_round_pts: |entry_price - nearest_round| in points
  - distance_to_round_R:   distance_to_round_pts / orb_size (R-multiples)
  - crosses_round:          bool — does ORB range straddle a round number?

Round-number definitions (locked before analysis):
  MGC: $10 increments (2050, 2060, 2070...)
  MNQ: 100pt increments (17500, 17600, 17700...)
  MES: 25pt increments (5000, 5025, 5050...)

Test battery (quant-audit-protocol):
  T0: Tautology check (corr with existing filters)
  T1: Win-rate monotonicity across quintiles
  T2: In-sample baseline
  T3: Out-of-sample / walk-forward
  T4: Sensitivity ±20%
  T5: Family comparison (all sessions)
  T6: Null floor (bootstrap permutation)
  T7: Per-year stability
  T8: Cross-instrument consistency

Usage:
  python research/research_round_number_proximity.py
  python research/research_round_number_proximity.py --instrument MNQ
  python research/research_round_number_proximity.py --us-only
"""

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.cost_model import COST_SPECS, get_cost_spec
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import GOLD_DB_PATH

# =============================================================================
# ROUND-NUMBER DEFINITIONS (locked before analysis)
# =============================================================================

ROUND_INCREMENTS = {
    "MGC": 10.0,    # Gold $10 increments (2050, 2060, 2070...)
    "MNQ": 100.0,   # Nasdaq 100pt increments (17500, 17600...)
    "MES": 25.0,    # S&P 25pt increments (5000, 5025, 5050...)
    "M2K": 10.0,    # Russell 10pt increments
}

# US sessions = the ones that showed anti-clustering in prior analysis
US_SESSIONS = {"COMEX_SETTLE", "NYSE_OPEN", "US_DATA_1000", "US_DATA_830", "NYSE_CLOSE", "CME_PRECLOSE"}
ASIAN_SESSIONS = {"TOKYO_OPEN", "SINGAPORE_OPEN", "CME_REOPEN", "LONDON_METALS", "EUROPE_FLOW", "BRISBANE_1025"}

# IS/OOS split — define BEFORE seeing results
IS_END = date(2024, 12, 31)   # In-sample ends 2024
OOS_START = date(2025, 1, 1)  # OOS = 2025+


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================


def nearest_round(price: float, increment: float) -> float:
    """Return the nearest round number to price."""
    return round(price / increment) * increment


def distance_to_nearest_round(price: float, increment: float) -> float:
    """Unsigned distance from price to nearest round number, in points."""
    nearest = nearest_round(price, increment)
    return abs(price - nearest)


def orb_crosses_round(orb_high: float, orb_low: float, increment: float) -> bool:
    """Does the ORB range contain (straddle) a round number?"""
    # Find the round number just above orb_low
    first_round_above_low = math.ceil(orb_low / increment) * increment
    return first_round_above_low <= orb_high


def compute_round_features(row: dict, symbol: str) -> dict:
    """Compute round-number proximity features for one trade."""
    increment = ROUND_INCREMENTS.get(symbol)
    if increment is None:
        return {}

    orb_high = row.get("orb_high")
    orb_low = row.get("orb_low")
    orb_size = row.get("orb_size")
    break_dir = row.get("break_dir")

    if orb_high is None or orb_low is None or orb_size is None or orb_size <= 0:
        return {}

    # Entry price = the ORB boundary in break direction
    if break_dir == "long":
        entry_price = orb_high
    elif break_dir == "short":
        entry_price = orb_low
    else:
        return {}

    dist_pts = distance_to_nearest_round(entry_price, increment)
    dist_r = dist_pts / orb_size
    crosses = orb_crosses_round(orb_high, orb_low, increment)

    # Also: distance from TARGET to nearest round (1R target = entry ± orb_size)
    if break_dir == "long":
        target_price = entry_price + orb_size
    else:
        target_price = entry_price - orb_size
    target_dist_pts = distance_to_nearest_round(target_price, increment)
    target_dist_r = target_dist_pts / orb_size

    return {
        "distance_to_round_pts": round(dist_pts, 4),
        "distance_to_round_R": round(dist_r, 4),
        "target_distance_to_round_R": round(target_dist_r, 4),
        "crosses_round": crosses,
    }


# =============================================================================
# DATA LOADING
# =============================================================================


def load_trade_data(instrument: str | None = None) -> pd.DataFrame:
    """Load orb_outcomes joined with daily_features for active instruments."""
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    instruments = [instrument] if instrument else list(ACTIVE_ORB_INSTRUMENTS)
    placeholders = ", ".join(["?"] * len(instruments))

    query = f"""
    SELECT
        o.trading_day,
        o.symbol,
        o.orb_minutes,
        o.orb_label,
        o.entry_model,
        o.confirm_bars,
        o.rr_target,
        o.entry_price,
        o.stop_price,
        o.target_price,
        o.pnl_r,
        o.outcome,
        -- Derive break direction from entry vs stop
        CASE WHEN o.entry_price > o.stop_price THEN 'long'
             WHEN o.entry_price < o.stop_price THEN 'short'
             ELSE NULL END AS break_dir,
        d.atr_20,
        CASE o.orb_label
            {" ".join(f"WHEN '{lbl}' THEN d.orb_{lbl}_high" for lbl in SESSION_CATALOG)}
        END AS orb_high,
        CASE o.orb_label
            {" ".join(f"WHEN '{lbl}' THEN d.orb_{lbl}_low" for lbl in SESSION_CATALOG)}
        END AS orb_low,
        CASE o.orb_label
            {" ".join(f"WHEN '{lbl}' THEN d.orb_{lbl}_size" for lbl in SESSION_CATALOG)}
        END AS orb_size,
        CASE o.orb_label
            {" ".join(f"WHEN '{lbl}' THEN d.orb_{lbl}_break_delay_min" for lbl in SESSION_CATALOG)}
        END AS break_delay_min
    FROM orb_outcomes o
    JOIN daily_features d
        ON o.trading_day = d.trading_day
        AND o.symbol = d.symbol
        AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol IN ({placeholders})
        AND o.entry_price IS NOT NULL
        AND o.pnl_r IS NOT NULL
        AND o.entry_model = 'E1'
        AND o.confirm_bars = 1
        AND o.rr_target = 2.0
        AND o.orb_minutes = 5
    ORDER BY o.trading_day, o.symbol, o.orb_label
    """

    df = con.execute(query, instruments).fetchdf()
    con.close()
    print(f"Loaded {len(df):,} trades across {df['symbol'].nunique()} instruments, "
          f"{df['orb_label'].nunique()} sessions")
    return df


# =============================================================================
# T0: TAUTOLOGY CHECK
# =============================================================================


def run_t0_tautology(df: pd.DataFrame) -> dict:
    """Check correlation with existing filters (orb_size, atr_20, break_delay)."""
    print("\n" + "=" * 60)
    print("T0: TAUTOLOGY CHECK")
    print("=" * 60)

    features_to_check = ["orb_size", "atr_20", "break_delay_min"]
    round_features = ["distance_to_round_pts", "distance_to_round_R", "target_distance_to_round_R"]

    results = {}
    for rf in round_features:
        if rf not in df.columns:
            continue
        for ef in features_to_check:
            if ef not in df.columns:
                continue
            mask = df[rf].notna() & df[ef].notna()
            if mask.sum() < 30:
                continue
            corr = df.loc[mask, rf].corr(df.loc[mask, ef])
            label = "DUPLICATE_FILTER" if abs(corr) > 0.70 else "PASS"
            results[f"{rf}_vs_{ef}"] = {"corr": round(corr, 4), "verdict": label}
            print(f"  corr({rf}, {ef}) = {corr:.4f}  → {label}")

    overall = "DUPLICATE_FILTER" if any(v["verdict"] == "DUPLICATE_FILTER" for v in results.values()) else "PASS"
    print(f"\n  T0 VERDICT: {overall}")
    return {"results": results, "verdict": overall}


# =============================================================================
# T1: WIN-RATE MONOTONICITY
# =============================================================================


def run_t1_wr_monotonicity(df: pd.DataFrame, feature: str = "distance_to_round_R",
                            session_regime: str = "ALL") -> dict:
    """Quintile analysis of win rate by round-number proximity."""
    print("\n" + "=" * 60)
    print(f"T1: WIN-RATE MONOTONICITY — {feature} ({session_regime} sessions)")
    print("=" * 60)

    if session_regime == "US":
        mask = df["orb_label"].isin(US_SESSIONS)
    elif session_regime == "ASIAN":
        mask = df["orb_label"].isin(ASIAN_SESSIONS)
    else:
        mask = pd.Series(True, index=df.index)

    sub = df.loc[mask & df[feature].notna()].copy()
    if len(sub) < 100:
        print(f"  Insufficient data: {len(sub)} trades")
        return {"verdict": "INSUFFICIENT_DATA", "n": len(sub)}

    sub["quintile"] = pd.qcut(sub[feature], 5, labels=False, duplicates="drop") + 1

    print(f"\n  {'Bin':>4} | {'N':>6} | {'WR':>6} | {'ExpR':>8} | {'AvgWinR':>8} | {'Mean':>8}")
    print("  " + "-" * 55)

    quintile_stats = []
    for q in sorted(sub["quintile"].unique()):
        qdf = sub[sub["quintile"] == q]
        n = len(qdf)
        wr = (qdf["outcome"] == "win").mean() * 100
        expr = qdf["pnl_r"].mean()
        avg_win = qdf.loc[qdf["outcome"] == "win", "pnl_r"].mean() if (qdf["outcome"] == "win").any() else 0
        mean_feat = qdf[feature].mean()
        quintile_stats.append({"q": q, "n": n, "wr": wr, "expr": expr, "avg_win": avg_win, "mean": mean_feat})
        print(f"  Q{q:>3} | {n:>6} | {wr:>5.1f}% | {expr:>+7.4f} | {avg_win:>+7.4f} | {mean_feat:>8.4f}")

    wr_values = [s["wr"] for s in quintile_stats]
    wr_spread = max(wr_values) - min(wr_values)
    expr_values = [s["expr"] for s in quintile_stats]

    # Check monotonicity direction
    if wr_values[0] > wr_values[-1]:
        direction = "INVERSE"  # low distance = high WR (near round = better)
    else:
        direction = "DIRECT"   # high distance = high WR (far from round = better)

    if wr_spread < 3.0:
        verdict = "ARITHMETIC_ONLY"
    elif wr_spread >= 5.0:
        verdict = "SIGNAL"
    else:
        verdict = "AMBIGUOUS"

    print(f"\n  WR spread: {wr_spread:.1f}% | Direction: {direction} | Verdict: {verdict}")
    print(f"  N total: {len(sub):,}")

    return {
        "quintile_stats": quintile_stats,
        "wr_spread": round(wr_spread, 2),
        "direction": direction,
        "verdict": verdict,
        "n": len(sub),
    }


# =============================================================================
# T1b: CROSSES_ROUND BINARY TEST
# =============================================================================


def run_t1b_crosses_round(df: pd.DataFrame, session_regime: str = "ALL") -> dict:
    """Binary test: does crossing a round number predict outcome?"""
    print("\n" + "=" * 60)
    print(f"T1b: CROSSES_ROUND BINARY TEST ({session_regime} sessions)")
    print("=" * 60)

    if session_regime == "US":
        mask = df["orb_label"].isin(US_SESSIONS)
    elif session_regime == "ASIAN":
        mask = df["orb_label"].isin(ASIAN_SESSIONS)
    else:
        mask = pd.Series(True, index=df.index)

    sub = df.loc[mask & df["crosses_round"].notna()].copy()
    if len(sub) < 100:
        print(f"  Insufficient data: {len(sub)} trades")
        return {"verdict": "INSUFFICIENT_DATA"}

    for val, label in [(True, "CROSSES"), (False, "NO_CROSS")]:
        g = sub[sub["crosses_round"] == val]
        n = len(g)
        wr = (g["outcome"] == "win").mean() * 100
        expr = g["pnl_r"].mean()
        print(f"  {label:>10}: N={n:>6}, WR={wr:>5.1f}%, ExpR={expr:>+.4f}")

    crosses = sub[sub["crosses_round"] == True]
    no_cross = sub[sub["crosses_round"] == False]
    wr_cross = (crosses["outcome"] == "win").mean() * 100
    wr_no = (no_cross["outcome"] == "win").mean() * 100
    wr_diff = wr_cross - wr_no

    print(f"\n  WR diff (CROSSES - NO_CROSS): {wr_diff:+.1f}%")

    # Quick significance: proportions z-test
    from scipy import stats
    n1, n2 = len(crosses), len(no_cross)
    w1 = (crosses["outcome"] == "win").sum()
    w2 = (no_cross["outcome"] == "win").sum()
    if n1 > 0 and n2 > 0:
        p_pooled = (w1 + w2) / (n1 + n2)
        se = (p_pooled * (1 - p_pooled) * (1/n1 + 1/n2)) ** 0.5
        if se > 0:
            z_stat = (w1/n1 - w2/n2) / se
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat, p_val = 0, 1.0
    else:
        z_stat, p_val = 0, 1.0

    print(f"  z={z_stat:.3f}, p={p_val:.4f}")

    return {
        "wr_crosses": round(wr_cross, 2),
        "wr_no_cross": round(wr_no, 2),
        "wr_diff": round(wr_diff, 2),
        "z": round(z_stat, 3),
        "p": round(p_val, 4),
        "n_crosses": n1,
        "n_no_cross": n2,
    }


# =============================================================================
# T2/T3: IN-SAMPLE + OUT-OF-SAMPLE
# =============================================================================


def run_t2_t3_is_oos(df: pd.DataFrame, feature: str = "distance_to_round_R",
                      session_regime: str = "ALL", threshold_pct: float = 50.0) -> dict:
    """IS/OOS split test. Threshold = percentile cutoff on feature."""
    print("\n" + "=" * 60)
    print(f"T2/T3: IS/OOS — {feature} ({session_regime} sessions)")
    print("=" * 60)

    if session_regime == "US":
        mask = df["orb_label"].isin(US_SESSIONS)
    elif session_regime == "ASIAN":
        mask = df["orb_label"].isin(ASIAN_SESSIONS)
    else:
        mask = pd.Series(True, index=df.index)

    sub = df.loc[mask & df[feature].notna()].copy()

    # Convert trading_day to date for comparison (DuckDB returns datetime64)
    td = pd.to_datetime(sub["trading_day"]).dt.date
    is_df = sub[td <= IS_END]
    oos_df = sub[td >= OOS_START]

    print(f"  IS:  {len(is_df):>6} trades ({is_df['trading_day'].min()} to {is_df['trading_day'].max()})")
    print(f"  OOS: {len(oos_df):>6} trades ({oos_df['trading_day'].min()} to {oos_df['trading_day'].max()})")

    results = {}
    for label, sdf in [("IS", is_df), ("OOS", oos_df)]:
        if len(sdf) < 30:
            print(f"  {label}: insufficient data ({len(sdf)})")
            results[label] = {"n": len(sdf), "verdict": "INSUFFICIENT"}
            continue

        # Compute median split (below median = closer to round)
        median_val = sdf[feature].median()
        close_mask = sdf[feature] <= median_val
        far_mask = sdf[feature] > median_val

        close = sdf[close_mask]
        far = sdf[far_mask]

        wr_close = (close["outcome"] == "win").mean() * 100
        wr_far = (far["outcome"] == "win").mean() * 100
        expr_close = close["pnl_r"].mean()
        expr_far = far["pnl_r"].mean()

        print(f"\n  {label} — median split at {feature}={median_val:.4f}:")
        print(f"    CLOSE (≤median): N={len(close):>5}, WR={wr_close:>5.1f}%, ExpR={expr_close:>+.4f}")
        print(f"    FAR   (>median): N={len(far):>5}, WR={wr_far:>5.1f}%, ExpR={expr_far:>+.4f}")
        print(f"    WR diff: {wr_close - wr_far:+.1f}%  ExpR diff: {expr_close - expr_far:+.4f}")

        results[label] = {
            "n": len(sdf),
            "median": round(median_val, 4),
            "wr_close": round(wr_close, 2),
            "wr_far": round(wr_far, 2),
            "expr_close": round(expr_close, 4),
            "expr_far": round(expr_far, 4),
        }

    # WFE: compare IS and OOS effect direction
    if "IS" in results and "OOS" in results:
        is_diff = results["IS"].get("expr_close", 0) - results["IS"].get("expr_far", 0)
        oos_diff = results["OOS"].get("expr_close", 0) - results["OOS"].get("expr_far", 0)
        same_direction = (is_diff > 0) == (oos_diff > 0) if (is_diff != 0 and oos_diff != 0) else False
        print(f"\n  IS effect: {is_diff:+.4f} | OOS effect: {oos_diff:+.4f} | Same direction: {same_direction}")

    return results


# =============================================================================
# T5: FAMILY COMPARISON (per-session)
# =============================================================================


def run_t5_family(df: pd.DataFrame, feature: str = "distance_to_round_R") -> dict:
    """Same feature across ALL sessions — not just the winners."""
    print("\n" + "=" * 60)
    print(f"T5: FAMILY COMPARISON — {feature} per session")
    print("=" * 60)

    results = {}
    print(f"\n  {'Session':<20} | {'N':>6} | {'WR_Q1':>6} | {'WR_Q5':>6} | {'Spread':>7} | {'ExpR_Q1':>8} | {'ExpR_Q5':>8} | {'Verdict'}")
    print("  " + "-" * 95)

    for session in sorted(df["orb_label"].unique()):
        sdf = df[(df["orb_label"] == session) & df[feature].notna()]
        if len(sdf) < 50:
            print(f"  {session:<20} | {len(sdf):>6} | {'---':>6} | {'---':>6} | {'---':>7} | {'---':>8} | {'---':>8} | SKIP (N<50)")
            continue

        sdf = sdf.copy()
        try:
            sdf["quintile"] = pd.qcut(sdf[feature], 5, labels=False, duplicates="drop") + 1
        except ValueError:
            print(f"  {session:<20} | {len(sdf):>6} | {'---':>6} | {'---':>6} | {'---':>7} | {'---':>8} | {'---':>8} | SKIP (qcut fail)")
            continue

        q1 = sdf[sdf["quintile"] == 1]
        q5 = sdf[sdf["quintile"] == sdf["quintile"].max()]
        wr1 = (q1["outcome"] == "win").mean() * 100
        wr5 = (q5["outcome"] == "win").mean() * 100
        expr1 = q1["pnl_r"].mean()
        expr5 = q5["pnl_r"].mean()
        spread = wr1 - wr5
        regime = "US" if session in US_SESSIONS else "ASIAN"

        verdict = "SIGNAL" if abs(spread) >= 5 else ("AMBIGUOUS" if abs(spread) >= 3 else "FLAT")
        results[session] = {
            "n": len(sdf), "wr_q1": round(wr1, 1), "wr_q5": round(wr5, 1),
            "spread": round(spread, 1), "regime": regime, "verdict": verdict,
        }
        print(f"  {session:<20} | {len(sdf):>6} | {wr1:>5.1f}% | {wr5:>5.1f}% | {spread:>+6.1f}% | {expr1:>+7.4f} | {expr5:>+7.4f} | {verdict} [{regime}]")

    signal_count = sum(1 for v in results.values() if v["verdict"] == "SIGNAL")
    total = len(results)
    print(f"\n  SIGNAL sessions: {signal_count}/{total}")

    return results


# =============================================================================
# T6: NULL FLOOR (bootstrap permutation)
# =============================================================================


def run_t6_null_floor(df: pd.DataFrame, feature: str = "distance_to_round_R",
                       session_regime: str = "ALL", n_perms: int = 1000) -> dict:
    """Shuffle pnl_r, compute WR spread each time, compare to observed."""
    print("\n" + "=" * 60)
    print(f"T6: NULL FLOOR — {feature} ({session_regime}, {n_perms} permutations)")
    print("=" * 60)

    if session_regime == "US":
        mask = df["orb_label"].isin(US_SESSIONS)
    elif session_regime == "ASIAN":
        mask = df["orb_label"].isin(ASIAN_SESSIONS)
    else:
        mask = pd.Series(True, index=df.index)

    sub = df.loc[mask & df[feature].notna()].copy()
    if len(sub) < 100:
        print(f"  Insufficient data: {len(sub)}")
        return {"verdict": "INSUFFICIENT_DATA"}

    # Observed: WR in Q1 (closest to round) vs Q5 (farthest)
    sub["quintile"] = pd.qcut(sub[feature], 5, labels=False, duplicates="drop") + 1
    q1 = sub[sub["quintile"] == 1]
    q5 = sub[sub["quintile"] == sub["quintile"].max()]
    observed_spread = (q1["outcome"] == "win").mean() - (q5["outcome"] == "win").mean()

    # Permutation test
    rng = np.random.default_rng(42)
    outcomes = np.array(sub["outcome"].values.tolist())
    quintiles = sub["quintile"].values
    q1_mask = quintiles == 1
    q5_mask = quintiles == sub["quintile"].max()

    exceeding = 0
    for _ in range(n_perms):
        rng.shuffle(outcomes)
        perm_wr1 = (outcomes[q1_mask] == "win").mean()
        perm_wr5 = (outcomes[q5_mask] == "win").mean()
        perm_spread = perm_wr1 - perm_wr5
        if abs(perm_spread) >= abs(observed_spread):
            exceeding += 1

    # Phipson & Smyth correction
    p_val = (exceeding + 1) / (n_perms + 1)

    print(f"  Observed WR spread (Q1-Q5): {observed_spread*100:+.2f}%")
    print(f"  Null P95 exceeded: {exceeding}/{n_perms}")
    print(f"  p-value: {p_val:.4f}")

    verdict = "BEATS_NULL" if p_val < 0.05 else "NO_EDGE"
    print(f"  Verdict: {verdict}")

    return {
        "observed_spread": round(observed_spread * 100, 2),
        "exceeding": exceeding,
        "n_perms": n_perms,
        "p_val": round(p_val, 4),
        "verdict": verdict,
    }


# =============================================================================
# T7: PER-YEAR STABILITY
# =============================================================================


def run_t7_per_year(df: pd.DataFrame, feature: str = "distance_to_round_R",
                     session_regime: str = "ALL") -> dict:
    """Check Q1 vs Q5 WR spread per year. Must be same direction in >=7/10 years."""
    print("\n" + "=" * 60)
    print(f"T7: PER-YEAR STABILITY — {feature} ({session_regime} sessions)")
    print("=" * 60)

    if session_regime == "US":
        mask = df["orb_label"].isin(US_SESSIONS)
    elif session_regime == "ASIAN":
        mask = df["orb_label"].isin(ASIAN_SESSIONS)
    else:
        mask = pd.Series(True, index=df.index)

    sub = df.loc[mask & df[feature].notna()].copy()
    sub["year"] = pd.to_datetime(sub["trading_day"]).dt.year

    # Compute overall direction first
    sub["quintile"] = pd.qcut(sub[feature], 5, labels=False, duplicates="drop") + 1
    q1_all = sub[sub["quintile"] == 1]
    q5_all = sub[sub["quintile"] == sub["quintile"].max()]
    overall_direction = (q1_all["outcome"] == "win").mean() > (q5_all["outcome"] == "win").mean()

    print(f"\n  Overall direction: Q1 {'>' if overall_direction else '<'} Q5")
    print(f"\n  {'Year':>6} | {'N':>6} | {'WR_Q1':>6} | {'WR_Q5':>6} | {'Spread':>7} | {'Consistent'}")
    print("  " + "-" * 55)

    consistent_years = 0
    total_years = 0
    for year in sorted(sub["year"].unique()):
        ydf = sub[sub["year"] == year]
        if len(ydf) < 20:
            continue
        total_years += 1

        try:
            ydf = ydf.copy()
            ydf["yq"] = pd.qcut(ydf[feature], 5, labels=False, duplicates="drop") + 1
        except ValueError:
            continue

        yq1 = ydf[ydf["yq"] == 1]
        yq5 = ydf[ydf["yq"] == ydf["yq"].max()]
        if len(yq1) == 0 or len(yq5) == 0:
            continue

        wr1 = (yq1["outcome"] == "win").mean() * 100
        wr5 = (yq5["outcome"] == "win").mean() * 100
        spread = wr1 - wr5
        year_consistent = (spread > 0) == overall_direction
        if year_consistent:
            consistent_years += 1

        print(f"  {year:>6} | {len(ydf):>6} | {wr1:>5.1f}% | {wr5:>5.1f}% | {spread:>+6.1f}% | {'✓' if year_consistent else '✗'}")

    print(f"\n  Consistent: {consistent_years}/{total_years} years")
    verdict = "STABLE" if total_years > 0 and consistent_years / total_years >= 0.7 else "ERA_DEPENDENT"
    print(f"  Verdict: {verdict}")

    return {
        "consistent": consistent_years,
        "total": total_years,
        "verdict": verdict,
    }


# =============================================================================
# T8: CROSS-INSTRUMENT
# =============================================================================


def run_t8_cross_instrument(df: pd.DataFrame, feature: str = "distance_to_round_R",
                             session_regime: str = "ALL") -> dict:
    """Same direction on all active instruments?"""
    print("\n" + "=" * 60)
    print(f"T8: CROSS-INSTRUMENT — {feature} ({session_regime} sessions)")
    print("=" * 60)

    if session_regime == "US":
        mask = df["orb_label"].isin(US_SESSIONS)
    elif session_regime == "ASIAN":
        mask = df["orb_label"].isin(ASIAN_SESSIONS)
    else:
        mask = pd.Series(True, index=df.index)

    sub = df.loc[mask & df[feature].notna()].copy()
    results = {}

    print(f"\n  {'Symbol':<8} | {'N':>6} | {'WR_Q1':>6} | {'WR_Q5':>6} | {'Spread':>7} | {'Direction'}")
    print("  " + "-" * 55)

    for sym in sorted(sub["symbol"].unique()):
        sdf = sub[sub["symbol"] == sym]
        if len(sdf) < 50:
            continue

        sdf = sdf.copy()
        try:
            sdf["quintile"] = pd.qcut(sdf[feature], 5, labels=False, duplicates="drop") + 1
        except ValueError:
            continue

        q1 = sdf[sdf["quintile"] == 1]
        q5 = sdf[sdf["quintile"] == sdf["quintile"].max()]
        wr1 = (q1["outcome"] == "win").mean() * 100
        wr5 = (q5["outcome"] == "win").mean() * 100
        spread = wr1 - wr5
        direction = "Q1>Q5" if spread > 0 else "Q5>Q1"
        results[sym] = {"spread": round(spread, 1), "direction": direction, "n": len(sdf)}

        print(f"  {sym:<8} | {len(sdf):>6} | {wr1:>5.1f}% | {wr5:>5.1f}% | {spread:>+6.1f}% | {direction}")

    directions = [v["direction"] for v in results.values()]
    consistent = len(set(directions)) == 1 if directions else False
    verdict = "CONSISTENT" if consistent else "INCONSISTENT"
    print(f"\n  Directions: {directions} → {verdict}")

    return {"results": results, "verdict": verdict}


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Round-number proximity T0-T8 audit")
    parser.add_argument("--instrument", type=str, default=None, help="Single instrument (default: all active)")
    parser.add_argument("--us-only", action="store_true", help="Only test US sessions")
    args = parser.parse_args()

    print("━" * 60)
    print("ROUND-NUMBER PROXIMITY — T0-T8 AUDIT")
    print("━" * 60)
    print(f"Instruments: {args.instrument or 'ALL ACTIVE'}")
    print(f"IS end: {IS_END} | OOS start: {OOS_START}")
    print(f"Round increments: {ROUND_INCREMENTS}")

    # Load data
    df = load_trade_data(args.instrument)

    # Compute round-number features
    print("\nComputing round-number features...")
    features_list = []
    for _, row in df.iterrows():
        feats = compute_round_features(row.to_dict(), row["symbol"])
        features_list.append(feats)

    feat_df = pd.DataFrame(features_list)
    for col in feat_df.columns:
        df[col] = feat_df[col].values

    has_features = df["distance_to_round_R"].notna().sum()
    print(f"  Features computed for {has_features:,}/{len(df):,} trades")

    # Quick distribution check
    print("\n  Feature distributions:")
    for col in ["distance_to_round_pts", "distance_to_round_R"]:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"    {col}: mean={vals.mean():.4f}, median={vals.median():.4f}, "
                  f"std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")
    if "crosses_round" in df.columns:
        cross_pct = df["crosses_round"].mean() * 100
        print(f"    crosses_round: {cross_pct:.1f}% of trades straddle a round number")

    # Run test battery
    regimes = ["US"] if args.us_only else ["ALL", "US", "ASIAN"]

    # T0: Tautology
    t0 = run_t0_tautology(df)
    if t0["verdict"] == "DUPLICATE_FILTER":
        print("\n\n⛔ T0 FAILED — DUPLICATE FILTER. Halting.")
        return

    for regime in regimes:
        print(f"\n\n{'▓' * 60}")
        print(f"  REGIME: {regime}")
        print(f"{'▓' * 60}")

        # T1: WR monotonicity
        t1_dist = run_t1_wr_monotonicity(df, "distance_to_round_R", regime)
        t1_target = run_t1_wr_monotonicity(df, "target_distance_to_round_R", regime)

        # T1b: crosses_round binary
        t1b = run_t1b_crosses_round(df, regime)

        # T2/T3: IS/OOS
        run_t2_t3_is_oos(df, "distance_to_round_R", regime)

        # T6: Null floor
        run_t6_null_floor(df, "distance_to_round_R", regime, n_perms=1000)

        # T7: Per-year
        run_t7_per_year(df, "distance_to_round_R", regime)

    # T5: Family comparison (all sessions, all instruments)
    run_t5_family(df, "distance_to_round_R")

    # T8: Cross-instrument
    for regime in regimes:
        run_t8_cross_instrument(df, "distance_to_round_R", regime)

    print("\n\n" + "━" * 60)
    print("AUDIT COMPLETE")
    print("━" * 60)


if __name__ == "__main__":
    main()
