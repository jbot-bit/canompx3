"""
research_win_direction.py -- Direction prediction for MGC 0900 prev=WIN days.

CORE QUESTION: On days where yesterday was a WINNING 0900 trade, can any
zero-look-ahead signal predict whether today's break will be LONG or SHORT?

If yes → one-sided E0 limit order → potentially highest single-trade expectancy.

PREDICTORS TESTED (all confirmed zero look-ahead — known before ORB forms):
  1. prev_break_dir     — yesterday's break direction (continuation vs reversal?)
  2. gap_open_points    — overnight gap direction
  3. rsi_14_at_0900    — RSI state at session open (overbought/oversold bias?)
  4. atr_velocity       — today ATR / yesterday ATR (expanding = momentum?)
  5. asia_position      — close of Asia session relative to its midpoint
  6. london_position    — London session high/low midpoint vs Asia midpoint
  7. day_of_week        — DOW directional bias
  8. us_dst             — DST regime (winter vs summer)

METHODOLOGY:
  - Filter: MGC 0900, G4+, prev_outcome = 'win'
  - Target: actual break direction (LONG=1, SHORT=0)
  - Test each predictor independently (logistic regression / t-test / chi-sq)
  - Apply Benjamini-Hochberg FDR correction (q=0.10) across all tests
  - Report survivors with mechanism hypothesis

Usage:
    python research/research_win_direction.py
"""

import sys
from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import false_discovery_control

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.paths import GOLD_DB_PATH

INSTRUMENT  = "MGC"
SESSION     = "0900"
ORB_MIN_PTS = 4.0


def load_data(con) -> pd.DataFrame:
    df = con.execute("""
        WITH base AS (
            SELECT
                d.trading_day,
                d.day_of_week,
                d.us_dst,
                d.orb_0900_break_dir                          AS break_dir,
                d.orb_0900_size                               AS orb_size,
                d.gap_open_points,
                d.rsi_14_at_0900,
                d.atr_20,
                d.session_asia_high,
                d.session_asia_low,
                d.session_london_high,
                d.session_london_low,
                LAG(d.orb_0900_outcome)   OVER (PARTITION BY d.symbol ORDER BY d.trading_day) AS prev_outcome,
                LAG(d.orb_0900_break_dir) OVER (PARTITION BY d.symbol ORDER BY d.trading_day) AS prev_break_dir,
                LAG(d.atr_20)             OVER (PARTITION BY d.symbol ORDER BY d.trading_day) AS prev_atr_20
            FROM daily_features d
            WHERE d.symbol = ? AND d.orb_minutes = 5
        )
        SELECT *
        FROM base
        WHERE prev_outcome = 'win'
          AND orb_size >= ?
          AND break_dir IS NOT NULL
        ORDER BY trading_day
    """, [INSTRUMENT, ORB_MIN_PTS]).df()

    # Encode target: LONG=1, SHORT=0
    df["is_long"] = (df["break_dir"].str.lower() == "long").astype(int)

    # Derived predictors
    df["atr_velocity"] = df["atr_20"] / df["prev_atr_20"].replace(0, np.nan)
    df["asia_mid"]     = (df["session_asia_high"] + df["session_asia_low"]) / 2
    df["london_mid"]   = (df["session_london_high"] + df["session_london_low"]) / 2
    # Asia position: positive = closed upper half of Asia range (long bias?)
    df["asia_position"] = (df["asia_mid"] - df["session_asia_low"]) / (
        df["session_asia_high"] - df["session_asia_low"] + 1e-9
    ) - 0.5  # centered at 0; positive = upper half
    # London vs Asia mid: positive = London sitting above Asia midpoint
    df["london_vs_asia"] = df["london_mid"] - df["asia_mid"]
    # prev_break_dir encoded
    df["prev_is_long"] = (df["prev_break_dir"].str.lower() == "long").astype(float)

    return df


def test_continuous(series: pd.Series, target: pd.Series, label: str):
    """t-test: do longs and shorts have different mean predictor values?"""
    clean = pd.DataFrame({"x": series, "y": target}).dropna()
    if len(clean) < 20:
        return label, np.nan, len(clean), "too few"
    longs  = clean.loc[clean["y"] == 1, "x"]
    shorts = clean.loc[clean["y"] == 0, "x"]
    t, p   = stats.ttest_ind(longs, shorts)
    direction = "LONG higher" if longs.mean() > shorts.mean() else "SHORT higher"
    desc = f"long_mean={longs.mean():.3f}  short_mean={shorts.mean():.3f}  ({direction})"
    return label, p, len(clean), desc


def test_categorical(series: pd.Series, target: pd.Series, label: str):
    """Chi-squared test for categorical predictor vs direction."""
    clean = pd.DataFrame({"x": series, "y": target}).dropna()
    if len(clean) < 20:
        return label, np.nan, len(clean), "too few"
    ct  = pd.crosstab(clean["x"], clean["y"])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    # Show breakdown
    pct_long = clean.groupby("x")["y"].mean().to_dict()
    desc = "  ".join(f"{k}:{v:.0%}L" for k, v in sorted(pct_long.items()))
    return label, p, len(clean), desc


def main():
    con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
    df  = load_data(con)
    con.close()

    n = len(df)
    n_long  = df["is_long"].sum()
    n_short = n - n_long
    print(f"\n{'='*80}")
    print(f"DIRECTION PREDICTION  --  {INSTRUMENT} {SESSION}  G4+  prev=WIN")
    print(f"{'='*80}")
    print(f"N = {n} days  |  LONG breaks: {n_long} ({n_long/n:.1%})  |  SHORT breaks: {n_short} ({n_short/n:.1%})")
    print(f"Baseline WR if always bet LONG: {n_long/n:.1%}")
    print()

    tests = []

    # 1. prev_break_dir (categorical: LONG / SHORT)
    tests.append(test_categorical(df["prev_break_dir"], df["is_long"], "prev_break_dir"))

    # 2. gap_open_points (continuous)
    tests.append(test_continuous(df["gap_open_points"], df["is_long"], "gap_open_points"))

    # 3. rsi_14_at_0900 (continuous)
    tests.append(test_continuous(df["rsi_14_at_0900"], df["is_long"], "rsi_14_at_0900"))

    # 4. atr_velocity (continuous)
    tests.append(test_continuous(df["atr_velocity"], df["is_long"], "atr_velocity"))

    # 5. asia_position (continuous: upper vs lower half of Asia range)
    tests.append(test_continuous(df["asia_position"], df["is_long"], "asia_position"))

    # 6. london_vs_asia (continuous)
    tests.append(test_continuous(df["london_vs_asia"], df["is_long"], "london_vs_asia"))

    # 7. day_of_week (categorical)
    tests.append(test_categorical(df["day_of_week"], df["is_long"], "day_of_week"))

    # 8. us_dst (categorical: 0=winter, 1=summer)
    tests.append(test_categorical(df["us_dst"].astype(str), df["is_long"], "us_dst"))

    # ── BH correction ─────────────────────────────────────────────────────────
    valid = [(lbl, p, n_, desc) for lbl, p, n_, desc in tests if not np.isnan(p)]
    labels_v, pvals_v, ns_v, descs_v = zip(*valid) if valid else ([], [], [], [])

    print(f"{'Predictor':<25}  {'p-raw':>8}  {'p-BH':>8}  {'N':>5}  Description")
    print("-" * 90)

    if pvals_v:
        p_adj  = false_discovery_control(list(pvals_v), method="bh")
        reject = [p <= 0.10 for p in p_adj]
        for lbl, p_raw, p_bh, rej, n_, desc in zip(labels_v, pvals_v, p_adj, reject, ns_v, descs_v):
            sig = "*** BH-SIG" if rej else ("** " if p_raw < 0.05 else "   ")
            print(f"  {lbl:<23}  {p_raw:>8.4f}  {p_bh:>8.4f}  {n_:>5}  {desc}  {sig}")

    # ── Per-day breakdown for strongest predictors ─────────────────────────────
    print()
    print("=" * 80)
    print("BREAKDOWN: prev_break_dir × actual direction")
    print("=" * 80)
    ct = pd.crosstab(df["prev_break_dir"], df["break_dir"], margins=True)
    print(ct)
    pct = pd.crosstab(df["prev_break_dir"], df["break_dir"], normalize="index")
    print("\nRow %:")
    print(pct.round(3))

    print()
    print("=" * 80)
    print("BREAKDOWN: day_of_week × actual direction")
    print("=" * 80)
    dow_map = {0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri"}
    df["dow_name"] = df["day_of_week"].map(dow_map)
    ct2 = pd.crosstab(df["dow_name"], df["break_dir"])
    pct2 = pd.crosstab(df["dow_name"], df["break_dir"], normalize="index")
    print(ct2)
    print("\nRow %:")
    print(pct2.round(3))

    print()
    print("=" * 80)
    print("OUTCOME BY DIRECTION (E0 baseline -- from inside entry audit)")
    print("Using prev=WIN directional P&L from inside entry simulation:")
    print("  Long fills  avgR = +0.536R")
    print("  Short fills avgR = +0.054R")
    print()
    print("If any predictor can call direction correctly 60%+ of time:")
    pred_acc = 0.60
    exp_r = pred_acc * 0.536 + (1 - pred_acc) * 0.054 - (1 - pred_acc) * 0.536 - pred_acc * 0.054
    # Simpler: E[R | one-sided correct 60%] vs random OCO
    # If we go LONG only when predictor says long:
    # 60% of time we're right → +0.536R
    # 40% of time we're wrong → fills short instead = we skip it (one-sided limit not filled on wrong side)
    # Actually with one-sided limit: only fills when predicted side fires
    # Hit rate = 60% of prev=WIN days (that subset) fill our limit
    # avgR on those = 0.536R
    # Miss rate = 40% → 0R (limit never fills, no trade)
    # So expected R per day = 0.60 * 0.536 = +0.322R per prev=WIN day
    # vs current OCO (all directions): ~0.40R blended
    print(f"  One-sided LONG (60% accuracy): 0.60 × 0.536R = +{0.60*0.536:.3f}R per prev=WIN day")
    print(f"  One-sided LONG (70% accuracy): 0.70 × 0.536R = +{0.70*0.536:.3f}R per prev=WIN day")
    print(f"  Random OCO (no predictor):     50% × 0.536R + 50% × 0.054R = +{0.5*0.536+0.5*0.054:.3f}R")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    bh_sigs = [lbl for lbl, p_bh, rej, n_, desc in
               zip(labels_v, p_adj, reject, ns_v, descs_v) if rej]
    if bh_sigs:
        print(f"BH SURVIVORS: {', '.join(bh_sigs)}")
        print("ACTION: Direction predictor found — test as one-sided limit filter")
    else:
        print("NO BH SURVIVORS — direction on prev=WIN days is not predictable with these features.")
        print("ACTION: Stick with OCO entry. Do not attempt one-sided directional bet.")


if __name__ == "__main__":
    main()
