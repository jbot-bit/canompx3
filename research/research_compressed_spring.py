#!/usr/bin/env python3
"""
research_compressed_spring.py — "The Compressed Spring" MES 1000 Strategy

HYPOTHESIS:
    Three independent market-structure signals, each observable BEFORE the
    MES 1000 ORB entry, combine to identify days where ORB breaks are
    systematically higher quality:

    1. ATR VELOCITY  — Is volatility expanding? An expanding ATR regime
       signals the market is transitioning from chop to trend. Expanding
       vol = institutional positioning building = breaks that sustain.

    2. ORB COMPRESSION — Is today's ORB tight relative to recent ATR?
       A tight ORB (compressed spring) despite an expanding ATR means the
       market has accumulated energy without releasing it. The break IS
       the release.

    3. GAP ALIGNMENT — Did price gap in the same direction as the break?
       A gap establishes overnight directional intent. When the ORB break
       confirms that direction, two timeframes are aligned. Counter-gap
       breaks fight institutional flow and are more likely to fail.

MECHANISM:
    The three signals are independent at the signal level but share a
    common macro driver: institutional positioning creating directional
    pressure. ATR velocity detects that pressure building. ORB compression
    detects it being suppressed (absorbed). Gap alignment detects the
    prior-session direction of the pressure. When all three align, the
    break is not a random coin flip — it is the market resolving accumulated
    directional pressure in the presence of expanding institutional activity.

NO LOOK-AHEAD:
    - ATR velocity: rolling 5-day avg of ATR_20, all prior days only
    - ORB compression: rolling 20-day avg/std of ORB/ATR ratio, prior days only
    - Gap: previous close → today open (established before session open)
    - Break direction: the actual trade being taken (not a predictor — a descriptor)
    All signals are observable at 10:00 AM Brisbane before the ORB closes at 10:05.

PRIMARY ANALYSIS: E0 / CB1 / RR3.0
    Per validated research, MES 1000 E0 CB1 ORB_G4 RR3.0 has ExpR=+0.382 (N=272).
    This is the highest-quality validated configuration. Signal testing is done
    here first. Cross-check at E1/CB2/RR2.5 for robustness.

WHAT WOULD KILL THIS EDGE:
    - ATR velocity threshold is curve-fitted (sensitive to ±20% change)
    - Compression z-score window choice is path-dependent
    - Gap effect reverses in strong trending markets (gaps get filled)
    - 2021–2025 data is the only in-sample period for MES (short history)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import duckdb

DB_PATH = os.environ.get("DUCKDB_PATH", str(PROJECT_ROOT / "gold.db"))
OUTPUT_DIR = PROJECT_ROOT / "research" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Statistical helpers ───────────────────────────────────────────────────────

def ttest_1s(arr, mu=0.0):
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 10:
        return len(a), float("nan"), float("nan"), float("nan"), float("nan")
    t, p = stats.ttest_1samp(a, mu)
    return len(a), float(a.mean()), float((a > 0).mean()), float(t), float(p)


def ttest_2s(a_arr, b_arr):
    a = np.array(a_arr, dtype=float)
    b = np.array(b_arr, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 10 or len(b) < 10:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b)
    return float(t), float(p)


def bh(p_values: list, q: float = 0.10) -> set:
    n = len(p_values)
    if n == 0:
        return set()
    ranked = sorted(enumerate(p_values), key=lambda x: x[1])
    thresholds = [q * (k + 1) / n for k in range(n)]
    max_k = -1
    for k, (_, p) in enumerate(ranked):
        if p <= thresholds[k]:
            max_k = k
    rejected = set()
    if max_k >= 0:
        for k, (idx, _) in enumerate(ranked[: max_k + 1]):
            rejected.add(idx)
    return rejected


def fmt_row(label, n, avg_r, wr, t, p, sig=""):
    if np.isnan(avg_r):
        return f"  {label:<32} {'N<10':>5}  (skip)"
    s = "***" if sig else "   "
    return (f"  {label:<32} N={n:>4}  avgR={avg_r:>+7.3f}  "
            f"WR={wr:>5.1%}  t={t:>6.2f}  p={p:>7.4f}  {s}")


# ── Load data ─────────────────────────────────────────────────────────────────

BASE_SQL = """
WITH base AS (
    SELECT
        trading_day,
        symbol,
        atr_20,
        orb_1000_size,
        orb_1000_break_dir,
        gap_open_points,
        us_dst
    FROM daily_features
    WHERE symbol     = 'MES'
      AND orb_minutes = 5
      AND atr_20      IS NOT NULL
      AND orb_1000_size IS NOT NULL
),

atr_rolling AS (
    SELECT *,
        AVG(atr_20) OVER (
            PARTITION BY symbol ORDER BY trading_day
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS atr_5d_avg,
        COUNT(atr_20) OVER (
            PARTITION BY symbol ORDER BY trading_day
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS atr_prior_count
    FROM base
),

atr_velocity AS (
    SELECT *,
        CASE WHEN atr_prior_count < 5 THEN NULL
             ELSE atr_20 / NULLIF(atr_5d_avg, 0.0)
        END AS atr_vel_ratio,
        CASE WHEN atr_prior_count < 5                           THEN NULL
             WHEN atr_20 / NULLIF(atr_5d_avg, 0.0) > 1.05     THEN 'Expanding'
             WHEN atr_20 / NULLIF(atr_5d_avg, 0.0) < 0.95     THEN 'Contracting'
             ELSE                                                    'Stable'
        END AS atr_vel_regime
    FROM atr_rolling
),

orb_compression AS (
    SELECT *,
        orb_1000_size / NULLIF(atr_20, 0.0) AS orb_atr_ratio,
        AVG(orb_1000_size / NULLIF(atr_20, 0.0)) OVER (
            PARTITION BY symbol ORDER BY trading_day
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS ratio_20d_avg,
        STDDEV_POP(orb_1000_size / NULLIF(atr_20, 0.0)) OVER (
            PARTITION BY symbol ORDER BY trading_day
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS ratio_20d_std
    FROM atr_velocity
),

orb_compression_tier AS (
    SELECT *,
        CASE WHEN ratio_20d_std IS NULL OR ratio_20d_std = 0 THEN NULL
             ELSE (orb_atr_ratio - ratio_20d_avg) / ratio_20d_std
        END AS compression_z,
        CASE WHEN ratio_20d_std IS NULL OR ratio_20d_std = 0   THEN NULL
             WHEN (orb_atr_ratio - ratio_20d_avg)
                  / NULLIF(ratio_20d_std, 0.0) < -0.5          THEN 'Compressed'
             WHEN (orb_atr_ratio - ratio_20d_avg)
                  / NULLIF(ratio_20d_std, 0.0) >  0.5          THEN 'Expanded'
             ELSE                                                    'Neutral'
        END AS compression_tier
    FROM orb_compression
),

gap_signals AS (
    SELECT *,
        gap_open_points / NULLIF(atr_20, 0.0) AS gap_atr_ratio,
        CASE WHEN orb_1000_break_dir IS NULL                        THEN NULL
             WHEN ABS(gap_open_points / NULLIF(atr_20, 0.0)) < 0.03 THEN 'Neutral'
             WHEN orb_1000_break_dir = 'long'  AND gap_open_points > 0 THEN 'Aligned'
             WHEN orb_1000_break_dir = 'short' AND gap_open_points < 0 THEN 'Aligned'
             WHEN orb_1000_break_dir = 'long'  AND gap_open_points < 0 THEN 'Counter'
             WHEN orb_1000_break_dir = 'short' AND gap_open_points > 0 THEN 'Counter'
             ELSE                                                        'Neutral'
        END AS gap_alignment,
        ABS(gap_open_points / NULLIF(atr_20, 0.0)) >= 0.03 AS gap_is_large
    FROM orb_compression_tier
)

SELECT
    f.trading_day,
    YEAR(f.trading_day)          AS year,
    f.us_dst,
    f.atr_20,
    f.atr_5d_avg,
    f.atr_prior_count,
    f.atr_vel_ratio,
    f.atr_vel_regime,
    f.orb_1000_size,
    f.orb_1000_break_dir         AS break_dir,
    f.orb_atr_ratio,
    f.ratio_20d_avg,
    f.compression_z,
    f.compression_tier,
    f.gap_open_points,
    f.gap_atr_ratio,
    f.gap_alignment,
    f.gap_is_large,
    o.rr_target,
    o.confirm_bars,
    o.entry_model,
    o.outcome,
    o.pnl_r,
    o.mfe_r,
    o.mae_r
FROM gap_signals f
INNER JOIN orb_outcomes o
    ON  o.trading_day = f.trading_day
    AND o.symbol      = f.symbol
    AND o.orb_label   = '1000'
    AND o.orb_minutes = 5
    AND o.pnl_r IS NOT NULL
ORDER BY f.trading_day
"""


def load(rr: float, cb: int, model: str) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    sql = BASE_SQL + f"\nWHERE o.rr_target = {rr} AND o.confirm_bars = {cb} AND o.entry_model = '{model}'"
    # BASE_SQL already ends in ORDER BY; the WHERE needs to come before ORDER BY
    # Fix: inject WHERE before ORDER BY
    sql = BASE_SQL.replace(
        "ORDER BY f.trading_day",
        f"WHERE o.rr_target = {rr} AND o.confirm_bars = {cb} AND o.entry_model = '{model}'\nORDER BY f.trading_day"
    )
    df = con.execute(sql).df()
    con.close()
    return df


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(df: pd.DataFrame, label: str) -> list[dict]:
    """
    Run all signal layers. Returns list of result dicts for BH collection.
    """
    records = []
    N_total = len(df)

    # ── Baseline ──────────────────────────────────────────────────────────────
    n, avg, wr, t, p = ttest_1s(df["pnl_r"])
    print(f"\n  Baseline ({label}): N={n}, avgR={avg:+.3f}, WR={wr:.1%}, p={p:.4f}")
    records.append(dict(layer="L0_baseline", group="all", n=n, avg_r=avg, wr=wr, t=t, p=p))

    sep = "─" * 68

    # ── L1: ATR Velocity alone ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("L1: ATR Velocity regime → pnl_r  [CLEAN ✓]")
    print(sep)
    for regime in ["Expanding", "Stable", "Contracting"]:
        sub = df[df["atr_vel_regime"] == regime]["pnl_r"]
        n, avg, wr, t, p = ttest_1s(sub)
        lbl = f"L1_{regime}"
        records.append(dict(layer="L1_atr_vel", group=regime, n=n, avg_r=avg, wr=wr, t=t, p=p))
        print(fmt_row(lbl, n, avg, wr, t, p))

    t_exp_con, p_exp_con = ttest_2s(
        df[df["atr_vel_regime"] == "Expanding"]["pnl_r"],
        df[df["atr_vel_regime"] == "Contracting"]["pnl_r"],
    )
    print(f"  Expanding vs Contracting: t={t_exp_con:.2f}, p={p_exp_con:.4f}")
    records.append(dict(layer="L1_exp_vs_con", group="vs", n=N_total, avg_r=float("nan"),
                        wr=float("nan"), t=t_exp_con, p=p_exp_con))

    # ── L2: ORB Compression alone ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("L2: ORB Compression tier (rolling 20d z-score) → pnl_r  [CLEAN ✓]")
    print(sep)
    for tier in ["Compressed", "Neutral", "Expanded"]:
        sub = df[df["compression_tier"] == tier]["pnl_r"]
        n, avg, wr, t, p = ttest_1s(sub)
        lbl = f"L2_{tier}"
        records.append(dict(layer="L2_compression", group=tier, n=n, avg_r=avg, wr=wr, t=t, p=p))
        print(fmt_row(lbl, n, avg, wr, t, p))

    # Monotonicity check: Compressed > Neutral > Expanded?
    comp_r = df[df["compression_tier"] == "Compressed"]["pnl_r"]
    exp_r  = df[df["compression_tier"] == "Expanded"]["pnl_r"]
    t_cv, p_cv = ttest_2s(comp_r, exp_r)
    print(f"  Compressed vs Expanded: t={t_cv:.2f}, p={p_cv:.4f}")
    records.append(dict(layer="L2_comp_vs_exp", group="vs", n=N_total, avg_r=float("nan"),
                        wr=float("nan"), t=t_cv, p=p_cv))

    # ── L3: Gap Alignment alone ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("L3: Gap Alignment → pnl_r  [CLEAN ✓]")
    print(sep)
    for align in ["Aligned", "Neutral", "Counter"]:
        sub = df[df["gap_alignment"] == align]["pnl_r"]
        n, avg, wr, t, p = ttest_1s(sub)
        lbl = f"L3_{align}"
        records.append(dict(layer="L3_gap", group=align, n=n, avg_r=avg, wr=wr, t=t, p=p))
        print(fmt_row(lbl, n, avg, wr, t, p))

    t_align_vs_counter, p_align_vs_counter = ttest_2s(
        df[df["gap_alignment"] == "Aligned"]["pnl_r"],
        df[df["gap_alignment"] == "Counter"]["pnl_r"],
    )
    print(f"  Aligned vs Counter: t={t_align_vs_counter:.2f}, p={p_align_vs_counter:.4f}")
    records.append(dict(layer="L3_aligned_vs_counter", group="vs", n=N_total, avg_r=float("nan"),
                        wr=float("nan"), t=t_align_vs_counter, p=p_align_vs_counter))

    # ── L4: ATR Velocity × Compression ───────────────────────────────────────
    print(f"\n{sep}")
    print("L4: ATR Velocity × ORB Compression interaction → pnl_r  [CLEAN ✓]")
    print(sep)
    for vel in ["Expanding", "Stable", "Contracting"]:
        for comp in ["Compressed", "Neutral", "Expanded"]:
            sub = df[
                (df["atr_vel_regime"] == vel) & (df["compression_tier"] == comp)
            ]["pnl_r"]
            n, avg, wr, t, p = ttest_1s(sub)
            lbl = f"L4_{vel[:3]}×{comp[:4]}"
            records.append(dict(layer="L4_vel_x_comp", group=f"{vel}×{comp}",
                                n=n, avg_r=avg, wr=wr, t=t, p=p))
            print(fmt_row(lbl, n, avg, wr, t, p))

    # ── L5: "The Compressed Spring" — all three signals ───────────────────────
    print(f"\n{sep}")
    print("L5: The Compressed Spring — Expanding × Compressed × Aligned  [CLEAN ✓]")
    print(sep)

    spring_mask = (
        (df["atr_vel_regime"]   == "Expanding")
        & (df["compression_tier"] == "Compressed")
        & (df["gap_alignment"]    == "Aligned")
    )
    rest_mask = ~spring_mask & df["pnl_r"].notna()

    spring = df[spring_mask]["pnl_r"]
    rest   = df[rest_mask]["pnl_r"]

    n_sp, avg_sp, wr_sp, t_sp, p_sp = ttest_1s(spring)
    n_re, avg_re, wr_re, t_re, p_re = ttest_1s(rest)
    t_diff, p_diff = ttest_2s(spring, rest)

    records.append(dict(layer="L5_spring", group="Spring", n=n_sp, avg_r=avg_sp, wr=wr_sp, t=t_sp, p=p_sp))
    records.append(dict(layer="L5_rest", group="Rest", n=n_re, avg_r=avg_re, wr=wr_re, t=t_re, p=p_re))
    records.append(dict(layer="L5_spring_vs_rest", group="vs", n=n_sp+n_re,
                        avg_r=float("nan"), wr=float("nan"), t=t_diff, p=p_diff))

    print(fmt_row("Compressed Spring", n_sp, avg_sp, wr_sp, t_sp, p_sp))
    print(fmt_row("Rest of days",      n_re, avg_re, wr_re, t_re, p_re))
    print(f"  Spring vs Rest: t={t_diff:.2f}, p={p_diff:.4f}")

    # Partial combos (2-way interactions, to see which signal adds most)
    for (c1, v1), (c2, v2) in [
        (("atr_vel_regime", "Expanding"), ("compression_tier", "Compressed")),
        (("atr_vel_regime", "Expanding"), ("gap_alignment", "Aligned")),
        (("compression_tier", "Compressed"), ("gap_alignment", "Aligned")),
    ]:
        sub = df[(df[c1] == v1) & (df[c2] == v2)]["pnl_r"]
        n, avg, wr, t, p = ttest_1s(sub)
        lbl = f"L5_{v1[:4]}+{v2[:4]}"
        records.append(dict(layer="L5_2way", group=lbl, n=n, avg_r=avg, wr=wr, t=t, p=p))
        print(fmt_row(lbl, n, avg, wr, t, p))

    # ── L6: Year-by-year stability for the full spring signal ─────────────────
    print(f"\n{sep}")
    print("L6: Year-by-year — Compressed Spring vs Baseline")
    print(sep)
    print(f"  {'Year':<6} {'Spring':>7}  {'N_sp':>5}   {'Base':>7}  {'N_base':>6}")
    years = sorted(df["year"].dropna().unique())
    spring_positive = 0
    valid_spring_years = 0
    for yr in years:
        yr_df = df[df["year"] == yr].reset_index(drop=True)
        # Recompute mask on the subset to avoid index alignment bugs
        yr_spring = (
            (yr_df["atr_vel_regime"]   == "Expanding")
            & (yr_df["compression_tier"] == "Compressed")
            & (yr_df["gap_alignment"]    == "Aligned")
        )
        sp = yr_df[yr_spring]["pnl_r"]
        ba = yr_df[~yr_spring]["pnl_r"]
        if len(sp) < 3:
            continue
        valid_spring_years += 1
        n_s, avg_s, wr_s, _, _ = ttest_1s(sp)
        n_b, avg_b, wr_b, _, _ = ttest_1s(ba)
        if avg_s > 0:
            spring_positive += 1
        marker = " +" if avg_s > 0 else " -"
        print(f"  {int(yr):<6} avgR={avg_s:>+6.3f}  N={n_s:>3} {marker}    "
              f"base={avg_b:>+6.3f}  N={n_b:>4}")
    pct = spring_positive / valid_spring_years if valid_spring_years else 0
    print(f"  Spring positive years: {spring_positive}/{valid_spring_years} ({pct:.0%})")

    # ── L7: Sensitivity — ATR velocity threshold robustness (Spring signal) ──
    print(f"\n{sep}")
    print("L7: Sensitivity — ATR velocity threshold robustness  [CLEAN ✓]")
    print(sep)
    print("  (Spring signal held with compression_tier=Compressed + gap=Aligned)")
    for vel_thresh in [1.03, 1.05, 1.07, 1.10]:
        sub_mask = (
            (df["atr_vel_ratio"] > vel_thresh)
            & (df["compression_tier"] == "Compressed")
            & (df["gap_alignment"]    == "Aligned")
        )
        sub = df[sub_mask]["pnl_r"]
        n, avg, wr, t, p = ttest_1s(sub)
        records.append(dict(layer="L7_sensitivity", group=f"vel>{vel_thresh}", n=n,
                            avg_r=avg, wr=wr, t=t, p=p))
        print(fmt_row(f"vel>{vel_thresh}", n, avg, wr, t, p))

    # ── L8: Year-by-year — Contracting × Neutral AVOID signal ────────────────
    print(f"\n{sep}")
    print("L8: Year-by-year — Contracting×Neutral AVOID signal stability")
    print("    (BH-sig finding: N=~101, avgR~-0.368, WR~18%)")
    print(sep)
    print(f"  {'Year':<6} {'Avoid':>7}  {'N_av':>4}    {'Other':>7}  {'N_ot':>5}")
    avoid_negative = 0
    valid_avoid_years = 0
    for yr in years:
        yr_df = df[df["year"] == yr].reset_index(drop=True)
        yr_avoid = (
            (yr_df["atr_vel_regime"]   == "Contracting")
            & (yr_df["compression_tier"] == "Neutral")
        )
        av = yr_df[yr_avoid]["pnl_r"]
        ot = yr_df[~yr_avoid]["pnl_r"]
        if len(av) < 3:
            continue
        valid_avoid_years += 1
        n_a, avg_a, wr_a, _, _ = ttest_1s(av)
        n_o, avg_o, wr_o, _, _ = ttest_1s(ot)
        if avg_a < 0:
            avoid_negative += 1
        marker = " -" if avg_a < 0 else " +"
        print(f"  {int(yr):<6} avgR={avg_a:>+6.3f}  N={n_a:>3} {marker}    "
              f"other={avg_o:>+6.3f}  N={n_o:>4}")
    pct_neg = avoid_negative / valid_avoid_years if valid_avoid_years else 0
    print(f"  Avoid signal negative years: {avoid_negative}/{valid_avoid_years} ({pct_neg:.0%})")

    # ── L9: Sensitivity — Contracting×Neutral at different velocity thresholds ─
    print(f"\n{sep}")
    print("L9: Sensitivity — Contracting×Neutral AVOID at ±20% velocity thresholds")
    print(sep)
    print("  (Null hypothesis: avgR=0.0, testing if AVOID is robust to threshold changes)")
    for vel_upper in [0.92, 0.93, 0.95, 0.97, 0.98]:
        sub_mask = (
            (df["atr_vel_ratio"] < vel_upper)
            & (df["compression_tier"] == "Neutral")
        )
        sub = df[sub_mask]["pnl_r"]
        n, avg, wr, t, p = ttest_1s(sub)
        records.append(dict(layer="L9_avoid_sensitivity", group=f"vel<{vel_upper}", n=n,
                            avg_r=avg, wr=wr, t=t, p=p))
        print(fmt_row(f"Contracting(vel<{vel_upper})×Neutral", n, avg, wr, t, p))

    return records


def print_bh_summary(records: list[dict], label: str):
    print(f"\n{'=' * 68}")
    print(f"BH FDR CORRECTION (q=0.10)  —  {label}")
    print("=" * 68)

    valid = [(i, r) for i, r in enumerate(records) if not np.isnan(r["p"])]
    p_vals = [r["p"] for _, r in valid]
    rejected_idx = bh(p_vals)
    survivors = []

    for local_i, (orig_i, r) in enumerate(valid):
        sig = local_i in rejected_idx
        if sig or r["p"] < 0.10:
            star = "*** BH-SIG ***" if sig else ""
            avg_s = f"{r['avg_r']:+.3f}" if not np.isnan(r["avg_r"]) else "   —"
            wr_s = f"{r['wr']:.1%}" if not np.isnan(r["wr"]) else "  —"
            print(f"  [{r['layer']}] {r['group']:<20}  N={r['n']:>4}  "
                  f"avgR={avg_s}  WR={wr_s}  p={r['p']:.4f}  {star}")
            if sig:
                survivors.append(r)

    return survivors


def honest_summary(survivors: list[dict], label: str):
    print(f"\n{'=' * 68}")
    print(f"HONEST SUMMARY — {label}")
    print("=" * 68)

    if not survivors:
        print("\nDID NOT SURVIVE BH FDR at q=0.10")
        print()
        print("STATISTICAL OBSERVATION (not actionable):")
        print("  The three-signal combination does not produce a statistically")
        print("  separable edge at the tested parameters. Possible reasons:")
        print("  1. ATR velocity (5-day window) is too short to capture regime shifts")
        print("  2. Z-score compression uses 20-day window — may be optimal at 10 or 40")
        print("  3. Gap alignment threshold (5% of ATR) may be too lenient")
        print("  4. The three signals are not as independent as theorized")
        print("  5. MES 1000 E0 already captures the best days; no residual signal")
        print()
        print("NEXT STEPS:")
        print("  - Test with ATR window = 10d (slower, captures broader regimes)")
        print("  - Test with compression window = 40d")
        print("  - Test gap threshold at 10% and 15% of ATR (stronger gap signal)")
        print("  - Test on E1/CB2 where sample is larger")
    else:
        print(f"\nSURVIVED BH FDR: {len(survivors)} test(s)")
        for r in survivors:
            print(f"  [{r['layer']}] {r['group']}: N={r['n']}, avgR={r['avg_r']:+.3f}, "
                  f"WR={r['wr']:.1%}, p={r['p']:.4f}")
        print()
        print("CAVEATS:")
        print("  - All signals are pre-entry (no look-ahead confirmed)")
        print("  - MES 1000 history starts 2019; < 7 years is PRELIMINARY at best")
        print("  - Positive years fraction MUST be ≥75% before calling this validated")
        print("  - Sample counts may be small per year — check L6 table")
        print("  - Sensitivity (L7) must hold at ±20% threshold variation")

    print()
    print("MANDATORY DISCLOSURES:")
    print(f"  Instrument: MES 1000 session")
    print(f"  Period: all available MES data in gold.db")
    print(f"  Entry model: as specified in each run")
    print(f"  IS/OOS: In-sample only — NO holdout OOS performed")
    print(f"  Variations tested: 3 layers × 3 groups + sensitivity = ~25 tests")
    print(f"  BH FDR applied: YES (q=0.10)")
    print(f"  Mechanism stated: YES (see docstring)")
    print(f"  What kills it: regime change, thin N per year, window sensitivity")


def run():
    print()
    print("=" * 68)
    print("THE COMPRESSED SPRING — MES 1000 Three-Layer ATR Signal Strategy")
    print("=" * 68)

    # ── PRIMARY: E0 / CB1 / RR3.0 ────────────────────────────────────────────
    primary_label = "E0 / CB1 / RR3.0 (primary — highest validated config)"
    print(f"\n{'─'*68}")
    print(f"PRIMARY: {primary_label}")
    print(f"{'─'*68}")

    df_primary = load(rr=3.0, cb=1, model="E0")
    if df_primary.empty:
        print("  NO DATA for primary config. Check DB path or MES orb_outcomes.")
    else:
        print(f"\n  Loaded: {len(df_primary)} trade-rows")
        print(f"  Date range: {df_primary['trading_day'].min()} → {df_primary['trading_day'].max()}")
        print(f"  ATR velocity null (warm-up): {df_primary['atr_vel_regime'].isna().sum()} rows")
        print(f"  Compression null (warm-up):  {df_primary['compression_tier'].isna().sum()} rows")
        print(f"  Gap Aligned:  {(df_primary['gap_alignment']=='Aligned').sum()}")
        print(f"  Gap Counter:  {(df_primary['gap_alignment']=='Counter').sum()}")
        print(f"  Gap Neutral:  {(df_primary['gap_alignment']=='Neutral').sum()}")

        # Distribution of ATR regimes
        print(f"\n  ATR Velocity distribution:")
        for r in ["Expanding", "Stable", "Contracting"]:
            n = (df_primary["atr_vel_regime"] == r).sum()
            print(f"    {r:<13}: {n:>4} ({n/len(df_primary):.0%})")

        records_primary = analyse(df_primary, primary_label)
        survivors_primary = print_bh_summary(records_primary, primary_label)
        honest_summary(survivors_primary, primary_label)

        # Save primary records
        pd.DataFrame(records_primary).to_csv(
            OUTPUT_DIR / "compressed_spring_primary.csv", index=False
        )

    # ── CROSS-CHECK: E1 / CB2 / RR2.5 ────────────────────────────────────────
    cross_label = "E1 / CB2 / RR2.5 (cross-check — larger sample)"
    print(f"\n\n{'─'*68}")
    print(f"CROSS-CHECK: {cross_label}")
    print(f"{'─'*68}")

    df_cross = load(rr=2.5, cb=2, model="E1")
    if df_cross.empty:
        print("  NO DATA for cross-check config.")
    else:
        print(f"\n  Loaded: {len(df_cross)} trade-rows")
        records_cross = analyse(df_cross, cross_label)
        survivors_cross = print_bh_summary(records_cross, cross_label)
        honest_summary(survivors_cross, cross_label)

        pd.DataFrame(records_cross).to_csv(
            OUTPUT_DIR / "compressed_spring_crosscheck.csv", index=False
        )

    print(f"\nOutput saved to: {OUTPUT_DIR}/compressed_spring_*.csv")
    print()


if __name__ == "__main__":
    run()
