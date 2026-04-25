#!/usr/bin/env python3
"""
research_mes_compressed_spring.py -- MES Compressed Spring AVOID Revalidation

PURPOSE:
    Revalidate contracting ATR AVOID signal for MES using:
    - E1/E2 entry models (E0 is PURGED)
    - Event-based session names (not old fixed-clock names)
    - All available MES data (~6 years, 2019-2025)

    Previous MES finding used E0 + old session "1000". That validation is STALE.
    MGC TOKYO_OPEN AVOID is confirmed (10/10 years). MNQ has NO signal (0/169 BH).
    MES needs fresh evidence with active entry models.

SIGNAL DEFINITION (no look-ahead):
    - ATR Velocity: atr_20 / 5-day prior avg. <0.95 = Contracting, >1.05 = Expanding
    - ORB Compression z-score: rolling 20-day z of (orb_size / atr_20).
      <-0.5 = Compressed, >+0.5 = Expanded, else Neutral
    - Both computed from prior days only (ROWS BETWEEN X PRECEDING AND 1 PRECEDING)

CONFIGS TESTED:
    Primary sessions: TOKYO_OPEN, NYSE_OPEN, SINGAPORE_OPEN (strongest MES sessions)
    Entry models: E1 CB1 RR3.0, E2 CB1 RR3.0
    Apertures: O5 (primary), O15 (cross-check)
    Context: All MES sessions with E2 CB1 RR3.0 O5

BH FDR: Single correction across ALL tests (q=0.10)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import duckdb

from pipeline.paths import GOLD_DB_PATH

DB_PATH = str(GOLD_DB_PATH)
OUTPUT_DIR = PROJECT_ROOT / "research" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -- Statistical helpers -------------------------------------------------------


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


def bh_fdr(p_values: list, q: float = 0.10) -> set:
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


def fmt(label, n, avg_r, wr, t, p, sig=""):
    if np.isnan(avg_r):
        return f"  {label:<35} N<10 (skip)"
    s = "*** BH-SIG" if sig else ""
    return f"  {label:<35} N={n:>4}  avgR={avg_r:>+7.3f}  WR={wr:>5.1%}  t={t:>6.2f}  p={p:>7.4f}  {s}"


SEP = "-" * 72


# -- SQL builder ---------------------------------------------------------------


def build_sql(instrument: str, orb_label: str, rr: float, cb: int, model: str, orb_minutes: int = 5) -> str:
    size_col = f"orb_{orb_label}_size"
    dir_col = f"orb_{orb_label}_break_dir"

    return f"""
WITH base AS (
    SELECT
        trading_day, symbol, atr_20,
        {size_col}  AS orb_size,
        {dir_col}   AS break_dir
    FROM daily_features
    WHERE symbol      = '{instrument}'
      AND orb_minutes = {orb_minutes}
      AND atr_20      IS NOT NULL
      AND {size_col}  IS NOT NULL
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
        orb_size / NULLIF(atr_20, 0.0) AS orb_atr_ratio,
        AVG(orb_size / NULLIF(atr_20, 0.0)) OVER (
            PARTITION BY symbol ORDER BY trading_day
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS ratio_20d_avg,
        STDDEV_POP(orb_size / NULLIF(atr_20, 0.0)) OVER (
            PARTITION BY symbol ORDER BY trading_day
            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
        ) AS ratio_20d_std
    FROM atr_velocity
),

compression_tier AS (
    SELECT *,
        CASE WHEN ratio_20d_std IS NULL OR ratio_20d_std = 0 THEN NULL
             ELSE (orb_atr_ratio - ratio_20d_avg) / ratio_20d_std
        END AS compression_z,
        CASE WHEN ratio_20d_std IS NULL OR ratio_20d_std = 0     THEN NULL
             WHEN (orb_atr_ratio - ratio_20d_avg)
                  / NULLIF(ratio_20d_std, 0.0) < -0.5            THEN 'Compressed'
             WHEN (orb_atr_ratio - ratio_20d_avg)
                  / NULLIF(ratio_20d_std, 0.0) >  0.5            THEN 'Expanded'
             ELSE                                                       'Neutral'
        END AS compression_tier
    FROM orb_compression
)

SELECT
    f.trading_day,
    YEAR(f.trading_day) AS year,
    f.atr_20,
    f.atr_vel_ratio,
    f.atr_vel_regime,
    f.orb_size,
    f.break_dir,
    f.orb_atr_ratio,
    f.compression_z,
    f.compression_tier,
    o.rr_target,
    o.confirm_bars,
    o.entry_model,
    o.outcome,
    o.pnl_r,
    o.mfe_r,
    o.mae_r
FROM compression_tier f
INNER JOIN orb_outcomes o
    ON  o.trading_day  = f.trading_day
    AND o.symbol       = f.symbol
    AND o.orb_label    = '{orb_label}'
    AND o.orb_minutes  = {orb_minutes}
    AND o.pnl_r        IS NOT NULL
    AND o.rr_target    = {rr}
    AND o.confirm_bars = {cb}
    AND o.entry_model  = '{model}'
ORDER BY f.trading_day
"""


def load(instrument, orb_label, rr, cb, model, orb_minutes=5):
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        df = con.execute(build_sql(instrument, orb_label, rr, cb, model, orb_minutes)).df()
    except Exception as e:
        print(f"  ERROR loading {instrument} {orb_label}: {e}")
        df = pd.DataFrame()
    finally:
        con.close()
    return df


# -- Analysis layers -----------------------------------------------------------


def analyse(df: pd.DataFrame, config_label: str) -> list[dict]:
    """Full analysis: ATR velocity, compression, 9-combo interaction, year-by-year."""
    records = []
    N = len(df)

    # L0: Baseline
    n, avg, wr, t, p = ttest_1s(df["pnl_r"])
    print(f"\n  Baseline ({config_label}): N={n}, avgR={avg:+.3f}, WR={wr:.1%}, p={p:.4f}")
    records.append(dict(config=config_label, layer="L0_baseline", group="all", n=n, avg_r=avg, wr=wr, t=t, p=p))

    # L1: ATR Velocity alone
    print(f"\n{SEP}")
    print(f"L1: ATR Velocity regime -> pnl_r  [{config_label}]")
    print(SEP)
    for regime in ["Expanding", "Stable", "Contracting"]:
        sub = df[df["atr_vel_regime"] == regime]["pnl_r"]
        n, avg, wr, t, p = ttest_1s(sub)
        records.append(dict(config=config_label, layer="L1_atr_vel", group=regime, n=n, avg_r=avg, wr=wr, t=t, p=p))
        print(fmt(f"  {regime}", n, avg, wr, t, p))

    # Expanding vs Contracting direct comparison
    t_ec, p_ec = ttest_2s(
        df[df["atr_vel_regime"] == "Expanding"]["pnl_r"],
        df[df["atr_vel_regime"] == "Contracting"]["pnl_r"],
    )
    print(f"  Expanding vs Contracting: t={t_ec:.2f}, p={p_ec:.4f}")
    records.append(
        dict(
            config=config_label,
            layer="L1_exp_vs_con",
            group="vs",
            n=N,
            avg_r=float("nan"),
            wr=float("nan"),
            t=t_ec,
            p=p_ec,
        )
    )

    # L2: ORB Compression alone
    print(f"\n{SEP}")
    print(f"L2: ORB Compression tier -> pnl_r  [{config_label}]")
    print(SEP)
    for tier in ["Compressed", "Neutral", "Expanded"]:
        sub = df[df["compression_tier"] == tier]["pnl_r"]
        n, avg, wr, t, p = ttest_1s(sub)
        records.append(dict(config=config_label, layer="L2_compression", group=tier, n=n, avg_r=avg, wr=wr, t=t, p=p))
        print(fmt(f"  {tier}", n, avg, wr, t, p))

    # L3: ATR Velocity x Compression (9 combos) -- the main test
    print(f"\n{SEP}")
    print(f"L3: ATR Velocity x ORB Compression  [{config_label}]")
    print(SEP)
    for vel in ["Expanding", "Stable", "Contracting"]:
        for comp in ["Compressed", "Neutral", "Expanded"]:
            sub = df[(df["atr_vel_regime"] == vel) & (df["compression_tier"] == comp)]["pnl_r"]
            n, avg, wr, t, p = ttest_1s(sub)
            group = f"{vel} x {comp}"
            records.append(
                dict(config=config_label, layer="L3_vel_x_comp", group=group, n=n, avg_r=avg, wr=wr, t=t, p=p)
            )
            tag = ""
            if not np.isnan(avg):
                if avg < -0.15 and n >= 20:
                    tag = " <-- AVOID?"
                elif avg > 0.15 and n >= 20:
                    tag = " <-- BOOST?"
            print(fmt(f"  {vel[:3]}x{comp[:4]}", n, avg, wr, t, p) + tag)

    # L4: Contracting-only (any compression) vs Rest
    print(f"\n{SEP}")
    print(f"L4: Contracting ATR (any compression) vs Rest  [{config_label}]")
    print(SEP)
    con_mask = df["atr_vel_regime"] == "Contracting"
    rest_mask = ~con_mask & df["atr_vel_regime"].notna()
    con_r = df[con_mask]["pnl_r"]
    rest_r = df[rest_mask]["pnl_r"]
    n_c, avg_c, wr_c, t_c, p_c = ttest_1s(con_r)
    n_r, avg_r_rest, wr_r, t_r, p_r = ttest_1s(rest_r)
    t_diff, p_diff = ttest_2s(con_r, rest_r)
    records.append(
        dict(
            config=config_label,
            layer="L4_contracting_all",
            group="Contracting",
            n=n_c,
            avg_r=avg_c,
            wr=wr_c,
            t=t_c,
            p=p_c,
        )
    )
    records.append(
        dict(config=config_label, layer="L4_rest", group="Rest", n=n_r, avg_r=avg_r_rest, wr=wr_r, t=t_r, p=p_r)
    )
    records.append(
        dict(
            config=config_label,
            layer="L4_con_vs_rest",
            group="vs",
            n=n_c + n_r,
            avg_r=float("nan"),
            wr=float("nan"),
            t=t_diff,
            p=p_diff,
        )
    )
    print(fmt("  Contracting (all comp)", n_c, avg_c, wr_c, t_c, p_c))
    print(fmt("  Rest (Expanding+Stable)", n_r, avg_r_rest, wr_r, t_r, p_r))
    print(f"  Contracting vs Rest: t={t_diff:.2f}, p={p_diff:.4f}")

    return records


def year_by_year(df: pd.DataFrame, label: str, vel_filter: str = "Contracting"):
    """Year-by-year stability for contracting ATR AVOID signal."""
    years = sorted(df["year"].dropna().unique())
    results = []
    print(f"\n{SEP}")
    print(f"YEAR-BY-YEAR: {vel_filter} ATR AVOID  [{label}]")
    print(SEP)
    print(f"  {'Year':<6} {'Avoid avgR':>10} {'N_av':>5}  {'Other avgR':>10} {'N_ot':>5}")

    neg_years = 0
    valid_years = 0
    for yr in years:
        yr_df = df[df["year"] == yr].reset_index(drop=True)
        avoid_mask = yr_df["atr_vel_regime"] == vel_filter
        av = yr_df[avoid_mask]["pnl_r"]
        ot = yr_df[~avoid_mask & yr_df["atr_vel_regime"].notna()]["pnl_r"]
        if len(av) < 3:
            continue
        valid_years += 1
        n_a, avg_a, wr_a, _, _ = ttest_1s(av)
        n_o, avg_o, wr_o, _, _ = ttest_1s(ot)
        if np.isnan(avg_a):
            avg_a = av.mean()
        if avg_a < 0:
            neg_years += 1
        marker = " -" if avg_a < 0 else " +"
        results.append((int(yr), n_a if not np.isnan(n_a) else len(av), avg_a))
        avg_o_s = f"{avg_o:>+10.3f}" if not np.isnan(avg_o) else "       N/A"
        print(f"  {int(yr):<6} {avg_a:>+10.3f} {len(av):>5}{marker}  {avg_o_s} {len(ot):>5}")

    pct = neg_years / valid_years if valid_years else 0
    print(f"  Negative years: {neg_years}/{valid_years} ({pct:.0%})")
    return results, neg_years, valid_years


def sensitivity(df: pd.DataFrame, label: str) -> list[dict]:
    """Test ATR velocity threshold robustness."""
    records = []
    print(f"\n{SEP}")
    print(f"SENSITIVITY: ATR velocity threshold  [{label}]")
    print(SEP)
    for vel_upper in [0.90, 0.92, 0.93, 0.95, 0.97, 0.98, 1.00]:
        sub = df[df["atr_vel_ratio"] < vel_upper]["pnl_r"]
        n, avg, wr, t, p = ttest_1s(sub)
        records.append(
            dict(config=label, layer="sensitivity", group=f"vel<{vel_upper}", n=n, avg_r=avg, wr=wr, t=t, p=p)
        )
        print(fmt(f"  vel < {vel_upper}", n, avg, wr, t, p))
    return records


# -- Main execution -------------------------------------------------------------

# Primary configs: MES strongest sessions with E1 and E2
PRIMARY_CONFIGS = [
    # (instrument, orb_label, rr, cb, model, orb_min, label)
    ("MES", "TOKYO_OPEN", 3.0, 1, "E2", 5, "MES_TOKYO_E2_CB1_RR3_O5 [PRIMARY]"),
    ("MES", "TOKYO_OPEN", 3.0, 1, "E1", 5, "MES_TOKYO_E1_CB1_RR3_O5 [XCHECK]"),
    ("MES", "NYSE_OPEN", 3.0, 1, "E2", 5, "MES_NYSE_E2_CB1_RR3_O5 [PRIMARY]"),
    ("MES", "NYSE_OPEN", 3.0, 1, "E1", 5, "MES_NYSE_E1_CB1_RR3_O5 [XCHECK]"),
    ("MES", "SINGAPORE_OPEN", 3.0, 1, "E2", 5, "MES_SING_E2_CB1_RR3_O5 [PRIMARY]"),
    ("MES", "SINGAPORE_OPEN", 3.0, 1, "E1", 5, "MES_SING_E1_CB1_RR3_O5 [XCHECK]"),
    # O15 cross-check on strongest session
    ("MES", "TOKYO_OPEN", 3.0, 1, "E2", 15, "MES_TOKYO_E2_CB1_RR3_O15 [XCHECK]"),
    ("MES", "NYSE_OPEN", 3.0, 1, "E2", 15, "MES_NYSE_E2_CB1_RR3_O15 [XCHECK]"),
]

# Cross-session context: all remaining MES sessions at E2 CB1 RR3.0 O5
CONTEXT_SESSIONS = [
    "LONDON_METALS",
    "CME_REOPEN",
    "US_DATA_830",
    "US_DATA_1000",
    "COMEX_SETTLE",
    "BRISBANE_1025",
    "CME_PRECLOSE",
    "NYSE_CLOSE",
]


def run():
    print()
    print("=" * 72)
    print("MES COMPRESSED SPRING AVOID -- Revalidation with E1/E2")
    print("Event-based sessions | ~6 years (2019-2025)")
    print("=" * 72)

    all_records = []
    yby_summaries = []

    # -- PART 1: Primary sessions + cross-checks --------------------------------
    print("\n" + "=" * 72)
    print("PART 1: Primary Sessions + Cross-checks")
    print("=" * 72)

    for instrument, orb_label, rr, cb, model, orb_min, label in PRIMARY_CONFIGS:
        print(f"\n{'=' * 72}")
        print(f"  {label}")
        print(f"{'=' * 72}")

        df = load(instrument, orb_label, rr, cb, model, orb_min)
        if df.empty or len(df) < 30:
            print(f"  SKIP -- insufficient data (N={len(df)})")
            continue

        print(f"  Loaded: {len(df)} trade-rows")
        print(f"  Date range: {df['trading_day'].min()} to {df['trading_day'].max()}")
        yrs = sorted(df["year"].dropna().unique())
        print(f"  Years: {[int(y) for y in yrs]}")

        # ATR regime distribution
        for r in ["Expanding", "Stable", "Contracting"]:
            n_r = (df["atr_vel_regime"] == r).sum()
            print(f"    {r:<13}: {n_r:>4} ({n_r / len(df):.0%})")

        records = analyse(df, label)
        all_records.extend(records)

        yby_results, neg_yrs, valid_yrs = year_by_year(df, label)
        yby_summaries.append(
            dict(
                config=label,
                neg_years=neg_yrs,
                valid_years=valid_yrs,
                pct_neg=neg_yrs / valid_yrs if valid_yrs else 0,
                years=yby_results,
            )
        )

        sens_records = sensitivity(df, label)
        all_records.extend(sens_records)

    # -- PART 2: Cross-session context -------------------------------------------
    print("\n\n" + "=" * 72)
    print("PART 2: CROSS-SESSION CONTEXT -- All MES sessions at E2 CB1 RR3.0 O5")
    print("=" * 72)

    for session in CONTEXT_SESSIONS:
        label = f"MES_{session}_E2_CB1_RR3_O5"
        print(f"\n{SEP}")
        print(f"  {label}")
        print(SEP)

        df = load("MES", session, 3.0, 1, "E2", 5)
        if df.empty or len(df) < 30:
            print(f"  SKIP -- insufficient data (N={len(df)})")
            continue

        print(f"  N={len(df)}  |  {df['trading_day'].min().date()} to {df['trading_day'].max().date()}")

        # Baseline
        n_base, avg_base, wr_base, t_base, p_base = ttest_1s(df["pnl_r"])
        print(f"  Baseline: N={n_base}, avgR={avg_base:+.3f}, WR={wr_base:.1%}")
        all_records.append(
            dict(
                config=label, layer="L0_baseline", group="all", n=n_base, avg_r=avg_base, wr=wr_base, t=t_base, p=p_base
            )
        )

        # 9 combos
        for vel in ["Expanding", "Stable", "Contracting"]:
            for comp in ["Compressed", "Neutral", "Expanded"]:
                sub = df[(df["atr_vel_regime"] == vel) & (df["compression_tier"] == comp)]["pnl_r"]
                n, avg, wr, t, p = ttest_1s(sub)
                group = f"{vel} x {comp}"
                all_records.append(
                    dict(config=label, layer="L3_vel_x_comp", group=group, n=n, avg_r=avg, wr=wr, t=t, p=p)
                )
                tag = ""
                if not np.isnan(avg):
                    if avg < -0.15 and n >= 20:
                        tag = " <-- AVOID?"
                    elif avg > 0.15 and n >= 20:
                        tag = " <-- BOOST?"
                if not np.isnan(avg):
                    print(f"    {group:<25}  N={n:>4}  avgR={avg:>+7.3f}  WR={wr:>5.1%}  t={t:>6.2f}  p={p:.4f}{tag}")
                else:
                    print(f"    {group:<25}  N={n:>4}  (skip)")

        # Contracting-only aggregate
        con_mask = df["atr_vel_regime"] == "Contracting"
        con_r = df[con_mask]["pnl_r"]
        n_c, avg_c, wr_c, t_c, p_c = ttest_1s(con_r)
        all_records.append(
            dict(
                config=label, layer="L4_contracting_all", group="Contracting", n=n_c, avg_r=avg_c, wr=wr_c, t=t_c, p=p_c
            )
        )
        if not np.isnan(avg_c):
            print(f"    ** Contracting (all): N={n_c}, avgR={avg_c:+.3f}, WR={wr_c:.1%}, p={p_c:.4f}")

        # Year-by-year
        yby_results, neg_yrs, valid_yrs = year_by_year(df, label)
        yby_summaries.append(
            dict(
                config=label,
                neg_years=neg_yrs,
                valid_years=valid_yrs,
                pct_neg=neg_yrs / valid_yrs if valid_yrs else 0,
                years=yby_results,
            )
        )

    # -- PART 3: Global BH FDR -------------------------------------------------
    print("\n\n" + "=" * 72)
    print("GLOBAL BH FDR (q=0.10)")
    print("=" * 72)

    valid = [(i, r) for i, r in enumerate(all_records) if not np.isnan(r.get("p", float("nan")))]
    p_vals = [r["p"] for _, r in valid]
    rejected_idx = bh_fdr(p_vals)

    print(f"  Total tests: {len(p_vals)}")
    print(f"  BH survivors: {len(rejected_idx)}")

    avoid_survivors = []
    boost_survivors = []
    other_survivors = []

    for local_i, (orig_i, r) in enumerate(valid):
        if local_i not in rejected_idx:
            continue
        if r["layer"] == "L0_baseline":
            continue
        avg = r.get("avg_r", float("nan"))
        if not np.isnan(avg):
            if avg < -0.10:
                avoid_survivors.append(r)
            elif avg > 0.10:
                boost_survivors.append(r)
            else:
                other_survivors.append(r)
        else:
            other_survivors.append(r)

    if avoid_survivors:
        print("\n  AVOID signals (BH-sig, avgR < -0.10):")
        for r in sorted(avoid_survivors, key=lambda x: x.get("avg_r", 0)):
            print(
                f"    [{r['config']:<45}]  {r['group']:<25}  "
                f"N={r['n']:>4}  avgR={r['avg_r']:>+7.3f}  WR={r['wr']:.1%}  p={r['p']:.4f}"
            )
    else:
        print("\n  No AVOID signals survived BH FDR.")

    if boost_survivors:
        print("\n  BOOST signals (BH-sig, avgR > +0.10):")
        for r in sorted(boost_survivors, key=lambda x: -x.get("avg_r", 0)):
            print(
                f"    [{r['config']:<45}]  {r['group']:<25}  "
                f"N={r['n']:>4}  avgR={r['avg_r']:>+7.3f}  WR={r['wr']:.1%}  p={r['p']:.4f}"
            )

    if other_survivors:
        print("\n  Other BH-sig (comparison tests):")
        for r in other_survivors:
            avg_s = f"{r['avg_r']:+.3f}" if not np.isnan(r.get("avg_r", float("nan"))) else "  ---"
            print(f"    [{r['config']:<45}]  {r['group']:<25}  p={r['p']:.4f}  avgR={avg_s}")

    # -- PART 4: Year-by-year summary -------------------------------------------
    print("\n\n" + "=" * 72)
    print("YEAR-BY-YEAR SUMMARY: Contracting ATR AVOID per session")
    print("=" * 72)
    print(f"  {'Config':<50} {'Neg':>4} {'Tot':>4} {'Hit%':>6}")
    for s in yby_summaries:
        print(f"  {s['config']:<50} {s['neg_years']:>4} {s['valid_years']:>4} {s['pct_neg']:>6.0%}")

    # -- PART 5: Anomaly scan ---------------------------------------------------
    print("\n\n" + "=" * 72)
    print("ANOMALY SCAN: vel x comp combos consistent across MES sessions")
    print("(>=4 sessions tested, >=60% directional consistency, |avgR|>0.10)")
    print("=" * 72)

    combo_results: dict[str, list[float]] = {}
    for r in all_records:
        if r["layer"] != "L3_vel_x_comp":
            continue
        if np.isnan(r.get("avg_r", float("nan"))) or r["n"] < 15:
            continue
        combo = r["group"]
        combo_results.setdefault(combo, []).append(r["avg_r"])

    print(f"\n  {'Combo':<25}  {'Sessions':>8}  {'% neg':>6}  {'% pos':>6}  {'Med avgR':>9}  {'Verdict'}")

    for combo, avg_rs in sorted(combo_results.items(), key=lambda x: np.median(x[1])):
        n_sess = len(avg_rs)
        if n_sess < 3:
            continue
        n_neg = sum(1 for r in avg_rs if r < 0)
        n_pos = sum(1 for r in avg_rs if r > 0)
        pct_neg = n_neg / n_sess
        pct_pos = n_pos / n_sess
        med_r = float(np.median(avg_rs))

        verdict = ""
        if pct_neg >= 0.70 and med_r < -0.10:
            verdict = "<-- CONSISTENT AVOID"
        elif pct_pos >= 0.70 and med_r > 0.10:
            verdict = "<-- CONSISTENT BOOST"
        elif pct_neg >= 0.60 and med_r < -0.10:
            verdict = "(weak avoid)"
        elif pct_pos >= 0.60 and med_r > 0.10:
            verdict = "(weak boost)"

        print(f"  {combo:<25}  {n_sess:>8}  {pct_neg:>6.0%}  {pct_pos:>6.0%}  {med_r:>+9.3f}  {verdict}")

    # -- PART 6: Honest summary ------------------------------------------------
    print("\n\n" + "=" * 72)
    print("HONEST SUMMARY")
    print("=" * 72)

    n_avoid = len(avoid_survivors)
    n_boost = len(boost_survivors)

    if n_avoid > 0:
        print(f"\n  SURVIVED BH FDR: {n_avoid} AVOID signal(s)")
        for r in avoid_survivors:
            print(
                f"    {r['config']} | {r['group']} | N={r['n']} | "
                f"avgR={r['avg_r']:+.3f} | WR={r['wr']:.1%} | p={r['p']:.4f}"
            )
    else:
        print("\n  NO AVOID signals survived BH FDR at q=0.10")

    if n_boost > 0:
        print(f"\n  BONUS: {n_boost} BOOST signal(s) survived")

    # Classification per primary session
    for session_name in ["TOKYO_OPEN", "NYSE_OPEN", "SINGAPORE_OPEN"]:
        sess_avoids = [r for r in avoid_survivors if session_name in r.get("config", "")]
        sess_yby = [s for s in yby_summaries if session_name in s["config"] and "PRIMARY" in s["config"]]

        if sess_avoids and sess_yby:
            s = sess_yby[0]
            if s["valid_years"] >= 5 and s["pct_neg"] >= 0.80:
                classification = "CONFIRMED (>=5 yrs, >=80% negative)"
            elif s["valid_years"] >= 4 and s["pct_neg"] >= 0.75:
                classification = "PRELIMINARY (>=4 yrs, >=75% negative)"
            elif s["valid_years"] >= 3 and s["pct_neg"] >= 0.67:
                classification = "REGIME (>=3 yrs, >=67% negative)"
            else:
                classification = "WATCH (insufficient stability)"
        elif sess_avoids:
            classification = "UNCLASSIFIED (no YBY data)"
        else:
            classification = "NO SIGNAL"

        print(f"\n  {session_name} AVOID classification: {classification}")

    print()
    print("  MANDATORY DISCLOSURES:")
    print("    Instrument: MES (Micro S&P 500)")
    print("    Sessions tested: TOKYO_OPEN, NYSE_OPEN, SINGAPORE_OPEN (primary)")
    print("                   + 8 context sessions")
    print("    Period: all available MES data in gold.db (~6 years)")
    print("    Entry models: E1, E2 (E0 is PURGED)")
    print("    IS/OOS: In-sample only -- NO holdout OOS performed")
    print("    BH FDR applied: YES (q=0.10, single correction across all tests)")
    print(f"    Total tests in FDR pool: {len(p_vals)}")
    print("    Mechanism: Contracting ATR = de-volatilizing market = breaks lack follow-through")
    print("    Prior stale finding: E0 on old '1000' session showed BH-sig AVOID")
    print("    This revalidation: E1/E2 on event-based sessions")

    # -- Save output -----------------------------------------------------------
    df_out = pd.DataFrame(all_records)
    sig_set = set()
    for local_i, (orig_i, _) in enumerate(valid):
        if local_i in rejected_idx:
            sig_set.add(orig_i)
    df_out["bh_sig"] = [i in sig_set for i in range(len(all_records))]
    df_out.to_csv(OUTPUT_DIR / "mes_compressed_spring_all.csv", index=False)

    # Year-by-year summary
    yby_rows = []
    for s in yby_summaries:
        for yr, n_yr, avg_yr in s.get("years", []):
            yby_rows.append(dict(config=s["config"], year=yr, n=n_yr, avg_r=avg_yr))
    if yby_rows:
        pd.DataFrame(yby_rows).to_csv(OUTPUT_DIR / "mes_compressed_spring_yearly.csv", index=False)

    print(f"\n  Output: {OUTPUT_DIR}/mes_compressed_spring_*.csv")
    print()

    return avoid_survivors, boost_survivors


if __name__ == "__main__":
    run()
