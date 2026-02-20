#!/usr/bin/env python3
"""
research_avoid_crosscheck.py — Cross-Instrument AVOID Signal Validation

QUESTION:
    The Contracting×Neutral ATR/compression signal was discovered on MES 1000
    (avgR=-0.368R, N=101, BH-sig, 5/5 years negative). Is this a universal
    market-structure effect, or MES-specific?

APPROACH:
    Run the same Contracting×Neutral test across all major validated sessions:
    MGC 0900/1000/1800, MNQ 0900/1000/1100/1800, MES 1000 (anchor).

    Also scan all 9 vel×comp combinations for each session — looking for
    any other anomalies that pop up sideways.

    Apply a SINGLE BH FDR correction across all tests from all sessions to
    account for the full multiple comparisons budget.

SIGNAL DEFINITION (no look-ahead):
    - ATR Velocity: atr_20 / 5-day prior avg. <0.95 = Contracting, >1.05 = Expanding
    - ORB Compression z-score: rolling 20-day z-score of (orb_size / atr_20).
      <-0.5 = Compressed, >+0.5 = Expanded, else Neutral
    - Both computed from prior days only (ROWS BETWEEN X PRECEDING AND 1 PRECEDING)

MECHANISM:
    De-volatilizing market (contracting ATR) + no tension (neutral compression) =
    neither momentum fuel nor coiled energy behind the break. Should be
    instrument-agnostic if the mechanism is real.
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

# Representative validated configs for each instrument × session
# Chosen from MEMORY.md / TRADING_RULES.md validated findings
CONFIGS = [
    # (instrument, orb_label, rr, cb, model,  label)
    ("MES", "1000", 3.0, 1, "E0", "MES_1000_E0 [ANCHOR]"),
    ("MGC", "0900", 2.5, 2, "E1", "MGC_0900_E1"),
    ("MGC", "0900", 2.5, 1, "E0", "MGC_0900_E0"),
    ("MGC", "1000", 3.0, 1, "E0", "MGC_1000_E0"),
    ("MGC", "1000", 2.5, 2, "E1", "MGC_1000_E1"),
    ("MGC", "1800", 2.5, 1, "E0", "MGC_1800_E0"),
    ("MNQ", "0900", 2.5, 2, "E0", "MNQ_0900_E0"),
    ("MNQ", "1000", 3.0, 1, "E0", "MNQ_1000_E0"),
    ("MNQ", "1000", 3.0, 2, "E1", "MNQ_1000_E1"),
    ("MNQ", "1100", 2.0, 2, "E1", "MNQ_1100_E1"),
    ("MNQ", "1800", 2.5, 2, "E0", "MNQ_1800_E0"),
]


# ── Statistical helpers ───────────────────────────────────────────────────────

def ttest_1s(arr, mu=0.0):
    a = np.array(arr, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) < 10:
        return len(a), float("nan"), float("nan"), float("nan"), float("nan")
    t, p = stats.ttest_1samp(a, mu)
    return len(a), float(a.mean()), float((a > 0).mean()), float(t), float(p)


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


# ── SQL builder ───────────────────────────────────────────────────────────────

def build_sql(instrument: str, orb_label: str, rr: float, cb: int, model: str) -> str:
    """Build CTE query parameterized by instrument and session."""
    size_col = f"orb_{orb_label}_size"
    dir_col  = f"orb_{orb_label}_break_dir"

    return f"""
WITH base AS (
    SELECT
        trading_day,
        symbol,
        atr_20,
        {size_col}   AS orb_size,
        {dir_col}    AS break_dir,
        gap_open_points,
        us_dst
    FROM daily_features
    WHERE symbol      = '{instrument}'
      AND orb_minutes = 5
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
        CASE WHEN atr_prior_count < 5                               THEN NULL
             WHEN atr_20 / NULLIF(atr_5d_avg, 0.0) > 1.05          THEN 'Expanding'
             WHEN atr_20 / NULLIF(atr_5d_avg, 0.0) < 0.95          THEN 'Contracting'
             ELSE                                                         'Stable'
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
    f.us_dst,
    f.atr_20,
    f.atr_vel_ratio,
    f.atr_vel_regime,
    f.orb_size,
    f.orb_atr_ratio,
    f.compression_z,
    f.compression_tier,
    o.rr_target,
    o.confirm_bars,
    o.entry_model,
    o.outcome,
    o.pnl_r
FROM compression_tier f
INNER JOIN orb_outcomes o
    ON  o.trading_day  = f.trading_day
    AND o.symbol       = f.symbol
    AND o.orb_label    = '{orb_label}'
    AND o.orb_minutes  = 5
    AND o.pnl_r        IS NOT NULL
    AND o.rr_target    = {rr}
    AND o.confirm_bars = {cb}
    AND o.entry_model  = '{model}'
ORDER BY f.trading_day
"""


def load(instrument, orb_label, rr, cb, model):
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        df = con.execute(build_sql(instrument, orb_label, rr, cb, model)).df()
    except Exception as e:
        print(f"  ERROR: {e}")
        df = pd.DataFrame()
    finally:
        con.close()
    return df


# ── Per-session analysis ───────────────────────────────────────────────────────

def analyse_session(df: pd.DataFrame, config_label: str) -> list[dict]:
    """Returns records for global BH pool. Prints detailed breakdown."""
    records = []
    n_base, avg_base, wr_base, t_base, p_base = ttest_1s(df["pnl_r"])
    print(f"  Baseline: N={n_base}, avgR={avg_base:+.3f}, WR={wr_base:.1%}, p={p_base:.4f}")

    records.append(dict(config=config_label, layer="baseline", group="all",
                        n=n_base, avg_r=avg_base, wr=wr_base, t=t_base, p=p_base))

    # All 9 vel×comp combinations
    for vel in ["Expanding", "Stable", "Contracting"]:
        for comp in ["Compressed", "Neutral", "Expanded"]:
            sub = df[
                (df["atr_vel_regime"] == vel) & (df["compression_tier"] == comp)
            ]["pnl_r"]
            n, avg, wr, t, p = ttest_1s(sub)
            group = f"{vel}×{comp}"
            records.append(dict(config=config_label, layer="vel_x_comp", group=group,
                                n=n, avg_r=avg, wr=wr, t=t, p=p))
            star = "  <-- AVOID?" if (not np.isnan(avg) and avg < -0.20 and n >= 30) else ""
            star = "  <-- BOOST?" if (not np.isnan(avg) and avg > 0.20 and n >= 30) else star
            if not np.isnan(avg):
                print(f"    {group:<25}  N={n:>4}  avgR={avg:>+7.3f}  WR={wr:>5.1%}  "
                      f"t={t:>6.2f}  p={p:.4f}{star}")
            else:
                print(f"    {group:<25}  N={n:>4}  (skip - too few)")

    return records


def year_by_year(df: pd.DataFrame, vel: str, comp: str, label: str):
    """Year-by-year stability for a specific vel×comp combination."""
    years = sorted(df["year"].dropna().unique())
    results = []
    for yr in years:
        yr_df = df[df["year"] == yr].reset_index(drop=True)
        mask = (yr_df["atr_vel_regime"] == vel) & (yr_df["compression_tier"] == comp)
        sub = yr_df[mask]["pnl_r"]
        if len(sub) < 3:
            continue
        n, avg, wr, t, p = ttest_1s(sub)
        results.append((int(yr), n, avg, wr))
    return results


def run():
    print()
    print("=" * 72)
    print("CROSS-INSTRUMENT AVOID SIGNAL — ATR Velocity × ORB Compression")
    print("=" * 72)
    print(f"Configs: {len(CONFIGS)} instrument×session combinations")
    print(f"Signal: Contracting ATR (vel<0.95) × Neutral compression (z in [-0.5, 0.5])")
    print(f"BH FDR: Single correction across ALL tests (q=0.10)")
    print()

    all_records = []
    session_dfs = {}

    # ── Pass 1: Load and analyse each session ─────────────────────────────────
    for instrument, orb_label, rr, cb, model, label in CONFIGS:
        print(f"\n{'─' * 72}")
        print(f"  {label}  (rr={rr}, cb={cb}, {model})")
        print(f"{'─' * 72}")

        df = load(instrument, orb_label, rr, cb, model)
        if df.empty or len(df) < 30:
            print(f"  SKIP — insufficient data (N={len(df)})")
            continue

        print(f"  N={len(df)} trades  |  "
              f"{df['trading_day'].min().date()} → {df['trading_day'].max().date()}")

        records = analyse_session(df, label)
        all_records.extend(records)
        session_dfs[label] = df

    # ── Pass 2: Global BH correction ──────────────────────────────────────────
    valid = [(i, r) for i, r in enumerate(all_records) if not np.isnan(r.get("p", float("nan")))]
    p_vals = [r["p"] for _, r in valid]
    rejected_idx = bh(p_vals)

    print(f"\n\n{'=' * 72}")
    print(f"GLOBAL BH FDR (q=0.10) — {len(p_vals)} tests across {len(CONFIGS)} configs")
    print("=" * 72)
    print(f"  Survivors (BH-sig): {len(rejected_idx)}")
    print()

    # Group survivors by type
    avoid_survivors = []
    boost_survivors = []
    other_survivors = []

    for local_i, (orig_i, r) in enumerate(valid):
        if local_i not in rejected_idx:
            continue
        if r["layer"] == "baseline":
            continue  # baseline significance is just config quality, not interesting
        avg = r["avg_r"]
        if not np.isnan(avg):
            if avg < -0.10:
                avoid_survivors.append(r)
            elif avg > 0.10:
                boost_survivors.append(r)
            else:
                other_survivors.append(r)

    if avoid_survivors:
        print("  AVOID signals (BH-sig, avgR < -0.10):")
        for r in sorted(avoid_survivors, key=lambda x: x["avg_r"]):
            print(f"    [{r['config']:<22}]  {r['group']:<25}  "
                  f"N={r['n']:>4}  avgR={r['avg_r']:>+7.3f}  WR={r['wr']:.1%}  p={r['p']:.4f}")

    if boost_survivors:
        print()
        print("  BOOST signals (BH-sig, avgR > +0.10):")
        for r in sorted(boost_survivors, key=lambda x: -x["avg_r"]):
            print(f"    [{r['config']:<22}]  {r['group']:<25}  "
                  f"N={r['n']:>4}  avgR={r['avg_r']:>+7.3f}  WR={r['wr']:.1%}  p={r['p']:.4f}")

    if other_survivors:
        print()
        print("  Other BH-sig (comparison tests etc.):")
        for r in other_survivors:
            avg_s = f"{r['avg_r']:+.3f}" if not np.isnan(r["avg_r"]) else "  —"
            print(f"    [{r['config']:<22}]  {r['group']:<25}  "
                  f"N={r['n']:>4}  avgR={avg_s}  p={r['p']:.4f}")

    # ── Pass 3: Year-by-year for Contracting×Neutral across all sessions ───────
    print(f"\n\n{'=' * 72}")
    print("YEAR-BY-YEAR — Contracting×Neutral AVOID signal per session")
    print("=" * 72)
    print(f"  {'Config':<22}  {'Yrs -ve':>7}  {'Yrs tot':>7}  {'Hit%':>6}  "
          f"{'Avg avoidR':>10}  {'Years detail'}")
    print(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*10}")

    summary_rows = []
    for label, df in session_dfs.items():
        yby = year_by_year(df, "Contracting", "Neutral", label)
        if not yby:
            continue
        n_neg = sum(1 for _, _, avg, _ in yby if avg < 0)
        n_tot = len(yby)
        hit_pct = n_neg / n_tot if n_tot else 0
        avg_r_all = np.mean([avg for _, _, avg, _ in yby])
        yr_detail = "  ".join(
            f"{yr}:{avg:+.2f}({'−' if avg < 0 else '+'})"
            for yr, _, avg, _ in yby
        )
        print(f"  {label:<22}  {n_neg:>7}  {n_tot:>7}  {hit_pct:>6.0%}  "
              f"{avg_r_all:>+10.3f}  {yr_detail}")
        summary_rows.append(dict(config=label, neg_years=n_neg, total_years=n_tot,
                                 hit_pct=hit_pct, avg_avoid_r=avg_r_all))

    # ── Pass 4: Scan for other consistent anomalies across sessions ────────────
    print(f"\n\n{'=' * 72}")
    print("ANOMALY SCAN — vel×comp combos that are consistently bad/good")
    print("(≥4 configs tested, ≥60% directional consistency, |avgR|>0.15)")
    print("=" * 72)

    # Collect per-combo results across sessions
    combo_results: dict[str, list[float]] = {}
    for r in all_records:
        if r["layer"] != "vel_x_comp":
            continue
        if np.isnan(r["avg_r"]) or r["n"] < 20:
            continue
        combo = r["group"]
        combo_results.setdefault(combo, []).append(r["avg_r"])

    print(f"\n  {'Combo':<25}  {'N_sessions':>10}  {'Pct -ve':>8}  "
          f"{'Pct +ve':>8}  {'Med avgR':>9}  {'Verdict'}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*9}")

    for combo, avg_rs in sorted(combo_results.items(), key=lambda x: np.median(x[1])):
        n_sess = len(avg_rs)
        if n_sess < 3:
            continue
        n_neg = sum(1 for r in avg_rs if r < 0)
        n_pos = sum(1 for r in avg_rs if r > 0)
        pct_neg = n_neg / n_sess
        pct_pos = n_pos / n_sess
        med_r = np.median(avg_rs)

        if abs(med_r) < 0.10 and max(pct_neg, pct_pos) < 0.70:
            continue  # boring

        verdict = ""
        if pct_neg >= 0.70 and med_r < -0.10:
            verdict = "<-- CONSISTENT AVOID"
        elif pct_pos >= 0.70 and med_r > 0.10:
            verdict = "<-- CONSISTENT BOOST"
        elif pct_neg >= 0.60 and med_r < -0.15:
            verdict = "  (weak avoid)"
        elif pct_pos >= 0.60 and med_r > 0.15:
            verdict = "  (weak boost)"

        print(f"  {combo:<25}  {n_sess:>10}  {pct_neg:>8.0%}  {pct_pos:>8.0%}  "
              f"{med_r:>+9.3f}  {verdict}")

    # ── Save output ───────────────────────────────────────────────────────────
    df_out = pd.DataFrame(all_records)
    # Add BH significance flag
    sig_set = set()
    for local_i, (orig_i, _) in enumerate(valid):
        if local_i in rejected_idx:
            sig_set.add(orig_i)
    df_out["bh_sig"] = [i in sig_set for i in range(len(all_records))]
    df_out.to_csv(OUTPUT_DIR / "avoid_crosscheck_all.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "avoid_crosscheck_yearly.csv", index=False
    )

    print(f"\nOutput: {OUTPUT_DIR}/avoid_crosscheck_*.csv")
    print()


if __name__ == "__main__":
    run()
