"""L2 MNQ_SINGAPORE_OPEN ATR_P50 filter stability audit.

Context: PR #52 found L2 is FULLY filter-dependent (unfiltered IS ExpR=-0.010,
Welch fire-vs-non-fire p=0.002 with Δ=+0.073R). If the ATR_P50 edge has
decayed in recent years, the lane has no remaining edge.

PR #47 described the fire-rate drift (29%→80% erratic) as "rolling-percentile
instability on sparse session data" — but atr_20_pct is instrument-wide
(rolling 252d on all MNQ trading days), not session-specific. So the fire
drift must be either (a) SINGAPORE_OPEN-eligible days sample a biased slice,
or (b) atr_20 distribution has shifted so more days exceed the 252d median.

This audit answers: does the filter's +0.073R IS lift hold in recent years?

Canonical truth only: orb_outcomes and daily_features (triple-joined on
trading_day, symbol, orb_minutes).
Read-only. No production code touched.
"""

from __future__ import annotations

import sys

import duckdb
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from research.filter_utils import filter_signal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
HOLDOUT = pd.Timestamp("2026-01-01")


def _welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a.values, b.values, equal_var=False)
    return float(t), float(p)


def main() -> None:
    print("=" * 80)
    print(f"L2 ATR_P50 STABILITY AUDIT  (ran {pd.Timestamp.now('UTC')})")
    print("=" * 80)

    # L2 universe
    q = """
    SELECT o.trading_day, o.pnl_r,
           CASE WHEN o.pnl_r > 0 THEN 1 ELSE 0 END AS win,
           d.*
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = 'SINGAPORE_OPEN'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.rr_target = 1.5
      AND o.confirm_bars = 1
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = DB.execute(q).fetchdf()
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    print(f"\nL2 universe: n={len(df)}, {df.trading_day.min()} → {df.trading_day.max()}")

    sig = filter_signal(df, "ATR_P50", "SINGAPORE_OPEN")
    df["fire"] = sig

    # All-MNQ comparison universe (for distribution-shift check)
    q_all = """
    SELECT DISTINCT d.trading_day, d.atr_20, d.atr_20_pct
    FROM daily_features d
    WHERE d.symbol = 'MNQ'
      AND d.orb_minutes = 5
      AND d.atr_20_pct IS NOT NULL
    ORDER BY d.trading_day
    """
    all_mnq = DB.execute(q_all).fetchdf()
    all_mnq["year"] = pd.to_datetime(all_mnq["trading_day"]).dt.year
    print(f"All-MNQ calendar n={len(all_mnq)}")

    # =============================================================
    # STEP 1 — Per-year fire rate, delta, Welch p, distribution
    # =============================================================
    print("\n" + "-" * 80)
    print("STEP 1 — PER-YEAR FIRE / LIFT / WELCH DECOMPOSITION")
    print("-" * 80)
    print(f"  {'Year':5s} {'N':>4s} {'fire%':>6s} "
          f"{'atr_pct μ':>9s} {'atr_pct p25':>11s} {'atr_pct p75':>11s} "
          f"{'ExpR_fire':>10s} {'ExpR_nonf':>10s} {'Δ':>8s} "
          f"{'Welch t':>8s} {'p':>7s}")
    year_rows = []
    for y, grp in df.groupby("year"):
        fire = grp[grp["fire"] == 1]["pnl_r"]
        nonf = grp[grp["fire"] == 0]["pnl_r"]
        fire_rate = len(fire) / len(grp)
        fire_expr = fire.mean() if len(fire) > 0 else float("nan")
        nonf_expr = nonf.mean() if len(nonf) > 0 else float("nan")
        delta = fire_expr - nonf_expr
        t, p = _welch(fire, nonf)
        pct_series = grp["atr_20_pct"].dropna()
        year_rows.append({
            "year": int(y), "n": len(grp), "fire_rate": fire_rate,
            "atr_pct_mean": pct_series.mean(),
            "atr_pct_p25": pct_series.quantile(0.25),
            "atr_pct_p75": pct_series.quantile(0.75),
            "fire_expr": fire_expr, "nonf_expr": nonf_expr,
            "delta": delta, "welch_t": t, "welch_p": p,
            "n_fire": len(fire), "n_nonf": len(nonf),
        })
        print(f"  {y:5d} {len(grp):>4d} {fire_rate*100:>5.1f}% "
              f"{pct_series.mean():>9.1f} {pct_series.quantile(0.25):>11.1f} {pct_series.quantile(0.75):>11.1f} "
              f"{fire_expr:>+10.4f} {nonf_expr:>+10.4f} {delta:>+8.4f} "
              f"{t:>+8.2f} {p:>7.4f}")

    # =============================================================
    # STEP 2 — Distribution-shift check (SINGAPORE_OPEN vs all MNQ)
    # =============================================================
    print("\n" + "-" * 80)
    print("STEP 2 — ATR_PCT DISTRIBUTION: SINGAPORE_OPEN-eligible vs ALL-MNQ-calendar")
    print("-" * 80)
    print("  If SINGAPORE_OPEN samples a biased slice, these will diverge.")
    print(f"  {'Year':5s} {'SGO μ':>7s} {'SGO med':>8s} {'ALL μ':>7s} {'ALL med':>8s} "
          f"{'Δμ':>7s} {'Δmed':>7s}")
    for y in sorted(set(year_rows_y := [r['year'] for r in year_rows])):
        sgo_pct = df[df.year == y]["atr_20_pct"].dropna()
        all_pct = all_mnq[all_mnq.year == y]["atr_20_pct"].dropna()
        if len(sgo_pct) == 0 or len(all_pct) == 0:
            continue
        sgo_mean = sgo_pct.mean(); all_mean = all_pct.mean()
        sgo_med = sgo_pct.median(); all_med = all_pct.median()
        print(f"  {y:5d} {sgo_mean:>7.1f} {sgo_med:>8.1f} "
              f"{all_mean:>7.1f} {all_med:>8.1f} "
              f"{sgo_mean-all_mean:>+7.2f} {sgo_med-all_med:>+7.2f}")

    # =============================================================
    # STEP 3 — Early-vs-late IS Welch test
    # =============================================================
    print("\n" + "-" * 80)
    print("STEP 3 — EARLY-vs-LATE IS WELCH TEST")
    print("-" * 80)
    df_is = df[df.trading_day < HOLDOUT].copy()
    median_day = df_is.trading_day.quantile(0.5)
    early = df_is[df_is.trading_day < median_day]
    late = df_is[df_is.trading_day >= median_day]
    print(f"  IS split at {pd.Timestamp(median_day).date()}")
    for label, sub in [("early", early), ("late", late)]:
        fire = sub[sub["fire"] == 1]["pnl_r"]
        nonf = sub[sub["fire"] == 0]["pnl_r"]
        t, p = _welch(fire, nonf)
        delta = fire.mean() - nonf.mean()
        print(f"  {label:5s}: n={len(sub)} (fire={len(fire)}, non={len(nonf)})  "
              f"fire_expr={fire.mean():+.4f} non_expr={nonf.mean():+.4f} "
              f"Δ={delta:+.4f} Welch t={t:+.3f} p={p:.4f}")

    # =============================================================
    # STEP 4 — Rolling 3-year Welch (per calendar year, trailing 3y)
    # =============================================================
    print("\n" + "-" * 80)
    print("STEP 4 — TRAILING 3-YEAR WELCH FIRE-vs-NON-FIRE")
    print("-" * 80)
    print(f"  {'end_year':9s} {'N':>5s} {'fire_expr':>10s} {'non_expr':>10s} "
          f"{'Δ':>8s} {'Welch t':>8s} {'p':>7s}")
    years_sorted = sorted(df.year.unique())
    for end_y in years_sorted:
        if end_y < years_sorted[0] + 2:
            continue
        window = df[(df.year >= end_y - 2) & (df.year <= end_y)]
        fire = window[window["fire"] == 1]["pnl_r"]
        nonf = window[window["fire"] == 0]["pnl_r"]
        if len(fire) < 30 or len(nonf) < 30:
            continue
        t, p = _welch(fire, nonf)
        print(f"  {end_y:>9d} {len(window):>5d} {fire.mean():>+10.4f} {nonf.mean():>+10.4f} "
              f"{fire.mean()-nonf.mean():>+8.4f} {t:>+8.2f} {p:>7.4f}")

    # =============================================================
    # Verdict
    # =============================================================
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    early_p = _welch(early[early["fire"] == 1]["pnl_r"],
                     early[early["fire"] == 0]["pnl_r"])[1]
    late_p = _welch(late[late["fire"] == 1]["pnl_r"],
                    late[late["fire"] == 0]["pnl_r"])[1]
    early_d = (early[early["fire"] == 1]["pnl_r"].mean()
               - early[early["fire"] == 0]["pnl_r"].mean())
    late_d = (late[late["fire"] == 1]["pnl_r"].mean()
              - late[late["fire"] == 0]["pnl_r"].mean())

    # Distribution bias
    sgo_is_pct = df_is["atr_20_pct"].dropna()
    # All MNQ days aligned to same IS range
    mnq_is = all_mnq[(all_mnq.trading_day >= df_is.trading_day.min())
                     & (all_mnq.trading_day < HOLDOUT)]
    ks_stat, ks_p = stats.ks_2samp(sgo_is_pct.values,
                                    mnq_is["atr_20_pct"].dropna().values)

    print(f"Early-half Welch p: {early_p:.4f}  Δ={early_d:+.4f}")
    print(f"Late-half  Welch p: {late_p:.4f}  Δ={late_d:+.4f}")
    print(f"KS test: SINGAPORE_OPEN atr_20_pct vs all-MNQ atr_20_pct  "
          f"D={ks_stat:.4f}  p={ks_p:.4g}")

    if early_p < 0.05 and late_p > 0.10 and late_d < early_d * 0.5:
        verdict = "DECAYING — early filter discriminated, late does not"
    elif early_p < 0.05 and late_p < 0.05:
        verdict = "HOLDING — filter discriminates in both halves"
    elif ks_p < 0.01:
        verdict = "UNSTABLE_DISTRIBUTION — SINGAPORE_OPEN samples biased atr_20_pct slice"
    else:
        verdict = "MIXED — see per-year and rolling-3y detail"

    print(f"\nFinal verdict: {verdict}")


if __name__ == "__main__":
    main()
