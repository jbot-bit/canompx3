"""L6 MNQ_US_DATA_1000 2026 breakdown — is the negative OOS noise or structural?

PR #52: L6 is only 2026-negative deployed lane (unfiltered ExpR -0.034R,
N=68). Full-sample IS t=+3.20 (strong). Tests whether 2026 is noise at
the given sample size or a structural break.

Canonical truth only: orb_outcomes + daily_features (triple-joined).
Read-only. No production code touched.
"""

from __future__ import annotations

import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
HOLDOUT = pd.Timestamp("2026-01-01")
rng = np.random.default_rng(42)


def _welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a.values, b.values, equal_var=False)
    return float(t), float(p)


def main() -> None:
    print("=" * 80)
    print(f"L6 US_DATA_1000 2026 BREAKDOWN  (ran {pd.Timestamp.now('UTC')})")
    print("=" * 80)

    q = """
    SELECT o.trading_day, o.pnl_r,
           CASE WHEN o.pnl_r > 0 THEN 1 ELSE 0 END AS win,
           d.is_nfp_day, d.is_opex_day, d.is_friday, d.is_monday, d.is_tuesday,
           d.day_of_week, d.atr_vel_regime, d.garch_forecast_vol_pct,
           d.atr_20_pct, d.prev_day_range
    FROM orb_outcomes o
    JOIN daily_features d
      ON o.trading_day = d.trading_day
     AND o.symbol = d.symbol
     AND o.orb_minutes = d.orb_minutes
    WHERE o.symbol = 'MNQ'
      AND o.orb_label = 'US_DATA_1000'
      AND o.orb_minutes = 5
      AND o.entry_model = 'E2'
      AND o.rr_target = 1.5
      AND o.confirm_bars = 1
      AND o.pnl_r IS NOT NULL
    ORDER BY o.trading_day
    """
    df = DB.execute(q).fetchdf()
    df["year"] = pd.to_datetime(df["trading_day"]).dt.year
    df["month"] = pd.to_datetime(df["trading_day"]).dt.month
    df["day_of_month"] = pd.to_datetime(df["trading_day"]).dt.day
    is_df = df[df.trading_day < HOLDOUT].copy()
    oos_df = df[df.trading_day >= HOLDOUT].copy()

    print(f"\nUniverse n={len(df)}  IS n={len(is_df)}  2026 n={len(oos_df)}")
    print(f"IS  ExpR={is_df.pnl_r.mean():+.4f}  t={stats.ttest_1samp(is_df.pnl_r.values, 0.0)[0]:+.2f}")
    print(f"OOS ExpR={oos_df.pnl_r.mean():+.4f}  t={stats.ttest_1samp(oos_df.pnl_r.values, 0.0)[0]:+.2f}")

    # =============================================================
    # STEP 1 — Noise check: bootstrap from IS, draw size N=68
    # =============================================================
    print("\n" + "-" * 80)
    print("STEP 1 — BOOTSTRAP NULL: 10,000 draws of size 68 from IS")
    print("-" * 80)
    obs_oos_mean = oos_df.pnl_r.mean()
    n_oos = len(oos_df)
    is_pnl = is_df.pnl_r.values
    n_trials = 10_000
    boot_means = np.array([
        rng.choice(is_pnl, size=n_oos, replace=True).mean()
        for _ in range(n_trials)
    ])
    # two-sided p-value: fraction of bootstrap means as extreme or more extreme
    # vs IS mean, measuring whether observed OOS is unusual given IS distribution
    is_mean = is_df.pnl_r.mean()
    one_sided_p = (boot_means <= obs_oos_mean).mean()
    print(f"  IS mean={is_mean:+.4f}  observed 2026 mean={obs_oos_mean:+.4f}")
    print(f"  Bootstrap p(mean ≤ observed | H0 = IS distribution) = {one_sided_p:.4f}")
    print(f"  Bootstrap 5th/25th/50th/75th/95th pct: "
          f"{np.percentile(boot_means, [5, 25, 50, 75, 95])}")
    if one_sided_p > 0.10:
        noise_verdict = "NOISE_CONSISTENT (obs within 10-90 pct of null)"
    elif one_sided_p < 0.01:
        noise_verdict = "STRUCTURAL_SIGNAL (obs < 1 pct of null)"
    else:
        noise_verdict = "BORDERLINE (obs in 1-10 pct tail)"
    print(f"  Noise verdict: {noise_verdict}")

    # =============================================================
    # STEP 2 — Per-year IS t-tests vs 2026
    # =============================================================
    print("\n" + "-" * 80)
    print("STEP 2 — PER-YEAR COMPARISON — is 2026 worse than any IS year?")
    print("-" * 80)
    print(f"  {'Year':5s} {'N':>4s} {'ExpR':>9s} {'t vs 0':>8s} {'p':>7s} "
          f"{'Welch vs 2026 t':>16s} {'p':>7s}")
    for y, grp in df.groupby("year"):
        t0, p0 = stats.ttest_1samp(grp.pnl_r.values, 0.0)
        if y == 2026:
            print(f"  {y:5d} {len(grp):>4d} {grp.pnl_r.mean():>+9.4f} {float(t0):>+8.2f} {float(p0):>7.4f} "
                  f"{'—':>16s} {'—':>7s}")
            continue
        t_w, p_w = _welch(grp.pnl_r, oos_df.pnl_r)
        print(f"  {y:5d} {len(grp):>4d} {grp.pnl_r.mean():>+9.4f} {float(t0):>+8.2f} {float(p0):>7.4f} "
              f"{t_w:>+16.2f} {p_w:>7.4f}")

    # =============================================================
    # STEP 3 — Calendar decomposition
    # =============================================================
    print("\n" + "-" * 80)
    print("STEP 3 — CALENDAR DECOMPOSITION (NFP, OPEX, day-of-week)")
    print("-" * 80)
    print(f"  {'Bucket':28s}  {'IS':>28s}  {'2026':>28s}")
    for tag, col, val in [
        ("NFP days", "is_nfp_day", True),
        ("non-NFP days", "is_nfp_day", False),
        ("OPEX days", "is_opex_day", True),
        ("non-OPEX days", "is_opex_day", False),
        ("Monday", "is_monday", True),
        ("Tuesday", "is_tuesday", True),
        ("Friday", "is_friday", True),
    ]:
        sub_is = is_df[is_df[col] == val]
        sub_oos = oos_df[oos_df[col] == val]
        is_str = f"n={len(sub_is):>4d} ExpR={sub_is.pnl_r.mean():+.4f}" if len(sub_is) else "n=0"
        if len(sub_oos):
            oos_str = f"n={len(sub_oos):>3d} ExpR={sub_oos.pnl_r.mean():+.4f}"
        else:
            oos_str = "n=0"
        print(f"  {tag:28s}  {is_str:>28s}  {oos_str:>28s}")

    # CPI-adjacent (day_of_month 10-15)
    cpi_is = is_df[(is_df.day_of_month >= 10) & (is_df.day_of_month <= 15)]
    cpi_oos = oos_df[(oos_df.day_of_month >= 10) & (oos_df.day_of_month <= 15)]
    print(f"  {'day_of_month 10-15 (CPI adj)':28s}  "
          f"n={len(cpi_is):>4d} ExpR={cpi_is.pnl_r.mean():+.4f}  "
          f"n={len(cpi_oos):>3d} ExpR={cpi_oos.pnl_r.mean():+.4f}")

    # =============================================================
    # STEP 4 — Volatility regime split
    # =============================================================
    print("\n" + "-" * 80)
    print("STEP 4 — VOLATILITY REGIME — does 2026 cluster in low-vol bins?")
    print("-" * 80)
    # atr_20_pct quartiles
    print(f"  atr_20_pct quartile breakdown (higher = higher recent vol vs 252d)")
    print(f"  {'Bin':10s} {'IS n':>5s} {'IS ExpR':>10s} {'2026 n':>7s} {'2026 ExpR':>10s} {'2026 %':>7s}")
    for lo, hi, tag in [(0, 25, "Q1"), (25, 50, "Q2"), (50, 75, "Q3"), (75, 101, "Q4")]:
        sub_is = is_df[(is_df.atr_20_pct >= lo) & (is_df.atr_20_pct < hi)]
        sub_oos = oos_df[(oos_df.atr_20_pct >= lo) & (oos_df.atr_20_pct < hi)]
        oos_pct = len(sub_oos) / len(oos_df) * 100 if len(oos_df) else float("nan")
        is_expr = sub_is.pnl_r.mean() if len(sub_is) else float("nan")
        oos_expr = sub_oos.pnl_r.mean() if len(sub_oos) else float("nan")
        print(f"  {tag:10s} {len(sub_is):>5d} {is_expr:>+10.4f} {len(sub_oos):>7d} {oos_expr:>+10.4f} {oos_pct:>6.1f}%")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"Noise check: {noise_verdict}")
    print(f"Bootstrap p one-sided (obs ≤ null): {one_sided_p:.4f}")


if __name__ == "__main__":
    main()
