#!/usr/bin/env python3
"""
Comprehensive calendar effect analysis on validated ORB strategies.

Tests every plausible calendar signal against validated strategy outcomes
using event-based sessions (E1/E2). Applies BH FDR correction across ALL
tests and checks year-by-year consistency for survivors.

Calendar signals tested:
  1. NFP day (first Friday of month)
  2. OPEX day (third Friday of month)
  3. FOMC day (announcement day + day after)
  4. Day of week (Mon/Tue/Wed/Thu/Fri — each vs rest)
  5. Month-end (last 2 trading days of month)
  6. Month-start (first 2 trading days of month)
  7. Quarter-end (last 2 trading days of quarter)
  8. OPEX week (Mon-Fri of OPEX week)
  9. CPI day (typically ~13th of month, hardcoded from BLS schedule)

Replaces: research/archive/research_bh_calendar.py (stale — E0, old sessions)

Usage:
  python research/research_calendar_effects.py
  python research/research_calendar_effects.py --db-path C:/db/gold.db
"""

import argparse
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================================================================
# Calendar date builders
# =========================================================================

# FOMC announcement dates
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# Last updated: 2026-03-08 (covers through 2026-03-18)
_FOMC_DATES_RAW = [
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18",
]

# CPI release dates (typically around 10th-13th of month)
# Source: https://www.bls.gov/schedule/news_release/cpi.htm
# Last updated: 2026-03-08 (covers through 2026-02-11)
_CPI_DATES_RAW = [
    # 2020
    "2020-01-14", "2020-02-13", "2020-03-11", "2020-04-10", "2020-05-12",
    "2020-06-10", "2020-07-14", "2020-08-12", "2020-09-11", "2020-10-13",
    "2020-11-12", "2020-12-10",
    # 2021
    "2021-01-13", "2021-02-10", "2021-03-10", "2021-04-13", "2021-05-12",
    "2021-06-10", "2021-07-13", "2021-08-11", "2021-09-14", "2021-10-13",
    "2021-11-10", "2021-12-10",
    # 2022
    "2022-01-12", "2022-02-10", "2022-03-10", "2022-04-12", "2022-05-11",
    "2022-06-10", "2022-07-13", "2022-08-10", "2022-09-13", "2022-10-13",
    "2022-11-10", "2022-12-13",
    # 2023
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10",
    "2023-06-13", "2023-07-12", "2023-08-10", "2023-09-13", "2023-10-12",
    "2023-11-14", "2023-12-12",
    # 2024
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15",
    "2024-06-12", "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10",
    "2024-11-13", "2024-12-11",
    # 2025
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13",
    "2025-06-11", "2025-07-15", "2025-08-12", "2025-09-10", "2025-10-14",
    "2025-11-12", "2025-12-10",
    # 2026
    "2026-01-13", "2026-02-11",
]


def _build_fomc_set() -> set[date]:
    """FOMC announcement day + day after."""
    dates = set()
    for d_str in _FOMC_DATES_RAW:
        d = date.fromisoformat(d_str)
        dates.add(d)
        dates.add(d + timedelta(days=1))
    return dates


def _build_cpi_set() -> set[date]:
    """CPI release day."""
    return {date.fromisoformat(d) for d in _CPI_DATES_RAW}


def _is_month_end(td: date, all_trading_days: set[date], window: int = 2) -> bool:
    """True if td is within the last `window` trading days of its month."""
    # Find last day of this month
    if td.month == 12:
        next_month_first = date(td.year + 1, 1, 1)
    else:
        next_month_first = date(td.year, td.month + 1, 1)
    # Count trading days from td to end of month
    count = 0
    check = td
    while check < next_month_first:
        if check in all_trading_days:
            count += 1
        check += timedelta(days=1)
    return count <= window


def _is_month_start(td: date, all_trading_days: set[date], window: int = 2) -> bool:
    """True if td is within the first `window` trading days of its month."""
    first_of_month = date(td.year, td.month, 1)
    count = 0
    check = first_of_month
    while check <= td:
        if check in all_trading_days:
            count += 1
        check += timedelta(days=1)
    return count <= window


def _is_quarter_end(td: date, all_trading_days: set[date], window: int = 2) -> bool:
    """True if td is within the last `window` trading days of a quarter."""
    if td.month not in (3, 6, 9, 12):
        return False
    return _is_month_end(td, all_trading_days, window)


def _opex_week_dates(yr_start: int = 2016, yr_end: int = 2027) -> set[date]:
    """All Mon-Fri of OPEX week (the week containing third Friday)."""
    dates = set()
    for year in range(yr_start, yr_end):
        for month in range(1, 13):
            d = date(year, month, 1)
            friday_count = 0
            while True:
                if d.weekday() == 4:
                    friday_count += 1
                    if friday_count == 3:
                        break
                d += timedelta(days=1)
            # d is the third Friday — get Mon-Fri of that week
            monday = d - timedelta(days=4)
            for i in range(5):
                dates.add(monday + timedelta(days=i))
    return dates


# =========================================================================
# Data loading
# =========================================================================

def load_validated_outcomes(db_path: Path) -> pd.DataFrame:
    """Load all outcomes for validated strategies (E1/E2)."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("""
        SELECT
            o.symbol, o.orb_label, o.entry_model, o.rr_target, o.confirm_bars,
            o.orb_minutes, o.trading_day, o.pnl_r,
            d.is_nfp_day, d.is_opex_day, d.is_friday, d.is_monday, d.is_tuesday,
            d.day_of_week
        FROM validated_setups vs
        JOIN orb_outcomes o
            ON o.symbol = vs.instrument
            AND o.orb_label = vs.orb_label
            AND o.entry_model = vs.entry_model
            AND o.rr_target = vs.rr_target
            AND o.confirm_bars = vs.confirm_bars
            AND o.orb_minutes = vs.orb_minutes
        JOIN daily_features d
            ON d.symbol = o.symbol
            AND d.trading_day = o.trading_day
            AND d.orb_minutes = o.orb_minutes
        WHERE o.pnl_r IS NOT NULL
            AND vs.entry_model IN ('E1', 'E2')
    """).fetchdf()
    con.close()
    return df


# =========================================================================
# Statistical tests
# =========================================================================

def run_test(on_pnl: np.ndarray, off_pnl: np.ndarray) -> dict:
    """Welch t-test between two groups."""
    if len(on_pnl) < 10 or len(off_pnl) < 30:
        return None
    t_stat, p_val = ttest_ind(on_pnl, off_pnl, equal_var=False)
    return {
        "n_on": len(on_pnl),
        "n_off": len(off_pnl),
        "mean_on": float(on_pnl.mean()),
        "mean_off": float(off_pnl.mean()),
        "diff": float(on_pnl.mean() - off_pnl.mean()),
        "wr_on": float((on_pnl > 0).mean()),
        "wr_off": float((off_pnl > 0).mean()),
        "t_stat": float(t_stat),
        "p_raw": float(p_val),
    }


def year_consistency(df_sub: pd.DataFrame, col: str, expected_negative: bool) -> tuple[int, int]:
    """Check how many years agree with the direction."""
    consistent = 0
    total = 0
    for yr in sorted(df_sub["yr"].unique()):
        ys = df_sub[df_sub.yr == yr]
        on = ys[ys[col]]["pnl_r"].values
        off = ys[~ys[col]]["pnl_r"].values
        if len(on) < 5 or len(off) < 20:
            continue
        total += 1
        diff = on.mean() - off.mean()
        if (expected_negative and diff < 0) or (not expected_negative and diff > 0):
            consistent += 1
    return consistent, total


# =========================================================================
# Main
# =========================================================================

def run(db_path: Path):
    t0 = time.time()

    instruments = list(ACTIVE_ORB_INSTRUMENTS)
    print("=" * 80)
    print("  COMPREHENSIVE CALENDAR EFFECT ANALYSIS")
    print(f"  DB: {db_path}")
    print(f"  Instruments: {', '.join(instruments)}")
    print(f"  Entry models: E1, E2 (validated strategies only)")
    print("=" * 80)

    # Load data
    print("\n  Loading validated strategy outcomes...")
    df = load_validated_outcomes(db_path)
    print(f"  Loaded {len(df):,} outcome rows across {df.symbol.nunique()} instruments")
    print(f"  Unique strategies: {df.groupby(['symbol','orb_label','entry_model','rr_target','confirm_bars','orb_minutes']).ngroups}")

    # Add year column
    df["yr"] = pd.to_datetime(df["trading_day"]).dt.year

    # Normalize trading_day to python date for calendar computations
    df["td_date"] = pd.to_datetime(df["trading_day"]).dt.date

    # Build calendar signal lookup on unique dates (fast)
    unique_dates = sorted(df["td_date"].unique())
    all_td_set = set(unique_dates)
    fomc_dates = _build_fomc_set()
    cpi_dates = _build_cpi_set()
    opex_week = _opex_week_dates()

    cal_lookup = {}
    for td in unique_dates:
        cal_lookup[td] = {
            "is_fomc": td in fomc_dates,
            "is_cpi": td in cpi_dates,
            "is_month_end": _is_month_end(td, all_td_set),
            "is_month_start": _is_month_start(td, all_td_set),
            "is_quarter_end": _is_quarter_end(td, all_td_set),
            "is_opex_week": td in opex_week,
        }

    cal_df = pd.DataFrame.from_dict(cal_lookup, orient="index")
    cal_df.index.name = "td_date"
    cal_df = cal_df.reset_index()

    df = df.merge(cal_df, on="td_date", how="left")

    # DOW dummies
    for dow_num, dow_name in [(0, "Monday"), (1, "Tuesday"), (2, "Wednesday"), (3, "Thursday"), (4, "Friday")]:
        df[f"is_{dow_name}"] = df["day_of_week"] == dow_num

    # Verify calendar signal counts
    print("\n  Calendar signal prevalence (unique trading days):")
    day_df = df.drop_duplicates(subset=["trading_day"])
    for sig in ["is_nfp_day", "is_opex_day", "is_fomc", "is_cpi", "is_month_end",
                "is_month_start", "is_quarter_end", "is_opex_week"]:
        n = day_df[sig].sum()
        print(f"    {sig:20s}: {n:>4} days ({n/len(day_df)*100:.1f}%)")
    for dow_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        n = day_df[f"is_{dow_name}"].sum()
        print(f"    is_{dow_name:9s}        : {n:>4} days ({n/len(day_df)*100:.1f}%)")

    # Define all calendar signals to test
    calendar_signals = [
        ("NFP", "is_nfp_day"),
        ("OPEX", "is_opex_day"),
        ("FOMC", "is_fomc"),
        ("CPI", "is_cpi"),
        ("MONTH_END", "is_month_end"),
        ("MONTH_START", "is_month_start"),
        ("QUARTER_END", "is_quarter_end"),
        ("OPEX_WEEK", "is_opex_week"),
        ("Monday", "is_Monday"),
        ("Tuesday", "is_Tuesday"),
        ("Wednesday", "is_Wednesday"),
        ("Thursday", "is_Thursday"),
        ("Friday", "is_Friday"),
    ]

    # Run all tests: signal x instrument x session
    results = []
    sessions = sorted(df.orb_label.unique())

    for sig_name, sig_col in calendar_signals:
        for inst in instruments:
            sub_inst = df[df.symbol == inst]
            for sess in sessions:
                sub = sub_inst[sub_inst.orb_label == sess]
                if len(sub) < 50:
                    continue
                on = sub[sub[sig_col]]["pnl_r"].values
                off = sub[~sub[sig_col]]["pnl_r"].values
                test = run_test(on, off)
                if test is None:
                    continue
                test["signal"] = sig_name
                test["instrument"] = inst
                test["session"] = sess
                results.append(test)

    rdf = pd.DataFrame(results)
    print(f"\n  Total tests: {len(rdf)}")

    # BH FDR correction
    reject, pvals_corr, _, _ = multipletests(rdf["p_raw"].values, alpha=0.10, method="fdr_bh")
    rdf["p_bh"] = pvals_corr
    rdf["fdr_sig"] = reject

    n_raw05 = (rdf.p_raw < 0.05).sum()
    n_bh10 = rdf.fdr_sig.sum()
    n_bh05 = (rdf.p_bh < 0.05).sum()
    print(f"  Raw p<0.05: {n_raw05}  |  BH q<0.10: {n_bh10}  |  BH q<0.05: {n_bh05}")

    # ===================================================================
    # BH FDR survivors with year-by-year consistency
    # ===================================================================
    survivors = rdf[rdf.fdr_sig].sort_values("p_bh").copy()

    print(f"\n{'=' * 100}")
    print(f"  BH FDR SURVIVORS (q=0.10) — {len(survivors)} found")
    print(f"{'=' * 100}")

    if survivors.empty:
        print("  None.")
    else:
        # Add year-by-year consistency
        consistency_results = []
        for idx, row in survivors.iterrows():
            sub = df[(df.symbol == row["instrument"]) & (df.orb_label == row["session"])]
            sig_col = [s[1] for s in calendar_signals if s[0] == row["signal"]][0]
            expected_neg = row["diff"] < 0
            cons, tot = year_consistency(sub, sig_col, expected_neg)
            consistency_results.append({"consistent": cons, "total": tot})

        survivors["yr_consistent"] = [c["consistent"] for c in consistency_results]
        survivors["yr_total"] = [c["total"] for c in consistency_results]
        survivors["yr_pct"] = survivors.apply(
            lambda r: r["yr_consistent"] / r["yr_total"] * 100 if r["yr_total"] > 0 else 0, axis=1
        )

        # Print header
        print(f"\n  {'Signal':12s} {'Inst':4s} {'Session':20s} "
              f"{'Diff':>8s} {'p_bh':>8s} {'N_on':>6s} {'N_off':>7s} "
              f"{'WR_on':>6s} {'WR_off':>7s} {'YrCons':>8s} {'Verdict':>12s}")
        print(f"  {'-' * 105}")

        for _, row in survivors.iterrows():
            direction = "WORSE" if row["diff"] < 0 else "BETTER"
            yr_str = f"{row['yr_consistent']:.0f}/{row['yr_total']:.0f} ({row['yr_pct']:.0f}%)"
            if row["yr_pct"] >= 75:
                verdict = "CONSISTENT"
            elif row["yr_pct"] >= 60:
                verdict = "WEAK"
            else:
                verdict = "NOISE"
            print(f"  {row['signal']:12s} {row['instrument']:4s} {row['session']:20s} "
                  f"{row['diff']:>+8.4f} {row['p_bh']:>8.4f} {row['n_on']:>6d} {row['n_off']:>7d} "
                  f"{row['wr_on']:>6.1%} {row['wr_off']:>7.1%} {yr_str:>8s}   {verdict} {direction}")

    # ===================================================================
    # Actionable summary: only CONSISTENT survivors
    # ===================================================================
    print(f"\n{'=' * 100}")
    print(f"  ACTIONABLE SIGNALS (BH FDR q<0.10 + >=75% year consistency)")
    print(f"{'=' * 100}")

    if not survivors.empty:
        actionable = survivors[survivors.yr_pct >= 75].copy()
        avoid = actionable[actionable["diff"] < 0].sort_values("diff")
        exploit = actionable[actionable["diff"] > 0].sort_values("diff", ascending=False)

        if not avoid.empty:
            print("\n  --- AVOID (skip these days) ---")
            for _, row in avoid.iterrows():
                print(f"    {row['signal']:12s} {row['instrument']:4s} {row['session']:20s} "
                      f"diff={row['diff']:+.4f}R  p_bh={row['p_bh']:.4f}  "
                      f"consistency={row['yr_consistent']:.0f}/{row['yr_total']:.0f}")

        if not exploit.empty:
            print("\n  --- EXPLOIT (trade these days — do NOT skip!) ---")
            for _, row in exploit.iterrows():
                print(f"    {row['signal']:12s} {row['instrument']:4s} {row['session']:20s} "
                      f"diff={row['diff']:+.4f}R  p_bh={row['p_bh']:.4f}  "
                      f"consistency={row['yr_consistent']:.0f}/{row['yr_total']:.0f}")

        # Weak signals
        weak = survivors[(survivors.yr_pct >= 60) & (survivors.yr_pct < 75)]
        if not weak.empty:
            print(f"\n  --- WEAK (60-74% consistency — monitor, don't act) ---")
            for _, row in weak.iterrows():
                direction = "WORSE" if row["diff"] < 0 else "BETTER"
                print(f"    {row['signal']:12s} {row['instrument']:4s} {row['session']:20s} "
                      f"diff={row['diff']:+.4f}R  consistency={row['yr_consistent']:.0f}/{row['yr_total']:.0f}  {direction}")

        noise = survivors[survivors.yr_pct < 60]
        if not noise.empty:
            print(f"\n  --- NOISE (<60% consistency — FDR survivor but not stable) ---")
            print(f"    {len(noise)} signals classified as noise (omitted for brevity)")
    else:
        print("\n  No actionable signals found.")

    # ===================================================================
    # Impact on current blanket skip
    # ===================================================================
    print(f"\n{'=' * 100}")
    print(f"  IMPACT ANALYSIS: Current CALENDAR_SKIP_NFP_OPEX blanket skip")
    print(f"{'=' * 100}")

    for cal_type, cal_col in [("NFP", "is_nfp_day"), ("OPEX", "is_opex_day")]:
        print(f"\n  --- {cal_type} blanket skip ---")
        for inst in instruments:
            sub = df[(df.symbol == inst) & (df[cal_col])]
            if len(sub) == 0:
                continue
            mean_r = sub["pnl_r"].mean()
            n = len(sub)
            non = df[(df.symbol == inst) & (~df[cal_col])]
            diff = mean_r - non["pnl_r"].mean()
            # Check if this inst has any session where skipping is harmful
            inst_survivors = survivors[
                (survivors.instrument == inst) &
                (survivors.signal == cal_type) &
                (survivors["diff"] > 0)
            ] if not survivors.empty else pd.DataFrame()
            harmful_sessions = list(inst_survivors["session"]) if not inst_survivors.empty else []
            print(f"    {inst}: overall {cal_type} diff={diff:+.4f}R  (N={n})")
            if harmful_sessions:
                print(f"         HARMFUL to skip at: {', '.join(harmful_sessions)}")

    # Save results
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    csv_path = out / "calendar_effects_comprehensive.csv"
    rdf.sort_values("p_bh").to_csv(csv_path, index=False)
    print(f"\n  Results saved: {csv_path}")
    print(f"  Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive calendar effect analysis")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    args = parser.parse_args()
    run(args.db_path)
