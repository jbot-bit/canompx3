#!/usr/bin/env python3
"""
Portfolio-level cascade scanner for calendar effects.

Replaces the flat 416-test approach in research_calendar_effects.py with a
3-level cascade (Harvey, Liu & Zhu methodology) that keeps the hypothesis
budget under 30 tests by decomposing only where significance is found.

Key improvement: per-day aggregation before testing. If 20 strategies fire on
the same NFP day, that's 1 observation -- not 20. This gives honest p-values
at the cost of reduced power.

Levels:
  1. Portfolio-wide: pool all instruments/sessions, 13 tests, BH FDR q=0.10
  2. Per-instrument: for rejected signals, 4 tests each, BH FDR q=0.10
  3. Per-session: for rejected instruments, up to 11 tests each, BH FDR q=0.10
     + year-by-year consistency check

Output: research/output/calendar_cascade_rules.json

Usage:
  python research/research_calendar_cascade.py
  python research/research_calendar_cascade.py --db-path C:/db/gold.db
"""

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH

# Reuse calendar date builders and data loader from the flat script
from research.research_calendar_effects import (
    _build_cpi_set,
    _build_fomc_set,
    _is_month_end,
    _is_month_start,
    _is_quarter_end,
    _opex_week_dates,
    load_validated_outcomes,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Minimum sample sizes for t-tests
MIN_ON = 10
MIN_OFF = 30

# Calendar signal definitions: (name, column)
CALENDAR_SIGNALS = [
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


# =========================================================================
# Per-day aggregation
# =========================================================================


def aggregate_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one PnL number per (instrument, session, trading_day).

    This is THE key improvement over the flat script. Without this, 20
    strategies firing on the same day count as 20 "independent" observations,
    inflating N and deflating p-values.
    """
    # Preserve calendar signal columns through aggregation
    signal_cols = [c for c in df.columns if c.startswith("is_")]
    # For signal columns, take first value (they're identical within a group)
    agg_dict = {"pnl_r": "mean"}
    for col in signal_cols:
        agg_dict[col] = "first"
    # Also preserve day_of_week and yr
    if "day_of_week" in df.columns:
        agg_dict["day_of_week"] = "first"
    if "yr" in df.columns:
        agg_dict["yr"] = "first"
    if "td_date" in df.columns:
        agg_dict["td_date"] = "first"

    daily = (
        df.groupby(["symbol", "orb_label", "trading_day"])
        .agg(agg_dict)
        .reset_index()
    )
    return daily


# =========================================================================
# Statistical helpers
# =========================================================================


def welch_ttest(on_pnl: np.ndarray, off_pnl: np.ndarray) -> dict | None:
    """Welch t-test between two groups. Returns None if insufficient samples."""
    if len(on_pnl) < MIN_ON or len(off_pnl) < MIN_OFF:
        return None
    t_stat, p_val = ttest_ind(on_pnl, off_pnl, equal_var=False)
    return {
        "n_on": len(on_pnl),
        "n_off": len(off_pnl),
        "mean_on": float(on_pnl.mean()),
        "mean_off": float(off_pnl.mean()),
        "diff": float(on_pnl.mean() - off_pnl.mean()),
        "t_stat": float(t_stat),
        "p_raw": float(p_val),
    }


def apply_bh_fdr(results: list[dict], q: float = 0.10) -> list[dict]:
    """Apply BH FDR correction to a list of test results."""
    if not results:
        return []
    pvals = np.array([r["p_raw"] for r in results])
    reject, pvals_corr, _, _ = multipletests(pvals, alpha=q, method="fdr_bh")
    for i, r in enumerate(results):
        r["p_bh"] = float(pvals_corr[i])
        r["rejected"] = bool(reject[i])
    return results


def year_consistency_check(
    daily_agg: pd.DataFrame, sig_col: str, expected_negative: bool
) -> tuple[int, int]:
    """Check how many years agree with the overall direction.

    Returns (consistent_years, total_years_with_sufficient_data).
    Requires >= 5 on-days and >= 20 off-days per year.
    """
    consistent = 0
    total = 0
    for yr in sorted(daily_agg["yr"].unique()):
        yr_data = daily_agg[daily_agg["yr"] == yr]
        on = yr_data[yr_data[sig_col]]["pnl_r"].values
        off = yr_data[~yr_data[sig_col]]["pnl_r"].values
        if len(on) < 5 or len(off) < 20:
            continue
        total += 1
        diff = on.mean() - off.mean()
        if (expected_negative and diff < 0) or (not expected_negative and diff > 0):
            consistent += 1
    return consistent, total


# =========================================================================
# Cascade levels
# =========================================================================


def level1_portfolio_wide(
    daily_agg: pd.DataFrame,
) -> tuple[list[dict], list[str]]:
    """Level 1: Pool ALL instruments and sessions, test each calendar signal.

    Returns (results, rejected_signal_names).
    """
    print("\n" + "=" * 80)
    print("  LEVEL 1: PORTFOLIO-WIDE (13 tests)")
    print("=" * 80)

    results = []
    for sig_name, sig_col in CALENDAR_SIGNALS:
        on = daily_agg[daily_agg[sig_col]]["pnl_r"].values
        off = daily_agg[~daily_agg[sig_col]]["pnl_r"].values
        test = welch_ttest(on, off)
        if test is None:
            print(f"    {sig_name:15s} -- SKIPPED (n_on={len(on)}, n_off={len(off)})")
            continue
        test["signal"] = sig_name
        test["sig_col"] = sig_col
        results.append(test)

    results = apply_bh_fdr(results)

    # Print results
    print(f"\n  {'Signal':15s} {'Diff':>8s} {'p_raw':>8s} {'p_bh':>8s} "
          f"{'N_on':>6s} {'N_off':>7s} {'Rejected':>10s}")
    print(f"  {'-' * 70}")

    rejected_signals = []
    for r in sorted(results, key=lambda x: x["p_bh"]):
        status = "REJECTED" if r["rejected"] else "neutral"
        print(f"  {r['signal']:15s} {r['diff']:>+8.4f} {r['p_raw']:>8.4f} "
              f"{r['p_bh']:>8.4f} {r['n_on']:>6d} {r['n_off']:>7d} {status:>10s}")
        if r["rejected"]:
            rejected_signals.append(r["signal"])

    n_rejected = len(rejected_signals)
    n_neutral = len(results) - n_rejected
    print(f"\n  Summary: {n_rejected} REJECTED, {n_neutral} NEUTRAL "
          f"(out of {len(results)} testable)")
    if rejected_signals:
        print(f"  Rejected signals -> Level 2: {', '.join(rejected_signals)}")
    else:
        print("  No signals rejected -> ALL NEUTRAL. Cascade stops here.")

    return results, rejected_signals


def level2_per_instrument(
    daily_agg: pd.DataFrame,
    rejected_signals: list[str],
    instruments: list[str],
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Level 2: For each rejected signal, test per-instrument.

    Returns (results, rejected_pairs) where pairs are (signal, instrument).
    """
    if not rejected_signals:
        return [], []

    # Build signal name -> column mapping
    sig_col_map = {name: col for name, col in CALENDAR_SIGNALS}

    print("\n" + "=" * 80)
    n_tests = len(rejected_signals) * len(instruments)
    print(f"  LEVEL 2: PER-INSTRUMENT ({n_tests} tests max)")
    print("=" * 80)

    results = []
    for sig_name in rejected_signals:
        sig_col = sig_col_map[sig_name]
        for inst in instruments:
            inst_data = daily_agg[daily_agg["symbol"] == inst]
            on = inst_data[inst_data[sig_col]]["pnl_r"].values
            off = inst_data[~inst_data[sig_col]]["pnl_r"].values
            test = welch_ttest(on, off)
            if test is None:
                continue
            test["signal"] = sig_name
            test["sig_col"] = sig_col
            test["instrument"] = inst
            results.append(test)

    results = apply_bh_fdr(results)

    # Print results grouped by signal
    rejected_pairs = []
    for sig_name in rejected_signals:
        sig_results = [r for r in results if r["signal"] == sig_name]
        if not sig_results:
            continue
        print(f"\n  --- {sig_name} ---")
        print(f"  {'Instrument':12s} {'Diff':>8s} {'p_raw':>8s} {'p_bh':>8s} "
              f"{'N_on':>6s} {'N_off':>7s} {'Rejected':>10s}")
        print(f"  {'-' * 60}")
        for r in sorted(sig_results, key=lambda x: x["p_bh"]):
            status = "REJECTED" if r["rejected"] else "neutral"
            print(f"  {r['instrument']:12s} {r['diff']:>+8.4f} {r['p_raw']:>8.4f} "
                  f"{r['p_bh']:>8.4f} {r['n_on']:>6d} {r['n_off']:>7d} {status:>10s}")
            if r["rejected"]:
                rejected_pairs.append((r["signal"], r["instrument"]))

    n_rejected = len(rejected_pairs)
    n_tested = len(results)
    print(f"\n  Summary: {n_rejected} REJECTED, {n_tested - n_rejected} NEUTRAL "
          f"(out of {n_tested} testable)")
    if rejected_pairs:
        pair_strs = [f"{s}x{i}" for s, i in rejected_pairs]
        print(f"  Rejected pairs -> Level 3: {', '.join(pair_strs)}")
    else:
        print("  No instrument-level rejections -> ALL NEUTRAL. Cascade stops here.")

    return results, rejected_pairs


def level3_per_session(
    daily_agg: pd.DataFrame,
    rejected_pairs: list[tuple[str, str]],
) -> list[dict]:
    """Level 3: For each rejected (signal, instrument), test per-session.

    Returns list of classified results with year-by-year consistency.
    """
    if not rejected_pairs:
        return []

    # Build signal name -> column mapping
    sig_col_map = {name: col for name, col in CALENDAR_SIGNALS}

    sessions = sorted(daily_agg["orb_label"].unique())

    print("\n" + "=" * 80)
    print(f"  LEVEL 3: PER-SESSION (up to {len(rejected_pairs) * len(sessions)} tests)")
    print("=" * 80)

    results = []
    for sig_name, inst in rejected_pairs:
        sig_col = sig_col_map[sig_name]
        inst_data = daily_agg[
            (daily_agg["symbol"] == inst)
        ]
        for sess in sessions:
            sess_data = inst_data[inst_data["orb_label"] == sess]
            on = sess_data[sess_data[sig_col]]["pnl_r"].values
            off = sess_data[~sess_data[sig_col]]["pnl_r"].values
            test = welch_ttest(on, off)
            if test is None:
                continue
            test["signal"] = sig_name
            test["sig_col"] = sig_col
            test["instrument"] = inst
            test["session"] = sess
            results.append(test)

    results = apply_bh_fdr(results)

    # Year-by-year consistency for all results
    for r in results:
        sig_col = r["sig_col"]
        sess_data = daily_agg[
            (daily_agg["symbol"] == r["instrument"])
            & (daily_agg["orb_label"] == r["session"])
        ]
        expected_neg = r["diff"] < 0
        yr_cons, yr_tot = year_consistency_check(sess_data, sig_col, expected_neg)
        r["yr_consistent"] = yr_cons
        r["yr_total"] = yr_tot
        r["yr_pct"] = (yr_cons / yr_tot * 100) if yr_tot > 0 else 0.0

        # Classify
        if (
            r["rejected"]
            and r["yr_pct"] >= 75
            and r["diff"] <= -0.15
        ):
            r["action"] = "SKIP"
        elif (
            r["rejected"]
            and r["yr_pct"] >= 60
            and r["diff"] < 0
        ):
            r["action"] = "HALF_SIZE"
        else:
            r["action"] = "NEUTRAL"

    # Print results grouped by (signal, instrument)
    for sig_name, inst in rejected_pairs:
        pair_results = [
            r for r in results if r["signal"] == sig_name and r["instrument"] == inst
        ]
        if not pair_results:
            print(f"\n  --- {sig_name} x {inst}: no testable sessions ---")
            continue

        print(f"\n  --- {sig_name} x {inst} ---")
        print(f"  {'Session':20s} {'Diff':>8s} {'p_raw':>8s} {'p_bh':>8s} "
              f"{'N_on':>6s} {'N_off':>7s} {'YrCons':>10s} {'Action':>10s}")
        print(f"  {'-' * 80}")

        for r in sorted(pair_results, key=lambda x: x["p_bh"]):
            yr_str = f"{r['yr_consistent']}/{r['yr_total']}" if r["yr_total"] > 0 else "n/a"
            yr_pct = f"({r['yr_pct']:.0f}%)" if r["yr_total"] > 0 else ""
            print(f"  {r['session']:20s} {r['diff']:>+8.4f} {r['p_raw']:>8.4f} "
                  f"{r['p_bh']:>8.4f} {r['n_on']:>6d} {r['n_off']:>7d} "
                  f"{yr_str:>5s} {yr_pct:<5s} {r['action']:>10s}")

    return results


# =========================================================================
# Output
# =========================================================================


def build_rules_json(
    l1_results: list[dict],
    l2_results: list[dict],
    l3_results: list[dict],
) -> dict:
    """Build the output JSON structure."""
    # Count total tests across all levels
    total_tests = len(l1_results) + len(l2_results) + len(l3_results)

    # Collect non-NEUTRAL rules from Level 3
    rules = []
    for r in l3_results:
        if r["action"] == "NEUTRAL":
            continue
        rules.append({
            "instrument": r["instrument"],
            "session": r["session"],
            "signal": r["signal"],
            "action": r["action"],
            "diff": round(r["diff"], 4),
            "p_bh": round(r["p_bh"], 4),
            "yr_consistent": r["yr_consistent"],
            "yr_total": r["yr_total"],
            "n_on": r["n_on"],
            "n_off": r["n_off"],
        })

    return {
        "generated": datetime.now(tz=None).astimezone().isoformat(timespec="seconds"),
        "method": "portfolio_cascade_bh_fdr",
        "levels": 3,
        "total_tests": total_tests,
        "rules": rules,
    }


def print_final_summary(
    l1_results: list[dict],
    rejected_signals: list[str],
    l2_results: list[dict],
    rejected_pairs: list[tuple[str, str]],
    l3_results: list[dict],
    rules_json: dict,
):
    """Print the final human-readable classification table."""
    print("\n" + "=" * 80)
    print("  FINAL CLASSIFICATION SUMMARY")
    print("=" * 80)

    # Cascade budget
    print(f"\n  Hypothesis budget:")
    print(f"    Level 1 (portfolio):    {len(l1_results):>3d} tests")
    print(f"    Level 2 (instrument):   {len(l2_results):>3d} tests")
    print(f"    Level 3 (session):      {len(l3_results):>3d} tests")
    print(f"    Total:                  {rules_json['total_tests']:>3d} tests")
    print(f"    (vs 416 in flat approach)")

    # Level 1 summary
    print(f"\n  Level 1 -- Portfolio-wide rejections:")
    if rejected_signals:
        for sig in rejected_signals:
            r = next(x for x in l1_results if x["signal"] == sig)
            print(f"    {sig:15s} diff={r['diff']:+.4f}R  p_bh={r['p_bh']:.4f}")
    else:
        print("    None -- all signals NEUTRAL at portfolio level")

    # Level 1 neutrals
    neutral_l1 = [r for r in l1_results if not r["rejected"]]
    if neutral_l1:
        print(f"\n  Level 1 -- NEUTRAL (not decomposed):")
        for r in sorted(neutral_l1, key=lambda x: x["p_bh"]):
            print(f"    {r['signal']:15s} diff={r['diff']:+.4f}R  p_bh={r['p_bh']:.4f}")

    # Level 2 summary
    if l2_results:
        print(f"\n  Level 2 -- Instrument-level rejections:")
        if rejected_pairs:
            for sig, inst in rejected_pairs:
                r = next(
                    x for x in l2_results
                    if x["signal"] == sig and x["instrument"] == inst
                )
                print(f"    {sig:15s} x {inst:4s}  diff={r['diff']:+.4f}R  p_bh={r['p_bh']:.4f}")
        else:
            print("    None -- all instruments NEUTRAL")

    # Final rules
    rules = rules_json["rules"]
    if rules:
        skip_rules = [r for r in rules if r["action"] == "SKIP"]
        half_rules = [r for r in rules if r["action"] == "HALF_SIZE"]

        if skip_rules:
            print(f"\n  SKIP rules ({len(skip_rules)}):")
            for r in skip_rules:
                print(f"    {r['instrument']:4s} {r['session']:20s} {r['signal']:12s} "
                      f"diff={r['diff']:+.4f}R  p_bh={r['p_bh']:.4f}  "
                      f"consistency={r['yr_consistent']}/{r['yr_total']}")

        if half_rules:
            print(f"\n  HALF_SIZE rules ({len(half_rules)}):")
            for r in half_rules:
                print(f"    {r['instrument']:4s} {r['session']:20s} {r['signal']:12s} "
                      f"diff={r['diff']:+.4f}R  p_bh={r['p_bh']:.4f}  "
                      f"consistency={r['yr_consistent']}/{r['yr_total']}")
    else:
        print("\n  No actionable rules (SKIP or HALF_SIZE) found.")
        print("  All calendar signals classified NEUTRAL.")
        print("  This means: remove the blanket NFP/OPEX skip -- it's hurting performance.")


# =========================================================================
# Main
# =========================================================================


def run(db_path: Path):
    t0 = time.time()

    instruments = list(ACTIVE_ORB_INSTRUMENTS)
    print("=" * 80)
    print("  CALENDAR CASCADE SCANNER")
    print(f"  Method: Harvey, Liu & Zhu 3-level cascade with BH FDR q=0.10")
    print(f"  DB: {db_path}")
    print(f"  Instruments: {', '.join(instruments)}")
    print(f"  Entry models: E1, E2 (validated strategies only)")
    print(f"  Key improvement: per-day aggregation (1 obs per instrument x session x day)")
    print("=" * 80)

    # Load data
    print("\n  Loading validated strategy outcomes...")
    df = load_validated_outcomes(db_path)
    n_strategies = df.groupby(
        ["symbol", "orb_label", "entry_model", "rr_target", "confirm_bars", "orb_minutes"]
    ).ngroups
    print(f"  Loaded {len(df):,} outcome rows across {df.symbol.nunique()} instruments")
    print(f"  Unique strategies: {n_strategies}")

    # Add derived columns
    df["yr"] = pd.to_datetime(df["trading_day"]).dt.year
    df["td_date"] = pd.to_datetime(df["trading_day"]).dt.date

    # Build calendar signal columns from date builders
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
    for dow_num, dow_name in [
        (0, "Monday"), (1, "Tuesday"), (2, "Wednesday"),
        (3, "Thursday"), (4, "Friday"),
    ]:
        df[f"is_{dow_name}"] = df["day_of_week"] == dow_num

    # Per-day aggregation -- THE key step
    print("\n  Aggregating to per-day level...")
    daily_agg = aggregate_per_day(df)
    print(f"  Before: {len(df):,} rows (strategy-level)")
    print(f"  After:  {len(daily_agg):,} rows (one per instrument x session x day)")
    print(f"  Compression ratio: {len(df) / len(daily_agg):.1f}x")

    # Run the 3-level cascade
    l1_results, rejected_signals = level1_portfolio_wide(daily_agg)
    l2_results, rejected_pairs = level2_per_instrument(
        daily_agg, rejected_signals, instruments
    )
    l3_results = level3_per_session(daily_agg, rejected_pairs)

    # Build and save output JSON
    rules_json = build_rules_json(l1_results, l2_results, l3_results)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "calendar_cascade_rules.json"
    with open(json_path, "w") as f:
        json.dump(rules_json, f, indent=2)
    print(f"\n  Rules saved: {json_path}")

    # Print final summary
    print_final_summary(
        l1_results, rejected_signals,
        l2_results, rejected_pairs,
        l3_results, rules_json,
    )

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"  Total tests: {rules_json['total_tests']}")
    print(f"  Actionable rules: {len(rules_json['rules'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calendar cascade scanner")
    parser.add_argument("--db-path", type=Path, default=GOLD_DB_PATH)
    args = parser.parse_args()
    run(args.db_path)
