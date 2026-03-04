#!/usr/bin/env python3
"""Session alignment audit: fixed vs dynamic ORB session performance during summer.

2026-02-25. Task 8 of session architecture overhaul.

Hypothesis: During summer (US/UK DST active), fixed Brisbane-time sessions
(0900, 1800, 0030, 2300) are ~1hr late relative to the actual market event.
Dynamic sessions (CME_OPEN, LONDON_OPEN, US_EQUITY_OPEN, US_DATA_OPEN)
track the correct time regardless of DST. If the market event timing matters,
dynamic sessions should show better performance during summer.

Control: Winter performance should be similar since fixed times align with
the market event during standard time.

Session pairs:
  0900 <-> CME_OPEN       (CME Globex open at 5pm CT, US DST)
  1800 <-> LONDON_OPEN    (London metals open at 8am London, UK DST)
  0030 <-> US_EQUITY_OPEN (NYSE cash open at 9:30am ET, US DST)
  2300 <-> US_DATA_OPEN   (US data at 8:30am ET, US DST)

All features computed from data available at decision time. No look-ahead.
"""

from __future__ import annotations

import sys

# Force unbuffered stdout for background runs
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from pipeline.paths import GOLD_DB_PATH
from pipeline.asset_configs import ASSET_CONFIGS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ACTIVE_INSTRUMENTS = ["MGC", "MES", "MNQ", "M2K"]

# Session pairs: (fixed_label, dynamic_label, dst_column, event_description)
# dst_column: which DST flag determines summer/winter for this pair
SESSION_PAIRS = [
    ("0900", "CME_OPEN",       "us_dst", "CME Globex open (5pm CT)"),
    ("1800", "LONDON_OPEN",    "uk_dst", "London metals open (8am London)"),
    ("0030", "US_EQUITY_OPEN", "us_dst", "NYSE cash open (9:30am ET)"),
    ("2300", "US_DATA_OPEN",   "us_dst", "US data release (8:30am ET)"),
]

# Baseline parameters (most data, per CLAUDE.md)
ENTRY_MODEL = "E0"
RR_TARGET = 1.0
CONFIRM_BARS = 1
ORB_MINUTES = 5

MIN_TRADES = 30  # RESEARCH_RULES.md: <30 = INVALID


def get_enabled_sessions(instrument: str) -> list[str]:
    """Return enabled sessions for an instrument."""
    cfg = ASSET_CONFIGS.get(instrument.upper())
    if cfg is None:
        return []
    return cfg.get("enabled_sessions", [])


def load_comparison_data(
    db: duckdb.DuckDBPyConnection,
    instrument: str,
    fixed_label: str,
    dynamic_label: str,
    dst_col: str,
) -> pd.DataFrame | None:
    """Load pnl_r for both fixed and dynamic sessions, joined with DST flag.

    Returns DataFrame with columns: trading_day, orb_label, pnl_r, is_summer
    or None if one or both sessions have no data.
    """
    # Check which sessions are enabled for this instrument
    enabled = get_enabled_sessions(instrument)
    have_fixed = fixed_label in enabled
    have_dynamic = dynamic_label in enabled

    if not have_fixed and not have_dynamic:
        return None

    labels_to_query = []
    if have_fixed:
        labels_to_query.append(fixed_label)
    if have_dynamic:
        labels_to_query.append(dynamic_label)

    labels_str = ", ".join(f"'{l}'" for l in labels_to_query)

    query = f"""
        SELECT
            o.trading_day,
            o.orb_label,
            o.pnl_r,
            d.{dst_col} AS is_summer
        FROM orb_outcomes o
        JOIN daily_features d
            ON o.trading_day = d.trading_day
            AND o.symbol = d.symbol
            AND o.orb_minutes = d.orb_minutes
        WHERE o.symbol = '{instrument}'
            AND o.orb_minutes = {ORB_MINUTES}
            AND o.entry_model = '{ENTRY_MODEL}'
            AND o.rr_target = {RR_TARGET}
            AND o.confirm_bars = {CONFIRM_BARS}
            AND o.orb_label IN ({labels_str})
            AND o.pnl_r IS NOT NULL
            AND d.orb_minutes = {ORB_MINUTES}
        ORDER BY o.trading_day, o.orb_label
    """
    df = db.sql(query).fetchdf()
    if df.empty:
        return None
    return df


def compute_stats(pnl_series: pd.Series) -> dict:
    """Compute summary stats for a pnl_r series."""
    n = len(pnl_series)
    if n == 0:
        return {"n": 0, "avg_r": np.nan, "std_r": np.nan, "wr": np.nan, "sharpe": np.nan}
    avg = pnl_series.mean()
    std = pnl_series.std()
    wr = (pnl_series > 0).mean()
    sharpe = avg / std * np.sqrt(252) if std > 0 else np.nan
    return {"n": n, "avg_r": avg, "std_r": std, "wr": wr, "sharpe": sharpe}


def main():
    print("=" * 90)
    print("SESSION ALIGNMENT AUDIT: Fixed vs Dynamic ORB Performance")
    print("=" * 90)
    print(f"Baseline: entry_model={ENTRY_MODEL}, rr_target={RR_TARGET}, "
          f"confirm_bars={CONFIRM_BARS}, orb_minutes={ORB_MINUTES}")
    print(f"Database: {GOLD_DB_PATH}")
    print(f"Instruments: {', '.join(ACTIVE_INSTRUMENTS)}")
    print(f"Min trades for valid comparison: {MIN_TRADES}")
    print()

    db = duckdb.connect(str(GOLD_DB_PATH), read_only=True)

    # Collect all results for summary table
    results = []

    for fixed_label, dynamic_label, dst_col, event_desc in SESSION_PAIRS:
        print("-" * 90)
        print(f"SESSION PAIR: {fixed_label} (fixed) vs {dynamic_label} (dynamic)")
        print(f"Event: {event_desc}")
        print(f"DST column: {dst_col} (True = summer / DST active)")
        print("-" * 90)

        for instrument in ACTIVE_INSTRUMENTS:
            enabled = get_enabled_sessions(instrument)
            have_fixed = fixed_label in enabled
            have_dynamic = dynamic_label in enabled

            if not have_fixed and not have_dynamic:
                print(f"  {instrument}: Neither session enabled — skipping")
                continue

            if not have_fixed:
                print(f"  {instrument}: {fixed_label} not enabled (only {dynamic_label}) — skipping pair")
                continue
            if not have_dynamic:
                print(f"  {instrument}: {dynamic_label} not enabled (only {fixed_label}) — skipping pair")
                continue

            df = load_comparison_data(db, instrument, fixed_label, dynamic_label, dst_col)
            if df is None or df.empty:
                print(f"  {instrument}: No data — skipping")
                continue

            for season_label, is_summer_val in [("SUMMER", True), ("WINTER", False)]:
                season_df = df[df["is_summer"] == is_summer_val]

                fixed_pnl = season_df.loc[season_df["orb_label"] == fixed_label, "pnl_r"]
                dynamic_pnl = season_df.loc[season_df["orb_label"] == dynamic_label, "pnl_r"]

                fixed_stats = compute_stats(fixed_pnl)
                dynamic_stats = compute_stats(dynamic_pnl)

                # t-test (only if both have sufficient data)
                t_stat = np.nan
                p_val = np.nan
                if fixed_stats["n"] >= MIN_TRADES and dynamic_stats["n"] >= MIN_TRADES:
                    t_stat, p_val = stats.ttest_ind(
                        fixed_pnl.values, dynamic_pnl.values, equal_var=False
                    )

                delta = dynamic_stats["avg_r"] - fixed_stats["avg_r"] if not (
                    np.isnan(dynamic_stats["avg_r"]) or np.isnan(fixed_stats["avg_r"])
                ) else np.nan

                results.append({
                    "fixed": fixed_label,
                    "dynamic": dynamic_label,
                    "instrument": instrument,
                    "season": season_label,
                    "fixed_n": fixed_stats["n"],
                    "fixed_avg_r": fixed_stats["avg_r"],
                    "fixed_wr": fixed_stats["wr"],
                    "dynamic_n": dynamic_stats["n"],
                    "dynamic_avg_r": dynamic_stats["avg_r"],
                    "dynamic_wr": dynamic_stats["wr"],
                    "delta_r": delta,
                    "t_stat": t_stat,
                    "p_val": p_val,
                })

        print()

    db.close()

    # =========================================================================
    # Print results table
    # =========================================================================
    if not results:
        print("NO RESULTS — no session pairs had data for comparison.")
        return

    rdf = pd.DataFrame(results)

    print()
    print("=" * 130)
    print("RESULTS TABLE")
    print("=" * 130)
    print(f"{'Pair':<22} {'Instr':<5} {'Season':<7} | "
          f"{'Fixed N':>7} {'Avg R':>8} {'WR':>6} | "
          f"{'Dyn N':>7} {'Avg R':>8} {'WR':>6} | "
          f"{'Delta':>8} {'t-stat':>8} {'p-val':>8} {'Flag':<6}")
    print("-" * 130)

    for _, row in rdf.iterrows():
        pair = f"{row['fixed']} vs {row['dynamic']}"

        # Flag significant results
        flag = ""
        if not np.isnan(row["p_val"]):
            if row["p_val"] < 0.005:
                flag = "***"
            elif row["p_val"] < 0.01:
                flag = "**"
            elif row["p_val"] < 0.05:
                flag = "*"

        # Mark insufficient sample
        if row["fixed_n"] < MIN_TRADES or row["dynamic_n"] < MIN_TRADES:
            flag = "LOW-N"

        print(f"{pair:<22} {row['instrument']:<5} {row['season']:<7} | "
              f"{row['fixed_n']:>7} {row['fixed_avg_r']:>8.4f} {row['fixed_wr']:>5.1%} | "
              f"{row['dynamic_n']:>7} {row['dynamic_avg_r']:>8.4f} {row['dynamic_wr']:>5.1%} | "
              f"{row['delta_r']:>8.4f} {row['t_stat']:>8.3f} {row['p_val']:>8.4f} {flag:<6}")

    # =========================================================================
    # Summer-only analysis: Does dynamic beat fixed?
    # =========================================================================
    print()
    print("=" * 90)
    print("SUMMER-ONLY ANALYSIS (hypothesis: dynamic > fixed during DST)")
    print("=" * 90)

    summer = rdf[rdf["season"] == "SUMMER"].copy()
    summer_valid = summer[
        (summer["fixed_n"] >= MIN_TRADES) & (summer["dynamic_n"] >= MIN_TRADES)
    ]

    if summer_valid.empty:
        print("No valid summer comparisons (all below min trades).")
    else:
        print(f"\nValid summer comparisons: {len(summer_valid)}")
        dynamic_better = summer_valid[summer_valid["delta_r"] > 0]
        print(f"Dynamic outperforms fixed: {len(dynamic_better)}/{len(summer_valid)} "
              f"({len(dynamic_better)/len(summer_valid):.0%})")

        sig_summer = summer_valid[summer_valid["p_val"] < 0.05]
        if len(sig_summer) > 0:
            print(f"\nStatistically significant summer differences (p<0.05):")
            for _, row in sig_summer.iterrows():
                direction = "DYNAMIC better" if row["delta_r"] > 0 else "FIXED better"
                print(f"  {row['instrument']} {row['fixed']} vs {row['dynamic']}: "
                      f"delta={row['delta_r']:+.4f}R, p={row['p_val']:.4f} -> {direction}")
        else:
            print("\nNo statistically significant summer differences at p<0.05.")

    # =========================================================================
    # Winter control: Should show no difference (times align)
    # =========================================================================
    print()
    print("=" * 90)
    print("WINTER CONTROL (times align — expect no difference)")
    print("=" * 90)

    winter = rdf[rdf["season"] == "WINTER"].copy()
    winter_valid = winter[
        (winter["fixed_n"] >= MIN_TRADES) & (winter["dynamic_n"] >= MIN_TRADES)
    ]

    if winter_valid.empty:
        print("No valid winter comparisons.")
    else:
        print(f"\nValid winter comparisons: {len(winter_valid)}")

        # Check if winter data is identical (expected: fixed and dynamic resolve
        # to the same Brisbane time during winter, producing identical ORBs)
        identical_winter = winter_valid[winter_valid["delta_r"] == 0.0]
        if len(identical_winter) > 0:
            print(f"\nNOTE: {len(identical_winter)}/{len(winter_valid)} winter pairs have "
                  f"IDENTICAL results (delta=0, p=1.0).")
            print("  This is EXPECTED: during standard time (no DST), fixed Brisbane times")
            print("  align exactly with market events, so both sessions capture the same")
            print("  5-minute ORB window and produce identical outcomes.")
            print("  This validates the control: the only difference is DST misalignment.")

        sig_winter = winter_valid[winter_valid["p_val"] < 0.05]
        if len(sig_winter) > 0:
            print(f"\nWARNING: {len(sig_winter)} significant winter difference(s) — "
                  f"unexpected if times truly align:")
            for _, row in sig_winter.iterrows():
                print(f"  {row['instrument']} {row['fixed']} vs {row['dynamic']}: "
                      f"delta={row['delta_r']:+.4f}R, p={row['p_val']:.4f}")
        else:
            print("\nNo significant winter differences. Control passes.")

    # =========================================================================
    # Summer-vs-winter delta-of-deltas (interaction test)
    # =========================================================================
    print()
    print("=" * 90)
    print("INTERACTION TEST: Summer delta vs Winter delta")
    print("  (If DST misalignment hurts fixed sessions, the summer delta should")
    print("   be more negative for fixed (or more positive for dynamic) than winter.)")
    print("=" * 90)

    for fixed_label, dynamic_label, dst_col, event_desc in SESSION_PAIRS:
        for instrument in ACTIVE_INSTRUMENTS:
            mask_s = (
                (rdf["fixed"] == fixed_label) &
                (rdf["dynamic"] == dynamic_label) &
                (rdf["instrument"] == instrument) &
                (rdf["season"] == "SUMMER")
            )
            mask_w = (
                (rdf["fixed"] == fixed_label) &
                (rdf["dynamic"] == dynamic_label) &
                (rdf["instrument"] == instrument) &
                (rdf["season"] == "WINTER")
            )
            if mask_s.sum() == 0 or mask_w.sum() == 0:
                continue

            summer_row = rdf[mask_s].iloc[0]
            winter_row = rdf[mask_w].iloc[0]

            # Skip if either season has insufficient data
            if (summer_row["fixed_n"] < MIN_TRADES or summer_row["dynamic_n"] < MIN_TRADES or
                    winter_row["fixed_n"] < MIN_TRADES or winter_row["dynamic_n"] < MIN_TRADES):
                continue

            summer_delta = summer_row["delta_r"]
            winter_delta = winter_row["delta_r"]
            interaction = summer_delta - winter_delta

            print(f"  {instrument} {fixed_label} vs {dynamic_label}: "
                  f"summer_delta={summer_delta:+.4f}R, "
                  f"winter_delta={winter_delta:+.4f}R, "
                  f"interaction={interaction:+.4f}R "
                  f"{'(dynamic gains in summer)' if interaction > 0 else '(no summer advantage for dynamic)'}")

    # =========================================================================
    # Honest summary
    # =========================================================================
    print()
    print("=" * 90)
    print("HONEST SUMMARY")
    print("=" * 90)
    print()

    # Count significant results
    all_valid = rdf[(rdf["fixed_n"] >= MIN_TRADES) & (rdf["dynamic_n"] >= MIN_TRADES)]
    summer_sig = summer_valid[summer_valid["p_val"] < 0.05] if not summer_valid.empty else pd.DataFrame()
    winter_sig = winter_valid[winter_valid["p_val"] < 0.05] if not winter_valid.empty else pd.DataFrame()

    n_summer_valid = len(summer_valid) if not summer_valid.empty else 0
    n_summer_sig = len(summer_sig) if not summer_sig.empty else 0
    n_winter_valid = len(winter_valid) if not winter_valid.empty else 0
    n_winter_sig = len(winter_sig) if not winter_sig.empty else 0
    n_summer_dyn_better = len(summer_valid[summer_valid["delta_r"] > 0]) if not summer_valid.empty else 0

    print(f"Total valid comparisons: {len(all_valid)} ({n_summer_valid} summer, {n_winter_valid} winter)")
    print(f"Summer: dynamic outperforms in {n_summer_dyn_better}/{n_summer_valid} pairs")
    print(f"Summer significant (p<0.05): {n_summer_sig}/{n_summer_valid}")
    print(f"Winter significant (p<0.05): {n_winter_sig}/{n_winter_valid}")
    print()
    print("SURVIVED SCRUTINY:")
    if n_summer_sig > 0:
        for _, row in summer_sig.iterrows():
            print(f"  - {row['instrument']} {row['fixed']} vs {row['dynamic']} summer: "
                  f"delta={row['delta_r']:+.4f}R, p={row['p_val']:.4f}, "
                  f"N_fixed={row['fixed_n']}, N_dynamic={row['dynamic_n']}")
    else:
        print("  - Nothing. No significant summer performance difference between fixed and dynamic sessions.")
    print()
    print("DID NOT SURVIVE:")
    print("  - All other comparisons: no statistically significant difference.")
    print()
    print("CAVEATS:")
    print("  - This is an in-sample analysis across all available data.")
    print("  - Comparisons use E0 CB1 RR1.0 only — other entry models may differ.")
    print("  - Some instrument/session pairs are not enabled (MGC has no CME_OPEN; etc).")
    print("  - 2300 vs US_DATA_OPEN is not a clean 'same event' comparison:")
    print("    2300 is 30min before data in winter, 30min after in summer.")
    print("  - Number of comparisons tested: up to 16 (4 pairs x 4 instruments).")
    print("    At p<0.05 we expect ~1 false positive by chance alone.")
    print()
    print("NEXT STEPS:")
    print("  - If significant summer differences found: consider replacing fixed sessions")
    print("    with dynamic equivalents in production (Task 10+).")
    print("  - If no differences: fixed sessions are adequate; dynamic sessions add")
    print("    complexity without measurable benefit for ORB breakout performance.")


if __name__ == "__main__":
    main()
