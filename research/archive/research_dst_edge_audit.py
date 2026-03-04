#!/usr/bin/env python3
"""
DST Edge Audit — Do confirmed edges degrade when fixed sessions drift from market events?

Fixed sessions (0900, 1800) measure ORBs at constant Brisbane times regardless of DST.
Dynamic sessions (CME_OPEN, LONDON_OPEN) shift with DST to track actual market events.

If the edge is really about the market event (e.g., CME Globex open), then during
summer when the fixed time is 1 hour off from the event, edge quality should degrade.

This script answers whether confirmed edges should switch to dynamic sessions.

Sections:
  1. DST Resolver Correctness Audit
  2. Winter vs Summer Edge Split (fixed sessions)
  3. Dynamic vs Fixed Head-to-Head (same-day comparison)
  4. DST Mismatch Windows (US vs UK differ)
  5. Recommendations
"""

import argparse
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np

from pipeline.dst import (
    SESSION_CATALOG,
    cme_open_brisbane,
    is_uk_dst,
    is_us_dst,
    london_open_brisbane,
    us_data_open_brisbane,
    us_equity_open_brisbane,
)
from pipeline.paths import GOLD_DB_PATH
from research._alt_strategy_utils import compute_strategy_metrics

# =========================================================================
# Constants
# =========================================================================

DYNAMIC_RESOLVERS = {
    "CME_OPEN": cme_open_brisbane,
    "LONDON_OPEN": london_open_brisbane,
    "US_EQUITY_OPEN": us_equity_open_brisbane,
    "US_DATA_OPEN": us_data_open_brisbane,
}

EXPECTED_HOURS = {
    "CME_OPEN": {8, 9},
    "LONDON_OPEN": {17, 18},
    "US_EQUITY_OPEN": {23, 0},
    "US_DATA_OPEN": {22, 23},
}

# Confirmed edge combos: (instrument, session, entry_model, rr_target, confirm_bars, size_gate_label, size_min, dst_flag)
EDGE_COMBOS = [
    ("MGC", "0900", "E1", 2.5, 2, "G5+", 5.0, "us_dst"),
    ("MGC", "1800", "E3", 2.0, 2, "G4+", 4.0, "uk_dst"),
    ("MNQ", "0900", "E1", 2.5, 2, "G4+", 4.0, "us_dst"),
    ("MES", "0900", "E1", 2.5, 2, "G3+", 3.0, "us_dst"),
]

# Head-to-head pairs: (fixed_session, dynamic_session, instrument, em, rr, cb)
H2H_PAIRS = [
    ("0900", "CME_OPEN", "MGC", "E1", 2.5, 2),
    ("1800", "LONDON_OPEN", "MGC", "E3", 2.0, 2),
    ("0030", "US_EQUITY_OPEN", "MES", "E1", 2.5, 2),
    ("0030", "US_EQUITY_OPEN", "MNQ", "E1", 2.5, 2),
]

_US_EASTERN = ZoneInfo("America/New_York")
_UK_LONDON = ZoneInfo("Europe/London")


# =========================================================================
# Helpers
# =========================================================================

def fmt_metrics(m: dict | None) -> str:
    """Format metrics dict as a compact summary line."""
    if m is None:
        return "N=0  (no data)"
    return (
        f"N={m['n']:<4d} avgR={m['expr']:+.3f}  "
        f"totR={m['total']:+.1f}  WR={m['wr'] * 100:.1f}%  "
        f"Sharpe={m['sharpe']:.3f}"
    )


def compute_dst_transition_dates(year: int) -> dict:
    """Compute US and UK DST transition dates for a given year."""
    from datetime import datetime

    # US: 2nd Sunday March, 1st Sunday November
    mar1 = date(year, 3, 1)
    sundays_mar = [mar1 + timedelta(days=d)
                   for d in range(31)
                   if (mar1 + timedelta(days=d)).weekday() == 6]
    us_spring = sundays_mar[1]  # 2nd Sunday

    nov1 = date(year, 11, 1)
    sundays_nov = [nov1 + timedelta(days=d)
                   for d in range(30)
                   if (nov1 + timedelta(days=d)).weekday() == 6]
    us_fall = sundays_nov[0]  # 1st Sunday

    # UK: Last Sunday March, Last Sunday October
    mar31 = date(year, 3, 31)
    uk_spring = mar31 - timedelta(days=(mar31.weekday() + 1) % 7)

    oct31 = date(year, 10, 31)
    uk_fall = oct31 - timedelta(days=(oct31.weekday() + 1) % 7)

    return {
        "us_spring": us_spring,
        "us_fall": us_fall,
        "uk_spring": uk_spring,
        "uk_fall": uk_fall,
    }


# =========================================================================
# Section 1: DST Resolver Correctness Audit
# =========================================================================

def section1_resolver_audit(con: duckdb.DuckDBPyConnection):
    print("=" * 60)
    print("  SECTION 1: DST RESOLVER CORRECTNESS")
    print("=" * 60)

    # Load all distinct trading days
    days_df = con.execute("""
        SELECT DISTINCT trading_day FROM daily_features
        ORDER BY trading_day
    """).fetchdf()
    trading_days = [d.date() if hasattr(d, 'date') else d
                    for d in days_df["trading_day"]]
    print(f"\n  Trading days in DB: {len(trading_days)} "
          f"({trading_days[0]} to {trading_days[-1]})")

    # Resolver hour distribution
    total_anomalies = 0
    for label, resolver in DYNAMIC_RESOLVERS.items():
        expected = EXPECTED_HOURS[label]
        hour_counts = defaultdict(int)
        anomalies = []

        for td in trading_days:
            h, m = resolver(td)
            hour_counts[h] += 1
            if h not in expected:
                anomalies.append((td, h, m))

        print(f"\n  {label} Brisbane hour distribution:")
        for h in sorted(hour_counts.keys()):
            marker = " ***" if h not in expected else ""
            print(f"    Hour {h:2d}: {hour_counts[h]:4d} days{marker}")
        print(f"    ANOMALIES: {len(anomalies)}")
        if anomalies:
            for td, h, m in anomalies[:5]:
                print(f"      {td} -> {h:02d}:{m:02d}")
            if len(anomalies) > 5:
                print(f"      ... and {len(anomalies) - 5} more")
        total_anomalies += len(anomalies)

    # DST transition dates table
    min_year = trading_days[0].year
    max_year = trading_days[-1].year
    print(f"\n  DST Transition Dates ({min_year}-{max_year}):")
    print(f"    {'Year':>4s} | {'US Spring':>10s} | {'US Fall':>10s} | "
          f"{'UK Spring':>10s} | {'UK Fall':>10s}")
    print(f"    {'-' * 4}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")

    all_mismatches = []
    for year in range(min_year, max_year + 1):
        t = compute_dst_transition_dates(year)
        print(f"    {year} | {t['us_spring']} | {t['us_fall']} | "
              f"{t['uk_spring']} | {t['uk_fall']}")

        # Spring mismatch: US goes DST first (mid-March), UK follows (late March)
        if t["us_spring"] < t["uk_spring"]:
            all_mismatches.append({
                "year": year, "season": "Spring",
                "start": t["us_spring"], "end": t["uk_spring"] - timedelta(days=1),
                "note": "US=DST, UK=not",
            })

        # Fall mismatch: UK reverts first (late Oct), US follows (early Nov)
        if t["uk_fall"] < t["us_fall"]:
            all_mismatches.append({
                "year": year, "season": "Fall",
                "start": t["uk_fall"], "end": t["us_fall"] - timedelta(days=1),
                "note": "UK=reverted, US=still DST",
            })

    # Mismatch windows
    print(f"\n  US/UK DST Mismatch Windows:")
    for mm in all_mismatches:
        days = (mm["end"] - mm["start"]).days + 1
        print(f"    {mm['year']} {mm['season']:6s}: "
              f"{mm['start']} - {mm['end']} ({days} days) — {mm['note']}")

    print(f"\n  TOTAL RESOLVER ANOMALIES: {total_anomalies}")
    return total_anomalies


# =========================================================================
# Section 2: Fixed Session Edge Split (Winter vs Summer)
# =========================================================================

def section2_winter_summer_split(con: duckdb.DuckDBPyConnection):
    print("\n" + "=" * 60)
    print("  SECTION 2: WINTER vs SUMMER EDGE SPLIT")
    print("=" * 60)

    verdicts = {}

    for instrument, session, em, rr, cb, gate_label, gate_min, dst_flag in EDGE_COMBOS:
        print(f"\n  {instrument} {session} {em} RR{rr} CB{cb} {gate_label}:")

        df = con.execute(f"""
            SELECT o.pnl_r, d.{dst_flag} AS is_dst
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.symbol = d.symbol
              AND o.trading_day = d.trading_day
              AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = ?
              AND o.orb_label = ?
              AND o.entry_model = ?
              AND o.rr_target = ?
              AND o.confirm_bars = ?
              AND o.outcome IN ('win', 'loss')
              AND d.orb_{session}_size >= ?
        """, [instrument, session, em, rr, cb, gate_min]).fetchdf()

        if len(df) == 0:
            print("    No data.")
            verdicts[(instrument, session)] = "NO DATA"
            continue

        winter = df[~df["is_dst"]]["pnl_r"].values
        summer = df[df["is_dst"]]["pnl_r"].values

        m_winter = compute_strategy_metrics(np.array(winter, dtype=float))
        m_summer = compute_strategy_metrics(np.array(summer, dtype=float))

        dst_off_label = "EST" if dst_flag == "us_dst" else "GMT"
        dst_on_label = "EDT" if dst_flag == "us_dst" else "BST"

        print(f"    Winter ({dst_off_label}): {fmt_metrics(m_winter)}")
        print(f"    Summer ({dst_on_label}): {fmt_metrics(m_summer)}")

        # Compute verdict
        if m_winter is None or m_summer is None:
            verdict = "INCONCLUSIVE (insufficient data in one period)"
        elif m_winter["n"] < 15 or m_summer["n"] < 15:
            verdict = "INCONCLUSIVE (< 15 trades in one period)"
        else:
            delta = m_winter["expr"] - m_summer["expr"]
            print(f"    Delta:        avgR={delta:+.3f}  (winter - summer)")

            if delta > 0.15 or (m_summer["expr"] > 0 and m_winter["expr"] / m_summer["expr"] > 2.0):
                verdict = "SWITCH TO DYNAMIC (winter >> summer)"
            elif abs(delta) <= 0.10:
                verdict = "KEEP FIXED (winter ~ summer)"
            else:
                verdict = "INCONCLUSIVE"

        print(f"    -> VERDICT: {verdict}")
        verdicts[(instrument, session)] = verdict

    return verdicts


# =========================================================================
# Section 3: Dynamic vs Fixed Head-to-Head
# =========================================================================

def section3_head_to_head(con: duckdb.DuckDBPyConnection):
    print("\n" + "=" * 60)
    print("  SECTION 3: DYNAMIC vs FIXED HEAD-TO-HEAD")
    print("=" * 60)

    h2h_results = {}

    for fixed, dynamic, instrument, em, rr, cb in H2H_PAIRS:
        print(f"\n  {instrument}: {fixed} vs {dynamic} ({em} RR{rr} CB{cb}):")

        # Self-join: same day, same params, different orb_label
        df = con.execute("""
            SELECT f.pnl_r AS fixed_pnl,
                   d.pnl_r AS dynamic_pnl,
                   f.trading_day
            FROM orb_outcomes f
            JOIN orb_outcomes d
              ON f.symbol = d.symbol
              AND f.trading_day = d.trading_day
              AND f.orb_minutes = d.orb_minutes
              AND f.rr_target = d.rr_target
              AND f.confirm_bars = d.confirm_bars
              AND f.entry_model = d.entry_model
            WHERE f.symbol = ?
              AND f.orb_label = ?
              AND d.orb_label = ?
              AND f.entry_model = ?
              AND f.rr_target = ?
              AND f.confirm_bars = ?
              AND f.outcome IN ('win', 'loss')
              AND d.outcome IN ('win', 'loss')
        """, [instrument, fixed, dynamic, em, rr, cb]).fetchdf()

        if len(df) == 0:
            print("    No overlapping resolved days.")
            h2h_results[(instrument, fixed, dynamic)] = None
            continue

        m_fixed = compute_strategy_metrics(np.array(df["fixed_pnl"].values, dtype=float))
        m_dynamic = compute_strategy_metrics(np.array(df["dynamic_pnl"].values, dtype=float))

        print(f"    Matched days: {len(df)}")
        print(f"    {fixed:>15s}: {fmt_metrics(m_fixed)}")
        print(f"    {dynamic:>15s}: {fmt_metrics(m_dynamic)}")

        if m_fixed and m_dynamic:
            delta = m_dynamic["expr"] - m_fixed["expr"]
            print(f"    Delta (dynamic - fixed): avgR={delta:+.3f}")
            h2h_results[(instrument, fixed, dynamic)] = {
                "fixed": m_fixed, "dynamic": m_dynamic,
                "delta": delta, "n_matched": len(df),
            }
        else:
            h2h_results[(instrument, fixed, dynamic)] = None

    return h2h_results


# =========================================================================
# Section 4: DST Mismatch Windows
# =========================================================================

def section4_mismatch_windows(con: duckdb.DuckDBPyConnection):
    print("\n" + "=" * 60)
    print("  SECTION 4: DST MISMATCH WINDOWS")
    print("=" * 60)

    # Find days where us_dst != uk_dst
    df = con.execute("""
        SELECT DISTINCT d.trading_day, d.us_dst, d.uk_dst
        FROM daily_features d
        WHERE d.symbol = 'MGC'
          AND d.orb_minutes = 5
          AND d.us_dst != d.uk_dst
        ORDER BY d.trading_day
    """).fetchdf()

    if len(df) == 0:
        print("\n  No mismatch days found in data.")
        return

    # Classify spring vs fall
    spring_days = []
    fall_days = []
    for _, row in df.iterrows():
        td = row["trading_day"]
        if hasattr(td, 'date'):
            td = td.date()
        month = td.month
        if month in (3, 4):  # Spring mismatch
            spring_days.append(td)
        elif month in (10, 11):  # Fall mismatch
            fall_days.append(td)

    print(f"\n  Total mismatch days: {len(df)}")
    print(f"    Spring (Mar-Apr): {len(spring_days)} days")
    print(f"    Fall (Oct-Nov):   {len(fall_days)} days")

    # Analyze MGC 0900 (us_dst matters) and MGC 1800 (uk_dst matters) during mismatches
    combos = [
        ("MGC", "0900", "E1", 2.5, 2, 5.0, "us_dst"),
        ("MGC", "1800", "E3", 2.0, 2, 4.0, "uk_dst"),
    ]

    for instrument, session, em, rr, cb, gate_min, dst_flag in combos:
        print(f"\n  {instrument} {session} ({dst_flag}) — mismatch vs normal:")

        # All resolved trades with size gate
        all_trades = con.execute(f"""
            SELECT o.pnl_r, d.us_dst, d.uk_dst, o.trading_day
            FROM orb_outcomes o
            JOIN daily_features d
              ON o.symbol = d.symbol
              AND o.trading_day = d.trading_day
              AND o.orb_minutes = d.orb_minutes
            WHERE o.symbol = ?
              AND o.orb_label = ?
              AND o.entry_model = ?
              AND o.rr_target = ?
              AND o.confirm_bars = ?
              AND o.outcome IN ('win', 'loss')
              AND d.orb_{session}_size >= ?
        """, [instrument, session, em, rr, cb, gate_min]).fetchdf()

        if len(all_trades) == 0:
            print("    No data.")
            continue

        # Mismatch = us_dst != uk_dst
        mismatch_mask = all_trades["us_dst"] != all_trades["uk_dst"]
        mismatch_pnls = all_trades[mismatch_mask]["pnl_r"].values
        normal_pnls = all_trades[~mismatch_mask]["pnl_r"].values

        m_mismatch = compute_strategy_metrics(np.array(mismatch_pnls, dtype=float))
        m_normal = compute_strategy_metrics(np.array(normal_pnls, dtype=float))

        print(f"    Mismatch: {fmt_metrics(m_mismatch)}")
        print(f"    Normal:   {fmt_metrics(m_normal)}")

        if m_mismatch and m_mismatch["n"] < 15:
            print("    NOTE: LOW SAMPLE in mismatch window — directional only")


# =========================================================================
# Section 5: Recommendations
# =========================================================================

def section5_recommendations(verdicts: dict, h2h_results: dict, anomaly_count: int):
    print("\n" + "=" * 60)
    print("  RECOMMENDATIONS")
    print("=" * 60)

    dynamic_map = {"0900": "CME_OPEN", "1800": "LONDON_OPEN", "0030": "US_EQUITY_OPEN"}

    for (instrument, session), verdict in verdicts.items():
        alt = dynamic_map.get(session, "?")
        if "SWITCH" in verdict:
            rec = f"SWITCH TO {alt}"
        elif "KEEP" in verdict:
            rec = "KEEP FIXED"
        else:
            rec = "INCONCLUSIVE"

        # Supplement with h2h if available
        h2h_key = None
        for key, val in h2h_results.items():
            if key[0] == instrument and key[1] == session:
                h2h_key = key
                break

        h2h_note = ""
        if h2h_key and h2h_results[h2h_key]:
            r = h2h_results[h2h_key]
            h2h_note = f"  (H2H delta: {r['delta']:+.3f}R on {r['n_matched']} matched days)"

        print(f"  {instrument} {session}: {rec}{h2h_note}")

    print(f"\n  DST Resolver Bugs: {'None found' if anomaly_count == 0 else f'{anomaly_count} anomalies'}")
    print()


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="DST Edge Audit")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to gold.db (default: auto-resolve)")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    con = duckdb.connect(str(db_path), read_only=True)

    print()
    print("=" * 60)
    print("  DST EDGE AUDIT")
    print(f"  Database: {db_path}")
    print("=" * 60)

    try:
        anomaly_count = section1_resolver_audit(con)
        verdicts = section2_winter_summer_split(con)
        h2h_results = section3_head_to_head(con)
        section4_mismatch_windows(con)
        section5_recommendations(verdicts, h2h_results, anomaly_count)
    finally:
        con.close()


if __name__ == "__main__":
    main()
