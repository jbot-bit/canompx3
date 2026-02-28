#!/usr/bin/env python3
"""
DOW Alignment Investigation — Brisbane DOW vs Exchange-Timezone DOW.

The question: when we skip "Brisbane Friday" at session X, are we actually
skipping the exchange's Friday session?  For sessions that straddle midnight
UTC, the Brisbane calendar date may not match the exchange's calendar date.

This script produces a canonical mapping:
  For each (session, Brisbane DOW) → what UTC time? → what exchange calendar day?

Then validates whether existing DOW filters (NOFRI@0900, NOMON@1800, NOTUE@1000)
are correctly aligned with the exchange events they intend to target.

Also runs a data-driven check: groups outcomes by BOTH Brisbane DOW and exchange
DOW and compares whether the edge signal is stronger on one vs the other.

Usage:
  python research/research_dow_alignment.py
  python research/research_dow_alignment.py --db-path C:/db/gold.db
"""

import argparse
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

# =========================================================================
# Timezone definitions
# =========================================================================

_BRISBANE = ZoneInfo("Australia/Brisbane")       # UTC+10, no DST ever
_US_EASTERN = ZoneInfo("America/New_York")       # EST UTC-5 / EDT UTC-4
_US_CHICAGO = ZoneInfo("America/Chicago")        # CST UTC-6 / CDT UTC-5
_UK_LONDON = ZoneInfo("Europe/London")           # GMT UTC+0 / BST UTC+1
_TOKYO = ZoneInfo("Asia/Tokyo")                  # JST UTC+9, no DST
_UTC = ZoneInfo("UTC")

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# =========================================================================
# Session → Exchange timezone mapping
# =========================================================================
# For each fixed session, define which exchange timezone matters and what
# local event it corresponds to.

SESSION_EXCHANGE_TZ = {
    "0900": {
        "tz": _US_CHICAGO,
        "event": "CME Globex open 5PM CT",
        "note": "CME names sessions by the NEXT business day — 5PM Thu CT = CME Friday session",
        "cme_next_day_convention": True,  # CME's 5PM = start of NEXT trading day
    },
    "1000": {
        "tz": _TOKYO,
        "event": "Tokyo open 9AM JST",
        "note": "Japan has no DST. Brisbane 1000 = 0000 UTC = 0900 JST. Same calendar day.",
        "cme_next_day_convention": False,
    },
    "1100": {
        "tz": ZoneInfo("Asia/Singapore"),
        "event": "Singapore/Shanghai open",
        "note": "No DST in Asia. Brisbane 1100 = 0100 UTC. Same calendar day.",
        "cme_next_day_convention": False,
    },
    "1130": {
        "tz": ZoneInfo("Asia/Hong_Kong"),
        "event": "HK/SG equity open 9:30 HKT",
        "note": "No DST. Brisbane 1130 = 0130 UTC = 0930 HKT. Same calendar day.",
        "cme_next_day_convention": False,
    },
    "1800": {
        "tz": _UK_LONDON,
        "event": "London metals open 8AM LT",
        "note": "Brisbane 1800 = 0800 UTC. Same calendar day (UTC day = London day in morning).",
        "cme_next_day_convention": False,
    },
    "2300": {
        "tz": _US_EASTERN,
        "event": "Near US data release 8:30 ET (never aligned — see DST notes)",
        "note": "Brisbane 2300 = 1300 UTC. Same calendar day as US (1300 UTC = 8AM ET winter / 9AM ET summer).",
        "cme_next_day_convention": False,
    },
    "0030": {
        "tz": _US_EASTERN,
        "event": "US equity open 9:30 ET",
        "note": "Brisbane 0030 = 1430 UTC = 9:30 AM ET (winter). CROSSES MIDNIGHT Brisbane: Brisbane date is +1 vs UTC date.",
        "cme_next_day_convention": False,
    },
}


def brisbane_to_utc(brisbane_date: date, hour: int, minute: int) -> datetime:
    """Convert Brisbane local time to UTC datetime."""
    bris_dt = datetime(brisbane_date.year, brisbane_date.month, brisbane_date.day,
                       hour, minute, 0, tzinfo=_BRISBANE)
    return bris_dt.astimezone(_UTC)


def utc_to_exchange(utc_dt: datetime, exchange_tz: ZoneInfo) -> datetime:
    """Convert UTC datetime to exchange local time."""
    return utc_dt.astimezone(exchange_tz)


# =========================================================================
# Part 1: Static mapping table
# =========================================================================

def build_static_mapping():
    """For each session, map Brisbane DOW → UTC time → exchange calendar day.

    Tests on a winter date and a summer date to show DST effects.
    """
    print(f"\n{'=' * 110}")
    print(f"  PART 1: STATIC DOW MAPPING — Brisbane DOW vs Exchange Calendar Day")
    print(f"{'=' * 110}")

    # Pick concrete dates: one winter (Jan, no DST anywhere except southern hemisphere),
    # one summer (Jul, US/UK DST active)
    # We want Mon-Fri for each.
    # Week of 2025-01-06 (Mon) = US winter, UK winter
    # Week of 2025-07-07 (Mon) = US summer (EDT), UK summer (BST)
    test_weeks = {
        "Winter (Jan 2025)": date(2025, 1, 6),  # Monday
        "Summer (Jul 2025)": date(2025, 7, 7),   # Monday
    }

    rows = []

    for season_label, monday in test_weeks.items():
        print(f"\n  --- {season_label} ---")

        for session_label, info in SESSION_EXCHANGE_TZ.items():
            bris_h = int(session_label[:2])
            bris_m = int(session_label[2:])
            exchange_tz = info["tz"]
            cme_next = info.get("cme_next_day_convention", False)

            print(f"\n  {session_label} ({info['event']}):")
            print(f"    {'Bris DOW':>10s} → {'Bris Date':>12s}  {'UTC':>18s}  {'Exch Local':>20s}  {'Exch DOW':>10s}  {'Match?':>7s}")
            print(f"    {'-' * 90}")

            for offset in range(5):  # Mon-Fri
                bris_date = monday + timedelta(days=offset)
                bris_dow = bris_date.weekday()

                utc_dt = brisbane_to_utc(bris_date, bris_h, bris_m)
                exch_dt = utc_to_exchange(utc_dt, exchange_tz)

                # For CME, the "trading day" is the NEXT business day when the session starts at 5PM
                if cme_next:
                    # CME convention: 5PM CT opens the NEXT trading day
                    # So Thu 5PM CT = Friday CME session
                    cme_trading_day = exch_dt.date() + timedelta(days=1)
                    exch_dow = cme_trading_day.weekday()
                    exch_dow_label = f"{DOW_NAMES[exch_dt.weekday()]}→{DOW_NAMES[exch_dow]}(CME)"
                else:
                    exch_dow = exch_dt.weekday()
                    exch_dow_label = DOW_NAMES[exch_dow]

                match = "✓" if bris_dow == exch_dow else "✗ DIFF"

                print(f"    {DOW_NAMES[bris_dow]:>10s} → {bris_date!s:>12s}  "
                      f"{utc_dt.strftime('%Y-%m-%d %H:%M'):>18s}  "
                      f"{exch_dt.strftime('%Y-%m-%d %H:%M %Z'):>20s}  "
                      f"{exch_dow_label:>10s}  {match:>7s}")

                rows.append({
                    "season": season_label,
                    "session": session_label,
                    "brisbane_date": str(bris_date),
                    "brisbane_dow": bris_dow,
                    "brisbane_dow_name": DOW_NAMES[bris_dow],
                    "utc_time": utc_dt.strftime('%Y-%m-%d %H:%M'),
                    "exchange_time": exch_dt.strftime('%Y-%m-%d %H:%M %Z'),
                    "exchange_dow": exch_dow,
                    "exchange_dow_name": DOW_NAMES[exch_dow],
                    "aligned": bris_dow == exch_dow,
                    "cme_convention": cme_next,
                })

    return rows


# =========================================================================
# Part 2: Validate existing DOW filters
# =========================================================================

def validate_existing_filters(static_rows):
    """Check whether NOFRI@0900, NOMON@1800, NOTUE@1000 target the right exchange day."""
    print(f"\n{'=' * 110}")
    print(f"  PART 2: EXISTING DOW FILTER ALIGNMENT CHECK")
    print(f"{'=' * 110}")

    filters_to_check = [
        ("0900", "NOFRI", 4, "Friday", "CME Friday session (position-squaring)"),
        ("1800", "NOMON", 0, "Monday", "London Monday open (thin, no follow-through)"),
        ("1000", "NOTUE", 1, "Tuesday", "Tokyo Tuesday (consistently weakest)"),
    ]

    for session, filter_name, skip_dow, skip_day_name, hypothesis in filters_to_check:
        print(f"\n  {filter_name} @ {session}: skip Brisbane-{skip_day_name}")
        print(f"  Hypothesis: {hypothesis}")

        # Check winter and summer
        for season in ["Winter (Jan 2025)", "Summer (Jul 2025)"]:
            matching = [r for r in static_rows
                       if r["session"] == session
                       and r["season"] == season
                       and r["brisbane_dow"] == skip_dow]

            if not matching:
                print(f"    {season}: NO DATA")
                continue

            r = matching[0]
            if r["aligned"]:
                verdict = f"✓ ALIGNED — Brisbane {skip_day_name} = Exchange {r['exchange_dow_name']}"
            else:
                verdict = (f"✗ MISALIGNED — Brisbane {skip_day_name} = "
                          f"Exchange {r['exchange_dow_name']} "
                          f"(UTC: {r['utc_time']}, Exch: {r['exchange_time']})")

            print(f"    {season}: {verdict}")

    # Special analysis for 0030
    print(f"\n  --- 0030 (US equity open) — NOT currently filtered, but important to document ---")
    for season in ["Winter (Jan 2025)", "Summer (Jul 2025)"]:
        for dow in range(5):
            matching = [r for r in static_rows
                       if r["session"] == "0030"
                       and r["season"] == season
                       and r["brisbane_dow"] == dow]
            if matching:
                r = matching[0]
                status = "✓" if r["aligned"] else f"→ Exchange {r['exchange_dow_name']}"
                print(f"    {season} Brisbane-{DOW_NAMES[dow]}: "
                      f"UTC={r['utc_time']}, Exchange={r['exchange_time']} {status}")


# =========================================================================
# Part 3: Data-driven — compare Brisbane DOW vs exchange DOW grouping
# =========================================================================

def load_outcomes(con, instrument, session):
    """Load orb_outcomes joined with daily_features for DOW analysis."""
    return con.execute("""
        SELECT
            oo.trading_day,
            oo.pnl_r,
            oo.orb_size_r,
            df.day_of_week as brisbane_dow
        FROM orb_outcomes oo
        JOIN daily_features df
            ON oo.symbol = df.symbol
            AND oo.trading_day = df.trading_day
            AND oo.orb_minutes = df.orb_minutes
        WHERE oo.symbol = ?
          AND oo.orb_label = ?
          AND oo.entry_model = 'E1'
          AND oo.rr_target = 2.0
          AND df.day_of_week IS NOT NULL
        ORDER BY oo.trading_day
    """, [instrument, session]).fetchdf()


def compute_exchange_dow(trading_day: date, session: str) -> int:
    """Compute the exchange-timezone day of week for a Brisbane trading day + session."""
    info = SESSION_EXCHANGE_TZ.get(session)
    if info is None:
        return trading_day.weekday()

    bris_h = int(session[:2])
    bris_m = int(session[2:])

    utc_dt = brisbane_to_utc(trading_day, bris_h, bris_m)
    exch_dt = utc_to_exchange(utc_dt, info["tz"])

    if info.get("cme_next_day_convention", False):
        return (exch_dt.date() + timedelta(days=1)).weekday()

    return exch_dt.weekday()


def q3_data_driven(con):
    """For sessions with DOW filters, group by both Brisbane DOW and exchange DOW.

    If they're aligned, the two groupings will give identical results.
    If not, the exchange DOW grouping should show the "real" pattern.
    """
    print(f"\n{'=' * 110}")
    print(f"  PART 3: DATA-DRIVEN — Brisbane DOW vs Exchange DOW Grouping")
    print(f"  (Do the two DOW definitions show different patterns?)")
    print(f"{'=' * 110}")

    all_rows = []

    for instrument in ["MGC", "MNQ", "MES"]:
        for session in ["0900", "1000", "1100", "1800", "0030", "2300"]:
            df = load_outcomes(con, instrument, session)
            if len(df) < 30:
                continue

            # Compute exchange DOW for each row
            df["exchange_dow"] = df["trading_day"].apply(
                lambda td: compute_exchange_dow(td, session)
            )

            # Filter to G4+ (orb_size_r >= 4.0)
            df_g4 = df[df["orb_size_r"] >= 4.0].copy()
            if len(df_g4) < 30:
                continue

            # Check if Brisbane DOW == Exchange DOW for all rows
            mismatches = (df_g4["brisbane_dow"] != df_g4["exchange_dow"]).sum()

            if mismatches == 0:
                # Perfect alignment — skip detailed comparison
                continue

            # There ARE mismatches — show both groupings
            print(f"\n  {instrument} {session} G4+ (N={len(df_g4)}, {mismatches} DOW mismatches)")
            print(f"    {'':>10s}  {'--- Brisbane DOW ---':>30s}  {'--- Exchange DOW ---':>30s}")
            print(f"    {'Day':>10s}  {'N':>5s} {'avgR':>7s} {'WR':>6s} {'totR':>8s}  "
                  f"{'N':>5s} {'avgR':>7s} {'WR':>6s} {'totR':>8s}")
            print(f"    {'-' * 75}")

            for dow in range(5):
                # Brisbane grouping
                b_mask = df_g4["brisbane_dow"] == dow
                b_rs = df_g4.loc[b_mask, "pnl_r"]
                b_n = len(b_rs)
                b_avg = b_rs.mean() if b_n > 0 else 0
                b_wr = (b_rs > 0).mean() if b_n > 0 else 0
                b_tot = b_rs.sum() if b_n > 0 else 0

                # Exchange grouping
                e_mask = df_g4["exchange_dow"] == dow
                e_rs = df_g4.loc[e_mask, "pnl_r"]
                e_n = len(e_rs)
                e_avg = e_rs.mean() if e_n > 0 else 0
                e_wr = (e_rs > 0).mean() if e_n > 0 else 0
                e_tot = e_rs.sum() if e_n > 0 else 0

                diff_marker = ""
                if b_n > 0 and e_n > 0 and abs(b_avg - e_avg) > 0.05:
                    diff_marker = " <<"

                print(f"    {DOW_NAMES[dow]:>10s}  "
                      f"{b_n:5d} {b_avg:+7.3f} {b_wr:5.1%} {b_tot:+8.1f}  "
                      f"{e_n:5d} {e_avg:+7.3f} {e_wr:5.1%} {e_tot:+8.1f}{diff_marker}")

                all_rows.append({
                    "instrument": instrument, "session": session,
                    "dow": dow, "dow_name": DOW_NAMES[dow],
                    "bris_n": b_n, "bris_avg_r": round(b_avg, 4),
                    "bris_wr": round(b_wr, 4), "bris_tot_r": round(b_tot, 2),
                    "exch_n": e_n, "exch_avg_r": round(e_avg, 4),
                    "exch_wr": round(e_wr, 4), "exch_tot_r": round(e_tot, 2),
                    "mismatches": mismatches,
                })

    return all_rows


# =========================================================================
# Part 4: Summary and recommendations
# =========================================================================

def print_summary():
    print(f"\n{'=' * 110}")
    print(f"  PART 4: CANONICAL DOW ALIGNMENT SUMMARY")
    print(f"{'=' * 110}")

    print("""
  SESSION-BY-SESSION ANALYSIS:

  0900 (CME open, 23:00 UTC):
    Brisbane Friday 09:00 = UTC Thursday 23:00 = CT Thursday 5:00 PM
    CME convention: 5PM CT Thursday = START of CME Friday session
    → Brisbane Friday = CME Friday ✓ ALIGNED
    NOFRI correctly skips CME Friday (position-squaring into weekend)

  1000 (Tokyo open, 00:00 UTC):
    Brisbane Tuesday 10:00 = UTC Tuesday 00:00 = JST Tuesday 09:00
    Japan has no DST.
    → Brisbane Tuesday = Tokyo Tuesday ✓ ALIGNED
    NOTUE correctly skips Tokyo Tuesday

  1100 (Singapore/Shanghai, 01:00 UTC):
    Brisbane DOW = Exchange DOW for all days ✓ ALIGNED
    No DST in SE Asia.

  1130 (HK/SG equity open, 01:30 UTC):
    Brisbane DOW = Exchange DOW for all days ✓ ALIGNED
    No DST in HK/SG.

  1800 (London open, 08:00 UTC):
    Brisbane Monday 18:00 = UTC Monday 08:00 = London Monday 08:00 (winter)
    Brisbane Monday 18:00 = UTC Monday 08:00 = London Monday 09:00 (summer BST)
    Either way, it's London Monday.
    → Brisbane Monday = London Monday ✓ ALIGNED
    NOMON correctly skips London Monday (thin open, no follow-through)

  2300 (US pre-data, 13:00 UTC):
    Brisbane Friday 23:00 = UTC Friday 13:00 = ET Friday 8:00 (winter) or 9:00 (summer)
    → Brisbane Friday = US Friday ✓ ALIGNED
    (No DOW filter currently applied — correct, no research basis)

  0030 (US equity open, 14:30 UTC):
    Brisbane Friday 00:30 = UTC Thursday 14:30 = ET Thursday 9:30 AM
    → Brisbane Friday = US THURSDAY ✗ MISALIGNED BY -1 DAY
    Brisbane DOW is always +1 relative to US DOW for this session.

    Implication: if DOW research at 0030 finds "Friday is bad," that's actually
    "US Thursday is bad." The DOW research script uses Brisbane DOW, so any
    findings are about the PREVIOUS US calendar day.

    This doesn't affect current production: 0030 has NO DOW filter in the grid.
    But it MUST be accounted for if DOW filters are ever added for 0030.

  CONCLUSION:
    All three ACTIVE DOW filters (NOFRI@0900, NOMON@1800, NOTUE@1000) are
    correctly aligned with their exchange-timezone targets.

    Only 0030 has a Brisbane↔Exchange DOW mismatch (-1 day). Since 0030 has
    no DOW filter, this is currently harmless. But the mapping must be
    documented and any future 0030 DOW analysis must use exchange DOW.
""")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="DOW Alignment Investigation")
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    if args.db_path:
        db_path = Path(args.db_path)
    else:
        try:
            from pipeline.paths import GOLD_DB_PATH
            db_path = GOLD_DB_PATH
        except ImportError:
            db_path = Path("gold.db")

    print(f"\n{'=' * 110}")
    print(f"  DOW ALIGNMENT INVESTIGATION — Brisbane DOW vs Exchange-Timezone DOW")
    print(f"  Database: {db_path}")
    print(f"{'=' * 110}")

    t_start = time.time()

    # Part 1: Static mapping
    static_rows = build_static_mapping()

    # Part 2: Validate existing filters
    validate_existing_filters(static_rows)

    # Part 3: Data-driven comparison (needs DB)
    data_rows = []
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            data_rows = q3_data_driven(con)
        finally:
            con.close()
    except Exception as e:
        print(f"\n  Part 3 skipped (DB error: {e})")

    # Part 4: Summary
    print_summary()

    # Save outputs
    out = Path("research/output")
    out.mkdir(parents=True, exist_ok=True)

    if static_rows:
        pd.DataFrame(static_rows).to_csv(
            out / "dow_alignment_static_mapping.csv",
            index=False)

    if data_rows:
        pd.DataFrame(data_rows).to_csv(
            out / "dow_alignment_data_comparison.csv",
            index=False, float_format="%.4f")

    print(f"\n  CSVs saved to research/output/dow_alignment_*.csv")
    print(f"  Total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
