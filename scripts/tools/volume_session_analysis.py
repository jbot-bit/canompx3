#!/usr/bin/env python3
"""
Volume Session Analysis — determine which ORB sessions are viable for an instrument.

Analyzes 1-minute bar volume across the 24-hour trading day in Brisbane time,
bucketed into 15-minute slots (96 total). For each slot, measures:
  - avg_volume: mean volume per bar across all trading days
  - bar_coverage: % of trading days that have bars at this time
  - orb_viability: % of trading days with all 5 consecutive 1m bars (needed for 5m ORB)

Maps viable times to SESSION_CATALOG sessions and recommends enabled_sessions.

Usage:
    python scripts/tools/volume_session_analysis.py --instrument SIL
    python scripts/tools/volume_session_analysis.py --instrument SIL --reference MGC
    python scripts/tools/volume_session_analysis.py --instrument SIL --db-path C:/db/gold.db
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = PROJECT_ROOT / "gold.db"

# 96 time slots across 24 hours
CANDIDATE_TIMES = [(h, m) for h in range(24) for m in (0, 15, 30, 45)]

# Session mappings: Brisbane (hour, minute) → session label
# Fixed sessions from SESSION_CATALOG
FIXED_SESSIONS = {
    (9, 0): "0900",
    (10, 0): "1000",
    (11, 0): "1100",
    (11, 30): "1130",
    (18, 0): "1800",
    (23, 0): "2300",
    (0, 30): "0030",
}

# Dynamic sessions and their Brisbane-time ranges (winter/summer)
# These shift with DST, so we show the range
DYNAMIC_SESSION_TIMES = {
    "CME_OPEN": {"winter": (9, 0), "summer": (8, 0)},
    "LONDON_OPEN": {"winter": (18, 0), "summer": (17, 0)},
    "US_EQUITY_OPEN": {"winter": (0, 30), "summer": (23, 30)},
    "US_DATA_OPEN": {"winter": (23, 30), "summer": (22, 30)},
    "US_POST_EQUITY": {"winter": (1, 0), "summer": (0, 0)},
    "CME_CLOSE": {"winter": (5, 45), "summer": (4, 45)},
}

# Viability thresholds
# Note: bar_coverage naturally caps ~88% even for liquid instruments (CME
# holidays, maintenance, early closes). 85% is the practical maximum.
VIABLE_BAR_COV = 85.0      # % of trading days with bars
VIABLE_AVG_VOL = 50         # mean volume per bar
VIABLE_ORB_OK = 80.0        # % of days with complete 5-bar ORB window
MARGINAL_BAR_COV = 70.0
MARGINAL_AVG_VOL = 20


# ── SQL Queries ──────────────────────────────────────────────────────

VOLUME_PROFILE_SQL = """
WITH bars AS (
    SELECT
        ts_utc,
        volume,
        EXTRACT(HOUR FROM ts_utc AT TIME ZONE 'Australia/Brisbane')::INT AS bris_h,
        EXTRACT(MINUTE FROM ts_utc AT TIME ZONE 'Australia/Brisbane')::INT AS bris_m,
        (EXTRACT(MINUTE FROM ts_utc AT TIME ZONE 'Australia/Brisbane')::INT // 15) * 15 AS bris_m15,
        CASE
            WHEN EXTRACT(HOUR FROM ts_utc AT TIME ZONE 'Australia/Brisbane') < 9
            THEN (ts_utc AT TIME ZONE 'Australia/Brisbane')::DATE - INTERVAL '1 day'
            ELSE (ts_utc AT TIME ZONE 'Australia/Brisbane')::DATE
        END AS trading_day
    FROM bars_1m
    WHERE symbol = $1
)
SELECT
    bris_h,
    bris_m15 AS bris_m,
    COUNT(*) AS total_bars,
    COUNT(DISTINCT trading_day) AS days_with_bars,
    SUM(volume) AS total_volume,
    AVG(volume) AS avg_volume
FROM bars
GROUP BY bris_h, bris_m15
ORDER BY bris_h, bris_m15
"""

TOTAL_TRADING_DAYS_SQL = """
SELECT COUNT(DISTINCT
    CASE
        WHEN EXTRACT(HOUR FROM ts_utc AT TIME ZONE 'Australia/Brisbane') < 9
        THEN (ts_utc AT TIME ZONE 'Australia/Brisbane')::DATE - INTERVAL '1 day'
        ELSE (ts_utc AT TIME ZONE 'Australia/Brisbane')::DATE
    END
) AS n_days
FROM bars_1m
WHERE symbol = $1
"""

# Check ORB viability: for each 15-min slot, count days where all 5
# consecutive 1m bars starting at that time exist
ORB_VIABILITY_SQL = """
WITH bars AS (
    SELECT
        ts_utc,
        EXTRACT(HOUR FROM ts_utc AT TIME ZONE 'Australia/Brisbane')::INT AS bris_h,
        EXTRACT(MINUTE FROM ts_utc AT TIME ZONE 'Australia/Brisbane')::INT AS bris_m,
        CASE
            WHEN EXTRACT(HOUR FROM ts_utc AT TIME ZONE 'Australia/Brisbane') < 9
            THEN (ts_utc AT TIME ZONE 'Australia/Brisbane')::DATE - INTERVAL '1 day'
            ELSE (ts_utc AT TIME ZONE 'Australia/Brisbane')::DATE
        END AS trading_day
    FROM bars_1m
    WHERE symbol = $1
),
-- For each trading day and each candidate start time, count bars in [start, start+5min)
orb_windows AS (
    SELECT
        b.trading_day,
        slot.bris_h AS slot_h,
        slot.bris_m AS slot_m,
        COUNT(*) AS bars_in_window
    FROM bars b
    CROSS JOIN (
        SELECT unnest(generate_series(0, 23)) AS bris_h,
               unnest_m AS bris_m
        FROM (SELECT unnest([0, 15, 30, 45]) AS unnest_m)
    ) slot
    WHERE b.bris_h = slot.bris_h
      AND b.bris_m >= slot.bris_m
      AND b.bris_m < slot.bris_m + 5
    GROUP BY b.trading_day, slot.bris_h, slot.bris_m
)
SELECT
    slot_h AS bris_h,
    slot_m AS bris_m,
    COUNT(*) AS days_checked,
    SUM(CASE WHEN bars_in_window >= 5 THEN 1 ELSE 0 END) AS days_with_5bars
FROM orb_windows
GROUP BY slot_h, slot_m
ORDER BY slot_h, slot_m
"""

# Simpler ORB viability: count bars in each 5-min window per trading day
ORB_VIABILITY_SIMPLE_SQL = """
WITH bars AS (
    SELECT
        EXTRACT(HOUR FROM ts_utc AT TIME ZONE 'Australia/Brisbane')::INT AS bris_h,
        EXTRACT(MINUTE FROM ts_utc AT TIME ZONE 'Australia/Brisbane')::INT AS bris_m,
        CASE
            WHEN EXTRACT(HOUR FROM ts_utc AT TIME ZONE 'Australia/Brisbane') < 9
            THEN (ts_utc AT TIME ZONE 'Australia/Brisbane')::DATE - INTERVAL '1 day'
            ELSE (ts_utc AT TIME ZONE 'Australia/Brisbane')::DATE
        END AS trading_day
    FROM bars_1m
    WHERE symbol = $1
),
-- For each slot's :00-:04 window, count bars per day
slot_bars AS (
    SELECT
        trading_day,
        bris_h,
        (bris_m // 15) * 15 AS slot_m,
        -- Count bars in the first 5 minutes of this 15-min slot
        SUM(CASE WHEN bris_m % 15 < 5 THEN 1 ELSE 0 END) AS first_5_bars
    FROM bars
    GROUP BY trading_day, bris_h, (bris_m // 15) * 15
)
SELECT
    bris_h,
    slot_m AS bris_m,
    COUNT(*) AS days_checked,
    SUM(CASE WHEN first_5_bars >= 5 THEN 1 ELSE 0 END) AS days_orb_ok
FROM slot_bars
GROUP BY bris_h, slot_m
ORDER BY bris_h, slot_m
"""


# ── Helpers ──────────────────────────────────────────────────────────

def get_db_path(args):
    if args.db_path:
        return Path(args.db_path)
    env = os.environ.get("DUCKDB_PATH")
    if env:
        return Path(env)
    return DEFAULT_DB


def classify_slot(bar_cov, avg_vol, orb_pct):
    """Classify a time slot's viability for ORB trading."""
    if bar_cov >= VIABLE_BAR_COV and avg_vol >= VIABLE_AVG_VOL and orb_pct >= VIABLE_ORB_OK:
        return "VIABLE"
    if bar_cov >= MARGINAL_BAR_COV and avg_vol >= MARGINAL_AVG_VOL:
        return "MARGINAL"
    return "DEAD"


def session_match(bris_h, bris_m):
    """Find which session(s) this time slot matches."""
    matches = []

    # Check fixed sessions
    fixed = FIXED_SESSIONS.get((bris_h, bris_m))
    if fixed:
        matches.append(fixed)

    # Check dynamic sessions
    for name, times in DYNAMIC_SESSION_TIMES.items():
        if times["winter"] == (bris_h, bris_m):
            matches.append(f"{name}(W)")
        elif times["summer"] == (bris_h, bris_m):
            matches.append(f"{name}(S)")

    return " / ".join(matches) if matches else ""


# ── Core Analysis ────────────────────────────────────────────────────

def analyze_instrument(con, instrument):
    """Run full volume profile analysis for an instrument.

    Returns dict with:
        total_days: number of trading days
        slots: list of slot dicts with volume/coverage/viability data
    """
    # Total trading days
    total_days = con.execute(TOTAL_TRADING_DAYS_SQL, [instrument]).fetchone()[0]
    if total_days == 0:
        print(f"  WARNING: No bars found for {instrument}")
        return {"total_days": 0, "slots": []}

    # Volume profile
    vol_rows = con.execute(VOLUME_PROFILE_SQL, [instrument]).fetchall()
    vol_by_slot = {}
    for bris_h, bris_m, total_bars, days_with, total_vol, avg_vol in vol_rows:
        vol_by_slot[(int(bris_h), int(bris_m))] = {
            "total_bars": int(total_bars),
            "days_with_bars": int(days_with),
            "total_volume": int(total_vol),
            "avg_volume": float(avg_vol),
        }

    # ORB viability
    orb_rows = con.execute(ORB_VIABILITY_SIMPLE_SQL, [instrument]).fetchall()
    orb_by_slot = {}
    for bris_h, bris_m, days_checked, days_ok in orb_rows:
        orb_by_slot[(int(bris_h), int(bris_m))] = {
            "days_checked": int(days_checked),
            "days_orb_ok": int(days_ok),
        }

    # Build combined slot data
    slots = []
    for bris_h, bris_m in CANDIDATE_TIMES:
        vol = vol_by_slot.get((bris_h, bris_m), {})
        orb = orb_by_slot.get((bris_h, bris_m), {})

        days_with = vol.get("days_with_bars", 0)
        avg_vol = vol.get("avg_volume", 0.0)
        bar_cov = (days_with / total_days * 100) if total_days > 0 else 0.0

        days_orb_ok = orb.get("days_orb_ok", 0)
        days_checked = orb.get("days_checked", 0)
        orb_pct = (days_orb_ok / days_checked * 100) if days_checked > 0 else 0.0

        status = classify_slot(bar_cov, avg_vol, orb_pct)
        sess = session_match(bris_h, bris_m)

        slots.append({
            "bris_h": bris_h,
            "bris_m": bris_m,
            "avg_volume": avg_vol,
            "bar_coverage": bar_cov,
            "orb_viability": orb_pct,
            "days_with_bars": days_with,
            "total_days": total_days,
            "status": status,
            "session": sess,
        })

    return {"total_days": total_days, "slots": slots}


def recommend_sessions(slots):
    """Determine recommended enabled_sessions based on viability.

    Returns list of session labels where the time slot is VIABLE.
    """
    recommended = []
    seen = set()

    for slot in slots:
        if slot["status"] != "VIABLE":
            continue
        bris_h, bris_m = slot["bris_h"], slot["bris_m"]

        # Check fixed sessions
        fixed = FIXED_SESSIONS.get((bris_h, bris_m))
        if fixed and fixed not in seen:
            recommended.append(fixed)
            seen.add(fixed)

        # Check dynamic sessions — recommend if EITHER winter or summer time is viable
        for name, times in DYNAMIC_SESSION_TIMES.items():
            if name in seen:
                continue
            if times["winter"] == (bris_h, bris_m) or times["summer"] == (bris_h, bris_m):
                recommended.append(name)
                seen.add(name)

    return recommended


# ── Output ───────────────────────────────────────────────────────────

def print_profile(instrument, result, reference_result=None, ref_instrument=None):
    """Print the full volume profile analysis."""
    slots = result["slots"]
    total_days = result["total_days"]

    print(f"\n{'='*72}")
    print(f"  {instrument} VOLUME PROFILE (Brisbane Time)")
    print(f"  {total_days} trading days analyzed")
    print(f"{'='*72}")
    print()

    # Thresholds legend
    print(f"  Thresholds: VIABLE = barCov>{VIABLE_BAR_COV}% + avgVol>{VIABLE_AVG_VOL}"
          f" + ORB5>{VIABLE_ORB_OK}%")
    print(f"              MARGINAL = barCov>{MARGINAL_BAR_COV}% + avgVol>{MARGINAL_AVG_VOL}")
    print(f"              DEAD = below marginal")
    print()

    # Header
    header = f"  {'Time':<7} {'AvgVol':>7} {'BarCov%':>8} {'ORB5ok%':>8} {'Status':<10} {'Session Match'}"
    print(header)
    print(f"  {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*10} {'-'*20}")

    # Only print non-DEAD slots (or all if --verbose)
    for slot in slots:
        if slot["status"] == "DEAD" and not slot["session"]:
            continue

        bh, bm = slot["bris_h"], slot["bris_m"]
        time_str = f"{bh:02d}:{bm:02d}"
        avg_vol = f"{slot['avg_volume']:.0f}"
        bar_cov = f"{slot['bar_coverage']:.1f}%"
        orb_ok = f"{slot['orb_viability']:.1f}%"
        status = slot["status"]
        sess = slot["session"]

        # Color-code status
        if status == "VIABLE":
            marker = "+"
        elif status == "MARGINAL":
            marker = "~"
        else:
            marker = " "

        print(f" {marker} {time_str:<7} {avg_vol:>7} {bar_cov:>8} {orb_ok:>8} {status:<10} {sess}")

    # Recommendations
    recommended = recommend_sessions(slots)
    print()
    print(f"  RECOMMENDED enabled_sessions for {instrument}:")
    if recommended:
        sessions_str = ", ".join(f'"{s}"' for s in recommended)
        print(f"    [{sessions_str}]")
    else:
        print("    [] (no sessions meet viability thresholds)")

    # Summary stats
    n_viable = sum(1 for s in slots if s["status"] == "VIABLE")
    n_marginal = sum(1 for s in slots if s["status"] == "MARGINAL")
    n_dead = sum(1 for s in slots if s["status"] == "DEAD")
    print()
    print(f"  Viable: {n_viable}/96 slots | Marginal: {n_marginal}/96 | Dead: {n_dead}/96")

    # Comparison with reference
    if reference_result and ref_instrument:
        ref_slots = reference_result["slots"]
        ref_viable = sum(1 for s in ref_slots if s["status"] == "VIABLE")
        ref_recommended = recommend_sessions(ref_slots)

        print()
        print(f"  COMPARISON vs {ref_instrument}:")
        print(f"    {instrument} viable: {n_viable}/96 | {ref_instrument} viable: {ref_viable}/96")
        print(f"    {instrument} sessions: {len(recommended)} | {ref_instrument} sessions: {len(ref_recommended)}")

        # Find dead zones unique to this instrument
        inst_dead_times = {(s["bris_h"], s["bris_m"]) for s in slots if s["status"] == "DEAD"}
        ref_viable_times = {(s["bris_h"], s["bris_m"]) for s in ref_slots if s["status"] == "VIABLE"}
        only_dead_here = inst_dead_times & ref_viable_times
        if only_dead_here:
            # Group consecutive dead slots
            sorted_dead = sorted(only_dead_here)
            print(f"    Dead in {instrument} but viable in {ref_instrument}: "
                  f"{len(sorted_dead)} slots")

    print()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze volume profile to determine viable ORB sessions"
    )
    parser.add_argument("--instrument", required=True, help="Instrument to analyze (e.g., SIL)")
    parser.add_argument("--reference", default=None, help="Reference instrument for comparison (e.g., MGC)")
    parser.add_argument("--db-path", default=None, help="Path to DuckDB database")
    args = parser.parse_args()

    db_path = get_db_path(args)
    instrument = args.instrument.upper()

    print(f"Database: {db_path}")
    print(f"Instrument: {instrument}")

    con = duckdb.connect(str(db_path), read_only=True)

    # Verify instrument has data
    n_bars = con.execute(
        "SELECT COUNT(*) FROM bars_1m WHERE symbol = $1", [instrument]
    ).fetchone()[0]
    print(f"  {instrument}: {n_bars:,} bars in bars_1m")

    if n_bars == 0:
        print(f"\nFATAL: No bars found for {instrument}. Run ingestion first.")
        con.close()
        sys.exit(1)

    # Analyze target instrument
    print(f"\nAnalyzing {instrument} volume profile...")
    result = analyze_instrument(con, instrument)

    # Analyze reference instrument if requested
    ref_result = None
    if args.reference:
        ref = args.reference.upper()
        ref_bars = con.execute(
            "SELECT COUNT(*) FROM bars_1m WHERE symbol = $1", [ref]
        ).fetchone()[0]
        print(f"  {ref}: {ref_bars:,} bars in bars_1m")
        if ref_bars > 0:
            print(f"Analyzing {ref} volume profile...")
            ref_result = analyze_instrument(con, ref)
        else:
            print(f"  WARNING: No bars for reference {ref}, skipping comparison")

    con.close()

    # Print results
    print_profile(instrument, result, ref_result, args.reference.upper() if args.reference else None)

    if ref_result and args.reference:
        print_profile(args.reference.upper(), ref_result)


if __name__ == "__main__":
    main()
