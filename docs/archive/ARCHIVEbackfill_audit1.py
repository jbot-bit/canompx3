#!/usr/bin/env python3
"""
Backfill Audit Script

Checks data coverage and identifies gaps in:
- Raw DBN files (1m OHLSV source files)
- bars_1m (raw price data in DB)
- bars_5m (aggregated bars)
- daily_features (processed features)

Reports:
- Date range covered
- Missing dates (gaps)
- Bar counts per day
- Coverage comparison between tables
- Raw file vs DB comparison
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent

import duckdb
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
from pipeline.paths import GOLD_DB_PATH

TZ_LOCAL = ZoneInfo("Australia/Brisbane")

def get_connection(db_path: str = GOLD_DB_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(db_path, read_only=True)

def audit_bars_1m(conn: duckdb.DuckDBPyConnection, symbol: str = "MGC") -> dict:
    """Audit bars_1m table for coverage and gaps."""

    # Get date range and daily counts
    query = """
        SELECT
            DATE_TRUNC('day', ts_utc AT TIME ZONE 'UTC' AT TIME ZONE 'Australia/Brisbane')::DATE AS date_local,
            COUNT(*) AS bar_count,
            MIN(ts_utc) AS first_bar,
            MAX(ts_utc) AS last_bar
        FROM bars_1m
        WHERE symbol = ?
        GROUP BY date_local
        ORDER BY date_local
    """
    rows = conn.execute(query, [symbol]).fetchall()

    if not rows:
        return {"error": f"No data found for {symbol}"}

    dates_with_data = {row[0]: row[1] for row in rows}
    first_date = min(dates_with_data.keys())
    last_date = max(dates_with_data.keys())

    # Find gaps (missing weekdays)
    gaps = []
    current = first_date
    while current <= last_date:
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5 and current not in dates_with_data:
            gaps.append(current)
        current += timedelta(days=1)

    # Categorize days by bar count
    low_bar_days = [(d, c) for d, c in dates_with_data.items() if c < 400]
    normal_days = [(d, c) for d, c in dates_with_data.items() if c >= 400]

    return {
        "symbol": symbol,
        "first_date": first_date,
        "last_date": last_date,
        "total_days": len(dates_with_data),
        "total_bars": sum(dates_with_data.values()),
        "gaps": gaps,
        "low_bar_days": low_bar_days,
        "normal_days_count": len(normal_days),
        "dates_with_data": dates_with_data,
    }

def audit_bars_5m(conn: duckdb.DuckDBPyConnection, symbol: str = "MGC") -> dict:
    """Audit bars_5m table for coverage."""

    query = """
        SELECT
            DATE_TRUNC('day', ts_utc AT TIME ZONE 'UTC' AT TIME ZONE 'Australia/Brisbane')::DATE AS date_local,
            COUNT(*) AS bar_count
        FROM bars_5m
        WHERE symbol = ?
        GROUP BY date_local
        ORDER BY date_local
    """
    rows = conn.execute(query, [symbol]).fetchall()

    if not rows:
        return {"total_days": 0, "first_date": None, "last_date": None}

    dates_with_data = {row[0]: row[1] for row in rows}

    return {
        "symbol": symbol,
        "first_date": min(dates_with_data.keys()),
        "last_date": max(dates_with_data.keys()),
        "total_days": len(dates_with_data),
        "total_bars": sum(dates_with_data.values()),
        "dates_with_data": dates_with_data,
    }

def audit_daily_features(conn: duckdb.DuckDBPyConnection, instrument: str = "MGC") -> dict:
    """Audit daily_features table for coverage."""

    query = """
        SELECT
            date_local,
            orb_0900_high IS NOT NULL AS has_0900,
            orb_1000_high IS NOT NULL AS has_1000,
            orb_1100_high IS NOT NULL AS has_1100,
            asia_high IS NOT NULL AS has_asia,
            london_high IS NOT NULL AS has_london,
            ny_high IS NOT NULL AS has_ny
        FROM daily_features
        WHERE instrument = ?
        ORDER BY date_local
    """
    rows = conn.execute(query, [instrument]).fetchall()

    if not rows:
        return {"total_days": 0, "first_date": None, "last_date": None}

    dates = [row[0] for row in rows]

    # Count coverage for each ORB time
    orb_coverage = {
        "0900": sum(1 for row in rows if row[1]),
        "1000": sum(1 for row in rows if row[2]),
        "1100": sum(1 for row in rows if row[3]),
    }

    session_coverage = {
        "asia": sum(1 for row in rows if row[4]),
        "london": sum(1 for row in rows if row[5]),
        "ny": sum(1 for row in rows if row[6]),
    }

    return {
        "instrument": instrument,
        "first_date": min(dates),
        "last_date": max(dates),
        "total_days": len(dates),
        "orb_coverage": orb_coverage,
        "session_coverage": session_coverage,
        "dates": set(dates),
    }

def audit_dbn_file(dbn_path: Path) -> dict:
    """Audit a DBN file for date coverage."""
    try:
        import databento as db
        import pandas as pd
    except ImportError:
        return {"error": "databento not installed - pip install databento"}

    if not dbn_path.exists():
        return {"error": f"File not found: {dbn_path}"}

    print(f"Reading DBN file: {dbn_path.name}...")
    try:
        store = db.DBNStore.from_file(dbn_path)
        df = store.to_df()
    except Exception as e:
        return {"error": f"Failed to read DBN file: {e}"}

    total_records = len(df)
    df = df.reset_index()

    # Get timestamps
    df['ts_utc_dt'] = pd.to_datetime(df['ts_event'], utc=True)

    # Determine trading days (09:00 local -> 09:00 next day)
    df['ts_local'] = df['ts_utc_dt'].dt.tz_convert(str(TZ_LOCAL))
    df['hour'] = df['ts_local'].dt.hour
    df['base_date'] = df['ts_local'].dt.date
    df['trading_day'] = df.apply(
        lambda row: row['base_date'] - timedelta(days=1) if row['hour'] < 9 else row['base_date'],
        axis=1
    )

    # Aggregate by trading day
    daily_counts = df.groupby('trading_day').size().to_dict()

    # Get symbols
    symbols = df['symbol'].unique().tolist() if 'symbol' in df.columns else []

    return {
        "file": str(dbn_path),
        "total_records": total_records,
        "first_date": min(daily_counts.keys()),
        "last_date": max(daily_counts.keys()),
        "total_days": len(daily_counts),
        "daily_counts": daily_counts,
        "symbols": symbols,
    }

def compare_coverage(bars_1m: dict, bars_5m: dict, daily_features: dict) -> list:
    """Compare coverage between tables and identify discrepancies."""

    issues = []

    if not bars_1m.get("dates_with_data"):
        issues.append("CRITICAL: No bars_1m data")
        return issues

    bars_1m_dates = set(bars_1m["dates_with_data"].keys())
    bars_5m_dates = set(bars_5m.get("dates_with_data", {}).keys())
    daily_dates = daily_features.get("dates", set())

    # Days with bars but no daily_features
    bars_no_features = bars_1m_dates - daily_dates
    if bars_no_features:
        issues.append(f"Days with bars_1m but no daily_features: {len(bars_no_features)}")
        recent_missing = sorted(bars_no_features)[-5:]
        issues.append(f"  Recent missing: {recent_missing}")

    # Days with features but no bars
    features_no_bars = daily_dates - bars_1m_dates
    if features_no_bars:
        issues.append(f"Days with daily_features but no bars_1m: {len(features_no_bars)}")

    # bars_1m vs bars_5m
    bars_1m_no_5m = bars_1m_dates - bars_5m_dates
    if bars_1m_no_5m:
        issues.append(f"Days with bars_1m but no bars_5m: {len(bars_1m_no_5m)}")

    return issues

def print_report(bars_1m: dict, bars_5m: dict, daily_features: dict, issues: list):
    """Print the audit report."""

    print("=" * 70)
    print("BACKFILL AUDIT REPORT")
    print("=" * 70)

    # bars_1m summary
    print("\n[bars_1m]")
    if "error" in bars_1m:
        print(f"  ERROR: {bars_1m['error']}")
    else:
        print(f"  Symbol: {bars_1m['symbol']}")
        print(f"  Date range: {bars_1m['first_date']} to {bars_1m['last_date']}")
        print(f"  Total trading days: {bars_1m['total_days']}")
        print(f"  Total bars: {bars_1m['total_bars']:,}")
        print(f"  Normal days (>=400 bars): {bars_1m['normal_days_count']}")

        if bars_1m['gaps']:
            print(f"\n  GAPS (missing weekdays): {len(bars_1m['gaps'])}")
            for gap in bars_1m['gaps'][:10]:
                print(f"    - {gap}")
            if len(bars_1m['gaps']) > 10:
                print(f"    ... and {len(bars_1m['gaps']) - 10} more")
        else:
            print(f"\n  GAPS: None found")

        if bars_1m['low_bar_days']:
            print(f"\n  LOW BAR DAYS (<400 bars): {len(bars_1m['low_bar_days'])}")
            for d, c in sorted(bars_1m['low_bar_days'])[-5:]:
                print(f"    - {d}: {c} bars")

    # bars_5m summary
    print("\n[bars_5m]")
    if bars_5m.get("total_days", 0) == 0:
        print("  No data")
    else:
        print(f"  Date range: {bars_5m['first_date']} to {bars_5m['last_date']}")
        print(f"  Total days: {bars_5m['total_days']}")
        print(f"  Total bars: {bars_5m['total_bars']:,}")

    # daily_features summary
    print("\n[daily_features]")
    if daily_features.get("total_days", 0) == 0:
        print("  No data")
    else:
        print(f"  Instrument: {daily_features['instrument']}")
        print(f"  Date range: {daily_features['first_date']} to {daily_features['last_date']}")
        print(f"  Total days: {daily_features['total_days']}")
        print(f"\n  ORB Coverage:")
        for orb, count in daily_features['orb_coverage'].items():
            pct = (count / daily_features['total_days'] * 100) if daily_features['total_days'] else 0
            print(f"    {orb}: {count}/{daily_features['total_days']} ({pct:.1f}%)")
        print(f"\n  Session Coverage:")
        for sess, count in daily_features['session_coverage'].items():
            pct = (count / daily_features['total_days'] * 100) if daily_features['total_days'] else 0
            print(f"    {sess}: {count}/{daily_features['total_days']} ({pct:.1f}%)")

    # Issues
    print("\n" + "=" * 70)
    print("ISSUES")
    print("=" * 70)
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  None - all tables aligned!")

    # Backfill recommendation
    print("\n" + "=" * 70)
    print("BACKFILL RECOMMENDATION")
    print("=" * 70)

    if "error" not in bars_1m:
        today = date.today()
        last_bar_date = bars_1m['last_date']
        days_behind = (today - last_bar_date).days

        if days_behind > 1:
            print(f"  Data is {days_behind} days behind (last: {last_bar_date})")
            print(f"  Suggested command:")
            print(f"    python pipeline/backfill_range.py {last_bar_date + timedelta(days=1)} {today - timedelta(days=1)}")
        else:
            print(f"  Data is current (last: {last_bar_date})")

    print()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Audit data backfill coverage")
    parser.add_argument("--db", default=GOLD_DB_PATH, help="Database path")
    parser.add_argument("--symbol", default="MGC", help="Symbol to audit")
    parser.add_argument("--dbn", help="Path to DBN file to audit")
    parser.add_argument("--check-mnq", action="store_true", help="Also check MNQ DBN file")
    args = parser.parse_args()

    print(f"Connecting to: {args.db}")
    conn = get_connection(args.db)

    # Check for DBN file
    dbn_result = None
    if args.dbn:
        dbn_result = audit_dbn_file(Path(args.dbn))
    elif args.check_mnq:
        mnq_dbn = PROJECT_ROOT / "data" / "MNQ db" / "glbx-mdp3-20240204-20260203.ohlcv-1m.dbn.zst"
        dbn_result = audit_dbn_file(mnq_dbn)

    # Print DBN audit if available
    if dbn_result:
        print("\n" + "=" * 70)
        print("RAW DBN FILE AUDIT")
        print("=" * 70)
        if "error" in dbn_result:
            print(f"  ERROR: {dbn_result['error']}")
        else:
            print(f"  File: {Path(dbn_result['file']).name}")
            print(f"  Total records: {dbn_result['total_records']:,}")
            print(f"  Date range: {dbn_result['first_date']} to {dbn_result['last_date']}")
            print(f"  Total trading days: {dbn_result['total_days']}")
            print(f"  Symbols: {len(dbn_result['symbols'])} unique contracts")

            # Show some daily stats
            daily = dbn_result['daily_counts']
            avg_bars = sum(daily.values()) / len(daily) if daily else 0
            print(f"  Avg bars/day: {avg_bars:.0f}")

            # Find gaps
            dates_in_file = set(dbn_result['daily_counts'].keys())
            first = dbn_result['first_date']
            last = dbn_result['last_date']
            gaps = []
            current = first
            while current <= last:
                if current.weekday() < 5 and current not in dates_in_file:
                    gaps.append(current)
                current += timedelta(days=1)
            if gaps:
                print(f"\n  GAPS in file: {len(gaps)}")
                for g in gaps[:10]:
                    print(f"    - {g}")
                if len(gaps) > 10:
                    print(f"    ... and {len(gaps) - 10} more")

    # Run DB audits
    bars_1m = audit_bars_1m(conn, args.symbol)
    bars_5m = audit_bars_5m(conn, args.symbol)
    daily_features = audit_daily_features(conn, args.symbol)

    # Compare coverage
    issues = compare_coverage(bars_1m, bars_5m, daily_features)

    # Add DBN vs DB comparison
    if dbn_result and "error" not in dbn_result and "error" not in bars_1m:
        dbn_dates = set(dbn_result['daily_counts'].keys())
        db_dates = set(bars_1m['dates_with_data'].keys())

        in_file_not_db = dbn_dates - db_dates
        in_db_not_file = db_dates - dbn_dates

        if in_file_not_db:
            issues.append(f"Days in DBN file but NOT in DB: {len(in_file_not_db)}")
            recent = sorted(in_file_not_db)[-5:]
            issues.append(f"  Recent missing from DB: {recent}")

        if in_db_not_file:
            issues.append(f"Days in DB but NOT in DBN file: {len(in_db_not_file)}")

    # Print report
    print_report(bars_1m, bars_5m, daily_features, issues)

    conn.close()

    # Return exit code based on issues
    return 1 if issues else 0

if __name__ == "__main__":
    sys.exit(main())
