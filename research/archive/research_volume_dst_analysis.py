#!/usr/bin/env python3
"""
Volume by DST Regime + New Session Candidate Analysis.

Task 1: Compare trading volume between winter and summer at key session times.
Task 2: Evaluate new session candidates (09:30, 19:00, 10:45) for pipeline addition.

Read-only: no writes to gold.db.

Usage:
    python research/research_volume_dst_analysis.py
    python research/research_volume_dst_analysis.py --db-path C:/db/gold.db
"""

import argparse
import csv
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

# ── Timezone setup ────────────────────────────────────────────────────

TZ_US_EAST = ZoneInfo("America/New_York")
TZ_UK = ZoneInfo("Europe/London")
TZ_BRIS = ZoneInfo("Australia/Brisbane")


def is_us_winter(trading_day: date) -> bool:
    """True if US Eastern is in standard time (EST, UTC-5)."""
    dt = datetime(trading_day.year, trading_day.month, trading_day.day,
                  12, 0, 0, tzinfo=TZ_US_EAST)
    return dt.utcoffset().total_seconds() == -5 * 3600


def is_uk_winter(trading_day: date) -> bool:
    """True if UK is in standard time (GMT, UTC+0)."""
    dt = datetime(trading_day.year, trading_day.month, trading_day.day,
                  12, 0, 0, tzinfo=TZ_UK)
    return dt.utcoffset().total_seconds() == 0


# ── Session time config ───────────────────────────────────────────────

# All times in Brisbane local (UTC+10, no DST)
# For each time: (hour, minute, dst_type)
# dst_type: "US" = split by US DST, "UK" = split by UK DST, "NONE" = no split (Asia)

ANALYSIS_TIMES = {
    # Current sessions
    "0900": (9, 0, "US"),
    "1000": (10, 0, "NONE"),
    "1100": (11, 0, "NONE"),
    "1800": (18, 0, "UK"),
    "0030": (0, 30, "US"),
    "2300": (23, 0, "US"),
    # New candidates
    "0830": (8, 30, "US"),
    "0930": (9, 30, "US"),
    "1045": (10, 45, "NONE"),
    "1700": (17, 0, "UK"),
    "1900": (19, 0, "UK"),
    # Dynamic session equivalents (computed per-day, placeholder times)
    # CME_OPEN: 0800 summer / 0900 winter
    # LONDON_OPEN: 1700 summer / 1800 winter
}

INSTRUMENTS = ["MGC", "MNQ", "MES"]


def brisbane_to_utc(hour: int, minute: int) -> tuple[int, int]:
    """Convert Brisbane time to UTC. Brisbane = UTC+10 always."""
    utc_hour = (hour - 10) % 24
    return utc_hour, minute


def get_volume_at_time(con, symbol: str, utc_hour: int, utc_minute: int,
                       n_bars: int = 5) -> list[dict]:
    """Get volume data for n_bars starting at given UTC time for all trading days.

    Returns list of dicts: {trading_day, orb_volume, post_orb_volume_60m}
    """
    # Query: for each day, get volume in the ORB window (first n_bars minutes)
    # and in the 60 minutes after
    rows = con.execute("""
        WITH day_bars AS (
            SELECT
                ts_utc::DATE AS bar_date,
                EXTRACT(HOUR FROM (ts_utc AT TIME ZONE 'UTC')) AS bar_hour,
                EXTRACT(MINUTE FROM (ts_utc AT TIME ZONE 'UTC')) AS bar_minute,
                volume
            FROM bars_1m
            WHERE symbol = ?
        ),
        orb_bars AS (
            SELECT bar_date, SUM(volume) AS orb_volume
            FROM day_bars
            WHERE (bar_hour * 60 + bar_minute) >= ?
              AND (bar_hour * 60 + bar_minute) < ?
            GROUP BY bar_date
        ),
        post_orb AS (
            SELECT bar_date, SUM(volume) / 60.0 AS post_orb_vol_per_min
            FROM day_bars
            WHERE (bar_hour * 60 + bar_minute) >= ?
              AND (bar_hour * 60 + bar_minute) < ?
            GROUP BY bar_date
        )
        SELECT o.bar_date, o.orb_volume, p.post_orb_vol_per_min
        FROM orb_bars o
        LEFT JOIN post_orb p ON o.bar_date = p.bar_date
        WHERE o.orb_volume > 0
        ORDER BY o.bar_date
    """, [
        symbol,
        utc_hour * 60 + utc_minute,
        utc_hour * 60 + utc_minute + n_bars,
        utc_hour * 60 + utc_minute + n_bars,
        utc_hour * 60 + utc_minute + n_bars + 60,
    ]).fetchall()

    return [{"trading_day": r[0], "orb_volume": r[1], "post_orb_vol": r[2]} for r in rows]


def compute_volume_stats(rows: list[dict]) -> dict:
    """Compute mean and median volume from list of volume dicts."""
    if not rows:
        return {"n": 0, "mean_vol": None, "median_vol": None,
                "mean_post": None, "median_post": None}

    vols = [r["orb_volume"] for r in rows if r["orb_volume"] is not None]
    posts = [r["post_orb_vol"] for r in rows if r["post_orb_vol"] is not None]

    def _median(lst):
        if not lst:
            return None
        s = sorted(lst)
        mid = len(s) // 2
        return (s[mid - 1] + s[mid]) / 2 if len(s) % 2 == 0 else s[mid]

    return {
        "n": len(rows),
        "mean_vol": sum(vols) / len(vols) if vols else None,
        "median_vol": _median(vols),
        "mean_post": sum(posts) / len(posts) if posts else None,
        "median_post": _median(posts),
    }


def run_task1(con, output_dir: Path):
    """Task 1: Volume by DST regime for all times and instruments."""
    print("=" * 80)
    print("TASK 1: Volume by DST Regime")
    print("=" * 80)

    all_results = []

    for symbol in INSTRUMENTS:
        print(f"\n{'=' * 60}")
        print(f"  {symbol}")
        print(f"{'=' * 60}")
        print(f"{'Time':<8} | {'W Vol (mean/med)':<22} | {'S Vol (mean/med)':<22} | "
              f"{'W/S Ratio':<10} | {'Post-ORB W':<12} | {'Post-ORB S':<12} | {'Post Ratio':<10}")
        print("-" * 110)

        for time_label, (hour, minute, dst_type) in sorted(ANALYSIS_TIMES.items()):
            utc_h, utc_m = brisbane_to_utc(hour, minute)
            vol_data = get_volume_at_time(con, symbol, utc_h, utc_m)

            if dst_type == "NONE":
                # No DST split needed — show combined only
                stats = compute_volume_stats(vol_data)
                mean_s = f"{stats['mean_vol']:.0f}" if stats['mean_vol'] else "N/A"
                med_s = f"{stats['median_vol']:.0f}" if stats['median_vol'] else "N/A"
                post_s = f"{stats['mean_post']:.1f}" if stats['mean_post'] else "N/A"
                print(f"{time_label:<8} | {mean_s + '/' + med_s:<22} | {'(no DST)':<22} | "
                      f"{'N/A':<10} | {post_s:<12} | {'N/A':<12} | {'N/A':<10}")
                all_results.append({
                    "symbol": symbol, "time": time_label,
                    "dst_type": dst_type,
                    "winter_n": stats["n"], "winter_mean": stats["mean_vol"],
                    "winter_median": stats["median_vol"],
                    "summer_n": None, "summer_mean": None, "summer_median": None,
                    "ratio": None,
                    "post_winter": stats["mean_post"], "post_summer": None,
                    "post_ratio": None,
                })
            else:
                # Split by winter/summer
                is_winter_fn = is_us_winter if dst_type == "US" else is_uk_winter

                winter_rows = []
                summer_rows = []
                for r in vol_data:
                    td = r["trading_day"]
                    if hasattr(td, 'date'):
                        td = td.date()
                    elif not isinstance(td, date):
                        td = date.fromisoformat(str(td)[:10])
                    if is_winter_fn(td):
                        winter_rows.append(r)
                    else:
                        summer_rows.append(r)

                w_stats = compute_volume_stats(winter_rows)
                s_stats = compute_volume_stats(summer_rows)

                w_mean = f"{w_stats['mean_vol']:.0f}" if w_stats['mean_vol'] else "N/A"
                w_med = f"{w_stats['median_vol']:.0f}" if w_stats['median_vol'] else "N/A"
                s_mean = f"{s_stats['mean_vol']:.0f}" if s_stats['mean_vol'] else "N/A"
                s_med = f"{s_stats['median_vol']:.0f}" if s_stats['median_vol'] else "N/A"

                ratio = None
                if w_stats['mean_vol'] and s_stats['mean_vol'] and s_stats['mean_vol'] > 0:
                    ratio = w_stats['mean_vol'] / s_stats['mean_vol']
                ratio_s = f"{ratio:.2f}" if ratio else "N/A"

                w_post = f"{w_stats['mean_post']:.1f}" if w_stats['mean_post'] else "N/A"
                s_post = f"{s_stats['mean_post']:.1f}" if s_stats['mean_post'] else "N/A"

                post_ratio = None
                if w_stats['mean_post'] and s_stats['mean_post'] and s_stats['mean_post'] > 0:
                    post_ratio = w_stats['mean_post'] / s_stats['mean_post']
                pr_s = f"{post_ratio:.2f}" if post_ratio else "N/A"

                print(f"{time_label:<8} | {w_mean + '/' + w_med:<22} | {s_mean + '/' + s_med:<22} | "
                      f"{ratio_s:<10} | {w_post:<12} | {s_post:<12} | {pr_s:<10}")

                all_results.append({
                    "symbol": symbol, "time": time_label,
                    "dst_type": dst_type,
                    "winter_n": w_stats["n"], "winter_mean": w_stats["mean_vol"],
                    "winter_median": w_stats["median_vol"],
                    "summer_n": s_stats["n"], "summer_mean": s_stats["mean_vol"],
                    "summer_median": s_stats["median_vol"],
                    "ratio": ratio,
                    "post_winter": w_stats["mean_post"], "post_summer": s_stats["mean_post"],
                    "post_ratio": post_ratio,
                })

    # Key analysis
    print("\n" + "=" * 80)
    print("KEY ANALYSIS QUESTIONS")
    print("=" * 80)

    for symbol in INSTRUMENTS:
        sym_results = {r["time"]: r for r in all_results if r["symbol"] == symbol}

        r0900 = sym_results.get("0900", {})
        r0800 = sym_results.get("0830", {})  # Close to 0800 summer CME open
        r1900 = sym_results.get("1900", {})
        r1800 = sym_results.get("1800", {})
        r1000 = sym_results.get("1000", {})
        r1100 = sym_results.get("1100", {})

        print(f"\n{symbol}:")
        # Does 0900 volume drop in summer?
        if r0900.get("winter_mean") and r0900.get("summer_mean"):
            pct_change = (r0900["summer_mean"] - r0900["winter_mean"]) / r0900["winter_mean"] * 100
            print(f"  0900 summer vs winter: {pct_change:+.1f}% volume change")

        # Is 19:00 volume comparable to 18:00?
        if r1900.get("winter_mean") and r1800.get("winter_mean"):
            pct = r1900["winter_mean"] / r1800["winter_mean"] * 100
            print(f"  19:00 volume = {pct:.0f}% of 18:00 volume (winter)")
        if r1900.get("summer_mean") and r1800.get("summer_mean"):
            pct = r1900["summer_mean"] / r1800["summer_mean"] * 100
            print(f"  19:00 volume = {pct:.0f}% of 18:00 volume (summer)")

        # Asia regime stability
        if r1000.get("winter_mean") and r1000.get("winter_n"):
            print(f"  1000: {r1000['winter_n']} days (no split, Asia clean)")

    # Save CSV
    csv_path = output_dir / "volume_dst_analysis.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "symbol", "time", "dst_type",
            "winter_n", "winter_mean", "winter_median",
            "summer_n", "summer_mean", "summer_median",
            "ratio", "post_winter", "post_summer", "post_ratio",
        ])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved: {csv_path}")

    return all_results


def run_task2(con, output_dir: Path):
    """Task 2: Evaluate new session candidates."""
    print("\n" + "=" * 80)
    print("TASK 2: New Session Candidate Evaluation")
    print("=" * 80)

    # Candidates from time scan results
    candidates = [
        {"time": "0930", "instrument": "MGC", "nearest": "0900",
         "scan_avg_w": 0.14, "scan_avg_s": 0.18, "stability": "STABLE"},
        {"time": "1900", "instrument": "MGC", "nearest": "1800",
         "scan_avg_w": 0.43, "scan_avg_s": 0.48, "stability": "STABLE"},
        {"time": "1045", "instrument": "MNQ", "nearest": "1000",
         "scan_avg_w": 0.33, "scan_avg_s": 0.11, "stability": "WINTER-DOM"},
        {"time": "1045", "instrument": "MES", "nearest": "1000",
         "scan_avg_w": 0.35, "scan_avg_s": 0.10, "stability": "WINTER-DOM"},
    ]

    results = []

    for cand in candidates:
        symbol = cand["instrument"]
        time_label = cand["time"]
        nearest = cand["nearest"]
        h, m, _ = ANALYSIS_TIMES[time_label]
        utc_h, utc_m = brisbane_to_utc(h, m)

        nh, nm, _ = ANALYSIS_TIMES[nearest]
        n_utc_h, n_utc_m = brisbane_to_utc(nh, nm)

        print(f"\n--- {symbol} {time_label} (nearest: {nearest}) ---")

        # 1. Count G4+ days (ORB size >= 4pt from 5-minute bars)
        # Query 5 bars, compute ORB, filter G4+
        orb_rows = con.execute("""
            WITH orb_bars AS (
                SELECT
                    ts_utc::DATE AS bar_date,
                    high, low
                FROM bars_1m
                WHERE symbol = ?
                  AND EXTRACT(HOUR FROM (ts_utc AT TIME ZONE 'UTC')) * 60 +
                      EXTRACT(MINUTE FROM (ts_utc AT TIME ZONE 'UTC'))
                      BETWEEN ? AND ?
            ),
            daily_orb AS (
                SELECT bar_date, MAX(high) - MIN(low) AS orb_size, COUNT(*) AS bar_count
                FROM orb_bars
                GROUP BY bar_date
                HAVING COUNT(*) >= 4
            )
            SELECT bar_date, orb_size
            FROM daily_orb
            WHERE orb_size >= 4.0
            ORDER BY bar_date
        """, [symbol, utc_h * 60 + utc_m, utc_h * 60 + utc_m + 4]).fetchall()

        g4_days = len(orb_rows)
        if orb_rows:
            first_day = orb_rows[0][0]
            last_day = orb_rows[-1][0]
            if hasattr(first_day, 'date'):
                first_day = first_day.date()
                last_day = last_day.date()
            span_years = max((last_day - first_day).days / 365.25, 0.5)
            g4_per_year = g4_days / span_years
            avg_orb = sum(r[1] for r in orb_rows) / g4_days
        else:
            g4_per_year = 0
            avg_orb = 0

        print(f"  G4+ days: {g4_days} total, ~{g4_per_year:.0f}/year")
        print(f"  Avg ORB size: {avg_orb:.1f} pts")

        # 2. Nearest session comparison
        nearest_orbs = con.execute("""
            WITH orb_bars AS (
                SELECT
                    ts_utc::DATE AS bar_date,
                    high, low
                FROM bars_1m
                WHERE symbol = ?
                  AND EXTRACT(HOUR FROM (ts_utc AT TIME ZONE 'UTC')) * 60 +
                      EXTRACT(MINUTE FROM (ts_utc AT TIME ZONE 'UTC'))
                      BETWEEN ? AND ?
            ),
            daily_orb AS (
                SELECT bar_date, MAX(high) - MIN(low) AS orb_size
                FROM orb_bars
                GROUP BY bar_date
                HAVING COUNT(*) >= 4
            )
            SELECT bar_date FROM daily_orb WHERE orb_size >= 4.0
        """, [symbol, n_utc_h * 60 + n_utc_m, n_utc_h * 60 + n_utc_m + 4]).fetchall()

        nearest_g4_days = set()
        for r in nearest_orbs:
            d = r[0]
            if hasattr(d, 'date'):
                d = d.date()
            nearest_g4_days.add(d)

        cand_g4_days = set()
        for r in orb_rows:
            d = r[0]
            if hasattr(d, 'date'):
                d = d.date()
            cand_g4_days.add(d)

        # 4. Overlap
        if cand_g4_days:
            overlap = len(cand_g4_days & nearest_g4_days) / len(cand_g4_days) * 100
        else:
            overlap = 0

        print(f"  Overlap with {nearest}: {overlap:.0f}% of G4+ days")

        # 5. Volume comparison
        cand_vol = get_volume_at_time(con, symbol, utc_h, utc_m)
        near_vol = get_volume_at_time(con, symbol, n_utc_h, n_utc_m)
        cand_mean = sum(r["orb_volume"] for r in cand_vol) / len(cand_vol) if cand_vol else 0
        near_mean = sum(r["orb_volume"] for r in near_vol) / len(near_vol) if near_vol else 1
        vol_ratio = cand_mean / near_mean if near_mean > 0 else 0

        print(f"  Volume ratio vs {nearest}: {vol_ratio:.2f}x")

        # Verdict
        if overlap > 80:
            verdict = "SKIP (>80% overlap)"
        elif g4_per_year < 20:
            verdict = "INVESTIGATE (low G4+ frequency)"
        elif vol_ratio < 0.3:
            verdict = "INVESTIGATE (low volume)"
        elif cand["stability"] == "STABLE":
            verdict = "ADD"
        else:
            verdict = "INVESTIGATE"

        print(f"  Verdict: {verdict}")

        results.append({
            "candidate": f"{symbol}_{time_label}",
            "g4_per_year": round(g4_per_year, 0),
            "avg_orb": round(avg_orb, 1),
            "overlap_pct": round(overlap, 0),
            "vol_ratio": round(vol_ratio, 2),
            "stability": cand["stability"],
            "verdict": verdict,
        })

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"{'Candidate':<15} | {'G4+/yr':<8} | {'Avg ORB':<8} | {'Overlap':<8} | "
          f"{'Vol Ratio':<10} | {'Stability':<12} | {'Verdict':<25}")
    print("-" * 90)
    for r in results:
        print(f"{r['candidate']:<15} | {r['g4_per_year']:<8.0f} | {r['avg_orb']:<8.1f} | "
              f"{r['overlap_pct']:<7.0f}% | {r['vol_ratio']:<10.2f} | "
              f"{r['stability']:<12} | {r['verdict']:<25}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Volume DST analysis + new session candidates")
    parser.add_argument("--db-path", type=str, default="C:/db/gold.db",
                        help="Database path")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    output_dir = PROJECT_ROOT / "research" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Database: {db_path}")
    print(f"Output:   {output_dir}")
    print()

    with duckdb.connect(str(db_path), read_only=True) as con:
        run_task1(con, output_dir)
        run_task2(con, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
