#!/usr/bin/env python3
"""
Deterministic volume spike detector.

Builds a minute-of-day volume profile from bars_1m, detects statistically
significant spikes via z-score against a circular rolling baseline, and
cross-references results against SESSION_CATALOG.

Read-only: opens DB with read_only=True.
No new dependencies: uses numpy + pandas + duckdb already in project.

Usage:
    python scripts/detect_volume_spikes.py --instrument MGC
    python scripts/detect_volume_spikes.py --all
    python scripts/detect_volume_spikes.py --instrument MGC --min-z 2.5 --min-distance 10
    python scripts/detect_volume_spikes.py --instrument MGC --db-path C:\\db\\gold.db
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.paths import GOLD_DB_PATH
from pipeline.dst import (
    SESSION_CATALOG,
    is_us_dst,
    is_uk_dst,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MINUTES_PER_DAY = 1440
ROLLING_WINDOW = 61  # +/-30 minutes centered
REPORTS_DIR = Path(__file__).parent.parent / "reports"


# ---------------------------------------------------------------------------
# Peak detection (no scipy)
# ---------------------------------------------------------------------------
def find_peaks(z_scores: np.ndarray, min_height: float = 3.0,
               min_distance: int = 15) -> list[int]:
    """Local maxima above threshold with minimum separation.

    Uses circular indexing so minute 0 and minute 1439 are neighbors.
    """
    n = len(z_scores)
    peaks = []
    for i in range(n):
        if z_scores[i] > min_height:
            prev_i = (i - 1) % n
            next_i = (i + 1) % n
            if z_scores[i] > z_scores[prev_i] and z_scores[i] >= z_scores[next_i]:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
    return peaks


# ---------------------------------------------------------------------------
# Circular rolling baseline
# ---------------------------------------------------------------------------
def circular_rolling(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute rolling mean and std with circular padding.

    Returns (rolling_mean, rolling_std) of length len(arr).
    """
    half = window // 2
    # Circular pad: wrap tail to front, head to tail
    padded = np.concatenate([arr[-half:], arr, arr[:half]])
    series = pd.Series(padded)
    rolling_mean = series.rolling(window, center=True, min_periods=1).mean().values[half:-half]
    rolling_std = series.rolling(window, center=True, min_periods=1).std().values[half:-half]
    # Avoid division by zero
    rolling_std = np.where(rolling_std < 1e-9, 1e-9, rolling_std)
    return rolling_mean, rolling_std


# ---------------------------------------------------------------------------
# Build volume profiles
# ---------------------------------------------------------------------------
def load_bars(con: duckdb.DuckDBPyConnection, instrument: str) -> pd.DataFrame:
    """Load minute_utc, trading_day, and volume from bars_1m.

    Computes trading_day and DST flags in SQL for performance (3.5M rows).
    Trading day = DATE(ts_utc AT TIME ZONE 'Australia/Brisbane' - INTERVAL 9 HOURS).
    """
    sql = """
        SELECT
            EXTRACT(HOUR FROM ts_utc AT TIME ZONE 'UTC')::INT * 60
              + EXTRACT(MINUTE FROM ts_utc AT TIME ZONE 'UTC')::INT AS minute_utc,
            CAST(
                (ts_utc AT TIME ZONE 'Australia/Brisbane' - INTERVAL '9 hours')
                AS DATE
            ) AS trading_day,
            EXTRACT(YEAR FROM
                CAST(
                    (ts_utc AT TIME ZONE 'Australia/Brisbane' - INTERVAL '9 hours')
                    AS DATE
                )
            )::INT AS year,
            volume
        FROM bars_1m
        WHERE symbol = ?
        ORDER BY ts_utc
    """
    df = con.execute(sql, [instrument]).fetchdf()
    if df.empty:
        return df

    # Vectorized DST: build lookup per unique trading day, then map
    unique_days = df["trading_day"].unique()
    us_dst_map = {d: is_us_dst(d) for d in unique_days}
    uk_dst_map = {d: is_uk_dst(d) for d in unique_days}
    df["is_us_summer"] = df["trading_day"].map(us_dst_map)
    df["is_uk_summer"] = df["trading_day"].map(uk_dst_map)
    return df


def build_profile(df: pd.DataFrame, mask: pd.Series | None = None) -> np.ndarray:
    """Return median volume per minute_utc (0-1439). NaN for missing."""
    subset = df if mask is None else df[mask]
    if subset.empty:
        return np.zeros(MINUTES_PER_DAY)
    grouped = subset.groupby("minute_utc")["volume"].median()
    profile = np.zeros(MINUTES_PER_DAY)
    for minute_utc, vol in grouped.items():
        profile[int(minute_utc)] = vol
    return profile


def build_day_count(df: pd.DataFrame, mask: pd.Series | None = None) -> np.ndarray:
    """Return number of distinct trading days per minute_utc."""
    subset = df if mask is None else df[mask]
    if subset.empty:
        return np.zeros(MINUTES_PER_DAY, dtype=int)
    grouped = subset.groupby("minute_utc")["trading_day"].nunique()
    counts = np.zeros(MINUTES_PER_DAY, dtype=int)
    for minute_utc, n in grouped.items():
        counts[int(minute_utc)] = n
    return counts


# ---------------------------------------------------------------------------
# Year-by-year stability
# ---------------------------------------------------------------------------
def compute_stability_and_years(df: pd.DataFrame, spike_minutes: list[int],
                                min_z_per_year: float = 2.0
                                ) -> tuple[dict[int, float], dict[int, list[int]]]:
    """For each spike minute, compute stability and active years.

    Pre-computes per-year profiles and z-scores once, then looks up spike minutes.
    """
    years = sorted(df["year"].unique())
    if not years:
        empty_stab = {m: 0.0 for m in spike_minutes}
        empty_years = {m: [] for m in spike_minutes}
        return empty_stab, empty_years

    # Pre-compute z-scores per year (one profile + rolling per year)
    year_z_scores = {}  # year -> z_score array (1440,)
    for year in years:
        year_mask = df["year"] == year
        year_profile = build_profile(df, year_mask)
        base_mean, base_std = circular_rolling(year_profile, ROLLING_WINDOW)
        year_z_scores[year] = (year_profile - base_mean) / base_std

    stability = {}
    years_active = {}
    for minute_utc in spike_minutes:
        active = [yr for yr in years if year_z_scores[yr][minute_utc] > min_z_per_year]
        years_active[minute_utc] = active
        stability[minute_utc] = len(active) / len(years)

    return stability, years_active


# ---------------------------------------------------------------------------
# Winter/summer shift detection
# ---------------------------------------------------------------------------
def _circular_shift(a: int, b: int) -> int:
    """Signed circular distance from a to b on a 1440-minute ring."""
    diff = b - a
    if diff > MINUTES_PER_DAY // 2:
        diff -= MINUTES_PER_DAY
    elif diff < -MINUTES_PER_DAY // 2:
        diff += MINUTES_PER_DAY
    return diff


def _nearest_peak(target: int, peaks: list[int], max_dist: int = 90) -> int | None:
    """Find the peak nearest to target (circular), within max_dist."""
    best = None
    best_dist = max_dist + 1
    for p in peaks:
        dist = min(abs(p - target), MINUTES_PER_DAY - abs(p - target))
        if dist < best_dist:
            best_dist = dist
            best = p
    return best


def detect_shift_for_spike(
    minute_utc: int,
    z_winter_val: float,
    z_summer_val: float,
    winter_peaks: list[int],
    summer_peaks: list[int],
    max_dist: int = 90,
) -> tuple[int | None, int | None, int | None, str]:
    """Classify a spike as fixed-UTC, DST-shifting, or season-only.

    Logic:
      - If z > 2 in BOTH seasons at this minute → fixed_utc (no shift)
      - If z > 2 in one season only → search the OTHER season's peaks for a pair
      - Shift = summer_minute - winter_minute (convention)
        -60 = US DST or UK BST (disambiguated by UTC hour)

    Returns (winter_min, summer_min, shift_minutes, shift_type).
    """
    STRONG = 2.0  # threshold for "present in this season"

    winter_strong = z_winter_val > STRONG
    summer_strong = z_summer_val > STRONG

    if winter_strong and summer_strong:
        # Same UTC minute is elevated in both seasons → fixed
        return minute_utc, minute_utc, 0, "fixed_utc"

    if not winter_strong and not summer_strong:
        # Appears in all-data profile but neither season strongly
        # (aggregation artifact or consistent weak signal) → treat as fixed
        return minute_utc, minute_utc, 0, "fixed_utc"

    # One season is strong, the other is weak → DST candidate
    if winter_strong:
        # Search summer peaks for the corresponding event
        pair = _nearest_peak(minute_utc, summer_peaks, max_dist)
        winter_min, summer_min = minute_utc, pair
    else:
        # Search winter peaks for the corresponding event
        pair = _nearest_peak(minute_utc, winter_peaks, max_dist)
        winter_min, summer_min = pair, minute_utc

    if winter_min is None or summer_min is None:
        label = "winter_only" if winter_strong else "summer_only"
        return winter_min, summer_min, None, label

    shift = _circular_shift(winter_min, summer_min)

    if shift == 0:
        shift_type = "fixed_utc"
    elif shift == -60:
        # Both US DST and UK BST shift -60 (summer UTC is 1h earlier).
        # Disambiguate by the winter UTC hour: London events are 06:00-10:00 UTC.
        if 360 <= winter_min <= 600:
            shift_type = "uk_bst"
        else:
            shift_type = "us_dst"
    else:
        shift_type = f"shift_{shift:+d}m"

    return winter_min, summer_min, shift, shift_type


# ---------------------------------------------------------------------------
# Map spikes to SESSION_CATALOG
# ---------------------------------------------------------------------------
def minute_utc_to_brisbane(minute_utc: int) -> tuple[int, int]:
    """Convert minute-of-day in UTC to (hour, minute) in Brisbane."""
    brisbane_min = (minute_utc + 600) % MINUTES_PER_DAY
    return brisbane_min // 60, brisbane_min % 60


def match_to_catalog(minute_utc: int, tolerance: int = 5) -> tuple[str | None, str | None]:
    """Match a UTC spike minute to SESSION_CATALOG entries.

    Returns (matched_dynamic_label, matched_fixed_label) or (None, None).
    """
    bris_h, bris_m = minute_utc_to_brisbane(minute_utc)
    bris_total = bris_h * 60 + bris_m

    matched_fixed = None
    matched_dynamic = None

    for label, entry in SESSION_CATALOG.items():
        if entry["type"] == "alias":
            continue

        if entry["type"] == "fixed":
            cat_h, cat_m = entry["brisbane"]
            cat_total = cat_h * 60 + cat_m
            dist = min(abs(bris_total - cat_total),
                       MINUTES_PER_DAY - abs(bris_total - cat_total))
            if dist <= tolerance:
                matched_fixed = label

        elif entry["type"] == "dynamic":
            # Check both winter and summer times
            for test_date in [date(2025, 1, 15), date(2025, 7, 15)]:
                res_h, res_m = entry["resolver"](test_date)
                # Convert resolver result (Brisbane) to UTC minute
                res_bris_total = res_h * 60 + res_m
                res_utc_total = (res_bris_total - 600) % MINUTES_PER_DAY
                dist = min(abs(minute_utc - res_utc_total),
                           MINUTES_PER_DAY - abs(minute_utc - res_utc_total))
                if dist <= tolerance:
                    matched_dynamic = label
                    break

    return matched_dynamic, matched_fixed


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------
def analyze_instrument(con: duckdb.DuckDBPyConnection, instrument: str,
                       min_z: float = 3.0, min_distance: int = 15) -> dict | None:
    """Run full volume spike analysis for one instrument."""
    print(f"\nLoading {instrument} bars_1m...")
    df = load_bars(con, instrument)
    if df.empty:
        print(f"  No data for {instrument}, skipping.")
        return None

    total_days = df["trading_day"].nunique()
    td_min = df["trading_day"].min()
    td_max = df["trading_day"].max()
    # DuckDB DATE comes back as datetime -- extract date portion
    date_range = (
        td_min.strftime("%Y-%m-%d") if hasattr(td_min, "strftime") else str(td_min),
        td_max.strftime("%Y-%m-%d") if hasattr(td_max, "strftime") else str(td_max),
    )
    print(f"  {len(df):,} bars, {total_days} trading days "
          f"({date_range[0]} to {date_range[1]})")

    # Build profiles: all / winter / summer
    profile_all = build_profile(df)
    profile_winter = build_profile(df, ~df["is_us_summer"])
    profile_summer = build_profile(df, df["is_us_summer"])

    day_count_all = build_day_count(df)

    # Z-scores
    base_mean_all, base_std_all = circular_rolling(profile_all, ROLLING_WINDOW)
    z_all = (profile_all - base_mean_all) / base_std_all

    base_mean_win, base_std_win = circular_rolling(profile_winter, ROLLING_WINDOW)
    z_winter = (profile_winter - base_mean_win) / base_std_win

    base_mean_sum, base_std_sum = circular_rolling(profile_summer, ROLLING_WINDOW)
    z_summer = (profile_summer - base_mean_sum) / base_std_sum

    # Peak detection on all three
    peaks_all = find_peaks(z_all, min_height=min_z, min_distance=min_distance)
    peaks_winter = find_peaks(z_winter, min_height=min_z, min_distance=min_distance)
    peaks_summer = find_peaks(z_summer, min_height=min_z, min_distance=min_distance)

    # Merge: union of all peak minutes, sorted by z_all descending
    all_peak_set = sorted(set(peaks_all) | set(peaks_winter) | set(peaks_summer))
    # Re-filter: keep only those with z_all > min_z OR z_winter > min_z OR z_summer > min_z
    significant = [m for m in all_peak_set
                   if z_all[m] > min_z or z_winter[m] > min_z or z_summer[m] > min_z]
    significant.sort(key=lambda m: -z_all[m])

    if not significant:
        print(f"  No spikes above z={min_z} detected.")
        return {"instrument": instrument, "data_range": [str(date_range[0]), str(date_range[1])],
                "total_trading_days": total_days, "spikes": [], "catalog_coverage": {}}

    # Stability
    print(f"  Computing year-by-year stability for {len(significant)} spikes...")
    stability, years_active = compute_stability_and_years(df, significant)

    # Shift detection
    spikes = []
    for rank, minute_utc in enumerate(significant, 1):
        winter_min, summer_min, shift_minutes, shift_type = detect_shift_for_spike(
            minute_utc, z_winter[minute_utc], z_summer[minute_utc],
            peaks_winter, peaks_summer,
        )

        matched_dynamic, matched_fixed = match_to_catalog(minute_utc)
        bris_h, bris_m = minute_utc_to_brisbane(minute_utc)

        spike = {
            "rank": rank,
            "minute_utc": int(minute_utc),
            "time_utc": f"{minute_utc // 60:02d}:{minute_utc % 60:02d}",
            "time_brisbane": f"{bris_h:02d}:{bris_m:02d}",
            "winter_minute_utc": int(winter_min) if winter_min is not None else None,
            "summer_minute_utc": int(summer_min) if summer_min is not None else None,
            "shift_minutes": int(shift_minutes) if shift_minutes is not None else None,
            "shift_type": shift_type,
            "z_score_all": round(float(z_all[minute_utc]), 2),
            "z_score_winter": round(float(z_winter[minute_utc]), 2),
            "z_score_summer": round(float(z_summer[minute_utc]), 2),
            "median_vol_all": round(float(profile_all[minute_utc]), 1),
            "n_days": int(day_count_all[minute_utc]),
            "stability": round(stability.get(minute_utc, 0.0), 2),
            "years_active": years_active.get(minute_utc, []),
            "matched_session": matched_dynamic,
            "matched_fixed_label": matched_fixed,
        }
        spikes.append(spike)

    # Catalog coverage
    catalog_coverage = {}
    for label, entry in SESSION_CATALOG.items():
        if entry["type"] == "alias":
            continue
        detected = False
        spike_rank = None
        best_z = 0.0
        note = None

        for sp in spikes:
            if sp["matched_session"] == label or sp["matched_fixed_label"] == label:
                detected = True
                spike_rank = sp["rank"]
                best_z = sp["z_score_all"]
                break

        if not detected:
            # Check what z-score we see at the catalog's expected UTC minute
            if entry["type"] == "fixed":
                cat_h, cat_m = entry["brisbane"]
                cat_utc_min = ((cat_h * 60 + cat_m) - 600) % MINUTES_PER_DAY
                best_z = round(float(z_all[cat_utc_min]), 2)
                note = f"z={best_z} (no matching peak)"
            elif entry["type"] == "dynamic":
                # Check both season times
                z_vals = []
                for test_date in [date(2025, 1, 15), date(2025, 7, 15)]:
                    res_h, res_m = entry["resolver"](test_date)
                    res_bris_total = res_h * 60 + res_m
                    res_utc_min = (res_bris_total - 600) % MINUTES_PER_DAY
                    z_vals.append(round(float(z_all[res_utc_min]), 2))
                best_z = max(z_vals)
                note = f"z_winter={z_vals[0]}, z_summer={z_vals[1]} (no matching peak)"

        catalog_coverage[label] = {
            "detected": detected,
            "spike_rank": spike_rank,
            "z_score": round(best_z, 2) if best_z else None,
            "note": note,
        }

    return {
        "instrument": instrument,
        "data_range": [str(date_range[0]), str(date_range[1])],
        "total_trading_days": total_days,
        "spikes": spikes,
        "catalog_coverage": catalog_coverage,
    }


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------
def print_report(result: dict, min_z: float) -> None:
    """Print human-readable spike report."""
    inst = result["instrument"]
    dr = result["data_range"]
    total = result["total_trading_days"]

    print(f"\n{'=' * 72}")
    print(f" {inst} Volume Spike Report ({dr[0]} to {dr[1]})")
    print(f" {total} trading days analyzed")
    print(f"{'=' * 72}")

    spikes = result["spikes"]
    if not spikes:
        print(f"\n  No spikes detected above z={min_z}")
        return

    # Table header
    print(f"\n  Top Spikes (z > {min_z}):")
    print(f"  {'#':>3}  {'UTC':>5}  {'Bris':>5}  {'Z-All':>6}  {'Z-Win':>6}  "
          f"{'Z-Sum':>6}  {'Shift':>6}  {'Type':<12}  {'Stab':>5}  {'Session Match'}")
    print(f"  {'---':>3}  {'-----':>5}  {'-----':>5}  {'------':>6}  {'------':>6}  "
          f"{'------':>6}  {'------':>6}  {'----------':<12}  {'-----':>5}  {'-------------'}")

    for sp in spikes:
        shift_str = f"{sp['shift_minutes']:+d}" if sp["shift_minutes"] is not None else "N/A"
        match_parts = []
        if sp["matched_session"]:
            match_parts.append(sp["matched_session"])
        if sp["matched_fixed_label"]:
            match_parts.append(sp["matched_fixed_label"])
        match_str = " / ".join(match_parts) if match_parts else "--"

        print(f"  {sp['rank']:>3}  {sp['time_utc']:>5}  {sp['time_brisbane']:>5}  "
              f"{sp['z_score_all']:>6.1f}  {sp['z_score_winter']:>6.1f}  "
              f"{sp['z_score_summer']:>6.1f}  {shift_str:>6}  {sp['shift_type']:<12}  "
              f"{sp['stability']:>5.2f}  {match_str}")

    # Catalog coverage
    print(f"\n  Catalog Coverage:")
    coverage = result["catalog_coverage"]
    for label, info in sorted(coverage.items()):
        if info["detected"]:
            print(f"    {label:<18} MATCHED (rank {info['spike_rank']}, z={info['z_score']})")
        else:
            note = info.get("note", "not detected")
            print(f"    {label:<18} NOT DETECTED ({note})")

    # Unmatched spikes
    unmatched = [sp for sp in spikes
                 if not sp["matched_session"] and not sp["matched_fixed_label"]]
    if unmatched:
        print(f"\n  Unmatched Spikes (potential new sessions):")
        for sp in unmatched:
            print(f"    {sp['time_utc']} UTC ({sp['time_brisbane']} Brisbane) "
                  f"z={sp['z_score_all']:.1f}, stability={sp['stability']:.2f}")


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------
def save_json(result: dict, output_dir: Path) -> Path:
    """Save JSON report and return the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"volume_spikes_{result['instrument']}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Detect volume spikes in bars_1m by minute-of-day"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instrument", type=str, help="Instrument symbol (e.g., MGC)")
    group.add_argument("--all", action="store_true", help="Run for all instruments with data")
    parser.add_argument("--db-path", type=str, default=None,
                        help=f"Database path (default: {GOLD_DB_PATH})")
    parser.add_argument("--min-z", type=float, default=3.0,
                        help="Minimum z-score for spike detection (default: 3.0)")
    parser.add_argument("--min-distance", type=int, default=15,
                        help="Minimum minutes between peaks (default: 15)")
    parser.add_argument("--no-json", action="store_true",
                        help="Skip JSON output")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else GOLD_DB_PATH
    if not db_path.exists():
        print(f"FATAL: Database not found at {db_path}")
        sys.exit(1)

    con = duckdb.connect(str(db_path), read_only=True)

    # Determine instruments to process
    if args.all:
        # Find instruments that actually have bars_1m data
        rows = con.execute(
            "SELECT DISTINCT symbol FROM bars_1m ORDER BY symbol"
        ).fetchall()
        instruments = [r[0] for r in rows]
        if not instruments:
            print("No instruments found in bars_1m.")
            sys.exit(0)
        print(f"Processing {len(instruments)} instruments: {', '.join(instruments)}")
    else:
        instruments = [args.instrument.upper()]

    for instrument in instruments:
        result = analyze_instrument(con, instrument,
                                    min_z=args.min_z,
                                    min_distance=args.min_distance)
        if result is None:
            continue

        print_report(result, args.min_z)

        if not args.no_json:
            path = save_json(result, REPORTS_DIR)
            print(f"\n  JSON saved: {path}")

    con.close()


if __name__ == "__main__":
    main()
