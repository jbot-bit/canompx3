#!/usr/bin/env python3
"""
Null test sigma calibration — reproducible derivation from raw data.

Computes the sigma parameter for the Gaussian null test (test_synthetic_null.py)
from actual 1-minute bar data, so the derivation is auditable and reproducible.

RULE (frozen Mar 24 2026, sigma-authority audit):
    p99-trimmed standard deviation of 1-minute close-to-close point changes,
    computed over the null test date range (2020-01-01 to 2025-12-31),
    separately per instrument.

SOURCE TABLE: bars_1m (gold.db)
COLUMN: close - LAG(close) OVER (PARTITION BY symbol ORDER BY ts_utc)
DATE RANGE: 2020-01-01 <= ts_utc < 2026-01-01  (matches test_synthetic_null.py defaults)
TRIM: two-tailed 1% — exclude point changes below P0.5 and above P99.5
UNIT: points per 1-minute bar
ROUNDING: 2 decimal places

WHY TRIMMED:
    Real financial data has fat tails (kurtosis 374-8950 for MGC). The Gaussian
    null cannot reproduce these tails. Matching the FULL std (inflated by extreme
    moves) to a Gaussian sigma would make the null's typical bars too volatile.
    Trimming the top/bottom 0.5% removes fat-tail contamination, giving a sigma
    that matches the BODY of the real distribution — which is what the Gaussian
    null actually generates. This is standard robust estimation for matching
    Gaussian models to heavy-tailed data.

Usage:
    python scripts/tools/calibrate_null_sigma.py              # Read-only comparison table
    python scripts/tools/calibrate_null_sigma.py --update-config  # (stubbed — not implemented yet)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH

# Current null params — imported for comparison only.
# Canonical source: scripts/tests/run_null_batch.py INSTRUMENT_NULL_PARAMS
CURRENT_SIGMAS: dict[str, float] = {
    "MGC": 1.2,
    "MNQ": 5.0,
    "MES": 1.1,
}

# Null test date range (from test_synthetic_null.py defaults)
NULL_START = "2020-01-01"
NULL_END = "2026-01-01"  # exclusive upper bound


def compute_sigma(db_path: Path, instrument: str) -> dict:
    """Compute sigma for one instrument from raw bars_1m data.

    Returns dict with full_std, p005, p995, trimmed_std, trimmed_n, total_n,
    proposed_sigma (rounded to 2dp), and date range coverage.
    """
    con = duckdb.connect(str(db_path), read_only=True)

    rows = con.execute(
        """
        WITH ordered AS (
            SELECT close,
                   LAG(close) OVER (PARTITION BY symbol ORDER BY ts_utc) AS prev_close
            FROM bars_1m
            WHERE symbol = ?
              AND ts_utc >= ?::TIMESTAMP
              AND ts_utc < ?::TIMESTAMP
        )
        SELECT close - prev_close AS point_change
        FROM ordered
        WHERE prev_close IS NOT NULL
        """,
        [instrument, NULL_START, NULL_END],
    ).fetchall()
    con.close()

    if not rows:
        return {"error": f"No bars_1m data for {instrument} in [{NULL_START}, {NULL_END})"}

    changes = np.array([r[0] for r in rows], dtype=np.float64)
    total_n = len(changes)

    full_std = float(np.std(changes, ddof=1))

    p005 = float(np.percentile(changes, 0.5))
    p995 = float(np.percentile(changes, 99.5))

    mask = (changes >= p005) & (changes <= p995)
    trimmed = changes[mask]
    trimmed_n = len(trimmed)
    trimmed_std = float(np.std(trimmed, ddof=1))

    proposed = round(trimmed_std, 2)

    return {
        "instrument": instrument,
        "total_n": total_n,
        "trimmed_n": trimmed_n,
        "full_std": full_std,
        "p005": p005,
        "p995": p995,
        "trimmed_std": trimmed_std,
        "proposed_sigma": proposed,
    }


def compute_sigma_per_year(db_path: Path, instrument: str) -> list[dict]:
    """Compute per-year sigma for C1 time-varying null.

    Each year's sigma is derived from that year's own real bars_1m data only.
    No cross-year pooling, no forward information leakage.
    2026 excluded (holdout sacred).

    NOTE: This is a pre-holdout-era null audit tool. If reused inside
    walk-forward later, sigma must be IS-only for that fold.
    """
    con = duckdb.connect(str(db_path), read_only=True)

    rows = con.execute(
        """
        WITH changes AS (
            SELECT
                EXTRACT(YEAR FROM ts_utc) AS yr,
                close - LAG(close) OVER (PARTITION BY symbol ORDER BY ts_utc) AS delta
            FROM bars_1m
            WHERE symbol = ?
              AND ts_utc >= ?::TIMESTAMP
              AND ts_utc < '2026-01-01'::TIMESTAMP
        )
        SELECT yr, ARRAY_AGG(delta) AS deltas
        FROM changes
        WHERE delta IS NOT NULL
        GROUP BY yr
        ORDER BY yr
        """,
        [instrument, NULL_START],
    ).fetchall()
    con.close()

    results = []
    for yr, deltas in rows:
        arr = np.array(deltas, dtype=np.float64)
        p005 = float(np.percentile(arr, 0.5))
        p995 = float(np.percentile(arr, 99.5))
        mask = (arr >= p005) & (arr <= p995)
        trimmed = arr[mask]
        trimmed_std = float(np.std(trimmed, ddof=1))
        results.append({
            "year": int(yr),
            "total_n": len(arr),
            "trimmed_n": len(trimmed),
            "trimmed_std": trimmed_std,
            "proposed_sigma": round(trimmed_std, 2),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute null test sigma from raw bars_1m data")
    parser.add_argument(
        "--per-year",
        action="store_true",
        help="Emit per-year sigma dict for C1 time-varying null (2026 excluded)",
    )
    args = parser.parse_args()

    instruments = list(ACTIVE_ORB_INSTRUMENTS)
    print("Null test sigma calibration")
    print(f"  DB: {GOLD_DB_PATH}")
    print(f"  Date range: [{NULL_START}, 2026-01-01) [2026 excluded — holdout sacred]")
    print(f"  Instruments: {instruments}")
    print("  Rule: p99-trimmed std (two-tailed 1%, P0.5-P99.5)")
    print()

    if args.per_year:
        print("MODE: per-year (C1 time-varying null)")
        print("  Each year's sigma from that year's own data only. No cross-year pooling.")
        print()
        for inst in instruments:
            yearly = compute_sigma_per_year(GOLD_DB_PATH, inst)
            if not yearly:
                print(f"  {inst}: no data")
                continue
            print(f"  {inst}:")
            sigma_dict = {}
            for r in yearly:
                sigma_dict[r["year"]] = r["proposed_sigma"]
                print(
                    f"    {r['year']}: N={r['total_n']:>10,}  "
                    f"trimmed_std={r['trimmed_std']:.4f}  sigma={r['proposed_sigma']}"
                )
            print()
            print(f"  SIGMA_BY_YEAR_{inst} = {{")
            for yr in sorted(sigma_dict):
                print(f"      {yr}: {sigma_dict[yr]},")
            print("  }")
            print()
        return

    # Original pooled mode
    results = []
    for inst in instruments:
        r = compute_sigma(GOLD_DB_PATH, inst)
        if "error" in r:
            print(f"  {inst}: {r['error']}")
            continue
        results.append(r)
        print(
            f"  {inst}: N={r['total_n']:,} bars, trimmed N={r['trimmed_n']:,}, "
            f"full_std={r['full_std']:.4f}, trimmed_std={r['trimmed_std']:.4f}, "
            f"P0.5={r['p005']:.2f}, P99.5={r['p995']:.2f}"
        )

    if not results:
        print("\nNo results. Check DB path and data availability.")
        sys.exit(1)

    print()
    print("=" * 85)
    print(
        f"{'Instrument':>10s} | {'Current S':>9s} | {'Full Std':>9s} | "
        f"{'Trimmed Std':>11s} | {'Proposed S':>10s} | {'Delta':>7s} | {'Delta %':>7s}"
    )
    print("-" * 85)

    for r in results:
        inst = r["instrument"]
        current = CURRENT_SIGMAS.get(inst)
        proposed = r["proposed_sigma"]
        if current is not None:
            delta = proposed - current
            pct = (delta / current) * 100
            print(
                f"{inst:>10s} | {current:9.2f} | {r['full_std']:9.4f} | "
                f"{r['trimmed_std']:11.4f} | {proposed:10.2f} | {delta:+7.2f} | {pct:+6.1f}%"
            )
        else:
            print(
                f"{inst:>10s} | {'N/A':>9s} | {r['full_std']:9.4f} | "
                f"{r['trimmed_std']:11.4f} | {proposed:10.2f} |     N/A |    N/A"
            )

    print("=" * 85)
    print()
    print("Rule: p99-trimmed std of 1m close-to-close point changes, per year, 2026 excluded")


if __name__ == "__main__":
    main()
