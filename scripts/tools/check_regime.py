#!/usr/bin/env python3
"""Pre-session regime check — run before trading day starts.

Shows ATR regime status for all active instruments + cross-asset gates.
Tells you which sessions are GO/SKIP based on ATR percentile filters.

Usage:
    python scripts/tools/check_regime.py
    python scripts/tools/check_regime.py --instrument MNQ
"""

import argparse
import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, CrossAssetATRFilter


def main():
    parser = argparse.ArgumentParser(description="Pre-session ATR regime check")
    parser.add_argument("--instrument", help="Single instrument (default: all active)")
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH))
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else sorted(ACTIVE_ORB_INSTRUMENTS)
    con = duckdb.connect(args.db_path, read_only=True)

    # Collect cross-asset source instruments from filter registry
    cross_sources = {f.source_instrument for f in ALL_FILTERS.values() if isinstance(f, CrossAssetATRFilter)}

    # Load latest ATR data for all instruments
    atr_data = {}
    for inst in sorted(set(instruments) | cross_sources):
        r = con.execute(
            """SELECT atr_20, atr_20_pct, atr_vel_regime, trading_day
               FROM daily_features
               WHERE symbol = ? AND orb_minutes = 5 AND atr_20 IS NOT NULL
               ORDER BY trading_day DESC LIMIT 1""",
            [inst],
        ).fetchone()
        if r:
            atr_data[inst] = {
                "atr": r[0],
                "pct": r[1],
                "vel": r[2],
                "day": r[3],
            }

    print("=" * 70)
    print("PRE-SESSION REGIME CHECK")
    print("=" * 70)

    # Per-instrument regime status
    print(f"\n{'Inst':5s} {'ATR_20':>8s} {'Pct':>5s} {'Velocity':>12s} {'ATR70':>7s} {'Date':>12s}")
    print("-" * 55)
    for inst in instruments:
        d = atr_data.get(inst)
        if not d:
            print(f"{inst:5s}  NO DATA")
            continue
        pct = d["pct"]
        gate = "GO" if pct is not None and pct >= 70 else "SKIP"
        pct_str = f"{pct:.0f}th" if pct is not None else "N/A"
        print(
            f"{inst:5s} {d['atr']:8.1f} {pct_str:>5s} {(d['vel'] or 'N/A'):>12s} {gate:>7s} {str(d['day'])[:10]:>12s}"
        )

    # Cross-asset gates for MNQ
    if "MNQ" in instruments:
        print(f"\n{'-' * 70}")
        print("CROSS-ASSET GATES (MNQ US sessions)")
        print(f"{'-' * 70}")

        cross_filters = {k: f for k, f in ALL_FILTERS.items() if isinstance(f, CrossAssetATRFilter)}

        for key, filt in sorted(cross_filters.items()):
            src = filt.source_instrument
            d = atr_data.get(src)
            if not d or d["pct"] is None:
                gate = "NO DATA"
            elif d["pct"] >= filt.min_pct:
                gate = "GO"
            else:
                gate = "SKIP"
            pct_str = f"{d['pct']:.0f}th" if d and d["pct"] is not None else "N/A"
            print(f"  {key:15s}  {src} ATR pct={pct_str:>5s}  threshold={filt.min_pct:.0f}th  [{gate}]")

        # Which MNQ sessions have cross-asset filters?
        from trading_app.config import get_filters_for_grid

        print("\n  Sessions with cross-asset filters:")
        from pipeline.dst import SESSION_CATALOG

        for session in sorted(SESSION_CATALOG):
            grid = get_filters_for_grid("MNQ", session)
            cross_in_grid = [k for k in grid if k.startswith("X_")]
            if cross_in_grid:
                print(f"    {session:20s}  {', '.join(cross_in_grid)}")

    # Trade management note
    print(f"\n{'-' * 70}")
    print("TRADE MANAGEMENT (post-entry)")
    print(f"{'-' * 70}")
    print("  rel_vol is known AFTER break bar closes (not pre-entry).")
    print("  High rel_vol (>=1.2) on ATR70 days -> 1.15x more MFE (p<0.05).")
    print("  Use as confidence signal: high vol = hold to target,")
    print("  low vol = consider tighter management.")

    con.close()


if __name__ == "__main__":
    main()
