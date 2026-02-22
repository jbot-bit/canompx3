"""Portfolio Assembly Research Report — Combines all session slots into a single portfolio equity curve with honest stats."""

import sys
import argparse
from pathlib import Path
from math import sqrt
from collections import defaultdict

import numpy as np
import pandas as pd
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.paths import GOLD_DB_PATH
from trading_app.config import ALL_FILTERS, VolumeFilter
from trading_app.strategy_discovery import (
    _build_filter_day_sets,
    _compute_relative_volumes,
    _load_daily_features,
)

# Add scripts/reports to sys.path for session_slots import
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reports"))
from report_edge_portfolio import session_slots

sys.stdout.reconfigure(line_buffering=True)

TRADING_DAYS_PER_YEAR = 252


def _get_strategy_params(con, strategy_id):
    """Look up strategy parameters from validated_setups."""
    row = con.execute("""
        SELECT instrument, orb_label, orb_minutes, entry_model,
               rr_target, confirm_bars, filter_type
        FROM validated_setups
        WHERE strategy_id = ?
    """, [strategy_id]).fetchone()
    if not row:
        return None
    cols = ["instrument", "orb_label", "orb_minutes", "entry_model",
            "rr_target", "confirm_bars", "filter_type"]
    return dict(zip(cols, row))


def load_slot_trades(con, slots):
    """Load trades for each slot's head strategy using canonical filter logic.

    Bypasses strategy_trade_days (incomplete for E0 strategies).
    Instead applies filters via daily_features using the same logic
    as strategy_discovery.py.

    Returns dict: strategy_id -> list of trade dicts.
    """
    trades_by_slot = {}

    # Group slots by instrument to batch filter computation
    by_instrument = defaultdict(list)
    for slot in slots:
        by_instrument[slot["instrument"]].append(slot)

    for instrument, inst_slots in by_instrument.items():
        # Collect filter types and sessions needed for this instrument
        slot_params = {}
        filter_types = set()
        orb_labels = set()
        for slot in inst_slots:
            params = _get_strategy_params(con, slot["head_strategy_id"])
            if params is None:
                print(f"  WARNING: {slot['head_strategy_id']} not in validated_setups")
                continue
            slot_params[slot["head_strategy_id"]] = params
            filter_types.add(params["filter_type"])
            orb_labels.add(params["orb_label"])

        if not slot_params:
            continue

        orb_labels = sorted(orb_labels)
        needed_filters = {k: v for k, v in ALL_FILTERS.items() if k in filter_types}

        # Handle filters not in ALL_FILTERS (e.g. DIR_LONG, ORB_G4_L12)
        missing = filter_types - set(needed_filters.keys())
        if missing:
            from trading_app.config import DirectionFilter, OrbSizeFilter, CompositeFilter, BreakSpeedFilter
            for ft in missing:
                if ft == "DIR_LONG":
                    needed_filters[ft] = DirectionFilter(
                        filter_type="DIR_LONG",
                        description="Long breaks only",
                        direction="long",
                    )
                elif ft == "DIR_SHORT":
                    needed_filters[ft] = DirectionFilter(
                        filter_type="DIR_SHORT",
                        description="Short breaks only",
                        direction="short",
                    )
                elif ft.startswith("ORB_G") and "_L" in ft:
                    # e.g. ORB_G4_L12 = G4 base + max break delay 12
                    parts = ft.split("_")
                    g_val = float(parts[1][1:])
                    l_val = float(parts[2][1:])
                    needed_filters[ft] = CompositeFilter(
                        filter_type=ft,
                        description=f"ORB >= {g_val} + delay <= {l_val}",
                        base=OrbSizeFilter(
                            filter_type=f"ORB_G{int(g_val)}",
                            description=f"ORB size >= {g_val}",
                            min_size=g_val,
                        ),
                        overlay=BreakSpeedFilter(
                            filter_type=f"BRK_FAST{int(l_val)}",
                            description=f"Break delay <= {l_val} min",
                            max_delay_minutes=l_val,
                        ),
                    )
                else:
                    print(f"  WARNING: Unknown filter type '{ft}', using NO_FILTER fallback")
                    needed_filters[ft] = ALL_FILTERS["NO_FILTER"]

        # Load daily features (canonical: orb_minutes=5)
        features = _load_daily_features(con, instrument, 5, None, None)

        # Compute relative volumes if needed
        has_vol = any(isinstance(f, VolumeFilter) for f in needed_filters.values())
        if has_vol:
            _compute_relative_volumes(con, features, instrument, orb_labels, needed_filters)

        # Build filter day sets
        filter_days = _build_filter_day_sets(features, orb_labels, needed_filters)

        # Load trades for each slot
        for slot in inst_slots:
            sid = slot["head_strategy_id"]
            params = slot_params.get(sid)
            if params is None:
                trades_by_slot[sid] = []
                continue

            # Get eligible days from filter
            eligible = filter_days.get(
                (params["filter_type"], params["orb_label"]), set()
            )

            # Query outcomes
            rows = con.execute("""
                SELECT trading_day, outcome, pnl_r
                FROM orb_outcomes
                WHERE symbol = ?
                  AND orb_label = ?
                  AND orb_minutes = ?
                  AND entry_model = ?
                  AND rr_target = ?
                  AND confirm_bars = ?
                  AND outcome IN ('win', 'loss')
                ORDER BY trading_day
            """, [
                params["instrument"], params["orb_label"], params["orb_minutes"],
                params["entry_model"], params["rr_target"], params["confirm_bars"],
            ]).fetchall()

            # Filter to eligible days
            trades_by_slot[sid] = [
                {
                    "trading_day": r[0],
                    "outcome": r[1],
                    "pnl_r": r[2],
                    "instrument": instrument,
                    "session": slot["session"],
                    "strategy_id": sid,
                }
                for r in rows
                if r[0] in eligible
            ]

    return trades_by_slot


def print_slot_inventory(slots, trades_by_slot):
    """Print formatted inventory table of all session slots and their trade counts."""
    print()
    print("=" * 110)
    print("SLOT INVENTORY")
    print("=" * 110)
    print(
        f"{'Instrument':<12} {'Session':<20} {'Strategy ID':<47} {'ExpR':>6} {'ShANN':>7} {'N':>6} {'Tier':<10}"
    )
    print("-" * 110)

    total_n = 0
    for slot in slots:
        sid = slot["head_strategy_id"]
        sid_display = sid[:45] if len(sid) > 45 else sid
        n = len(trades_by_slot.get(sid, []))
        total_n += n

        print(
            f"{slot['instrument']:<12} "
            f"{slot['session']:<20} "
            f"{sid_display:<47} "
            f"{slot['head_expectancy_r']:>+6.3f} "
            f"{slot['head_sharpe_ann']:>7.2f} "
            f"{n:>6d} "
            f"{slot['trade_tier']:<10}"
        )

    print("-" * 110)
    print(f"{'TOTAL':<12} {'':<20} {len(slots):>47d} slots {'':<6} {'':<7} {total_n:>6d}")
    print("=" * 110)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Assembly Research Report"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(GOLD_DB_PATH),
        help="Path to DuckDB database (default: project gold.db)",
    )
    parser.add_argument(
        "--exclude-regime",
        action="store_true",
        help="Exclude REGIME-tier slots (keep only CORE)",
    )
    args = parser.parse_args()

    db_path = args.db_path

    print(f"Database: {db_path}")
    print(f"Exclude REGIME: {args.exclude_regime}")

    # Load session slots
    slots = session_slots(db_path)
    print(f"Loaded {len(slots)} session slots")

    if args.exclude_regime:
        before = len(slots)
        slots = [s for s in slots if s["trade_tier"] != "REGIME"]
        print(f"Filtered to {len(slots)} slots (excluded {before - len(slots)} REGIME)")

    if not slots:
        print("No slots found. Exiting.")
        return

    # Open DB read-only and load trades
    con = duckdb.connect(db_path, read_only=True)
    try:
        trades_by_slot = load_slot_trades(con, slots)
        print_slot_inventory(slots, trades_by_slot)
    finally:
        con.close()


if __name__ == "__main__":
    main()
