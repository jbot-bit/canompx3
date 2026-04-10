"""
DISCOVERY SAFETY: UNSAFE — reads validated_setups, writes edge_families (both derived).
Do not use edge_families output as discovery truth. See CLAUDE.md Project Truth Protocol.

CLI wrapper for canonical edge-family assignment.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH
from trading_app.edge_families import (
    build_edge_families_for_instrument,
)
from trading_app.edge_families import (
    classify_family as _classify_family,
)
from trading_app.edge_families import (
    classify_trade_tier as _classify_trade_tier,
)
from trading_app.edge_families import (
    compute_family_hash as _compute_family_hash,
)
from trading_app.edge_families import (
    elect_median_head as _edge_elect_median_head,
)

# Force unbuffered stdout (Windows cp1252 buffering issue)
sys.stdout.reconfigure(line_buffering=True)

classify_family = _classify_family
classify_trade_tier = _classify_trade_tier
compute_family_hash = _compute_family_hash
_elect_median_head = _edge_elect_median_head


def build_edge_families(db_path: str, instrument: str) -> int:
    """Build edge families for one instrument."""
    con = duckdb.connect(str(db_path))
    try:
        count = build_edge_families_for_instrument(con, instrument)
        con.commit()
        return count
    finally:
        con.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build edge families with robustness filter")
    parser.add_argument("--instrument", help="Instrument symbol")
    parser.add_argument("--db-path", default=str(GOLD_DB_PATH), help="Database path")
    parser.add_argument("--all", action="store_true", help="Run for all instruments")
    args = parser.parse_args()

    if not args.all and not args.instrument:
        parser.error("Either --instrument or --all is required")

    if args.all:
        total = 0
        for inst in ACTIVE_ORB_INSTRUMENTS:
            total += build_edge_families(args.db_path, inst)
            print()
        print(f"Grand total: {total} unique edge families")
    else:
        build_edge_families(args.db_path, args.instrument)


if __name__ == "__main__":
    main()
