"""
Build edge families by hashing strategy trade-day lists.

Groups validated strategies that share identical post-filter trade-day
patterns. Elects a cluster head (best ExpR) per family.

Usage:
    python scripts/build_edge_families.py --instrument MGC --db-path C:/db/gold.db
    python scripts/build_edge_families.py --all --db-path C:/db/gold.db
"""

import sys
import hashlib
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

# Force unbuffered stdout (Windows cp1252 buffering issue)
sys.stdout.reconfigure(line_buffering=True)


def compute_family_hash(days: list) -> str:
    """Compute deterministic MD5 hash of sorted trade-day list."""
    day_str = ",".join(str(d) for d in sorted(days))
    return hashlib.md5(day_str.encode()).hexdigest()


def _migrate_columns(con):
    """Add family_hash and is_family_head columns if missing (existing DB migration)."""
    cols = {
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'validated_setups'"
        ).fetchall()
    }
    if "family_hash" not in cols:
        con.execute("ALTER TABLE validated_setups ADD COLUMN family_hash TEXT")
    if "is_family_head" not in cols:
        con.execute(
            "ALTER TABLE validated_setups ADD COLUMN is_family_head BOOLEAN DEFAULT FALSE"
        )
    con.commit()


def build_edge_families(db_path: str, instrument: str) -> int:
    """
    Build edge families for one instrument.

    Returns number of unique families found.
    """
    con = duckdb.connect(str(db_path))
    try:
        _migrate_columns(con)

        # Ensure edge_families table exists
        con.execute("""
            CREATE TABLE IF NOT EXISTS edge_families (
                family_hash       TEXT        PRIMARY KEY,
                instrument        TEXT        NOT NULL,
                member_count      INTEGER     NOT NULL,
                trade_day_count   INTEGER     NOT NULL,
                head_strategy_id  TEXT        NOT NULL,
                head_expectancy_r DOUBLE,
                head_sharpe_ann   DOUBLE,
                created_at        TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 1. Load validated strategies
        strategies = con.execute("""
            SELECT strategy_id, expectancy_r, sharpe_ann
            FROM validated_setups
            WHERE instrument = ? AND LOWER(status) = 'active'
            ORDER BY strategy_id
        """, [instrument]).fetchall()

        print(f"Building edge families for {len(strategies)} {instrument} strategies")

        if not strategies:
            print(f"No active strategies for {instrument}")
            return 0

        # 2. Compute hash per strategy
        hash_map = {}  # strategy_id -> family_hash
        for sid, expr, shann in strategies:
            days = con.execute("""
                SELECT trading_day FROM strategy_trade_days
                WHERE strategy_id = ?
                ORDER BY trading_day
            """, [sid]).fetchall()

            day_list = [r[0] for r in days]
            h = compute_family_hash(day_list)
            hash_map[sid] = h

        # 3. Group by hash
        families = defaultdict(list)  # hash -> [(sid, expr, shann)]
        for sid, expr, shann in strategies:
            families[hash_map[sid]].append((sid, expr, shann))

        print(f"  {len(strategies)} strategies -> {len(families)} unique families")

        # 4. Clear existing families for this instrument
        con.execute(
            "DELETE FROM edge_families WHERE instrument = ?", [instrument]
        )

        # 5. Reset family columns on validated_setups
        con.execute("""
            UPDATE validated_setups
            SET family_hash = NULL, is_family_head = FALSE
            WHERE instrument = ?
        """, [instrument])

        # 6. For each family: elect head, insert edge_families, update validated_setups
        for family_hash, members in families.items():
            # Head = best ExpR (among members)
            members_sorted = sorted(members, key=lambda m: m[1] or 0, reverse=True)
            head_sid, head_expr, head_shann = members_sorted[0]

            # Trade day count (all members share the same days, pick any)
            trade_day_count = con.execute("""
                SELECT COUNT(*) FROM strategy_trade_days
                WHERE strategy_id = ?
            """, [head_sid]).fetchone()[0]

            # Insert edge family
            con.execute("""
                INSERT INTO edge_families
                (family_hash, instrument, member_count, trade_day_count,
                 head_strategy_id, head_expectancy_r, head_sharpe_ann)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                family_hash, instrument, len(members), trade_day_count,
                head_sid, head_expr, head_shann,
            ])

            # Tag all members with family_hash
            for sid, _, _ in members:
                is_head = sid == head_sid
                con.execute("""
                    UPDATE validated_setups
                    SET family_hash = ?, is_family_head = ?
                    WHERE strategy_id = ?
                """, [family_hash, is_head, sid])

        con.commit()

        # 7. Summary
        size_dist = sorted(
            [len(m) for m in families.values()], reverse=True
        )
        print(f"  Family sizes: max={size_dist[0]}, "
              f"median={size_dist[len(size_dist)//2]}, "
              f"singletons={size_dist.count(1)}")

        return len(families)

    finally:
        con.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build edge families from strategy trade-day hashes"
    )
    parser.add_argument("--instrument", help="Instrument symbol")
    parser.add_argument(
        "--db-path", default="C:/db/gold.db", help="Database path"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run for all instruments"
    )
    args = parser.parse_args()

    if not args.all and not args.instrument:
        parser.error("Either --instrument or --all is required")

    if args.all:
        total = 0
        for inst in ["MGC", "MNQ", "MES", "MCL"]:
            total += build_edge_families(args.db_path, inst)
            print()
        print(f"Grand total: {total} unique edge families")
    else:
        build_edge_families(args.db_path, args.instrument)


if __name__ == "__main__":
    main()
