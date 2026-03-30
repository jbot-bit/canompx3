"""
DISCOVERY SAFETY: UNSAFE — reads validated_setups, writes edge_families (both derived).
Do not use edge_families output as discovery truth. See CLAUDE.md Project Truth Protocol.

Build edge families by hashing strategy trade-day lists.

Groups validated strategies that share identical post-filter trade-day
patterns. Elects cluster head by MEDIAN ExpR (not max — avoids Winner's
Curse). Applies robustness filter: N>=5 = ROBUST, N in [3,4] whitelisted
if ShANN>=0.8 AND CV<=0.5 AND min_trades>=50. Trade tier: CORE/REGIME/INVALID.

Usage:
    python scripts/tools/build_edge_families.py --instrument MGC
    python scripts/tools/build_edge_families.py --all
"""

import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import duckdb

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
from pipeline.paths import GOLD_DB_PATH
from trading_app.config import CORE_MIN_SAMPLES, REGIME_MIN_SAMPLES
from trading_app.db_manager import compute_trade_day_hash

# Force unbuffered stdout (Windows cp1252 buffering issue)
sys.stdout.reconfigure(line_buffering=True)

# Robustness thresholds (Duke Protocol #3c)
# @research-source build_edge_families calibration — BH FDR-validated strategies only;
#   ShANN >= 0.8 (WHITELIST and SINGLETON) from Bailey et al. 2014 PBO analysis
#   (cited source: the_probability_of_backtest_overfitting.pdf §4.2 — NOT in resources/,
#    threshold aligned with WHITELIST 2026-03-31, see regime-validation-design.md)
#   CV <= 0.5 (coefficient of variation) from Carver Systematic Trading Ch.4 (member consistency)
#   min_trades thresholds align with CORE_MIN_SAMPLES=100 / REGIME_MIN_SAMPLES=30 from config.py
# @revalidated-for E1/E2 event-based sessions (2026-03-31)
MIN_FAMILY_SIZE = 5
WHITELIST_MIN_MEMBERS = 3  # N>=3 to avoid "one lucky sibling" at higher CV
WHITELIST_MIN_SHANN = 0.8
WHITELIST_MAX_CV = 0.5
WHITELIST_MIN_TRADES = 50
SINGLETON_MIN_TRADES = 100  # N=1 family: isolated edge quality bar
SINGLETON_MIN_SHANN = 0.8  # N=1 family: aligned with WHITELIST (was 1.0, ungrounded gap)

# Trade tier thresholds — imported from config.py (single source of truth)
CORE_MIN_TRADES = CORE_MIN_SAMPLES
REGIME_MIN_TRADES = REGIME_MIN_SAMPLES

compute_family_hash = compute_trade_day_hash  # public alias for backward compat


def classify_family(member_count, avg_shann, cv_expr, min_trades):
    """Classify family robustness status.

    ROBUST:      N>=5 (parameter-stable across 5+ combos)
    WHITELISTED: N in [3,4] with strong metrics (structurally small family)
    SINGLETON:   N=1 with quality bar (isolated edge, not overfit)
    PURGED:      everything else
    """
    if member_count >= MIN_FAMILY_SIZE:
        return "ROBUST"
    if (
        member_count >= WHITELIST_MIN_MEMBERS
        and avg_shann is not None
        and avg_shann >= WHITELIST_MIN_SHANN
        and cv_expr is not None
        and cv_expr <= WHITELIST_MAX_CV
        and min_trades is not None
        and min_trades >= WHITELIST_MIN_TRADES
    ):
        return "WHITELISTED"
    if (
        member_count == 1
        and min_trades is not None
        and min_trades >= SINGLETON_MIN_TRADES
        and avg_shann is not None
        and avg_shann >= SINGLETON_MIN_SHANN
    ):
        return "SINGLETON"
    return "PURGED"


def classify_trade_tier(min_trades):
    """Classify family trade tier by minimum member trade count.

    CORE: min_trades >= 100 (standalone portfolio weight)
    REGIME: 30 <= min_trades < 100 (conditional overlay / signal only)
    INVALID: min_trades < 30 (not tradeable)
    """
    if min_trades is None:
        return "INVALID"
    if min_trades >= CORE_MIN_TRADES:
        return "CORE"
    if min_trades >= REGIME_MIN_TRADES:
        return "REGIME"
    return "INVALID"


def _elect_median_head(members):
    """Elect head as strategy closest to median ExpR.

    Avoids Winner's Curse (selecting max = selection bias).
    Tiebreak: lower strategy_id (deterministic).
    """
    exprs = [m[1] or 0 for m in members]
    med = statistics.median(exprs)

    # Find member closest to median
    best = None
    best_dist = float("inf")
    for sid, expr, shann, sample in members:
        dist = abs((expr or 0) - med)
        if dist < best_dist or (dist == best_dist and (best is None or sid < best[0])):
            best = (sid, expr, shann, sample)
            best_dist = dist

    return best, med


def _migrate_columns(con):
    """Add family_hash and is_family_head columns if missing (existing DB migration)."""
    # validated_setups columns
    vs_cols = {
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'validated_setups'"
        ).fetchall()
    }
    if "family_hash" not in vs_cols:
        con.execute("ALTER TABLE validated_setups ADD COLUMN family_hash TEXT")
    if "is_family_head" not in vs_cols:
        con.execute("ALTER TABLE validated_setups ADD COLUMN is_family_head BOOLEAN DEFAULT FALSE")

    # edge_families robustness columns
    ef_tables = {
        r[0]
        for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
    }
    if "edge_families" in ef_tables:
        ef_cols = {
            r[0]
            for r in con.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'edge_families'"
            ).fetchall()
        }
        for col, typ, default in [
            ("robustness_status", "TEXT", "'PENDING'"),
            ("cv_expectancy", "DOUBLE", None),
            ("median_expectancy_r", "DOUBLE", None),
            ("avg_sharpe_ann", "DOUBLE", None),
            ("min_member_trades", "INTEGER", None),
            ("trade_tier", "TEXT", "'PENDING'"),
        ]:
            if col not in ef_cols:
                dflt = f" DEFAULT {default}" if default else ""
                con.execute(f"ALTER TABLE edge_families ADD COLUMN {col} {typ}{dflt}")

    con.commit()


def build_edge_families(db_path: str, instrument: str) -> int:
    """
    Build edge families for one instrument.

    Returns number of unique families found.
    """
    con = duckdb.connect(str(db_path))
    try:
        _migrate_columns(con)

        # Ensure edge_families table exists (with robustness columns)
        con.execute("""
            CREATE TABLE IF NOT EXISTS edge_families (
                family_hash       TEXT        PRIMARY KEY,
                instrument        TEXT        NOT NULL,
                member_count      INTEGER     NOT NULL,
                trade_day_count   INTEGER     NOT NULL,
                head_strategy_id  TEXT        NOT NULL,
                head_expectancy_r DOUBLE,
                head_sharpe_ann   DOUBLE,
                robustness_status   TEXT      DEFAULT 'PENDING',
                cv_expectancy       DOUBLE,
                median_expectancy_r DOUBLE,
                avg_sharpe_ann      DOUBLE,
                min_member_trades   INTEGER,
                trade_tier          TEXT      DEFAULT 'PENDING',
                created_at        TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 1. Load validated strategies; JOIN experimental_strategies for pre-computed hash
        #    Include orb_minutes to prevent cross-duration family contamination
        #    (5m and 15m strategies must NEVER share a family hash).
        strategies = con.execute(
            """
            SELECT vs.strategy_id, vs.expectancy_r, vs.sharpe_ann, vs.sample_size,
                   es.trade_day_hash, vs.orb_minutes
            FROM validated_setups vs
            LEFT JOIN experimental_strategies es ON vs.strategy_id = es.strategy_id
            WHERE vs.instrument = ? AND LOWER(vs.status) = 'active'
            ORDER BY vs.strategy_id
        """,
            [instrument],
        ).fetchall()

        print(f"Building edge families for {len(strategies)} {instrument} strategies")

        if not strategies:
            print(f"No active strategies for {instrument}")
            return 0

        # 2. Resolve hash per strategy: use pre-computed hash from experimental_strategies
        #    (100% populated for --no-walkforward runs); fall back to strategy_trade_days
        #    for legacy walk-forward strategies that didn't store trade_day_hash.
        hash_map = {}
        fallback_count = 0
        for sid, _expr, _shann, _sample, precomputed_hash, _orb_min in strategies:
            if precomputed_hash:
                hash_map[sid] = precomputed_hash
            else:
                days = con.execute(
                    """
                    SELECT trading_day FROM strategy_trade_days
                    WHERE strategy_id = ? ORDER BY trading_day
                """,
                    [sid],
                ).fetchall()
                hash_map[sid] = compute_family_hash([r[0] for r in days])
                fallback_count += 1

        if fallback_count:
            print(f"  WARNING: {fallback_count} strategies used strategy_trade_days fallback")

        # 3. Group by instrument+orb_minutes-prefixed hash -> [(sid, expr, shann, sample_size)]
        #    Prefix prevents cross-instrument and cross-duration PRIMARY KEY collisions.
        #    Without orb_minutes prefix, a 5m and 15m strategy with identical trade days
        #    would merge into one family — wrong (different ORB apertures).
        families = defaultdict(list)
        for sid, expr, shann, sample, _, orb_min in strategies:
            family_key = f"{instrument}_{orb_min or 5}m_{hash_map[sid]}"
            families[family_key].append((sid, expr, shann, sample))

        print(f"  {len(strategies)} strategies -> {len(families)} unique families")

        # 4. Clear existing families for this instrument
        con.execute("DELETE FROM edge_families WHERE instrument = ?", [instrument])

        # 5. Reset family columns on validated_setups
        con.execute(
            """
            UPDATE validated_setups
            SET family_hash = NULL, is_family_head = FALSE
            WHERE instrument = ?
        """,
            [instrument],
        )

        # 6. For each family: compute metrics, elect median head, classify
        status_counts = defaultdict(int)
        family_heads = {}  # family_hash -> head_sid

        for family_hash, members in families.items():
            # Compute family metrics
            exprs = [m[1] or 0 for m in members]
            shanns = [m[2] for m in members if m[2] is not None]
            samples = [m[3] for m in members if m[3] is not None]

            med_expr = statistics.median(exprs)
            avg_shann = statistics.mean(shanns) if shanns else None
            min_trades = min(samples) if samples else None

            if len(exprs) > 1:
                std_expr = statistics.stdev(exprs)
                mean_expr = statistics.mean(exprs)
                cv_expr = std_expr / mean_expr if mean_expr > 0 else float("inf")
            else:
                cv_expr = None  # Singletons have no CV

            # Elect head by MEDIAN (not max)
            (head_sid, head_expr, head_shann, _), _ = _elect_median_head(members)
            family_heads[family_hash] = head_sid

            # Classify robustness + trade tier
            status = classify_family(len(members), avg_shann, cv_expr, min_trades)
            tier = classify_trade_tier(min_trades)
            status_counts[status] += 1

            # Trade day count — strategy_trade_days may be absent for --no-walkforward runs;
            # fall back to head strategy's sample_size from validated_setups.
            trade_day_count = con.execute(
                """
                SELECT COUNT(*) FROM strategy_trade_days WHERE strategy_id = ?
            """,
                [head_sid],
            ).fetchone()[0]

            if trade_day_count == 0:
                head_sample = next(m[3] for m in members if m[0] == head_sid) or 0
                trade_day_count = head_sample

            # Insert edge family
            con.execute(
                """
                INSERT INTO edge_families
                (family_hash, instrument, member_count, trade_day_count,
                 head_strategy_id, head_expectancy_r, head_sharpe_ann,
                 robustness_status, cv_expectancy, median_expectancy_r,
                 avg_sharpe_ann, min_member_trades, trade_tier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    family_hash,
                    instrument,
                    len(members),
                    trade_day_count,
                    head_sid,
                    head_expr,
                    head_shann,
                    status,
                    cv_expr,
                    med_expr,
                    avg_shann,
                    min_trades,
                    tier,
                ],
            )

        # 6a. Batch-tag all members via temp table (replaces row-by-row UPDATEs)
        member_updates = []
        for family_hash, members in families.items():
            head_sid = family_heads.get(family_hash)
            if head_sid is None:
                # Loop 1 was interrupted before processing this family — skip
                print(f"  WARNING: family {family_hash} missing head (partial failure in loop 1)")
                continue
            for sid, _, _, _ in members:
                member_updates.append((sid, family_hash, sid == head_sid))

        if member_updates:
            con.execute("""
                CREATE TEMP TABLE _family_tags (
                    strategy_id TEXT,
                    family_hash TEXT,
                    is_family_head BOOLEAN
                )
            """)
            con.executemany(
                "INSERT INTO _family_tags VALUES (?, ?, ?)",
                member_updates,
            )
            con.execute("""
                UPDATE validated_setups vs
                SET family_hash = ft.family_hash,
                    is_family_head = ft.is_family_head
                FROM _family_tags ft
                WHERE vs.strategy_id = ft.strategy_id
            """)
            con.execute("DROP TABLE _family_tags")

        # 6b. Compute PBO for families with 2+ members (Bailey et al. 2014)
        from trading_app.pbo import compute_family_pbo

        pbo_computed = 0
        pbo_high = 0  # PBO > 0.50 (likely overfit selection)
        for family_hash, members in families.items():
            if len(members) < 2:
                continue

            pbo_result = compute_family_pbo(con, family_hash, instrument)
            pbo_val = pbo_result.get("pbo")
            if pbo_val is not None:
                con.execute(
                    "UPDATE edge_families SET pbo = ? WHERE family_hash = ?",
                    [pbo_val, family_hash],
                )
                pbo_computed += 1
                if pbo_val > 0.50:
                    pbo_high += 1

        if pbo_computed:
            print(f"  PBO computed for {pbo_computed} families ({pbo_high} with PBO > 0.50)")

        # 7. Fail gates — abort before commit if data quality checks fail
        total_fam = len(families)
        singleton_count = status_counts.get("SINGLETON", 0)
        max_members = max(len(m) for m in families.values()) if families else 0

        if max_members > 100:
            raise RuntimeError(
                f"ABORT: Mega-family detected — max_members={max_members} > 100. "
                "Possible hash collision or fallback to empty hash."
            )
        if total_fam > 50 and singleton_count / total_fam > 0.70:
            raise RuntimeError(
                f"ABORT: SINGLETON rate {singleton_count}/{total_fam} "
                f"({100 * singleton_count // total_fam}%) exceeds 50% guard. "
                "Review singleton thresholds."
            )

        con.commit()

        # 8. Summary
        size_dist = sorted([len(m) for m in families.values()], reverse=True)
        print(
            f"  Family sizes: max={size_dist[0]}, "
            f"median={size_dist[len(size_dist) // 2]}, "
            f"singletons={size_dist.count(1)}"
        )
        for status in ["ROBUST", "WHITELISTED", "SINGLETON", "PURGED"]:
            print(f"  {status}: {status_counts.get(status, 0)} families")

        return len(families)

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
