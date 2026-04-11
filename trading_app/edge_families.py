"""Canonical edge-family builder for validated strategies.

This module owns family assignment for the validated shelf. The CLI wrapper in
``scripts/tools/build_edge_families.py`` delegates here so validator writes and
manual rebuilds use the same logic.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from trading_app.config import CORE_MIN_SAMPLES, REGIME_MIN_SAMPLES
from trading_app.db_manager import compute_trade_day_hash
from trading_app.validated_shelf import deployable_validated_relation

# Robustness thresholds (Duke Protocol #3c)
# @research-source build_edge_families calibration — BH FDR-validated strategies only;
#   ShANN >= 0.8 (WHITELIST and SINGLETON) from Bailey et al. 2014 PBO analysis
#   (cited source: the_probability_of_backtest_overfitting.pdf §4.2 — NOT in resources/,
#    threshold aligned with WHITELIST 2026-03-31, see regime-validation-design.md)
#   CV <= 0.5 (coefficient of variation) from Carver Systematic Trading Ch.4 (member consistency)
#   min_trades thresholds align with CORE_MIN_SAMPLES=100 / REGIME_MIN_SAMPLES=30 from config.py
# @revalidated-for E1/E2 event-based sessions (2026-03-31)
MIN_FAMILY_SIZE = 5
WHITELIST_MIN_MEMBERS = 3
WHITELIST_MIN_SHANN = 0.8
WHITELIST_MAX_CV = 0.5
WHITELIST_MIN_TRADES = 50
SINGLETON_MIN_TRADES = 100
SINGLETON_MIN_SHANN = 0.8

CORE_MIN_TRADES = CORE_MIN_SAMPLES
REGIME_MIN_TRADES = REGIME_MIN_SAMPLES

compute_family_hash = compute_trade_day_hash


def classify_family(member_count, avg_shann, cv_expr, min_trades):
    """Classify family robustness status."""
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
    """Classify family trade tier by minimum member trade count."""
    if min_trades is None:
        return "INVALID"
    if min_trades >= CORE_MIN_TRADES:
        return "CORE"
    if min_trades >= REGIME_MIN_TRADES:
        return "REGIME"
    return "INVALID"


def elect_median_head(members):
    """Elect head as strategy closest to median ExpR."""
    exprs = [m[1] or 0 for m in members]
    med = statistics.median(exprs)

    best = None
    best_dist = float("inf")
    for sid, expr, shann, sample in members:
        dist = abs((expr or 0) - med)
        if dist < best_dist or (dist == best_dist and (best is None or sid < best[0])):
            best = (sid, expr, shann, sample)
            best_dist = dist

    return best, med


def _migrate_columns(con):
    """Ensure family columns exist on validated_setups and edge_families."""
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


def _ensure_edge_families_table(con) -> None:
    con.execute(
        """
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
            created_at        TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            pbo DOUBLE
        )
        """
    )


def build_edge_families_for_instrument(con, instrument: str) -> int:
    """Build edge families for one instrument using an existing DB connection."""
    _migrate_columns(con)
    _ensure_edge_families_table(con)

    shelf_relation = deployable_validated_relation(con, alias="vs")
    strategies = con.execute(
        f"""
        SELECT vs.strategy_id, vs.expectancy_r, vs.sharpe_ann, vs.sample_size,
               es.trade_day_hash, vs.orb_minutes
        FROM {shelf_relation}
        LEFT JOIN experimental_strategies es ON vs.strategy_id = es.strategy_id
        WHERE vs.instrument = ?
        ORDER BY vs.strategy_id
        """,
        [instrument],
    ).fetchall()

    con.execute("DELETE FROM edge_families WHERE instrument = ?", [instrument])
    con.execute(
        """
        UPDATE validated_setups
        SET family_hash = NULL, is_family_head = FALSE
        WHERE instrument = ?
        """,
        [instrument],
    )

    if not strategies:
        return 0

    hash_map = {}
    for sid, _expr, _shann, _sample, precomputed_hash, _orb_min in strategies:
        if precomputed_hash:
            hash_map[sid] = precomputed_hash
            continue
        days = con.execute(
            """
            SELECT trading_day FROM strategy_trade_days
            WHERE strategy_id = ? ORDER BY trading_day
            """,
            [sid],
        ).fetchall()
        hash_map[sid] = compute_family_hash([r[0] for r in days])

    families = defaultdict(list)
    for sid, expr, shann, sample, _precomputed_hash, orb_min in strategies:
        family_key = f"{instrument}_{orb_min or 5}m_{hash_map[sid]}"
        families[family_key].append((sid, expr, shann, sample))

    status_counts = defaultdict(int)
    family_heads = {}

    for family_hash, members in families.items():
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
            cv_expr = None

        (head_sid, head_expr, head_shann, _), _ = elect_median_head(members)
        family_heads[family_hash] = head_sid

        status = classify_family(len(members), avg_shann, cv_expr, min_trades)
        tier = classify_trade_tier(min_trades)
        status_counts[status] += 1

        trade_day_count = con.execute(
            "SELECT COUNT(*) FROM strategy_trade_days WHERE strategy_id = ?",
            [head_sid],
        ).fetchone()[0]
        if trade_day_count == 0:
            head_sample = next(m[3] for m in members if m[0] == head_sid) or 0
            trade_day_count = head_sample

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

    member_updates = []
    for family_hash, members in families.items():
        head_sid = family_heads.get(family_hash)
        if head_sid is None:
            continue
        for sid, _, _, _ in members:
            member_updates.append((sid, family_hash, sid == head_sid))

    if member_updates:
        con.execute(
            """
            CREATE TEMP TABLE _family_tags (
                strategy_id TEXT,
                family_hash TEXT,
                is_family_head BOOLEAN
            )
            """
        )
        con.executemany("INSERT INTO _family_tags VALUES (?, ?, ?)", member_updates)
        con.execute(
            """
            UPDATE validated_setups vs
            SET family_hash = ft.family_hash,
                is_family_head = ft.is_family_head
            FROM _family_tags ft
            WHERE vs.strategy_id = ft.strategy_id
            """
        )
        con.execute("DROP TABLE _family_tags")

    from trading_app.pbo import compute_family_pbo

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

    return len(families)
