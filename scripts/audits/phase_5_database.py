#!/usr/bin/env python3
"""Phase 5 — Database Integrity.

Source: SYSTEM_AUDIT.md Phase 5 (lines 257-286)

Schema alignment, row count ratios, temporal coverage, orphan detection.
Beyond what audit_integrity.py covers.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS
from trading_app.config import ALL_FILTERS, ENTRY_MODELS

from scripts.audits import AuditPhase, Severity, db_connect

_SQL_IN = ", ".join(f"'{s}'" for s in ACTIVE_ORB_INSTRUMENTS)


def main():
    audit = AuditPhase(phase_num=5, name="Database Integrity")
    audit.print_header()

    con = db_connect()
    try:
        _check_schema_alignment(audit, con)
        _check_row_ratios(audit, con)
        _check_temporal_coverage(audit, con)
        _check_orphans(audit, con)
    finally:
        con.close()

    audit.run_and_exit()


def _check_schema_alignment(audit: AuditPhase, con):
    """5A — Schema alignment (init_db.py vs actual DB)."""
    print("\n--- 5A. Schema Alignment ---")

    # Get actual tables in DB
    tables_result = con.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """).fetchall()
    actual_tables = {row[0] for row in tables_result}

    expected_tables = {
        "bars_1m",
        "bars_5m",
        "daily_features",
        "orb_outcomes",
        "experimental_strategies",
        "validated_setups",
        "edge_families",
        "prospective_signals",
        "family_rr_locks",
        "rebuild_manifest",
    }

    missing = expected_tables - actual_tables
    extra = actual_tables - expected_tables

    if missing:
        audit.check_failed(f"Missing tables: {sorted(missing)}")
        audit.add_finding(
            Severity.CRITICAL,
            "SCHEMA_BEHIND",
            claimed=f"DB has all {len(expected_tables)} expected tables",
            actual=f"Missing: {sorted(missing)}",
            evidence="information_schema.tables",
            fix_type="CODE_FIX",
        )
    else:
        audit.check_passed(f"All {len(expected_tables)} expected tables present")

    if extra:
        audit.check_info(f"Extra tables in DB (not in expected set): {sorted(extra)}")

    # Check column counts for key tables
    for table in sorted(expected_tables & actual_tables):
        cols = con.execute(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = '{table}' AND table_schema = 'main'
        """).fetchall()
        audit.check_info(f"{table}: {len(cols)} columns")


def _check_row_ratios(audit: AuditPhase, con):
    """5B — Row count ratios."""
    print("\n--- 5B. Row Count Ratios ---")

    for inst in ACTIVE_ORB_INSTRUMENTS:
        symbol = ASSET_CONFIGS[inst]["symbol"]

        # bars_1m vs bars_5m ratio
        r1m = con.execute("SELECT COUNT(*) FROM bars_1m WHERE symbol = ?", [symbol]).fetchone()[0]
        r5m = con.execute("SELECT COUNT(*) FROM bars_5m WHERE symbol = ?", [symbol]).fetchone()[0]

        if r1m > 0 and r5m > 0:
            ratio = r1m / r5m
            expected = 5.0
            tolerance = 0.25  # ±5% of 5.0
            if abs(ratio - expected) <= tolerance:
                audit.check_passed(f"{inst}: bars_1m/bars_5m ratio = {ratio:.2f} (expected ~5.0)")
            else:
                audit.check_failed(f"{inst}: bars_1m/bars_5m ratio = {ratio:.2f} (expected ~5.0)")
                audit.add_finding(
                    Severity.HIGH,
                    "COUNT_ANOMALY",
                    claimed=f"{inst} bars_1m ≈ 5× bars_5m",
                    actual=f"Ratio = {ratio:.2f} ({r1m:,} / {r5m:,})",
                    evidence=f"bars_1m={r1m:,}, bars_5m={r5m:,}",
                    fix_type="REBUILD_NEEDED",
                )
        elif r1m == 0:
            audit.check_failed(f"{inst}: 0 rows in bars_1m")
        elif r5m == 0:
            audit.check_failed(f"{inst}: 0 rows in bars_5m")

    # daily_features: rows should be divisible by number of orb_minutes variants
    for inst in ACTIVE_ORB_INSTRUMENTS:
        symbol = ASSET_CONFIGS[inst]["symbol"]
        orb_counts = con.execute("""
            SELECT orb_minutes, COUNT(*) FROM daily_features
            WHERE symbol = ? GROUP BY orb_minutes ORDER BY orb_minutes
        """, [symbol]).fetchall()

        if orb_counts:
            counts = [c for _, c in orb_counts]
            # All orb_minutes should have same count (one per trading_day)
            if len(set(counts)) == 1:
                audit.check_passed(f"{inst}: daily_features balanced ({len(orb_counts)} apertures × {counts[0]:,} days)")
            else:
                audit.check_failed(f"{inst}: daily_features unbalanced: {dict(orb_counts)}")
                audit.add_finding(
                    Severity.HIGH,
                    "COUNT_ANOMALY",
                    claimed=f"{inst} daily_features balanced across orb_minutes",
                    actual=f"Counts: {dict(orb_counts)}",
                    evidence="daily_features GROUP BY orb_minutes",
                    fix_type="REBUILD_NEEDED",
                )

    # validated < experimental
    v_count = con.execute("SELECT COUNT(*) FROM validated_setups").fetchone()[0]
    e_count = con.execute("SELECT COUNT(*) FROM experimental_strategies").fetchone()[0]
    if v_count < e_count:
        audit.check_passed(f"validated ({v_count:,}) < experimental ({e_count:,})")
    else:
        audit.check_failed(f"validated ({v_count:,}) >= experimental ({e_count:,})")
        audit.add_finding(
            Severity.HIGH,
            "COUNT_ANOMALY",
            claimed="validated_setups < experimental_strategies",
            actual=f"validated={v_count:,}, experimental={e_count:,}",
            evidence="SELECT COUNT(*) FROM each table",
            fix_type="REBUILD_NEEDED",
        )

    # edge_families < validated
    ef_count = con.execute("SELECT COUNT(*) FROM edge_families").fetchone()[0]
    if ef_count <= v_count:
        audit.check_passed(f"edge_families ({ef_count:,}) <= validated ({v_count:,})")
    else:
        audit.check_failed(f"edge_families ({ef_count:,}) > validated ({v_count:,})")
        audit.add_finding(
            Severity.MEDIUM,
            "COUNT_ANOMALY",
            claimed="edge_families <= validated_setups",
            actual=f"edge_families={ef_count:,}, validated={v_count:,}",
            evidence="SELECT COUNT(*) FROM each table",
            fix_type="REBUILD_NEEDED",
        )


def _check_temporal_coverage(audit: AuditPhase, con):
    """5C — Temporal coverage per instrument."""
    print("\n--- 5C. Temporal Coverage ---")

    tables = ["bars_1m", "bars_5m", "daily_features", "orb_outcomes"]

    for inst in ACTIVE_ORB_INSTRUMENTS:
        symbol = ASSET_CONFIGS[inst]["symbol"]
        print(f"\n  {inst} ({symbol}):")
        dates = {}
        for table in tables:
            col = "trading_day" if table in ("daily_features", "orb_outcomes") else "ts_utc"
            if col == "ts_utc":
                r = con.execute(f"""
                    SELECT MIN({col})::DATE, MAX({col})::DATE
                    FROM {table} WHERE symbol = ?
                """, [symbol]).fetchone()
            else:
                r = con.execute(f"""
                    SELECT MIN({col}), MAX({col})
                    FROM {table} WHERE symbol = ?
                """, [symbol]).fetchone()
            if r and r[0]:
                dates[table] = (str(r[0]), str(r[1]))
                print(f"    {table}: {r[0]} → {r[1]}")
            else:
                dates[table] = (None, None)
                print(f"    {table}: NO DATA")

        # Check orb_outcomes latest == daily_features latest
        df_latest = dates.get("daily_features", (None, None))[1]
        oo_latest = dates.get("orb_outcomes", (None, None))[1]
        if df_latest and oo_latest:
            if df_latest == oo_latest:
                audit.check_passed(f"{inst}: orb_outcomes latest == daily_features latest ({df_latest})")
            else:
                audit.check_failed(f"{inst}: outcomes ({oo_latest}) != features ({df_latest})")
                audit.add_finding(
                    Severity.HIGH,
                    "STALE_OUTCOMES",
                    claimed=f"{inst} orb_outcomes up to date with daily_features",
                    actual=f"outcomes latest={oo_latest}, features latest={df_latest}",
                    evidence=f"MAX(trading_day) from both tables WHERE symbol='{symbol}'",
                    fix_type="REBUILD_NEEDED",
                )


def _check_orphans(audit: AuditPhase, con):
    """5D — Orphan detection."""
    print("\n--- 5D. Orphan Detection ---")

    config_filters = set(ALL_FILTERS.keys())
    entry_models = set(ENTRY_MODELS)

    # Validated with unknown filter_type
    r = con.execute("""
        SELECT DISTINCT filter_type FROM validated_setups WHERE status='active'
    """).fetchall()
    db_filters = {row[0] for row in r}
    orphan_filters = db_filters - config_filters
    if orphan_filters:
        audit.check_failed(f"validated_setups: {len(orphan_filters)} unknown filter_type(s)")
        for f in sorted(orphan_filters):
            print(f"         ORPHAN: {f}")
        audit.add_finding(
            Severity.HIGH,
            "ORPHAN_STRATEGY",
            claimed="All validated filter_types in ALL_FILTERS",
            actual=f"Unknown: {sorted(orphan_filters)}",
            evidence="DISTINCT filter_type FROM validated_setups",
            fix_type="DATA_FIX",
        )
    else:
        audit.check_passed(f"All {len(db_filters)} validated filter_types in ALL_FILTERS")

    # Validated with unknown entry_model
    r = con.execute("""
        SELECT DISTINCT entry_model FROM validated_setups WHERE status='active'
    """).fetchall()
    db_models = {row[0] for row in r}
    orphan_models = db_models - entry_models
    if orphan_models:
        audit.check_failed(f"validated_setups: unknown entry_model(s): {sorted(orphan_models)}")
        audit.add_finding(
            Severity.CRITICAL,
            "ORPHAN_STRATEGY",
            claimed="All validated entry_models in ENTRY_MODELS",
            actual=f"Unknown: {sorted(orphan_models)}",
            evidence="DISTINCT entry_model FROM validated_setups",
            fix_type="DATA_FIX",
        )
    else:
        audit.check_passed(f"All validated entry_models in ENTRY_MODELS: {sorted(db_models)}")

    # Edge families referencing nonexistent validated_setups
    try:
        r = con.execute("""
            SELECT COUNT(*) FROM edge_families ef
            LEFT JOIN validated_setups vs ON ef.family_hash = vs.family_hash
            WHERE vs.family_hash IS NULL
        """).fetchone()[0]
        if r > 0:
            audit.check_failed(f"edge_families: {r} orphan families (no matching validated_setups)")
            audit.add_finding(
                Severity.HIGH,
                "ORPHAN_FAMILY",
                claimed="All edge_families reference valid validated_setups",
                actual=f"{r} orphan families",
                evidence="LEFT JOIN edge_families ON validated_setups.family_hash WHERE NULL",
                fix_type="DATA_FIX",
            )
        else:
            audit.check_passed("No orphan edge_families")
    except Exception as e:
        audit.check_info(f"Edge family orphan check skipped: {e}")


if __name__ == "__main__":
    main()
