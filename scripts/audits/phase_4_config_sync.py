#!/usr/bin/env python3
"""Phase 4 — Configuration Sync.

Source: SYSTEM_AUDIT.md Phase 4 (lines 218-254)

Cross-validates filter registry, entry models, session/ORB labels,
grid parameters, classification thresholds, and MCP allowlists.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS
from pipeline.dst import SESSION_CATALOG
from trading_app.config import (
    ALL_FILTERS,
    CORE_MIN_SAMPLES,
    ENTRY_MODELS,
    ORB_DURATION_MINUTES,
    REGIME_MIN_SAMPLES,
)
from trading_app.outcome_builder import CONFIRM_BARS_OPTIONS, RR_TARGETS

from scripts.audits import AuditPhase, Severity, db_connect

# Import MCP allowlists
from trading_app.ai.sql_adapter import (
    VALID_CONFIRM_BARS,
    VALID_ENTRY_MODELS,
    VALID_INSTRUMENTS,
    VALID_ORB_LABELS,
    VALID_RR_TARGETS,
)


def main():
    audit = AuditPhase(phase_num=4, name="Config Sync")
    audit.print_header()

    con = db_connect()
    try:
        _check_filter_registry(audit, con)
        _check_entry_model_sync(audit, con)
        _check_session_sync(audit)
        _check_grid_sync(audit)
        _check_threshold_sync(audit)
        _check_mcp_allowlists(audit)
    finally:
        con.close()

    audit.run_and_exit()


def _check_filter_registry(audit: AuditPhase, con):
    """4A — Filter Registry (beyond drift check 12)."""
    print("\n--- 4A. Filter Registry ---")

    config_filters = set(ALL_FILTERS.keys())
    audit.check_info(f"ALL_FILTERS has {len(config_filters)} entries")

    # DISTINCT filter_type from DB
    for table in ["validated_setups", "experimental_strategies"]:
        r = con.execute(f"SELECT DISTINCT filter_type FROM {table}").fetchall()
        db_filters = {row[0] for row in r}

        # Every DB value should be in ALL_FILTERS
        orphan = db_filters - config_filters
        if orphan:
            audit.check_failed(f"{table}: {len(orphan)} filter(s) in DB but not in ALL_FILTERS")
            for f in sorted(orphan):
                print(f"         PHANTOM: {f}")
            audit.add_finding(
                Severity.HIGH,
                "PHANTOM_FILTER",
                claimed="All DB filter_types in ALL_FILTERS",
                actual=f"{table} has unknown filters: {sorted(orphan)}",
                evidence=f"SELECT DISTINCT filter_type FROM {table}",
                fix_type="CONFIG_FIX",
            )
        else:
            audit.check_passed(f"{table}: all {len(db_filters)} DB filters in ALL_FILTERS")

    # Filters with zero rows in experimental (dead filters)
    dead_filters = []
    for f_name in sorted(config_filters):
        r = con.execute(
            "SELECT COUNT(*) FROM experimental_strategies WHERE filter_type = ?",
            [f_name],
        ).fetchone()[0]
        if r == 0:
            dead_filters.append(f_name)
    if dead_filters:
        audit.check_info(f"{len(dead_filters)} filter(s) in ALL_FILTERS with 0 experimental rows")
        for f in dead_filters[:10]:
            print(f"         DEAD_FILTER: {f}")
        if len(dead_filters) > 10:
            print(f"         ... and {len(dead_filters) - 10} more")
    else:
        audit.check_passed("All ALL_FILTERS entries have experimental rows")


def _check_entry_model_sync(audit: AuditPhase, con):
    """4B — Entry Model Sync (beyond drift check 13)."""
    print("\n--- 4B. Entry Model Sync ---")

    if ENTRY_MODELS == ["E1", "E2", "E3"]:
        audit.check_passed("ENTRY_MODELS = ['E1', 'E2', 'E3']")
    else:
        audit.check_failed(f"ENTRY_MODELS = {ENTRY_MODELS}")
        audit.add_finding(
            Severity.CRITICAL,
            "CONFIG_DRIFT",
            claimed="ENTRY_MODELS = ['E1', 'E2', 'E3']",
            actual=f"ENTRY_MODELS = {ENTRY_MODELS}",
            evidence="trading_app/config.py:ENTRY_MODELS",
            fix_type="CONFIG_FIX",
        )

    # Zero E0 in production
    for table in ["orb_outcomes", "experimental_strategies", "validated_setups"]:
        r = con.execute(f"SELECT COUNT(*) FROM {table} WHERE entry_model = 'E0'").fetchone()[0]
        if r > 0:
            audit.check_failed(f"{table}: {r} E0 rows")
            audit.add_finding(
                Severity.CRITICAL,
                "ZOMBIE_E0",
                claimed=f"Zero E0 in {table}",
                actual=f"{r} E0 rows",
                evidence=f"SELECT COUNT(*) FROM {table} WHERE entry_model='E0'",
                fix_type="DATA_FIX",
            )
        else:
            audit.check_passed(f"{table}: zero E0 rows")


def _check_session_sync(audit: AuditPhase):
    """4C — Session / ORB Label Sync (beyond drift check 32)."""
    print("\n--- 4C. Session / ORB Label Sync ---")

    catalog_keys = set(SESSION_CATALOG.keys())
    orb_duration_keys = set(ORB_DURATION_MINUTES.keys())
    enabled_union = set()
    for inst in ACTIVE_ORB_INSTRUMENTS:
        enabled_union.update(ASSET_CONFIGS[inst].get("enabled_sessions", []))

    # Three-way comparison
    all_sessions = catalog_keys | orb_duration_keys | enabled_union

    mismatches = []
    for s in sorted(all_sessions):
        in_catalog = s in catalog_keys
        in_orb = s in orb_duration_keys
        in_enabled = s in enabled_union
        if not (in_catalog and in_orb and in_enabled):
            parts = []
            if not in_catalog:
                parts.append("missing from SESSION_CATALOG")
            if not in_orb:
                parts.append("missing from ORB_DURATION_MINUTES")
            if not in_enabled:
                parts.append("not enabled for any instrument")
            mismatches.append((s, ", ".join(parts)))

    if mismatches:
        for s, detail in mismatches:
            audit.check_failed(f"{s}: {detail}")
            audit.add_finding(
                Severity.HIGH,
                "SESSION_LABEL_DRIFT",
                claimed="Three-way session sync (catalog, ORB durations, enabled)",
                actual=f"{s}: {detail}",
                evidence="SESSION_CATALOG vs ORB_DURATION_MINUTES vs enabled_sessions",
                fix_type="CONFIG_FIX",
            )
    else:
        audit.check_passed(f"Three-way session sync: {len(all_sessions)} sessions aligned")


def _check_grid_sync(audit: AuditPhase):
    """4D — Grid Sync."""
    print("\n--- 4D. Grid Sync ---")

    # RR_TARGETS
    expected_rr = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    if RR_TARGETS == expected_rr:
        audit.check_passed(f"RR_TARGETS = {RR_TARGETS}")
    else:
        audit.check_info(f"RR_TARGETS = {RR_TARGETS}")

    # CB_LEVELS
    expected_cb = [1, 2, 3, 4, 5]
    if CONFIRM_BARS_OPTIONS == expected_cb:
        audit.check_passed(f"CONFIRM_BARS_OPTIONS = {CONFIRM_BARS_OPTIONS}")
    else:
        audit.check_info(f"CONFIRM_BARS_OPTIONS = {CONFIRM_BARS_OPTIONS}")


def _check_threshold_sync(audit: AuditPhase):
    """4E — Threshold Sync."""
    print("\n--- 4E. Threshold Sync ---")

    if CORE_MIN_SAMPLES == 100:
        audit.check_passed(f"CORE_MIN_SAMPLES = {CORE_MIN_SAMPLES}")
    else:
        audit.check_failed(f"CORE_MIN_SAMPLES = {CORE_MIN_SAMPLES} (expected 100)")
        audit.add_finding(
            Severity.HIGH,
            "THRESHOLD_DRIFT",
            claimed="CORE_MIN_SAMPLES = 100",
            actual=f"CORE_MIN_SAMPLES = {CORE_MIN_SAMPLES}",
            evidence="trading_app/config.py:CORE_MIN_SAMPLES",
            fix_type="CONFIG_FIX",
        )

    if REGIME_MIN_SAMPLES == 30:
        audit.check_passed(f"REGIME_MIN_SAMPLES = {REGIME_MIN_SAMPLES}")
    else:
        audit.check_failed(f"REGIME_MIN_SAMPLES = {REGIME_MIN_SAMPLES} (expected 30)")
        audit.add_finding(
            Severity.HIGH,
            "THRESHOLD_DRIFT",
            claimed="REGIME_MIN_SAMPLES = 30",
            actual=f"REGIME_MIN_SAMPLES = {REGIME_MIN_SAMPLES}",
            evidence="trading_app/config.py:REGIME_MIN_SAMPLES",
            fix_type="CONFIG_FIX",
        )


def _check_mcp_allowlists(audit: AuditPhase):
    """4F — MCP Parameter Allowlists."""
    print("\n--- 4F. MCP Allowlists ---")

    # VALID_ENTRY_MODELS should match ENTRY_MODELS
    config_set = set(ENTRY_MODELS)
    if VALID_ENTRY_MODELS == config_set:
        audit.check_passed(f"VALID_ENTRY_MODELS matches ENTRY_MODELS: {sorted(config_set)}")
    else:
        diff = VALID_ENTRY_MODELS.symmetric_difference(config_set)
        audit.check_failed(f"VALID_ENTRY_MODELS mismatch: diff={sorted(diff)}")
        audit.add_finding(
            Severity.HIGH,
            "MCP_ALLOWLIST_STALE",
            claimed="VALID_ENTRY_MODELS == ENTRY_MODELS",
            actual=f"Difference: {sorted(diff)}",
            evidence="sql_adapter.py:VALID_ENTRY_MODELS vs config.py:ENTRY_MODELS",
            fix_type="CONFIG_FIX",
        )

    # VALID_INSTRUMENTS should match ACTIVE_ORB_INSTRUMENTS
    active_set = set(ACTIVE_ORB_INSTRUMENTS)
    if VALID_INSTRUMENTS == active_set:
        audit.check_passed(f"VALID_INSTRUMENTS matches ACTIVE_ORB_INSTRUMENTS: {sorted(active_set)}")
    else:
        diff = VALID_INSTRUMENTS.symmetric_difference(active_set)
        audit.check_failed(f"VALID_INSTRUMENTS mismatch: diff={sorted(diff)}")
        audit.add_finding(
            Severity.HIGH,
            "MCP_ALLOWLIST_STALE",
            claimed="VALID_INSTRUMENTS == ACTIVE_ORB_INSTRUMENTS",
            actual=f"Difference: {sorted(diff)}",
            evidence="sql_adapter.py:VALID_INSTRUMENTS vs asset_configs.py:ACTIVE_ORB_INSTRUMENTS",
            fix_type="CONFIG_FIX",
        )

    # VALID_ORB_LABELS should match SESSION_CATALOG keys
    catalog_set = set(SESSION_CATALOG.keys())
    if VALID_ORB_LABELS == catalog_set:
        audit.check_passed(f"VALID_ORB_LABELS matches SESSION_CATALOG: {len(catalog_set)} sessions")
    else:
        diff = VALID_ORB_LABELS.symmetric_difference(catalog_set)
        audit.check_failed(f"VALID_ORB_LABELS mismatch: diff={sorted(diff)}")
        audit.add_finding(
            Severity.HIGH,
            "MCP_ALLOWLIST_STALE",
            claimed="VALID_ORB_LABELS == SESSION_CATALOG keys",
            actual=f"Difference: {sorted(diff)}",
            evidence="sql_adapter.py:VALID_ORB_LABELS vs dst.py:SESSION_CATALOG",
            fix_type="CONFIG_FIX",
        )

    # VALID_RR_TARGETS should match RR_TARGETS
    rr_set = set(RR_TARGETS)
    if VALID_RR_TARGETS == rr_set:
        audit.check_passed(f"VALID_RR_TARGETS matches RR_TARGETS: {sorted(rr_set)}")
    else:
        diff = VALID_RR_TARGETS.symmetric_difference(rr_set)
        audit.check_failed(f"VALID_RR_TARGETS mismatch: diff={sorted(diff)}")
        audit.add_finding(
            Severity.HIGH,
            "MCP_ALLOWLIST_STALE",
            claimed="VALID_RR_TARGETS == RR_TARGETS",
            actual=f"Difference: {sorted(diff)}",
            evidence="sql_adapter.py:VALID_RR_TARGETS vs outcome_builder.py:RR_TARGETS",
            fix_type="CONFIG_FIX",
        )

    # VALID_CONFIRM_BARS should match CONFIRM_BARS_OPTIONS
    cb_set = set(CONFIRM_BARS_OPTIONS)
    if VALID_CONFIRM_BARS == cb_set:
        audit.check_passed(f"VALID_CONFIRM_BARS matches CONFIRM_BARS_OPTIONS: {sorted(cb_set)}")
    else:
        diff = VALID_CONFIRM_BARS.symmetric_difference(cb_set)
        audit.check_failed(f"VALID_CONFIRM_BARS mismatch: diff={sorted(diff)}")
        audit.add_finding(
            Severity.HIGH,
            "MCP_ALLOWLIST_STALE",
            claimed="VALID_CONFIRM_BARS == CONFIRM_BARS_OPTIONS",
            actual=f"Difference: {sorted(diff)}",
            evidence="sql_adapter.py:VALID_CONFIRM_BARS vs outcome_builder.py:CONFIRM_BARS_OPTIONS",
            fix_type="CONFIG_FIX",
        )


if __name__ == "__main__":
    main()
