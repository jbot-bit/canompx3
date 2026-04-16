#!/usr/bin/env python3
"""Phase 7 — Live Trading Readiness.

Source: SYSTEM_AUDIT.md Phase 7 (lines 316-355)

Live config coherence, execution engine wiring, risk manager,
and backtest↔live feature parity.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.dst import SESSION_CATALOG
from pipeline.paths import PROJECT_ROOT
from scripts.audits import AuditPhase, Severity, db_connect
from trading_app.config import ALL_FILTERS, ENTRY_MODELS, EXCLUDED_FROM_FITNESS, get_excluded_sessions
from trading_app.prop_profiles import get_active_profile_ids, get_profile_lane_definitions
from trading_app.validated_shelf import deployable_validated_relation


def main():
    audit = AuditPhase(phase_num=7, name="Live Trading Readiness")
    audit.print_header()

    con = db_connect()
    try:
        _check_live_config_coherence(audit, con)
        _check_execution_wiring(audit)
        _check_feature_parity(audit)
    finally:
        con.close()

    audit.run_and_exit()


def _check_live_config_coherence(audit: AuditPhase, con):
    """7A — Runtime lane coherence against canonical profile + shelf authorities."""
    print("\n--- 7A. Live Config Coherence ---")

    active_profiles = get_active_profile_ids(require_daily_lanes=False, exclude_self_funded=False)
    active_profiles_with_lanes = get_active_profile_ids(require_daily_lanes=True, exclude_self_funded=False)
    if not active_profiles:
        audit.check_info("No active execution profiles")
        return

    audit.check_info(f"Active execution profiles: {len(active_profiles)}")
    if not active_profiles_with_lanes:
        audit.check_info("No active profiles with explicit daily lanes")
        return

    lane_defs: list[dict] = []
    for profile_id in active_profiles_with_lanes:
        lanes = get_profile_lane_definitions(profile_id)
        lane_defs.extend(lanes)
        audit.check_info(f"{profile_id}: {len(lanes)} configured lane(s)")

    catalog_keys = set(SESSION_CATALOG.keys())
    filter_keys = set(ALL_FILTERS.keys())
    entry_set = set(ENTRY_MODELS)

    for lane in lane_defs:
        issues = []

        if lane["orb_label"] not in catalog_keys:
            issues.append(f"orb_label '{lane['orb_label']}' not in SESSION_CATALOG")
        if lane["entry_model"] not in entry_set:
            issues.append(f"entry_model '{lane['entry_model']}' not in ENTRY_MODELS")
        if lane["filter_type"] not in filter_keys:
            issues.append(f"filter_type '{lane['filter_type']}' not in ALL_FILTERS")

        if issues:
            audit.check_failed(f"{lane['strategy_id']}: {'; '.join(issues)}")
            audit.add_finding(
                Severity.HIGH,
                "CONFIG_DRIFT",
                claimed=f"{lane['strategy_id']} references valid runtime config values",
                actual="; ".join(issues),
                evidence="trading_app/prop_profiles.py daily_lanes",
                fix_type="CONFIG_FIX",
            )

    # Per-instrument session exclusion checks
    for inst, excluded_sessions in EXCLUDED_FROM_FITNESS.items():
        for sess in sorted(excluded_sessions):
            audit.check_passed(f"{sess} excluded from fitness for {inst}")

    # Verify MGC SINGAPORE_OPEN is still excluded (documented: 74% double-break)
    mgc_excluded = get_excluded_sessions("MGC")
    if "SINGAPORE_OPEN" in mgc_excluded:
        audit.check_passed("SINGAPORE_OPEN excluded from MGC fitness (74% double-break)")
    else:
        audit.check_failed("SINGAPORE_OPEN NOT excluded from MGC fitness")

    shelf_relation = deployable_validated_relation(con, alias="vs")
    valid_issues = 0
    for lane in lane_defs:
        r = con.execute(
            f"""
            SELECT instrument, orb_label, entry_model, filter_type
            FROM {shelf_relation}
            WHERE vs.strategy_id = ?
        """,
            [lane["strategy_id"]],
        ).fetchone()
        if r is None:
            valid_issues += 1
            audit.check_failed(f"{lane['strategy_id']}: no matching deployable validated strategy")
            continue

        lane_tuple = (
            lane["instrument"],
            lane["orb_label"],
            lane["entry_model"],
            lane["filter_type"],
        )
        shelf_tuple = tuple(r)
        if lane_tuple != shelf_tuple:
            valid_issues += 1
            audit.check_failed(
                f"{lane['strategy_id']}: profile lane metadata does not match deployable shelf row"
            )
            audit.add_finding(
                Severity.HIGH,
                "PARITY_VIOLATION",
                claimed=f"{lane['strategy_id']} lane metadata matches deployable shelf row",
                actual=f"lane={lane_tuple!r}, shelf={shelf_tuple!r}",
                evidence="trading_app/prop_profiles.py × deployable_validated_setups",
                fix_type="CONFIG_FIX",
            )

    if valid_issues == 0:
        audit.check_passed("All active profile lanes have matching deployable validated strategies")
    else:
        audit.add_finding(
            Severity.HIGH,
            "ORPHAN_STRATEGY",
            claimed="All active profile lanes resolve to deployable validated strategies",
            actual=f"{valid_issues} lane(s) missing or mismatched against deployable_validated_setups",
            evidence="trading_app/prop_profiles.py daily_lanes × deployable_validated_setups",
            fix_type="REBUILD_NEEDED",
        )


def _check_execution_wiring(audit: AuditPhase):
    """7B — Execution Engine Wiring."""
    print("\n--- 7B. Execution Engine Wiring ---")

    ee_path = PROJECT_ROOT / "trading_app" / "execution_engine.py"
    if not ee_path.exists():
        audit.check_info("execution_engine.py not found")
        return

    content = ee_path.read_text(encoding="utf-8")

    # Early exit rules
    if "EARLY_EXIT_MINUTES" in content:
        audit.check_passed("Execution engine imports EARLY_EXIT_MINUTES")
    else:
        audit.check_failed("EARLY_EXIT_MINUTES not referenced in execution_engine.py")
        audit.add_finding(
            Severity.HIGH,
            "FEATURE_PARITY_VIOLATION",
            claimed="Execution engine uses timed early exit",
            actual="EARLY_EXIT_MINUTES not found",
            evidence="grep EARLY_EXIT_MINUTES trading_app/execution_engine.py",
            fix_type="CODE_FIX",
        )

    # SINGAPORE_OPEN exclusion
    if "SINGAPORE_OPEN" in content:
        audit.check_passed("SINGAPORE_OPEN referenced in execution_engine.py (verify exclusion)")
    else:
        audit.check_info("SINGAPORE_OPEN not explicitly referenced in execution_engine.py")

    # Entry rules import
    if "entry_rules" in content or "detect_break" in content or "detect_entry" in content:
        audit.check_passed("Execution engine references entry detection logic")
    else:
        audit.check_info("No entry_rules reference in execution_engine.py")


def _check_feature_parity(audit: AuditPhase):
    """7D — Feature parity (backtest vs live)."""
    print("\n--- 7D. Feature Parity ---")

    # Check that outcome_builder and execution_engine use same entry logic source
    ob_path = PROJECT_ROOT / "trading_app" / "outcome_builder.py"
    ee_path = PROJECT_ROOT / "trading_app" / "execution_engine.py"
    er_path = PROJECT_ROOT / "trading_app" / "entry_rules.py"

    if not all(p.exists() for p in [ob_path, ee_path, er_path]):
        audit.check_info("Cannot verify parity — missing files")
        return

    ob_content = ob_path.read_text(encoding="utf-8")
    ee_content = ee_path.read_text(encoding="utf-8")

    # Both should reference entry_rules
    ob_uses_er = "entry_rules" in ob_content or "detect_break" in ob_content
    ee_uses_er = "entry_rules" in ee_content or "detect_break" in ee_content

    if ob_uses_er and ee_uses_er:
        audit.check_passed("Both outcome_builder and execution_engine reference entry_rules")
    elif ob_uses_er and not ee_uses_er:
        audit.check_info("outcome_builder uses entry_rules; execution_engine may have inline logic")
    elif not ob_uses_er:
        audit.check_failed("outcome_builder does not reference entry_rules")
        audit.add_finding(
            Severity.HIGH,
            "FEATURE_PARITY_VIOLATION",
            claimed="outcome_builder uses canonical entry_rules",
            actual="No entry_rules reference found",
            evidence="grep entry_rules trading_app/outcome_builder.py",
            fix_type="CODE_FIX",
        )

    # Both should reference cost model
    ob_cost = "cost_model" in ob_content or "friction" in ob_content or "COST_SPECS" in ob_content
    ee_cost = "cost_model" in ee_content or "friction" in ee_content or "COST_SPECS" in ee_content
    if ob_cost and ee_cost:
        audit.check_passed("Both paths reference cost model")
    else:
        audit.check_info(f"Cost model: outcome_builder={ob_cost}, execution_engine={ee_cost}")

    # Both should reference EARLY_EXIT_MINUTES
    ob_early = "EARLY_EXIT_MINUTES" in ob_content or "early_exit" in ob_content
    ee_early = "EARLY_EXIT_MINUTES" in ee_content or "early_exit" in ee_content
    if ob_early and ee_early:
        audit.check_passed("Both paths reference early exit logic")
    else:
        if not ob_early:
            audit.check_info("outcome_builder may not apply early exit (verify)")
        if not ee_early:
            audit.check_failed("execution_engine missing early exit logic")


if __name__ == "__main__":
    main()
