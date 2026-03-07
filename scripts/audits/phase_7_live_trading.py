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
from trading_app.config import ALL_FILTERS, ENTRY_MODELS, EXCLUDED_FROM_FITNESS
from trading_app.live_config import LIVE_PORTFOLIO


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
    """7A — Live Config Coherence."""
    print("\n--- 7A. Live Config Coherence ---")

    if not LIVE_PORTFOLIO:
        audit.check_info("LIVE_PORTFOLIO is empty")
        return

    audit.check_info(f"LIVE_PORTFOLIO has {len(LIVE_PORTFOLIO)} specs")

    # Tier breakdown
    tiers = {}
    for spec in LIVE_PORTFOLIO:
        tiers.setdefault(spec.tier, []).append(spec)
    for tier, specs in sorted(tiers.items()):
        print(f"         {tier}: {len(specs)} specs")

    # Validate each spec
    catalog_keys = set(SESSION_CATALOG.keys())
    filter_keys = set(ALL_FILTERS.keys())
    entry_set = set(ENTRY_MODELS)

    for spec in LIVE_PORTFOLIO:
        issues = []

        if spec.orb_label not in catalog_keys:
            issues.append(f"orb_label '{spec.orb_label}' not in SESSION_CATALOG")
        if spec.entry_model not in entry_set:
            issues.append(f"entry_model '{spec.entry_model}' not in ENTRY_MODELS")
        if spec.filter_type not in filter_keys:
            issues.append(f"filter_type '{spec.filter_type}' not in ALL_FILTERS")

        if issues:
            audit.check_failed(f"{spec.family_id}: {'; '.join(issues)}")
            audit.add_finding(
                Severity.HIGH,
                "CONFIG_DRIFT",
                claimed=f"{spec.family_id} references valid config values",
                actual="; ".join(issues),
                evidence="trading_app/live_config.py:LIVE_PORTFOLIO",
                fix_type="CONFIG_FIX",
            )

    # SINGAPORE_OPEN exclusion
    singapore_specs = [s for s in LIVE_PORTFOLIO if s.orb_label == "SINGAPORE_OPEN"]
    if singapore_specs:
        audit.check_failed(f"SINGAPORE_OPEN in LIVE_PORTFOLIO ({len(singapore_specs)} specs)")
        audit.add_finding(
            Severity.HIGH,
            "CONFIG_DRIFT",
            claimed="SINGAPORE_OPEN excluded from live portfolio",
            actual=f"{len(singapore_specs)} SINGAPORE_OPEN specs found",
            evidence="trading_app/live_config.py:LIVE_PORTFOLIO",
            fix_type="CONFIG_FIX",
        )
    else:
        audit.check_passed("SINGAPORE_OPEN excluded from LIVE_PORTFOLIO")

    # Check EXCLUDED_FROM_FITNESS
    if "SINGAPORE_OPEN" in EXCLUDED_FROM_FITNESS:
        audit.check_passed("SINGAPORE_OPEN in EXCLUDED_FROM_FITNESS")
    else:
        audit.check_failed("SINGAPORE_OPEN NOT in EXCLUDED_FROM_FITNESS")

    # Verify families exist in validated_setups
    valid_issues = 0
    for spec in LIVE_PORTFOLIO:
        r = con.execute("""
            SELECT COUNT(*) FROM validated_setups
            WHERE orb_label = ? AND entry_model = ? AND filter_type = ? AND status = 'active'
        """, [spec.orb_label, spec.entry_model, spec.filter_type]).fetchone()[0]
        if r == 0:
            valid_issues += 1
            audit.check_failed(f"{spec.family_id}: no matching active validated_setups")

    if valid_issues == 0:
        audit.check_passed("All LIVE_PORTFOLIO specs have matching validated_setups")
    else:
        audit.add_finding(
            Severity.HIGH,
            "ORPHAN_STRATEGY",
            claimed="All live specs have validated strategies",
            actual=f"{valid_issues} spec(s) without matching validated_setups",
            evidence="SELECT FROM validated_setups for each LIVE_PORTFOLIO spec",
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
