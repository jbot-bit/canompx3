#!/usr/bin/env python3
"""Phase 2 — Infrastructure Config Audit.

Source: SYSTEM_AUDIT.md Phase 2 (lines 122-154)

Validates asset_configs, cost_model, DST/sessions, and paths
against code and database reality.
"""

import sys
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS, DEAD_ORB_INSTRUMENTS
from pipeline.cost_model import COST_SPECS, SESSION_SLIPPAGE_MULT
from pipeline.dst import (
    DOW_ALIGNED_SESSIONS,
    DOW_MISALIGNED_SESSIONS,
    DYNAMIC_ORB_RESOLVERS,
    SESSION_CATALOG,
)
from pipeline.paths import GOLD_DB_PATH, PROJECT_ROOT
from scripts.audits import AuditPhase, Severity, db_connect


def main():
    audit = AuditPhase(phase_num=2, name="Infrastructure Config")
    audit.print_header()

    con = db_connect()
    try:
        _check_asset_configs(audit, con)
        _check_cost_model(audit)
        _check_dst_sessions(audit)
        _check_paths(audit)
    finally:
        con.close()

    audit.run_and_exit()


def _check_asset_configs(audit: AuditPhase, con):
    """2A — Asset Configs validation."""
    print("\n--- 2A. Asset Configs ---")

    for inst in ACTIVE_ORB_INSTRUMENTS:
        cfg = ASSET_CONFIGS[inst]

        # Symbol matches DB
        r = con.execute(
            "SELECT DISTINCT symbol FROM bars_1m WHERE symbol = ? LIMIT 1",
            [cfg["symbol"]],
        ).fetchone()
        if r:
            audit.check_passed(f"{inst}: symbol '{cfg['symbol']}' found in bars_1m")
        else:
            audit.check_failed(f"{inst}: symbol '{cfg['symbol']}' NOT in bars_1m")
            audit.add_finding(
                Severity.HIGH,
                "CONFIG_DRIFT",
                claimed=f"{inst} symbol '{cfg['symbol']}' in bars_1m",
                actual="No matching rows",
                evidence=f"SELECT DISTINCT symbol FROM bars_1m WHERE symbol='{cfg['symbol']}'",
                fix_type="CONFIG_FIX",
            )

        # enabled_sessions subset of SESSION_CATALOG
        enabled = set(cfg.get("enabled_sessions", []))
        catalog_keys = set(SESSION_CATALOG.keys())
        invalid_sessions = enabled - catalog_keys
        if invalid_sessions:
            audit.check_failed(f"{inst}: enabled_sessions not in SESSION_CATALOG: {sorted(invalid_sessions)}")
            audit.add_finding(
                Severity.HIGH,
                "SESSION_CONFIG_DRIFT",
                claimed=f"{inst} enabled_sessions ⊂ SESSION_CATALOG",
                actual=f"Invalid sessions: {sorted(invalid_sessions)}",
                evidence=f"asset_configs.py:{inst}[enabled_sessions] vs dst.py:SESSION_CATALOG",
                fix_type="CONFIG_FIX",
            )
        else:
            audit.check_passed(f"{inst}: {len(enabled)} enabled sessions all in SESSION_CATALOG")

    # Dead instruments should have no active validated_setups
    dead_in = ", ".join(f"'{d}'" for d in DEAD_ORB_INSTRUMENTS)
    r = con.execute(f"""
        SELECT instrument, COUNT(*) FROM validated_setups
        WHERE status='active' AND instrument IN ({dead_in})
        GROUP BY instrument
    """).fetchall()
    if r:
        for inst, n in r:
            audit.check_failed(f"Dead instrument {inst} has {n} active validated strategies")
            audit.add_finding(
                Severity.HIGH,
                "ZOMBIE_STRATEGY",
                claimed=f"{inst} is dead — no active strategies",
                actual=f"{n} active validated_setups",
                evidence=f"SELECT COUNT(*) FROM validated_setups WHERE instrument='{inst}' AND status='active'",
                fix_type="DATA_FIX",
            )
    else:
        audit.check_passed(f"Dead instruments ({', '.join(sorted(DEAD_ORB_INSTRUMENTS))}): no active strategies")


def _check_cost_model(audit: AuditPhase):
    """2B — Cost Model validation."""
    print("\n--- 2B. Cost Model ---")

    # COST_SPECS covers all active instruments
    active_set = set(ACTIVE_ORB_INSTRUMENTS)
    cost_set = set(COST_SPECS.keys())
    missing = active_set - cost_set
    if missing:
        audit.check_failed(f"Missing COST_SPECS for: {sorted(missing)}")
        audit.add_finding(
            Severity.HIGH,
            "COST_MODEL_DRIFT",
            claimed="COST_SPECS covers all active instruments",
            actual=f"Missing: {sorted(missing)}",
            evidence="pipeline/cost_model.py:COST_SPECS",
            fix_type="CONFIG_FIX",
        )
    else:
        audit.check_passed(f"COST_SPECS covers all {len(active_set)} active instruments")

    # Validate each spec
    for inst in ACTIVE_ORB_INSTRUMENTS:
        spec = COST_SPECS.get(inst)
        if not spec:
            continue
        issues = []
        if spec.point_value <= 0:
            issues.append(f"point_value={spec.point_value}")
        if spec.tick_size <= 0:
            issues.append(f"tick_size={spec.tick_size}")
        if spec.total_friction <= 0:
            issues.append(f"total_friction={spec.total_friction}")
        if issues:
            audit.check_failed(f"{inst}: invalid cost spec: {', '.join(issues)}")
            audit.add_finding(
                Severity.HIGH,
                "COST_MODEL_DRIFT",
                claimed=f"{inst} cost spec has positive values",
                actual=f"Issues: {', '.join(issues)}",
                evidence=f"pipeline/cost_model.py:COST_SPECS['{inst}']",
                fix_type="CONFIG_FIX",
            )
        else:
            audit.check_passed(
                f"{inst}: ${spec.total_friction:.2f}/RT "
                f"(pv={spec.point_value}, tick={spec.tick_size})"
            )

    # SESSION_SLIPPAGE_MULT sessions are valid
    catalog_keys = set(SESSION_CATALOG.keys())
    for inst, sessions in SESSION_SLIPPAGE_MULT.items():
        invalid = set(sessions.keys()) - catalog_keys
        if invalid:
            audit.check_failed(f"SESSION_SLIPPAGE_MULT[{inst}]: invalid sessions {sorted(invalid)}")
            audit.add_finding(
                Severity.MEDIUM,
                "SESSION_CONFIG_DRIFT",
                claimed="SESSION_SLIPPAGE_MULT sessions ⊂ SESSION_CATALOG",
                actual=f"Invalid: {sorted(invalid)}",
                evidence="pipeline/cost_model.py:SESSION_SLIPPAGE_MULT",
                fix_type="CONFIG_FIX",
            )
        else:
            audit.check_passed(f"SESSION_SLIPPAGE_MULT[{inst}]: {len(sessions)} session overrides valid")


def _check_dst_sessions(audit: AuditPhase):
    """2C — DST/Session validation."""
    print("\n--- 2C. DST & Sessions ---")

    # SESSION_CATALOG count
    n_sessions = len(SESSION_CATALOG)
    audit.check_info(f"SESSION_CATALOG has {n_sessions} entries")
    for name in sorted(SESSION_CATALOG.keys()):
        entry = SESSION_CATALOG[name]
        print(f"         {name}: type={entry.get('type', '?')}")

    # All have resolvers in DYNAMIC_ORB_RESOLVERS
    for name, entry in SESSION_CATALOG.items():
        resolver_name = entry.get("resolver")
        if resolver_name and resolver_name in DYNAMIC_ORB_RESOLVERS:
            pass  # OK
        elif resolver_name:
            audit.check_failed(f"{name}: resolver '{resolver_name}' not in DYNAMIC_ORB_RESOLVERS")
            audit.add_finding(
                Severity.HIGH,
                "SESSION_CONFIG_DRIFT",
                claimed=f"{name} resolver in DYNAMIC_ORB_RESOLVERS",
                actual=f"Resolver '{resolver_name}' missing",
                evidence="pipeline/dst.py:DYNAMIC_ORB_RESOLVERS",
                fix_type="CONFIG_FIX",
            )
        else:
            audit.check_failed(f"{name}: no resolver defined")
            audit.add_finding(
                Severity.HIGH,
                "SESSION_CONFIG_DRIFT",
                claimed=f"{name} has a resolver",
                actual="No resolver field",
                evidence=f"pipeline/dst.py:SESSION_CATALOG['{name}']",
                fix_type="CONFIG_FIX",
            )

    all_valid = all(
        entry.get("resolver") in DYNAMIC_ORB_RESOLVERS
        for entry in SESSION_CATALOG.values()
        if entry.get("resolver")
    )
    if all_valid:
        audit.check_passed("All sessions have valid resolvers")

    # Spot-check resolvers for summer/winter dates
    test_dates = [
        (date(2025, 7, 15), "summer"),
        (date(2025, 1, 15), "winter"),
    ]
    for td, season in test_dates:
        for name, resolver_fn in DYNAMIC_ORB_RESOLVERS.items():
            try:
                result = resolver_fn(td)
                if not (isinstance(result, tuple) and len(result) == 2):
                    audit.check_failed(f"{name}({td}): returned {type(result)} not tuple(h, m)")
                    audit.add_finding(
                        Severity.HIGH,
                        "SESSION_CONFIG_DRIFT",
                        claimed=f"{name} resolver returns (hour, minute)",
                        actual=f"Returned {type(result).__name__}: {result}",
                        evidence=f"DYNAMIC_ORB_RESOLVERS['{name}']({td})",
                        fix_type="CODE_FIX",
                    )
            except Exception as e:
                audit.check_failed(f"{name}({td}): raised {e}")
                audit.add_finding(
                    Severity.HIGH,
                    "SESSION_CONFIG_DRIFT",
                    claimed=f"{name} resolver works for {season}",
                    actual=f"Exception: {e}",
                    evidence=f"DYNAMIC_ORB_RESOLVERS['{name}']({td})",
                    fix_type="CODE_FIX",
                )
        audit.check_passed(f"All resolvers return valid (h, m) for {season} ({td})")

    # DOW_MISALIGNED_SESSIONS should only contain NYSE_OPEN
    if set(DOW_MISALIGNED_SESSIONS.keys()) == {"NYSE_OPEN"}:
        audit.check_passed("DOW_MISALIGNED_SESSIONS = {NYSE_OPEN: -1}")
    else:
        audit.check_failed(f"DOW_MISALIGNED_SESSIONS = {DOW_MISALIGNED_SESSIONS}")
        audit.add_finding(
            Severity.HIGH,
            "SESSION_CONFIG_DRIFT",
            claimed="Only NYSE_OPEN is DOW-misaligned",
            actual=f"Keys: {sorted(DOW_MISALIGNED_SESSIONS.keys())}",
            evidence="pipeline/dst.py:DOW_MISALIGNED_SESSIONS",
            fix_type="CODE_FIX",
        )


def _check_paths(audit: AuditPhase):
    """2D — Paths validation."""
    print("\n--- 2D. Paths ---")

    if GOLD_DB_PATH.exists():
        size_mb = GOLD_DB_PATH.stat().st_size / 1024 / 1024
        audit.check_passed(f"GOLD_DB_PATH exists: {GOLD_DB_PATH} ({size_mb:.1f} MB)")
    else:
        audit.check_failed(f"GOLD_DB_PATH does not exist: {GOLD_DB_PATH}")
        audit.add_finding(
            Severity.CRITICAL,
            "CONFIG_DRIFT",
            claimed="gold.db exists at GOLD_DB_PATH",
            actual=f"Not found: {GOLD_DB_PATH}",
            evidence=str(GOLD_DB_PATH),
            fix_type="CONFIG_FIX",
        )

    if PROJECT_ROOT.exists():
        audit.check_passed(f"PROJECT_ROOT: {PROJECT_ROOT}")
    else:
        audit.check_failed(f"PROJECT_ROOT does not exist: {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
