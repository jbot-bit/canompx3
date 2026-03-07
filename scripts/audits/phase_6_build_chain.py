#!/usr/bin/env python3
"""Phase 6 — Build Chain Staleness.

Source: SYSTEM_AUDIT.md Phase 6 (lines 289-313)

Per-instrument pipeline status, code-change-triggered rebuilds,
and validator flag verification.
"""

import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, ASSET_CONFIGS
from pipeline.paths import PROJECT_ROOT
from scripts.audits import AuditPhase, Severity, db_connect


def _git_log_file(filepath: str, days: int = 60) -> list[str]:
    """Get git log for a specific file."""
    r = subprocess.run(
        ["git", "log", "--oneline", f"--since={days} days ago", "--", filepath],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=15,
    )
    return [line for line in (r.stdout or "").strip().splitlines() if line]


def main():
    audit = AuditPhase(phase_num=6, name="Build Chain Staleness")
    audit.print_header()

    con = db_connect()
    try:
        _check_per_instrument_status(audit, con)
        _check_code_change_rebuilds(audit)
    finally:
        con.close()

    audit.run_and_exit()


def _check_per_instrument_status(audit: AuditPhase, con):
    """6A — Per-instrument build chain status."""
    print("\n--- 6A. Per-Instrument Build Chain ---")

    # Table → date column mapping
    pipeline_tables = {
        "bars_1m": "ts_utc",
        "bars_5m": "ts_utc",
        "daily_features": "trading_day",
        "orb_outcomes": "trading_day",
    }
    # These use instrument column, not symbol
    strategy_tables = {
        "experimental_strategies": "instrument",
        "validated_setups": "instrument",
    }

    print(f"\n  {'Instrument':<12} {'bars_1m':<12} {'bars_5m':<12} {'features':<12} {'outcomes':<12} {'experi':<12} {'valid':<12} {'Status'}")
    print("  " + "-" * 96)

    for inst in ACTIVE_ORB_INSTRUMENTS:
        symbol = ASSET_CONFIGS[inst]["symbol"]
        dates = {}

        # Pipeline tables (use symbol)
        for table, col in pipeline_tables.items():
            if col == "ts_utc":
                r = con.execute(f"SELECT MAX({col})::DATE FROM {table} WHERE symbol = ?", [symbol]).fetchone()
            else:
                r = con.execute(f"SELECT MAX({col}) FROM {table} WHERE symbol = ?", [symbol]).fetchone()
            dates[table] = str(r[0]) if r and r[0] else "N/A"

        # Strategy tables (use instrument)
        for table, col_name in strategy_tables.items():
            try:
                r = con.execute(f"""
                    SELECT MAX(trading_day) FROM {table}
                    WHERE {col_name} = ?
                """, [inst]).fetchone()
                dates[table] = str(r[0]) if r and r[0] else "N/A"
            except Exception:
                # Some tables might not have trading_day
                dates[table] = "N/A"

        # Determine status
        valid_dates = [d for d in dates.values() if d != "N/A"]
        if valid_dates and len(set(valid_dates)) <= 2:  # Allow small variance
            status = "UP_TO_DATE"
        elif valid_dates:
            status = "BUILD_CHAIN_GAP"
        else:
            status = "NO_DATA"

        print(
            f"  {inst:<12} "
            f"{dates.get('bars_1m', 'N/A'):<12} "
            f"{dates.get('bars_5m', 'N/A'):<12} "
            f"{dates.get('daily_features', 'N/A'):<12} "
            f"{dates.get('orb_outcomes', 'N/A'):<12} "
            f"{dates.get('experimental_strategies', 'N/A'):<12} "
            f"{dates.get('validated_setups', 'N/A'):<12} "
            f"{status}"
        )

        if status == "BUILD_CHAIN_GAP":
            audit.add_finding(
                Severity.HIGH,
                "BUILD_CHAIN_GAP",
                claimed=f"{inst} build chain up to date",
                actual=f"Dates vary: {dates}",
                evidence=f"MAX(trading_day/ts_utc) per table for {inst}",
                fix_type="REBUILD_NEEDED",
            )
        elif status == "UP_TO_DATE":
            audit.check_passed(f"{inst}: build chain up to date")
        elif status == "NO_DATA":
            audit.check_failed(f"{inst}: no data in pipeline tables")


def _check_code_change_rebuilds(audit: AuditPhase):
    """6B — Code changes that should trigger rebuilds."""
    print("\n--- 6B. Code-Change-Triggered Rebuilds ---")

    critical_files = {
        "trading_app/outcome_builder.py": "outcomes",
        "trading_app/strategy_discovery.py": "discovery",
        "trading_app/strategy_validator.py": "validation",
        "scripts/tools/build_edge_families.py": "edge families",
        "pipeline/build_daily_features.py": "daily features (cascades to everything)",
        "trading_app/entry_rules.py": "outcomes (entry logic changed)",
    }

    changes_found = False
    for filepath, affects in critical_files.items():
        commits = _git_log_file(filepath, days=60)
        if commits:
            changes_found = True
            audit.check_info(f"{filepath}: {len(commits)} commit(s) in last 60 days → may need {affects} rebuild")
            for c in commits[:3]:
                print(f"         {c}")
            audit.add_finding(
                Severity.MEDIUM,
                "REBUILD_NEEDED",
                claimed=f"Rebuild triggered after {filepath} changes",
                actual=f"{len(commits)} commit(s) in last 60 days",
                evidence=f"git log --since='60 days ago' -- {filepath}",
                fix_type="REBUILD_NEEDED",
            )
        else:
            audit.check_passed(f"{filepath}: no changes in 60 days")

    if not changes_found:
        print("\n  No critical pipeline files changed in last 60 days.")


if __name__ == "__main__":
    main()
