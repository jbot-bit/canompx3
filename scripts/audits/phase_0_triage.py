#!/usr/bin/env python3
"""Phase 0 — Triage: What changed since last audit.

Source: SYSTEM_AUDIT.md Phase 0 (lines 15-64)

Inventories existing automated checks, identifies what changed via git log,
and locates the last audit report to focus effort on changed files.
"""

import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.audits import AuditPhase, Severity, PROJECT_ROOT


def _git_log_since(paths: list[str], days: int = 30) -> list[str]:
    """Get git log --oneline for given paths since N days ago."""
    cmd = ["git", "log", "--oneline", f"--since={days} days ago", "--"] + paths
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=15)
    if r.returncode != 0:
        return []
    return [line for line in r.stdout.strip().splitlines() if line]


def _check_tool_exists(script_path: str) -> bool:
    """Verify a Python script exists and is importable."""
    return (PROJECT_ROOT / script_path).exists()


def main():
    audit = AuditPhase(phase_num=0, name="Triage")
    audit.print_header()

    # ── 0A: Inventory existing automated checks ──
    print("\n--- 0A. Existing Automated Checks ---")
    tools = {
        "pipeline/check_drift.py": "Static code/config drift detection",
        "pipeline/health_check.py": "Infrastructure health (7 checks)",
        "scripts/tools/audit_integrity.py": "Data integrity (10 enforcing + 7 informational)",
        "scripts/tools/audit_behavioral.py": "Anti-pattern scanner (6 checks)",
    }
    for script, desc in tools.items():
        if _check_tool_exists(script):
            audit.check_passed(f"{script} — {desc}")
        else:
            audit.check_failed(f"{script} — MISSING")
            audit.add_finding(
                Severity.HIGH,
                "GATE_MISSING",
                claimed=f"{script} should exist",
                actual="File not found",
                evidence=f"ls {PROJECT_ROOT / script}",
                fix_type="CODE_FIX",
            )

    # ── 0B: What changed since last audit ──
    print("\n--- 0B. Changes in Last 30 Days ---")
    change_groups = {
        "Production (pipeline/)": ["pipeline/"],
        "Production (trading_app/)": ["trading_app/"],
        "Docs (authority)": [
            "CLAUDE.md",
            "TRADING_RULES.md",
            "RESEARCH_RULES.md",
            "ROADMAP.md",
        ],
        "Tests": ["tests/"],
        "Rules & prompts": [".claude/rules/", "docs/prompts/"],
        "Infra configs": [
            "pipeline/asset_configs.py",
            "pipeline/cost_model.py",
            "pipeline/dst.py",
        ],
        "Scripts": ["scripts/"],
    }

    total_changes = 0
    for group_name, paths in change_groups.items():
        commits = _git_log_since(paths)
        count = len(commits)
        total_changes += count
        if count > 0:
            audit.check_info(f"{group_name}: {count} commits")
            for c in commits[:5]:  # Show first 5
                print(f"         {c}")
            if count > 5:
                print(f"         ... and {count - 5} more")
        else:
            audit.check_info(f"{group_name}: no changes")

    print(f"\n  Total commits touching production/docs/tests: {total_changes}")
    if total_changes == 0:
        print("  → No changes detected. Quick audit may suffice.")
    else:
        print("  → Focus 80% of manual effort on changed files.")

    # ── 0C: Locate last audit ──
    print("\n--- 0C. Last Audit ---")
    audit_pattern = PROJECT_ROOT / "research" / "output"
    audit_files = sorted(audit_pattern.glob("HIGH_LEVEL_AUDIT_*.md"), reverse=True)
    if audit_files:
        latest = audit_files[0]
        audit.check_info(f"Last audit: {latest.name}")
        # Extract date from filename (HIGH_LEVEL_AUDIT_YYYY-MM-DD.md)
        parts = latest.stem.split("_")
        if len(parts) >= 4:
            date_str = "-".join(parts[3:])
            print(f"         Date: {date_str}")
            print(f"         Path: {latest.relative_to(PROJECT_ROOT)}")
        if len(audit_files) > 1:
            print(f"         ({len(audit_files)} total audit reports found)")
    else:
        audit.check_info("No previous audit reports found in research/output/")
        print("         This may be the first formal audit.")

    audit.run_and_exit()


if __name__ == "__main__":
    main()
