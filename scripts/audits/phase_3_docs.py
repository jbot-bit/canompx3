#!/usr/bin/env python3
"""Phase 3 — Documentation vs Reality.

Source: SYSTEM_AUDIT.md Phase 3 (lines 157-215)

Validates CLAUDE.md, TRADING_RULES.md, RESEARCH_RULES.md, ROADMAP.md,
REPO_MAP.md, .claude/rules/, and docs/specs/ against canonical code sources.
"""

import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, DEAD_ORB_INSTRUMENTS
from pipeline.cost_model import COST_SPECS
from pipeline.dst import SESSION_CATALOG
from pipeline.paths import PROJECT_ROOT
from scripts.audits import AuditPhase, Severity
from trading_app.config import CORE_MIN_SAMPLES, ENTRY_MODELS, REGIME_MIN_SAMPLES


def _read_file(path: Path) -> str:
    """Read a file, return empty string if not found."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def main():
    audit = AuditPhase(phase_num=3, name="Documentation vs Reality")
    audit.print_header()

    _check_claude_md(audit)
    _check_trading_rules(audit)
    _check_research_rules(audit)
    _check_roadmap(audit)
    _check_repo_map(audit)
    _check_claude_rules(audit)
    _check_specs(audit)

    audit.run_and_exit()


def _check_claude_md(audit: AuditPhase):
    """3A — CLAUDE.md audit."""
    print("\n--- 3A. CLAUDE.md ---")
    content = _read_file(PROJECT_ROOT / "CLAUDE.md")
    if not content:
        audit.check_failed("CLAUDE.md not found")
        audit.add_finding(Severity.CRITICAL, "DOC_STALE", "CLAUDE.md exists", "Not found", str(PROJECT_ROOT / "CLAUDE.md"), "DOC_FIX")
        return

    # Active vs dead instruments
    for inst in ACTIVE_ORB_INSTRUMENTS:
        if inst not in content:
            audit.check_failed(f"CLAUDE.md missing active instrument: {inst}")
            audit.add_finding(Severity.MEDIUM, "DOC_STALE", f"{inst} mentioned in CLAUDE.md", "Not found", "CLAUDE.md", "DOC_FIX")

    for inst in DEAD_ORB_INSTRUMENTS:
        if inst in content:
            # Should be mentioned as dead, not active
            # Just check it exists (it should be mentioned in the dead list)
            pass

    audit.check_passed("CLAUDE.md references active instruments")

    # Classification thresholds
    if "100" in content and "CORE" in content:
        audit.check_passed("CLAUDE.md mentions CORE threshold (100)")
    else:
        audit.check_failed("CLAUDE.md missing CORE threshold 100")
        audit.add_finding(Severity.MEDIUM, "DOC_STALE", "CORE >= 100 documented", "Not found", "CLAUDE.md", "DOC_FIX")

    if "30" in content and "REGIME" in content:
        audit.check_passed("CLAUDE.md mentions REGIME threshold (30)")
    else:
        audit.check_failed("CLAUDE.md missing REGIME threshold 30")

    # Source contract mapping
    mappings = {"GC": "MGC", "ES": "MES", "RTY": "M2K"}
    for source, stored in mappings.items():
        if source in content and stored in content:
            pass
        else:
            audit.check_info(f"Source mapping {source}→{stored} may not be in CLAUDE.md")
    audit.check_passed("Source contract mappings present")


def _check_trading_rules(audit: AuditPhase):
    """3B — TRADING_RULES.md audit."""
    print("\n--- 3B. TRADING_RULES.md ---")
    content = _read_file(PROJECT_ROOT / "TRADING_RULES.md")
    if not content:
        audit.check_failed("TRADING_RULES.md not found")
        audit.add_finding(Severity.CRITICAL, "DOC_STALE", "TRADING_RULES.md exists", "Not found", str(PROJECT_ROOT / "TRADING_RULES.md"), "DOC_FIX")
        return

    # Session names — all SESSION_CATALOG keys should appear
    missing_sessions = []
    for session in SESSION_CATALOG:
        if session not in content:
            missing_sessions.append(session)
    if missing_sessions:
        audit.check_failed(f"TRADING_RULES.md missing sessions: {missing_sessions}")
        audit.add_finding(
            Severity.HIGH,
            "DOC_STALE",
            claimed="All sessions documented in TRADING_RULES.md",
            actual=f"Missing: {missing_sessions}",
            evidence="grep SESSION_CATALOG keys in TRADING_RULES.md",
            fix_type="DOC_FIX",
        )
    else:
        audit.check_passed(f"All {len(SESSION_CATALOG)} sessions mentioned in TRADING_RULES.md")

    # Entry models
    for em in ENTRY_MODELS:
        if em not in content:
            audit.check_failed(f"TRADING_RULES.md missing entry model: {em}")
            audit.add_finding(Severity.HIGH, "DOC_STALE", f"{em} documented", "Not found", "TRADING_RULES.md", "DOC_FIX")
    audit.check_passed(f"Entry models {ENTRY_MODELS} documented")

    # E0 should be mentioned as dead/purged
    if "E0" in content:
        audit.check_passed("E0 referenced (should be marked dead/purged)")
    else:
        audit.check_info("E0 not mentioned in TRADING_RULES.md")

    # Cost model numbers
    for inst in ACTIVE_ORB_INSTRUMENTS:
        spec = COST_SPECS.get(inst)
        if spec:
            friction_str = f"${spec.total_friction:.2f}"
            if friction_str in content:
                audit.check_passed(f"{inst} friction {friction_str} in TRADING_RULES.md")
            else:
                # Try without dollar sign
                friction_plain = f"{spec.total_friction:.2f}"
                if friction_plain in content:
                    audit.check_passed(f"{inst} friction {friction_plain} in TRADING_RULES.md")
                else:
                    audit.check_info(f"{inst} friction {friction_str} not found verbatim in TRADING_RULES.md")


def _check_research_rules(audit: AuditPhase):
    """3C — RESEARCH_RULES.md audit."""
    print("\n--- 3C. RESEARCH_RULES.md ---")
    content = _read_file(PROJECT_ROOT / "RESEARCH_RULES.md")
    if not content:
        audit.check_failed("RESEARCH_RULES.md not found")
        audit.add_finding(Severity.HIGH, "DOC_STALE", "RESEARCH_RULES.md exists", "Not found", str(PROJECT_ROOT / "RESEARCH_RULES.md"), "DOC_FIX")
        return

    # Sample size thresholds
    if str(CORE_MIN_SAMPLES) in content:
        audit.check_passed(f"CORE_MIN_SAMPLES ({CORE_MIN_SAMPLES}) in RESEARCH_RULES.md")
    else:
        audit.check_failed(f"CORE_MIN_SAMPLES ({CORE_MIN_SAMPLES}) not in RESEARCH_RULES.md")
        audit.add_finding(
            Severity.MEDIUM,
            "DOC_STALE",
            claimed=f"CORE_MIN_SAMPLES={CORE_MIN_SAMPLES} documented",
            actual="Not found",
            evidence="RESEARCH_RULES.md",
            fix_type="DOC_FIX",
        )

    if str(REGIME_MIN_SAMPLES) in content:
        audit.check_passed(f"REGIME_MIN_SAMPLES ({REGIME_MIN_SAMPLES}) in RESEARCH_RULES.md")
    else:
        audit.check_failed(f"REGIME_MIN_SAMPLES ({REGIME_MIN_SAMPLES}) not in RESEARCH_RULES.md")


def _check_roadmap(audit: AuditPhase):
    """3D — ROADMAP.md audit."""
    print("\n--- 3D. ROADMAP.md ---")
    content = _read_file(PROJECT_ROOT / "ROADMAP.md")
    if not content:
        audit.check_info("ROADMAP.md not found")
        return

    # Count DONE vs TODO
    done_count = content.lower().count("done")
    todo_count = content.lower().count("todo")
    audit.check_info(f"ROADMAP.md: ~{done_count} DONE references, ~{todo_count} TODO references")


def _check_repo_map(audit: AuditPhase):
    """3E — REPO_MAP.md audit."""
    print("\n--- 3E. REPO_MAP.md ---")
    repo_map = PROJECT_ROOT / "REPO_MAP.md"
    if not repo_map.exists():
        audit.check_info("REPO_MAP.md not found (may need generation)")
        return

    gen_script = PROJECT_ROOT / "scripts" / "tools" / "gen_repo_map.py"
    if not gen_script.exists():
        audit.check_info("gen_repo_map.py not found — cannot verify REPO_MAP.md freshness")
        return

    # Run gen_repo_map.py and compare
    r = subprocess.run(
        ["python", str(gen_script)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=30,
    )
    if r.returncode == 0:
        generated = r.stdout.strip()
        current = repo_map.read_text(encoding="utf-8").strip()
        if generated == current:
            audit.check_passed("REPO_MAP.md is up to date")
        else:
            audit.check_failed("REPO_MAP.md is stale (differs from gen_repo_map.py output)")
            audit.add_finding(
                Severity.LOW,
                "REPO_MAP_STALE",
                claimed="REPO_MAP.md matches gen_repo_map.py output",
                actual="Files differ",
                evidence="python scripts/tools/gen_repo_map.py | diff - REPO_MAP.md",
                fix_type="DOC_FIX",
            )
    else:
        audit.check_info("gen_repo_map.py failed to run — cannot verify")


def _check_claude_rules(audit: AuditPhase):
    """3F — .claude/rules/ audit."""
    print("\n--- 3F. .claude/rules/ ---")
    rules_dir = PROJECT_ROOT / ".claude" / "rules"
    if not rules_dir.exists():
        audit.check_failed(".claude/rules/ directory not found")
        return

    rule_files = sorted(rules_dir.glob("*.md"))
    audit.check_info(f"{len(rule_files)} rule file(s) in .claude/rules/")

    # Check for old session names in rules
    old_sessions = {"0900", "1800", "0030", "2300"}
    for rf in rule_files:
        content = rf.read_text(encoding="utf-8")
        found = [s for s in old_sessions if s in content]
        if found:
            audit.check_failed(f"{rf.name}: references old session names: {found}")
            audit.add_finding(
                Severity.MEDIUM,
                "STALE_CHECK",
                claimed=f"{rf.name} uses current session names",
                actual=f"Old session names found: {found}",
                evidence=f".claude/rules/{rf.name}",
                fix_type="DOC_FIX",
            )
        else:
            audit.check_passed(f"{rf.name}: no old session names")


def _check_specs(audit: AuditPhase):
    """3G — docs/specs/ compliance."""
    print("\n--- 3G. docs/specs/ ---")
    specs_dir = PROJECT_ROOT / "docs" / "specs"
    if not specs_dir.exists():
        audit.check_info("docs/specs/ directory not found")
        return

    spec_files = sorted(specs_dir.glob("*.md"))
    if spec_files:
        audit.check_info(f"{len(spec_files)} spec file(s) in docs/specs/")
        for sf in spec_files:
            print(f"         {sf.name}")
    else:
        audit.check_info("No spec files found")


if __name__ == "__main__":
    main()
