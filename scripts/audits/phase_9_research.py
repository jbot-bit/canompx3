#!/usr/bin/env python3
"""Phase 9 — Research & Script Hygiene.

Source: SYSTEM_AUDIT.md Phase 9 (lines 376-407)

NO-GO enforcement, data snooping quarantine, pending item staleness.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.paths import PROJECT_ROOT
from scripts.audits import AuditPhase, Severity

# Production directories to scan for NO-GO zombies
PROD_DIRS = [PROJECT_ROOT / "pipeline", PROJECT_ROOT / "trading_app"]


def main():
    audit = AuditPhase(phase_num=9, name="Research & Script Hygiene")
    audit.print_header()

    _check_nogo_enforcement(audit)
    _check_data_snooping(audit)
    _check_research_scripts(audit)

    audit.run_and_exit()


def _read_nogo_items(audit: AuditPhase) -> list[str]:
    """Extract NO-GO items from TRADING_RULES.md."""
    tr_path = PROJECT_ROOT / "TRADING_RULES.md"
    if not tr_path.exists():
        audit.check_info("TRADING_RULES.md not found — cannot check NO-GOs")
        return []

    content = tr_path.read_text(encoding="utf-8")

    # Look for NO-GO table entries
    # Pattern: lines in NO-GO section with feature names
    nogo_items = []

    # Find the NO-GO section
    nogo_start = content.find("NO-GO")
    if nogo_start == -1:
        nogo_start = content.find("No-Go")
    if nogo_start == -1:
        audit.check_info("No NO-GO section found in TRADING_RULES.md")
        return []

    # Extract items from the table after NO-GO heading
    nogo_section = content[nogo_start : nogo_start + 5000]  # Read ahead
    # Look for table rows with | delimiters
    for line in nogo_section.splitlines():
        if "|" in line and "---" not in line and "NO-GO" not in line.upper().split("|")[0]:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if cells:
                # First cell is usually the feature name
                item = cells[0].strip("`*_ ")
                if item and len(item) > 2 and not item.startswith("Feature"):
                    nogo_items.append(item)

    return nogo_items


def _check_nogo_enforcement(audit: AuditPhase):
    """9C — NO-GO enforcement in production code."""
    print("\n--- 9C. NO-GO Enforcement ---")

    nogo_items = _read_nogo_items(audit)
    if not nogo_items:
        audit.check_info("No NO-GO items to check")
        return

    audit.check_info(f"Found {len(nogo_items)} NO-GO items to check")

    zombies = []
    for item in nogo_items:
        # Sanitize for grep — use word-like pattern
        search_term = item.lower().replace(" ", "_").replace("-", "_")
        if len(search_term) < 4:
            continue  # Too short to be meaningful

        for d in PROD_DIRS:
            if not d.exists():
                continue
            for py in d.rglob("*.py"):
                try:
                    content = py.read_text(encoding="utf-8").lower()
                    if search_term in content:
                        rel = py.relative_to(PROJECT_ROOT)
                        # Exclude if it's in a comment or docstring warning about the NO-GO
                        # Simple heuristic: check if the line contains "no-go" or "dead" or "removed"
                        lines = content.splitlines()
                        for i, line in enumerate(lines):
                            if search_term in line:
                                if any(w in line for w in ["#", "no-go", "dead", "removed", "purged", "deprecated"]):
                                    continue  # It's a warning comment, OK
                                zombies.append((item, str(rel), i + 1))
                except (UnicodeDecodeError, PermissionError):
                    continue

    if zombies:
        audit.check_failed(f"{len(zombies)} potential NO-GO zombie(s) in production")
        for item, path, line in zombies[:10]:
            print(f"         NO_GO_ZOMBIE: '{item}' in {path}:{line}")
        audit.add_finding(
            Severity.HIGH,
            "NO_GO_ZOMBIE",
            claimed="NO-GO features not in production code",
            actual=f"{len(zombies)} potential zombie(s)",
            evidence="grep NO-GO items in pipeline/ trading_app/",
            fix_type="CODE_FIX",
        )
    else:
        audit.check_passed("No NO-GO zombies detected in production code")


def _check_data_snooping(audit: AuditPhase):
    """9D — Data snooping quarantine."""
    print("\n--- 9D. Data Snooping Quarantine ---")

    # strategy_discovery.py should NOT import from strategy_validator walk-forward logic
    discovery_path = PROJECT_ROOT / "trading_app" / "strategy_discovery.py"
    if discovery_path.exists():
        content = discovery_path.read_text(encoding="utf-8")

        # Check for imports from strategy_validator
        if "from trading_app.strategy_validator" in content or "import strategy_validator" in content:
            audit.check_failed("strategy_discovery.py imports from strategy_validator")
            audit.add_finding(
                Severity.CRITICAL,
                "DATA_LEAK_RISK",
                claimed="Discovery does not import from validator",
                actual="Import found — potential data leak",
                evidence="trading_app/strategy_discovery.py imports strategy_validator",
                fix_type="CODE_FIX",
            )
        else:
            audit.check_passed("strategy_discovery.py does not import strategy_validator")

        # Check for walk-forward boundary references
        wf_patterns = ["walk_forward", "holdout", "oos_start", "oos_end", "out_of_sample"]
        wf_found = [p for p in wf_patterns if p in content.lower()]
        if wf_found:
            audit.check_info(f"strategy_discovery.py references: {wf_found} (verify not accessing OOS data)")
        else:
            audit.check_passed("No walk-forward boundary references in discovery")
    else:
        audit.check_info("strategy_discovery.py not found")

    # strategy_validator.py walk-forward splits should be deterministic
    validator_path = PROJECT_ROOT / "trading_app" / "strategy_validator.py"
    if validator_path.exists():
        content = validator_path.read_text(encoding="utf-8")
        if "random" in content.lower() and "split" in content.lower():
            audit.check_info("strategy_validator.py contains 'random' + 'split' — verify walk-forward is deterministic")
        else:
            audit.check_passed("No random split logic detected in validator")
    else:
        audit.check_info("strategy_validator.py not found")


def _check_research_scripts(audit: AuditPhase):
    """9A — Research script inventory."""
    print("\n--- 9A. Research Script Inventory ---")

    research_dir = PROJECT_ROOT / "research"
    if not research_dir.exists():
        audit.check_info("research/ directory not found")
        return

    scripts = list(research_dir.glob("*.py"))
    outputs = list((research_dir / "output").glob("*")) if (research_dir / "output").exists() else []

    audit.check_info(f"research/: {len(scripts)} scripts, {len(outputs)} output files")

    # Check for stale references in script names
    stale_keywords = ["e0", "0900", "1800", "0030", "2300", "nodbl"]
    stale_scripts = []
    for s in scripts:
        name_lower = s.stem.lower()
        for kw in stale_keywords:
            if kw in name_lower:
                stale_scripts.append((s.name, kw))
                break

    if stale_scripts:
        audit.check_info(f"{len(stale_scripts)} research script(s) with potentially stale names")
        for name, kw in stale_scripts[:5]:
            print(f"         {name} (contains '{kw}')")
    else:
        audit.check_passed("No research scripts with stale naming patterns")


if __name__ == "__main__":
    main()
