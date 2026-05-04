"""Rebuild import centrality index for Ralph Loop targeting.

Walks all production Python files, counts how many other production files
import each one, assigns tier (critical/high/medium/low), writes JSON.

Usage: python scripts/tools/ralph_build_centrality.py
"""

import json
import re
import subprocess
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_PATH = PROJECT_ROOT / "docs" / "ralph-loop" / "import_centrality.json"
PROD_DIRS = ("pipeline", "trading_app", "scripts")


def _find_production_files() -> list[Path]:
    """Find all .py files in production directories."""
    files = []
    for d in PROD_DIRS:
        root = PROJECT_ROOT / d
        if root.exists():
            files.extend(root.rglob("*.py"))
    return sorted(files)


def _module_name(path: Path) -> str:
    """Convert file path to dotted module name relative to project root."""
    rel = path.relative_to(PROJECT_ROOT)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def _count_importers(target: Path) -> int:
    """Count how many production files import from the target module."""
    mod = _module_name(target)
    if not mod:
        return 0

    # Build patterns: "from module import", "import module"
    # For nested modules, also match parent imports
    patterns = [
        rf"from\s+{re.escape(mod)}\s+import",
        rf"import\s+{re.escape(mod)}\b",
    ]
    # Also match "from parent.module import" for __init__.py packages
    rel = str(target.relative_to(PROJECT_ROOT)).replace("\\", "/")

    count = 0
    combined = "|".join(patterns)
    try:
        result = subprocess.run(
            ["grep", "-rlE", combined, "pipeline/", "trading_app/", "scripts/", "--include=*.py"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        # Exclude self
        target_rel = rel
        matches = [line for line in result.stdout.strip().split("\n") if line and line.replace("\\", "/") != target_rel]
        count = len(matches)
    except Exception:
        count = 0

    return count


def _assign_tier(importers: int) -> str:
    if importers >= 10:
        return "critical"
    if importers >= 5:
        return "high"
    if importers >= 2:
        return "medium"
    return "low"


def main() -> None:
    all_files = _find_production_files()
    print(f"Found {len(all_files)} production files")

    files_data: dict[str, dict] = {}
    for i, f in enumerate(all_files):
        rel = str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")
        importers = _count_importers(f)
        # Count imports FROM this file (how many modules it depends on)
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            imports = len(re.findall(r"^(?:from|import)\s+\S+", text, re.MULTILINE))
        except Exception:
            imports = 0

        tier = _assign_tier(importers)
        files_data[rel] = {
            "importers": importers,
            "imports": imports,
            "centrality": tier,
        }
        if (i + 1) % 50 == 0 or importers >= 5:
            print(f"  [{i + 1}/{len(all_files)}] {rel}: {importers} importers ({tier})")

    # Summary
    tiers = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for info in files_data.values():
        tiers[info["centrality"]] += 1
    print(f"\nTiers: {tiers}")

    output = {
        "generated": date.today().isoformat(),
        "files": dict(sorted(files_data.items())),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(files_data)} files)")


if __name__ == "__main__":
    main()
