"""Build an internal import centrality graph for the project.

Scans .py files in pipeline/, trading_app/, scripts/tools/, scripts/reports/.
Parses import statements, builds a directed graph of internal imports,
and outputs centrality scores to docs/ralph-loop/import_centrality.json.

Usage:
    python scripts/tools/import_graph.py
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCAN_DIRS = [
    PROJECT_ROOT / "pipeline",
    PROJECT_ROOT / "trading_app",
    PROJECT_ROOT / "scripts" / "tools",
    PROJECT_ROOT / "scripts" / "reports",
]

# Top-level packages that count as internal
INTERNAL_PACKAGES = {"pipeline", "trading_app", "scripts"}


def collect_py_files() -> list[Path]:
    """Collect all .py files from scan directories."""
    files = []
    for d in SCAN_DIRS:
        if d.exists():
            files.extend(sorted(d.rglob("*.py")))
    return files


def file_to_module(path: Path) -> str:
    """Convert a file path to a dotted module path relative to project root.

    e.g. pipeline/paths.py -> pipeline.paths
         scripts/tools/import_graph.py -> scripts.tools.import_graph
    """
    rel = path.relative_to(PROJECT_ROOT)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def module_to_relpath(module: str) -> str:
    """Convert a dotted module to a relative file path for display.

    e.g. pipeline.paths -> pipeline/paths.py
    """
    parts = module.split(".")
    # Could be a package (__init__.py) or a module (.py)
    candidate = PROJECT_ROOT / Path(*parts)
    if candidate.is_dir() and (candidate / "__init__.py").exists():
        return "/".join(parts) + "/__init__.py"
    return "/".join(parts) + ".py"


def resolve_import_to_module(import_name: str) -> str | None:
    """Resolve an import name to an internal module, or None if external.

    Handles cases like:
      - pipeline.paths -> pipeline.paths (if pipeline/paths.py exists)
      - pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS -> pipeline.asset_configs
      - trading_app.config -> trading_app.config
    """
    parts = import_name.split(".")
    if parts[0] not in INTERNAL_PACKAGES:
        return None

    # Walk from longest to shortest to find the actual module/package
    for i in range(len(parts), 0, -1):
        candidate = ".".join(parts[:i])
        mod_path = PROJECT_ROOT / Path(*parts[:i])

        # Check if it's a .py file
        if mod_path.with_suffix(".py").exists():
            return candidate

        # Check if it's a package
        if mod_path.is_dir() and (mod_path / "__init__.py").exists():
            return candidate

    return None


def extract_imports(filepath: Path) -> set[str]:
    """Parse a .py file and return the set of internal modules it imports."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    internal_imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = resolve_import_to_module(alias.name)
                if mod:
                    internal_imports.add(mod)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = resolve_import_to_module(node.module)
                if mod:
                    internal_imports.add(mod)

    return internal_imports


def classify_tier(importer_count: int) -> str:
    if importer_count >= 10:
        return "critical"
    if importer_count >= 5:
        return "high"
    if importer_count >= 2:
        return "medium"
    return "low"


def main() -> None:
    py_files = collect_py_files()

    # module_name -> set of modules that import it
    imported_by: dict[str, set[str]] = defaultdict(set)
    # module_name -> set of modules it imports
    imports_of: dict[str, set[str]] = defaultdict(set)

    # Map each file to its module name
    file_modules: dict[str, Path] = {}
    for f in py_files:
        mod = file_to_module(f)
        file_modules[mod] = f

    # Build the graph
    for mod, filepath in file_modules.items():
        deps = extract_imports(filepath)
        # Don't count self-imports
        deps.discard(mod)
        imports_of[mod] = deps
        for dep in deps:
            imported_by[dep].add(mod)

    # Build the output
    files_output: dict[str, dict] = {}
    tiers: dict[str, list[str]] = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": [],
    }

    # Include all modules that are either scanned or imported by scanned files
    all_modules = set(file_modules.keys()) | set(imported_by.keys())

    for mod in sorted(all_modules):
        relpath = module_to_relpath(mod)
        importer_count = len(imported_by.get(mod, set()))
        import_count = len(imports_of.get(mod, set()))
        tier = classify_tier(importer_count)

        files_output[relpath] = {
            "importers": importer_count,
            "imports": import_count,
            "centrality": tier,
        }
        tiers[tier].append(relpath)

    # Sort tier lists by importer count descending
    for tier_name in tiers:
        tiers[tier_name].sort(
            key=lambda p: files_output[p]["importers"], reverse=True
        )

    result = {
        "generated": date.today().isoformat(),
        "files": files_output,
        "tiers": tiers,
    }

    out_path = PROJECT_ROOT / "docs" / "ralph-loop" / "import_centrality.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # Print summary
    print(f"Scanned {len(file_modules)} files, found {len(all_modules)} modules")
    print(f"Output: {out_path.relative_to(PROJECT_ROOT)}")
    print()
    for tier_name in ["critical", "high", "medium", "low"]:
        entries = tiers[tier_name]
        print(f"  {tier_name.upper()} ({len(entries)}):")
        for relpath in entries:
            info = files_output[relpath]
            print(f"    {relpath}: {info['importers']} importers, {info['imports']} imports")
    print()
    print(f"Total: {sum(len(v) for v in tiers.values())} modules")


if __name__ == "__main__":
    sys.exit(main() or 0)
