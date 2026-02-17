"""Auto-generate REPO_MAP.md from Python source using the ast module.

Usage:
    python scripts/tools/gen_repo_map.py          # Regenerate REPO_MAP.md
    python scripts/tools/gen_repo_map.py --check  # Exit 1 if REPO_MAP.md is stale
"""
from __future__ import annotations

import ast
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directories to scan (relative to project root)
SCAN_DIRS = ["pipeline", "trading_app", "scripts", "research", "tests"]

# Budget limits (lines) per section — auto-truncate if exceeded
TREE_BUDGET = 50
MODULE_BUDGET = 120
DEPS_BUDGET = 50
CLI_BUDGET = 25


def _count_loc(path: Path) -> int:
    """Count non-blank, non-comment lines."""
    count = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    except Exception:
        pass
    return count


def _parse_module(path: Path) -> dict:
    """Parse a Python file with ast. Returns info dict."""
    info = {
        "path": path,
        "loc": _count_loc(path),
        "summary": "",
        "exports": [],
        "imports": [],
        "has_main": False,
        "main_desc": "",
        "parse_error": False,
    }
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        info["parse_error"] = True
        return info

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        info["parse_error"] = True
        return info

    # Module docstring
    docstring = ast.get_docstring(tree)
    if docstring:
        first_sentence = docstring.strip().split("\n")[0].split(". ")[0]
        if len(first_sentence) > 80:
            first_sentence = first_sentence[:77] + "..."
        info["summary"] = first_sentence

    # Top-level classes, functions, imports
    exports = []
    imports = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                exports.append(node.name)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                exports.append(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                imports.add(node.module.split(".")[0])
        # Detect if __name__ == "__main__"
        elif isinstance(node, ast.If):
            try:
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == "__main__"
                ):
                    info["has_main"] = True
                    # Try to find argparse description
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Call):
                            func = sub.func
                            if isinstance(func, ast.Attribute) and func.attr == "ArgumentParser":
                                for kw in sub.keywords:
                                    if kw.arg == "description" and isinstance(kw.value, ast.Constant):
                                        desc = str(kw.value.value)
                                        if len(desc) > 60:
                                            desc = desc[:57] + "..."
                                        info["main_desc"] = desc
            except Exception:
                pass

    info["exports"] = exports[:5]  # Top 5 public exports
    # Filter imports to project-local only
    project_packages = {"pipeline", "trading_app", "scripts"}
    info["imports"] = sorted(imports & project_packages)
    return info


def _build_tree(root: Path, scan_dirs: list[str]) -> list[str]:
    """Build a directory tree view."""
    lines = []
    for scan_dir in sorted(scan_dirs):
        dirpath = root / scan_dir
        if not dirpath.is_dir():
            continue
        _walk_tree(dirpath, root, lines, prefix="")
    return lines


def _walk_tree(dirpath: Path, root: Path, lines: list[str], prefix: str) -> None:
    """Recursively build tree lines for a directory."""
    rel = dirpath.relative_to(root).as_posix()
    lines.append(f"{prefix}{rel}/")
    entries = sorted(dirpath.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    # Filter out __pycache__, .pyc
    entries = [e for e in entries if e.name != "__pycache__" and not e.name.endswith(".pyc")]
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "  "
        if entry.is_dir():
            _walk_tree(entry, root, lines, prefix=prefix + "  ")
        else:
            lines.append(f"{prefix}  {entry.name}")


def _collect_modules(root: Path, scan_dirs: list[str]) -> list[dict]:
    """Collect and parse all .py files in scan directories."""
    modules = []
    for scan_dir in scan_dirs:
        dirpath = root / scan_dir
        if not dirpath.is_dir():
            continue
        for py_file in sorted(dirpath.rglob("*.py")):
            # Skip __init__.py with zero LOC
            if py_file.name == "__init__.py" and _count_loc(py_file) == 0:
                continue
            modules.append(_parse_module(py_file))
    return modules


def _truncate_lines(lines: list[str], budget: int, label: str) -> list[str]:
    """Truncate a list of lines to budget, adding a note if truncated."""
    if len(lines) <= budget:
        return lines
    kept = lines[: budget - 1]
    kept.append(f"... and {len(lines) - budget + 1} more {label}")
    return kept


def generate_repo_map(root: Path) -> str:
    """Generate the full REPO_MAP.md content."""
    sections = []

    # Header
    sections.append("# REPO_MAP.md")
    sections.append("")
    sections.append("Auto-generated by `scripts/tools/gen_repo_map.py`. Do not edit manually.")
    sections.append("")

    # Section 1: Directory tree
    sections.append("## Directory Tree")
    sections.append("")
    sections.append("```")
    tree_lines = _build_tree(root, SCAN_DIRS)
    tree_lines = _truncate_lines(tree_lines, TREE_BUDGET, "entries")
    sections.extend(tree_lines)
    sections.append("```")
    sections.append("")

    # Section 2: Module index
    modules = _collect_modules(root, SCAN_DIRS)

    sections.append("## Module Index")
    sections.append("")
    sections.append("| Path | LOC | Summary | Key Exports |")
    sections.append("|------|-----|---------|-------------|")
    table_lines = []
    for mod in modules:
        rel_path = mod["path"].relative_to(root).as_posix()
        loc = mod["loc"]
        summary = mod["summary"] if not mod["parse_error"] else "(parse error)"
        exports = ", ".join(mod["exports"]) if mod["exports"] else ""
        table_lines.append(f"| `{rel_path}` | {loc} | {summary} | {exports} |")
    table_lines = _truncate_lines(table_lines, MODULE_BUDGET, "modules")
    sections.extend(table_lines)
    sections.append("")

    # Section 3: Dependency edges (cross-module imports)
    sections.append("## Cross-Package Dependencies")
    sections.append("")
    dep_lines = []
    for mod in modules:
        if mod["imports"]:
            rel_path = mod["path"].relative_to(root).as_posix()
            targets = ", ".join(mod["imports"])
            dep_lines.append(f"- `{rel_path}` -> {targets}")
    dep_lines = _truncate_lines(dep_lines, DEPS_BUDGET, "edges")
    if dep_lines:
        sections.extend(dep_lines)
    else:
        sections.append("(no cross-package dependencies found)")
    sections.append("")

    # Section 4: CLI entry points
    sections.append("## CLI Entry Points")
    sections.append("")
    cli_lines = []
    for mod in modules:
        if mod["has_main"]:
            rel_path = mod["path"].relative_to(root).as_posix()
            desc = mod["main_desc"] if mod["main_desc"] else "(no argparse description)"
            cli_lines.append(f"- `python {rel_path}` -- {desc}")
    cli_lines = _truncate_lines(cli_lines, CLI_BUDGET, "entry points")
    if cli_lines:
        sections.extend(cli_lines)
    else:
        sections.append("(no CLI entry points found)")
    sections.append("")

    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate REPO_MAP.md from source")
    parser.add_argument("--check", action="store_true", help="Check if REPO_MAP.md is stale (exit 1 if so)")
    args = parser.parse_args()

    content = generate_repo_map(PROJECT_ROOT)
    output_path = PROJECT_ROOT / "REPO_MAP.md"

    if args.check:
        if not output_path.exists():
            print("REPO_MAP.md does not exist. Run: python scripts/tools/gen_repo_map.py")
            sys.exit(1)
        existing = output_path.read_text(encoding="utf-8")
        if existing.rstrip() != content.rstrip():
            print("REPO_MAP.md is stale. Run: python scripts/tools/gen_repo_map.py")
            sys.exit(1)
        print("REPO_MAP.md is up to date.")
        sys.exit(0)
    else:
        output_path.write_text(content, encoding="utf-8")
        print(f"Generated {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
