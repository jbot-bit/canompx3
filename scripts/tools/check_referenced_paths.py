#!/usr/bin/env python3
"""Check that file paths referenced in docs/rules/memory actually exist.

Scans CLAUDE.md, .claude/rules/*.md for backtick-quoted path references
and validates each one exists on disk.

Returns exit code 0 if all referenced paths exist, 1 if any are broken.
Prints broken references to stdout (one per line).

Usage:
    python scripts/tools/check_referenced_paths.py [--verbose]

Design notes:
- Only backtick-quoted paths are checked (e.g. `pipeline/foo.py`).
- Paths containing glob wildcards (*) are skipped — not literal references.
- Python dotted module paths (pipeline.dst.SESSION_CATALOG) are skipped.
- Paths starting with http/https are skipped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PATH_RE = re.compile(r"`([^`]+)`")

_FILE_EXTENSIONS = {
    ".py",
    ".md",
    ".sh",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".toml",
    ".cfg",
    ".ini",
    ".env",
    ".sql",
    ".csv",
    ".db",
    ".patch",
}


def _looks_like_path(token: str) -> bool:
    if not token:
        return False
    if token.startswith(("http://", "https://")):
        return False
    if "*" in token or "?" in token:
        return False
    # Python module refs: contain . but no /
    if "." in token and "/" not in token and "\\" not in token:
        return False
    if token.startswith(("$", "{", "<")):
        return False
    has_path_prefix = token.startswith(
        (
            "pipeline/",
            "trading_app/",
            "scripts/",
            "docs/",
            ".claude/",
            ".githooks/",
            "tests/",
            "resources/",
            "research/",
            "evaluate/",
            "CLAUDE.md",
            "TRADING_RULES.md",
            "RESEARCH_RULES.md",
            "HANDOFF.md",
            "ROADMAP.md",
            "REPO_MAP.md",
            "AGENTS.md",
        )
    )
    has_ext = any(token.endswith(ext) for ext in _FILE_EXTENSIONS)
    return has_path_prefix or has_ext


def _extract_refs(text: str) -> list[str]:
    return [m.group(1) for m in _PATH_RE.finditer(text) if _looks_like_path(m.group(1))]


def _scan_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return _extract_refs(text)


def _files_to_scan() -> list[Path]:
    targets: list[Path] = []
    claude_md = PROJECT_ROOT / "CLAUDE.md"
    if claude_md.exists():
        targets.append(claude_md)
    rules_dir = PROJECT_ROOT / ".claude" / "rules"
    if rules_dir.is_dir():
        targets.extend(sorted(rules_dir.glob("*.md")))
    return targets


def main(verbose: bool = False) -> int:
    files = _files_to_scan()
    broken: list[tuple[Path, str]] = []

    for src in files:
        refs = _scan_file(src)
        for ref in refs:
            candidate = PROJECT_ROOT / ref
            if not candidate.exists():
                broken.append((src, ref))
                if verbose:
                    print(f"  MISSING  {ref}  (in {src.relative_to(PROJECT_ROOT)})")

    if not broken:
        if verbose:
            print(f"OK: all referenced paths exist ({len(files)} files scanned)")
        return 0

    if not verbose:
        for _, ref in broken:
            print(ref)
    else:
        print(f"\n{len(broken)} broken reference(s) found across {len(files)} files.")

    return 1


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    sys.exit(main(verbose=verbose))
