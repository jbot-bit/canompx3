#!/usr/bin/env python3
"""Check that file paths referenced in docs/rules actually exist.

Scans CLAUDE.md and .claude/rules/*.md for backtick-quoted path references
and validates each one exists on disk.

Returns exit code 0 if all referenced paths exist, 1 if any are broken.
Prints broken references to stdout (one per line in non-verbose mode).

Usage:
    python scripts/tools/check_referenced_paths.py [--verbose]

Token normalization (applied before existence check):
- CLI invocations: split on whitespace, take first token (`python foo.py` -> `foo.py`).
- Line-number suffixes: strip `:digits` or `:digits-digits` (`foo.py:510` -> `foo.py`).
- Qualified-name refs: split on `::`, take left half (`foo.py::bar` -> `foo.py`).
  NOTE: symbol existence is NOT validated. The right canonical source for that
  is `code-review-graph` Function nodes; not wired here to keep the tool
  stdlib-only. If a symbol-rename bug bites, add CRG validation with evidence.

Skip rules:
- Templates: tokens containing `<`, `>`, `YYYY`, or `|` (e.g. `<slug>`,
  `YYYY-MM-DD-<slug>.yaml|.md`).
- Anti-pattern citations: paths in ANTI_PATTERN_PATHS — referenced
  intentionally as "never use this" (see ANTI_PATTERN_PATHS for the set).
- Memory paths: `memory/` lives in user dir, not the repo.
- URLs and shell variables.
- Glob patterns containing `*` or `?`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PATH_RE = re.compile(r"`([^`]+)`")

# Trailing line-number suffix: `:N` or `:N-M` (and any further `:N-M` segments
# joined by commas, like `:586-594, :612-616, :357-381`). Trailing commas /
# whitespace after the last segment are also tolerated.
_LINE_SUFFIX_RE = re.compile(r"(:\d+(?:-\d+)?)(?:\s*,\s*:\d+(?:-\d+)?)*[\s,]*$")

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

# Paths cited intentionally as anti-patterns ("never use this").
# Validating them as broken refs would create permanent false positives.
ANTI_PATTERN_PATHS = frozenset(
    {
        "/tmp/gold.db",
        "C:\\db\\gold.db",
    }
)


def _normalize(token: str) -> str:
    """Normalize a token to a candidate file path.

    Strategy: split on whitespace and find the first sub-token that looks
    like a path (has a recognized file extension). This handles both
    `python foo.py` (skip `python`) and `foo.py --flag` (take `foo.py`).
    Then strip qualified-name `::suffix` and line-number `:N-M` suffixes.
    Returns an empty string if no path-like sub-token is found.
    """
    # 1. CLI / argument tokenization. If whitespace present, find the first
    # sub-token that has a file extension; otherwise return empty.
    if " " in token or "\t" in token:
        sub_tokens = token.split()
        token = ""
        for sub in sub_tokens:
            # Strip any trailing punctuation that breaks ext detection
            cleaned = sub.rstrip(",;.")
            # Quick path-likeness: ends with a recognized extension OR has /
            if any(cleaned.endswith(ext) for ext in _FILE_EXTENSIONS) or "/" in cleaned:
                # Also need an extension somewhere or it's just a flag
                stem = cleaned.split("::")[0]
                stem = _LINE_SUFFIX_RE.sub("", stem)
                if any(stem.endswith(ext) for ext in _FILE_EXTENSIONS):
                    token = cleaned
                    break
        if not token:
            return ""

    # 2. Qualified-name refs: take the file half.
    if "::" in token:
        token = token.split("::", 1)[0]

    # 3. Line-number suffixes: strip trailing `:N(-M)?(, :N(-M)?)*`.
    token = _LINE_SUFFIX_RE.sub("", token)

    return token


def _is_template(token: str) -> bool:
    if "<" in token or ">" in token:
        return True
    if "YYYY" in token:
        return True
    # Pipe inside an unbacktick'd token suggests a "foo|bar" template.
    # We can't know for sure, but combined with file-extension dual-suffixes
    # like `.yaml|.md` it's a strong signal.
    if "|" in token:
        return True
    return False


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
    if token.startswith(("$", "{")):
        return False
    if _is_template(token):
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


def _is_skipped(token: str) -> bool:
    """Return True if the token is intentionally skipped (not a real broken ref)."""
    if token in ANTI_PATTERN_PATHS:
        return True
    if token.startswith("memory/"):
        # Memory files live in user dir, not the repo.
        return True
    if token.startswith("stages/"):
        # Relative path inside docs/runtime/; rule docs use it as shorthand.
        # Not resolvable as a project-root path.
        return True
    return False


def _extract_refs(text: str) -> list[str]:
    """Yield (normalized) candidate paths for existence-checking."""
    out = []
    for m in _PATH_RE.finditer(text):
        raw = m.group(1)
        normalized = _normalize(raw)
        if not _looks_like_path(normalized):
            continue
        if _is_skipped(normalized):
            continue
        out.append(normalized)
    return out


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
