#!/usr/bin/env python3
"""PreToolUse hook: block edits to canonical production paths from research/* branches.

Wiring: NOT active yet — must be added to .claude/settings.json PreToolUse
matcher for Edit|Write tools. Per spec Decision-Ledger 2026-04-29: activation
is a separate user-approved step so the user can observe one session first.

Block logic: git-only (works even when CRG is down).
  - Current branch starts with research/ or session/ AND
  - Target file is under pipeline/, trading_app/, or scripts/
  → BLOCK with clear message and override instruction.

Escape hatch: BRANCH_CONTEXT_OVERRIDE=1 env var bypasses the block.

CRG is NOT called here: hub-node identification would add latency and
require CRG to be running. The path-prefix list IS the canonical-file
heuristic per spec §F2 notes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Canonical production paths — edits from research/session branches are blocked
_CANONICAL_PREFIXES = (
    "pipeline/",
    "trading_app/",
    "scripts/",
)

# Branches whose edits to canonical paths are blocked
_RESEARCH_PREFIXES = (
    "research/",
    "session/",
)


def _current_branch() -> str | None:
    try:
        r = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def _is_canonical(file_path: str) -> bool:
    norm = file_path.replace("\\", "/")
    # Strip absolute project root prefix if present
    root = str(PROJECT_ROOT).replace("\\", "/").rstrip("/") + "/"
    if norm.lower().startswith(root.lower()):
        norm = norm[len(root) :]
    return any(norm.startswith(pfx) for pfx in _CANONICAL_PREFIXES)


def _is_research_branch(branch: str) -> bool:
    return any(branch.startswith(pfx) for pfx in _RESEARCH_PREFIXES)


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)  # fail-open on malformed input

    # Override escape hatch
    if os.environ.get("BRANCH_CONTEXT_OVERRIDE"):
        sys.exit(0)

    file_path = event.get("tool_input", {}).get("file_path", "")
    if not file_path:
        sys.exit(0)

    if not _is_canonical(file_path):
        sys.exit(0)

    branch = _current_branch()
    if not branch:
        sys.exit(0)  # can't determine branch — fail-open

    if not _is_research_branch(branch):
        sys.exit(0)  # non-research branch — allow

    # BLOCK
    short_path = file_path.replace("\\", "/")
    root = str(PROJECT_ROOT).replace("\\", "/").rstrip("/") + "/"
    if short_path.lower().startswith(root.lower()):
        short_path = short_path[len(root) :]

    msg = (
        f"\n"
        f"  BLOCKED: branch-context guard\n"
        f"  ─────────────────────────────────────────────\n"
        f"  Branch:  {branch}\n"
        f"  File:    {short_path}\n"
        f"  ─────────────────────────────────────────────\n"
        f"  Editing canonical production paths ({', '.join(_CANONICAL_PREFIXES[:-1])}, "
        f"{_CANONICAL_PREFIXES[-1]}) from a research/* or session/* branch is blocked.\n"
        f"\n"
        f"  This guard prevents accidental production changes during research sessions.\n"
        f"\n"
        f"  Resolutions:\n"
        f"    1. Switch to main or a feature/* branch:\n"
        f"         git checkout main\n"
        f"    2. If you intentionally need to edit this file from this branch:\n"
        f"         BRANCH_CONTEXT_OVERRIDE=1  (set in your shell env)\n"
        f"  ─────────────────────────────────────────────\n"
    )
    print(msg, file=sys.stderr)

    # Return blocking response (Claude Code hook protocol: non-zero exit blocks the tool)
    sys.exit(2)


if __name__ == "__main__":
    main()
