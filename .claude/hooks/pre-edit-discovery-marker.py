#!/usr/bin/env python3
"""Pre-Edit Discovery Marker — Tier 2 of discovery-loop hardening.

Blocks Edit/Write to pipeline/ or trading_app/ when the session has not
produced one of three discovery-convergence artifacts:

  (a) REPRO:                        — failing command + actual vs expected
  (b) `context_resolver.py` output  — narrowed blast radius
  (c) TRIVIAL:                      — declaration with diff <100 lines

Reads the session JSONL transcript at
~/.claude/projects/<project-slug>/<session_id>.jsonl and walks the most
recent records.

Fail-open guarantees:
- Missing/short transcript (<5 records): pass (session just started).
- Marker file present and valid: pass (manual escape hatch).
- Trivial paths (docs/, tests/, scripts/tools/, .claude/, *.md/yaml/json): pass.
- Any exception (transcript missing, JSON parse error, OS error): pass.

Refs: docs/plans/discovery-loop-hardening.md § Tier 2.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SLUG = "C--Users-joshd-canompx3"
TRANSCRIPT_DIR = Path.home() / ".claude" / "projects" / PROJECT_SLUG
MARKER_FILE = PROJECT_ROOT / ".claude" / "scratch" / "discovery-marker.json"

# Walk the last N transcript records (cheap; one turn ~ 6-15 records).
SCAN_RECORD_BUDGET = 200

# Conditions for marker validity.
REPRO_RE = re.compile(r"\bREPRO:", re.IGNORECASE)
TRIVIAL_RE = re.compile(r"\bTRIVIAL:", re.IGNORECASE)
CONTEXT_RESOLVER_RE = re.compile(r"context_resolver\.py", re.IGNORECASE)

# Trivial path exclusions (gate does not apply).
TRIVIAL_PATH_PREFIXES = (
    "docs/",
    "tests/",
    "scripts/tools/",
    ".claude/",
    ".github/",
    ".codex/",
    "memory/",
)
TRIVIAL_FILE_SUFFIXES = (
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".txt",
)

# Production-code prefixes that the gate guards.
GUARDED_PATH_PREFIXES = (
    "pipeline/",
    "trading_app/",
)


def _norm(path: str) -> str:
    return (path or "").replace("\\", "/").lstrip("./")


def _is_guarded_path(file_path: str) -> bool:
    norm = _norm(file_path)
    if not norm:
        return False
    if any(norm.startswith(p) for p in TRIVIAL_PATH_PREFIXES):
        return False
    if norm.endswith(TRIVIAL_FILE_SUFFIXES):
        return False
    return any(norm.startswith(p) or f"/{p}" in f"/{norm}" for p in GUARDED_PATH_PREFIXES)


def _marker_file_active() -> bool:
    """True if .claude/scratch/discovery-marker.json exists with a future
    valid_until timestamp."""
    try:
        data = json.loads(MARKER_FILE.read_text(encoding="utf-8"))
        valid_until = datetime.fromisoformat(data["valid_until"])
        return valid_until > datetime.now(UTC)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError, OSError):
        return False


def _iter_transcript_text(session_id: str) -> list[str]:
    """Return up to SCAN_RECORD_BUDGET extracted text fragments from the
    session transcript, oldest-to-newest within the budget window.

    Each fragment is a flat string we can regex against:
    - User message text content
    - Assistant tool_use Bash commands (the `command` field)
    """
    transcript = TRANSCRIPT_DIR / f"{session_id}.jsonl"
    if not transcript.exists():
        return []

    fragments: list[str] = []
    record_count = 0
    try:
        # Read all lines; for typical sessions this is <2 MB. For very long
        # sessions we keep only the tail.
        with transcript.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []

    record_count = len(lines)
    if record_count < 5:
        # Fail-open signal: session just started.
        return ["__TRANSCRIPT_TOO_SHORT__"]

    tail = lines[-SCAN_RECORD_BUDGET:]
    for line in tail:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        if rec.get("type") not in ("user", "assistant"):
            continue
        msg = rec.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            fragments.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    fragments.append(block.get("text", ""))
                elif btype == "tool_use":
                    # Capture Bash command + Read/Edit file paths so
                    # context_resolver.py invocations are detectable.
                    inp = block.get("input") or {}
                    cmd = inp.get("command") or ""
                    fp = inp.get("file_path") or ""
                    if cmd:
                        fragments.append(cmd)
                    if fp:
                        fragments.append(fp)
                elif btype == "tool_result":
                    # tool_result content can carry context_resolver output;
                    # capture stringified version cheaply.
                    rc = block.get("content")
                    if isinstance(rc, str):
                        fragments.append(rc[:2000])
                    elif isinstance(rc, list):
                        for rb in rc:
                            if isinstance(rb, dict) and rb.get("type") == "text":
                                fragments.append(rb.get("text", "")[:2000])
    return fragments


def _staged_diff_under_100() -> bool:
    """True if `git diff --cached --shortstat` reports <100 net lines.

    Fail-open: any error returns True (we can't prove it's >=100).
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--shortstat"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=PROJECT_ROOT,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return True
    if result.returncode != 0:
        return True
    out = result.stdout or ""
    # Format: " 3 files changed, 47 insertions(+), 12 deletions(-)"
    ins = re.search(r"(\d+)\s+insertion", out)
    dels = re.search(r"(\d+)\s+deletion", out)
    net = (int(ins.group(1)) if ins else 0) + (int(dels.group(1)) if dels else 0)
    return net < 100


def _has_marker(fragments: list[str]) -> tuple[bool, str]:
    """Walk fragments newest-to-oldest; return (passed, reason)."""
    # Empty list = transcript missing/unreadable → fail-open.
    if not fragments:
        return True, "transcript empty or unreadable, fail-open"
    # Sentinel from _iter_transcript_text for short sessions.
    if fragments == ["__TRANSCRIPT_TOO_SHORT__"]:
        return True, "transcript too short — session just started, fail-open"

    # Walk from newest backward.
    for frag in reversed(fragments):
        if CONTEXT_RESOLVER_RE.search(frag):
            return True, "context_resolver.py invoked in session"
        if REPRO_RE.search(frag):
            return True, "REPRO: declaration found"
        if TRIVIAL_RE.search(frag):
            if _staged_diff_under_100():
                return True, "TRIVIAL: declaration found, staged diff <100 lines"
            return False, "TRIVIAL: declared but staged diff >=100 lines"
    return False, "no discovery-convergence artifact found in session"


BLOCK_MESSAGE_TEMPLATE = """DISCOVERY-MARKER GUARD: about to edit {file_path} but no discovery-convergence artifact found in this session.

Reason: {reason}

Before editing production code (pipeline/ or trading_app/), produce ONE of:
  (a) REPRO:        failing command + actual vs expected output
  (b) context_resolver.py output narrowing the blast radius:
        python scripts/tools/context_resolver.py --task "<x>" --format markdown
  (c) TRIVIAL: declaration with file list and diff <100 lines

Manual escape (use sparingly):
  Create .claude/scratch/discovery-marker.json with:
    {{"valid_until": "<ISO timestamp 1h in future>"}}

See docs/plans/discovery-loop-hardening.md § Tier 2.
"""


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    file_path = event.get("tool_input", {}).get("file_path", "")
    if not _is_guarded_path(file_path):
        sys.exit(0)

    if _marker_file_active():
        sys.exit(0)

    session_id = event.get("session_id", "") or os.environ.get("CLAUDE_SESSION_ID", "")
    if not session_id:
        # Cannot read transcript without session_id — fail-open.
        sys.exit(0)

    try:
        fragments = _iter_transcript_text(session_id)
    except Exception:
        sys.exit(0)

    passed, reason = _has_marker(fragments)
    if passed:
        sys.exit(0)

    print(
        BLOCK_MESSAGE_TEMPLATE.format(file_path=file_path, reason=reason),
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
