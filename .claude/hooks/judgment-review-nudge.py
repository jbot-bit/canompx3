#!/usr/bin/env python3
"""Judgment-review nudge: PostToolUse(Bash) — nudge after [judgment] commits that touch capital-class paths.

Fires after any `git commit` Bash call. Reads HEAD via `git log -1 --format=%s%n%b`
+ `git show --name-only HEAD`, and if BOTH apply:

  1) commit subject or body contains the literal token `[judgment]` (the
     severity-tag convention used by `.claude/rules/adversarial-audit-gate.md`).
  2) at least one file changed under a capital-class path:
       trading_app/live/, trading_app/risk_manager.py,
       trading_app/execution_engine.py, trading_app/session_orchestrator.py,
       pipeline/

THEN check whether judgment-class review was likely run recently:

  - commit body already mentions code-review / capital-review / evidence-auditor
    / ultrareview / adversarial-audit (suppresses — review was performed)
  - sibling marker file `.claude/scratch/.judgment-review-ts` modified within
    the last 60 minutes (suppresses — manual silence)

Otherwise emit a SINGLE-LINE additionalContext nudge to stdout suggesting
`/code-review` or `/capital-review`. Never blocks, never exits non-zero.

Fail-safe contract: every read error, missing git context, malformed event,
subprocess failure, or unexpected exception exits 0 with no output. The hook
must never interrupt a session it cannot reason about.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MARKER_PATH = _PROJECT_ROOT / ".claude" / "scratch" / ".judgment-review-ts"
_SUPPRESS_SECONDS = 60 * 60  # 60 minutes

_CAPITAL_PATH_PREFIXES = (
    "trading_app/live/",
    "trading_app/risk_manager.py",
    "trading_app/execution_engine.py",
    "trading_app/session_orchestrator.py",
    "pipeline/",
)

# Mentions in the commit body that indicate review was already done.
_REVIEW_MENTION_PATTERNS = (
    re.compile(r"\bcode[-\s]?review\b", re.IGNORECASE),
    re.compile(r"\bcapital[-\s]?review\b", re.IGNORECASE),
    re.compile(r"\bevidence[-\s]?auditor\b", re.IGNORECASE),
    re.compile(r"\bultrareview\b", re.IGNORECASE),
    re.compile(r"\badversarial[-\s]audit\b", re.IGNORECASE),
)


def _git(*args: str) -> str | None:
    """Run a git command in the project root; return stdout or None on failure."""
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        return r.stdout
    except Exception:
        return None


def _looks_like_commit(command: str) -> bool:
    """Cheap check that the Bash call was a `git commit` (not `git log`/`git status`/`git show`)."""
    if "git" not in command:
        return False
    if "commit" not in command:
        return False
    if "--amend" in command:
        return False
    for bad in (" log ", " show ", " status", " diff ", " reflog"):
        if bad in command:
            return False
    return True


def _suppressed_by_marker() -> bool:
    if not _MARKER_PATH.exists():
        return False
    try:
        age = time.time() - _MARKER_PATH.stat().st_mtime
        return age < _SUPPRESS_SECONDS
    except Exception:
        return False


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if event.get("tool_name") != "Bash":
        sys.exit(0)

    command = event.get("tool_input", {}).get("command", "") or ""
    if not _looks_like_commit(command):
        sys.exit(0)

    # Read HEAD commit message + changed files.
    msg = _git("log", "-1", "--format=%s%n%b")
    if msg is None:
        sys.exit(0)

    # Trigger 1: judgment-tag in subject or body.
    if "[judgment]" not in msg:
        sys.exit(0)

    # Suppress if commit body already cites a review.
    if any(p.search(msg) for p in _REVIEW_MENTION_PATTERNS):
        sys.exit(0)

    # Suppress if a skill marker was written recently.
    if _suppressed_by_marker():
        sys.exit(0)

    # Trigger 2: at least one capital-class path touched.
    files_out = _git("show", "--name-only", "--format=", "HEAD")
    if not files_out:
        sys.exit(0)
    changed = [line.strip() for line in files_out.splitlines() if line.strip()]
    touches_capital = any(
        any(f.startswith(prefix) for prefix in _CAPITAL_PATH_PREFIXES)
        for f in changed
    )
    if not touches_capital:
        sys.exit(0)

    # Emit nudge. Single-line to stdout; this lands in additionalContext.
    short_sha = _git("rev-parse", "--short", "HEAD")
    sha_disp = short_sha.strip() if short_sha else "HEAD"
    print(
        f"[judgment-review nudge] commit {sha_disp} is [judgment] on a capital-class path "
        f"with no code-review / capital-review / evidence-auditor mention. Consider "
        f"`/code-review` or `/capital-review` before the next change. (Suppress for 60m "
        f"by `touch .claude/scratch/.judgment-review-ts`.)"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
