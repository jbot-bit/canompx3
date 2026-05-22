#!/usr/bin/env python3
"""Judgment-review soft-block: PreToolUse(Bash) — block `[judgment]` capital-class
commits that lack a code-review / capital-review / evidence-auditor mention.

Promotes the PostToolUse nudge (`judgment-review-nudge.py`, retained as a
second-layer catch for IDE-driven commits) to a forcing function that fires
BEFORE `git commit` lands. Mirrors the structural template of
`shared-state-commit-guard.py`: stdin JSON event, command-string extraction,
trailing flag-marker override, stderr WARN block, exit 2 BLOCK.

Trigger predicates (ALL four must be true to block):

  1. Commit message contains the literal token `[judgment]`.
  2. At least one staged file path matches a `_CAPITAL_PATH_PREFIXES` entry
     (sourced canonically from the sibling nudge module — no inline copy).
  3. Commit message does NOT match any `_REVIEW_MENTION_PATTERNS` regex.
  4. The marker file `.claude/scratch/.judgment-review-ts` was NOT touched
     within `_SUPPRESS_SECONDS` of now.

Override: trailing `# --audit-acknowledged` token on the bash command bypasses
the block (mirrors `shared-state-commit-guard.py` `# --shared-state-ack`).

Canonical-source delegation (institutional-rigor.md § 4): the four constants
`_CAPITAL_PATH_PREFIXES`, `_REVIEW_MENTION_PATTERNS`, `_MARKER_PATH`,
`_SUPPRESS_SECONDS` are imported at module load from the sibling
`judgment-review-nudge.py` via `importlib.util.spec_from_file_location` (the
nudge filename's hyphens prohibit a direct `from … import`). A drift parity
check (`check_judgment_review_capital_paths_parity` in `pipeline/check_drift.py`)
guards against accidental future inlining.

Fail-open contract (`branch-flip-protection.md` § "Fail-safe guarantee"): every
read error, missing git context, malformed event JSON, subprocess failure,
unmatched here-string, unreadable `-F` file, or unexpected exception exits 0.
The hook can never block a session it cannot reason about.

Test seam: `JUDGMENT_REVIEW_SCRATCH_DIR` env var overrides the marker-file
directory so the suppression path is testable in a tempdir. Production runs
ignore the env var and use the canonical `_MARKER_PATH` from the nudge.

Known gaps (documented, not bugs):
  - IDE-driven commits (VSCode source-control panel, GitHub Desktop) bypass
    the Bash tool entirely and therefore this hook. The PostToolUse nudge
    catches them after the fact.
  - `git commit --amend` is skipped by `_looks_like_commit` for consistency
    with the nudge. An operator can launder a `[judgment]` commit through an
    amend; the nudge still fires post-amend.

Doctrine grounding:
- `.claude/rules/adversarial-audit-gate.md` — the doctrine being mechanised.
- `.claude/rules/institutional-rigor.md` § 2 (after any fix, review the fix),
  § 4 (delegate to canonical sources), § 6 (no silent failures).
- `memory/project_review_enforcement_gaps_and_plan_2026_05_23.md` — the plan.
- `memory/feedback_n3_same_class_doctrine_threshold.md` — n=3 forcing-function
  threshold for "doctrine present, mechanism missing" class.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_HOOK_DIR = Path(__file__).resolve().parent
_NUDGE_PATH = _HOOK_DIR / "judgment-review-nudge.py"


def _load_nudge_constants():
    """Load canonical constants from the sibling nudge via importlib shim.

    Hyphenated filename forbids a plain `from … import`; spec_from_file_location
    is the canonical workaround. Fail-closed at import: if the nudge module
    cannot load, the soft-block has no canonical source and must not run with
    silently-inlined fallback values (institutional-rigor § 4 + § 6).
    """
    spec = importlib.util.spec_from_file_location(
        "judgment_review_nudge", str(_NUDGE_PATH)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"judgment-review-soft-block: could not build importlib spec for "
            f"sibling nudge at {_NUDGE_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _NUDGE = _load_nudge_constants()
    _CAPITAL_PATH_PREFIXES = _NUDGE._CAPITAL_PATH_PREFIXES
    _REVIEW_MENTION_PATTERNS = _NUDGE._REVIEW_MENTION_PATTERNS
    _DEFAULT_MARKER_PATH = _NUDGE._MARKER_PATH
    _SUPPRESS_SECONDS = _NUDGE._SUPPRESS_SECONDS
except Exception:
    # Fail-open at module-load: if the canonical source is unreadable, the
    # hook degrades to a no-op rather than blocking commits with inlined
    # fallback values. Logged to stderr for operator awareness.
    print(
        "[judgment-review-soft-block] WARN: could not load canonical "
        "constants from judgment-review-nudge.py; hook is a no-op.",
        file=sys.stderr,
    )
    sys.exit(0)


_OVERRIDE_TOKEN = "--audit-acknowledged"

_LIVE_TRADING_PREFIXES = (
    "trading_app/live/",
    "trading_app/risk_manager.py",
    "trading_app/execution_engine.py",
    "trading_app/session_orchestrator.py",
)


def _marker_path() -> Path:
    """Resolve the suppression marker path, honoring the test env override."""
    override = os.environ.get("JUDGMENT_REVIEW_SCRATCH_DIR")
    if override:
        return Path(override) / ".judgment-review-ts"
    return _DEFAULT_MARKER_PATH


def _suppressed_by_marker() -> bool:
    path = _marker_path()
    if not path.exists():
        return False
    try:
        age = time.time() - path.stat().st_mtime
        return age < _SUPPRESS_SECONDS
    except Exception:
        return False


def _looks_like_commit(command: str) -> bool:
    """Cheap allow-list: is this a `git commit` (not log/status/diff/amend)?"""
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


def _strip_override(command: str) -> tuple[str, bool]:
    """Remove the trailing `# --audit-acknowledged` token; return (stripped, was_present)."""
    if _OVERRIDE_TOKEN not in command:
        return command, False
    pattern = re.compile(r"#\s*--audit-acknowledged\b.*$", re.MULTILINE)
    stripped = pattern.sub("", command).rstrip()
    return stripped, True


# Matches: -m "msg", -m 'msg', -m msg, --message="msg", --message=msg, -F path
_MSG_RE_DOUBLE = re.compile(r'(?:^|\s)(?:-m|--message)\s*=?\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
_MSG_RE_SINGLE = re.compile(r"(?:^|\s)(?:-m|--message)\s*=?\s*'((?:[^'\\]|\\.)*)'", re.DOTALL)
_MSG_RE_BARE = re.compile(r"(?:^|\s)(?:-m|--message)\s*=?\s*([^\s'\"@-][^\s]*)")
_MSG_RE_PS_HERESTRING = re.compile(r"(?:^|\s)(?:-m|--message)\s*=?\s*@'(.*?)'@", re.DOTALL)
_MSG_FILE_RE = re.compile(r"(?:^|\s)-F\s+([^\s]+)")


def _extract_message(command: str, project_root: Path) -> str:
    """Concatenate every message payload found in the command string.

    Returns empty string when no payload can be extracted. We OR-concatenate
    all matches with newlines because operators occasionally use multiple
    `-m` flags (git treats each as a separate paragraph).
    """
    pieces: list[str] = []

    for m in _MSG_RE_PS_HERESTRING.finditer(command):
        pieces.append(m.group(1))

    no_here = _MSG_RE_PS_HERESTRING.sub(" ", command)

    for m in _MSG_RE_DOUBLE.finditer(no_here):
        pieces.append(m.group(1))
    no_double = _MSG_RE_DOUBLE.sub(" ", no_here)

    for m in _MSG_RE_SINGLE.finditer(no_double):
        pieces.append(m.group(1))
    no_quoted = _MSG_RE_SINGLE.sub(" ", no_double)

    for m in _MSG_RE_BARE.finditer(no_quoted):
        pieces.append(m.group(1))

    for m in _MSG_FILE_RE.finditer(command):
        fpath = m.group(1).strip("'\"")
        candidate = Path(fpath)
        if not candidate.is_absolute():
            candidate = project_root / fpath
        try:
            pieces.append(candidate.read_text(encoding="utf-8"))
        except Exception:
            # Unreadable -F file is fail-open: we cannot reason about the
            # message content, so we let the commit through.
            continue

    return "\n".join(p for p in pieces if p)


def _staged_paths(project_root: Path) -> list[str] | None:
    """Return staged file paths, or None on subprocess failure (fail-open)."""
    try:
        out = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0:
            return None
        return [
            line.strip().replace("\\", "/")
            for line in out.stdout.splitlines()
            if line.strip()
        ]
    except Exception:
        return None


def _project_root() -> Path:
    """Resolve project root from the current git worktree (cwd-aware)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip()).resolve()
    except Exception:
        pass
    return Path(__file__).resolve().parents[2]


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    if event.get("tool_name") != "Bash":
        sys.exit(0)

    raw_command = event.get("tool_input", {}).get("command", "") or ""
    if not isinstance(raw_command, str):
        sys.exit(0)

    command, override_present = _strip_override(raw_command)
    if override_present:
        sys.exit(0)

    if not _looks_like_commit(command):
        sys.exit(0)

    if _suppressed_by_marker():
        sys.exit(0)

    project_root = _project_root()
    message = _extract_message(command, project_root)
    if not message:
        # No extractable payload — fail-open. Examples: `git commit` with no
        # `-m` (would normally drop into $EDITOR; can't reason about content).
        sys.exit(0)

    # Predicate 1: [judgment] tag.
    if "[judgment]" not in message:
        sys.exit(0)

    # Predicate 3: explicit review mention suppresses.
    if any(p.search(message) for p in _REVIEW_MENTION_PATTERNS):
        sys.exit(0)

    # Predicate 2: at least one staged path matches a capital-class prefix.
    staged = _staged_paths(project_root)
    if staged is None:
        sys.exit(0)
    capital_hits = [
        f for f in staged
        if any(f.startswith(prefix) for prefix in _CAPITAL_PATH_PREFIXES)
    ]
    if not capital_hits:
        sys.exit(0)

    live_hits = [
        f for f in capital_hits
        if any(f.startswith(prefix) for prefix in _LIVE_TRADING_PREFIXES)
    ]
    primary_skill = "/capital-review" if live_hits else "/code-review"
    secondary_skill = "/code-review" if live_hits else "/capital-review"

    print("", file=sys.stderr)
    print("  ====================================================================", file=sys.stderr)
    print("  JUDGMENT-REVIEW SOFT-BLOCK: [judgment] commit on capital-class path.", file=sys.stderr)
    print("  --------------------------------------------------------------------", file=sys.stderr)
    print(f"  Staged capital-class files ({len(capital_hits)}):", file=sys.stderr)
    for f in capital_hits[:8]:
        print(f"      {f}", file=sys.stderr)
    if len(capital_hits) > 8:
        print(f"      ... and {len(capital_hits) - 8} more", file=sys.stderr)
    print("  --------------------------------------------------------------------", file=sys.stderr)
    print("  Per .claude/rules/adversarial-audit-gate.md and institutional-rigor.md § 2:", file=sys.stderr)
    print(f"    1. Run {primary_skill} (or {secondary_skill}) before this commit lands.", file=sys.stderr)
    print("    2. After review, mention the skill name in the commit body, e.g.:", file=sys.stderr)
    print('         "... includes /code-review pass — no blocking findings."', file=sys.stderr)
    print("    3. To suppress for 60 min after a manual review, run:", file=sys.stderr)
    print("         touch .claude/scratch/.judgment-review-ts", file=sys.stderr)
    print("    4. To override this single commit (post-review acknowledgement),", file=sys.stderr)
    print("       append the marker token as a trailing bash comment:", file=sys.stderr)
    print("         git commit -m \"[judgment] ...\"  # --audit-acknowledged", file=sys.stderr)
    print("  ====================================================================", file=sys.stderr)
    print("", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
