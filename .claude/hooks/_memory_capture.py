#!/usr/bin/env python3
"""Shared signal-gathering + breadcrumb I/O for the auto-memory-capture loop.

ONE source of truth imported by both hook files (institutional-rigor §4 — no
parallel re-encoding across the three events):
  - memory-capture-advisory.py     (PreCompact warning + SessionEnd breadcrumb)
  - memory-capture-sessionstart.py (Claude-visible cue on next session)

Design contract (all verified against the Stage-0 probe + official hooks docs):
  - Gates on FACTUAL git state only (commit count, diff names, stage/doctrine
    globs). NEVER parses transcript prose for salience — deterministic and
    unfoolable by narrative (no-look-ahead). "Capture nothing" is a valid
    outcome; the cue is a JUDGE prompt, never an auto-writer.
  - Fail-open at every step: any git/IO error -> empty/zero, never raises.
  - All git calls cwd=PROJECT_ROOT, short timeout.

State files (all under .claude/hooks/state/, gitignored contents):
  - memory-capture-pending.json   breadcrumb: {session_id, counts, ts, consumed}
  - memory-capture.log            JSONL telemetry (one line per SessionEnd)
  - memory-capture-advisory.json  PreCompact dedup: {advised_sessions: [...]}
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# .claude/hooks/_memory_capture.py -> parents[2] == project root (worktree root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / ".claude" / "hooks" / "state"
PENDING_PATH = STATE_DIR / "memory-capture-pending.json"
TELEMETRY_PATH = STATE_DIR / "memory-capture.log"
ADVISORY_PATH = STATE_DIR / "memory-capture-advisory.json"

_GIT_TIMEOUT = 3
_BREADCRUMB_MAX_AGE_HOURS = 24
_ADVISED_CAP = 200

# Diff-name prefixes that signal stage / doctrine work worth a capture judgement.
_STAGE_PREFIX = "docs/runtime/stages/"
_RULES_PREFIX = ".claude/rules/"
_DOCTRINE_TOKENS = ("doctrine", "rule")


def _git(args: list[str]) -> tuple[int, str]:
    """Run a git command at PROJECT_ROOT. Returns (rc, stdout). (1, "") on error."""
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
        return r.returncode, r.stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return 1, ""


def _git_dir() -> Path | None:
    """Resolve THIS worktree's git-dir (where .claude.pid lives).

    For a linked worktree this is `<repo>/.git/worktrees/<name>/`, NOT the
    top-level `.git/`. Mirrors session-start.py:_git_dir(). None on failure.
    """
    rc, out = _git(["rev-parse", "--git-dir"])
    if rc != 0 or not out.strip():
        return None
    p = Path(out.strip())
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _head_at_start() -> str | None:
    """Read `head_at_start` from the session lock, if present. None on any miss."""
    git_dir = _git_dir()
    if git_dir is None:
        return None
    lock_path = git_dir / ".claude.pid"
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None
    sha = data.get("head_at_start")
    return sha.strip() if isinstance(sha, str) and sha.strip() else None


def _commit_count() -> int:
    """Commits made this session: <head_at_start>..HEAD, else origin/main..HEAD."""
    base = _head_at_start()
    if base:
        rc, out = _git(["rev-list", "--count", f"{base}..HEAD"])
        if rc == 0 and out.strip().isdigit():
            return int(out.strip())
    # Fallback: no usable lock baseline -> compare against the remote main tip.
    rc, out = _git(["rev-list", "--count", "origin/main..HEAD"])
    if rc == 0 and out.strip().isdigit():
        return int(out.strip())
    return 0


def _diff_names() -> list[str]:
    """Uncommitted-edit file paths (staged + unstaged). Empty on failure."""
    rc, out = _git(["diff", "--name-only", "HEAD"])
    if rc != 0:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def _is_stage_file(norm: str) -> bool:
    return norm.startswith(_STAGE_PREFIX)


def _is_doctrine_file(norm: str) -> bool:
    if norm.startswith(_RULES_PREFIX):
        return True
    base = norm.rsplit("/", 1)[-1].lower()
    return base.endswith(".md") and any(tok in base for tok in _DOCTRINE_TOKENS)


def gather_signals() -> dict:
    """Collect factual session-work signals. Always returns a dict; never raises."""
    try:
        commits = _commit_count()
        names = [n.replace("\\", "/") for n in _diff_names()]
        stage_files = [n for n in names if _is_stage_file(n)]
        doctrine_files = [n for n in names if _is_doctrine_file(n)]
        return {
            "commits": commits,
            "files": len(names),
            "stage_files": stage_files,
            "doctrine_files": doctrine_files,
        }
    except BaseException:  # pragma: no cover - fail-open
        return {"commits": 0, "files": 0, "stage_files": [], "doctrine_files": []}


def signal_meets_threshold(sig: dict) -> bool:
    """True if the session did enough durable work to warrant a capture judgement."""
    return bool(
        sig.get("commits", 0) >= 1
        or sig.get("files", 0) >= 3
        or sig.get("stage_files")
        or sig.get("doctrine_files")
    )


def describe_signals(sig: dict) -> str:
    """Human/Claude-readable one-liner of the accrued work."""
    parts = [
        f"{sig.get('commits', 0)} commit(s)",
        f"{sig.get('files', 0)} file(s) edited",
    ]
    if sig.get("stage_files"):
        parts.append(f"{len(sig['stage_files'])} stage file(s)")
    if sig.get("doctrine_files"):
        parts.append(f"{len(sig['doctrine_files'])} doctrine/rule file(s)")
    return ", ".join(parts)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Breadcrumb I/O (SessionEnd writes, SessionStart consumes)
# ---------------------------------------------------------------------------
def write_breadcrumb(session_id: str, sig: dict) -> None:
    """Persist a pending capture-judgement breadcrumb. Swallows all IO errors."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        PENDING_PATH.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "counts": sig,
                    "ts": _now_iso(),
                    "consumed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def read_breadcrumb() -> dict | None:
    """Load the pending breadcrumb. None if missing/unreadable/corrupt."""
    try:
        return json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return None


def mark_breadcrumb_consumed() -> None:
    """One-shot: flip consumed=true so the cue fires at most once per breadcrumb."""
    crumb = read_breadcrumb()
    if crumb is None:
        return
    crumb["consumed"] = True
    try:
        PENDING_PATH.write_text(json.dumps(crumb, indent=2), encoding="utf-8")
    except OSError:
        pass


def breadcrumb_is_fresh(crumb: dict) -> bool:
    """True if the breadcrumb is younger than the 24h expiry window."""
    ts = crumb.get("ts")
    if not isinstance(ts, str):
        return False
    try:
        when = datetime.fromisoformat(ts)
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    age_hours = (datetime.now(UTC) - when).total_seconds() / 3600.0
    return age_hours < _BREADCRUMB_MAX_AGE_HOURS


# ---------------------------------------------------------------------------
# Telemetry (SessionEnd always appends one line)
# ---------------------------------------------------------------------------
def append_telemetry(record: dict) -> None:
    """Append one JSONL telemetry line. Swallows all IO errors."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with TELEMETRY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# PreCompact dedup (advise once per session)
# ---------------------------------------------------------------------------
def already_advised(session_id: str) -> bool:
    try:
        data = json.loads(ADVISORY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        return False
    return session_id in data.get("advised_sessions", [])


def record_advised(session_id: str) -> None:
    """Append session_id to the advised list (capped). Swallows all IO errors."""
    try:
        data = json.loads(ADVISORY_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
        data = {}
    advised = data.get("advised_sessions", [])
    if session_id not in advised:
        advised.append(session_id)
    data["advised_sessions"] = advised[-_ADVISED_CAP:]
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        ADVISORY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass
