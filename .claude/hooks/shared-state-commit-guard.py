#!/usr/bin/env python3
"""Shared-state commit guard: PreToolUse(Bash) — warn before committing files
under shared-state directories without a peer-coordination check.

Triggers on Bash commands matching `git add` or `git commit` that include
paths under `docs/runtime/` or `docs/audit/`. Runs three checks:

  1. Sibling-commit drift: any new commits on the current branch since the
     session-start lock was written? If yes, the agent's parent SHA is stale.
  2. Peer scope-lock: any active stage file in `docs/runtime/stages/` declares
     `scope_lock` on the target file from a DIFFERENT session?
  3. Sibling working-tree heat: any sibling worktree (per `git worktree list`)
     has a dirty status on the same file path?

On any HIT the hook prints a WARN block via stderr and exits 2 (BLOCK).
The agent can override by re-running with `--shared-state-ack` appended to
the bash command (parsed and stripped before git sees it via a hook-side
recommendation — agent must remove the flag manually to proceed).

Fail-safe: every read error / parse error / non-git context exits 0 (pass).
Triggered by: `feedback_multi_terminal_shared_file_thrash_2026_05_21.md` n=1.

Doctrine grounding:
- `.claude/rules/multi-terminal-shared-file-hygiene.md` (companion rule)
- `.claude/rules/parallel-session-isolation.md` (worktree-level rule)
- `.claude/rules/branch-flip-protection.md` (branch-level rule)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

SHARED_STATE_DIRS = (
    "docs/runtime/",
    "docs/audit/",
)

# Files inside SHARED_STATE_DIRS that are NEVER concurrently written by other
# terminals (per repo convention) - skip the guard for these to avoid false
# positives. Add paths here if they prove to be safe.
SAFE_SUBPATHS_PREFIX = (
    "docs/runtime/stages/",  # stage files claim their own scope; one terminal per stage
    "docs/audit/hypotheses/drafts/",  # draft preregs are author-owned
)


def _git_dir() -> Path | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if out.returncode != 0:
            return None
        return Path(out.stdout.strip()).resolve()
    except Exception:
        return None


def _read_session_lock(git_dir: Path) -> dict | None:
    lock = git_dir / ".claude.pid"
    if not lock.exists():
        return None
    try:
        return json.loads(lock.read_text(encoding="utf-8"))
    except Exception:
        return None


def _commits_since(ref: str, exclude_author_email: str | None = None) -> list[str]:
    """SHAs on HEAD since the given commit, exclusive.

    When `exclude_author_email` is provided, commits authored by that email
    are filtered out — they are this session's own work, not peer noise.
    """
    try:
        out = subprocess.run(
            ["git", "log", f"{ref}..HEAD", "--format=%H|%ae|%s"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode != 0:
            return []
        results: list[str] = []
        for raw in out.stdout.splitlines():
            if not raw.strip():
                continue
            parts = raw.split("|", 2)
            if len(parts) != 3:
                continue
            sha, ae, subj = parts
            if exclude_author_email and ae.strip().lower() == exclude_author_email.lower():
                continue
            results.append(f"{sha} {subj}")
        return results
    except Exception:
        return []


def _current_author_email() -> str | None:
    try:
        out = subprocess.run(
            ["git", "config", "user.email"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if out.returncode != 0:
            return None
        v = out.stdout.strip()
        return v or None
    except Exception:
        return None


_PATH_TRIM_CHARS = " \t\r\n'\"`;&|<>(){}[]"


def _extract_paths(command: str) -> list[str]:
    """Find paths under SHARED_STATE_DIRS mentioned in a `git add|commit` command.

    Tokens are split on whitespace and aggressively trimmed of shell/JSON
    punctuation. A token only counts as a path if (a) it starts with a
    non-flag char, (b) contains at least one SHARED_STATE_DIRS prefix as a
    substring after a path-boundary char (start or '/' or '\\'), and (c) does
    NOT match any SAFE_SUBPATHS_PREFIX.
    """
    paths: list[str] = []
    normalized = command.replace("\\", "/")
    for raw in normalized.split():
        token = raw.strip(_PATH_TRIM_CHARS)
        if not token or token.startswith("-"):
            continue
        # Skip paths under safe subpaths (stage files, draft preregs)
        if any(s in token for s in SAFE_SUBPATHS_PREFIX):
            continue
        for d in SHARED_STATE_DIRS:
            # Match `d` only at a path boundary to avoid e.g. "foo.docs/runtime"
            idx = token.find(d)
            if idx == -1:
                continue
            if idx > 0 and token[idx - 1] != "/":
                continue
            paths.append(token[idx:])
            break
    return sorted(set(paths))


_MODE_RE = re.compile(r"^mode:\s*(\S+)", re.MULTILINE)
_CLOSED_MODES = {"CLOSED", "DONE", "ARCHIVED", "COMPLETE"}


def _stage_files_claiming_path(project_root: Path, target: str) -> list[str]:
    """Return list of ACTIVE stage file names that claim scope_lock on the target.

    Filters out stage files whose `mode:` is in _CLOSED_MODES — those are
    retained-for-canonical-source files (e.g. Stage 2A.3 retained as Check
    #173's STATUS_VALUES enum source per the inline-copy parity rule) and
    do not represent in-flight peer work.
    """
    stages = project_root / "docs" / "runtime" / "stages"
    if not stages.is_dir():
        return []
    hits = []
    target_normalized = target.replace("\\", "/").lstrip("./")
    for sf in stages.glob("*.md"):
        if sf.name == ".gitkeep":
            continue
        try:
            content = sf.read_text(encoding="utf-8")
        except Exception:
            continue
        if "scope_lock" not in content.lower():
            continue
        if target_normalized not in content.replace("\\", "/"):
            continue
        m = _MODE_RE.search(content)
        mode = (m.group(1).strip().upper() if m else "")
        if mode in _CLOSED_MODES:
            continue
        hits.append(sf.name)
    return hits


def _sibling_worktrees_with_dirty_path(target: str) -> list[str]:
    """Return list of sibling worktree paths that have the target file dirty."""
    try:
        out = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode != 0:
            return []
    except Exception:
        return []
    worktrees: list[str] = []
    current_wt = None
    try:
        cur = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if cur.returncode == 0:
            current_wt = Path(cur.stdout.strip()).resolve()
    except Exception:
        pass
    for line in out.stdout.splitlines():
        if line.startswith("worktree "):
            wt = Path(line[len("worktree "):].strip()).resolve()
            if current_wt and wt == current_wt:
                continue
            worktrees.append(str(wt))
    dirty: list[str] = []
    for wt in worktrees:
        try:
            st = subprocess.run(
                ["git", "-C", wt, "status", "--porcelain", "--", target],
                capture_output=True, text=True, timeout=3, check=False,
            )
            if st.returncode == 0 and st.stdout.strip():
                dirty.append(wt)
        except Exception:
            continue
    return dirty


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    tool_name = event.get("tool_name", "")
    if tool_name != "Bash":
        sys.exit(0)

    command = event.get("tool_input", {}).get("command", "") or ""
    # Only fire on git add / git commit (not log, status, diff, etc.)
    if not re.search(r"\bgit\s+(add|commit)\b", command):
        sys.exit(0)

    # Quick rejection: if the command carries the explicit ACK suffix, the
    # user has already been warned once and is overriding.
    if "--shared-state-ack" in command:
        sys.exit(0)

    paths = _extract_paths(command)
    if not paths:
        # Maybe a generic `git commit -m "..."` without explicit paths. In that
        # case staging happened earlier; we cannot know what's staged without a
        # git call.
        if re.search(r"\bgit\s+commit\b", command):
            try:
                st = subprocess.run(
                    ["git", "diff", "--cached", "--name-only"],
                    capture_output=True, text=True, timeout=3, check=False,
                )
                if st.returncode == 0:
                    for line in st.stdout.splitlines():
                        normalized = line.strip().replace("\\", "/")
                        for d in SHARED_STATE_DIRS:
                            if (
                                normalized.startswith(d)
                                and not any(normalized.startswith(s) for s in SAFE_SUBPATHS_PREFIX)
                            ):
                                paths.append(normalized)
            except Exception:
                pass
        if not paths:
            sys.exit(0)

    git_dir = _git_dir()
    if git_dir is None:
        sys.exit(0)

    # Resolve project_root from the current git worktree (respects CWD), with
    # the hook-file-location as the fallback. This lets the hook work both
    # under tests (synthetic repo via CWD) and under real Claude invocations.
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if top.returncode == 0 and top.stdout.strip():
            project_root = Path(top.stdout.strip()).resolve()
        else:
            project_root = Path(__file__).resolve().parents[2]
    except Exception:
        project_root = Path(__file__).resolve().parents[2]

    lock = _read_session_lock(git_dir)

    findings: list[str] = []

    # Check 1: sibling commits since session start.
    # Prefer the canonical lock field (`head_at_start`, added 2026-05-21).
    # Skip Check 1 entirely if the lock predates this field AND is >24h old —
    # the reflog fallback would produce false-positive floods (every commit
    # in a multi-day-stale lock window would count as "sibling").
    head_at_start = (lock or {}).get("head_at_start") or (lock or {}).get("head_sha")
    if not head_at_start:
        iso_started = (lock or {}).get("iso_started")
        if iso_started:
            try:
                from datetime import datetime as _dt, timezone as _tz
                started = _dt.fromisoformat(iso_started.replace("Z", "+00:00"))
                age_hours = (_dt.now(_tz.utc) - started).total_seconds() / 3600
                if age_hours <= 24:
                    out = subprocess.run(
                        ["git", "rev-parse", f"HEAD@{{{iso_started}}}"],
                        capture_output=True, text=True, timeout=3, check=False,
                    )
                    if out.returncode == 0 and out.stdout.strip():
                        head_at_start = out.stdout.strip()
            except Exception:
                head_at_start = None
    if head_at_start:
        new = _commits_since(head_at_start, exclude_author_email=_current_author_email())
        if new:
            findings.append(
                f"  - Sibling commits (different author) since session start "
                f"({head_at_start[:10]}): {len(new)}"
            )
            for line in new[:3]:
                findings.append(f"      {line[:90]}")

    # Check 2: peer scope_lock claims on the target paths
    for p in paths:
        claimants = _stage_files_claiming_path(project_root, p)
        if claimants:
            findings.append(f"  - Active stage(s) claim scope_lock on {p}:")
            for c in claimants:
                findings.append(f"      docs/runtime/stages/{c}")

    # Check 3: sibling worktree dirty on the same path
    for p in paths:
        dirty_wts = _sibling_worktrees_with_dirty_path(p)
        if dirty_wts:
            findings.append(f"  - Sibling worktree(s) have dirty {p}:")
            for wt in dirty_wts:
                findings.append(f"      {wt}")

    if not findings:
        sys.exit(0)

    print("", file=sys.stderr)
    print("  ====================================================================", file=sys.stderr)
    print("  SHARED-STATE COMMIT GUARD: peer-coordination check FOUND signals.", file=sys.stderr)
    print("  --------------------------------------------------------------------", file=sys.stderr)
    print(f"  Target paths: {', '.join(paths)}", file=sys.stderr)
    print("  Findings:", file=sys.stderr)
    for f in findings:
        print(f, file=sys.stderr)
    print("  --------------------------------------------------------------------", file=sys.stderr)
    print("  Per .claude/rules/multi-terminal-shared-file-hygiene.md:", file=sys.stderr)
    print("    1. Re-fetch + log: 'git fetch origin && git log --oneline -5'", file=sys.stderr)
    print("    2. Confirm with peer terminal whether they own this file", file=sys.stderr)
    print("    3. Categorize provenance of the diff (smoke run vs drift-loop side effect)", file=sys.stderr)
    print("    4. If safe to proceed, re-run the command with '--shared-state-ack'", file=sys.stderr)
    print("       appended (the hook strips this flag-marker; git sees it as garbage", file=sys.stderr)
    print("       and will reject — so put the ack in a separate echo / comment first):", file=sys.stderr)
    print("         git commit ... # --shared-state-ack", file=sys.stderr)
    print("  ====================================================================", file=sys.stderr)
    print("", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
