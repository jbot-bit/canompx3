#!/usr/bin/env python3
"""Session start hook: inject a concise workspace brief on entry/reset."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress task_route_packet's _ensure_repo_python() respawn when imported from
# hook context. The respawn passes argv[1:] which (for hooks) is [] — that makes
# the CLI run in --clear mode and raise SystemExit, killing the hook silently.
# Hooks only need read-only file access; they don't need venv-specific deps.
os.environ.setdefault("CANOMPX3_BOOTSTRAP_DONE", "1")

try:
    from scripts.tools.claude_superpower_brief import build_brief
except BaseException:  # pragma: no cover - hook fallback path (catches SystemExit too)
    build_brief = None

try:
    from scripts.tools.task_route_packet import read_task_route_packet
except BaseException:  # pragma: no cover - hook fallback path (catches SystemExit too)
    read_task_route_packet = None


def _legacy_startup_lines() -> list[str]:
    lines = ["NEW SESSION — Auto-orientation:"]

    # Read all stage files (stages/*.md + legacy STAGE_STATE.md)
    stages_dir = PROJECT_ROOT / "docs" / "runtime" / "stages"
    legacy_file = PROJECT_ROOT / "docs" / "runtime" / "STAGE_STATE.md"
    found_any = False

    if stages_dir.is_dir():
        for sf in sorted(stages_dir.glob("*.md")):
            if sf.name == ".gitkeep":
                continue
            content = sf.read_text(encoding="utf-8")
            for field in ("mode", "task"):
                for line in content.splitlines():
                    if line.strip().startswith(f"{field}:"):
                        lines.append(f"  Active stage [{sf.stem}]: {line.strip()}")
                        found_any = True
                        break

    if legacy_file.exists():
        content = legacy_file.read_text(encoding="utf-8")
        for field in ("mode", "task"):
            for line in content.splitlines():
                if line.strip().startswith(f"{field}:"):
                    lines.append(f"  Active stage [legacy]: {line.strip()}")
                    found_any = True
                    break

    if not found_any:
        lines.append("  No active stage.")

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.stdout.strip():
            files = result.stdout.strip().splitlines()
            lines.append(f"  Uncommitted: {len(files)} files")
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-3"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.stdout.strip():
            lines.append("  Recent commits:")
            lines.extend(f"    {commit}" for commit in result.stdout.strip().splitlines())
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return lines


def _superpower_lines(mode: str) -> list[str]:
    if build_brief is None:
        return []
    try:
        return build_brief(root=PROJECT_ROOT, mode=mode).splitlines()
    except Exception:
        return []


def _task_route_lines() -> list[str]:
    if read_task_route_packet is None:
        return []
    try:
        return read_task_route_packet(PROJECT_ROOT)
    except Exception:
        return []


def _git(args: list[str], timeout: int = 5) -> tuple[int, str]:
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return r.returncode, r.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return 1, ""


def _origin_drift_lines() -> list[str]:
    """Fetch, then auto-fast-forward when safe; otherwise warn.

    Safe-FF preconditions (ALL must hold):
      1. Working tree is clean (no modified, no untracked-blocking files).
      2. Local has 0 ahead commits vs origin/main.
      3. Current branch tracks origin/main.

    When all three hold, run `git pull --ff-only`. No data loss possible:
    we are replaying remote commits onto identical local state.

    When any fails, fall back to warn-only with a specific next step.
    Never auto-rebases or auto-merges (risk of clobbering parallel sessions).
    """
    rc_fetch, _ = _git(["fetch", "origin", "--quiet"], timeout=10)
    if rc_fetch != 0:
        return ["  Origin: fetch failed (offline?) — skipping drift check"]

    rc_branch, branch = _git(["branch", "--show-current"])
    if rc_branch != 0:
        return []
    if not branch:
        return ["  Origin: detached HEAD — checkout a branch before committing"]

    rc_count, count_out = _git(["rev-list", "--left-right", "--count", "HEAD...origin/main"])
    if rc_count != 0 or not count_out:
        return []
    try:
        ahead_n, behind_n = (int(x) for x in count_out.split())
    except ValueError:
        return []

    if ahead_n == 0 and behind_n == 0:
        return ["  Origin: in sync with origin/main"]

    rc_status, status_out = _git(["status", "--porcelain"])
    dirty = rc_status == 0 and bool(status_out)

    can_ff = branch == "main" and ahead_n == 0 and behind_n > 0 and not dirty
    if can_ff:
        rc_pull, _ = _git(["pull", "--ff-only", "origin", "main"], timeout=15)
        if rc_pull == 0:
            return [f"  Origin: auto-fast-forwarded {behind_n} commit(s) from origin/main"]
        return [f"  Origin: {behind_n} behind on main, ff-pull failed — run `git pull --ff-only` manually"]

    parts = []
    if behind_n:
        parts.append(f"{behind_n} behind")
    if ahead_n:
        parts.append(f"{ahead_n} ahead")
    state = ", ".join(parts)

    if branch != "main":
        guidance = f"on branch `{branch}` — verify base before pushing"
    elif ahead_n > 0 and behind_n > 0:
        guidance = "diverged — rebase ahead commits onto origin/main on a fresh branch"
    elif dirty and behind_n > 0:
        guidance = "dirty working tree blocks ff-pull — stash/commit WIP, then pull"
    elif ahead_n > 0 and behind_n == 0:
        guidance = "unpushed commits on main — push when ready"
    else:
        guidance = "branch from origin/main per .claude/rules/branch-discipline.md"

    return [f"  Origin: {state} vs origin/main — {guidance}"]


def _env_drift_lines() -> list[str]:
    """Detect venv drift vs uv.lock. Read-only; never modifies env.

    `uv sync --frozen --check` prints a diff when the venv has drifted from
    the lock and "Would make no changes" when in sync. Silent drift (caused
    by raw `pip install` or `uv pip install` against the venv) is the #1
    cause of irreproducible backtests — surfacing it at session start makes
    it loud instead of buried.
    """
    if not (PROJECT_ROOT / "uv.lock").exists():
        return []
    try:
        r = subprocess.run(
            ["uv", "sync", "--frozen", "--check"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    output = r.stdout + r.stderr
    if "Would make no changes" in output:
        return ["  Env: in sync with uv.lock"]

    drift_count = sum(1 for line in output.splitlines() if line.startswith(" - "))
    if drift_count == 0:
        return []
    return [f"  Env: {drift_count} pkg(s) drifted from uv.lock — run `uv sync --frozen` to repair"]


def _git_dir() -> Path | None:
    """Return per-worktree git directory (`.git/` for main, `.git/worktrees/<name>/`
    for linked worktrees). Used for per-worktree state files.
    """
    rc, out = _git(["rev-parse", "--git-dir"])
    if rc != 0 or not out:
        return None
    p = Path(out.strip())
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def _git_common_dir() -> Path | None:
    """Return per-repo git common directory, shared across all worktrees.

    Differs from `_git_dir()`: linked worktrees have their own `.git/worktrees/<name>/`,
    but the COMMON dir is the repo's main `.git/`. Use this for repo-state caches
    that all worktrees should share (e.g. CI status of `main` is repo state, not
    worktree state — without this, every worktree would do its own gh API call).
    """
    rc, out = _git(["rev-parse", "--git-common-dir"])
    if rc != 0 or not out:
        return None
    p = Path(out.strip())
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


_MAIN_CI_CACHE_TTL_SECS = 300  # 5 min — bounds API hits without staling beyond useful


def _format_ci_line(payload: dict) -> list[str]:
    """Render a cached or fresh CI payload to one stderr line. Pure function:
    no side effects, deterministic on input. Kept separate from
    `_main_ci_status_lines` so the cache-hit and cache-miss paths share a
    single source of truth for the output format.
    """
    conclusion = payload.get("conclusion", "unknown")
    if conclusion == "success":
        return ["  Main CI: green"]
    if conclusion == "failure":
        run_id = payload.get("run_id", "?")
        workflow = payload.get("workflow", "?")
        return [f"  Main CI: RED on run {run_id} ({workflow}) — verify before PR work; `gh run view {run_id}`"]
    return [f"  Main CI: last completed run = {conclusion}"]


def _main_ci_status_lines() -> list[str]:
    """Surface main-branch CI status at session start to prevent the
    PR-#108-class cascade where work is sunk into a PR before discovering
    `main` is red and downstream PRs inherit the break.

    Mechanism: query `gh run list --branch main --limit 1 --status completed`
    for the conclusion of the most recent COMPLETED run on origin/main. The
    `--status completed` filter excludes in-progress runs so we always report
    the last verdict. Cache the result repo-wide via `git-common-dir` (NOT
    per-worktree — CI status is repo state) for `_MAIN_CI_CACHE_TTL_SECS` to
    bound the API hit rate to roughly 12/hour/worktree even on aggressive
    session-flipping.

    Atomic cache write via `tempfile.NamedTemporaryFile` + `os.replace` so
    no interrupt mid-write can leave a partial JSON file.

    Silent on every failure path (offline, gh missing, gh unauth, no GitHub
    remote, parse error, timeout). This matches the offline-tolerance
    pattern of every other helper in this module. Anthropic's 2026 hooks
    documentation publishes this exact pattern (curl + warn-don't-block) as
    a sanctioned SessionStart template.
    """
    common_dir = _git_common_dir()
    if common_dir is None:
        return []
    cache_path = common_dir / ".claude.main-ci-status"
    now = int(time.time())

    # Cache hit path: read cached result if fresh.
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and now - int(cached.get("timestamp", 0)) < _MAIN_CI_CACHE_TTL_SECS:
            return _format_ci_line(cached)
    except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
        pass  # fall through to fresh fetch

    # Cache miss path: query gh CLI.
    try:
        r = subprocess.run(
            [
                "gh",
                "run",
                "list",
                "--branch",
                "main",
                "--limit",
                "1",
                "--status",
                "completed",
                "--json",
                "conclusion,databaseId,name",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return []

    if r.returncode != 0:
        return []  # gh unauth, no remote, network failure — silent skip

    try:
        runs = json.loads(r.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(runs, list) or not runs:
        return []  # no completed runs on main yet
    run = runs[0]
    if not isinstance(run, dict):
        return []

    payload = {
        "timestamp": now,
        "conclusion": run.get("conclusion", "unknown"),
        "run_id": run.get("databaseId", 0),
        "workflow": run.get("name", "?"),
    }

    # Atomic cache write: write to temp file in same dir, then atomic rename.
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(common_dir),
            prefix=".claude.main-ci-status.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            json.dump(payload, tmp)
            tmp_path = tmp.name
        os.replace(tmp_path, cache_path)
    except OSError:
        pass  # cache write failure is non-fatal — we still report this fetch's result

    return _format_ci_line(payload)


def _session_lock_lines() -> tuple[list[str], bool]:
    """Detect concurrent Claude sessions in THIS worktree. Hard-block if found.

    Mechanism: write a PID file at `<git-dir>/.claude.pid` on session startup.
    On the next startup, if the file already exists, BLOCK with cleanup
    instructions (returns block=True so main() exits non-zero).

    Cleanup is intentionally MANUAL: if Claude crashed without releasing the
    lock, the user deletes the file. Auto-cleanup via PID liveness was
    considered and rejected — Windows process trees (Claude → bash → python
    hook) make `os.getppid()` ambiguous, and parsing `tasklist` for Claude's
    actual PID is brittle. Manual delete is the safe default; the BLOCK
    message includes the exact `rm` command.

    Why this exists: per `feedback_shared_worktree_concurrent_commits.md`,
    two Claude sessions in the SAME worktree corrupt `.git/index` and produce
    `cannot lock ref 'HEAD'` mid-commit. This is the dominant collision
    pattern (per 2026-04-26 incident) — clones don't fix it; only a mutex
    does. Aligns with 2026 industry consensus on AI-agent runtime isolation:
    git-level isolation is necessary but not sufficient; per-runtime locks
    are required to make parallel agents safe.

    Returns: (lines_to_print, should_block).
    """
    git_dir = _git_dir()
    if git_dir is None:
        return [], False  # not in a git repo — skip (no contention possible)

    lock_path = git_dir / ".claude.pid"
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "iso_started": datetime.now(UTC).isoformat(),
            "worktree": str(PROJECT_ROOT),
        },
        indent=2,
    ).encode("utf-8")

    # Atomic create-or-fail. O_EXCL closes the TOCTOU race that a check-then-
    # write pattern would have: two simultaneous startups can't both believe
    # they hold the lock.
    try:
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return [], False  # we own the lock
    except FileExistsError:
        pass  # conflict — fall through to BLOCK message
    except OSError as e:
        # Transient FS issue (perms, disk full, etc.). Warn loudly but don't
        # block: a write-side failure here would otherwise lock out every
        # future session, which is a worse failure mode than the contention
        # we're trying to prevent.
        return ([f"  WARNING: could not write Claude session lock at {lock_path}: {e}"], False)

    # Conflict path: read existing lock for the BLOCK message. Tolerate
    # corruption (treat as "unknown other session" — still safer to BLOCK).
    try:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        other_pid = existing.get("pid", "?")
        other_started = existing.get("iso_started", "?")
        other_worktree = existing.get("worktree", "?")
    except (json.JSONDecodeError, OSError, ValueError):
        other_pid = "(corrupted lock file)"
        other_started = "?"
        other_worktree = "?"

    return (
        [
            "",
            "  ====================================================================",
            "  BLOCKED: Another Claude session is active in this worktree.",
            "  --------------------------------------------------------------------",
            f"  Active PID:   {other_pid}",
            f"  Started:      {other_started}",
            f"  Worktree:     {other_worktree}",
            f"  Lock file:    {lock_path}",
            "  --------------------------------------------------------------------",
            "  Two Claudes in one worktree corrupt .git/index. See:",
            "  memory/feedback_shared_worktree_concurrent_commits.md",
            "",
            "  Resolutions (pick one):",
            "    1. Switch to the other session and continue there.",
            "    2. If that session is dead, run:",
            f"         rm '{lock_path}'",
            "       (manual delete — PID liveness is unreliable across Windows",
            "       process trees, so we don't auto-clean.)",
            "    3. Spawn a fresh worktree:",
            "         scripts/tools/new_session.sh",
            "  ====================================================================",
            "",
        ],
        True,
    )


def _action_queue_ready_lines() -> list[str]:
    """Surface action-queue items with status:ready as a 1-line nudge.

    Prevents the 2026-04-26 stale-status-ready miss (PR #140 follow-up): a
    P1 ready item sat for 2+ days because no startup signal surfaced it.

    Delegates parsing to `pipeline.work_queue.load_queue()` (canonical source
    per institutional-rigor §4 — never re-encode YAML schema logic).

    Returns: list of lines (typically 0 or 1). On any failure (missing file,
    malformed YAML, import error, schema drift), returns [] — must never
    block session start.
    """
    try:
        from pipeline.work_queue import load_queue
    except BaseException:  # pragma: no cover - hook fallback path
        return []
    try:
        queue = load_queue(PROJECT_ROOT)
    except BaseException:  # pragma: no cover - hook fallback path
        return []

    ready_ids = [item.id for item in queue.items if item.status == "ready"]
    if not ready_ids:
        return []
    return [f"  Action queue READY: {', '.join(ready_ids)}"]


def _parallel_session_lines() -> list[str]:
    """Detect other active worktrees and warn on cross-session collision risk.

    Reports each worktree (other than the current one), its branch, and whether
    it has uncommitted changes. If 2+ worktrees are dirty simultaneously this
    is the documented "open 2 terminals and start working" failure mode — it
    causes CRLF noise, lost stashes, and merge conflicts.

    Output is informational at start, escalating to a warning when 2+ dirty
    worktrees coexist.
    """
    rc, out = _git(["worktree", "list", "--porcelain"])
    if rc != 0 or not out:
        return []

    rc_pwd, current_path = _git(["rev-parse", "--show-toplevel"])
    if rc_pwd != 0:
        return []
    current_path = current_path.strip()

    # Parse worktree blocks: each starts with `worktree <path>`
    worktrees: list[dict[str, str]] = []
    block: dict[str, str] = {}
    for line in out.splitlines():
        if not line.strip():
            if block:
                worktrees.append(block)
                block = {}
        elif line.startswith("worktree "):
            block["path"] = line[len("worktree ") :].strip()
        elif line.startswith("branch "):
            block["branch"] = line[len("branch ") :].strip().replace("refs/heads/", "")
        elif line.startswith("HEAD "):
            block["head"] = line[len("HEAD ") :].strip()[:8]
    if block:
        worktrees.append(block)

    others = [w for w in worktrees if w.get("path") and w["path"] != current_path]
    if not others:
        return []

    # Check dirtiness of each other worktree
    lines = [f"  Parallel worktrees: {len(others)} other active"]
    dirty_count = 0
    for w in others[:6]:  # cap noise
        wt_path = w.get("path", "?")
        wt_branch = w.get("branch", w.get("head", "?"))
        rc_st, st = _git(["-C", wt_path, "status", "--porcelain"], timeout=3)
        is_dirty = rc_st == 0 and bool(st.strip())
        if is_dirty:
            dirty_count += 1
        marker = " [DIRTY]" if is_dirty else ""
        # Trim path for display
        display_path = wt_path.split("/")[-1] if "/" in wt_path else wt_path
        lines.append(f"    - {display_path} on {wt_branch}{marker}")

    rc_self_st, self_st = _git(["status", "--porcelain"])
    self_dirty = rc_self_st == 0 and bool(self_st.strip())

    if self_dirty and dirty_count >= 1:
        lines.append(f"  WARNING: {dirty_count + 1} dirty worktrees active — edit collision/CRLF/stash-loss risk.")
        lines.append("  Each Claude session should work in its own worktree. Spawn one: scripts/tools/new_session.sh")
    return lines


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    session_type = event.get("session_type", "startup")
    lines: list[str] = []
    task_route_lines = _task_route_lines()

    if session_type == "startup":
        # Hard-block FIRST if another Claude session holds the worktree lock.
        # Doing this before any context generation prevents the user from
        # accidentally starting work in a soon-to-be-corrupted state.
        block_lines, should_block = _session_lock_lines()
        if should_block:
            print("\n".join(block_lines), file=sys.stderr)
            sys.exit(2)
        lines = task_route_lines or _superpower_lines("session-start") or _legacy_startup_lines()
        if block_lines:  # warning lines (e.g., lock-write failure) — surface them
            lines.extend(block_lines)
    elif session_type == "resume":
        if task_route_lines:
            lines = ["RESUMED SESSION — Task route restored:"]
            lines.extend(task_route_lines)
        else:
            lines = ["RESUMED SESSION — Re-grounding context:"]
            lines.extend(_superpower_lines("interactive") or ["Check HANDOFF.md for last known state."])
    elif session_type == "compact":
        pass
    elif session_type == "clear":
        if task_route_lines:
            lines = ["CONTEXT CLEARED — Task route restored:"]
            lines.extend(task_route_lines)
        else:
            lines = ["CONTEXT CLEARED — Re-grounding context:"]
            lines.extend(
                _superpower_lines("interactive") or ["Re-read docs/runtime/stages/*.md if active work exists."]
            )

    if lines:
        lines.extend(_origin_drift_lines())
        lines.extend(_env_drift_lines())
        lines.extend(_action_queue_ready_lines())
        lines.extend(_parallel_session_lines())
        lines.extend(_main_ci_status_lines())
        print("\n".join(lines), file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
