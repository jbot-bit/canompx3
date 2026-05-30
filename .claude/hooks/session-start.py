#!/usr/bin/env python3
"""Session start hook: inject a concise workspace brief on entry/reset."""

from __future__ import annotations

import json
import os
import re
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

try:
    from pipeline.system_brief import build_system_brief
except BaseException:  # pragma: no cover - hook fallback path (catches SystemExit too)
    build_system_brief = None


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


# Stale-lock auto-recovery threshold. Locks held by a dead PID are reclaimed
# only after this many hours, so a momentarily-dead-then-restarted session
# (e.g. /clear) cannot accidentally reclaim a fresh sibling's lock. The 12h
# floor matches the documented "leave it overnight" pattern; the dead-PID
# requirement is the actual safety property. Origin: 2026-05-14 incident
# where a 3-day-stale lock from a dead PID silently disabled mutex protection
# while two concurrent live Claude sessions wrote to the same .git/index.
_STALE_LOCK_RECLAIM_HOURS = 12


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform best-effort PID liveness check.

    POSIX: ``os.kill(pid, 0)`` raises ProcessLookupError on dead PIDs and
    PermissionError on PIDs that are alive but owned by another user. Both
    "alive" and "permission denied" mean "do not reclaim the lock."

    Windows: ``os.kill`` is partially supported. Use it when available;
    fall back to a conservative True (assume alive) on any error so we
    NEVER reclaim a lock we cannot prove dead. This deliberately favours
    false BLOCKs over false reclaims — clobbering a live session's lock
    is the catastrophic failure mode we are guarding against.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # PID exists, owned by another user — treat as alive
    except (OSError, AttributeError):
        return True  # conservative: cannot prove dead, do not reclaim


def _lock_age_hours(iso_started: str) -> float | None:
    """Hours since ``iso_started`` per the SAME-class clock the writer used.

    Uses ``hook.datetime`` rather than a direct import so the test suite
    can monkey-patch a pinned clock; using a pinned clock for tests is the
    only way to deterministically exercise the stale-recovery branch.
    Returns None when the timestamp is unparseable (treat as not-stale).
    """
    try:
        # Strip a trailing Z for fromisoformat tolerance.
        ts = datetime.fromisoformat(iso_started.replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - ts
    return delta.total_seconds() / 3600.0


def _acquire_lock_fd(lock_path: Path) -> int:
    # Atomic create-or-fail. O_EXCL closes the TOCTOU race that a check-then-
    # write pattern would have: two simultaneous startups can't both believe
    # they hold the lock.
    #
    # Extracted as a seam so tests can simulate FS errors (ENOSPC, EACCES)
    # by patching this single function instead of the singleton ``os.open``.
    return os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)


def _session_lock_lines() -> tuple[list[str], bool]:
    """Detect concurrent Claude sessions in THIS worktree. Hard-block if found.

    Mechanism: write a PID file at `<git-dir>/.claude.pid` on session startup.
    On the next startup, if the file already exists, check whether the holder
    is still alive (``_pid_is_alive``); reclaim the lock atomically when it is
    proven dead AND older than ``_STALE_LOCK_RECLAIM_HOURS`` hours. Otherwise
    BLOCK with cleanup instructions (returns block=True so main() exits
    non-zero).

    Reclaim conditions are strict by design:
      - PID must be PROVEN dead (not "could not check" — see _pid_is_alive)
      - Lock must be older than _STALE_LOCK_RECLAIM_HOURS hours
    Either failing -> BLOCK. This deliberately favours false BLOCKs over
    false reclaims; clobbering a live session's lock is the catastrophic
    failure mode we are guarding against.

    Why this exists: per `feedback_shared_worktree_concurrent_commits.md`,
    two Claude sessions in the SAME worktree corrupt `.git/index` and produce
    `cannot lock ref 'HEAD'` mid-commit. This was reproduced 2026-05-14 when
    a 3-day-stale lock from a dead PID silently disabled mutex protection
    while two concurrent live Claude sessions wrote to the same index — one
    session's `git reset HEAD <file>` un-staged the other's mid-commit work.

    Returns: (lines_to_print, should_block).
    """
    git_dir = _git_dir()
    if git_dir is None:
        return [], False  # not in a git repo — skip (no contention possible)

    lock_path = git_dir / ".claude.pid"
    _, current_branch = _git(["branch", "--show-current"])
    _, head_sha = _git(["rev-parse", "HEAD"])
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "iso_started": datetime.now(UTC).isoformat(),
            "worktree": str(PROJECT_ROOT),
            "branch_at_start": current_branch.strip(),
            "head_at_start": head_sha.strip(),
        },
        indent=2,
    ).encode("utf-8")

    try:
        fd = _acquire_lock_fd(lock_path)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        return [], False  # we own the lock
    except FileExistsError:
        # Lock exists. Could be: (a) another live session, (b) stale lock from a
        # crashed session, or (c) THIS session's own lock from a previous start
        # (e.g. after /clear). Read it and check.
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            existing = {}
        # Guard health check: if the existing lock is for THIS worktree but has
        # no branch_at_start (pre-Phase-1 lock or corrupted write), the branch-
        # flip-guard exits 0 silently — silence #2 from the gap audit. Surface
        # that the guard is inactive and tell the user how to restore it.
        # Worktree match (not PID) is the right comparator: PID liveness is
        # unreliable across Windows process trees per the existing comments.
        if existing.get("worktree") == str(PROJECT_ROOT) and not existing.get("branch_at_start"):
            return (
                [
                    "  WARNING: branch-flip-guard inactive — session lock predates the",
                    f"    branch_at_start field. Run:  rm '{lock_path}'  then restart",
                    "    this session to enable mid-session branch-flip detection.",
                ],
                False,
            )

        # Stale-lock auto-recovery: reclaim ONLY when the holder PID is proven
        # dead AND the lock is older than _STALE_LOCK_RECLAIM_HOURS. Both must
        # hold; either alone is insufficient (a live PID at any age must
        # block; a dead PID within the freshness window may be a /clear-and-
        # restart of the same shell and should still block to surface the
        # transition).
        held_pid = existing.get("pid")
        held_iso = existing.get("iso_started", "")
        age_hours = _lock_age_hours(held_iso)
        is_dead = isinstance(held_pid, int) and not _pid_is_alive(held_pid)
        is_old_enough = age_hours is not None and age_hours >= _STALE_LOCK_RECLAIM_HOURS
        if is_dead and is_old_enough:
            try:
                # Atomic replace via write-to-temp + os.replace (POSIX rename(2)
                # semantics; on Windows NTFS, os.replace uses MoveFileExW with
                # MOVEFILE_REPLACE_EXISTING which is atomic within the same
                # volume). O_WRONLY|O_TRUNC was NOT atomic: two simultaneous
                # restarts could both truncate and overwrite, reintroducing the
                # race the mutex guards against.
                tmp_path = lock_path.with_suffix(".pid.tmp")
                fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
                try:
                    os.write(fd, payload)
                finally:
                    os.close(fd)
                os.replace(str(tmp_path), str(lock_path))
            except OSError as exc:
                return (
                    [
                        f"  WARNING: failed to reclaim stale Claude lock at {lock_path}: {exc}",
                        f"    Holder PID {held_pid} is dead and lock is {age_hours:.1f}h old.",
                        f"    Manual recovery: rm '{lock_path}' and restart.",
                    ],
                    False,
                )
            return (
                [
                    f"  Recovered stale session lock: PID {held_pid} dead, age {age_hours:.1f}h",
                    f"    (threshold: PID must be dead AND age >= {_STALE_LOCK_RECLAIM_HOURS}h)",
                ],
                False,
            )

        pass  # genuine conflict — fall through to BLOCK message
    except OSError as e:
        # Transient FS issue (perms, disk full, etc.). Warn loudly but don't
        # block: a write-side failure here would otherwise lock out every
        # future session, which is a worse failure mode than the contention
        # we're trying to prevent.
        return ([f"  WARNING: could not write Claude session lock at {lock_path}: {e}"], False)

    # Conflict path: read existing lock for the BLOCK message.
    #
    # Two distinct failure modes converge here and DESERVE distinct messages:
    #   (a) the lock parses → a real (possibly live) concurrent session.
    #   (b) the lock is present but UNPARSEABLE (malformed JSON) → almost
    #       always a stale/corrupt lock, NOT a concurrent session. The live
    #       2026-05-24 incident was an older writer emitting unescaped Windows
    #       backslashes (`"worktree": "C:\Users\..."` → invalid \U \j \c
    #       escapes), which `json.loads` rejects. Critically, a malformed lock
    #       also SILENTLY disables the per-tool branch-flip / head-flip guards
    #       (they read `branch_at_start` / `head_at_start` via `_branch_state`,
    #       which return None on parse failure and exit 0). The operator must
    #       be told the guards are inactive — the old "Another session is
    #       active" framing sent them chasing a phantom concurrent session.
    #
    # We still BLOCK in BOTH cases: a corrupt lock cannot prove no live session
    # exists, and blocking-conservatively is the mutex contract
    # (`test_corrupted_lock_still_blocks_and_degrades_gracefully`). The fix is
    # message ACCURACY, not the block decision.
    try:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_malformed = False
    except (json.JSONDecodeError, OSError, ValueError):
        existing = {}
        lock_malformed = True

    if lock_malformed:
        return (
            [
                "",
                "  ====================================================================",
                "  BLOCKED: session lock is malformed (unparseable JSON).",
                "  --------------------------------------------------------------------",
                "  Branch-flip and head-flip guards are INACTIVE — they read",
                "  branch_at_start / head_at_start from this lock and fail open on",
                "  a parse error. No live session could be confirmed; this is almost",
                "  certainly a stale/corrupt lock, NOT a concurrent session.",
                f"  Lock file:    {lock_path}",
                "  --------------------------------------------------------------------",
                "  Common cause: an older writer emitted unescaped Windows paths",
                "  (e.g. \"worktree\": \"C:\\Users\\...\") — invalid JSON escapes.",
                "",
                "  Recover (restores guards):",
                f"    1. rm '{lock_path}'",
                "    2. Restart this session — session-start rewrites a clean,",
                "       properly-escaped lock and re-enables the guards.",
                "  ====================================================================",
                "",
            ],
            True,
        )

    other_pid = existing.get("pid", "?")
    other_started = existing.get("iso_started", "?")
    other_worktree = existing.get("worktree", "?")
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


# A sibling worktree counts as ACTIVELY HOT if it has uncommitted changes AND a
# tracked file modified within this many seconds. mtime — not PID liveness — is
# the signal: PIDs go dead across Windows process trees constantly (the existing
# PID lock under-reports for exactly this reason), but a file touched two minutes
# ago in a dirty tree is an unambiguous live editor. Stale dirty worktrees
# abandoned hours/days ago fall outside the window and never trip the block.
_ACTIVE_SIBLING_WINDOW_SECS = 15 * 60


def _most_recent_tracked_mtime(wt_path: str, dirty_rel_paths: list[str]) -> float | None:
    """Return the newest mtime (epoch secs) among a worktree's dirty TRACKED files.

    Only the paths git reports as modified/added are stat'd — untracked churn
    (logs, .venv, generated artifacts) is excluded by the caller so background
    file writes can't masquerade as a live editor. Returns None if nothing
    could be stat'd (all paths vanished / unreadable).
    """
    newest: float | None = None
    for rel in dirty_rel_paths:
        full = os.path.join(wt_path, rel)
        try:
            m = os.stat(full).st_mtime
        except OSError:
            continue  # file vanished mid-scan (rename/delete) — skip, don't fail
        if newest is None or m > newest:
            newest = m
    return newest


def _dirty_tracked_paths(wt_path: str) -> list[str]:
    """Relative paths of TRACKED files with uncommitted changes in a worktree.

    Uses `--untracked-files=no` so brand-new untracked files (the background-
    writer false-positive class) are ignored — only edits to files git already
    knows about count as a human/agent actively working.
    """
    rc, out = _git(["-C", wt_path, "status", "--porcelain", "--untracked-files=no"], timeout=3)
    if rc != 0 or not out:
        return []
    paths: list[str] = []
    for line in out.splitlines():
        # Porcelain v1: 2 status chars, a space, then the path (or "orig -> new").
        if len(line) < 4:
            continue
        rel = line[3:]
        if " -> " in rel:  # rename: take the destination
            rel = rel.split(" -> ", 1)[1]
        rel = rel.strip().strip('"')
        if rel:
            paths.append(rel)
    return paths


def _active_sibling_lines(now_epoch: float | None = None) -> tuple[list[str], bool]:
    """Hard-block when another worktree of this repo is ACTIVELY HOT.

    The PID lock (`_session_lock_lines`) guards two sessions in the SAME tree but
    relies on PID liveness, which is unreliable on Windows. This complements it
    with a mtime signal across ALL worktrees: if a sibling has uncommitted
    changes to a tracked file touched within `_ACTIVE_SIBLING_WINDOW_SECS`, a
    live session is almost certainly working there right now — starting a second
    session risks the index-corruption / lost-stash class this repo has hit
    repeatedly (`feedback_multi_terminal_shared_file_thrash_2026_05_21.md`).

    Returns (lines, should_block). Hard-block (True) is reserved for the
    SAME-worktree case — two sessions in one tree is the corruption risk.
    A DIFFERENT hot worktree is the SANCTIONED parallel pattern
    (`scripts/tools/new_session.sh`), so it WARNS but never refuses.

    Override: set CLAUDE_ALLOW_CONCURRENT=1 to skip the block entirely.

    Fail-open on every error path — a guard that can't read state must never
    wedge session start.
    """
    if os.environ.get("CLAUDE_ALLOW_CONCURRENT") == "1":
        return [], False

    rc, out = _git(["worktree", "list", "--porcelain"])
    if rc != 0 or not out:
        return [], False

    rc_pwd, current_path = _git(["rev-parse", "--show-toplevel"])
    if rc_pwd != 0 or not current_path.strip():
        return [], False
    current_path = current_path.strip()

    # Parse worktree paths (porcelain blocks separated by blank lines).
    wt_paths: list[str] = []
    for line in out.splitlines():
        if line.startswith("worktree "):
            wt_paths.append(line[len("worktree ") :].strip())

    if now_epoch is None:
        now_epoch = time.time()

    hot_self: list[tuple[str, float]] = []
    hot_siblings: list[tuple[str, float]] = []
    for wt in wt_paths:
        if not wt:
            continue
        # Normalise both sides for the self/sibling comparison: git may report
        # forward slashes on Windows while show-toplevel agrees, but be defensive.
        is_self = os.path.normcase(os.path.normpath(wt)) == os.path.normcase(
            os.path.normpath(current_path)
        )
        dirty = _dirty_tracked_paths(wt)
        if not dirty:
            continue
        newest = _most_recent_tracked_mtime(wt, dirty)
        if newest is None:
            continue
        age = now_epoch - newest
        # Reject garbage / clock-skew: a file "modified in the future" or absurdly
        # old age is not a trustworthy active signal — don't block on it.
        if age < -60 or age > _ACTIVE_SIBLING_WINDOW_SECS:
            continue
        if is_self:
            hot_self.append((wt, age))
        else:
            hot_siblings.append((wt, age))

    # SAME worktree freshly hot but our PID lock didn't catch it (dead-PID
    # reclaim, pre-field lock, crashed session). This IS the corruption case.
    if hot_self:
        wt, age = hot_self[0]
        mins = int(age // 60)
        return (
            [
                "",
                "  ====================================================================",
                f"  BLOCKED: this worktree was edited < {max(mins, 1)} min ago by another session.",
                "  --------------------------------------------------------------------",
                f"  Worktree:  {wt}",
                "  Tracked files here changed minutes ago but no live PID lock holds",
                "  them — a crashed/parallel session in THIS tree. Two sessions in one",
                "  worktree corrupt .git/index (feedback_shared_worktree_concurrent_commits.md).",
                "",
                "  Resolutions (pick one):",
                "    1. Return to the other terminal working this tree and continue there.",
                "    2. If it's truly gone, commit/stash its WIP, then restart this session.",
                "    3. Work in your own tree:  scripts/tools/new_session.sh",
                "    4. Override (you're certain it's safe):  CLAUDE_ALLOW_CONCURRENT=1",
                "  ====================================================================",
                "",
            ],
            True,
        )

    if hot_siblings:
        lines = [
            "",
            "  --------------------------------------------------------------------",
            f"  NOTE: {len(hot_siblings)} other worktree(s) edited in the last "
            f"{_ACTIVE_SIBLING_WINDOW_SECS // 60} min — a live parallel session:",
        ]
        for wt, age in sorted(hot_siblings, key=lambda t: t[1])[:4]:
            mins = int(age // 60)
            label = "just now" if mins == 0 else f"{mins} min ago"
            lines.append(f"    - {wt}  (edited {label})")
        lines.append("  Parallel worktrees are fine; just don't open a 2nd session in any ONE")
        lines.append("  of them. Coordinate shared-state commits per multi-terminal-shared-file-hygiene.md.")
        lines.append("  --------------------------------------------------------------------")
        return lines, False

    return [], False


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


def _literature_corpus_lines() -> list[str]:
    """One-line nudge that the institutional literature corpus exists.

    Trimmed 2026-05-13 from a 5-line MCP-tool-name dump (~120 tokens) to a single
    count line (~25 tokens). MCP tool names are discoverable via `auto-skill-routing.md`
    and the research-catalog MCP server's own `list_literature_sources` — repeating
    them every session is pure noise after the first acquaintance. The count itself
    still load-bearing: it's the visible signal that the corpus exists.

    Fail-silent on every error path.
    """
    try:
        lit_dir = PROJECT_ROOT / "docs" / "institutional" / "literature"
        if not lit_dir.is_dir():
            return []
        extracts = [
            p for p in lit_dir.glob("*.md") if not p.name.startswith("PENDING_ACQUISITION")
        ]
        n = len(extracts)
        if n == 0:
            return []
        return [f"  Literature: {n} extracts (mcp__research-catalog__* to query)"]
    except BaseException:  # pragma: no cover - hook fallback path
        return []


def _handoff_next_step_line() -> list[str]:
    """Extract the first bullet from HANDOFF.md '## Next Steps — Active' as a resume cue.

    Per the intent-router-hooks plan: surfacing the operator's own pre-recorded
    "where to start" pointer at session start eliminates the cognitive tax of
    re-reading HANDOFF. Returns a single line (~80-char truncated bullet) or [].

    Fail-silent on every error path — missing file, missing section, malformed
    markdown, encoding error.
    """
    try:
        handoff = PROJECT_ROOT / "HANDOFF.md"
        if not handoff.exists():
            return []
        lines = handoff.read_text(encoding="utf-8").splitlines()
        in_section = False
        bullet_re = re.compile(r"^\s*\d+\.\s+(.*)$")
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith("## Next Steps"):
                in_section = True
                continue
            if in_section and (stripped.startswith("## ") or stripped.startswith("---")):
                break  # left the section without finding a numbered bullet
            if in_section:
                m = bullet_re.match(ln)
                if m:
                    body = m.group(1).strip()
                    # Strip leading bold markers for compactness.
                    body = re.sub(r"^\*\*(.+?)\*\*\s*:?\s*", r"\1: ", body)
                    snippet = body[:90].rstrip()
                    if len(body) > 90:
                        snippet += "…"
                    return [f"  Resume → /next  ·  top: {snippet}"]
        return []
    except BaseException:  # pragma: no cover - hook fallback path
        return []


def _startup_packet_lines() -> list[str]:
    """Compact summary of the repo-state startup packet (NUGGET 4).

    Calls `pipeline.system_brief.build_system_brief` in-process — the same
    function `repo-state` MCP `get_startup_packet` exposes. The function is
    engineered for hook-level usage (222ms typical vs 250ms internal budget)
    so we avoid the subprocess+timeout dance other helpers use.

    Surfaces only the highest-signal fields: blocking/warning issue counts,
    task kind, latency, budget compliance. Anything more is noise that
    competes with /orient on session start.

    Fail-silent on every error path. BaseException intentional — matches
    the surrounding hook idiom; this line is cosmetic and must never abort
    session start.
    """
    if build_system_brief is None:
        return []
    try:
        brief = build_system_brief(
            PROJECT_ROOT,
            task_text=None,
            task_id=None,
            briefing_level="read_only",
            context_name="generic",
            active_tool="session-start-hook",
            active_mode="read-only",
        )
    except BaseException:  # pragma: no cover - hook fallback path
        return []
    if not isinstance(brief, dict):
        return []

    blocking = brief.get("blocking_issues") or []
    warnings = brief.get("warning_issues") or []
    latency_ms = brief.get("startup_latency_ms")
    budget = brief.get("orientation_cost_budget")
    budget = budget if isinstance(budget, dict) else {}
    within = budget.get("within_budget")

    parts = []
    if isinstance(blocking, list) and blocking:
        parts.append(f"{len(blocking)} BLOCKING")
    if isinstance(warnings, list) and warnings:
        parts.append(f"{len(warnings)} warning")
    if not parts:
        parts.append("no blockers")
    if isinstance(latency_ms, (int, float)):
        budget_marker = "" if within is None else (" within budget" if within else " OVER BUDGET")
        parts.append(f"{int(latency_ms)}ms{budget_marker}")

    lines = [f"  Startup packet: {' | '.join(parts)}"]

    if isinstance(blocking, list):
        for issue in blocking[:2]:
            if isinstance(issue, dict):
                code = issue.get("code", "?")
                msg = str(issue.get("message", "")).strip().splitlines()[0][:120]
                lines.append(f"    BLOCK [{code}] {msg}")
    if isinstance(warnings, list):
        for issue in warnings[:2]:
            if isinstance(issue, dict):
                code = issue.get("code", "?")
                msg = str(issue.get("message", "")).strip().splitlines()[0][:120]
                lines.append(f"    warn  [{code}] {msg}")

    return lines


def _crg_context_lines() -> list[str]:
    """One-line CRG graph status: nodes, files, and freshness vs the working tree.

    Returns either:
      - "  Graph: <N> nodes / <F> files — fresh"   (last_updated within 7 days)
      - "  Graph: <N> nodes / <F> files — stale (<D>d old)"
      - "  Graph: missing — run `code-review-graph build`"
      - []                                          on any failure (fail-silent)

    Reads `.code-review-graph/graph.db` directly via the same find_repo_root
    semantics CRG uses, so worktree-local DBs are reported correctly.
    """
    try:
        import sqlite3
        from datetime import datetime as _dt

        # Resolve canonical CRG root: if this is a sibling worktree
        # (`canompx3-<descriptor>`) and the canonical `canompx3/` exists with
        # a `.code-review-graph/` dir, use the canonical sibling so all
        # worktrees report ONE shared graph rather than per-worktree
        # fragments. Mirrors `.githooks/pre-commit` step 3b sibling-detection
        # and `_crg_canonical_root()` in `.claude/hooks/post-edit-pipeline.py`.
        # Source-of-truth: `.githooks/pre-commit` lines 222–225. If that
        # logic changes, update both hook sites.
        crg_root = PROJECT_ROOT
        if PROJECT_ROOT.name.startswith("canompx3-"):
            sibling = PROJECT_ROOT.parent / "canompx3"
            if (sibling / ".code-review-graph").exists():
                crg_root = sibling

        db_path = crg_root / ".code-review-graph" / "graph.db"
        if not db_path.exists():
            return ["  Graph: missing — run `code-review-graph build`"]

        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        try:
            cur = conn.cursor()
            try:
                node_count = cur.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            except sqlite3.Error:
                node_count = 0
            try:
                file_count = cur.execute(
                    "SELECT COUNT(*) FROM nodes WHERE type='file'"
                ).fetchone()[0]
            except sqlite3.Error:
                file_count = 0
            last_updated = None
            try:
                row = cur.execute(
                    "SELECT value FROM metadata WHERE key='last_updated'"
                ).fetchone()
                if row and row[0]:
                    last_updated = row[0]
            except sqlite3.Error:
                last_updated = None
        finally:
            conn.close()

        freshness = "unknown"
        if last_updated:
            try:
                ts = _dt.fromisoformat(last_updated.replace("Z", ""))
                age_days = (_dt.now() - ts).days
                freshness = "fresh" if age_days <= 7 else f"stale ({age_days}d old)"
            except (ValueError, TypeError):
                freshness = "unknown"

        return [f"  Graph: {node_count} nodes / {file_count} files — {freshness}"]
    except BaseException:  # pragma: no cover - hook fallback path
        # BaseException (incl. KeyboardInterrupt/SystemExit) intentional —
        # this is a cosmetic startup line and must NEVER abort session start.
        return []


def _worktree_lease_lines(session_id: str = "") -> tuple[list[str], bool]:
    """Acquire (or detect peer hold of) the worktree concurrency lease.

    Delegates entirely to `scripts/tools/worktree_guard.py` — institutional-
    rigor §4 (no inline copies). Returns (lines, should_block).

    The lease is now keyed on (session_id, ppid)+heartbeat, NOT an OS file-lock
    (the old lock was held by an ephemeral hook subprocess and provided no
    mutual exclusion — see the canonical module docstring, n=2 incident
    2026-05-29/30). Passing `session_id` here is what lets a SECOND session in
    this worktree be recognised as a different owner and BLOCKED at startup,
    even when the peer's working tree is clean (e.g. mid git-surgery after a
    stash) — the exact case `_active_sibling_lines()`'s mtime signal misses.

    The 12h `_session_lock_lines()` PID mutex remains as a separate safety net
    for the longer-lived dead-lock incident class. Both coexist intentionally.
    Fail-open on any error.
    """
    try:
        import importlib.util as _ilu

        spec = _ilu.spec_from_file_location(
            "worktree_guard",
            str(PROJECT_ROOT / "scripts" / "tools" / "worktree_guard.py"),
        )
        if spec is None or spec.loader is None:
            return [], False
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        status_str, payload, msg = mod.acquire(PROJECT_ROOT, session_id=session_id)
    except BaseException:
        return [], False

    if status_str == "blocked":
        pl: dict = payload if isinstance(payload, dict) else {}
        pid = pl.get("pid", "?")
        sid = pl.get("session_id", "?")
        wt = pl.get("worktree", "?")
        lines = [
            "",
            "  ====================================================================",
            "  BLOCKED: another Claude session holds the worktree concurrency lease.",
            f"  Peer session {sid} (PID {pid}) on worktree {wt}",
            "  Recovery (after confirming peer is truly gone):",
            "    python scripts/tools/worktree_guard.py --force-release",
            "  Or spawn a fresh worktree: scripts/tools/new_session.sh",
            "  ====================================================================",
            "",
        ]
        return lines, True
    if status_str in ("acquired", "refreshed", "reclaimed"):
        return [f"  Worktree lease: {status_str} ({msg})"], False
    return [], False  # skipped / error -> fail-open silently


def _stale_process_reaper_lines() -> list[str]:
    """Run the stale-process reaper in DRY-RUN and surface its summary line.

    Stale MCP / node / fork-worker trees from prior sessions hold read-only
    gold.db handles and contend for the single-writer DuckDB lock — the
    documented root cause of "commit/drift is slow because a sibling holds the
    lock" (see memory/feedback_stale_mcp_node_process_accumulation_slows_session_2026_05_29.md).

    This wires the already-committed, dry-run-by-default
    `scripts/tools/reap_stale_claude_processes.py` to FIRE at session start so
    the operator sees how many stale candidates exist. It deliberately does NOT
    pass `--apply` — first-wire policy is to prove the dry-run classification is
    correct on this machine (no live-bot false-positive) before any kill is
    enabled. The reaper's own contract hard-excludes capital-path processes and
    is fail-open; this wrapper adds a second fail-silent layer.

    Returns the summary line (+ any candidate detail) or [] on any error.
    """
    try:
        reaper = PROJECT_ROOT / "scripts" / "tools" / "reap_stale_claude_processes.py"
        if not reaper.exists():
            return []
        r = subprocess.run(
            [sys.executable, str(reaper)],  # dry-run (no --apply)
            capture_output=True,
            text=True,
            timeout=35,
            cwd=str(PROJECT_ROOT),
        )
        if r.returncode != 0:
            return []
        out_lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        if not out_lines:
            return []
        # The reaper's last line is its summary ("... N candidate(s) ...").
        summary = out_lines[-1].strip()
        lines = [f"  Process reaper (dry-run): {summary}"]
        # If candidates exist, surface up to 3 KILL/DRY reasons so the operator
        # can eyeball the classification before opting into --apply.
        detail = [ln.strip() for ln in out_lines if "[DRY " in ln or "[KILL" in ln]
        for d in detail[:3]:
            lines.append(f"    {d}")
        # Only nudge --apply when there is genuinely something to clean up. The
        # candidate count is the integer immediately before "candidate(s)".
        m = re.search(r",\s*(\d+)\s+candidate\(s\)", summary)
        if m and int(m.group(1)) > 0:
            lines.append(
                "    (to clean up: python scripts/tools/reap_stale_claude_processes.py --apply)"
            )
        return lines
    except BaseException:  # pragma: no cover - hook fallback path
        return []


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, Exception):
        sys.exit(0)

    session_type = event.get("session_type", "startup")
    session_id = event.get("session_id", "") or os.environ.get("CLAUDE_SESSION_ID", "")
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
        # (session_id, ppid)+heartbeat lease — the real worktree mutex, also
        # enforced PreToolUse by `.claude/hooks/worktree_guard.py`. Passing this
        # session's id lets the canonical module recognise a DIFFERENT live
        # session in this tree and BLOCK us at startup, even when the peer's
        # tree is clean (mid git-surgery after a stash) — the case the
        # mtime-based `_active_sibling_lines()` below structurally misses.
        lease_lines, lease_block = _worktree_lease_lines(session_id)
        if lease_block:
            print("\n".join(lease_lines), file=sys.stderr)
            sys.exit(2)
        # mtime-based active-worktree gate — catches the live sessions the PID
        # lock misses (dead-PID reclaim, crashed session). Hard-blocks only the
        # SAME-tree case; a hot SIBLING tree warns (sanctioned parallel pattern).
        active_lines, active_block = _active_sibling_lines()
        if active_block:
            print("\n".join(active_lines), file=sys.stderr)
            sys.exit(2)
        lines = task_route_lines or _superpower_lines("session-start") or _legacy_startup_lines()
        if active_lines:  # non-blocking hot-sibling warning — surface it
            lines.extend(active_lines)
        if block_lines:  # warning lines (e.g., lock-write failure) — surface them
            lines.extend(block_lines)
        if lease_lines:
            lines.extend(lease_lines)
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
        lines.extend(_startup_packet_lines())
        lines.extend(_crg_context_lines())
        lines.extend(_literature_corpus_lines())
        lines.extend(_main_ci_status_lines())
        lines.extend(_handoff_next_step_line())
        if session_type == "startup":
            lines.extend(_stale_process_reaper_lines())
        print("\n".join(lines), file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
