#!/usr/bin/env python3
"""Session start hook: inject a concise workspace brief on entry/reset."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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

    can_ff = (
        branch == "main"
        and ahead_n == 0
        and behind_n > 0
        and not dirty
    )
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
            block["path"] = line[len("worktree "):].strip()
        elif line.startswith("branch "):
            block["branch"] = line[len("branch "):].strip().replace("refs/heads/", "")
        elif line.startswith("HEAD "):
            block["head"] = line[len("HEAD "):].strip()[:8]
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
        lines.append(
            f"  WARNING: {dirty_count + 1} dirty worktrees active — "
            "edit collision/CRLF/stash-loss risk."
        )
        lines.append(
            "  Each Claude session should work in its own worktree. "
            "Spawn one: scripts/tools/new_session.sh"
        )
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
        lines = task_route_lines or _superpower_lines("session-start") or _legacy_startup_lines()
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
        lines.extend(_parallel_session_lines())
        print("\n".join(lines), file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
