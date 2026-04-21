#!/usr/bin/env python3
"""Session start hook: inject a concise workspace brief on entry/reset."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.tools.claude_superpower_brief import build_brief
except Exception:  # pragma: no cover - hook fallback path
    build_brief = None

try:
    from scripts.tools.task_route_packet import read_task_route_packet
except Exception:  # pragma: no cover - hook fallback path
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


def _parallel_session_lines() -> list[str]:
    """Detect other-terminal collisions and emit warnings.

    Triggers: (a) more than one git worktree active, (b) recent branch
    switch in this worktree, (c) stashes referencing a branch other than
    the current one.

    Mirrors two MEMORY.md feedback rules:
      - feedback_parallel_session_awareness.md
      - feedback_no_push_to_other_terminal_branch.md
    """
    lines: list[str] = []

    # (a) Other active worktrees
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        worktrees = [
            line[9:] for line in result.stdout.splitlines() if line.startswith("worktree ")
        ]
        if len(worktrees) > 1:
            others = [w for w in worktrees if Path(w).resolve() != PROJECT_ROOT]
            if others:
                lines.append(
                    f"  PARALLEL: {len(others)} other worktree(s) active: "
                    + ", ".join(Path(w).name for w in others[:3])
                )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    # (b) Most-recent branch checkout in reflog
    try:
        result = subprocess.run(
            ["git", "reflog", "-5"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        for line in result.stdout.splitlines():
            if "checkout: moving from" in line:
                lines.append(f"  RECENT CHECKOUT: {line.split(':', 1)[-1].strip()[:120]}")
                break  # most recent only
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    # (c) Stashes from other branches
    try:
        cur = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.strip()
        stash = subprocess.run(
            ["git", "stash", "list"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        other_branch_stashes = [
            s for s in stash.stdout.splitlines()
            if cur and f"On {cur}:" not in s and "On " in s
        ]
        if other_branch_stashes:
            lines.append(
                f"  STASHES FROM OTHER BRANCHES: {len(other_branch_stashes)} "
                f"(inspect `git stash list` before destructive ops)"
            )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    if lines:
        return ["PARALLEL-SESSION AWARENESS:"] + lines
    return []


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)
    except Exception as exc:
        print(f"[session-start] unexpected: {exc}", file=sys.stderr)
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

    # Parallel-session awareness — append for every session_type so the user
    # sees collision signals on startup, resume, and clear.
    if session_type != "compact":
        lines.extend(_parallel_session_lines())

    if lines:
        print("\n".join(lines), file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
