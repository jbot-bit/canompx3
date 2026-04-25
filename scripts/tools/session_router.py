#!/usr/bin/env python3
"""Auto-route concurrent mutating sessions into managed worktrees.

The repo already has worktree and claim machinery. This script turns that into
an operator-safe launch path: if a mutating session is started from the main
checkout while that branch is already claimed, route the new session into a
managed worktree instead of letting terminals pile into the same root.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.system_context import list_claims
from scripts.tools import worktree_manager


@dataclass(frozen=True)
class RouteDecision:
    requested_root: str
    resolved_root: str
    branch: str
    routed: bool
    reason: str
    workstream_name: str | None = None


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def branch_name(root: Path) -> str:
    result = _run_git(root, "branch", "--show-current")
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "detached"


def canonical_root(root: Path) -> Path:
    result = _run_git(root, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        return root.resolve()
    return Path(result.stdout.strip()).resolve()


def repo_anchor(root: Path) -> Path | None:
    result = _run_git(root, "rev-parse", "--git-common-dir")
    if result.returncode != 0:
        git_dir = root / ".git"
        return git_dir.resolve() if git_dir.exists() else None
    common_dir = Path(result.stdout.strip())
    if not common_dir.is_absolute():
        common_dir = (root / common_dir).resolve()
    else:
        common_dir = common_dir.resolve()
    return common_dir


def in_linked_worktree(root: Path) -> bool:
    return canonical_root(root) != root.resolve()


def _same_repo_claims(root: Path) -> list[object]:
    anchor = repo_anchor(root)
    if anchor is None:
        return []

    matches: list[object] = []
    for claim in list_claims(fresh_only=True):
        claim_root = getattr(claim, "root", "")
        if not claim_root:
            continue
        try:
            claim_anchor = repo_anchor(Path(claim_root))
        except OSError:
            continue
        if claim_anchor == anchor:
            matches.append(claim)
    return matches


def conflicting_mutating_claims(root: Path, branch: str) -> list[object]:
    requested_root = root.resolve()
    if branch in {"", "unknown", "detached"}:
        return []

    claims = []
    for claim in _same_repo_claims(root):
        if getattr(claim, "mode", "") != "mutating":
            continue
        if getattr(claim, "branch", "") != branch:
            continue
        claim_root = str(getattr(claim, "root", ""))
        if claim_root and Path(claim_root).resolve() == requested_root:
            claims.append(claim)
            continue
        claims.append(claim)
    return claims


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text or "session"


def derive_workstream_name(task: str | None) -> str:
    if task and task.strip():
        return slugify(task)[:48]
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"parallel-{timestamp}"


def route_session(root: Path, *, tool: str, mode: str, task: str | None = None) -> RouteDecision:
    resolved_root = root.resolve()
    branch = branch_name(resolved_root)
    if mode != "mutating":
        return RouteDecision(
            requested_root=str(resolved_root),
            resolved_root=str(resolved_root),
            branch=branch,
            routed=False,
            reason="read_only_mode",
        )

    if in_linked_worktree(resolved_root):
        return RouteDecision(
            requested_root=str(resolved_root),
            resolved_root=str(resolved_root),
            branch=branch,
            routed=False,
            reason="already_isolated_in_worktree",
        )

    conflicts = conflicting_mutating_claims(resolved_root, branch)
    if not conflicts:
        return RouteDecision(
            requested_root=str(resolved_root),
            resolved_root=str(resolved_root),
            branch=branch,
            routed=False,
            reason="no_conflict",
        )

    workstream_name = derive_workstream_name(task)
    purpose = task.strip() if task and task.strip() else f"Auto-routed concurrent {tool} session"
    worktree_path = worktree_manager.create_worktree(tool=tool, name=workstream_name, purpose=purpose)
    return RouteDecision(
        requested_root=str(resolved_root),
        resolved_root=str(worktree_path.resolve()),
        branch=branch,
        routed=True,
        reason="parallel_mutating_claim_detected",
        workstream_name=workstream_name,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-route concurrent sessions into managed worktrees")
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="Repo root or current checkout")
    parser.add_argument("--tool", required=True, help="Launching tool name, e.g. codex")
    parser.add_argument("--mode", choices=("read-only", "mutating"), default="mutating")
    parser.add_argument("--task", default=None, help="Optional task text used for worktree naming")
    parser.add_argument("--format", choices=("path", "json"), default="path")
    args = parser.parse_args()

    decision = route_session(Path(args.root), tool=args.tool, mode=args.mode, task=args.task)
    if args.format == "json":
        print(json.dumps(asdict(decision), indent=2))
    else:
        print(decision.resolved_root)
        if decision.routed:
            print(
                f"Auto-routed concurrent session to managed worktree '{decision.workstream_name}'",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
