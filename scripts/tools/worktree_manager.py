#!/usr/bin/env python3
"""Managed git worktree helper for canompx3."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKTREE_ROOT = PROJECT_ROOT / ".worktrees"
WORKTREE_META = ".canompx3-worktree.json"
SYMLINK_TARGETS = [".venv", ".venv-wsl"]
# Files/dirs linked into worktrees via hard links (files) or junctions (dirs).
# These don't need admin on Windows, unlike symlinks.
HARDLINK_FILES = ["gold.db"]
JUNCTION_DIRS = ["models"]


@dataclass(frozen=True)
class WorktreeInfo:
    path: str
    head: str | None = None
    branch: str | None = None
    prunable: str | None = None


@dataclass(frozen=True)
class ManagedWorktreeInfo:
    tool: str
    name: str
    path: str
    path_tail: str
    branch: str
    head: str | None = None
    created_at: str | None = None
    last_opened_at: str | None = None
    purpose: str | None = None
    dirty: bool = False


def _run_git(*args: str, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text or "session"


def build_branch_name(tool: str, name: str) -> str:
    return f"wt-{slugify(tool)}-{slugify(name)}"


def build_worktree_path(tool: str, name: str) -> Path:
    return WORKTREE_ROOT / slugify(tool) / slugify(name)


def current_branch(root: Path = PROJECT_ROOT) -> str:
    result = _run_git("branch", "--show-current", cwd=root)
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "detached"


def parse_worktree_list(output: str) -> list[WorktreeInfo]:
    entries: list[WorktreeInfo] = []
    current: dict[str, str] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                entries.append(
                    WorktreeInfo(
                        path=current["worktree"],
                        head=current.get("HEAD"),
                        branch=current.get("branch"),
                        prunable=current.get("prunable"),
                    )
                )
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    if current:
        entries.append(
            WorktreeInfo(
                path=current["worktree"],
                head=current.get("HEAD"),
                branch=current.get("branch"),
                prunable=current.get("prunable"),
            )
        )
    return entries


def list_worktrees(root: Path = PROJECT_ROOT) -> list[WorktreeInfo]:
    result = _run_git("worktree", "list", "--porcelain", cwd=root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git worktree list failed")
    return parse_worktree_list(result.stdout)


def prune_worktrees(root: Path = PROJECT_ROOT) -> None:
    _run_git("worktree", "prune", cwd=root)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_symlink(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    if not target.exists():
        return
    rel_target = os.path.relpath(target, link_path.parent)
    try:
        link_path.symlink_to(rel_target, target_is_directory=target.is_dir())
    except OSError:
        pass


def ensure_hardlink(target: Path, link_path: Path) -> None:
    """Create a hard link for a file. No admin needed on Windows (same volume)."""
    if link_path.exists():
        return
    if not target.exists():
        return
    try:
        os.link(target, link_path)
    except OSError:
        pass


def ensure_junction(target: Path, link_path: Path) -> None:
    """Create a directory junction. No admin needed on Windows (same volume)."""
    if link_path.exists() or link_path.is_symlink():
        return
    if not target.exists() or not target.is_dir():
        return
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target)],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            # Fall back to symlink
            ensure_symlink(target, link_path)
    else:
        ensure_symlink(target, link_path)


def write_metadata(path: Path, tool: str, name: str, branch: str, base_ref: str, purpose: str | None = None) -> None:
    existing = read_metadata(path) or {}
    now = datetime.now(UTC).isoformat()
    meta = {
        "tool": tool,
        "name": name,
        "branch": branch,
        "base_ref": base_ref,
        "created_at": existing.get("created_at", now),
        "last_opened_at": now,
        "purpose": purpose or existing.get("purpose"),
        "repo_root": str(PROJECT_ROOT),
    }
    (path / WORKTREE_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_metadata(path: Path) -> dict[str, str] | None:
    meta_path = path / WORKTREE_META
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_metadata_for(tool: str, name: str) -> dict[str, str] | None:
    return read_metadata(build_worktree_path(tool, name))


def _ensure_scaffold(path: Path, tool: str, name: str, branch: str, base_ref: str, purpose: str | None = None) -> None:
    for item in SYMLINK_TARGETS:
        ensure_symlink(PROJECT_ROOT / item, path / item)
    for item in HARDLINK_FILES:
        ensure_hardlink(PROJECT_ROOT / item, path / item)
    for item in JUNCTION_DIRS:
        ensure_junction(PROJECT_ROOT / item, path / item)
    write_metadata(path, tool=tool, name=name, branch=branch, base_ref=base_ref, purpose=purpose)


def create_worktree(tool: str, name: str, base_ref: str = "HEAD", purpose: str | None = None) -> Path:
    prune_worktrees()
    branch = build_branch_name(tool, name)
    path = build_worktree_path(tool, name)
    active_paths = {Path(wt.path).resolve() for wt in list_worktrees()}
    if path.exists():
        if path.resolve() not in active_paths:
            raise RuntimeError(
                f"Managed worktree path exists but is not active: {path}. Run prune or remove the stale directory."
            )
        _ensure_scaffold(path, tool=tool, name=name, branch=branch, base_ref=base_ref, purpose=purpose)
        return path

    ensure_parent(path)
    result = _run_git("worktree", "add", "-b", branch, str(path), base_ref)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "already exists" in stderr or "a branch named" in stderr:
            retry = _run_git("worktree", "add", str(path), branch)
            if retry.returncode != 0:
                retry_stderr = retry.stderr.strip()
                raise RuntimeError(retry_stderr or retry.stdout.strip() or "git worktree add failed")
        elif "is already checked out" not in stderr:
            raise RuntimeError(stderr or result.stdout.strip() or "git worktree add failed")

    _ensure_scaffold(path, tool=tool, name=name, branch=branch, base_ref=base_ref, purpose=purpose)
    return path


def list_managed_worktrees(root: Path = PROJECT_ROOT) -> list[ManagedWorktreeInfo]:
    active = {Path(item.path).resolve(): item for item in list_worktrees(root)}
    managed: list[ManagedWorktreeInfo] = []
    if not WORKTREE_ROOT.exists():
        return managed

    for meta_path in sorted(WORKTREE_ROOT.rglob(WORKTREE_META)):
        path = meta_path.parent.resolve()
        active_info = active.get(path)
        if active_info is None:
            continue
        meta = read_metadata(path)
        if meta is None:
            continue
        managed.append(
            ManagedWorktreeInfo(
                tool=str(meta.get("tool", "unknown")),
                name=str(meta.get("name", path.name)),
                path=str(path),
                path_tail="/".join(path.parts[-2:]),
                branch=str(meta.get("branch", active_info.branch or "")),
                head=active_info.head,
                created_at=meta.get("created_at"),
                last_opened_at=meta.get("last_opened_at"),
                purpose=meta.get("purpose"),
                dirty=bool(worktree_status(path)),
            )
        )
    managed.sort(key=lambda item: item.last_opened_at or item.created_at or "", reverse=True)
    return managed


def _resolve_worktree(path: str | None = None, name: str | None = None, tool: str | None = None) -> Path:
    if path:
        return Path(path).resolve()
    if name and tool:
        return build_worktree_path(tool, name)
    raise ValueError("Provide --path or both --tool and --name")


def worktree_status(path: Path) -> list[str]:
    result = _run_git("status", "--short", cwd=path)
    if result.returncode != 0:
        return [result.stderr.strip() or "unable to read status"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def close_worktree(path: Path, force: bool = False, drop_branch: bool = False) -> None:
    if not path.exists():
        raise RuntimeError(f"Worktree path does not exist: {path}")

    if path.resolve() == PROJECT_ROOT.resolve():
        raise RuntimeError("Refusing to remove the main repo worktree")

    dirty = worktree_status(path)
    if dirty and not force:
        raise RuntimeError("Worktree has uncommitted changes. Use --force if you really want removal.")

    branch_result = _run_git("branch", "--show-current", cwd=path)
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else ""

    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(path))
    result = _run_git(*args)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git worktree remove failed")

    if drop_branch and branch.startswith("wt-"):
        _run_git("branch", "-D", branch)

    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def cmd_list(_args: argparse.Namespace) -> int:
    args = _args
    prune_worktrees()
    if args.managed_only:
        managed = list_managed_worktrees()
        if args.json:
            print(json.dumps([item.__dict__ for item in managed], indent=2))
            return 0
        print("Managed workstreams:")
        if not managed:
            print("(none)")
            return 0
        for item in managed:
            status = "dirty" if item.dirty else "clean"
            purpose = f" | {item.purpose}" if item.purpose else ""
            opened = item.last_opened_at or item.created_at or "-"
            print(f"[{item.tool}] {item.name} | {status} | {opened} | {item.path_tail}{purpose}")
        return 0

    worktrees = list_worktrees()
    if args.json:
        print(json.dumps([item.__dict__ for item in worktrees], indent=2))
        return 0
    print(f"Main branch: {current_branch()}")
    for wt in worktrees:
        branch = wt.branch.removeprefix("refs/heads/") if wt.branch else "-"
        prunable = f" | prunable: {wt.prunable}" if wt.prunable else ""
        print(f"{wt.path} | {branch} | {wt.head or '-'}{prunable}")
    return 0


def cmd_create(args: argparse.Namespace) -> int:
    path = create_worktree(tool=args.tool, name=args.name, base_ref=args.base_ref, purpose=args.purpose)
    if args.json:
        meta = read_metadata(path) or {}
        print(
            json.dumps(
                {
                    "path": str(path),
                    "branch": build_branch_name(args.tool, args.name),
                    "purpose": meta.get("purpose"),
                    "last_opened_at": meta.get("last_opened_at"),
                },
                indent=2,
            )
        )
    else:
        print(path)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    meta = read_metadata_for(args.tool, args.name)
    if meta is None:
        print("{}")
        return 1
    print(json.dumps(meta, indent=2))
    return 0


def cmd_prune(_args: argparse.Namespace) -> int:
    before = list_worktrees()
    prune_worktrees()
    after = list_worktrees()
    print(f"Pruned worktree metadata. Before={len(before)} After={len(after)}")
    return 0


def cmd_close(args: argparse.Namespace) -> int:
    path = _resolve_worktree(path=args.path, name=args.name, tool=args.tool)
    if not path.exists():
        if args.drop_branch and args.tool and args.name:
            branch = build_branch_name(args.tool, args.name)
            _run_git("branch", "-D", branch)
        print(f"Already absent: {path}")
        return 0
    close_worktree(path, force=args.force, drop_branch=args.drop_branch)
    print(f"Closed {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Managed git worktree helper for canompx3")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List worktrees")
    p_list.add_argument("--json", action="store_true", help="Print machine-readable result")
    p_list.add_argument("--managed-only", action="store_true", help="Show only repo-managed task worktrees")
    p_list.set_defaults(func=cmd_list)

    p_create = sub.add_parser("create", help="Create or reuse a managed worktree")
    p_create.add_argument("--tool", required=True, choices=["claude", "codex"], help="Owning tool")
    p_create.add_argument("--name", required=True, help="Task/session name")
    p_create.add_argument("--purpose", default=None, help="Short purpose for the workstream")
    p_create.add_argument("--base-ref", default="HEAD", help="Base ref for the new branch/worktree")
    p_create.add_argument("--json", action="store_true", help="Print machine-readable result")
    p_create.set_defaults(func=cmd_create)

    p_show = sub.add_parser("show", help="Show managed worktree metadata")
    p_show.add_argument("--tool", required=True, choices=["claude", "codex"], help="Owning tool")
    p_show.add_argument("--name", required=True, help="Managed workstream name")
    p_show.set_defaults(func=cmd_show)

    p_prune = sub.add_parser("prune", help="Prune stale worktree metadata")
    p_prune.set_defaults(func=cmd_prune)

    p_close = sub.add_parser("close", help="Close and remove a managed worktree")
    p_close.add_argument("--path", default=None, help="Explicit worktree path")
    p_close.add_argument("--tool", choices=["claude", "codex"], default=None, help="Tool namespace for --name")
    p_close.add_argument("--name", default=None, help="Managed worktree name")
    p_close.add_argument("--force", action="store_true", help="Remove even if dirty")
    p_close.add_argument("--drop-branch", action="store_true", help="Delete the worktree branch after removal")
    p_close.set_defaults(func=cmd_close)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
