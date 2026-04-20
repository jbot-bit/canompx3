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
LOCAL_WORKTREE_META = ".canompx3-worktree.local.json"
TASK_NAMESPACE = "tasks"
KNOWN_TOOLS = ("claude", "codex")
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
    state: str | None = None
    handoff_note: str | None = None
    last_actor_tool: str | None = None
    dirty: bool = False


def orphaned_pruned_worktree_artifacts(worktrees: list[WorktreeInfo]) -> list[str]:
    reports: list[str] = []
    for wt in worktrees:
        if not wt.prunable:
            continue
        path = Path(wt.path)
        if not path.exists():
            continue

        markers: list[str] = []
        if (path / WORKTREE_META).exists():
            markers.append(WORKTREE_META)
        if (path / LOCAL_WORKTREE_META).exists():
            markers.append(LOCAL_WORKTREE_META)

        claim_dir = path / ".canompx3-runtime" / "active-sessions"
        claim_count = len(list(claim_dir.glob("*.json"))) if claim_dir.exists() else 0
        if claim_count:
            noun = "claim" if claim_count == 1 else "claims"
            markers.append(f"{claim_count} active-session {noun}")

        if markers:
            reports.append(f"{path} ({', '.join(markers)})")
    return reports


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
    if tool:
        return WORKTREE_ROOT / TASK_NAMESPACE / slugify(tool) / slugify(name)
    return WORKTREE_ROOT / TASK_NAMESPACE / slugify(name)


def build_generic_worktree_path(name: str) -> Path:
    return WORKTREE_ROOT / TASK_NAMESPACE / slugify(name)


def build_legacy_worktree_path(tool: str, name: str) -> Path:
    return WORKTREE_ROOT / slugify(tool) / slugify(name)


def _candidate_worktree_paths(name: str, tool: str | None = None) -> list[Path]:
    candidates: list[Path] = [build_worktree_path(tool or "", name), build_generic_worktree_path(name)]
    if tool:
        candidates.append(build_legacy_worktree_path(tool, name))
    for known_tool in KNOWN_TOOLS:
        legacy = build_legacy_worktree_path(known_tool, name)
        if legacy not in candidates:
            candidates.append(legacy)
    return candidates


def find_managed_worktree_path(name: str, tool: str | None = None) -> Path | None:
    matches: list[Path] = []
    seen: set[Path] = set()

    def _matches(meta: dict[str, str] | None) -> bool:
        if meta is None:
            return False
        if str(meta.get("name")) != name:
            return False
        if tool is not None and str(meta.get("tool")) != tool:
            return False
        return True

    for path in _candidate_worktree_paths(name, tool):
        meta_path = path / WORKTREE_META
        if meta_path.exists():
            meta = read_metadata(path)
            resolved = path.resolve()
            if _matches(meta) and resolved not in seen:
                matches.append(resolved)
                seen.add(resolved)

    if WORKTREE_ROOT.exists():
        for meta_path in sorted(WORKTREE_ROOT.rglob(WORKTREE_META)):
            path = meta_path.parent.resolve()
            if path in seen:
                continue
            meta = read_metadata(path)
            if _matches(meta):
                matches.append(path)
                seen.add(path)

    if len(matches) > 1:
        joined = ", ".join(str(path) for path in matches)
        scope = f"{tool}:{name}" if tool else repr(name)
        raise RuntimeError(f"Multiple managed worktrees share the name {scope}: {joined}")
    return matches[0] if matches else None


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
    # Prefer a relative symlink so the worktree stays portable. On Windows,
    # os.path.relpath raises ValueError when target and link_path.parent are
    # on different drives (e.g. CI project root on D:\ vs pytest tmp_path on
    # C:\). In that case fall back to an absolute target — still a valid
    # symlink, just not portable across rename of either root.
    try:
        link_target: str | Path = os.path.relpath(target, link_path.parent)
    except ValueError:
        link_target = target.resolve()
    try:
        link_path.symlink_to(link_target, target_is_directory=target.is_dir())
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


def write_metadata(
    path: Path,
    tool: str,
    name: str,
    branch: str,
    base_ref: str,
    purpose: str | None = None,
    state: str | None = "active",
    handoff_note: str | None = None,
    last_actor_tool: str | None = None,
) -> None:
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
        "state": state or existing.get("state") or "active",
        "handoff_note": handoff_note if handoff_note is not None else existing.get("handoff_note"),
        "last_actor_tool": last_actor_tool or existing.get("last_actor_tool") or tool,
        "repo_root": str(PROJECT_ROOT),
    }
    payload = json.dumps(meta, indent=2)
    (path / WORKTREE_META).write_text(payload, encoding="utf-8")
    (path / LOCAL_WORKTREE_META).write_text(payload, encoding="utf-8")


def read_metadata(path: Path) -> dict[str, str] | None:
    meta_path = path / WORKTREE_META
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_metadata_for(tool: str, name: str) -> dict[str, str] | None:
    path = find_managed_worktree_path(name, tool=tool)
    if path is None:
        return None
    return read_metadata(path)


def set_metadata_fields(path: Path, **changes: str | None) -> dict[str, str]:
    meta = read_metadata(path)
    if meta is None:
        raise RuntimeError(f"Missing metadata for managed worktree: {path}")
    meta.update({key: value for key, value in changes.items() if value is not None})
    meta["last_opened_at"] = datetime.now(UTC).isoformat()
    (path / WORKTREE_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _ensure_scaffold(path: Path, tool: str, name: str, branch: str, base_ref: str, purpose: str | None = None) -> None:
    for item in SYMLINK_TARGETS:
        ensure_symlink(PROJECT_ROOT / item, path / item)
    for item in HARDLINK_FILES:
        ensure_hardlink(PROJECT_ROOT / item, path / item)
    for item in JUNCTION_DIRS:
        ensure_junction(PROJECT_ROOT / item, path / item)
    write_metadata(
        path,
        tool=tool,
        name=name,
        branch=branch,
        base_ref=base_ref,
        purpose=purpose,
        state="active",
        last_actor_tool=tool,
    )
    try:
        from pipeline.work_capsule import ensure_work_capsule_scaffold

        capsule_path, stage_path = ensure_work_capsule_scaffold(
            path,
            tool=tool,
            name=name,
            branch=branch,
            purpose=purpose,
        )
        meta = read_metadata(path) or {}
        meta["capsule_path"] = capsule_path.relative_to(path).as_posix()
        meta["stage_path"] = stage_path.relative_to(path).as_posix()
        payload = json.dumps(meta, indent=2)
        (path / WORKTREE_META).write_text(payload, encoding="utf-8")
        (path / LOCAL_WORKTREE_META).write_text(payload, encoding="utf-8")
    except Exception:
        pass


def create_worktree(tool: str, name: str, base_ref: str = "HEAD", purpose: str | None = None) -> Path:
    prune_worktrees()
    path = find_managed_worktree_path(name, tool=tool)
    if path is None:
        for candidate in _candidate_worktree_paths(name, tool):
            if candidate.exists():
                path = candidate
                break
    if path is None:
        path = build_worktree_path(tool, name)
    active_paths = {Path(wt.path).resolve() for wt in list_worktrees()}
    if path.exists():
        if path.resolve() not in active_paths:
            raise RuntimeError(
                f"Managed worktree path exists but is not active: {path}. Run prune or remove the stale directory."
            )
        existing = read_metadata(path) or {}
        branch = str(existing.get("branch") or build_branch_name(tool, name))
        base_ref = str(existing.get("base_ref") or base_ref)
        _ensure_scaffold(path, tool=tool, name=name, branch=branch, base_ref=base_ref, purpose=purpose)
        return path

    branch = build_branch_name(tool, name)
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
                state=meta.get("state"),
                handoff_note=meta.get("handoff_note"),
                last_actor_tool=meta.get("last_actor_tool"),
                dirty=bool(worktree_status(path)),
            )
        )
    managed.sort(key=lambda item: item.last_opened_at or item.created_at or "", reverse=True)
    return managed


def _resolve_worktree(path: str | None = None, name: str | None = None, tool: str | None = None) -> Path:
    if path:
        return Path(path).resolve()
    if name and tool:
        existing = find_managed_worktree_path(name, tool=tool)
        return existing or build_worktree_path(tool, name)
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


def handoff_worktree(
    path: Path, target_tool: str, purpose: str | None = None, note: str | None = None
) -> dict[str, str]:
    meta = read_metadata(path)
    if meta is None:
        raise RuntimeError(f"Missing metadata for managed worktree: {path}")

    previous_tool = str(meta.get("tool") or "unknown")
    return set_metadata_fields(
        path,
        tool=target_tool,
        purpose=purpose or str(meta.get("purpose") or ""),
        state="handoff",
        handoff_note=note if note is not None else str(meta.get("handoff_note") or ""),
        last_actor_tool=previous_tool,
    )


def ship_worktree(path: Path, merge_target: str = "main", commit_message: str | None = None) -> dict[str, str]:
    meta = read_metadata(path)
    if meta is None:
        raise RuntimeError(f"Missing metadata for managed worktree: {path}")

    dirty = worktree_status(path)
    if dirty and not commit_message:
        raise RuntimeError("Worktree has uncommitted changes. Provide --commit-message to ship it.")

    root_dirty = worktree_status(PROJECT_ROOT)
    if root_dirty:
        raise RuntimeError("Main repo worktree is dirty. Clean it before shipping a task workstream.")

    current = current_branch(PROJECT_ROOT)
    if current != merge_target:
        raise RuntimeError(f"Main repo is on {current}, expected {merge_target}. Refusing to ship.")

    if dirty:
        add_result = _run_git("add", "-A", cwd=path)
        if add_result.returncode != 0:
            raise RuntimeError(add_result.stderr.strip() or add_result.stdout.strip() or "git add failed")
        commit_result = _run_git("commit", "-m", commit_message, cwd=path)
        if commit_result.returncode != 0:
            raise RuntimeError(commit_result.stderr.strip() or commit_result.stdout.strip() or "git commit failed")

    branch_result = _run_git("branch", "--show-current", cwd=path)
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else ""
    if not branch:
        raise RuntimeError(f"Unable to determine worktree branch for {path}")

    merge_result = _run_git("merge", "--no-ff", "--no-edit", branch, cwd=PROJECT_ROOT)
    if merge_result.returncode != 0:
        raise RuntimeError(merge_result.stderr.strip() or merge_result.stdout.strip() or "git merge failed")

    close_worktree(path, force=False, drop_branch=True)
    return {
        "path": str(path),
        "branch": branch,
        "merge_target": merge_target,
        "name": str(meta.get("name") or path.name),
    }


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


def cmd_handoff(args: argparse.Namespace) -> int:
    path = _resolve_worktree(path=args.path, name=args.name, tool=args.tool)
    meta = handoff_worktree(path, target_tool=args.target_tool, purpose=args.purpose, note=args.note)
    print(json.dumps(meta, indent=2) if args.json else f"Handoff set: {meta.get('name')} -> {meta.get('tool')}")
    return 0


def cmd_prune(_args: argparse.Namespace) -> int:
    before = list_worktrees()
    orphan_reports = orphaned_pruned_worktree_artifacts(before)
    prune_worktrees()
    after = list_worktrees()
    print(f"Pruned worktree metadata. Before={len(before)} After={len(after)}")
    if orphan_reports:
        print("Orphaned worktree directories remain on disk:")
        for report in orphan_reports:
            print(f"  {report}")
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


def cmd_ship(args: argparse.Namespace) -> int:
    path = _resolve_worktree(path=args.path, name=args.name, tool=args.tool)
    result = ship_worktree(path, merge_target=args.merge_target, commit_message=args.commit_message)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Shipped {result['name']} into {result['merge_target']} and closed {result['path']}")
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

    p_handoff = sub.add_parser("handoff", help="Reassign a managed workstream to another tool")
    p_handoff.add_argument("--path", default=None, help="Explicit worktree path")
    p_handoff.add_argument(
        "--tool", choices=["claude", "codex"], default=None, help="Current tool namespace for --name"
    )
    p_handoff.add_argument("--name", default=None, help="Managed workstream name")
    p_handoff.add_argument("--target-tool", required=True, choices=["claude", "codex"], help="New owning tool")
    p_handoff.add_argument("--purpose", default=None, help="Optional updated purpose")
    p_handoff.add_argument("--note", default=None, help="Optional baton note")
    p_handoff.add_argument("--json", action="store_true", help="Print machine-readable result")
    p_handoff.set_defaults(func=cmd_handoff)

    p_prune = sub.add_parser("prune", help="Prune stale worktree metadata")
    p_prune.set_defaults(func=cmd_prune)

    p_close = sub.add_parser("close", help="Close and remove a managed worktree")
    p_close.add_argument("--path", default=None, help="Explicit worktree path")
    p_close.add_argument("--tool", choices=["claude", "codex"], default=None, help="Tool namespace for --name")
    p_close.add_argument("--name", default=None, help="Managed worktree name")
    p_close.add_argument("--force", action="store_true", help="Remove even if dirty")
    p_close.add_argument("--drop-branch", action="store_true", help="Delete the worktree branch after removal")
    p_close.set_defaults(func=cmd_close)

    p_ship = sub.add_parser("ship", help="Commit if needed, merge into main, and close the worktree")
    p_ship.add_argument("--path", default=None, help="Explicit worktree path")
    p_ship.add_argument("--tool", choices=["claude", "codex"], default=None, help="Tool namespace for --name")
    p_ship.add_argument("--name", default=None, help="Managed workstream name")
    p_ship.add_argument("--commit-message", default=None, help="Commit message to use if the worktree is dirty")
    p_ship.add_argument("--merge-target", default="main", help="Target branch to merge into")
    p_ship.add_argument("--json", action="store_true", help="Print machine-readable result")
    p_ship.set_defaults(func=cmd_ship)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
