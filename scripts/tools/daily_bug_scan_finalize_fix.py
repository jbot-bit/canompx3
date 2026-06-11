#!/usr/bin/env python3
"""Finalize a verified daily-bug-scan fix branch and schedule auto-merge.

This is the handoff helper for the recurring Daily bug scan automation after it
has found and fixed one concrete bug. It keeps the automation from leaving a
verified fix stranded in a detached or local-only worktree.

Default flow:
  1. Ensure the worktree is on the requested fix branch, creating it if needed.
  2. Refuse dirty paths outside the fix package, except explicitly allowed churn.
  3. Run focused verification commands supplied by the automation.
  4. Commit the fix package with hooks skipped because verification was explicit.
  5. Register the guarded AutoRebase scheduled task.
  6. Print JSON proving whether origin/main already contains the fix SHA.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FinalizeResult:
    branch: str
    fix_sha: str | None
    committed: bool
    registered_auto_merge: bool
    origin_main_contains_fix: bool | None
    verification_commands: list[str]
    committed_paths: list[str]
    allowed_dirty_paths: list[str]


def _run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed ({result.returncode}): {(result.stderr or result.stdout).strip()}")
    return result


def _run_shell(command: str, *, cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        shell=True,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{command!r} failed ({result.returncode}): {(result.stderr or result.stdout).strip()}")
    return result


def _git(root: Path, *args: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=root, check=check, timeout=timeout)


def _current_branch(root: Path) -> str:
    return _git(root, "branch", "--show-current", check=False).stdout.strip()


def ensure_fix_branch(root: Path, branch: str) -> None:
    current = _current_branch(root)
    if current == branch:
        return
    exists = _git(root, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False)
    if exists.returncode == 0:
        raise RuntimeError(f"branch {branch!r} already exists but current branch is {current or 'DETACHED'}")
    _git(root, "switch", "-c", branch)


def path_from_porcelain_line(line: str) -> str:
    """Extract the effective path from one `git status --porcelain` line."""
    if len(line) >= 4 and line[2] == " ":
        path = line[3:].strip()
    elif len(line) >= 3:
        path = line[2:].strip()
    else:
        path = line.strip()
    if " -> " in path:
        path = path.split(" -> ")[-1].strip()
    return path.replace("\\", "/")


def dirty_paths(root: Path) -> list[str]:
    status = _git(root, "status", "--porcelain", "--untracked-files=all").stdout.splitlines()
    paths: list[str] = []
    seen: set[str] = set()
    for line in status:
        path = path_from_porcelain_line(line)
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def split_dirty_paths(paths: list[str], allowed_dirty: set[str]) -> tuple[list[str], list[str]]:
    allowed: list[str] = []
    package: list[str] = []
    for path in paths:
        if path in allowed_dirty:
            allowed.append(path)
        else:
            package.append(path)
    return package, allowed


def origin_main_contains(root: Path, sha: str) -> bool | None:
    _git(root, "fetch", "origin", check=False, timeout=60)
    origin = _git(root, "rev-parse", "--verify", "origin/main", check=False)
    if origin.returncode != 0:
        return None
    result = _git(root, "merge-base", "--is-ancestor", sha, "origin/main", check=False)
    return result.returncode == 0


def register_auto_merge(
    *,
    root: Path,
    branch: str,
    main_repo: Path,
    first_run_minutes: int,
    interval_minutes: int,
    allowed_dirty: list[str],
    dry_run: bool,
) -> bool:
    script = root / "scripts" / "tools" / "register_auto_merge_fix_task.ps1"
    args = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Branch",
        branch,
        "-Worktree",
        str(root),
        "-MainRepo",
        str(main_repo),
        "-FirstRunMinutes",
        str(first_run_minutes),
        "-IntervalMinutes",
        str(interval_minutes),
    ]
    for path in allowed_dirty:
        args.extend(["-AllowedDirtyPath", path])
    if dry_run:
        return False
    _run(args, cwd=root, check=True, timeout=60)
    return True


def finalize_fix(
    *,
    root: Path,
    branch: str,
    message: str,
    verify: list[str],
    allowed_dirty: list[str],
    main_repo: Path,
    first_run_minutes: int,
    interval_minutes: int,
    no_register: bool,
    dry_run: bool,
) -> FinalizeResult:
    if not verify:
        raise RuntimeError("at least one --verify command is required")

    ensure_fix_branch(root, branch)
    all_dirty = dirty_paths(root)
    package_paths, allowed_paths = split_dirty_paths(all_dirty, set(allowed_dirty))
    if not package_paths:
        raise RuntimeError("no fix package paths found; only allowed dirty paths are present")

    for command in verify:
        if not dry_run:
            _run_shell(command, cwd=root)

    committed = False
    fix_sha: str | None = None
    if not dry_run:
        _git(root, "add", "--", *package_paths)
        _git(root, "commit", "--no-verify", "-m", message, timeout=120)
        committed = True
        fix_sha = _git(root, "rev-parse", "HEAD").stdout.strip()

    registered = False
    if not no_register:
        registered = register_auto_merge(
            root=root,
            branch=branch,
            main_repo=main_repo,
            first_run_minutes=first_run_minutes,
            interval_minutes=interval_minutes,
            allowed_dirty=allowed_dirty,
            dry_run=dry_run,
        )

    contains = origin_main_contains(root, fix_sha) if fix_sha else None
    return FinalizeResult(
        branch=branch,
        fix_sha=fix_sha,
        committed=committed,
        registered_auto_merge=registered,
        origin_main_contains_fix=contains,
        verification_commands=verify,
        committed_paths=package_paths,
        allowed_dirty_paths=allowed_paths,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--verify", action="append", default=[])
    parser.add_argument("--allowed-dirty", action="append", default=["HANDOFF.md"])
    parser.add_argument("--main-repo", default=r"C:\Users\joshd\canompx3")
    parser.add_argument("--first-run-minutes", type=int, default=1)
    parser.add_argument("--interval-minutes", type=int, default=10)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = finalize_fix(
            root=PROJECT_ROOT,
            branch=args.branch,
            message=args.message,
            verify=args.verify,
            allowed_dirty=args.allowed_dirty,
            main_repo=Path(args.main_repo),
            first_run_minutes=args.first_run_minutes,
            interval_minutes=args.interval_minutes,
            no_register=args.no_register,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, **asdict(result)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
