#!/usr/bin/env python3
"""Fail fast when a WSL-backed repo mount is read-only or unstable."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class MountEntry:
    source: str
    mount_point: str
    fs_type: str
    options: tuple[str, ...]


def _decode_mount_field(value: str) -> str:
    return value.replace("\\040", " ")


def parse_proc_mounts(text: str) -> list[MountEntry]:
    entries: list[MountEntry] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        source, mount_point, fs_type, options = parts[:4]
        entries.append(
            MountEntry(
                source=_decode_mount_field(source),
                mount_point=_decode_mount_field(mount_point),
                fs_type=_decode_mount_field(fs_type),
                options=tuple(option for option in options.split(",") if option),
            )
        )
    return entries


def read_proc_mounts() -> str:
    return Path("/proc/mounts").read_text(encoding="utf-8")


def find_mount_entry(path: PurePosixPath, entries: list[MountEntry]) -> MountEntry | None:
    best: MountEntry | None = None
    for entry in entries:
        mount_point = PurePosixPath(entry.mount_point)
        if path == mount_point or mount_point in path.parents:
            if best is None or len(entry.mount_point) > len(best.mount_point):
                best = entry
    return best


def duplicate_mounts(mount_point: str, entries: list[MountEntry]) -> list[MountEntry]:
    return [entry for entry in entries if entry.mount_point == mount_point]


def resolve_git_dir(root: Path) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--absolute-git-dir"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    git_dir = result.stdout.strip()
    if not git_dir:
        return None
    return Path(git_dir)


def probe_write_access(path: Path) -> str | None:
    if not path.exists():
        return f"path missing: {path}"
    if not path.is_dir():
        return f"path is not a directory: {path}"

    try:
        fd, probe_path = tempfile.mkstemp(prefix=".codex-mount-probe-", dir=path)
        os.close(fd)
        Path(probe_path).unlink()
    except OSError as exc:
        return f"{path}: {exc.strerror or exc}"
    return None


def collect_mount_issues(
    root: Path,
    *,
    mount_text: str | None = None,
    git_dir: Path | None = None,
) -> list[str]:
    issues: list[str] = []
    posix_root = PurePosixPath(root.as_posix())

    if mount_text is None:
        try:
            mount_text = read_proc_mounts()
        except OSError:
            mount_text = None

    if mount_text:
        entries = parse_proc_mounts(mount_text)
        mount_entry = find_mount_entry(posix_root, entries)
        if mount_entry is not None:
            if "ro" in mount_entry.options:
                issues.append(
                    f"mount {mount_entry.mount_point} is read-only "
                    f"(fs={mount_entry.fs_type}, opts={','.join(mount_entry.options)})"
                )
            duplicates = duplicate_mounts(mount_entry.mount_point, entries)
            if len(duplicates) > 1:
                rendered = "; ".join(
                    f"{entry.fs_type}:{','.join(entry.options)}" for entry in duplicates
                )
                issues.append(
                    f"mount {mount_entry.mount_point} has multiple active entries ({rendered})"
                )

    root_probe_error = probe_write_access(root)
    if root_probe_error:
        issues.append(f"repo root write probe failed: {root_probe_error}")

    resolved_git_dir = git_dir if git_dir is not None else resolve_git_dir(root)
    if resolved_git_dir is not None:
        git_probe_error = probe_write_access(resolved_git_dir)
        if git_probe_error:
            issues.append(f"git dir write probe failed: {git_probe_error}")

    return issues


def format_failure(root: Path, issues: list[str]) -> str:
    recovery_root = root.as_posix()
    return "\n".join(
        [
            f"ERROR: WSL mount health check failed for {root}",
            "",
            "Why launch is blocked:",
            *[f"  - {issue}" for issue in issues],
            "",
            "Recovery:",
            "  1. Close terminals or tools using this repo.",
            "  2. In Windows PowerShell, run: wsl --shutdown",
            "  3. Restart WSL and verify the mount is writable again.",
            f"  4. Test from WSL: touch {recovery_root}/.mount-write-test && rm {recovery_root}/.mount-write-test",
            "  5. If this keeps happening on /mnt/c, move the Codex repo to the WSL filesystem",
            "     such as ~/canompx3 and launch it there instead.",
            "",
            "Microsoft's WSL guidance is to keep Linux-command-line projects in the Linux filesystem",
            "and Windows-command-line projects in the Windows filesystem.",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail fast on unhealthy WSL repo mounts")
    parser.add_argument("--root", required=True, help="Repo root to validate")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    issues = collect_mount_issues(root)
    if not issues:
        return 0
    print(format_failure(root, issues))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
