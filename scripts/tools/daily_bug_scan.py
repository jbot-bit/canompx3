#!/usr/bin/env python3
"""Deterministic recent-commit scanner for the daily bug scan automation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_SUFFIXES = {".md", ".rst", ".txt", ".adoc"}
TEST_PATH_PARTS = ("tests/",)
CODE_PATH_PARTS = ("scripts/", "trading_app/", "pipeline/", "research/", ".claude/hooks/")
NON_PRODUCTION_PATH_PARTS = ("artifacts/research/",)
CODE_SUFFIXES = {".py", ".ps1", ".bat", ".sh", ".json", ".yaml", ".yml", ".toml"}


@dataclass(frozen=True)
class ScanWindow:
    since_iso: str
    source: str
    hours: int


@dataclass(frozen=True)
class VerificationStatus:
    mode: str
    reason: str
    repo_python: str | None = None
    current_python: str | None = None


@dataclass(frozen=True)
class CandidateCommit:
    sha: str
    committed_at: str
    subject: str
    touched_code_paths: list[str]
    touched_test_paths: list[str]
    diff_stats: dict[str, int]
    local_only: bool = False
    source: str = "base_ref"


@dataclass(frozen=True)
class SkippedCommit:
    sha: str
    committed_at: str
    subject: str
    reason: str
    touched_paths: list[str]


@dataclass(frozen=True)
class ScanPacket:
    generated_at: str
    window: ScanWindow
    git_context: dict[str, Any]
    verification: VerificationStatus
    scanned_commits: list[str]
    skipped_commits: list[SkippedCommit] = field(default_factory=list)
    candidate_commits: list[CandidateCommit] = field(default_factory=list)
    review_next: list[str] = field(default_factory=list)
    total_candidate_count: int = 0
    omitted_candidate_count: int = 0
    risk_reason: list[str] = field(default_factory=list)


def _run_git(root: Path, args: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)
    return result.returncode, (result.stdout or "").rstrip("\n"), (result.stderr or "").strip()


def _parse_since(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO timestamp: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def resolve_window(*, since: str | None, hours: int, now: datetime | None = None) -> ScanWindow:
    current = now or datetime.now(UTC)
    if since:
        since_dt = _parse_since(since)
        return ScanWindow(since_iso=since_dt.isoformat(), source="explicit-since", hours=hours)
    return ScanWindow(since_iso=(current - timedelta(hours=hours)).isoformat(), source="hours-fallback", hours=hours)


def _preferred_repo_python(root: Path) -> Path | None:
    candidates = []
    if os.name == "nt":
        candidates.extend(
            [
                root / ".venv" / "Scripts" / "python.exe",
                root / ".venv-wsl" / "bin" / "python",
            ]
        )
    else:
        candidates.extend(
            [
                root / ".venv-wsl" / "bin" / "python",
                root / ".venv" / "Scripts" / "python.exe",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _verification_status(root: Path) -> VerificationStatus:
    git_rc, _git_root, git_err = _run_git(root, ["rev-parse", "--show-toplevel"])
    repo_python = _preferred_repo_python(root)
    current_python = str(Path(sys.executable).resolve())
    if git_rc != 0:
        return VerificationStatus(
            mode="blocked",
            reason=f"git unavailable: {git_err or git_rc}",
            repo_python=str(repo_python) if repo_python else None,
            current_python=current_python,
        )
    if repo_python is None:
        return VerificationStatus(
            mode="static_only",
            reason="no repo-managed interpreter detected",
            repo_python=None,
            current_python=current_python,
        )
    repo_python_text = str(repo_python.resolve())
    current_lower = current_python.lower()
    repo_lower = repo_python_text.lower()
    if current_lower == repo_lower:
        return VerificationStatus(
            mode="full",
            reason="running under repo-managed interpreter",
            repo_python=repo_python_text,
            current_python=current_python,
        )
    return VerificationStatus(
        mode="static_only",
        reason="repo interpreter exists but current process is outside it",
        repo_python=repo_python_text,
        current_python=current_python,
    )


def _git_context(root: Path, base_ref: str) -> dict[str, Any]:
    head_rc, head, _ = _run_git(root, ["rev-parse", "--short", "HEAD"])
    branch_rc, branch, _ = _run_git(root, ["branch", "--show-current"])
    status_rc, status, _ = _run_git(root, ["status", "--short"])
    return {
        "head": head if head_rc == 0 else None,
        "branch": branch if branch_rc == 0 and branch else None,
        "detached": branch_rc == 0 and not branch,
        "dirty": bool(status) if status_rc == 0 else None,
        "base_ref": base_ref,
    }


def _classify_paths(paths: list[str]) -> tuple[list[str], list[str], bool]:
    code_paths: list[str] = []
    test_paths: list[str] = []
    doc_only = True
    for raw in paths:
        path = raw.replace("\\", "/")
        if any(path.startswith(part) for part in NON_PRODUCTION_PATH_PARTS):
            continue
        if any(path.startswith(part) for part in TEST_PATH_PARTS):
            test_paths.append(path)
            doc_only = False
            continue
        suffix = Path(path).suffix.lower()
        if any(path.startswith(part) for part in CODE_PATH_PARTS) or suffix in CODE_SUFFIXES:
            code_paths.append(path)
            doc_only = False
            continue
        if suffix not in DOC_SUFFIXES and not path.startswith("docs/"):
            doc_only = False
    return code_paths, test_paths, doc_only


def _dedupe_paths(raw_paths: list[str]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for raw in raw_paths:
        path = raw.strip().replace("\\", "/")
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _changed_paths_for_commit(root: Path, sha: str) -> list[str]:
    rc, out, _err = _run_git(
        root,
        ["diff-tree", "--no-commit-id", "--name-only", "-r", "-m", "--root", sha],
    )
    paths = _dedupe_paths(out.splitlines()) if rc == 0 else []
    if paths:
        return paths
    fallback_rc, fallback_out, _ = _run_git(root, ["show", "--format=", "--name-only", sha])
    return _dedupe_paths(fallback_out.splitlines()) if fallback_rc == 0 else []


def _diff_stats(root: Path, sha: str) -> dict[str, int]:
    rc, out, _err = _run_git(root, ["diff-tree", "--numstat", "--no-commit-id", "-r", "-m", "--root", sha])
    if rc != 0 or not out.strip():
        rc, out, _err = _run_git(root, ["show", "--numstat", "--format=", sha])
    added = 0
    deleted = 0
    file_count = 0
    if rc != 0:
        return {"files": 0, "added": 0, "deleted": 0}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        file_count += 1
        if parts[0].isdigit():
            added += int(parts[0])
        if parts[1].isdigit():
            deleted += int(parts[1])
    return {"files": file_count, "added": added, "deleted": deleted}


def _collect_commit_rows(root: Path, *, since_iso: str, refspec: str, first_parent: bool) -> list[tuple[str, str, str]]:
    args = ["log", f"--since={since_iso}", "--format=%H%x1f%cI%x1f%s"]
    if first_parent:
        args.append("--first-parent")
    args.append(refspec)
    rc, out, _err = _run_git(root, args)
    if rc != 0 or not out.strip():
        return []
    rows: list[tuple[str, str, str]] = []
    for line in out.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 3:
            continue
        rows.append((parts[0], parts[1], parts[2]))
    return rows


def _working_tree_paths(root: Path) -> list[str]:
    collected: list[str] = []
    for args in (
        ["diff", "--name-only"],
        ["diff", "--cached", "--name-only"],
        ["ls-files", "--others", "--exclude-standard"],
    ):
        rc, out, _ = _run_git(root, args)
        if rc == 0:
            collected.extend(out.splitlines())
    return _dedupe_paths(collected)


def _working_tree_diff_stats(paths: list[str]) -> dict[str, int]:
    return {"files": len(paths), "added": 0, "deleted": 0}


def build_scan_packet(
    *,
    root: Path,
    since: str | None = None,
    hours: int = 24,
    base_ref: str = "origin/main",
    include_local_head: bool = False,
    include_working_tree: bool = True,
    max_commits: int = 5,
) -> ScanPacket:
    window = resolve_window(since=since, hours=hours)
    git_context = _git_context(root, base_ref)
    verification = _verification_status(root)

    rows = _collect_commit_rows(root, since_iso=window.since_iso, refspec=base_ref, first_parent=True)
    local_only_rows: list[tuple[str, str, str]] = []
    if include_local_head:
        local_only_rows = _collect_commit_rows(
            root, since_iso=window.since_iso, refspec=f"{base_ref}..HEAD", first_parent=False
        )

    seen: set[str] = set()
    scanned: list[str] = []
    skipped: list[SkippedCommit] = []
    candidates: list[CandidateCommit] = []

    def ingest(commit_rows: list[tuple[str, str, str]], *, local_only: bool, source: str) -> None:
        for sha, committed_at, subject in commit_rows:
            if sha in seen:
                continue
            seen.add(sha)
            scanned.append(sha)
            touched_paths = _changed_paths_for_commit(root, sha)
            code_paths, test_paths, doc_only = _classify_paths(touched_paths)
            if doc_only or not code_paths:
                skipped.append(
                    SkippedCommit(
                        sha=sha,
                        committed_at=committed_at,
                        subject=subject,
                        reason="doc-only/no production code",
                        touched_paths=touched_paths,
                    )
                )
                continue
            candidates.append(
                CandidateCommit(
                    sha=sha,
                    committed_at=committed_at,
                    subject=subject,
                    touched_code_paths=code_paths,
                    touched_test_paths=test_paths,
                    diff_stats=_diff_stats(root, sha),
                    local_only=local_only,
                    source=source,
                )
            )

    ingest(rows, local_only=False, source="base_ref")
    ingest(local_only_rows, local_only=True, source="local_head")

    if include_working_tree:
        touched_paths = _working_tree_paths(root)
        if touched_paths:
            code_paths, test_paths, doc_only = _classify_paths(touched_paths)
            if doc_only or not code_paths:
                skipped.append(
                    SkippedCommit(
                        sha="WORKTREE",
                        committed_at=datetime.now(UTC).isoformat(),
                        subject="Uncommitted working tree changes",
                        reason="doc-only/no production code",
                        touched_paths=touched_paths,
                    )
                )
            else:
                candidates.append(
                    CandidateCommit(
                        sha="WORKTREE",
                        committed_at=datetime.now(UTC).isoformat(),
                        subject="Uncommitted working tree changes",
                        touched_code_paths=code_paths,
                        touched_test_paths=test_paths,
                        diff_stats=_working_tree_diff_stats(touched_paths),
                        local_only=True,
                        source="working_tree",
                    )
                )

    source_priority = {"working_tree": 3, "local_head": 2, "base_ref": 1}
    candidates.sort(
        key=lambda item: (
            source_priority.get(item.source, 0),
            0 if item.touched_test_paths else 1,
            len(item.touched_code_paths),
            item.diff_stats["files"],
        ),
        reverse=True,
    )
    total_candidate_count = len(candidates)
    bounded_candidates = candidates[:max_commits]
    omitted_candidate_count = max(0, total_candidate_count - len(bounded_candidates))
    risk_reason: list[str] = []
    if omitted_candidate_count:
        risk_reason.append(f"{omitted_candidate_count} candidate(s) omitted by max_commits={max_commits}")
    review_next = [item.sha for item in bounded_candidates]
    return ScanPacket(
        generated_at=datetime.now(UTC).isoformat(),
        window=window,
        git_context=git_context,
        verification=verification,
        scanned_commits=scanned,
        skipped_commits=skipped,
        candidate_commits=bounded_candidates,
        review_next=review_next,
        total_candidate_count=total_candidate_count,
        omitted_candidate_count=omitted_candidate_count,
        risk_reason=risk_reason,
    )


def render_json(packet: ScanPacket) -> str:
    payload = asdict(packet)
    return json.dumps(payload, indent=2, sort_keys=True)


def render_text(packet: ScanPacket) -> str:
    lines = [
        f"Daily Bug Scan | generated_at={packet.generated_at}",
        f"Window: {packet.window.since_iso} ({packet.window.source})",
        (f"Verification: {packet.verification.mode} | {packet.verification.reason}"),
        (
            "Git: "
            f"base_ref={packet.git_context.get('base_ref')} "
            f"head={packet.git_context.get('head') or 'unknown'} "
            f"branch={packet.git_context.get('branch') or 'DETACHED'}"
        ),
        f"Candidate count: shown={len(packet.candidate_commits)} total={packet.total_candidate_count} omitted={packet.omitted_candidate_count}",
        "",
        "Candidates:",
    ]
    if not packet.candidate_commits:
        lines.append("- none")
    for candidate in packet.candidate_commits:
        lines.append(
            "- "
            f"{candidate.sha[:12]} | {candidate.subject} | "
            f"code={len(candidate.touched_code_paths)} test={len(candidate.touched_test_paths)} "
            f"local_only={'yes' if candidate.local_only else 'no'} source={candidate.source}"
        )
    if packet.risk_reason:
        lines.append("")
        lines.append("Risk notes:")
        lines.extend(f"- {item}" for item in packet.risk_reason)
    lines.append("")
    lines.append("Skipped:")
    if not packet.skipped_commits:
        lines.append("- none")
    for skipped in packet.skipped_commits:
        lines.append(f"- {skipped.sha[:12]} | {skipped.subject} | {skipped.reason}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--include-local-head", action="store_true")
    parser.add_argument("--no-working-tree", action="store_true")
    parser.add_argument("--max-commits", type=int, default=5)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    packet = build_scan_packet(
        root=PROJECT_ROOT,
        since=args.since,
        hours=args.hours,
        base_ref=args.base_ref,
        include_local_head=args.include_local_head,
        include_working_tree=not args.no_working_tree,
        max_commits=args.max_commits,
    )
    print(render_json(packet) if args.format == "json" else render_text(packet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
