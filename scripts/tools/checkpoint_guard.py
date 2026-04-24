#!/usr/bin/env python3
"""Cheap commit-time guard for closeout hygiene and provable session conflicts.

Purpose:
- block commits that stage result artifacts without also staging a durable repo
  surface that records the conclusion
- block commits when fresh mutating claims prove that multiple sessions are
  still editing the same branch

This intentionally stays narrow. It does not try to infer every bad workflow.
The stronger protection for concurrent terminals is session-time auto-routing
into worktrees; this hook only enforces what the repo can prove at commit time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.system_context import list_claims

RESULT_ROOTS = ("docs/audit/results/",)
DURABLE_ROOTS = (
    "HANDOFF.md",
    "docs/runtime/action-queue.yaml",
    "docs/runtime/decision-ledger.md",
    "docs/runtime/debt-ledger.md",
    "docs/plans/",
)


@dataclass(frozen=True)
class GuardReport:
    branch: str
    staged_files: list[str]
    result_files: list[str]
    durable_files: list[str]
    mutating_claims_same_branch: list[str]
    blockers: list[str]
    warnings: list[str]


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _repo_anchor(root: Path) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        git_dir = root / ".git"
        return git_dir.resolve() if git_dir.exists() else None
    common_dir = Path(result.stdout.strip())
    if not common_dir.is_absolute():
        common_dir = (root / common_dir).resolve()
    else:
        common_dir = common_dir.resolve()
    return common_dir


def staged_files() -> list[str]:
    result = _run_git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    if result.returncode != 0:
        return []
    return [item for item in result.stdout.split("\x00") if item]


def classify_closeout_state(files: list[str]) -> tuple[list[str], list[str], list[str]]:
    result_files = [path for path in files if path.startswith(RESULT_ROOTS)]
    durable_files = [
        path
        for path in files
        if path == "HANDOFF.md"
        or path == "docs/runtime/action-queue.yaml"
        or path == "docs/runtime/decision-ledger.md"
        or path == "docs/runtime/debt-ledger.md"
        or path.startswith("docs/plans/")
    ]
    blockers: list[str] = []
    if result_files and not durable_files:
        blockers.append(
            "Result artifacts are staged without any durable closeout surface. "
            "Stage at least one of: HANDOFF.md, docs/runtime/action-queue.yaml, docs/runtime/decision-ledger.md, docs/runtime/debt-ledger.md, docs/plans/*."
        )
    return result_files, durable_files, blockers


def _same_repo_claims() -> list[object]:
    anchor = _repo_anchor(PROJECT_ROOT)
    if anchor is None:
        return []

    same_repo: list[object] = []
    for claim in list_claims(fresh_only=True):
        claim_root = getattr(claim, "root", "")
        if not claim_root:
            continue
        try:
            claim_anchor = _repo_anchor(Path(claim_root))
        except OSError:
            continue
        if claim_anchor == anchor:
            same_repo.append(claim)
    return same_repo


def mutating_claim_conflicts(branch: str) -> tuple[list[str], list[str]]:
    if branch in {"", "unknown", "detached"}:
        return [], []

    same_branch = [
        claim
        for claim in _same_repo_claims()
        if getattr(claim, "mode", "") == "mutating" and getattr(claim, "branch", "") == branch
    ]
    details = [f"{getattr(claim, 'tool', 'unknown')}@{getattr(claim, 'branch', 'unknown')}" for claim in same_branch]
    blockers: list[str] = []
    if len(same_branch) > 1:
        blockers.append(
            "Multiple fresh mutating session claims exist on this branch. "
            "Finish or hand off one session, or route the work into a managed worktree."
        )
    return details, blockers


def build_report(files: list[str] | None = None) -> GuardReport:
    files = staged_files() if files is None else files
    branch_result = _run_git("branch", "--show-current")
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
    result_files, durable_files, closeout_blockers = classify_closeout_state(files)
    claim_details, claim_blockers = mutating_claim_conflicts(branch)
    warnings: list[str] = []
    if result_files and durable_files:
        warnings.append("Result artifacts include a staged durable closeout surface.")
    return GuardReport(
        branch=branch,
        staged_files=files,
        result_files=result_files,
        durable_files=durable_files,
        mutating_claims_same_branch=claim_details,
        blockers=[*closeout_blockers, *claim_blockers],
        warnings=warnings,
    )


def _render_text(report: GuardReport) -> str:
    lines: list[str] = []
    lines.append(f"branch: {report.branch}")
    lines.append(f"staged_files: {len(report.staged_files)}")
    if report.result_files:
        lines.append("result_files:")
        lines.extend(f"  - {path}" for path in report.result_files)
    if report.durable_files:
        lines.append("durable_files:")
        lines.extend(f"  - {path}" for path in report.durable_files)
    if report.mutating_claims_same_branch:
        lines.append("mutating_claims_same_branch:")
        lines.extend(f"  - {item}" for item in report.mutating_claims_same_branch)
    if report.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {item}" for item in report.warnings)
    if report.blockers:
        lines.append("blockers:")
        lines.extend(f"  - {item}" for item in report.blockers)
    else:
        lines.append("status: PASS")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Commit-time checkpoint/concurrency guard")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    report = build_report()
    if args.format == "json":
        print(json.dumps(asdict(report), indent=2))
    else:
        print(_render_text(report))
    return 1 if report.blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
