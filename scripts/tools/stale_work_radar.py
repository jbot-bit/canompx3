#!/usr/bin/env python3
"""Stale-work radar — surface git work at risk of being silently lost.

This repo runs many parallel AI sessions across ~18 git worktrees and ~30
branches. Work accumulates on `session/*` and `codex/*` branches and goes cold:
unpushed local commits (recoverable only via reflog ~90d if the worktree is
pruned), branches lapped >100 commits behind main (rebase-debt minefields), and
worktrees carrying huge uncommitted diffs (the `git add -A` artifact-blob trap,
e.g. the chordia 2.99M-line incident).

`worktree_manager.py list` reports path/branch/head only — no divergence, no
unpushed detection, no age, no risk scoring. This radar fills that gap.

READ-ONLY. No network by default: divergence is measured against the *cached*
`origin/main` ref, so the radar is safe to run from inside a worktree-lease that
blocks index writes. `--fetch` opts into a refresh, but a `git fetch` is an
index-mutating op and will be blocked by `worktree_guard.py` if run from a
session whose worktree is leased by a live peer — run it from an unleased tree.

Risk score is explainable by design (not a black box): each contributing term is
printed alongside the total so an operator can see *why* a branch ranks high.

Usage:
    python scripts/tools/stale_work_radar.py                 # ranked table
    python scripts/tools/stale_work_radar.py --top 10        # worst 10 only
    python scripts/tools/stale_work_radar.py --json          # machine-readable
    python scripts/tools/stale_work_radar.py --fetch         # refresh origin first
    python scripts/tools/stale_work_radar.py --base origin/main
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Risk-score weights. Tuned so the dominant signal is *unrecoverable* work
# (unpushed commits), then *active* uncommitted work, then rot (age / rebase
# debt). Each term is surfaced in the output so the score is auditable.
W_UNPUSHED_COMMIT = 10.0  # per local commit not on any remote — highest: reflog-only if pruned
W_UPSTREAM_GONE = 15.0  # tracked remote was deleted — local commits orphaned, look "pushed" but aren't
W_DIRTY_FLAG = 5.0  # has uncommitted changes at all
W_AGE_PER_DAY = 0.1  # gentle decay so a fresh ahead=1 doesn't outrank a stale ahead=5
W_REBASE_DEBT = 3.0  # applied once when behind > REBASE_DEBT_THRESHOLD
REBASE_DEBT_THRESHOLD = 100  # commits behind main past which a merge is a conflict minefield
BLOB_TRAP_THRESHOLD = 1000  # uncommitted changed-line count past which it's probably an `add -A` artifact trap


def _scrubbed_git_env() -> dict[str, str]:
    """os.environ minus GIT_DIR/GIT_WORK_TREE/GIT_INDEX_FILE.

    Pre-commit hooks pre-populate these and would override cwd-based repo
    resolution. Mirrors worktree_manager._scrubbed_git_env.
    """
    env = os.environ.copy()
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(var, None)
    return env


def _run_git(*args: str, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=_scrubbed_git_env(),
        timeout=30,
    )


@dataclass
class BranchReport:
    branch: str
    ahead: int  # commits on branch not on base
    behind: int  # commits on base not on branch
    unpushed: int  # commits not reachable from the branch's remote tip
    has_remote: bool  # a refs/remotes/origin/<branch> exists
    upstream_gone: bool  # git reports the tracked upstream ref was deleted ([gone])
    last_commit_age_days: int
    last_commit_date: str
    worktree_path: str | None  # checked-out worktree, if any
    dirty_lines: int  # uncommitted changed lines in that worktree (0 if none / not checked out)
    merged_into_base: bool
    risk_score: float = 0.0
    risk_breakdown: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def _now() -> datetime:
    return datetime.now(UTC)


def base_exists(base: str) -> bool:
    return _run_git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}").returncode == 0


def list_local_branches() -> list[str]:
    res = _run_git("for-each-ref", "--format=%(refname:short)", "refs/heads/")
    if res.returncode != 0:
        return []
    return [b for b in res.stdout.splitlines() if b.strip()]


# Field separator unlikely to appear in a ref name or date — used to parse the
# single batched for-each-ref pass below.
_FER_SEP = "\x1f"


def branch_metadata() -> dict[str, dict[str, str]]:
    """One `git for-each-ref` pass for all local branches (efficiency).

    Canonical git idiom (git-scm.com/docs/git-for-each-ref): a single call emits
    refname, committerdate, the upstream ref, and the upstream:track token —
    which prints "[gone]" when the upstream ref was deleted. Replaces N per-branch
    rev-parse/log calls with one. Returns {branch: {"date","upstream","track"}}.
    """
    fmt = _FER_SEP.join(["%(refname:short)", "%(committerdate:iso-strict)", "%(upstream:short)", "%(upstream:track)"])
    res = _run_git("for-each-ref", f"--format={fmt}", "refs/heads/")
    meta: dict[str, dict[str, str]] = {}
    if res.returncode != 0:
        return meta
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split(_FER_SEP)
        if len(parts) != 4:
            continue
        name, date, upstream, track = parts
        meta[name] = {"date": date, "upstream": upstream, "track": track}
    return meta


def upstream_gone(track: str) -> bool:
    """True when git reports the branch's upstream ref was deleted ([gone])."""
    return "gone" in track.lower()


def ahead_behind(branch: str, base: str) -> tuple[int, int]:
    """(ahead, behind) of `branch` relative to `base`. (0,0) on failure."""
    res = _run_git("rev-list", "--left-right", "--count", f"{base}...{branch}")
    if res.returncode != 0:
        return (0, 0)
    parts = res.stdout.split()
    if len(parts) != 2:
        return (0, 0)
    behind, ahead = parts  # left=base-only=behind, right=branch-only=ahead
    return (int(ahead), int(behind))


def remote_tip(branch: str) -> str | None:
    res = _run_git("rev-parse", "--verify", "--quiet", f"origin/{branch}")
    return res.stdout.strip() if res.returncode == 0 else None


def unpushed_count(branch: str) -> tuple[int, bool]:
    """(commits not on the remote tip, has_remote).

    If no remote tracking ref exists, every commit unique to this branch vs base
    is unpushed — but we can't cheaply know "vs base" here, so we report the full
    commit count reachable from the branch that the remote does not have. With no
    remote at all, we fall back to counting commits ahead of the base in the
    caller (LOCAL-ONLY branches surface via has_remote=False + ahead).
    """
    tip = remote_tip(branch)
    if tip is None:
        return (0, False)
    res = _run_git("rev-list", "--count", f"origin/{branch}..{branch}")
    if res.returncode != 0:
        return (0, True)
    return (int(res.stdout.strip() or "0"), True)


def age_from_iso(iso: str) -> tuple[int, str]:
    """(age_in_days, iso_date) from a committerdate string. (0,'unknown') on parse fail."""
    iso = (iso or "").strip()
    if not iso:
        return (0, "unknown")
    try:
        dt = datetime.fromisoformat(iso)
        age = (_now() - dt).days
        return (max(age, 0), dt.date().isoformat())
    except ValueError:
        return (0, iso)


def is_merged(branch: str, base: str) -> bool:
    return _run_git("merge-base", "--is-ancestor", branch, base).returncode == 0


def worktree_for_branch() -> dict[str, str]:
    """Map branch-name -> worktree path for every checked-out worktree."""
    res = _run_git("worktree", "list", "--porcelain")
    mapping: dict[str, str] = {}
    if res.returncode != 0:
        return mapping
    cur_path: str | None = None
    for line in res.stdout.splitlines():
        if line.startswith("worktree "):
            cur_path = line[len("worktree ") :].strip()
        elif line.startswith("branch ") and cur_path:
            br = line[len("branch ") :].strip().removeprefix("refs/heads/")
            mapping[br] = cur_path
    return mapping


def dirty_line_count(worktree_path: str) -> int:
    """Count of uncommitted changed paths in a worktree (porcelain lines)."""
    res = _run_git("status", "--porcelain", cwd=Path(worktree_path))
    if res.returncode != 0:
        return 0
    return sum(1 for ln in res.stdout.splitlines() if ln.strip())


def score(report: BranchReport) -> None:
    """Populate risk_score + risk_breakdown + flags in place."""
    breakdown: dict[str, float] = {}

    # Unpushed: LOCAL-ONLY branches expose loss-risk via `ahead`; remote-tracked
    # via the precise unpushed count. Take whichever applies.
    unpushed = report.unpushed if report.has_remote else report.ahead
    if unpushed > 0:
        breakdown["unpushed"] = round(unpushed * W_UNPUSHED_COMMIT, 2)
        report.flags.append(
            f"{unpushed} unpushed commit(s)" + ("" if report.has_remote else " (LOCAL-ONLY: reflog-only if pruned)")
        )

    if report.upstream_gone:
        breakdown["upstream_gone"] = W_UPSTREAM_GONE
        report.flags.append("upstream [gone] — tracked remote was deleted; local commits orphaned")

    if report.dirty_lines > 0:
        breakdown["dirty"] = W_DIRTY_FLAG
        if report.dirty_lines > BLOB_TRAP_THRESHOLD:
            report.flags.append(
                f"{report.dirty_lines} uncommitted paths (>{BLOB_TRAP_THRESHOLD}: probable `git add -A` artifact-blob trap)"
            )
        else:
            report.flags.append(f"{report.dirty_lines} uncommitted path(s)")

    if report.last_commit_age_days > 0:
        breakdown["age"] = round(report.last_commit_age_days * W_AGE_PER_DAY, 2)

    if report.behind > REBASE_DEBT_THRESHOLD:
        breakdown["rebase_debt"] = W_REBASE_DEBT
        report.flags.append(f"{report.behind} behind base (>{REBASE_DEBT_THRESHOLD}: rebase-conflict risk)")

    if report.merged_into_base and unpushed == 0 and report.dirty_lines == 0 and not report.upstream_gone:
        report.flags.append("merged into base — safe to prune")

    report.risk_breakdown = breakdown
    report.risk_score = round(sum(breakdown.values()), 2)


def build_reports(base: str) -> list[BranchReport]:
    wt_map = worktree_for_branch()
    meta = branch_metadata()  # single for-each-ref pass: date + upstream:track for all branches
    reports: list[BranchReport] = []
    for branch in list_local_branches():
        bmeta = meta.get(branch, {})
        ahead, behind = ahead_behind(branch, base)
        unpushed, has_remote = unpushed_count(branch)
        age_days, age_date = age_from_iso(bmeta.get("date", ""))
        gone = upstream_gone(bmeta.get("track", ""))
        wt_path = wt_map.get(branch)
        dirty = dirty_line_count(wt_path) if wt_path else 0
        merged = is_merged(branch, base)
        rpt = BranchReport(
            branch=branch,
            ahead=ahead,
            behind=behind,
            unpushed=unpushed,
            has_remote=has_remote,
            upstream_gone=gone,
            last_commit_age_days=age_days,
            last_commit_date=age_date,
            worktree_path=wt_path,
            dirty_lines=dirty,
            merged_into_base=merged,
        )
        score(rpt)
        reports.append(rpt)
    reports.sort(key=lambda r: r.risk_score, reverse=True)
    return reports


def render_table(reports: list[BranchReport], base: str) -> str:
    lines = [f"Stale-Work Radar — {len(reports)} branches vs {base}  ({_now().date().isoformat()})", ""]
    header = f"{'RISK':>6}  {'BRANCH':<48} {'A/B':>9} {'AGE':>5} {'DIRTY':>6}  FLAGS"
    lines.append(header)
    lines.append("-" * len(header))
    for r in reports:
        ab = f"{r.ahead}/{r.behind}"
        age = f"{r.last_commit_age_days}d"
        dirty = str(r.dirty_lines) if r.dirty_lines else "-"
        flag_str = "; ".join(r.flags) if r.flags else ("clean" if r.risk_score == 0 else "")
        lines.append(f"{r.risk_score:>6.1f}  {r.branch:<48} {ab:>9} {age:>5} {dirty:>6}  {flag_str}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Surface git work at risk of being silently lost.")
    parser.add_argument(
        "--base", default="origin/main", help="Base ref to measure divergence against (default origin/main)"
    )
    parser.add_argument(
        "--fetch", action="store_true", help="git fetch origin first (BLOCKED under a peer worktree lease)"
    )
    parser.add_argument("--top", type=int, default=None, help="Show only the N highest-risk branches")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)

    if args.fetch:
        res = _run_git("fetch", "origin", "--quiet")
        if res.returncode != 0:
            print(f"WARN: git fetch failed (continuing on cached refs): {res.stderr.strip()}", file=sys.stderr)

    if not base_exists(args.base):
        print(f"WARN: base ref {args.base!r} not found — cannot measure divergence. Exiting clean.", file=sys.stderr)
        return 0

    reports = build_reports(args.base)
    if args.top is not None:
        reports = reports[: args.top]

    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2))
    else:
        print(render_table(reports, args.base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
