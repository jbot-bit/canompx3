#!/usr/bin/env python3
"""Thin Codex publish assistant for canompx3.

This script does not replace git hooks or GitHub CLI. It classifies repo state,
runs small Codex-scoped preflight checks, records local evidence under .git/,
and delegates publish actions to normal `git push` / `gh pr` commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tools import session_preflight  # noqa: E402

CODEX_TOOL = "codex"
CODEX_CONTEXT = "generic"
DEFAULT_LABELS = ("codex", "codex-automation")


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class GitState:
    root: str
    branch: str
    head: str
    upstream: str | None
    staged: list[str]
    unstaged: list[str]
    status: list[str]

    @property
    def detached(self) -> bool:
        return self.branch == "HEAD"


@dataclass(frozen=True)
class SessionPolicy:
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PublishPlan:
    action: str
    blockers: list[str]
    warnings: list[str]
    commands: list[list[str]]
    labels: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blockers


def run_command(command: list[str], *, cwd: Path = ROOT, timeout: float | None = None) -> CommandResult:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
        return CommandResult(
            command=command,
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            elapsed_s=time.perf_counter() - start,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=command,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or f"timeout after {timeout}s",
            elapsed_s=time.perf_counter() - start,
        )
    except OSError as exc:
        return CommandResult(
            command=command,
            returncode=127,
            stderr=str(exc),
            elapsed_s=time.perf_counter() - start,
        )


def _git(args: list[str], *, cwd: Path = ROOT, timeout: float | None = 10.0) -> CommandResult:
    return run_command(["git", *args], cwd=cwd, timeout=timeout)


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _git_output(args: list[str], *, cwd: Path = ROOT, timeout: float | None = 10.0) -> str | None:
    result = _git(args, cwd=cwd, timeout=timeout)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def repo_root() -> Path:
    out = _git_output(["rev-parse", "--show-toplevel"], cwd=ROOT)
    return Path(out).resolve() if out else ROOT.resolve()


def collect_git_state(root: Path) -> GitState:
    branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root) or "HEAD"
    head = _git_output(["rev-parse", "--short=12", "HEAD"], cwd=root) or "<unknown>"
    upstream = _git_output(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd=root)
    staged = _lines(_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"], cwd=root).stdout)
    unstaged = _lines(_git(["diff", "--name-only"], cwd=root).stdout)
    status = _lines(_git(["status", "--short"], cwd=root).stdout)
    return GitState(
        root=str(root),
        branch=branch,
        head=head,
        upstream=upstream,
        staged=staged,
        unstaged=unstaged,
        status=status,
    )


def codex_python_files(paths: list[str]) -> list[str]:
    selected: list[str] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if not normalized.endswith(".py"):
            continue
        if normalized.startswith("scripts/infra/codex_") or normalized.startswith("tests/test_tools/test_codex"):
            selected.append(path)
    return selected


def has_local_handoff_churn(state: GitState) -> bool:
    if "HANDOFF.md" in state.staged:
        return False
    return any(line.endswith("HANDOFF.md") for line in state.status)


def classify_state(state: GitState) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if state.detached:
        blockers.append("Detached HEAD: create or switch to a named branch before publishing.")
    if not state.upstream:
        warnings.append("No upstream branch is configured; push may need --set-upstream.")
    mixed = sorted(set(codex_python_files(state.staged)).intersection(state.unstaged))
    if mixed:
        blockers.append("Staged Codex Python file(s) also have unstaged changes: " + ", ".join(mixed))
    if "HANDOFF.md" in state.staged:
        warnings.append("HANDOFF.md is staged; confirm this is deliberate shared baton content.")
    elif has_local_handoff_churn(state):
        warnings.append("HANDOFF.md has local unstaged churn; it will be left out unless explicitly staged.")
    return blockers, warnings


def check_session_policy(root: Path) -> SessionPolicy:
    blockers = session_preflight.build_blockers(
        root,
        context=CODEX_CONTEXT,
        active_tool=CODEX_TOOL,
        active_mode="mutating",
    )
    warnings = session_preflight.build_warnings(
        root,
        context=CODEX_CONTEXT,
        active_tool=CODEX_TOOL,
        active_mode="mutating",
    )
    return SessionPolicy(blockers=blockers, warnings=warnings)


def available_labels(root: Path) -> set[str]:
    result = run_command(["gh", "label", "list", "--limit", "200", "--json", "name"], cwd=root, timeout=15.0)
    if result.returncode != 0:
        return set()
    try:
        rows = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return set()
    return {str(row.get("name", "")).strip() for row in rows if str(row.get("name", "")).strip()}


def select_labels(root: Path, wanted: tuple[str, ...] = DEFAULT_LABELS) -> tuple[list[str], list[str]]:
    labels = available_labels(root)
    present = [label for label in wanted if label in labels]
    missing = [label for label in wanted if label not in labels]
    return present, missing


def existing_pr_number(root: Path, branch: str) -> str | None:
    result = run_command(
        ["gh", "pr", "list", "--head", branch, "--json", "number", "--limit", "1"],
        cwd=root,
        timeout=15.0,
    )
    if result.returncode != 0:
        return None
    try:
        rows = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return None
    if not rows:
        return None
    number = rows[0].get("number")
    return str(number) if number else None


def evidence_dir(root: Path) -> Path:
    git_dir = _git_output(["rev-parse", "--git-dir"], cwd=root) or ".git"
    git_path = Path(git_dir)
    if not git_path.is_absolute():
        git_path = root / git_path
    return git_path / "canompx3" / "codex_publish"


def evidence_key(state: GitState, action: str) -> str:
    payload = "\n".join([state.head, state.branch, action, *state.staged])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{action}-{state.head}-{digest}.json"


def write_evidence(root: Path, state: GitState, plan: PublishPlan, results: list[CommandResult]) -> Path:
    target_dir = evidence_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / evidence_key(state, plan.action)
    payload = {
        "created_at_epoch": time.time(),
        "state": asdict(state),
        "plan": asdict(plan),
        "results": [asdict(result) for result in results],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def preflight_commands(state: GitState) -> list[list[str]]:
    commands: list[list[str]] = [["git", "diff", "--check"]]
    py_files = codex_python_files(state.staged)
    if py_files:
        commands.append(["ruff", "check", *py_files])
        commands.append([sys.executable, "-m", "py_compile", *py_files])
        commands.append([sys.executable, "-m", "pytest", "tests/test_tools/test_codex_publish.py", "-q"])
    return commands


def build_plan(root: Path, action: str) -> PublishPlan:
    state = collect_git_state(root)
    blockers, warnings = classify_state(state)
    session_policy = check_session_policy(root)
    blockers.extend(session_policy.blockers)
    warnings.extend(session_policy.warnings)

    commands: list[list[str]] = []
    labels: list[str] = []
    missing_labels: list[str] = []
    if action == "preflight":
        commands = preflight_commands(state)
    elif action == "push":
        commands = [] if blockers else [["git", "push"]]
    elif action == "pr":
        if not blockers:
            labels, missing_labels = select_labels(root)
            pr_number = existing_pr_number(root, state.branch)
            if pr_number:
                label_args = [arg for label in labels for arg in ("--add-label", label)]
                commands = [["gh", "pr", "edit", pr_number, *label_args]]
            else:
                label_args = [arg for label in labels for arg in ("--label", label)]
                commands = [["gh", "pr", "create", "--base", "main", "--head", state.branch, "--fill", *label_args]]
    elif action == "status":
        commands = []
    else:
        blockers.append(f"Unsupported action: {action}")

    return PublishPlan(
        action=action,
        blockers=blockers,
        warnings=warnings,
        commands=commands,
        labels=labels,
        missing_labels=missing_labels,
    )


def run_plan(root: Path, state: GitState, plan: PublishPlan, *, dry_run: bool) -> tuple[int, list[CommandResult]]:
    if plan.action == "status":
        return 0, []
    if plan.blockers:
        return 2, []
    if dry_run:
        return 0, []
    results: list[CommandResult] = []
    for command in plan.commands:
        result = run_command(command, cwd=root)
        results.append(result)
        if result.returncode != 0:
            write_evidence(root, state, plan, results)
            return result.returncode, results
    write_evidence(root, state, plan, results)
    return 0, results


def print_summary(state: GitState, plan: PublishPlan, results: list[CommandResult] | None = None) -> None:
    print(f"codex_publish {plan.action}")
    print(f"root: {state.root}")
    print(f"branch: {state.branch}")
    print(f"head: {state.head}")
    print(f"upstream: {state.upstream or '<none>'}")
    if state.staged:
        print("staged:")
        for path in state.staged:
            print(f"  {path}")
    if plan.blockers:
        print("blockers:")
        for blocker in plan.blockers:
            print(f"  XX {blocker}")
    if plan.warnings:
        print("warnings:")
        for warning in plan.warnings:
            print(f"  !! {warning}")
    if plan.missing_labels:
        print("missing labels:")
        for label in plan.missing_labels:
            print(f"  {label}")
    if plan.commands:
        print("commands:")
        for command in plan.commands:
            print("  " + " ".join(command))
    if results:
        print("results:")
        for result in results:
            print(f"  rc={result.returncode} elapsed={result.elapsed_s:.3f}s :: {' '.join(result.command)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thin Codex publish assistant for canompx3")
    parser.add_argument("--root", default=None, help="Repo root override")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "preflight", "push", "pr"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--dry-run", action="store_true", help="Print planned actions without running commands")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve() if args.root else repo_root()
    state = collect_git_state(root)
    plan = build_plan(root, args.command)
    dry_run = bool(args.dry_run or args.command == "status")
    rc, results = run_plan(root, state, plan, dry_run=dry_run)
    print_summary(state, plan, results)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
