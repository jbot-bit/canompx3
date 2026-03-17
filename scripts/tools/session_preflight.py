#!/usr/bin/env python3
"""Session preflight and stale-state guard for canompx3."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("CANOMPX3_ROOT", Path.cwd()))
ACTIVE_SESSION_FILE = Path(os.environ.get("CANOMPX3_ACTIVE_SESSION_FILE", "/tmp/canompx3-active-session.json"))
CLAIM_FRESHNESS = timedelta(hours=8)
GIT_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class HandoffSnapshot:
    tool: str | None = None
    date: str | None = None
    summary: str | None = None


@dataclass(frozen=True)
class SessionClaim:
    tool: str
    branch: str
    head_sha: str
    started_at: str
    pid: int


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def recent_commits(root: Path) -> list[str]:
    result = _run_git(root, "log", "--oneline", "-10")
    if result is None or result.returncode != 0:
        return ["<unavailable>"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def branch_name(root: Path) -> str:
    result = _run_git(root, "branch", "--show-current")
    if result is None or result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "detached"


def head_sha(root: Path) -> str:
    result = _run_git(root, "rev-parse", "HEAD")
    if result is None or result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def dirty_files(root: Path) -> list[str]:
    result = _run_git(root, "status", "--short")
    if result is None:
        return ["git status unavailable or timed out"]
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git status failed"
        return [message]
    return [line for line in result.stdout.splitlines() if line.strip()]


def extract_handoff_snapshot(handoff_path: Path) -> HandoffSnapshot:
    if not handoff_path.exists():
        return HandoffSnapshot()

    tool = date = summary = None
    for line in handoff_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("- **Tool:** "):
            tool = line.removeprefix("- **Tool:** ").strip()
        elif line.startswith("- **Date:** "):
            date = line.removeprefix("- **Date:** ").strip()
        elif line.startswith("- **Summary:** "):
            summary = line.removeprefix("- **Summary:** ").strip()
        if tool and date and summary:
            break
    return HandoffSnapshot(tool=tool, date=date, summary=summary)


def write_claim(claim_path: Path, tool: str, branch: str, head: str) -> SessionClaim:
    claim = SessionClaim(
        tool=tool,
        branch=branch,
        head_sha=head,
        started_at=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
    )
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text(json.dumps(asdict(claim), indent=2), encoding="utf-8")
    return claim


def read_claim(claim_path: Path) -> SessionClaim | None:
    if not claim_path.exists():
        return None
    try:
        data = json.loads(claim_path.read_text(encoding="utf-8"))
        return SessionClaim(
            tool=str(data["tool"]),
            branch=str(data["branch"]),
            head_sha=str(data["head_sha"]),
            started_at=str(data["started_at"]),
            pid=int(data["pid"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _claim_is_fresh(claim: SessionClaim) -> bool:
    try:
        started = datetime.fromisoformat(claim.started_at)
    except ValueError:
        return False
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return datetime.now(UTC) - started <= CLAIM_FRESHNESS


def verify_claim(
    root: Path,
    active_tool: str,
    claim_path: Path = ACTIVE_SESSION_FILE,
) -> tuple[bool, list[str]]:
    claim = read_claim(claim_path)
    if claim is None:
        return True, []

    warnings: list[str] = []
    ok = True
    current_branch = branch_name(root)
    current_head = head_sha(root)

    if claim.tool != active_tool:
        warnings.append(f"tool mismatch: claim={claim.tool} current={active_tool}")
        ok = False
    if claim.branch != current_branch:
        warnings.append(f"Branch mismatch: claim={claim.branch} current={current_branch}")
        ok = False
    if claim.head_sha != current_head:
        warnings.append(f"HEAD mismatch: claim={claim.head_sha} current={current_head}")
        ok = False

    return ok, warnings


def build_warnings(
    root: Path,
    context: str,
    active_tool: str | None = None,
    claim_path: Path = ACTIVE_SESSION_FILE,
) -> list[str]:
    warnings: list[str] = []
    handoff_path = root / "HANDOFF.md"
    if not handoff_path.exists():
        warnings.append("HANDOFF.md missing.")

    if dirty_files(root):
        warnings.append("Working tree is dirty. Re-read changed files before editing.")

    if "wsl" in context and not (root / ".venv-wsl" / "bin" / "python").exists():
        warnings.append("WSL context but .venv-wsl/bin/python is missing.")

    if active_tool:
        claim = read_claim(claim_path)
        if (
            claim is not None
            and claim.tool != active_tool
            and claim.branch == branch_name(root)
            and _claim_is_fresh(claim)
        ):
            warnings.append(
                f"Concurrent session risk: active {claim.tool} claim already exists on branch {claim.branch}."
            )

    return warnings


def print_report(
    root: Path,
    context: str,
    claim_tool: str | None = None,
    verify_only: bool = False,
    claim_path: Path = ACTIVE_SESSION_FILE,
) -> int:
    print("=== SESSION PREFLIGHT ===")
    print(f"Root: {root}")
    print(f"Context: {context}")
    print(f"Branch: {branch_name(root)}")
    print(f"HEAD: {head_sha(root)}")
    print("Recent commits:")
    for line in recent_commits(root):
        print(f"  {line}")

    handoff = extract_handoff_snapshot(root / "HANDOFF.md")
    if handoff.tool or handoff.date or handoff.summary:
        print("Handoff:")
        if handoff.tool:
            print(f"  Tool: {handoff.tool}")
        if handoff.date:
            print(f"  Date: {handoff.date}")
        if handoff.summary:
            print(f"  Summary: {handoff.summary}")

    windows_env = (root / ".venv" / "Scripts" / "python.exe").exists()
    wsl_env = (root / ".venv-wsl" / "bin" / "python").exists()
    print(f"Env: .venv={'yes' if windows_env else 'no'} | .venv-wsl={'yes' if wsl_env else 'no'}")

    warnings = build_warnings(root, context=context, active_tool=claim_tool, claim_path=claim_path)
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Status: clean")

    exit_code = 0
    if verify_only and claim_tool:
        ok, claim_warnings = verify_claim(root, active_tool=claim_tool, claim_path=claim_path)
        if ok:
            print("SESSION CLAIM OK")
        else:
            print("SESSION CLAIM STALE")
            for warning in claim_warnings:
                print(f"  - {warning}")
            exit_code = 1

    if claim_tool and not verify_only:
        claim = write_claim(
            claim_path,
            tool=claim_tool,
            branch=branch_name(root),
            head=head_sha(root),
        )
        print(f"Claim updated: tool={claim.tool} branch={claim.branch} head={claim.head_sha} file={claim_path}")

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Session preflight for canompx3")
    parser.add_argument("--context", default="generic", help="Startup context label")
    parser.add_argument("--claim", default=None, help="Write or verify a session claim for this tool")
    parser.add_argument("--verify-claim", action="store_true", help="Verify current HEAD against the stored claim")
    parser.add_argument("--root", default=None, help="Override repo root")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else DEFAULT_ROOT.resolve()
    return print_report(
        root,
        context=args.context,
        claim_tool=args.claim,
        verify_only=args.verify_claim,
    )


if __name__ == "__main__":
    raise SystemExit(main())
