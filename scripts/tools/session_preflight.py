#!/usr/bin/env python3
"""Session preflight and stale-state guard for canompx3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("CANOMPX3_ROOT", Path.cwd()))
import tempfile

ACTIVE_SESSION_DIR = Path(
    os.environ.get(
        "CANOMPX3_ACTIVE_SESSION_DIR",
        os.path.join(tempfile.gettempdir(), "canompx3-active-sessions"),
    )
)
CLAIM_FRESHNESS = timedelta(hours=8)
GIT_TIMEOUT_SECONDS = 2.0
CLAIM_MODES = {"read-only", "mutating"}


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
    mode: str = "read-only"
    root: str = ""


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


def infer_claim_mode(tool: str | None) -> str:
    label = (tool or "").lower()
    if "search" in label or "shell" in label:
        return "read-only"
    return "mutating"


def _active_claim_key(root: Path, tool: str) -> str:
    payload = f"{tool}|{root.resolve()}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    safe_tool = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in tool)
    return f"{safe_tool}-{digest}.json"


def active_claim_path(
    root: Path,
    tool: str,
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> Path:
    return claim_dir / _active_claim_key(root, tool)


def write_claim(
    claim_path: Path,
    tool: str,
    branch: str,
    head: str,
    *,
    mode: str = "read-only",
    root: str | None = None,
) -> SessionClaim:
    if mode not in CLAIM_MODES:
        raise ValueError(f"Invalid claim mode: {mode}")
    claim = SessionClaim(
        tool=tool,
        branch=branch,
        head_sha=head,
        started_at=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
        mode=mode,
        root=root or "",
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
            mode=str(data.get("mode") or infer_claim_mode(str(data["tool"]))),
            root=str(data.get("root") or ""),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def list_claims(claim_dir: Path = ACTIVE_SESSION_DIR, *, fresh_only: bool = False) -> list[SessionClaim]:
    if not claim_dir.exists():
        return []

    claims: list[SessionClaim] = []
    for path in sorted(claim_dir.glob("*.json")):
        claim = read_claim(path)
        if claim is None:
            continue
        if fresh_only and not _claim_is_fresh(claim):
            continue
        claims.append(claim)
    return claims


def write_active_claim(
    root: Path,
    tool: str,
    *,
    mode: str,
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> SessionClaim:
    return write_claim(
        active_claim_path(root, tool, claim_dir=claim_dir),
        tool=tool,
        branch=branch_name(root),
        head=head_sha(root),
        mode=mode,
        root=str(root.resolve()),
    )


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
    claim_path: Path | None = None,
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> tuple[bool, list[str]]:
    claim = read_claim(claim_path or active_claim_path(root, active_tool, claim_dir=claim_dir))
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
    current_root = str(root.resolve())
    if claim.root and claim.root != current_root:
        warnings.append(f"Root mismatch: claim={claim.root} current={current_root}")
        ok = False

    return ok, warnings


def build_blockers(
    root: Path,
    *,
    active_tool: str | None = None,
    active_mode: str = "read-only",
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> list[str]:
    if not active_tool or active_mode != "mutating":
        return []

    current_branch = branch_name(root)
    current_root = str(root.resolve())
    blockers: list[str] = []
    for claim in list_claims(claim_dir, fresh_only=True):
        if claim.tool == active_tool and claim.root == current_root:
            continue
        if claim.branch != current_branch:
            continue
        if claim.mode != "mutating":
            continue
        blockers.append(
            "Concurrent mutating session blocked: "
            f"{claim.tool} already holds a fresh mutating claim on branch {claim.branch}. "
            "Use a worktree or handoff/finish the other session first."
        )
    return blockers


def build_warnings(
    root: Path,
    context: str,
    active_tool: str | None = None,
    active_mode: str = "read-only",
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> list[str]:
    warnings: list[str] = []
    handoff_path = root / "HANDOFF.md"
    if not handoff_path.exists():
        warnings.append("HANDOFF.md missing.")

    if dirty_files(root):
        warnings.append("Working tree is dirty. Re-read changed files before editing.")

    if "wsl" in context and not (root / ".venv-wsl" / "bin" / "python").exists():
        warnings.append("WSL context but .venv-wsl/bin/python is missing.")
    elif "wsl" in context:
        expected_python = (root / ".venv-wsl" / "bin" / "python").resolve()
        current_python = Path(sys.executable).resolve()
        if current_python != expected_python:
            warnings.append(
                "WSL context is using the wrong interpreter. "
                f"current={current_python} expected={expected_python}. "
                "Use the repo launcher, 'uv run python ...', or '.venv-wsl/bin/python ...'."
            )

    if active_tool:
        current_branch = branch_name(root)
        current_root = str(root.resolve())
        for claim in list_claims(claim_dir, fresh_only=True):
            if claim.tool == active_tool and claim.root == current_root:
                continue
            if claim.branch != current_branch:
                continue
            if active_mode == "read-only" or claim.mode == "read-only":
                warnings.append(
                    "Parallel session present on this branch: "
                    f"{claim.tool} ({claim.mode}) already has a fresh claim on {claim.branch}. "
                    "Keep this session read-only or move to a worktree before editing."
                )
                break

    return warnings


def print_report(
    root: Path,
    context: str,
    claim_tool: str | None = None,
    claim_mode: str = "read-only",
    verify_only: bool = False,
    quiet: bool = False,
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> int:
    blockers = build_blockers(root, active_tool=claim_tool, active_mode=claim_mode, claim_dir=claim_dir)
    warnings = build_warnings(root, context=context, active_tool=claim_tool, active_mode=claim_mode, claim_dir=claim_dir)

    if quiet:
        # Quiet mode: only print warnings, silently write claims
        if blockers:
            for blocker in blockers:
                print(f"  XX {blocker}")
        if warnings:
            for warning in warnings:
                print(f"  !! {warning}")
        if blockers:
            return 2
        if claim_tool and not verify_only:
            write_active_claim(root, tool=claim_tool, mode=claim_mode, claim_dir=claim_dir)
        if verify_only and claim_tool:
            ok, _ = verify_claim(root, active_tool=claim_tool, claim_dir=claim_dir)
            return 0 if ok else 1
        return 0

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
    print(f"Interpreter: {Path(sys.executable).resolve()}")
    python3_path = shutil.which("python3")
    if python3_path:
        print(f"python3 -> {Path(python3_path).resolve()}")
    python_path = shutil.which("python")
    if python_path:
        print(f"python  -> {Path(python_path).resolve()}")

    if blockers:
        print("Blockers:")
        for blocker in blockers:
            print(f"  - {blocker}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    elif not blockers:
        print("Status: clean")

    exit_code = 1 if blockers else 0
    if verify_only and claim_tool:
        ok, claim_warnings = verify_claim(root, active_tool=claim_tool, claim_dir=claim_dir)
        if ok:
            print("SESSION CLAIM OK")
        else:
            print("SESSION CLAIM STALE")
            for warning in claim_warnings:
                print(f"  - {warning}")
            exit_code = 1

    if claim_tool and not verify_only and not blockers:
        claim = write_active_claim(root, tool=claim_tool, mode=claim_mode, claim_dir=claim_dir)
        claim_path = active_claim_path(root, claim_tool, claim_dir=claim_dir)
        print(
            "Claim updated: "
            f"tool={claim.tool} mode={claim.mode} branch={claim.branch} "
            f"head={claim.head_sha} file={claim_path}"
        )

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Session preflight for canompx3")
    parser.add_argument("--context", default="generic", help="Startup context label")
    parser.add_argument("--claim", default=None, help="Write or verify a session claim for this tool")
    parser.add_argument("--mode", choices=sorted(CLAIM_MODES), default=None, help="Claim mode: read-only or mutating")
    parser.add_argument("--verify-claim", action="store_true", help="Verify current HEAD against the stored claim")
    parser.add_argument("--quiet", action="store_true", help="Only print warnings, suppress verbose output")
    parser.add_argument("--with-pulse", action="store_true", help="Append project pulse summary")
    parser.add_argument("--root", default=None, help="Override repo root")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else DEFAULT_ROOT.resolve()
    claim_mode = args.mode or infer_claim_mode(args.claim)
    exit_code = print_report(
        root,
        context=args.context,
        claim_tool=args.claim,
        claim_mode=claim_mode,
        verify_only=args.verify_claim,
        quiet=args.quiet,
    )

    if args.with_pulse:
        try:
            _tools_dir = str(Path(__file__).resolve().parent)
            if _tools_dir not in sys.path:
                sys.path.insert(0, _tools_dir)
            from project_pulse import build_pulse, format_text

            context_lower = (args.context or "").lower()
            tool_name = "codex" if "codex" in context_lower else ("claude" if "claude" in context_lower else "unknown")
            report = build_pulse(root, skip_drift=True, skip_tests=True, tool_name=tool_name)
            print()
            print(format_text(report))
        except Exception as exc:
            print(f"Pulse: unavailable ({type(exc).__name__})")  # Advisory — preflight unaffected

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
