#!/usr/bin/env python3
"""Session preflight and stale-state guard for canompx3."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _preferred_repo_python() -> Path | None:
    if os.name == "nt":
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_ROOT / ".venv-wsl" / "bin" / "python"
    return candidate if candidate.exists() else None


def _preferred_repo_prefix(expected_python: Path) -> Path:
    return expected_python.parent.parent.resolve()


def _ensure_repo_python() -> None:
    """Re-exec into repo venv only for direct script invocations."""
    if __name__ != "__main__":
        return
    expected_python = _preferred_repo_python()
    if expected_python is None:
        return
    current_python = Path(sys.executable).resolve()
    current_prefix = Path(sys.prefix).resolve()
    expected_prefix = _preferred_repo_prefix(expected_python)
    if current_prefix == expected_prefix or os.environ.get("CANOMPX3_BOOTSTRAP_DONE") == "1":
        return

    env = os.environ.copy()
    env["CANOMPX3_BOOTSTRAP_DONE"] = "1"
    env.setdefault("CANOMPX3_BOOTSTRAPPED_FROM", str(current_python))
    raise SystemExit(
        subprocess.call(
            [str(expected_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    )


_ensure_repo_python()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.system_context import (
    ACTIVE_SESSION_DIR,
    SessionClaim as SystemSessionClaim,
    active_claim_path as system_active_claim_path,
    branch_name as system_branch_name,
    build_system_context,
    dirty_files as system_dirty_files,
    evaluate_system_policy,
    head_sha as system_head_sha,
    list_claims as system_list_claims,
    read_claim as system_read_claim,
    verify_claim as system_verify_claim,
    write_active_claim as system_write_active_claim,
    write_claim as system_write_claim,
)

DEFAULT_ROOT = Path(os.environ.get("CANOMPX3_ROOT", Path.cwd()))
GIT_TIMEOUT_SECONDS = 2.0
CLAIM_MODES = {"read-only", "mutating"}
KNOWN_CONTEXTS = {"generic", "codex-wsl", "claude-windows", "claude-shell", "unknown"}


@dataclass(frozen=True)
class HandoffSnapshot:
    tool: str | None = None
    date: str | None = None
    summary: str | None = None


SessionClaim = SystemSessionClaim


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
    return system_branch_name(root)


def head_sha(root: Path) -> str:
    return system_head_sha(root)


def dirty_files(root: Path) -> list[str]:
    return system_dirty_files(root)


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


def active_claim_path(
    root: Path,
    tool: str,
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> Path:
    return system_active_claim_path(root, tool, claim_dir=claim_dir)


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
    return system_write_claim(claim_path, tool, branch, head, mode=mode, root=root)


def read_claim(claim_path: Path) -> SessionClaim | None:
    claim = system_read_claim(claim_path)
    if claim is None:
        return None
    if not claim.mode:
        return claim.model_copy(update={"mode": infer_claim_mode(claim.tool)})
    return claim


def list_claims(claim_dir: Path = ACTIVE_SESSION_DIR, *, fresh_only: bool = False) -> list[SessionClaim]:
    return system_list_claims(claim_dir=claim_dir, fresh_only=fresh_only)


def write_active_claim(
    root: Path,
    tool: str,
    *,
    mode: str,
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> SessionClaim:
    return system_write_active_claim(root, tool=tool, mode=mode, claim_dir=claim_dir)


def verify_claim(
    root: Path,
    active_tool: str,
    claim_path: Path | None = None,
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> tuple[bool, list[str]]:
    return system_verify_claim(root, active_tool=active_tool, claim_path=claim_path, claim_dir=claim_dir)


def _normalize_context(context: str) -> str:
    return context if context in KNOWN_CONTEXTS else "unknown"


def _policy_action_for_mode(active_mode: str) -> str:
    return "session_start_mutating" if active_mode == "mutating" else "session_start_read_only"


def _format_policy_messages(issues: list[object]) -> list[str]:
    messages: list[str] = []
    for issue in issues:
        message = str(getattr(issue, "message", "")).strip()
        detail = getattr(issue, "detail", None)
        if detail:
            messages.append(f"{message} ({detail})")
        else:
            messages.append(message)
    return messages


def _evaluate_preflight_policy(
    root: Path,
    *,
    context: str,
    active_tool: str | None,
    active_mode: str,
    claim_dir: Path,
) -> tuple[list[str], list[str]]:
    snapshot = build_system_context(
        root,
        context_name=_normalize_context(context),
        active_tool=active_tool,
        active_mode=active_mode if active_mode in CLAIM_MODES else "read-only",
        claim_dir=claim_dir,
    )
    decision = evaluate_system_policy(snapshot, _policy_action_for_mode(active_mode))
    return _format_policy_messages(decision.blockers), _format_policy_messages(decision.warnings)


def build_blockers(
    root: Path,
    *,
    context: str = "generic",
    active_tool: str | None = None,
    active_mode: str = "read-only",
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> list[str]:
    blockers, _warnings = _evaluate_preflight_policy(
        root,
        context=context,
        active_tool=active_tool,
        active_mode=active_mode,
        claim_dir=claim_dir,
    )
    return blockers


def build_warnings(
    root: Path,
    context: str,
    active_tool: str | None = None,
    active_mode: str = "read-only",
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> list[str]:
    _blockers, warnings = _evaluate_preflight_policy(
        root,
        context=context,
        active_tool=active_tool,
        active_mode=active_mode,
        claim_dir=claim_dir,
    )
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
    blockers, warnings = _evaluate_preflight_policy(
        root,
        context=context,
        active_tool=claim_tool,
        active_mode=claim_mode,
        claim_dir=claim_dir,
    )

    if quiet:
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
    bootstrap_from = os.environ.get("CANOMPX3_BOOTSTRAPPED_FROM")
    if bootstrap_from:
        print(f"Bootstrapped from: {bootstrap_from}")
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
            print(f"Pulse: unavailable ({type(exc).__name__})")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
