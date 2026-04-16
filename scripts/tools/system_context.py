#!/usr/bin/env python3
"""CLI/read model for the canonical project system context."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
    build_system_context,
    evaluate_system_policy,
    format_system_context_text,
    write_decision_log,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show canonical project system context")
    parser.add_argument(
        "--context",
        default="generic",
        choices=["generic", "codex-wsl", "claude-windows", "claude-shell", "unknown"],
        help="Execution context used to resolve interpreter/env expectations.",
    )
    parser.add_argument(
        "--action",
        default=None,
        choices=["orientation", "session_start_read_only", "session_start_mutating"],
        help="Optional policy action to evaluate against the snapshot.",
    )
    parser.add_argument("--tool", default=None, help="Active tool label for policy evaluation.")
    parser.add_argument(
        "--mode",
        default="read-only",
        choices=["read-only", "mutating"],
        help="Active session mode for policy evaluation.",
    )
    parser.add_argument("--format", default="text", choices=["text", "json"], help="Output format.")
    parser.add_argument(
        "--claim-dir",
        default=str(ACTIVE_SESSION_DIR),
        help="Override active session claim directory.",
    )
    parser.add_argument(
        "--log-decision",
        action="store_true",
        help="Append the evaluated decision and snapshot to data/state/system_context_decisions.jsonl.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path.cwd()
    snapshot = build_system_context(
        root,
        context_name=args.context,
        active_tool=args.tool,
        active_mode=args.mode,
        claim_dir=Path(args.claim_dir),
    )
    decision = evaluate_system_policy(snapshot, args.action) if args.action else None

    if args.log_decision:
        if decision is None:
            raise SystemExit("--log-decision requires --action")
        write_decision_log(root, snapshot, decision)

    if args.format == "json":
        payload = {"snapshot": snapshot.model_dump(mode="json")}
        if decision is not None:
            payload["decision"] = decision.model_dump(mode="json")
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_system_context_text(snapshot, decision))

    if decision is not None and not decision.allowed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
