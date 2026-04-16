#!/usr/bin/env python3
"""CLI for the derived startup system brief."""

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
    current_prefix = Path(sys.prefix).resolve()
    expected_prefix = _preferred_repo_prefix(expected_python)
    if current_prefix == expected_prefix or os.environ.get("CANOMPX3_BOOTSTRAP_DONE") == "1":
        return

    env = os.environ.copy()
    env["CANOMPX3_BOOTSTRAP_DONE"] = "1"
    env.setdefault("CANOMPX3_BOOTSTRAPPED_FROM", str(Path(sys.executable).resolve()))
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

from pipeline.system_brief import build_system_brief
from pipeline.system_context import infer_context_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show the derived startup system brief")
    parser.add_argument("--task", default=None, help="Natural-language task text")
    parser.add_argument("--task-id", default=None, help="Explicit task id")
    parser.add_argument(
        "--briefing-level", default="read_only", choices=["trivial", "read_only", "non_trivial", "mutating"]
    )
    parser.add_argument("--format", default="json", choices=["json", "text"])
    parser.add_argument("--root", default=None, help="Override repo root")
    return parser


def _format_text(payload: dict[str, object]) -> str:
    lines = [
        f"System brief: {payload['task_id']} [{payload['briefing_level']}]",
        f"Verification profile: {payload['verification_profile']}",
        f"Work capsule: {payload.get('work_capsule_ref') or 'none'}",
        "Doctrine chain:",
    ]
    lines.extend(f"  - {item}" for item in payload["doctrine_chain"])
    lines.append("Canonical owners:")
    lines.extend(f"  - {item}" for item in payload["canonical_owners"])
    lines.append("Required live views:")
    lines.extend(f"  - {item}" for item in payload["required_live_views"])
    if payload["blocking_issues"]:
        lines.append("Blockers:")
        lines.extend(f"  - {issue['message']}" for issue in payload["blocking_issues"])
    if payload["warning_issues"]:
        lines.append("Warnings:")
        lines.extend(f"  - {issue['message']}" for issue in payload["warning_issues"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    payload = build_system_brief(
        root,
        task_text=args.task,
        task_id=args.task_id,
        briefing_level=args.briefing_level,  # type: ignore[arg-type]
        context_name=infer_context_name(root, Path(sys.executable)),
        active_mode="mutating" if args.briefing_level == "mutating" else "read-only",
    )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_text(payload))
    return 1 if payload["blocking_issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
