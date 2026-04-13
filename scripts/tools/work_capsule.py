#!/usr/bin/env python3
"""CLI wrapper for work capsule summary and scaffold operations."""

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
    if __name__ != "__main__":
        return
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

from pipeline.work_capsule import current_branch, ensure_work_capsule_scaffold, evaluate_current_capsule  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show or scaffold the current work capsule.")
    parser.add_argument("--create-name", default=None)
    parser.add_argument("--purpose", default="Scaffold a work capsule.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.create_name:
        capsule, stage = ensure_work_capsule_scaffold(
            PROJECT_ROOT,
            tool="codex",
            name=args.create_name,
            branch=current_branch(PROJECT_ROOT),
            purpose=args.purpose,
        )
        print(json.dumps({"capsule": str(capsule), "stage": str(stage)}, indent=2))
        return 0
    summary, issues = evaluate_current_capsule(PROJECT_ROOT)
    print(json.dumps({"summary": summary, "issues": [issue.__dict__ for issue in issues]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
