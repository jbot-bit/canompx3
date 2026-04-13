#!/usr/bin/env python3
"""CLI wrapper for the generated startup brief."""

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

from pipeline.system_brief import build_system_brief  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show the generated startup brief.")
    parser.add_argument("--task", default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--format", default="text", choices=["text", "json", "markdown"])
    parser.add_argument("--touched-path", action="append", default=[])
    return parser


def _render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# System Brief",
        "",
        f"- **Task:** `{payload['task_id']}`",
        f"- **Verification profile:** `{payload['verification_profile']}`",
        f"- **Briefing level:** `{payload['briefing_level']}`",
        "",
        "## Doctrine",
        "",
        *[f"- `{item}`" for item in payload["doctrine_chain"]],
        "",
        "## Canonical Owners",
        "",
        *[f"- `{item}`" for item in payload["canonical_owners"]],
    ]
    return "\n".join(lines)


def _render_text(payload: dict[str, object]) -> str:
    return "\n".join(
        [
            f"System brief: {payload['task_id']}",
            f"Verification profile: {payload['verification_profile']}",
            f"Latency: {payload['startup_latency_ms']}ms",
        ]
    )


def main() -> int:
    args = build_parser().parse_args()
    payload = build_system_brief(
        PROJECT_ROOT,
        task_text=args.task,
        task_id=args.task_id,
        touched_paths=args.touched_path,
    )
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.format == "markdown":
        print(_render_markdown(payload))
    else:
        print(_render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
