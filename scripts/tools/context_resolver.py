#!/usr/bin/env python3
"""Resolve a user task into a deterministic repo context route."""

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

from context.registry import (  # noqa: E402
    FALLBACK_READ_SET,
    TASKS,
    render_route_json,
    render_route_markdown,
    render_route_text,
    resolve_from_text,
    resolve_task,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve a task into the exact repo context to read.")
    parser.add_argument("--task", default=None, help="Natural-language task description.")
    parser.add_argument(
        "--task-id",
        default=None,
        choices=sorted(TASKS),
        help="Explicit task manifest ID. Overrides --task matching.",
    )
    parser.add_argument("--format", default="text", choices=["text", "markdown", "json"])
    return parser


def _render_no_match(task: str, output_format: str, candidates=()) -> str:
    ambiguous = len(candidates) > 1 and candidates[1].score == candidates[0].score
    reason = "Ambiguous task match." if ambiguous else "No deterministic task match found."
    if output_format == "json":
        return json.dumps(
            {
                "matched": False,
                "task": task,
                "reason": reason,
                "fallback_read_set": list(FALLBACK_READ_SET),
                "available_task_ids": sorted(TASKS),
                "candidates": [candidate.__dict__ for candidate in candidates],
            },
            indent=2,
            sort_keys=True,
        )
    if output_format == "markdown":
        lines = [
            "# Context Route",
            "",
            reason,
            "",
            "Fallback read set:",
            "",
        ]
        if candidates:
            lines.extend(["Candidate routes:", ""])
            lines.extend(
                f"- `{candidate.task_id}` (score={candidate.score}) via {', '.join(f'`{term}`' for term in candidate.matched_terms)}"
                for candidate in candidates
            )
            lines.append("")
        lines.extend(f"- `{path}`" for path in FALLBACK_READ_SET)
        return "\n".join(lines)
    lines = [reason, ""]
    if candidates:
        lines.append("Candidate routes:")
        lines.extend(
            f"  - {candidate.task_id} (score={candidate.score}) via {', '.join(candidate.matched_terms)}"
            for candidate in candidates
        )
        lines.append("")
    lines.append("Fallback read set:")
    lines.extend(f"  - {path}" for path in FALLBACK_READ_SET)
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()

    if args.task_id:
        route = resolve_task(args.task_id)
        candidates = ()
    else:
        if not args.task:
            raise SystemExit("Provide --task or --task-id.")
        route, candidates = resolve_from_text(args.task)
        if route is None:
            print(_render_no_match(args.task, args.format, candidates))
            return 2

    if args.format == "json":
        print(render_route_json(route, candidates))
    elif args.format == "markdown":
        print(render_route_markdown(route, candidates))
    else:
        print(render_route_text(route, candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
