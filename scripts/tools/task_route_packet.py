#!/usr/bin/env python3
"""Write and read a compact startup task-route packet."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET_RELATIVE_PATH = Path(".session/task-route.md")


def _preferred_repo_python() -> Path | None:
    if os.name == "nt":
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_ROOT / ".venv-wsl" / "bin" / "python"
    return candidate if candidate.exists() else None


def _preferred_repo_prefix(expected_python: Path) -> Path:
    return expected_python.parent.parent.resolve()


def _ensure_repo_python() -> None:
    if "pytest" in sys.modules:
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


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.system_brief import build_system_brief  # noqa: E402
from pipeline.system_context import infer_context_name  # noqa: E402


def default_packet_path(root: Path) -> Path:
    return root / DEFAULT_PACKET_RELATIVE_PATH


def clear_task_route_packet(root: Path, output_path: Path | None = None) -> Path:
    packet_path = output_path or default_packet_path(root)
    if packet_path.exists():
        packet_path.unlink()
    return packet_path


def _route_status(payload: dict[str, object]) -> str:
    blocker_codes = {issue.get("code") for issue in payload.get("blocking_issues", []) if isinstance(issue, dict)}
    if "ambiguous_route" in blocker_codes:
        return "fallback (ambiguous)"
    if "missing_route" in blocker_codes:
        return "fallback (unmatched)"
    return "matched"


def _render_packet_markdown(*, task_text: str, tool: str, payload: dict[str, object]) -> str:
    warnings = [issue.get("message") for issue in payload.get("warning_issues", []) if isinstance(issue, dict)]
    blockers = [issue.get("message") for issue in payload.get("blocking_issues", []) if isinstance(issue, dict)]
    live_views = payload.get("required_live_views") or []
    doctrine_chain = payload.get("doctrine_chain") or []
    canonical_owners = payload.get("canonical_owners") or []
    verification_steps = payload.get("verification_steps") or []

    lines = [
        "# Startup Task Route",
        f"- Tool: `{tool}`",
        f"- Task: {task_text}",
        f"- Route status: `{_route_status(payload)}`",
        f"- Route id: `{payload['route_id']}`",
        f"- Task kind: `{payload['task_kind']}`",
        f"- Briefing: `{payload['briefing_contract']}`",
        f"- Verification: `{payload['verification_profile']}` via {', '.join(f'`{step}`' for step in verification_steps)}",
        f"- Live views: {', '.join(f'`{view}`' for view in live_views) if live_views else '`none`'}",
        f"- Doctrine: {', '.join(f'`{path}`' for path in doctrine_chain)}",
        f"- Canonical owners: {', '.join(f'`{path}`' for path in canonical_owners)}",
    ]
    if blockers:
        lines.append(f"- Blockers: {' | '.join(str(item) for item in blockers)}")
    if warnings:
        lines.append(f"- Warnings: {' | '.join(str(item) for item in warnings)}")
    return "\n".join(lines) + "\n"


def write_task_route_packet(
    root: Path,
    *,
    task_text: str,
    tool: str,
    briefing_level: str = "mutating",
    output_path: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    packet_path = output_path or default_packet_path(root)
    payload = build_system_brief(
        root,
        task_text=task_text,
        briefing_level=briefing_level,  # type: ignore[arg-type]
        context_name=infer_context_name(root, Path(sys.executable)),
        active_tool=tool,
        active_mode="mutating" if briefing_level == "mutating" else "read-only",
    )
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        _render_packet_markdown(task_text=task_text, tool=tool, payload=payload),
        encoding="utf-8",
    )
    return packet_path, payload


def read_task_route_packet(root: Path, output_path: Path | None = None) -> list[str]:
    packet_path = output_path or default_packet_path(root)
    if not packet_path.exists():
        return []
    return [
        line.rstrip() for line in packet_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a compact startup task-route packet.")
    parser.add_argument("--root", default=None, help="Override repo root")
    parser.add_argument("--task", default=None, help="Natural-language task text")
    parser.add_argument("--tool", default="generic", help="Tool name for metadata")
    parser.add_argument(
        "--briefing-level",
        default="mutating",
        choices=["trivial", "read_only", "non_trivial", "mutating"],
        help="System brief level to use for the packet",
    )
    parser.add_argument("--output", default=None, help="Optional packet output path")
    parser.add_argument("--clear", action="store_true", help="Remove any existing packet and exit")
    parser.add_argument("--format", default="text", choices=["text", "json"], help="CLI output format")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    output_path = Path(args.output).resolve() if args.output else None

    if args.clear or not (args.task and args.task.strip()):
        path = clear_task_route_packet(root, output_path)
        if args.format == "json":
            print(json.dumps({"cleared": True, "path": str(path)}, indent=2, sort_keys=True))
        else:
            print(path)
        return 0

    path, payload = write_task_route_packet(
        root,
        task_text=args.task.strip(),
        tool=args.tool,
        briefing_level=args.briefing_level,
        output_path=output_path,
    )
    if args.format == "json":
        print(json.dumps({"path": str(path), "payload": payload}, indent=2, sort_keys=True))
    else:
        print(path)
    return 0


if __name__ == "__main__":
    # Bootstrap the repo venv only when invoked as CLI. Importing this module
    # (e.g. from a hook) must NOT re-exec the process — see PR #125 postmortem
    # where the silent SystemExit broke session-start.py under system python.
    _ensure_repo_python()
    raise SystemExit(main())
