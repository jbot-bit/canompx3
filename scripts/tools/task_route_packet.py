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
METADATA_START = "<!-- task-route-metadata"
METADATA_END = "-->"


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


_ensure_repo_python()
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


def _bounded_text(value: str, *, limit: int = 120) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _build_packet_metadata(
    *,
    task_text: str,
    tool: str,
    briefing_level: str,
    queue_item: str | None,
    override_note: str | None,
) -> dict[str, object]:
    return {
        "version": 2,
        "tool": tool,
        "task_text": task_text,
        "briefing_level": briefing_level,
        "queue_item": queue_item,
        "override_note": override_note,
    }


def _render_metadata_block(metadata: dict[str, object]) -> str:
    return f"{METADATA_START}\n{json.dumps(metadata, sort_keys=True)}\n{METADATA_END}"


def _extract_metadata_block(text: str) -> dict[str, object] | None:
    start = text.find(METADATA_START)
    if start == -1:
        return None
    block_start = start + len(METADATA_START)
    end = text.find(METADATA_END, block_start)
    if end == -1:
        return None
    raw = text[block_start:end].strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _read_packet_text(root: Path, output_path: Path | None = None) -> str | None:
    packet_path = output_path or default_packet_path(root)
    if not packet_path.exists():
        return None
    return packet_path.read_text(encoding="utf-8", errors="replace")


def _render_packet_markdown(
    *,
    task_text: str,
    tool: str,
    payload: dict[str, object],
    metadata: dict[str, object],
) -> str:
    warnings = [issue.get("message") for issue in payload.get("warning_issues", []) if isinstance(issue, dict)]
    blockers = [issue.get("message") for issue in payload.get("blocking_issues", []) if isinstance(issue, dict)]
    live_views = payload.get("required_live_views") or []
    doctrine_chain = payload.get("doctrine_chain") or []
    canonical_owners = payload.get("canonical_owners") or []
    verification_steps = payload.get("verification_steps") or []
    queue_item = metadata.get("queue_item")
    override_note = metadata.get("override_note")

    lines = [
        "# Startup Task Route",
        _render_metadata_block(metadata),
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
    if isinstance(queue_item, str) and queue_item.strip():
        lines.append(f"- Queue item: `{queue_item}`")
    if isinstance(override_note, str) and override_note.strip():
        lines.append(f"- Override note: {_bounded_text(override_note)}")
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
    queue_item: str | None = None,
    override_note: str | None = None,
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
    metadata = _build_packet_metadata(
        task_text=task_text,
        tool=tool,
        briefing_level=briefing_level,
        queue_item=queue_item,
        override_note=override_note,
    )
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        _render_packet_markdown(task_text=task_text, tool=tool, payload=payload, metadata=metadata),
        encoding="utf-8",
    )
    return packet_path, payload


def read_task_route_packet(root: Path, output_path: Path | None = None) -> list[str]:
    text = _read_packet_text(root, output_path)
    if text is None:
        return []
    lines: list[str] = []
    in_metadata = False
    for raw in text.splitlines():
        stripped = raw.rstrip()
        if stripped == METADATA_START:
            in_metadata = True
            continue
        if in_metadata:
            if stripped == METADATA_END:
                in_metadata = False
            continue
        if stripped:
            lines.append(stripped)
    return lines


def read_task_route_packet_metadata(root: Path, output_path: Path | None = None) -> dict[str, object]:
    text = _read_packet_text(root, output_path)
    if text is None:
        return {}
    metadata = _extract_metadata_block(text)
    return metadata or {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a compact startup task-route packet.")
    parser.add_argument("--root", default=None, help="Override repo root")
    parser.add_argument("--task", default=None, help="Natural-language task text")
    parser.add_argument("--tool", default="generic", help="Tool name for metadata")
    parser.add_argument("--queue-item", default=None, help="Explicit canonical queue item id for startup metadata")
    parser.add_argument("--override-note", default=None, help="Optional queue override note for startup metadata")
    parser.add_argument(
        "--briefing-level",
        default="mutating",
        choices=["trivial", "read_only", "non_trivial", "mutating"],
        help="System brief level to use for the packet",
    )
    parser.add_argument("--output", default=None, help="Optional packet output path")
    parser.add_argument("--clear", action="store_true", help="Remove any existing packet and exit")
    parser.add_argument("--read", action="store_true", help="Read the current packet instead of writing one")
    parser.add_argument(
        "--field",
        default=None,
        choices=["briefing_level", "override_note", "queue_item", "task_text", "tool", "version"],
        help="Structured packet metadata field to emit when reading",
    )
    parser.add_argument("--format", default="text", choices=["text", "json"], help="CLI output format")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    output_path = Path(args.output).resolve() if args.output else None

    if args.read:
        metadata = read_task_route_packet_metadata(root, output_path)
        lines = read_task_route_packet(root, output_path)
        if args.field:
            value = metadata.get(args.field)
            if args.format == "json":
                print(json.dumps({"field": args.field, "value": value}, indent=2, sort_keys=True))
            elif value is not None:
                print(value)
            return 0
        if args.format == "json":
            print(
                json.dumps(
                    {"path": str(output_path or default_packet_path(root)), "metadata": metadata, "lines": lines},
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            for line in lines:
                print(line)
        return 0

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
        queue_item=args.queue_item,
        override_note=args.override_note,
        output_path=output_path,
    )
    if args.format == "json":
        print(
            json.dumps(
                {
                    "path": str(path),
                    "metadata": read_task_route_packet_metadata(root, output_path),
                    "payload": payload,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
