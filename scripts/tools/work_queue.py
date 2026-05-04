#!/usr/bin/env python3
"""CLI for the canonical active-work queue and local session leases."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.work_queue import (
    claim_item,
    close_first_open_items,
    close_item,
    fresh_leases,
    heartbeat_lease,
    lease_path,
    load_queue,
    queue_path,
    record_override,
    release_session,
    render_handoff_text,
    stale_items,
    top_baton_items,
    write_rendered_handoff,
)


def _root() -> Path:
    return Path.cwd().resolve()


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=_root(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _default_session_id(tool: str) -> str:
    branch = _run_git("branch", "--show-current")
    head = _run_git("rev-parse", "--short", "HEAD")
    return f"{tool}:{branch}:{head}"


def cmd_status(_args: argparse.Namespace) -> int:
    root = _root()
    queue = load_queue(root)
    print(f"Queue: {queue_path(root)}")
    print(f"Lease file: {lease_path(root)}")
    print(f"Open items: {len([item for item in queue.items if item.is_open])}")
    print("Top baton:")
    for item in top_baton_items(queue):
        print(f"  - {item.id} [{item.priority}/{item.status}] {item.title}")
    close_first = close_first_open_items(queue)
    if close_first:
        print("Close-first open:")
        for item in close_first:
            print(f"  - {item.id} [{item.priority}]")
    stale = stale_items(queue)
    if stale:
        print("Stale:")
        for item in stale:
            print(f"  - {item.id} (last_verified_at={item.last_verified_at})")
    leases = fresh_leases(root)
    if leases:
        print("Fresh leases:")
        for lease in leases:
            claimed = ", ".join(lease.claimed_item_ids) or "<none>"
            print(f"  - {lease.session_id} {lease.tool}@{lease.branch} -> {claimed}")
    return 0


def cmd_render_handoff(args: argparse.Namespace) -> int:
    root = _root()
    if args.write:
        path = write_rendered_handoff(root, tool=args.tool, date=args.date, summary=args.summary)
        print(path.relative_to(root).as_posix())
        return 0
    print(render_handoff_text(root, tool=args.tool, date=args.date, summary=args.summary))
    return 0


def cmd_claim(args: argparse.Namespace) -> int:
    root = _root()
    session_id = args.session_id or _default_session_id(args.tool)
    branch = args.branch or _run_git("branch", "--show-current")
    try:
        lease = claim_item(
            root,
            item_id=args.item,
            session_id=session_id,
            tool=args.tool,
            branch=branch,
            worktree=str(PROJECT_ROOT),
            override_note=args.override_note,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{lease.session_id}: {', '.join(lease.claimed_item_ids)}")
    return 0


def cmd_heartbeat(args: argparse.Namespace) -> int:
    root = _root()
    session_id = args.session_id or _default_session_id(args.tool)
    lease = heartbeat_lease(root, session_id=session_id)
    if lease is None:
        print(f"No lease found for {session_id}", file=sys.stderr)
        return 1
    print(f"{lease.session_id}: {lease.last_heartbeat_at}")
    return 0


def cmd_close(args: argparse.Namespace) -> int:
    item = close_item(_root(), item_id=args.item, status="closed")
    print(f"{item.id}: {item.status}")
    return 0


def cmd_supersede(args: argparse.Namespace) -> int:
    item = close_item(_root(), item_id=args.item, status="superseded", override_note=f"superseded by {args.by}")
    print(f"{item.id}: {item.status} ({args.by})")
    return 0


def cmd_override(args: argparse.Namespace) -> int:
    item = record_override(_root(), item_id=args.item, note=args.reason)
    print(f"{item.id}: override recorded")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    root = _root()
    session_id = args.session_id or _default_session_id(args.tool)
    release_session(root, session_id=session_id)
    print(f"released {session_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show queue summary")
    status.set_defaults(func=cmd_status)

    render = sub.add_parser("render-handoff", help="Render the thin baton from the queue")
    render.add_argument("--write", action="store_true", help="Write to HANDOFF.md instead of stdout")
    render.add_argument("--tool", default=None)
    render.add_argument("--date", default=None)
    render.add_argument("--summary", default=None)
    render.set_defaults(func=cmd_render_handoff)

    claim = sub.add_parser("claim", help="Claim a queue item in the local lease file")
    claim.add_argument("--item", required=True)
    claim.add_argument("--tool", default="codex")
    claim.add_argument("--session-id", default=None)
    claim.add_argument("--branch", default=None)
    claim.add_argument("--override-note", default=None)
    claim.set_defaults(func=cmd_claim)

    heartbeat = sub.add_parser("heartbeat", help="Refresh the heartbeat for a session lease")
    heartbeat.add_argument("--tool", default="codex")
    heartbeat.add_argument("--session-id", default=None)
    heartbeat.set_defaults(func=cmd_heartbeat)

    close = sub.add_parser("close", help="Close a queue item")
    close.add_argument("--item", required=True)
    close.set_defaults(func=cmd_close)

    supersede = sub.add_parser("supersede", help="Mark a queue item superseded")
    supersede.add_argument("--item", required=True)
    supersede.add_argument("--by", required=True)
    supersede.set_defaults(func=cmd_supersede)

    override = sub.add_parser("override", help="Record an override note on a queue item")
    override.add_argument("--item", required=True)
    override.add_argument("--reason", required=True)
    override.set_defaults(func=cmd_override)

    release = sub.add_parser("release", help="Release a local session lease")
    release.add_argument("--tool", default="codex")
    release.add_argument("--session-id", default=None)
    release.set_defaults(func=cmd_release)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
