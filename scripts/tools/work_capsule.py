#!/usr/bin/env python3
"""CLI for the canonical work capsule read model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.work_capsule import evaluate_current_capsule, format_capsule_text, list_work_capsules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show task-scoped work capsule context")
    parser.add_argument("--root", default=None, help="Override repo root")
    parser.add_argument("--format", default="text", choices=["text", "json"], help="Output format")
    parser.add_argument("--list", action="store_true", help="List available capsules instead of selecting current")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path.cwd().resolve()
    if args.list:
        capsules = list_work_capsules(root)
        payload = [
            {
                "capsule_id": capsule.capsule_id,
                "title": capsule.title,
                "status": capsule.status,
                "branch": capsule.branch,
                "worktree_name": capsule.worktree_name,
                "path": Path(capsule.path).relative_to(root).as_posix(),
            }
            for capsule in capsules
        ]
        if args.format == "json":
            print(json.dumps(payload, indent=2))
        else:
            if not payload:
                print("No work capsules.")
            for item in payload:
                print(f"{item['title']} [{item['status']}] {item['path']}")
        return 0

    summary, issues = evaluate_current_capsule(root)
    if args.format == "json":
        print(json.dumps({"summary": summary, "issues": [issue.__dict__ for issue in issues]}, indent=2))
    else:
        print(format_capsule_text(summary, issues))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
