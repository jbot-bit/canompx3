#!/usr/bin/env python3
"""List stale docs under `docs/plans/active/` by `last_reviewed` age."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_PLANS_ROOT = PROJECT_ROOT / "docs" / "plans" / "active"


@dataclass
class StaleDoc:
    path: Path
    owner: str
    last_reviewed: str
    age_days: int


def _parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}
    block = text[4:end]
    data: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data


def _iter_markdown_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))


def _collect_stale_docs(threshold_days: int) -> list[StaleDoc]:
    now = datetime.now(UTC).date()
    stale: list[StaleDoc] = []
    for path in _iter_markdown_files(ACTIVE_PLANS_ROOT):
        meta = _parse_frontmatter(path)
        # Active folder is expected to contain only active docs.
        # Keep this guard so accidental misplacement is ignored safely.
        if meta.get("status") != "active":
            continue
        last_reviewed = meta.get("last_reviewed", "")
        if not last_reviewed:
            continue
        try:
            reviewed = datetime.strptime(last_reviewed, "%Y-%m-%d").date()
        except ValueError:
            continue
        age_days = (now - reviewed).days
        if age_days > threshold_days:
            stale.append(
                StaleDoc(
                    path=path.relative_to(PROJECT_ROOT),
                    owner=meta.get("owner", ""),
                    last_reviewed=last_reviewed,
                    age_days=age_days,
                )
            )
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-days", type=int, default=14, help="Age threshold in days (default: 14)")
    args = parser.parse_args()

    stale_docs = _collect_stale_docs(args.threshold_days)
    if not stale_docs:
        print(f"No stale active docs found (threshold={args.threshold_days} days).")
        return 0

    print(f"Stale active docs (threshold={args.threshold_days} days):")
    for item in stale_docs:
        owner = item.owner or "unowned"
        print(f"- {item.path} | owner={owner} | last_reviewed={item.last_reviewed} | age={item.age_days}d")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
