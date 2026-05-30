#!/usr/bin/env python
"""Report resource-to-literature coverage for local grounding."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = PROJECT_ROOT / "resources" / "INDEX.md"
LOCAL_PC_RESOURCE_ROOT = Path("C:/Users/joshd/canompx3/resources")


@dataclass(frozen=True)
class ResourceCoverage:
    resource: str
    topic: str
    curated_extract: str | None


def _resource_roots() -> list[Path]:
    roots = [PROJECT_ROOT / "resources"]
    env_root = os.environ.get("CANOMPX3_RESOURCE_ROOT")
    if env_root:
        roots.append(Path(env_root))
    if LOCAL_PC_RESOURCE_ROOT.exists():
        roots.append(LOCAL_PC_RESOURCE_ROOT)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(root)
    return deduped


def _resource_exists(resource: str) -> bool:
    clean = resource.rstrip("/")
    for root in _resource_roots():
        if (root / clean).exists():
            return True
    return False


def _parse_index() -> list[ResourceCoverage]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing {INDEX_PATH.relative_to(PROJECT_ROOT)}")

    rows: list[ResourceCoverage] = []
    for line in INDEX_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        resource = cells[0].strip("`")
        topic = cells[1]
        extract_cell = cells[2]
        match = re.search(r"`([^`]+)`", extract_cell)
        rows.append(ResourceCoverage(resource, topic, match.group(1) if match else None))
    return rows


def _format_markdown(rows: list[ResourceCoverage]) -> str:
    covered = [row for row in rows if row.curated_extract]
    missing = [row for row in rows if not row.curated_extract]
    present = [row for row in rows if _resource_exists(row.resource)]
    absent = [row for row in rows if not _resource_exists(row.resource)]
    lines = [
        "# Literature Coverage",
        "",
        f"- Resources indexed: {len(rows)}",
        f"- With curated extract: {len(covered)}",
        f"- Without curated extract: {len(missing)}",
        f"- Resource files present on this machine: {len(present)}",
        f"- Resource files missing from this machine: {len(absent)}",
        "",
        "Note: large `resources/` PDFs are local-PC assets and may not exist on",
        "remote/CI checkouts. If a resource file is missing locally, do not imply",
        "raw-PDF verification; use tracked curated extracts only or rerun on the PC",
        "that has the corpus.",
        "",
    ]
    if absent:
        lines.extend(["## Missing Local Resource File", "", "| Resource | Topic |", "|---|---|"])
        for row in absent:
            lines.append(f"| `{row.resource}` | {row.topic} |")
        lines.append("")

    if missing:
        lines.extend(
            [
                "## Missing Curated Extract",
                "",
                "| Resource | Topic | Required fallback |",
                "|---|---|---|",
            ]
        )
        for row in missing:
            lines.append(
                f"| `{row.resource}` | {row.topic} | Read source directly; for PDFs extract TOC + 3 relevant/mid pages before characterizing. |"
            )
        lines.append("")

    lines.extend(["## Covered", "", "| Resource | Curated extract |", "|---|---|"])
    for row in covered:
        lines.append(f"| `{row.resource}` | `{row.curated_extract}` |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit 1 when any indexed resource lacks a curated extract.",
    )
    parser.add_argument(
        "--fail-on-missing-files",
        action="store_true",
        help="Exit 1 when any indexed resource file is not present on this machine.",
    )
    args = parser.parse_args(argv)

    try:
        rows = _parse_index()
    except FileNotFoundError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1

    print(_format_markdown(rows))
    has_missing = any(not row.curated_extract for row in rows)
    has_missing_files = any(not _resource_exists(row.resource) for row in rows)
    if args.fail_on_missing and has_missing:
        return 1
    if args.fail_on_missing_files and has_missing_files:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
