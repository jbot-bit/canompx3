#!/usr/bin/env python3
"""Check that Codex indexes the Claude capability surfaces it relies on."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Surface:
    name: str
    source_glob: str
    index_file: str
    ignore_parts: tuple[str, ...] = ()


SURFACES: tuple[Surface, ...] = (
    Surface("commands", ".claude/commands/*.md", ".codex/COMMANDS.md"),
    Surface("agents", ".claude/agents/*.md", ".codex/AGENTS.md"),
    Surface("rules", ".claude/rules/*.md", ".codex/RULES.md"),
    Surface("skills", ".claude/skills/*/SKILL.md", ".codex/skills/canompx3-claude-parity/SKILL.md"),
    Surface("hooks", ".claude/hooks/*.py", ".codex/HOOKS.md", ("tests", "state")),
)

REQUIRED_CODEX_FILES: tuple[str, ...] = (
    ".codex/AGENTS.md",
    ".codex/HOOKS.md",
    ".codex/COMMANDS.md",
    ".codex/RULES.md",
    ".codex/WORKFLOWS.md",
    ".codex/INTEGRATIONS.md",
    ".codex/skills/canompx3-claude-parity/SKILL.md",
    ".codex/skills/canompx3-claude-parity/agents/openai.yaml",
    ".agents/skills/canompx3-claude-parity/SKILL.md",
)


def _repo_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _source_paths(root: Path, surface: Surface) -> list[str]:
    paths: list[str] = []
    for path in root.glob(surface.source_glob):
        rel = _repo_rel(path, root)
        if any(part in path.parts for part in surface.ignore_parts):
            continue
        paths.append(rel)
    return sorted(paths)


def check_parity(root: Path = ROOT) -> dict[str, object]:
    missing_files = [rel for rel in REQUIRED_CODEX_FILES if not (root / rel).exists()]
    surfaces: dict[str, dict[str, object]] = {}
    missing_refs: dict[str, list[str]] = {}

    for surface in SURFACES:
        index_path = root / surface.index_file
        text = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
        sources = _source_paths(root, surface)
        missing = [rel for rel in sources if rel not in text]
        surfaces[surface.name] = {
            "source_count": len(sources),
            "index_file": surface.index_file,
            "missing_count": len(missing),
        }
        if missing:
            missing_refs[surface.name] = missing

    ok = not missing_files and not missing_refs
    return {
        "ok": ok,
        "missing_files": missing_files,
        "missing_refs": missing_refs,
        "surfaces": surfaces,
    }


def format_text(report: dict[str, object]) -> str:
    lines = ["CODEX PARITY CHECK"]
    lines.append("OK" if report["ok"] else "FAIL")

    missing_files = report["missing_files"]
    if missing_files:
        lines.append("Missing Codex files:")
        lines.extend(f"- {path}" for path in missing_files)  # type: ignore[union-attr]

    missing_refs = report["missing_refs"]
    if missing_refs:
        lines.append("Missing Claude source references:")
        for surface, paths in missing_refs.items():  # type: ignore[union-attr]
            lines.append(f"- {surface}:")
            lines.extend(f"  - {path}" for path in paths)

    lines.append("Surface coverage:")
    for surface, data in report["surfaces"].items():  # type: ignore[union-attr]
        lines.append(
            f"- {surface}: {data['source_count']} source(s), {data['missing_count']} missing in {data['index_file']}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check_parity(args.root)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_text(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
