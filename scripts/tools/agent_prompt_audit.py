#!/usr/bin/env python3
"""Audit project agent prompts for silent skips, stale facts, and tool drift."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

AGENT_DIRS = (Path(".claude/agents"), Path(".codex/agents"))
RESEARCH_TERMS = re.compile(
    r"\b(research|methodolog|strategy claim|validation claim|multiplicity|leakage|FDR|DSR|MinBTL)\b",
    re.IGNORECASE,
)
LOCAL_LIT_TERMS = re.compile(
    r"docs/institutional/literature/|resources/|RESEARCH_RULES\.md",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    path: Path
    rule: str
    message: str

    def format(self) -> str:
        return f"{self.path}: {self.rule}: {self.message}"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _markdown_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}
    data: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def _tools_from_frontmatter(value: str) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def _toml_value(text: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}\s*=\s*[\"']([^\"']+)[\"']", text)
    return match.group(1) if match else ""


def _is_read_only_markdown(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "read-only",
            "cannot edit",
            "you never edit",
            "never edits files",
            "no edit, no write",
            "no edit / no write",
            "stay read-only",
        )
    )


def _audit_markdown(path: Path) -> list[Finding]:
    text = _read_text(path)
    findings: list[Finding] = []
    frontmatter = _markdown_frontmatter(text)
    tools = _tools_from_frontmatter(frontmatter.get("tools", ""))

    if _is_read_only_markdown(text) and ({"Edit", "Write"} & tools):
        findings.append(
            Finding(
                path,
                "read-only-tools",
                "read-only agent grants Edit/Write tools",
            )
        )

    return findings


def _audit_toml(path: Path) -> list[Finding]:
    text = _read_text(path)
    findings: list[Finding] = []
    sandbox_mode = _toml_value(text, "sandbox_mode")
    if "read-only" in text.lower() and sandbox_mode and sandbox_mode != "read-only":
        findings.append(
            Finding(
                path,
                "read-only-sandbox",
                f"read-only custom agent uses sandbox_mode={sandbox_mode!r}",
            )
        )
    return findings


def audit_path(path: Path) -> list[Finding]:
    text = _read_text(path)
    findings: list[Finding] = []

    if re.search(r"\bskip silently\b|\bsilently skip\b", text, re.IGNORECASE):
        findings.append(Finding(path, "silent-skip", "silent skipped checks are not allowed"))

    if re.search(r"\b(all\s+4|4\s+instruments|currently\s+\d+\s*:\s*)\b", text, re.IGNORECASE):
        findings.append(
            Finding(
                path,
                "volatile-fact",
                "hardcoded volatile instrument count; query canonical source instead",
            )
        )

    for line in text.splitlines():
        if re.search(r"\bprobably fine\b|\bprobably ready\b", line, re.IGNORECASE) and not re.search(
            r"\b(never|refuse|saying)\b", line, re.IGNORECASE
        ):
            findings.append(Finding(path, "soft-claim", "soft readiness language must be evidence-labeled"))

    if RESEARCH_TERMS.search(text) and not LOCAL_LIT_TERMS.search(text):
        findings.append(
            Finding(
                path,
                "research-lit-grounding",
                "research/methodology agent lacks RESEARCH_RULES or local literature/resource grounding",
            )
        )

    if path.suffix == ".md":
        findings.extend(_audit_markdown(path))
    elif path.suffix == ".toml":
        findings.extend(_audit_toml(path))

    return findings


def iter_agent_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel_dir in AGENT_DIRS:
        directory = root / rel_dir
        if not directory.exists():
            continue
        files.extend(sorted(path for path in directory.iterdir() if path.suffix in {".md", ".toml"}))
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    findings: list[Finding] = []
    for path in iter_agent_files(args.root):
        findings.extend(audit_path(path))

    if findings:
        for finding in findings:
            print(finding.format())
        return 1

    print("agent prompt audit: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
