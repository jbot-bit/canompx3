"""Canonical task-scoped work capsule contract for active workstreams."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

WORK_CAPSULE_DIR_RELATIVE = Path("docs/runtime/capsules")
STAGE_DIR_RELATIVE = Path("docs/runtime/stages")
WORKTREE_META = ".canompx3-worktree.local.json"
LEGACY_WORKTREE_META = ".canompx3-worktree.json"

CapsuleStatus = Literal["proposed", "active", "paused", "handoff", "done"]
IssueLevel = Literal["blocker", "warning", "info"]
VALID_STATUSES: set[str] = {"proposed", "active", "paused", "handoff", "done"}

DEFAULT_AUTHORITIES: tuple[str, ...] = (
    "AGENTS.md",
    "HANDOFF.md",
    "CLAUDE.md",
    "docs/governance/system_authority_map.md",
)


@dataclass(frozen=True)
class WorkCapsule:
    capsule_id: str
    title: str
    status: CapsuleStatus
    branch: str
    worktree_name: str
    tool: str
    task_id: str
    route_id: str
    briefing_level: str
    purpose: str
    summary: str
    objective: str
    next_step: str
    created_at: str
    updated_at: str
    stage_path: str
    task_domains: list[str] = field(default_factory=list)
    authorities: list[str] = field(default_factory=list)
    scope_paths: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    decision_refs: list[str] = field(default_factory=list)
    debt_refs: list[str] = field(default_factory=list)
    history_window: str = "current"
    path: str = ""


@dataclass(frozen=True)
class CapsuleIssue:
    level: IssueLevel
    code: str
    message: str
    detail: str | None = None


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text or "work-capsule"


def read_worktree_metadata(root: Path) -> dict[str, str] | None:
    for meta_name in (WORKTREE_META, LEGACY_WORKTREE_META):
        meta_path = root / meta_name
        if not meta_path.exists():
            continue
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        return {str(key): str(value) for key, value in data.items() if value is not None}
    return None


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError:
        return None


def current_branch(root: Path) -> str:
    result = _run_git(root, "branch", "--show-current")
    if result is None or result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "detached"


def work_capsule_dir(root: Path) -> Path:
    return root / WORK_CAPSULE_DIR_RELATIVE


def stage_dir(root: Path) -> Path:
    return root / STAGE_DIR_RELATIVE


def build_capsule_path(root: Path, name: str) -> Path:
    return work_capsule_dir(root) / f"{slugify(name)}.md"


def build_stage_path(root: Path, name: str) -> Path:
    return stage_dir(root) / f"{slugify(name)}.md"


def _split_front_matter(text: str) -> tuple[str, str] | None:
    match = re.match(r"^\+\+\+\r?\n(.*?)\r?\n\+\+\+\r?\n?(.*)$", text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1), match.group(2)


def _expect_string(metadata: dict[str, object], key: str, *, default: str | None = None) -> str:
    value = metadata.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for {key}")
    return value.strip()


def _expect_string_list(metadata: dict[str, object], key: str) -> list[str]:
    value = metadata.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"Expected list for {key}")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Expected string entries for {key}")
        items.append(item.strip())
    return items


def _normalize_repo_relative_path(value: str, *, key: str) -> str:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"Expected repo-relative path for {key}, got absolute path: {value}")
    normalized = Path(*[part for part in path.parts if part not in ("", ".")])
    if any(part == ".." for part in normalized.parts):
        raise ValueError(f"Expected repo-relative path for {key}, got parent traversal: {value}")
    text = normalized.as_posix()
    if not text:
        raise ValueError(f"Expected repo-relative path for {key}")
    return text


def _expect_repo_relative_path(metadata: dict[str, object], key: str) -> str:
    return _normalize_repo_relative_path(_expect_string(metadata, key), key=key)


def _expect_repo_relative_path_list(metadata: dict[str, object], key: str) -> list[str]:
    return [_normalize_repo_relative_path(item, key=key) for item in _expect_string_list(metadata, key)]


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temp_name = handle.name
    os.replace(temp_name, path)


def read_work_capsule(path: Path) -> WorkCapsule:
    text = path.read_text(encoding="utf-8")
    split = _split_front_matter(text)
    if split is None:
        raise ValueError(f"Missing TOML front matter: {path}")
    front_matter, _body = split
    metadata = tomllib.loads(front_matter)
    status = _expect_string(metadata, "status")
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid capsule status in {path}: {status}")
    return WorkCapsule(
        capsule_id=_expect_string(metadata, "capsule_id"),
        title=_expect_string(metadata, "title"),
        status=status,  # type: ignore[arg-type]
        branch=_expect_string(metadata, "branch"),
        worktree_name=_expect_string(metadata, "worktree_name"),
        tool=_expect_string(metadata, "tool"),
        task_id=_expect_string(metadata, "task_id", default="system_orientation"),
        route_id=_expect_string(metadata, "route_id", default=_expect_string(metadata, "task_id", default="system_orientation")),
        briefing_level=_expect_string(metadata, "briefing_level", default="mutating"),
        purpose=_expect_string(metadata, "purpose"),
        summary=_expect_string(metadata, "summary"),
        objective=_expect_string(metadata, "objective"),
        next_step=_expect_string(metadata, "next_step"),
        created_at=_expect_string(metadata, "created_at"),
        updated_at=_expect_string(metadata, "updated_at"),
        stage_path=_expect_repo_relative_path(metadata, "stage_path"),
        task_domains=_expect_string_list(metadata, "task_domains"),
        authorities=_expect_repo_relative_path_list(metadata, "authorities"),
        scope_paths=_expect_repo_relative_path_list(metadata, "scope_paths"),
        out_of_scope=_expect_repo_relative_path_list(metadata, "out_of_scope"),
        verification_commands=_expect_string_list(metadata, "verification_commands"),
        acceptance_criteria=_expect_string_list(metadata, "acceptance_criteria"),
        risks=_expect_string_list(metadata, "risks"),
        references=_expect_string_list(metadata, "references"),
        decision_refs=_expect_string_list(metadata, "decision_refs"),
        debt_refs=_expect_string_list(metadata, "debt_refs"),
        history_window=_expect_string(metadata, "history_window", default="current"),
        path=str(path),
    )


def list_work_capsules(root: Path) -> list[WorkCapsule]:
    capsule_root = work_capsule_dir(root)
    if not capsule_root.exists():
        return []
    capsules: list[WorkCapsule] = []
    for path in sorted(capsule_root.glob("*.md")):
        try:
            capsules.append(read_work_capsule(path))
        except ValueError:
            continue
    return capsules


def render_stage_markdown(name: str, tool: str, capsule_path: Path) -> str:
    updated = datetime.now(UTC).isoformat()
    return "\n".join(
        [
            "---",
            f"task: {name}",
            "mode: IMPLEMENTATION",
            f"agent: {tool}",
            f"updated: {updated}",
            f"capsule: {capsule_path.as_posix()}",
            "scope_lock: []",
            "---",
            "",
            "# Stage Notes",
            "",
            "Replace the empty `scope_lock` with the concrete files this workstream owns.",
            "",
        ]
    )


def render_work_capsule_markdown(
    *,
    name: str,
    tool: str,
    branch: str,
    purpose: str,
    stage_path: Path,
) -> str:
    created = datetime.now(UTC).isoformat()
    return "\n".join(
        [
            "+++",
            f'capsule_id = "{slugify(name)}"',
            f'title = "{name}"',
            'status = "active"',
            f'branch = "{branch}"',
            f'worktree_name = "{name}"',
            f'tool = "{tool}"',
            'task_id = "system_orientation"',
            'route_id = "system_orientation"',
            'briefing_level = "mutating"',
            f'purpose = "{purpose}"',
            'summary = "Explain the current workstream in one factual sentence."',
            'objective = "Define the concrete outcome this workstream must produce."',
            'next_step = "Replace this with the highest-value next step."',
            f'created_at = "{created}"',
            f'updated_at = "{created}"',
            f'stage_path = "{stage_path.as_posix()}"',
            'task_domains = ["repo_governance"]',
            'authorities = ["AGENTS.md", "HANDOFF.md", "CLAUDE.md", "CODEX.md", "docs/governance/system_authority_map.md"]',
            "scope_paths = []",
            'out_of_scope = [".claude/", "trading logic beyond this workstream unless explicitly added"]',
            "verification_commands = []",
            "acceptance_criteria = []",
            "risks = []",
            "references = []",
            'decision_refs = ["docs/runtime/decision-ledger.md#current"]',
            'debt_refs = ["docs/runtime/debt-ledger.md#open-debt"]',
            'history_window = "current"',
            "+++",
            "",
            "# Context",
            "",
            "Describe why this work exists and what repo/runtime problem it is solving.",
            "",
            "# Design",
            "",
            "Explain the intended approach, invariants, and interfaces.",
            "",
            "# Decision Ledger",
            "",
            "- Record durable choices here.",
            "",
            "# Verification Ledger",
            "",
            "- Record real commands/results here.",
            "",
            "# Open Questions",
            "",
            "- List remaining uncertainties or explicitly state `None`.",
            "",
        ]
    )


def ensure_work_capsule_scaffold(root: Path, *, tool: str, name: str, branch: str, purpose: str) -> tuple[Path, Path]:
    capsule_path = build_capsule_path(root, name)
    stage_path = build_stage_path(root, name)
    if not capsule_path.exists():
        _atomic_write_text(
            capsule_path,
            render_work_capsule_markdown(name=name, tool=tool, branch=branch, purpose=purpose, stage_path=stage_path.relative_to(root)),
        )
    if not stage_path.exists():
        _atomic_write_text(stage_path, render_stage_markdown(name=name, tool=tool, capsule_path=capsule_path.relative_to(root)))
    return capsule_path, stage_path


def evaluate_current_capsule(root: Path) -> tuple[dict | None, list[CapsuleIssue]]:
    issues: list[CapsuleIssue] = []
    metadata = read_worktree_metadata(root)
    if metadata is None:
        return None, issues
    capsule_path_str = metadata.get("capsule_path")
    if not capsule_path_str:
        return None, [CapsuleIssue("warning", "capsule_missing_path", "Managed worktree metadata has no capsule path.")]
    capsule_path = root / capsule_path_str
    if not capsule_path.exists():
        return None, [CapsuleIssue("blocker", "capsule_missing", "Declared work capsule is missing.", capsule_path_str)]
    capsule = read_work_capsule(capsule_path)
    if not capsule.scope_paths:
        issues.append(CapsuleIssue("warning", "capsule_missing_scope", "Work capsule has no scoped file ownership yet."))
    if not capsule.verification_commands:
        issues.append(CapsuleIssue("warning", "capsule_missing_verification", "Work capsule has no declared verification commands yet."))
    return {
        "path": capsule_path_str,
        "task_id": capsule.task_id,
        "route_id": capsule.route_id,
        "briefing_level": capsule.briefing_level,
        "status": capsule.status,
    }, issues
