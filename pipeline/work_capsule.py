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


@dataclass(frozen=True)
class CapsuleSelection:
    capsule: WorkCapsule | None
    issues: list[CapsuleIssue]
    candidates: list[str] = field(default_factory=list)


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
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid metadata object: {path}")

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
        route_id=_expect_string(
            metadata, "route_id", default=_expect_string(metadata, "task_id", default="system_orientation")
        ),
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


def _match_capsule(capsule: WorkCapsule, *, branch: str, worktree_name: str | None) -> tuple[int, str]:
    score = 0
    reason = ""
    if worktree_name and capsule.worktree_name == worktree_name:
        score += 3
        reason = "worktree_name"
    if capsule.branch == branch:
        score += 2
        reason = f"{reason}+branch" if reason else "branch"
    return score, reason


def _parse_stage_capsule_reference(stage_path: Path) -> str | None:
    if not stage_path.exists():
        return None
    for raw in stage_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("capsule:"):
            return line.partition(":")[2].strip().strip('"').strip("'")
    return None


def select_current_capsule(root: Path) -> CapsuleSelection:
    meta = read_worktree_metadata(root)
    branch = current_branch(root)
    local_meta_exists = (root / WORKTREE_META).exists()
    if meta is not None and not local_meta_exists and meta.get("branch") and meta.get("branch") != branch:
        meta = None
    worktree_name = meta.get("name") if meta else None
    capsule_path_hint = meta.get("capsule_path") if meta else None

    issues: list[CapsuleIssue] = []
    if capsule_path_hint:
        candidate = root / capsule_path_hint
        if candidate.exists():
            try:
                capsule = read_work_capsule(candidate)
            except ValueError as exc:
                issues.append(
                    CapsuleIssue(
                        "blocker",
                        "invalid_capsule_hint",
                        "Managed worktree metadata points to an invalid work capsule.",
                        str(exc),
                    )
                )
            else:
                score, reason = _match_capsule(capsule, branch=branch, worktree_name=worktree_name)
                if meta is not None and score <= 0:
                    issues.append(
                        CapsuleIssue(
                            "blocker",
                            "capsule_hint_mismatch",
                            "Managed worktree metadata points to a capsule that does not match this branch/worktree.",
                            f"branch={branch} worktree={worktree_name or '-'} capsule={candidate.name}",
                        )
                    )
                else:
                    issues.append(
                        CapsuleIssue(
                            "info",
                            "capsule_match",
                            "Current work capsule matched.",
                            f"metadata_hint+{reason}" if reason else "metadata_hint",
                        )
                    )
                    return CapsuleSelection(capsule=capsule, issues=issues, candidates=[str(candidate)])
        else:
            issues.append(
                CapsuleIssue(
                    "blocker",
                    "missing_capsule_hint",
                    "Managed worktree metadata points to a missing work capsule.",
                    capsule_path_hint,
                )
            )

    candidates = list_work_capsules(root)
    if not candidates:
        if meta is not None:
            issues.append(
                CapsuleIssue(
                    "blocker",
                    "missing_capsule",
                    "Managed worktree has no work capsule.",
                    str(build_capsule_path(root, worktree_name or branch)),
                )
            )
        return CapsuleSelection(capsule=None, issues=issues, candidates=[])

    scored = []
    for capsule in candidates:
        score, reason = _match_capsule(capsule, branch=branch, worktree_name=worktree_name)
        if score > 0:
            scored.append((score, capsule, reason))
    if not scored:
        if meta is not None:
            issues.append(
                CapsuleIssue(
                    "blocker",
                    "unmatched_capsule",
                    "Managed worktree has capsules, but none match the current branch/worktree.",
                    f"branch={branch} worktree={worktree_name or '-'}",
                )
            )
        return CapsuleSelection(capsule=None, issues=issues, candidates=[capsule.path for capsule in candidates])

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_capsule, best_reason = scored[0]
    tied = [capsule for score, capsule, _reason in scored if score == best_score]
    if len(tied) > 1:
        issues.append(
            CapsuleIssue(
                "blocker",
                "ambiguous_capsule",
                "Multiple work capsules match the current worktree equally.",
                ", ".join(Path(capsule.path).name for capsule in tied),
            )
        )
    else:
        issues.append(CapsuleIssue("info", "capsule_match", "Current work capsule matched.", best_reason))
    return CapsuleSelection(capsule=best_capsule, issues=issues, candidates=[capsule.path for capsule in tied])


def _stage_scope_lock(stage_path: Path) -> list[str]:
    if not stage_path.exists():
        return []
    lines = stage_path.read_text(encoding="utf-8", errors="replace").splitlines()
    values: list[str] = []
    inside = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("scope_lock:"):
            inside = True
            if stripped.endswith("[]"):
                return []
            continue
        if not inside:
            continue
        if stripped.startswith("- "):
            values.append(stripped[2:].strip())
            continue
        if stripped.startswith("---") or (stripped and not raw.startswith("  ")):
            break
    return values


def evaluate_current_capsule(root: Path) -> tuple[dict[str, object] | None, list[CapsuleIssue]]:
    selection = select_current_capsule(root)
    issues = list(selection.issues)
    capsule = selection.capsule
    if capsule is None:
        return None, issues

    if not capsule.authorities:
        issues.append(CapsuleIssue("warning", "capsule_missing_authorities", "Work capsule has no authorities listed."))
    if not capsule.scope_paths:
        issues.append(CapsuleIssue("warning", "capsule_missing_scope", "Work capsule has no scoped paths yet."))
    if not capsule.verification_commands:
        issues.append(
            CapsuleIssue("warning", "capsule_missing_verification", "Work capsule has no verification commands listed.")
        )
    if not capsule.decision_refs:
        issues.append(
            CapsuleIssue("warning", "capsule_missing_decision_refs", "Work capsule has no decision ledger references.")
        )
    if not capsule.debt_refs:
        issues.append(
            CapsuleIssue("warning", "capsule_missing_debt_refs", "Work capsule has no debt ledger references.")
        )
    stage_path = root / capsule.stage_path
    if not stage_path.exists():
        issues.append(
            CapsuleIssue(
                "warning", "capsule_stage_missing", "Work capsule points to a missing stage file.", capsule.stage_path
            )
        )
    else:
        stage_capsule_ref = _parse_stage_capsule_reference(stage_path)
        if stage_capsule_ref and stage_capsule_ref != Path(capsule.path).relative_to(root).as_posix():
            issues.append(
                CapsuleIssue(
                    "warning",
                    "capsule_stage_mismatch",
                    "Stage file capsule reference does not match the selected work capsule.",
                    f"stage={stage_capsule_ref} capsule={Path(capsule.path).relative_to(root).as_posix()}",
                )
            )
        stage_scope = _stage_scope_lock(stage_path)
        missing_from_capsule = [item for item in stage_scope if item not in capsule.scope_paths]
        if missing_from_capsule:
            issues.append(
                CapsuleIssue(
                    "warning",
                    "capsule_scope_drift",
                    "Stage scope_lock contains paths not declared in the work capsule scope.",
                    ", ".join(missing_from_capsule),
                )
            )

    summary = {
        "capsule_id": capsule.capsule_id,
        "title": capsule.title,
        "status": capsule.status,
        "path": Path(capsule.path).relative_to(root).as_posix(),
        "branch": capsule.branch,
        "worktree_name": capsule.worktree_name,
        "tool": capsule.tool,
        "task_id": capsule.task_id,
        "route_id": capsule.route_id,
        "briefing_level": capsule.briefing_level,
        "purpose": capsule.purpose,
        "summary": capsule.summary,
        "objective": capsule.objective,
        "next_step": capsule.next_step,
        "stage_path": capsule.stage_path,
        "task_domain_count": len(capsule.task_domains),
        "authorities_count": len(capsule.authorities),
        "scope_count": len(capsule.scope_paths),
        "verification_count": len(capsule.verification_commands),
        "risk_count": len(capsule.risks),
        "decision_refs": list(capsule.decision_refs),
        "debt_refs": list(capsule.debt_refs),
        "history_window": capsule.history_window,
    }
    return summary, issues


def render_work_capsule_markdown(
    *,
    name: str,
    tool: str,
    branch: str,
    purpose: str,
    stage_path: Path,
    task_id: str = "system_orientation",
    route_id: str = "system_orientation",
    briefing_level: str = "mutating",
    created_at: str | None = None,
) -> str:
    slug = slugify(name)
    now = created_at or datetime.now(UTC).isoformat()
    authorities = list(DEFAULT_AUTHORITIES)
    if tool.lower() == "codex":
        authorities.insert(3, "CODEX.md")

    metadata_lines = [
        "+++",
        f'capsule_id = "{slug}"',
        f'title = "{name}"',
        'status = "active"',
        f'branch = "{branch}"',
        f'worktree_name = "{name}"',
        f'tool = "{tool}"',
        f'task_id = "{task_id}"',
        f'route_id = "{route_id}"',
        f'briefing_level = "{briefing_level}"',
        f'purpose = "{purpose or "Document the workstream purpose explicitly."}"',
        'summary = "Explain the current workstream in one factual sentence."',
        'objective = "Define the concrete outcome this workstream must produce."',
        'next_step = "Replace this with the highest-value next step."',
        f'created_at = "{now}"',
        f'updated_at = "{now}"',
        f'stage_path = "{stage_path.as_posix()}"',
        'task_domains = ["repo_governance"]',
        f"authorities = {json.dumps(authorities)}",
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
    return "\n".join(metadata_lines)


def render_stage_markdown(*, name: str, tool: str, capsule_path: Path, updated_at: str | None = None) -> str:
    now = updated_at or datetime.now(UTC).isoformat()
    lines = [
        "---",
        f"task: {name}",
        "mode: IMPLEMENTATION",
        f"agent: {tool}",
        f"updated: {now}",
        f"capsule: {capsule_path.as_posix()}",
        "scope_lock: []",
        "---",
        "",
        "# Stage Notes",
        "",
        "Replace the empty `scope_lock` with the concrete files this workstream owns.",
        "",
    ]
    return "\n".join(lines)


def ensure_work_capsule_scaffold(
    root: Path,
    *,
    tool: str,
    name: str,
    branch: str,
    purpose: str | None = None,
    task_id: str = "system_orientation",
    route_id: str = "system_orientation",
    briefing_level: str = "mutating",
) -> tuple[Path, Path]:
    capsule_path = build_capsule_path(root, name)
    stage_path = build_stage_path(root, name)
    capsule_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    if not capsule_path.exists():
        _atomic_write_text(
            capsule_path,
            render_work_capsule_markdown(
                name=name,
                tool=tool,
                branch=branch,
                purpose=purpose or "Document the workstream purpose explicitly.",
                stage_path=stage_path.relative_to(root),
                task_id=task_id,
                route_id=route_id,
                briefing_level=briefing_level,
            ),
        )
    if not stage_path.exists():
        _atomic_write_text(
            stage_path, render_stage_markdown(name=name, tool=tool, capsule_path=capsule_path.relative_to(root))
        )
    return capsule_path, stage_path


def format_capsule_text(summary: dict[str, object] | None, issues: list[CapsuleIssue]) -> str:
    lines: list[str] = []
    if summary is None:
        lines.append("Work capsule: none")
    else:
        lines.append(f"Work capsule: {summary['title']} [{summary['status']}]")
        lines.append(f"  Path: {summary['path']}")
        lines.append(f"  Task: {summary['task_id']} / {summary['route_id']} [{summary['briefing_level']}]")
        lines.append(f"  Purpose: {summary['purpose']}")
        lines.append(f"  Next: {summary['next_step']}")
        lines.append(
            "  Counts: "
            f"authorities={summary['authorities_count']} "
            f"task_domains={summary['task_domain_count']} "
            f"scope={summary['scope_count']} "
            f"verification={summary['verification_count']} "
            f"risks={summary['risk_count']}"
        )
    if issues:
        lines.append("Issues:")
        for issue in issues:
            detail = f" ({issue.detail})" if issue.detail else ""
            lines.append(f"  - [{issue.level}] {issue.message}{detail}")
    return "\n".join(lines)
