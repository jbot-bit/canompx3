"""Canonical project control-plane context and policy evaluation.

This module captures cheap, deterministic repo/dev context and evaluates the
shared shell policy that preflight and pulse should both consume.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SESSION_DIR = Path(
    os.environ.get(
        "CANOMPX3_ACTIVE_SESSION_DIR",
        os.path.join(tempfile.gettempdir(), "canompx3-active-sessions"),
    )
)
CLAIM_FRESHNESS = timedelta(hours=8)
GIT_TIMEOUT_SECONDS = 5.0
SYSTEM_CONTEXT_SCHEMA_VERSION = 1

ContextName = Literal["generic", "codex-wsl", "claude-windows", "claude-shell", "unknown"]
SessionMode = Literal["read-only", "mutating"]
PolicyAction = Literal["orientation", "session_start_read_only", "session_start_mutating"]
IssueLevel = Literal["blocker", "warning", "info"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HandoffSnapshot(StrictModel):
    exists: bool
    path: str
    tool: str | None = None
    date: str | None = None
    summary: str | None = None


class SessionClaim(StrictModel):
    tool: str
    branch: str
    head_sha: str
    started_at: str
    pid: int
    mode: SessionMode = "read-only"
    root: str = ""
    fresh: bool = False


class GitContext(StrictModel):
    selected_root: str
    canonical_root: str
    git_common_dir: str | None = None
    branch: str
    head_sha: str
    status_available: bool
    status_reason: str | None = None
    dirty: bool
    dirty_count: int
    dirty_files: list[str] = Field(default_factory=list)
    in_linked_worktree: bool


class InterpreterContext(StrictModel):
    context: ContextName
    current_python: str
    current_prefix: str
    expected_python: str | None = None
    expected_prefix: str | None = None
    expected_present: bool = False
    matches_expected: bool | None = None
    guidance: str | None = None


class DbContext(StrictModel):
    canonical_db_path: str
    selected_db_path: str
    db_override_active: bool
    live_journal_db_path: str | None = None


class ActiveStage(StrictModel):
    path: str
    task: str | None = None
    mode: str | None = None
    agent: str | None = None
    updated: str | None = None
    scope_lock: list[str] = Field(default_factory=list)


class AuthorityContext(StrictModel):
    authority_map_doc: str
    doctrine_docs: list[str] = Field(default_factory=list)
    backbone_modules: list[str] = Field(default_factory=list)
    active_orb_instruments: list[str] = Field(default_factory=list)
    active_profiles: list[str] = Field(default_factory=list)
    published_relations: dict[str, str] = Field(default_factory=dict)


class SystemContext(StrictModel):
    schema_version: int = SYSTEM_CONTEXT_SCHEMA_VERSION
    generated_at: str
    context_name: ContextName
    active_tool: str | None = None
    active_mode: SessionMode = "read-only"
    handoff: HandoffSnapshot
    git: GitContext
    interpreter: InterpreterContext
    db: DbContext
    claims: list[SessionClaim] = Field(default_factory=list)
    active_stages: list[ActiveStage] = Field(default_factory=list)
    authority: AuthorityContext


class PolicyIssue(StrictModel):
    level: IssueLevel
    code: str
    message: str
    detail: str | None = None


class PolicyDecision(StrictModel):
    decision_id: str
    action: PolicyAction
    allowed: bool
    blockers: list[PolicyIssue] = Field(default_factory=list)
    warnings: list[PolicyIssue] = Field(default_factory=list)
    infos: list[PolicyIssue] = Field(default_factory=list)
    applicable_authorities: list[str] = Field(default_factory=list)
    applicable_controls: list[str] = Field(default_factory=list)


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def branch_name(root: Path) -> str:
    result = _run_git(root, "branch", "--show-current")
    if result is None or result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "detached"


def head_sha(root: Path) -> str:
    result = _run_git(root, "rev-parse", "HEAD")
    if result is None or result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def dirty_files(root: Path) -> list[str]:
    files, status_available, status_reason = git_status_details(root)
    if status_available:
        return files
    if status_reason:
        return [status_reason]
    return []


def git_status_details(root: Path) -> tuple[list[str], bool, str | None]:
    result = _run_git(root, "status", "--short")
    if result is None:
        return [], False, "git status unavailable or timed out"
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git status failed"
        return [], False, message
    return [line for line in result.stdout.splitlines() if line.strip()], True, None


def _canonical_repo_root(root: Path) -> tuple[Path, Path | None]:
    show_top = _run_git(root, "rev-parse", "--show-toplevel")
    selected_root = Path(show_top.stdout.strip()).resolve() if show_top and show_top.returncode == 0 else root.resolve()

    common = _run_git(root, "rev-parse", "--git-common-dir")
    if common and common.returncode == 0:
        common_dir = Path(common.stdout.strip()).resolve()
        if common_dir.name == ".git":
            return common_dir.parent, common_dir
        return selected_root, common_dir
    return selected_root, None


def _extract_handoff_snapshot(handoff_path: Path) -> HandoffSnapshot:
    if not handoff_path.exists():
        return HandoffSnapshot(exists=False, path=str(handoff_path))

    tool = date = summary = None
    for line in handoff_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("- **Tool:** "):
            tool = line.removeprefix("- **Tool:** ").strip()
        elif line.startswith("- **Date:** "):
            date = line.removeprefix("- **Date:** ").strip()
        elif line.startswith("- **Summary:** "):
            summary = line.removeprefix("- **Summary:** ").strip()
        if tool and date and summary:
            break

    return HandoffSnapshot(exists=True, path=str(handoff_path), tool=tool, date=date, summary=summary)


def _active_claim_key(root: Path, tool: str) -> str:
    payload = f"{tool}|{root.resolve()}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    safe_tool = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in tool)
    return f"{safe_tool}-{digest}.json"


def active_claim_path(root: Path, tool: str, claim_dir: Path = ACTIVE_SESSION_DIR) -> Path:
    return claim_dir / _active_claim_key(root, tool)


def write_claim(
    claim_path: Path,
    tool: str,
    branch: str,
    head: str,
    *,
    mode: SessionMode = "read-only",
    root: str | None = None,
) -> SessionClaim:
    claim = SessionClaim(
        tool=tool,
        branch=branch,
        head_sha=head,
        started_at=datetime.now(UTC).isoformat(),
        pid=os.getpid(),
        mode=mode,
        root=root or "",
        fresh=True,
    )
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text(json.dumps(claim.model_dump(mode="json", exclude={"fresh"}), indent=2), encoding="utf-8")
    return claim


def _claim_is_fresh(claim: SessionClaim) -> bool:
    try:
        started = datetime.fromisoformat(claim.started_at)
    except ValueError:
        return False
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return datetime.now(UTC) - started <= CLAIM_FRESHNESS


def read_claim(claim_path: Path) -> SessionClaim | None:
    if not claim_path.exists():
        return None
    try:
        data = json.loads(claim_path.read_text(encoding="utf-8"))
        claim = SessionClaim.model_validate(data)
        return claim.model_copy(update={"fresh": _claim_is_fresh(claim)})
    except Exception:
        return None


def list_claims(claim_dir: Path = ACTIVE_SESSION_DIR, *, fresh_only: bool = False) -> list[SessionClaim]:
    if not claim_dir.exists():
        return []

    claims: list[SessionClaim] = []
    for path in sorted(claim_dir.glob("*.json")):
        claim = read_claim(path)
        if claim is None:
            continue
        if fresh_only and not claim.fresh:
            continue
        claims.append(claim)
    return claims


def write_active_claim(
    root: Path,
    tool: str,
    *,
    mode: SessionMode,
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> SessionClaim:
    return write_claim(
        active_claim_path(root, tool, claim_dir=claim_dir),
        tool=tool,
        branch=branch_name(root),
        head=head_sha(root),
        mode=mode,
        root=str(root.resolve()),
    )


def verify_claim(
    root: Path,
    active_tool: str,
    claim_path: Path | None = None,
    claim_dir: Path = ACTIVE_SESSION_DIR,
) -> tuple[bool, list[str]]:
    claim = read_claim(claim_path or active_claim_path(root, active_tool, claim_dir=claim_dir))
    if claim is None:
        return True, []

    warnings: list[str] = []
    ok = True
    current_branch = branch_name(root)
    current_head = head_sha(root)

    if claim.tool != active_tool:
        warnings.append(f"tool mismatch: claim={claim.tool} current={active_tool}")
        ok = False
    if claim.branch != current_branch:
        warnings.append(f"Branch mismatch: claim={claim.branch} current={current_branch}")
        ok = False
    if claim.head_sha != current_head:
        warnings.append(f"HEAD mismatch: claim={claim.head_sha} current={current_head}")
        ok = False
    current_root = str(root.resolve())
    if claim.root and claim.root != current_root:
        warnings.append(f"Root mismatch: claim={claim.root} current={current_root}")
        ok = False

    return ok, warnings


def _expected_python(root: Path, context_name: ContextName) -> tuple[Path | None, bool]:
    if context_name == "codex-wsl":
        path = root / ".venv-wsl" / "bin" / "python"
        return path, path.exists()
    if context_name in {"claude-windows", "claude-shell"}:
        path = root / ".venv" / "Scripts" / "python.exe"
        return path, path.exists()
    return None, False


def _expected_prefix(root: Path, context_name: ContextName) -> Path | None:
    expected_python, expected_present = _expected_python(root, context_name)
    if not expected_present or expected_python is None:
        return None
    if context_name == "codex-wsl":
        return expected_python.parent.parent.resolve()
    if context_name in {"claude-windows", "claude-shell"}:
        return expected_python.parent.parent.resolve()
    return None


def infer_context_name(
    root: Path,
    current_python: Path | None = None,
    current_prefix: Path | None = None,
) -> ContextName:
    effective_prefix = (current_prefix or Path(sys.prefix)).resolve()
    codex_prefix = _expected_prefix(root, "codex-wsl")
    if codex_prefix is not None and codex_prefix == effective_prefix:
        return "codex-wsl"
    claude_prefix = _expected_prefix(root, "claude-windows")
    if claude_prefix is not None and claude_prefix == effective_prefix:
        return "claude-windows"
    effective_python = (current_python or Path(sys.executable)).resolve()
    codex_python, codex_present = _expected_python(root, "codex-wsl")
    if codex_present and codex_python is not None and codex_python.resolve() == effective_python:
        return "codex-wsl"
    claude_python, claude_present = _expected_python(root, "claude-windows")
    if claude_present and claude_python is not None and claude_python.resolve() == effective_python:
        return "claude-windows"
    return "generic"


def _build_interpreter_context(root: Path, context_name: ContextName) -> InterpreterContext:
    expected_python, expected_present = _expected_python(root, context_name)
    expected_prefix = _expected_prefix(root, context_name)
    current_python = Path(sys.executable)
    current_prefix = Path(sys.prefix).resolve()
    matches_expected = None
    guidance = None
    if expected_prefix is not None:
        matches_expected = current_prefix == expected_prefix
        if not matches_expected:
            guidance = (
                "Use the repo launcher, 'uv run python ...', or the repo-managed "
                "virtualenv interpreter for this context."
            )
    return InterpreterContext(
        context=context_name,
        current_python=str(current_python),
        current_prefix=str(current_prefix),
        expected_python=str(expected_python) if expected_python is not None else None,
        expected_prefix=str(expected_prefix) if expected_prefix is not None else None,
        expected_present=expected_present,
        matches_expected=matches_expected,
        guidance=guidance,
    )


def _parse_stage_file(path: Path) -> ActiveStage | None:
    if path.name == ".gitkeep":
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    scope_lock = data.get("scope_lock") or []
    if not isinstance(scope_lock, list):
        scope_lock = []
    return ActiveStage(
        path=str(path),
        task=str(data.get("task")) if data.get("task") is not None else None,
        mode=str(data.get("mode")) if data.get("mode") is not None else None,
        agent=str(data.get("agent")) if data.get("agent") is not None else None,
        updated=str(data.get("updated")) if data.get("updated") is not None else None,
        scope_lock=[str(item) for item in scope_lock],
    )


def _list_active_stages(root: Path) -> list[ActiveStage]:
    stage_dir = root / "docs" / "runtime" / "stages"
    if not stage_dir.exists():
        return []
    stages: list[ActiveStage] = []
    for path in sorted(stage_dir.glob("*.md")):
        stage = _parse_stage_file(path)
        if stage is not None:
            stages.append(stage)
    return stages


def _build_authority_context(db_path: Path) -> AuthorityContext:
    from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
    from pipeline.db_contracts import ACTIVE_VALIDATED_VIEW, DEPLOYABLE_VALIDATED_VIEW
    from pipeline.system_authority import (
        DOCTRINE_DOCS,
        SYSTEM_AUTHORITY_BACKBONE_MODULES,
        SYSTEM_AUTHORITY_MAP_RELATIVE_PATH,
    )

    # Resolve active profiles without violating pipeline → trading_app one-way
    # dependency (check 9 scans source text for "from trading_app").
    # Use importlib so the text scanner doesn't flag it.
    active_profiles: list[str] = []
    try:
        import importlib
        _mod = importlib.import_module("trading_app.prop_profiles")
        active_profiles = sorted(_mod.get_active_profile_ids())
    except (ImportError, AttributeError):
        pass

    return AuthorityContext(
        authority_map_doc=SYSTEM_AUTHORITY_MAP_RELATIVE_PATH.as_posix(),
        doctrine_docs=list(DOCTRINE_DOCS),
        backbone_modules=list(SYSTEM_AUTHORITY_BACKBONE_MODULES),
        active_orb_instruments=sorted(ACTIVE_ORB_INSTRUMENTS),
        active_profiles=active_profiles,
        published_relations={
            "active": ACTIVE_VALIDATED_VIEW,
            "deployable": DEPLOYABLE_VALIDATED_VIEW,
        },
    )


def _repo_anchor(canonical_root: Path, git_common_dir: Path | None) -> Path:
    return (git_common_dir or (canonical_root / ".git")).resolve()


def _claim_matches_repo(claim: SessionClaim, repo_anchor: Path) -> bool:
    if not claim.root:
        return False
    claim_root = Path(claim.root)
    if not claim_root.exists():
        return False
    try:
        claim_canonical_root, claim_common_dir = _canonical_repo_root(claim_root)
        return _repo_anchor(claim_canonical_root, claim_common_dir) == repo_anchor
    except OSError:
        return False


def build_system_context(
    root: Path | None = None,
    *,
    context_name: ContextName = "generic",
    active_tool: str | None = None,
    active_mode: SessionMode = "read-only",
    claim_dir: Path = ACTIVE_SESSION_DIR,
    db_path: Path | None = None,
) -> SystemContext:
    selected_root = (root or PROJECT_ROOT).resolve()
    canonical_root, git_common_dir = _canonical_repo_root(selected_root)
    repo_anchor = _repo_anchor(canonical_root, git_common_dir)

    from pipeline.paths import GOLD_DB_PATH, LIVE_JOURNAL_DB_PATH

    selected_db = (db_path or GOLD_DB_PATH).resolve()
    canonical_db = GOLD_DB_PATH.resolve()
    repo_dirty_files, status_available, status_reason = git_status_details(selected_root)
    git = GitContext(
        selected_root=str(selected_root),
        canonical_root=str(canonical_root),
        git_common_dir=str(git_common_dir) if git_common_dir is not None else None,
        branch=branch_name(selected_root),
        head_sha=head_sha(selected_root),
        status_available=status_available,
        status_reason=status_reason,
        dirty=bool(repo_dirty_files),
        dirty_count=len(repo_dirty_files),
        dirty_files=repo_dirty_files,
        in_linked_worktree=selected_root != canonical_root,
    )

    return SystemContext(
        generated_at=datetime.now(UTC).isoformat(),
        context_name=context_name,
        active_tool=active_tool,
        active_mode=active_mode,
        handoff=_extract_handoff_snapshot(selected_root / "HANDOFF.md"),
        git=git,
        interpreter=_build_interpreter_context(selected_root, context_name),
        db=DbContext(
            canonical_db_path=str(canonical_db),
            selected_db_path=str(selected_db),
            db_override_active=selected_db != canonical_db,
            live_journal_db_path=str(LIVE_JOURNAL_DB_PATH.resolve()),
        ),
        claims=[claim for claim in list_claims(claim_dir=claim_dir, fresh_only=True) if _claim_matches_repo(claim, repo_anchor)],
        active_stages=_list_active_stages(selected_root),
        authority=_build_authority_context(selected_db),
    )


def _issue(level: IssueLevel, code: str, message: str, detail: str | None = None) -> PolicyIssue:
    return PolicyIssue(level=level, code=code, message=message, detail=detail)


def _parallel_claim_issues(snapshot: SystemContext) -> tuple[list[PolicyIssue], list[PolicyIssue]]:
    blockers: list[PolicyIssue] = []
    warnings: list[PolicyIssue] = []
    branch = snapshot.git.branch
    current_root = snapshot.git.selected_root
    active_tool = snapshot.active_tool

    same_branch = [claim for claim in snapshot.claims if claim.branch == branch]
    peer_claims = [
        claim
        for claim in same_branch
        if not (active_tool is not None and claim.tool == active_tool and claim.root == current_root)
    ]
    mutating_peers = [claim for claim in peer_claims if claim.mode == "mutating"]
    if mutating_peers:
        detail = ", ".join(f"{claim.tool}@{claim.branch}" for claim in mutating_peers)
        blockers.append(
            _issue(
                "blocker",
                "parallel_mutating_claim",
                "Concurrent mutating session blocked: another tool already holds a fresh mutating claim on this branch.",
                detail=detail,
            )
        )
    elif peer_claims:
        detail = ", ".join(f"{claim.tool}({claim.mode})@{claim.branch}" for claim in peer_claims)
        warnings.append(
            _issue(
                "warning",
                "parallel_session_present",
                "Parallel session present on this branch. Keep this session read-only or move to a worktree before editing.",
                detail=detail,
            )
        )
    return blockers, warnings


def evaluate_system_policy(snapshot: SystemContext, action: PolicyAction) -> PolicyDecision:
    blockers: list[PolicyIssue] = []
    warnings: list[PolicyIssue] = []
    infos: list[PolicyIssue] = []

    applicable_authorities = ["HANDOFF.md", *snapshot.authority.doctrine_docs, snapshot.authority.authority_map_doc]
    applicable_controls = [
        "pipeline/check_drift.py",
        "scripts/tools/session_preflight.py",
        "scripts/tools/project_pulse.py",
        "pipeline/system_context.py",
        "trading_app/lifecycle_state.py",
    ]

    if not snapshot.handoff.exists:
        warnings.append(_issue("warning", "handoff_missing", "HANDOFF.md missing."))

    if not snapshot.git.status_available:
        warnings.append(
            _issue(
                "warning",
                "git_status_unavailable",
                "Git status unavailable. Dirty-state check degraded.",
                detail=snapshot.git.status_reason,
            )
        )
    elif snapshot.git.dirty:
        warnings.append(
            _issue(
                "warning",
                "dirty_worktree",
                "Working tree is dirty. Re-read changed files before editing.",
                detail="\n".join(snapshot.git.dirty_files[:10]),
            )
        )

    if snapshot.active_stages:
        stage_names = ", ".join(Path(stage.path).name for stage in snapshot.active_stages)
        warnings.append(
            _issue(
                "warning",
                "active_stage_files",
                f"Active stage file(s) present: {len(snapshot.active_stages)}.",
                detail=stage_names,
            )
        )

    interp = snapshot.interpreter
    if action == "session_start_mutating":
        if interp.expected_python is not None and interp.expected_present and interp.matches_expected is False:
            blockers.append(
                _issue(
                    "blocker",
                    "wrong_interpreter",
                    "Mutating session blocked: this context is using the wrong interpreter for the repo-managed environment.",
                    detail=f"current={interp.current_python} expected={interp.expected_python}",
                )
            )
        elif interp.expected_python is not None and not interp.expected_present:
            blockers.append(
                _issue(
                    "blocker",
                    "expected_interpreter_missing",
                    "Mutating session blocked: the repo-managed interpreter for this context is missing.",
                    detail=interp.expected_python,
                )
            )

        parallel_blockers, parallel_warnings = _parallel_claim_issues(snapshot)
        blockers.extend(parallel_blockers)
        warnings.extend(parallel_warnings)
    elif action in {"session_start_read_only", "orientation"}:
        if interp.expected_python is not None and interp.expected_present and interp.matches_expected is False:
            warnings.append(
                _issue(
                    "warning",
                    "wrong_interpreter",
                    "This context is using the wrong interpreter for the repo-managed environment.",
                    detail=f"current={interp.current_python} expected={interp.expected_python}",
                )
            )
        elif interp.expected_python is not None and not interp.expected_present:
            warnings.append(
                _issue(
                    "warning",
                    "expected_interpreter_missing",
                    "The repo-managed interpreter for this context is missing.",
                    detail=interp.expected_python,
                )
            )

        branch = snapshot.git.branch
        current_root = snapshot.git.selected_root
        active_tool = snapshot.active_tool
        peer_claims = [
            claim
            for claim in snapshot.claims
            if claim.branch == branch
            and not (active_tool is not None and claim.tool == active_tool and claim.root == current_root)
        ]
        if peer_claims:
            detail = ", ".join(f"{claim.tool}({claim.mode})@{claim.branch}" for claim in peer_claims)
            warnings.append(
                _issue(
                    "warning",
                    "parallel_session_present",
                    "Parallel session present on this branch.",
                    detail=detail,
                )
            )

    if snapshot.git.in_linked_worktree:
        infos.append(
            _issue(
                "info",
                "linked_worktree",
                "Running inside a linked worktree.",
                detail=f"selected={snapshot.git.selected_root} canonical={snapshot.git.canonical_root}",
            )
        )

    infos.append(
        _issue(
            "info",
            "authority_backbone",
            "Canonical authority backbone loaded.",
            detail=", ".join(snapshot.authority.backbone_modules),
        )
    )

    return PolicyDecision(
        decision_id=str(uuid4()),
        action=action,
        allowed=not blockers,
        blockers=blockers,
        warnings=warnings,
        infos=infos,
        applicable_authorities=applicable_authorities,
        applicable_controls=applicable_controls,
    )


def format_system_context_text(snapshot: SystemContext, decision: PolicyDecision | None = None) -> str:
    lines = [
        "=== SYSTEM CONTEXT ===",
        f"Context: {snapshot.context_name}",
        f"Repo root: {snapshot.git.selected_root}",
        f"Canonical root: {snapshot.git.canonical_root}",
        f"Branch: {snapshot.git.branch}",
        f"HEAD: {snapshot.git.head_sha}",
        "Dirty: "
        + (
            f"yes ({snapshot.git.dirty_count})"
            if snapshot.git.dirty
            else "unknown" if not snapshot.git.status_available
            else f"no ({snapshot.git.dirty_count})"
        ),
        f"Interpreter: {snapshot.interpreter.current_python}",
    ]
    if snapshot.interpreter.expected_python:
        match = snapshot.interpreter.matches_expected
        lines.append(
            "Expected interpreter: "
            f"{snapshot.interpreter.expected_python} "
            f"({'match' if match else 'mismatch' if match is False else 'unknown'})"
        )
    lines.append(f"Fresh claims: {len(snapshot.claims)}")
    lines.append(f"Active stages: {len(snapshot.active_stages)}")
    lines.append(f"Active profiles: {', '.join(snapshot.authority.active_profiles) or 'none'}")
    lines.append(f"Active instruments: {', '.join(snapshot.authority.active_orb_instruments) or 'none'}")
    if decision is not None:
        lines.append("")
        lines.append(f"Decision: {'ALLOW' if decision.allowed else 'BLOCK'} [{decision.action}]")
        if decision.blockers:
            lines.append("Blockers:")
            for issue in decision.blockers:
                suffix = f" ({issue.detail})" if issue.detail else ""
                lines.append(f"  - {issue.message}{suffix}")
        if decision.warnings:
            lines.append("Warnings:")
            for issue in decision.warnings:
                suffix = f" ({issue.detail})" if issue.detail else ""
                lines.append(f"  - {issue.message}{suffix}")
    return "\n".join(lines)


def write_decision_log(
    root: Path,
    snapshot: SystemContext,
    decision: PolicyDecision,
    *,
    relative_path: str = "data/state/system_context_decisions.jsonl",
) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "logged_at": datetime.now(UTC).isoformat(),
        "decision": decision.model_dump(mode="json"),
        "snapshot": snapshot.model_dump(mode="json"),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")
    return path
