#!/usr/bin/env python3
"""Report-only project improvement reviewer for canompx3.

This tool is intentionally conservative. It does not import application code,
open databases, run discovery, or mutate project state unless the operator
explicitly passes ``--out`` for a derived Markdown report.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path("docs/runtime/project_reviews")
DEFAULT_SINCE_REF = "HEAD~10"
DEFAULT_MAX_FILE_BYTES = 500_000

CATEGORIES = (
    "Source-of-truth integrity",
    "Git/worktree hygiene",
    "Test/CI gaps",
    "Research integrity",
    "Code quality",
    "Literature/resource grounding",
    "Workflow speed",
)

SEVERITY_RANK = {
    "BLOCKER": 50,
    "HIGH": 40,
    "MEDIUM": 30,
    "LOW": 20,
    "INFO": 10,
}

AUTHORITY_FILES = (
    "HANDOFF.md",
    "CLAUDE.md",
    "CODEX.md",
    "RESEARCH_RULES.md",
    ".claude/rules/backtesting-methodology.md",
    "docs/governance/document_authority.md",
    "docs/governance/system_authority_map.md",
    ".codex/CODEX_IMPROVEMENT_PLAN.md",
)

EXCLUDED_DIR_PARTS = {
    ".git",
    ".venv",
    ".venv-wsl",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
}

EXCLUDED_SUFFIXES = {
    ".db",
    ".duckdb",
    ".parquet",
    ".feather",
    ".dbn",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".pyc",
    ".pyd",
    ".dll",
    ".exe",
}

LIVE_CLAIM_RE = re.compile(
    r"\b(LIVE_SAFE|go live|currently live|current truth|live safe|ready to deploy)\b",
    re.IGNORECASE,
)
EVIDENCE_RE = re.compile(
    r"\b(pytest|check_drift|ruff|source:|canonical|commit|evidence|generated from|snapshot|dated|verified)\b",
    re.IGNORECASE,
)
OOS_TUNING_RE = re.compile(
    r"\b(tune|optimi[sz]e|rescue|adjust)\b.{0,80}\b(OOS|out[- ]of[- ]sample|holdout)\b"
    r"|\b(OOS|out[- ]of[- ]sample|holdout)\b.{0,80}\b(tune|optimi[sz]e|rescue|adjust)\b",
    re.IGNORECASE | re.DOTALL,
)
OOS_DECISION_CONTEXT_RE = re.compile(
    r"\b(threshold|filter|session|entry_model|parameter|lane|profile|select|choose|promot|deploy|go[- ]live|validated|strategy)\b",
    re.IGNORECASE,
)
NO_GO_REOPEN_RE = re.compile(
    r"\b(NO[- ]GO|NOGO|graveyard|dead)\b.{0,120}\b(reopen|revive|retry)\b", re.IGNORECASE | re.DOTALL
)
PRE_REG_RE = re.compile(
    r"\b(pre[- ]?reg|pre_registered|K accounting|expected_trial_count|total_expected_trials)\b", re.IGNORECASE
)
TRADING_HYPOTHESIS_RE = re.compile(r"\b(hypothesis|strategy|filter|threshold|entry_model|ORB|OOS)\b", re.IGNORECASE)
RESEARCH_CLAIM_RE = re.compile(
    r"\b(research shows|literature says|studies show|memory says|from memory)\b", re.IGNORECASE
)
LOCAL_SOURCE_RE = re.compile(
    r"\b(docs/institutional/literature/|resources/|arxiv\.org|doi\.org|https?://)\b", re.IGNORECASE
)
BROAD_EXCEPTION_RE = re.compile(r"except\s+Exception\s*(?:as\s+\w+)?\s*:\s*(?:\n\s+.*){0,4}", re.MULTILINE)
HARDCODED_LITERAL_RE = re.compile(
    r"\b("
    r"CME_REOPEN|SINGAPORE_OPEN|TOKYO_OPEN|LONDON_OPEN|EUROPE_FLOW|NYSE_OPEN|COMEX_SETTLE|"
    r"US_DATA_830|US_DATA_1000|E2_SLIPPAGE_TICKS|MNQ|MES|MGC|topstep_50k"
    r")\b"
)
REPEATED_PROMPT_RE = re.compile(r"\b(ask Claude|ask Codex|manual prompt|copy/paste|repeat this)\b", re.IGNORECASE)
READ_ONLY_TOKEN = "read_only=True"
DB_CONNECT_RE = re.compile(r"\bduckdb\s*\.\s*connect\s*\(")
E2_LITERAL_RE = re.compile(r"entry_model\s*[=:]\s*['\"]E2['\"]|['\"]entry_model['\"]\s*:\s*['\"]E2['\"]")
E2_TAINTED_RE = re.compile(
    r"\b(rel_vol_[A-Z][A-Z0-9_]*|break_bar_volume|break_bar_continues|break_delay_min|orb_\w+_break_dir)\b"
)
E2_POLICY_RE = re.compile(
    r"#\s*e2-lookahead-policy:\s*(cleared|late-fill-only|not-predictor|tainted)\b",
    re.IGNORECASE,
)

PROTECTED_MUTATION_HINTS = (
    "lane_" + "allocation.json",
    "prop_profiles",
    "live_config",
    "Pinecone",
    "pinecone",
    "go-live",
    "go_live",
)

DB_CONNECT_REVIEW_ROOTS = (
    "research/",
    "scripts/tools/",
    "trading_app/",
)

DB_CONNECT_ALLOWLIST_PREFIXES = (
    "tests/",
    "trading_app/db_manager.py",
    "trading_app/strategy_validator.py",
    "trading_app/log_trade.py",
    "trading_app/nested/",
)

DB_CONNECT_ALLOWLIST_EXACT = {
    "pipeline/init_db.py",
    "pipeline/build_daily_features.py",
}

PROTECTED_MUTATION_ALLOWLIST = {
    "scripts/tools/live_readiness_report.py",
    "trading_app/live/planned_launch.py",
}


@dataclass(frozen=True)
class Finding:
    category: str
    severity: str
    confidence: str
    evidence_path: str
    evidence_snippet: str
    rationale: str
    suggested_patch: str
    suggested_test: str
    stop_condition: str


@dataclass(frozen=True)
class ReviewConfig:
    root: Path = PROJECT_ROOT
    scope: str = "recent"
    since_ref: str = DEFAULT_SINCE_REF
    output_path: Path | None = None
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES


@dataclass(frozen=True)
class GitState:
    branch: str
    detached: bool
    git_available: bool
    context_status: str
    staged_files: tuple[str, ...] = ()
    dirty_files: tuple[str, ...] = ()
    untracked_files: tuple[str, ...] = ()
    changed_since_ref: tuple[str, ...] = ()
    sibling_dirty: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewReport:
    generated_at: str
    config: ReviewConfig
    git_state: GitState
    scanned_files: tuple[str, ...]
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def highest_ev(self) -> Finding:
        if not self.findings:
            return Finding(
                category="Highest-EV next action",
                severity="INFO",
                confidence="HIGH",
                evidence_path="repo",
                evidence_snippet="No findings emitted by static project-improvement review.",
                rationale="No deterministic pattern exceeded the v1 thresholds.",
                suggested_patch="No patch from this report. Continue normal targeted verification.",
                suggested_test="Run the task-specific tests and `python pipeline/check_drift.py --fast`.",
                stop_condition="Stop if a fresh command surfaces a blocker.",
            )
        return sorted(
            self.findings,
            key=lambda item: (
                SEVERITY_RANK.get(item.severity, 0),
                1 if item.confidence == "HIGH" else 0,
                -len(item.suggested_patch),
            ),
            reverse=True,
        )[0]


def _run_git(root: Path, args: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return 127, "", str(exc)
    return result.returncode, (result.stdout or "").rstrip("\n"), (result.stderr or "").strip()


def _normalize_status_path(raw: str) -> str:
    path = raw.strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.replace("\\", "/")


def _parse_status_porcelain(stdout: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    staged: set[str] = set()
    dirty: set[str] = set()
    untracked: set[str] = set()
    for line in stdout.splitlines():
        if not line:
            continue
        status = line[:2]
        path = _normalize_status_path(line[3:] if len(line) > 3 else "")
        if not path:
            continue
        if status == "??":
            untracked.add(path)
            continue
        if status[0] != " ":
            staged.add(path)
        if status[1] != " ":
            dirty.add(path)
    return tuple(sorted(staged)), tuple(sorted(dirty)), tuple(sorted(untracked))


def _changed_since_ref(root: Path, since_ref: str, branch: str) -> tuple[tuple[str, ...], str | None]:
    diff_expr = f"{since_ref}..HEAD"
    if since_ref == DEFAULT_SINCE_REF and branch not in {"main", "master", "DETACHED"}:
        rc_base, base_out, _ = _run_git(root, ["merge-base", "origin/main", "HEAD"])
        if rc_base == 0 and base_out:
            diff_expr = f"{base_out}..HEAD"
    rc, out, err = _run_git(root, ["diff", "--name-only", diff_expr])
    if rc != 0:
        return (), err or out or f"git diff failed for {diff_expr}"
    return tuple(sorted(path.replace("\\", "/") for path in out.splitlines() if path.strip())), None


def _dirty_sibling_worktrees(root: Path) -> tuple[str, ...]:
    rc, out, _ = _run_git(root, ["worktree", "list", "--porcelain"])
    if rc != 0 or not out:
        return ()
    paths: list[str] = []
    for line in out.splitlines():
        if line.startswith("worktree "):
            paths.append(line.removeprefix("worktree ").strip())
    dirty: list[str] = []
    current = str(root.resolve()).lower()
    for item in paths:
        path = Path(item)
        try:
            if str(path.resolve()).lower() == current:
                continue
        except OSError:
            continue
        rc_status, status, _ = _run_git(path, ["status", "--short"])
        if rc_status == 0 and status.strip():
            dirty.append(item)
    return tuple(sorted(dirty))


def gather_git_state(root: Path, since_ref: str = DEFAULT_SINCE_REF) -> GitState:
    errors: list[str] = []
    rc_branch, branch_out, branch_err = _run_git(root, ["symbolic-ref", "--short", "HEAD"])
    git_available = rc_branch != 127
    detached = rc_branch != 0
    branch = branch_out if rc_branch == 0 else "DETACHED"
    if rc_branch != 0:
        errors.append(branch_err or branch_out or "HEAD is detached or branch cannot be resolved")

    rc_status, status_out, status_err = _run_git(root, ["status", "--porcelain=v1"])
    if rc_status == 0:
        staged, dirty, untracked = _parse_status_porcelain(status_out)
    else:
        staged, dirty, untracked = (), (), ()
        errors.append(status_err or status_out or "git status failed")

    changed, changed_err = _changed_since_ref(root, since_ref, branch)
    if changed_err:
        errors.append(changed_err)

    context = "OK"
    if not git_available or detached:
        context = "UNKNOWN/BLOCKED CONTEXT"
    elif errors:
        context = "DEGRADED CONTEXT"

    return GitState(
        branch=branch,
        detached=detached,
        git_available=git_available,
        context_status=context,
        staged_files=staged,
        dirty_files=dirty,
        untracked_files=untracked,
        changed_since_ref=changed,
        sibling_dirty=_dirty_sibling_worktrees(root) if git_available else (),
        errors=tuple(errors),
    )


def _is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_DIR_PARTS:
        return True
    return path.suffix.lower() in EXCLUDED_SUFFIXES


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _candidate_files(config: ReviewConfig, git_state: GitState) -> tuple[Path, ...]:
    candidates: set[Path] = set()
    if config.scope == "all":
        for path in config.root.rglob("*"):
            if path.is_file() and not _is_excluded(path):
                candidates.add(path)
    else:
        for name in (
            *git_state.staged_files,
            *git_state.dirty_files,
            *git_state.untracked_files,
            *git_state.changed_since_ref,
            *AUTHORITY_FILES,
        ):
            path = config.root / name
            if path.exists() and not path.is_file():
                continue
            if not _is_excluded(path):
                candidates.add(path)
    return tuple(sorted(candidates, key=lambda item: _rel(item, config.root)))


def _read_text_or_finding(path: Path, config: ReviewConfig) -> tuple[str | None, Finding | None]:
    rel = _rel(path, config.root)
    if not path.exists():
        return None, Finding(
            "Workflow speed",
            "LOW",
            "HIGH",
            rel,
            "candidate path is missing or deleted",
            "Recent/deleted paths can hide stale references or incomplete work from static review.",
            "Confirm the path was intentionally removed or restore it before relying on this report.",
            "Run targeted tests for any deleted production surface.",
            "Stop if the deleted path is still referenced by code or docs.",
        )
    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, _scan_silence(rel, f"stat failed: {type(exc).__name__}")
    if size > config.max_file_bytes:
        return None, _scan_silence(rel, f"file exceeds {config.max_file_bytes} bytes")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, _scan_silence(rel, f"read failed: {type(exc).__name__}")
    if b"\x00" in raw[:4096]:
        return None, _scan_silence(rel, "binary/null-byte content")
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), _scan_silence(rel, "utf-8 decode replacement used")


def _scan_silence(rel: str, reason: str) -> Finding:
    return Finding(
        "Workflow speed",
        "LOW",
        "HIGH",
        rel,
        reason,
        "The reviewer skipped this file, so absence of findings is not proof the file is clean.",
        "Inspect the skipped file manually if it is in the active workstream.",
        "Add a narrow fixture if this file type should be scanned automatically.",
        "Stop if the skipped file is a production or research authority surface.",
    )


def _snippet(text: str, match: re.Match[str], limit: int = 180) -> str:
    start = max(0, match.start() - 60)
    end = min(len(text), match.end() + 60)
    return " ".join(text[start:end].split())[:limit]


def _nearby_has_evidence(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 300) : min(len(text), end + 300)]
    return bool(EVIDENCE_RE.search(window))


def _nearby_is_prohibition(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 120) : min(len(text), end + 120)].lower()
    return any(
        token in window
        for token in (
            "never ",
            "must not",
            "do not",
            "does not",
            "not ",
            "no ",
            "without ",
            "banned",
            "forbid",
            "blocked",
            "unchanged threshold",
            "thresholds are read-only",
        )
    )


def _oos_tuning_match(text: str) -> re.Match[str] | None:
    """Return the first high-signal OOS tuning risk.

    Audit docs often say "no OOS tuning" or "did not rescue"; those are safety
    assertions, not evidence of contamination. Treat "rescue" as high-risk only
    when the nearby text also references a concrete decision surface.
    """
    for match in OOS_TUNING_RE.finditer(text):
        if _nearby_is_prohibition(text, match.start(), match.end()):
            continue
        window = text[max(0, match.start() - 180) : min(len(text), match.end() + 180)]
        matched_text = match.group(0).lower()
        if "rescue" in matched_text and not OOS_DECISION_CONTEXT_RE.search(window):
            continue
        return match
    return None


def _is_reviewer_own_surface(path: str) -> bool:
    return path.startswith("docs/runtime/project_reviews/") or path in {
        "scripts/tools/project_improvement_review.py",
        "tests/test_tools/test_project_improvement_review.py",
    }


def _should_review_db_connect(path: str) -> bool:
    if path in DB_CONNECT_ALLOWLIST_EXACT:
        return False
    if any(path.startswith(prefix) for prefix in DB_CONNECT_ALLOWLIST_PREFIXES):
        return False
    return any(path.startswith(prefix) for prefix in DB_CONNECT_REVIEW_ROOTS)


def _finding(
    category: str,
    severity: str,
    confidence: str,
    path: str,
    snippet: str,
    rationale: str,
    patch: str,
    test: str,
    stop: str,
) -> Finding:
    return Finding(category, severity, confidence, path, snippet, rationale, patch, test, stop)


def _scan_source_truth(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    if _is_reviewer_own_surface(path):
        return findings
    if not path.endswith((".md", ".txt", ".yaml", ".yml")):
        return findings
    for match in LIVE_CLAIM_RE.finditer(text):
        if _nearby_has_evidence(text, match.start(), match.end()):
            continue
        findings.append(
            _finding(
                "Source-of-truth integrity",
                "HIGH",
                "MEDIUM",
                path,
                _snippet(text, match),
                "Static pattern risk: live/current claims need canonical source or dated command evidence.",
                "Rewrite the claim as a dated snapshot or link the canonical code/data source that owns the fact.",
                "Run `uv run python scripts/tools/project_improvement_review.py` and the relevant targeted test.",
                "Stop if the claim affects live trading, capital, or research promotion state.",
            )
        )
        break
    return findings


def _scan_research_integrity(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    if _is_reviewer_own_surface(path):
        return findings
    is_research_surface = path.startswith(
        ("research/", "docs/audit/", "docs/institutional/", "chatgpt_bundle/")
    ) or path.endswith(("RESEARCH_RULES.md", "STRATEGY_BLUEPRINT.md"))
    if match := _oos_tuning_match(text):
        findings.append(
            _finding(
                "Research integrity",
                "BLOCKER",
                "MEDIUM",
                path,
                _snippet(text, match),
                "Static pattern risk: OOS/holdout language appears near tuning or rescue language.",
                "Refactor into a pre-registered hypothesis draft with explicit K accounting before any result inspection.",
                "Run `uv run python pipeline/check_drift.py --fast` and targeted research tests.",
                "Stop if OOS influenced a threshold, filter, session, or lane-selection decision.",
            )
        )
    if NO_GO_REOPEN_RE.search(text):
        findings.append(
            _finding(
                "Research integrity",
                "HIGH",
                "MEDIUM",
                path,
                _snippet(text, NO_GO_REOPEN_RE.search(text) or re.search(".", text)),
                "Static pattern risk: dead/NO-GO work appears to be reopened without an explicit critique gate.",
                "Add the critique of the original NO-GO verdict before authoring new research or code.",
                "Run the relevant NO-GO/dead-registry check and project review again.",
                "Stop if the reopened idea lacks a falsifiable reason the old verdict no longer applies.",
            )
        )
    if (
        is_research_surface
        and TRADING_HYPOTHESIS_RE.search(text)
        and "hypothesis" in text.lower()
        and not PRE_REG_RE.search(text)
    ):
        findings.append(
            _finding(
                "Research integrity",
                "MEDIUM",
                "LOW",
                path,
                "trading hypothesis language without nearby pre-reg/K accounting token",
                "Static pattern risk: hypothesis discussion should route to pre-registration and K accounting.",
                "Convert any proposed trading improvement into a pre-reg draft before testing.",
                "Add a fixture or hypothesis-file test if this is an executable research path.",
                "Stop if the hypothesis is already being tested or promoted.",
            )
        )
    return findings


def _scan_code_quality(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    if _is_reviewer_own_surface(path):
        return findings
    if path.endswith(".py"):
        db_match = DB_CONNECT_RE.search(text)
        if db_match and READ_ONLY_TOKEN not in text and _should_review_db_connect(path):
            findings.append(
                _finding(
                    "Code quality",
                    "HIGH",
                    "MEDIUM",
                    path,
                    _snippet(text, db_match),
                    "Static pattern risk: database connection appears without read-only mode. `check_drift.py` remains canonical.",
                    "Either route through an owned writer path or make the reader connection explicitly read-only.",
                    "Run `uv run python pipeline/check_drift.py --fast` plus the companion test for the touched surface.",
                    "Stop if this path touches canonical `gold.db` outside an approved writer.",
                )
            )
        broad_match = BROAD_EXCEPTION_RE.search(text)
        if broad_match:
            block = broad_match.group(0)
            if not re.search(r"\b(raise|return\s+1|sys\.exit|fatal|blocked|fail)", block, re.IGNORECASE):
                findings.append(
                    _finding(
                        "Code quality",
                        "MEDIUM",
                        "MEDIUM",
                        path,
                        " ".join(block.split())[:180],
                        "Static pattern risk: broad exception handling may swallow failures without a fail-closed result.",
                        "Narrow the exception or return an explicit blocked/error state with evidence.",
                        "Add a failure-mode unit test for the exception path.",
                        "Stop if the broad handler wraps trading, DB, git, or verification state.",
                    )
                )
        if HARDCODED_LITERAL_RE.search(text) and path.startswith(("scripts/", "trading_app/", "pipeline/")):
            findings.append(
                _finding(
                    "Code quality",
                    "LOW",
                    "LOW",
                    path,
                    _snippet(text, HARDCODED_LITERAL_RE.search(text) or re.search(".", text)),
                    "Static pattern risk: canonical session/instrument/profile literals can drift when duplicated.",
                    "Verify this literal is fixture-only or delegated to the canonical registry.",
                    "Run the companion tests for the touched module.",
                    "Stop if the literal controls production routing or live strategy eligibility.",
                )
            )
    return findings


def _scan_literature_grounding(path: str, text: str) -> list[Finding]:
    if _is_reviewer_own_surface(path):
        return []
    if not path.endswith((".md", ".txt", ".yaml", ".yml", ".py")):
        return []
    match = RESEARCH_CLAIM_RE.search(text)
    if match and _nearby_is_prohibition(text, match.start(), match.end()):
        return []
    if match and not LOCAL_SOURCE_RE.search(text[max(0, match.start() - 400) : min(len(text), match.end() + 800)]):
        return [
            _finding(
                "Literature/resource grounding",
                "MEDIUM",
                "MEDIUM",
                path,
                _snippet(text, match),
                "Static pattern risk: a research/resource claim appears without a local source or external citation nearby.",
                "Cite the local literature extract/resource path or label the claim as unsupported.",
                "Run the relevant doc/reference test or manually verify cited paths.",
                "Stop if the claim justifies research promotion, capital, or live-readiness decisions.",
            )
        ]
    return []


def _scan_workflow_speed(path: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    if _is_reviewer_own_surface(path):
        return findings
    match = REPEATED_PROMPT_RE.search(text)
    if match:
        findings.append(
            _finding(
                "Workflow speed",
                "LOW",
                "LOW",
                path,
                _snippet(text, match),
                "Static pattern risk: repeated manual operator prompting may be better captured as a script or skill.",
                "If this workflow has repeated twice, extract the deterministic steps into a small tool or skill.",
                "Add a smoke test for any new deterministic tool.",
                "Stop if the manual step controls verification or git hygiene.",
            )
        )
    return findings


def _scan_e2_lookahead(path: str, text: str) -> list[Finding]:
    if _is_reviewer_own_surface(path):
        return []
    if not path.endswith(".py"):
        return []
    if E2_LITERAL_RE.search(text) and E2_TAINTED_RE.search(text) and not E2_POLICY_RE.search(text):
        return [
            _finding(
                "Research integrity",
                "HIGH",
                "MEDIUM",
                path,
                "entry_model='E2' appears with break-bar-derived predictor token and no e2-lookahead-policy annotation",
                "Static pattern risk: E2 stop-market entries can occur before break-bar-derived features are known.",
                "Route through canonical filter utilities or add a verified e2-lookahead-policy annotation.",
                "Run `uv run python pipeline/check_drift.py --fast` because drift owns this canonical class.",
                "Stop if the feature is used to select, tune, or promote E2 trades.",
            )
        ]
    return []


def _scan_protected_mutation(path: str, text: str) -> list[Finding]:
    if _is_reviewer_own_surface(path):
        return []
    if path in PROTECTED_MUTATION_ALLOWLIST:
        return []
    if path.startswith(("docs/", "chatgpt_bundle/", "tests/")):
        return []
    if not path.endswith((".py", ".yaml", ".yml", ".json")):
        return []
    hits: list[str] = []
    for token in PROTECTED_MUTATION_HINTS:
        for match in re.finditer(re.escape(token), text, re.IGNORECASE):
            window = text[max(0, match.start() - 180) : min(len(text), match.end() + 180)]
            if _nearby_is_prohibition(text, match.start(), match.end()):
                continue
            if re.search(
                r"\b(write|dump|apply|mutate|deploy|sync|upsert|delete|go_live|go-live)\b"
                r"|\b(write|dump|apply|sync|upsert|delete)_",
                window,
                re.IGNORECASE,
            ):
                hits.append(token)
                break
    if not hits:
        return []
    return [
        _finding(
            "Git/worktree hygiene",
            "HIGH",
            "LOW",
            path,
            f"protected surface token(s): {', '.join(sorted(set(hits)))}",
            "Static pattern risk: protected live/allocation/profile/Pinecone surfaces appear near mutation language.",
            "Confirm this is report-only. Move any mutation into an approved operator flow.",
            "Run targeted tests plus `uv run python pipeline/check_drift.py --fast`.",
            "Stop if the path can change live config, allocator state, Pinecone, or profile routing.",
        )
    ]


def _scan_test_gap(scanned: tuple[str, ...], git_state: GitState) -> list[Finding]:
    changed = set(git_state.staged_files) | set(git_state.dirty_files) | set(git_state.changed_since_ref)
    changed_code = sorted(path for path in changed if path.endswith(".py") and not path.startswith("tests/"))
    changed_tests = {path for path in changed if path.startswith("tests/") and path.endswith(".py")}
    if not changed_code or changed_tests:
        return []
    sample = ", ".join(changed_code[:5])
    return [
        _finding(
            "Test/CI gaps",
            "MEDIUM",
            "MEDIUM",
            "git",
            f"changed code without changed tests in scan window: {sample}",
            "Static pattern risk: recent code changes do not show an adjacent targeted-test change in the same window.",
            "Add or identify the targeted test covering the changed code before claiming completion.",
            "Run the targeted pytest slice before broad drift/lint.",
            "Stop if the changed code is a shared canonical or live/runtime surface.",
        )
    ]


def _git_findings(git_state: GitState) -> list[Finding]:
    findings: list[Finding] = []
    if git_state.context_status != "OK":
        findings.append(
            _finding(
                "Git/worktree hygiene",
                "HIGH" if git_state.detached else "MEDIUM",
                "HIGH",
                "git",
                git_state.context_status,
                "Git context is degraded; review conclusions may miss branch/workstream intent.",
                "Attach the work to an intentional branch before commit/push work, or keep this as report-only.",
                "Run `git status --short --branch` and `git log --oneline -10`.",
                "Stop if you need to commit or publish from a detached or unknown context.",
            )
        )
    if "HANDOFF.md" in git_state.staged_files:
        findings.append(
            _finding(
                "Git/worktree hygiene",
                "HIGH",
                "HIGH",
                "HANDOFF.md",
                "HANDOFF.md is staged",
                "Cross-tool baton staging can accidentally mix session state into unrelated code commits.",
                "Unstage or commit HANDOFF.md only with an intentional baton/update commit.",
                "Run `git diff --cached -- HANDOFF.md` before commit.",
                "Stop if staged HANDOFF.md is unrelated to the code patch.",
            )
        )
    protected_staged = sorted(
        path
        for path in (*git_state.staged_files, *git_state.dirty_files)
        if any(
            token.lower() in path.lower() for token in ("lane_allocation", "live_config", "prop_profiles", "pinecone")
        )
    )
    if protected_staged:
        findings.append(
            _finding(
                "Git/worktree hygiene",
                "BLOCKER",
                "HIGH",
                "git",
                ", ".join(protected_staged[:5]),
                "Protected live/allocation/profile surfaces are modified in the worktree.",
                "Separate these changes from report-only review work unless explicitly approved.",
                "Run targeted tests and drift after isolating the protected surface.",
                "Stop if this patch is supposed to be report-only.",
            )
        )
    if git_state.sibling_dirty:
        findings.append(
            _finding(
                "Git/worktree hygiene",
                "MEDIUM",
                "HIGH",
                "git worktree",
                ", ".join(git_state.sibling_dirty[:3]),
                "Dirty sibling worktrees can hide parallel edits or stale context.",
                "Inspect sibling worktree status before broad refactors or commits.",
                "Run `git worktree list --porcelain` and targeted sibling `git status` checks.",
                "Stop if sibling work touches the same files.",
            )
        )
    review_artifacts = [
        path for path in git_state.untracked_files if "review" in path.lower() or path.startswith("docs/runtime/")
    ]
    if review_artifacts:
        findings.append(
            _finding(
                "Git/worktree hygiene",
                "LOW",
                "MEDIUM",
                "git",
                ", ".join(review_artifacts[:5]),
                "Untracked review/runtime artifacts can be accidentally missed or accidentally committed.",
                "Decide whether each artifact is disposable, ignored, or intentionally derived output.",
                "Run `git status --short` before staging.",
                "Stop if an artifact is used as evidence but remains untracked.",
            )
        )
    return findings


def review(config: ReviewConfig) -> ReviewReport:
    git_state = gather_git_state(config.root, config.since_ref)
    candidates = _candidate_files(config, git_state)
    findings: list[Finding] = []
    findings.extend(_git_findings(git_state))
    scanned: list[str] = []

    for path in candidates:
        rel = _rel(path, config.root)
        text, silence = _read_text_or_finding(path, config)
        if silence:
            findings.append(silence)
        if text is None:
            continue
        scanned.append(rel)
        findings.extend(_scan_source_truth(rel, text))
        findings.extend(_scan_research_integrity(rel, text))
        findings.extend(_scan_code_quality(rel, text))
        findings.extend(_scan_literature_grounding(rel, text))
        findings.extend(_scan_workflow_speed(rel, text))
        findings.extend(_scan_e2_lookahead(rel, text))
        findings.extend(_scan_protected_mutation(rel, text))

    findings.extend(_scan_test_gap(tuple(scanned), git_state))
    if "scripts/tools/context_resolver.py" not in scanned:
        findings.append(
            _finding(
                "Workflow speed",
                "LOW",
                "HIGH",
                "scripts/tools/context_resolver.py",
                "context resolver route not part of v1 scan",
                "The current task had no deterministic route during planning; v1 should remain tool-only until report quality is proven.",
                "After several useful reports, add a context_resolver route for this review task.",
                "Add resolver tests only in the follow-up route patch.",
                "Stop if the route would duplicate or override canonical Claude/Codex rules.",
            )
        )
    return ReviewReport(
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        config=config,
        git_state=git_state,
        scanned_files=tuple(sorted(scanned)),
        findings=tuple(findings),
    )


def render_markdown(report: ReviewReport) -> str:
    lines: list[str] = []
    lines.append("# Project Improvement Review")
    lines.append("")
    lines.append(f"- Generated: `{report.generated_at}`")
    lines.append(f"- Scope: `{report.config.scope}`")
    lines.append(f"- Since ref: `{report.config.since_ref}`")
    lines.append(f"- Git context: `{report.git_state.context_status}`")
    lines.append(f"- Branch: `{report.git_state.branch}`")
    lines.append(f"- Scanned files: `{len(report.scanned_files)}`")
    lines.append("")
    if report.git_state.errors:
        lines.append("## State Notes")
        for err in report.git_state.errors[:5]:
            lines.append(f"- {err}")
        lines.append("")
    for category in CATEGORIES:
        lines.append(f"## {category}")
        category_findings = [item for item in report.findings if item.category == category]
        if not category_findings:
            lines.append("- No deterministic v1 finding.")
            lines.append("")
            continue
        for item in sorted(category_findings, key=lambda found: SEVERITY_RANK.get(found.severity, 0), reverse=True):
            lines.append(f"- **{item.severity}** `{item.evidence_path}` ({item.confidence})")
            lines.append(f"  - Evidence: {item.evidence_snippet}")
            lines.append(f"  - Why: {item.rationale}")
            lines.append(f"  - Patch: {item.suggested_patch}")
            lines.append(f"  - Test: {item.suggested_test}")
            lines.append(f"  - Stop: {item.stop_condition}")
        lines.append("")
    highest = report.highest_ev
    lines.append("## Highest-EV next action")
    lines.append(f"- **{highest.severity}** `{highest.evidence_path}` ({highest.confidence})")
    lines.append(f"- Rationale: {highest.rationale}")
    lines.append(f"- Patch: {highest.suggested_patch}")
    lines.append(f"- Tests: {highest.suggested_test}")
    lines.append(f"- Stop condition: {highest.stop_condition}")
    lines.append("")
    lines.append("## Verification commands")
    lines.append("```powershell")
    lines.append("uv run python scripts/tools/project_improvement_review.py")
    lines.append("uv run python pipeline/check_drift.py --fast")
    lines.append("uv run python -m ruff check . --quiet")
    lines.append("```")
    lines.append("")
    lines.append(
        "> Report-only output. Static pattern risks are not semantic proof; verify with targeted commands before claiming fixes."
    )
    return "\n".join(lines) + "\n"


def _resolve_output_path(root: Path, output: str | None) -> Path | None:
    if not output:
        return None
    report_root = (root / REPORT_DIR).resolve()
    target = (root / output).resolve()
    try:
        target.relative_to(report_root)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--out must resolve under {REPORT_DIR.as_posix()}") from exc
    if target.suffix.lower() != ".md":
        raise argparse.ArgumentTypeError("--out must be a Markdown .md path")
    return target


def parse_args(argv: list[str] | None = None) -> ReviewConfig:
    parser = argparse.ArgumentParser(description="Report-only canompx3 project improvement review")
    parser.add_argument("--scope", choices=("recent", "all"), default="recent")
    parser.add_argument("--since-ref", default=DEFAULT_SINCE_REF)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    args = parser.parse_args(argv)
    try:
        out = _resolve_output_path(PROJECT_ROOT, args.out)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return ReviewConfig(
        root=PROJECT_ROOT,
        scope=args.scope,
        since_ref=args.since_ref,
        output_path=out,
        max_file_bytes=args.max_file_bytes,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    report = review(config)
    markdown = render_markdown(report)
    if config.output_path:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(markdown, encoding="utf-8")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
