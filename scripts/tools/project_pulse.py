#!/usr/bin/env python3
"""Project pulse — synthesize project state from linked repo/runtime signals.

Answers four questions:
  1. Is anything broken?     (drift, tests)
  2. What was I doing?       (handoff, next steps, worktrees)
  3. Is my data/edge fresh?  (staleness, fitness)
  4. What's on deck?         (action queue, ralph deferred)

Usage:
    python scripts/tools/project_pulse.py                  # full (~35s first, <3s cached)
    python scripts/tools/project_pulse.py --fast            # skip drift+tests (~3s)
    python scripts/tools/project_pulse.py --format json     # for /orient skill
    python scripts/tools/project_pulse.py --format markdown --out PROJECT_PULSE.md
    python scripts/tools/project_pulse.py --deep            # full fitness (slow)
    python scripts/tools/project_pulse.py --no-cache        # force re-run drift+tests
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _preferred_repo_python() -> Path | None:
    if os.name == "nt":
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_ROOT / ".venv-wsl" / "bin" / "python"
    return candidate if candidate.exists() else None


def _preferred_repo_prefix(expected_python: Path) -> Path:
    return expected_python.parent.parent.resolve()


def _ensure_repo_python() -> None:
    """Re-exec into the repo venv when run as a direct script.

    Only activates for direct ``python project_pulse.py`` invocations
    (``__name__ == "__main__"``).  When imported as a library (pytest,
    ``build_brief()``, hooks), the re-exec is skipped — the caller's
    Python already has the repo on sys.path via the path-insert below.
    """
    if __name__ != "__main__":
        return
    expected_python = _preferred_repo_python()
    if expected_python is None:
        return
    current_prefix = Path(sys.prefix).resolve()
    expected_prefix = _preferred_repo_prefix(expected_python)
    if current_prefix == expected_prefix or os.environ.get("CANOMPX3_BOOTSTRAP_DONE") == "1":
        return

    env = os.environ.copy()
    env["CANOMPX3_BOOTSTRAP_DONE"] = "1"
    env.setdefault("CANOMPX3_BOOTSTRAPPED_FROM", str(Path(sys.executable).resolve()))
    raise SystemExit(
        subprocess.call(
            [str(expected_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    )


_ensure_repo_python()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# staleness_engine lives in scripts/tools/ (same dir as this file)
_SCRIPTS_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_TOOLS_DIR)

from pipeline.paths import GOLD_DB_PATH, LIVE_JOURNAL_DB_PATH
from pipeline.work_queue import (
    handoff_active_steps_match,
    load_queue,
    top_baton_items,
)
from pipeline.work_queue import (
    stale_items as queue_stale_items,
)
from trading_app.validated_shelf import deployable_validated_relation

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

CATEGORIES = ("broken", "decaying", "ready", "unactioned", "paused")


SKILL_SUGGESTIONS: dict[str, str] = {
    "staleness": "/rebuild-outcomes {inst}",
    "fitness": "/regime-check",
    "drift": "/verify",
    "tests": "/verify quick",
    "handoff": "/orient --full",
    "ralph": "/audit quick",
    "control_state": "python scripts/tools/refresh_control_state.py --profile topstep_50k_mnq_auto",
    "sr_monitor": "python -m trading_app.sr_monitor",
    "criterion11": "python -m trading_app.account_survival --profile topstep_50k_mnq_auto",
}


@dataclass
class PulseItem:
    category: str  # broken / decaying / ready / unactioned / paused
    severity: str  # high / medium / low
    source: str  # which collector found it
    summary: str  # one-line human description
    detail: str | None = None  # optional extra context
    action: str | None = None  # suggested skill/command to resolve


@dataclass
class NextAction:
    kind: str  # queue / command / reconcile
    label: str
    command: str | None = None
    queue_id: str | None = None
    source: str | None = None


@dataclass
class DebtEntry:
    debt_id: str | None
    text: str
    line_no: int
    has_owner: bool
    has_status: bool
    has_parking_decision: bool


@dataclass
class PlanEntry:
    path: Path
    title: str
    owner: str
    status: str
    last_reviewed: str


@dataclass
class PulseReport:
    generated_at: str
    cache_hit: bool
    git_head: str
    git_branch: str
    items: list[PulseItem] = field(default_factory=list)
    system_identity: dict | None = None
    # Handoff context (not a categorized item)
    handoff_tool: str | None = None
    handoff_date: str | None = None
    handoff_summary: str | None = None
    handoff_next_steps: list[str] = field(default_factory=list)
    # Fitness summary (fast proxy or deep)
    fitness_summary: dict | None = None
    deployment_summary: dict | None = None
    survival_summary: dict | None = None
    sr_summary: dict | None = None
    pause_summary: dict | None = None
    execution_summary: dict | None = None
    capital_packet_summary: dict | None = None
    capital_recommendation: str | None = None
    live_readiness_summary: dict | None = None
    # Trading day context
    upcoming_sessions: list[dict] = field(default_factory=list)
    # Single recommendation
    recommendation: str | None = None
    # Health metrics
    time_since_green: str | None = None
    session_delta: list[str] = field(default_factory=list)
    work_capsule_summary: dict | None = None
    system_brief_summary: dict | None = None
    history_debt_summary: dict | None = None
    startup_latency_ms: int | None = None
    orientation_cost_budget: dict | None = None
    next_actions: list[NextAction] = field(default_factory=list)

    @property
    def broken(self) -> list[PulseItem]:
        return [i for i in self.items if i.category == "broken"]

    @property
    def decaying(self) -> list[PulseItem]:
        return [i for i in self.items if i.category == "decaying"]

    @property
    def ready(self) -> list[PulseItem]:
        return [i for i in self.items if i.category == "ready"]

    @property
    def unactioned(self) -> list[PulseItem]:
        return [i for i in self.items if i.category == "unactioned"]

    @property
    def paused(self) -> list[PulseItem]:
        return [i for i in self.items if i.category == "paused"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GIT_TIMEOUT = 5
SUBPROCESS_TIMEOUT = 60


def _scrubbed_git_env() -> dict[str, str]:
    """os.environ minus GIT_DIR/GIT_WORK_TREE/GIT_INDEX_FILE.

    These are pre-populated by pre-commit hooks and other git contexts; if
    inherited, they override cwd-based repo resolution and make subprocess
    git calls report the wrong repo. See PR #126 postmortem.
    """
    env = os.environ.copy()
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        env.pop(var, None)
    return env


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT,
            check=False,
            env=_scrubbed_git_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git_head(root: Path) -> str:
    r = _run_git(root, "rev-parse", "--short", "HEAD")
    return r.stdout.strip() if r and r.returncode == 0 else "unknown"


def _git_branch(root: Path) -> str:
    r = _run_git(root, "branch", "--show-current")
    return r.stdout.strip() if r and r.returncode == 0 else "unknown"


def _canonical_repo_root(root: Path) -> Path:
    """Resolve the real repo root, even from inside a worktree."""
    r = _run_git(root, "rev-parse", "--git-common-dir")
    if r and r.returncode == 0:
        common = Path(r.stdout.strip()).resolve()
        if common.name == ".git":
            return common.parent
    return root.resolve()


def _find_memory_md(canonical: Path) -> Path | None:
    """Derive Claude auto-memory MEMORY.md path from canonical repo root.

    Claude stores project memory at ~/.claude/projects/<sanitized>/memory/MEMORY.md
    where <sanitized> replaces : and path separators with dashes.
    E.g. C:\\Users\\joshd\\canompx3 -> C--Users-joshd-canompx3
    """
    home = Path.home()
    projects_dir = home / ".claude" / "projects"
    if not projects_dir.exists():
        return None

    # Build sanitized project key matching Claude's format
    resolved_str = str(canonical.resolve())
    # Replace drive colon+separator: "C:\" -> "C--"
    sanitized = resolved_str.replace(":\\", "--").replace(":/", "--")
    # Replace remaining separators with single dash
    sanitized = sanitized.replace("\\", "-").replace("/", "-")

    # Try exact match first
    candidate = projects_dir / sanitized / "memory" / "MEMORY.md"
    if candidate.exists():
        return candidate

    # Try case-insensitive match
    target = sanitized.lower()
    for d in projects_dir.iterdir():
        if d.is_dir() and d.name.lower() == target:
            mem = d / "memory" / "MEMORY.md"
            if mem.exists():
                return mem

    # Fallback: match on repo name, prefer longest (most specific) match
    repo_name = canonical.name.lower()
    best: Path | None = None
    best_len = 0
    for d in projects_dir.iterdir():
        if d.is_dir() and repo_name in d.name.lower():
            mem = d / "memory" / "MEMORY.md"
            if mem.exists() and len(d.name) > best_len:
                best = mem
                best_len = len(d.name)
    return best


def _is_db_locked(e: Exception) -> bool:
    msg = str(e)
    return "being used by another process" in msg or "Cannot open file" in msg


# ---------------------------------------------------------------------------
# Collectors — each returns list[PulseItem], never raises
# ---------------------------------------------------------------------------


def collect_git_state(root: Path) -> list[PulseItem]:
    """Git dirty files and stashes."""
    items: list[PulseItem] = []

    # Dirty files (exclude the pulse cache file itself)
    r = _run_git(root, "status", "--short")
    if r and r.returncode == 0:
        dirty = [line for line in r.stdout.splitlines() if line.strip() and ".pulse_cache.json" not in line]
        if dirty:
            items.append(
                PulseItem(
                    category="paused",
                    severity="low",
                    source="git",
                    summary=f"{len(dirty)} uncommitted file(s)",
                    detail="\n".join(dirty[:10]) + ("\n..." if len(dirty) > 10 else ""),
                )
            )

    # Stashes
    r = _run_git(root, "stash", "list")
    if r and r.returncode == 0:
        stashes = [line for line in r.stdout.splitlines() if line.strip()]
        if stashes:
            items.append(
                PulseItem(
                    category="paused",
                    severity="low",
                    source="git",
                    summary=f"{len(stashes)} git stash(es) — possible forgotten work",
                    detail="\n".join(stashes[:5]),
                )
            )

    return items


_UPDATE_HEADER_RE = re.compile(r"^## (?:Current Session )?Update \((\d{4}-\d{2}-\d{2})(?:[^)]*?\s+[—-]\s+(.*?))?\)$")


def _strip_handoff_markup(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = text.replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def _normalize_handoff_step(text: str) -> str:
    stripped = _strip_handoff_markup(text)
    bullet_match = re.match(r"^(?:[-*]|\d+\.)\s+(.+)$", stripped)
    return bullet_match.group(1) if bullet_match else stripped


def _extract_first_paragraph(lines: list[str], start_index: int) -> str | None:
    paragraph: list[str] = []
    started = False
    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped:
            if started:
                break
            continue
        if stripped.startswith("#"):
            if started:
                break
            continue
        paragraph.append(_strip_handoff_markup(stripped))
        started = True
    return " ".join(paragraph) if paragraph else None


def _parse_rolling_handoff(lines: list[str]) -> dict:
    """Parse the current rolling-update HANDOFF format."""
    context: dict = {}
    update_candidates: list[tuple[int, re.Match[str]]] = []
    for idx, line in enumerate(lines):
        match = _UPDATE_HEADER_RE.match(line.strip())
        if match:
            update_candidates.append((idx, match))

    if not update_candidates:
        return context

    selected_lines: list[str] = []
    selected_match: re.Match[str] | None = None
    fallback_lines: list[str] = []
    fallback_match: re.Match[str] | None = None

    for update_index, update_match in update_candidates:
        update_lines: list[str] = []
        for line in lines[update_index + 1 :]:
            if line.startswith("## "):
                break
            update_lines.append(line.rstrip())

        if fallback_match is None:
            fallback_match = update_match
            fallback_lines = update_lines

        substantive = [
            ln.strip()
            for ln in update_lines
            if ln.strip() and not ln.strip().startswith("### ") and not ln.strip().startswith("#### ")
        ]
        if substantive:
            selected_match = update_match
            selected_lines = update_lines
            break

    if selected_match is None:
        selected_match = fallback_match
        selected_lines = fallback_lines

    if selected_match is None:
        return context

    date_str = selected_match.group(1)
    title = _strip_handoff_markup(selected_match.group(2) or "")
    headline: str | None = None
    next_steps: list[str] = []

    for idx, line in enumerate(selected_lines):
        stripped = line.strip()
        lower = stripped.lower()
        if lower == "### headline":
            headline = _extract_first_paragraph(selected_lines, idx + 1)
            break

    in_next_steps = False
    pending_step_heading: str | None = None
    for line in selected_lines:
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("### next move") or lower.startswith("### next steps"):
            in_next_steps = True
            pending_step_heading = None
            continue

        if in_next_steps and stripped.startswith("### "):
            break

        if not in_next_steps:
            continue

        if not stripped:
            if pending_step_heading:
                next_steps.append(pending_step_heading)
                pending_step_heading = None
            continue

        if "highest-value next step" in lower:
            continue

        if stripped.startswith("#### "):
            heading = _strip_handoff_markup(stripped.removeprefix("#### ").strip())
            heading = re.sub(r"^\d+\.\s*", "", heading)
            pending_step_heading = heading
            continue

        bullet_match = re.match(r"^(?:[-*]|\d+\.)\s+(.+)$", stripped)
        if bullet_match:
            bullet = _strip_handoff_markup(bullet_match.group(1))
            if pending_step_heading:
                next_steps.append(f"{pending_step_heading} — {bullet}")
                pending_step_heading = None
            else:
                next_steps.append(bullet)
            continue

        if pending_step_heading:
            next_steps.append(f"{pending_step_heading} — {_strip_handoff_markup(stripped)}")
            pending_step_heading = None

    if pending_step_heading:
        next_steps.append(pending_step_heading)

    context["tool"] = "Update log"
    context["date"] = date_str
    if headline:
        context["summary"] = headline
    elif title:
        context["summary"] = title
    if next_steps:
        context["next_steps"] = next_steps

    return context


def _parse_legacy_handoff(lines: list[str]) -> dict:
    """Parse the older metadata-style HANDOFF format."""
    context: dict = {}
    next_steps: list[str] = []
    section: str | None = None

    for line in lines:
        if line.startswith("## Last Session"):
            section = "metadata"
            continue
        if re.match(r"^## Next Steps", line):
            section = "next_steps"
            continue
        if line.startswith("## "):
            section = None
            continue

        if section == "metadata":
            if line.startswith("- **Tool:** "):
                context["tool"] = line.removeprefix("- **Tool:** ").strip()
            elif line.startswith("- **Date:** "):
                context["date"] = line.removeprefix("- **Date:** ").strip()
            elif line.startswith("- **Summary:** "):
                context["summary"] = line.removeprefix("- **Summary:** ").strip()
        elif section == "next_steps":
            stripped = line.strip()
            if stripped and not stripped.startswith("Phases "):
                next_steps.append(stripped)

    if next_steps:
        context["next_steps"] = next_steps
    return context


def collect_handoff(root: Path) -> tuple[dict, list[PulseItem]]:
    """Parse HANDOFF.md for context + next steps. Single pass."""
    context: dict = {}
    items: list[PulseItem] = []
    handoff_path = root / "HANDOFF.md"
    queue_path = root / "docs" / "runtime" / "action-queue.yaml"
    queue_steps: list[str] = []
    queue = None
    if queue_path.exists():
        try:
            queue = load_queue(root)
            queue_steps = [f"{item.title} — {item.next_action}" for item in top_baton_items(queue)]
        except Exception:
            queue_steps = []
    if not handoff_path.exists():
        if queue_steps:
            context = {
                "tool": "Queue",
                "summary": "Canonical action queue available; generated baton missing.",
                "next_steps": queue_steps,
            }
            items.append(
                PulseItem(
                    category="decaying",
                    severity="medium",
                    source="handoff",
                    summary="HANDOFF.md missing; falling back to canonical action queue",
                )
            )
            return context, items
        items.append(
            PulseItem(
                category="broken",
                severity="medium",
                source="handoff",
                summary="HANDOFF.md missing",
            )
        )
        return context, items

    text = handoff_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    blocker_keywords = {"failure", "broken", "missing", "error", "cannot", "blocked"}
    context = _parse_rolling_handoff(lines) or _parse_legacy_handoff(lines)
    if queue_steps and queue is not None:
        parsed_steps = context.get("next_steps", [])
        active_steps_match = handoff_active_steps_match(root, queue)
        if parsed_steps and active_steps_match is False:
            items.append(
                PulseItem(
                    category="decaying",
                    severity="medium",
                    source="handoff",
                    summary="HANDOFF next steps drifted from canonical action queue",
                )
            )
        context["next_steps"] = queue_steps

    section: str | None = None
    for line in lines:
        if line.startswith("## Blockers") or line.startswith("## Blockers / Warnings"):
            section = "blockers"
            continue
        if line.startswith("## "):
            section = None
            continue
        if section != "blockers":
            continue

        stripped = line.strip()
        if stripped.startswith("- "):
            note = stripped.lstrip("- ")
            if any(kw in note.lower() for kw in blocker_keywords):
                items.append(
                    PulseItem(
                        category="broken",
                        severity="medium",
                        source="handoff",
                        summary=note,
                    )
                )

    return context, items


def collect_drift(root: Path) -> list[PulseItem]:
    """Run check_drift.py and report pass/fail."""
    drift_script = root / "pipeline" / "check_drift.py"
    if not drift_script.exists():
        return []

    try:
        r = subprocess.run(
            [sys.executable, str(drift_script)],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return [
            PulseItem(
                category="broken",
                severity="high",
                source="drift",
                summary=f"Drift check timed out (>{SUBPROCESS_TIMEOUT}s)",
            )
        ]
    except OSError as e:
        return [
            PulseItem(
                category="broken",
                severity="high",
                source="drift",
                summary=f"Drift check failed to run: {e}",
            )
        ]

    if r.returncode == 0:
        return []

    # Drift detected — count only actual FAILED lines, not the summary
    failures = [line.strip() for line in r.stdout.splitlines() if "FAILED:" in line]

    return [
        PulseItem(
            category="broken",
            severity="high",
            source="drift",
            summary=f"Drift check FAILED ({len(failures)} violation(s))",
            detail="\n".join(failures[:10]) if failures else None,
        )
    ]


def collect_tests(root: Path) -> list[PulseItem]:
    """Run pytest with minimal output and report pass/fail."""
    tests_dir = root / "tests"
    if not tests_dir.exists():
        return []

    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", str(tests_dir), "-x", "-q", "--tb=line", "--no-header"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        # Timeout means the suite is larger than the 120s pulse budget, not
        # that any test is broken. Categorize as `paused` (the check is
        # deferred) rather than `broken` (which would fire the FIX NOW
        # siren). Users should run pytest directly to verify pass/fail.
        return [
            PulseItem(
                category="paused",
                severity="low",
                source="tests",
                summary=(
                    "Test health check skipped: suite exceeds 120s pulse "
                    "budget. Run `python -m pytest` directly to verify."
                ),
            )
        ]
    except OSError:
        return []

    if r.returncode == 0:
        return []

    # Extract failure summary from last few lines
    lines = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    summary_line = lines[-1] if lines else "Tests failed"
    failure_lines = [line for line in lines if "FAILED" in line]

    return [
        PulseItem(
            category="broken",
            severity="high",
            source="tests",
            summary=summary_line,
            detail="\n".join(failure_lines[:5]) if failure_lines else None,
        )
    ]


def collect_staleness(root: Path, db_path: Path) -> list[PulseItem]:
    """Check pipeline staleness per instrument."""
    items: list[PulseItem] = []
    if not db_path.exists():
        return [
            PulseItem(
                category="broken",
                severity="high",
                source="staleness",
                summary=f"gold.db not found at {db_path}",
            )
        ]

    try:
        import duckdb
        from pipeline_status import staleness_engine

        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, DEPLOYABLE_ORB_INSTRUMENTS

        deployable_set = set(DEPLOYABLE_ORB_INSTRUMENTS)
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            for inst in ACTIVE_ORB_INSTRUMENTS:
                status = staleness_engine(con, inst)
                stale = list(status.get("stale_steps", []))
                # For research-only instruments, the validated_setups shelf is
                # expected empty (insufficient real-micro data horizon for T7
                # survival). Filter that entry out instead of alerting. Any
                # OTHER stale step for the same instrument still surfaces.
                if inst not in deployable_set:
                    stale = [s for s in stale if s != "validated_setups"]
                if stale:
                    items.append(
                        PulseItem(
                            category="decaying",
                            severity="medium" if len(stale) <= 2 else "high",
                            source="staleness",
                            summary=f"{inst}: {len(stale)} stale step(s) — {', '.join(stale[:3])}",
                        )
                    )
        finally:
            con.close()
    except Exception as e:
        if _is_db_locked(e):
            items.append(
                PulseItem(
                    category="paused",
                    severity="low",
                    source="staleness",
                    summary="DB locked — staleness check skipped",
                )
            )
        else:
            items.append(
                PulseItem(
                    category="broken",
                    severity="medium",
                    source="staleness",
                    summary=f"Staleness check error: {type(e).__name__}: {e}",
                )
            )

    return items


def collect_fitness_fast(db_path: Path) -> tuple[dict, list[PulseItem]]:
    """Fast proxy: count active validated strategies per instrument.

    Alerts only on DEPLOYABLE_ORB_INSTRUMENTS — research-only instruments
    (e.g. MGC) still appear in the summary dict when they have rows, but
    an empty deployable shelf for a research-only instrument is by-design
    expected state, not an alert condition.
    """
    summary: dict = {}
    items: list[PulseItem] = []
    if not db_path.exists():
        return summary, items

    try:
        import duckdb

        from pipeline.asset_configs import DEPLOYABLE_ORB_INSTRUMENTS

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            shelf_relation = deployable_validated_relation(con)
            rows = con.execute(
                f"SELECT instrument, COUNT(*) as n FROM {shelf_relation} GROUP BY instrument ORDER BY instrument"
            ).fetchall()
            for inst, n in rows:
                summary[inst] = {"active_strategies": n}
            for inst in DEPLOYABLE_ORB_INSTRUMENTS:
                if inst not in summary:
                    items.append(
                        PulseItem(
                            category="decaying",
                            severity="high",
                            source="fitness",
                            summary=f"{inst}: 0 active validated strategies",
                        )
                    )
        finally:
            con.close()
    except Exception as e:
        if _is_db_locked(e):
            items.append(
                PulseItem(
                    category="paused",
                    severity="low",
                    source="fitness",
                    summary="DB locked — fitness check skipped",
                )
            )
        else:
            items.append(
                PulseItem(
                    category="broken",
                    severity="low",
                    source="fitness",
                    summary=f"Fitness check error: {type(e).__name__}",
                )
            )

    return summary, items


EXECUTION_STALE_AFTER_DAYS = 7
REPORT_ONLY_CAPITAL_BOUNDARY = "REPORT_ONLY_NOT_DEPLOYMENT_AUTHORITY"
AUTHORIZED_PROMOTE_CAPITAL_BOUNDARIES: frozenset[str] = frozenset()


def _parse_day(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _collect_execution_evidence(
    con,
    *,
    profile_id: str,
    deployed_ids: list[str],
    live_journal_path: Path | None = None,
    today: date | None = None,
) -> dict:
    """Read deployed-lane execution attribution from paper_trades and live journal."""
    effective_today = today or date.today()

    def _set_last_day_age(row: dict) -> None:
        parsed_last_day = _parse_day(row["last_day"])
        row["last_day_age_days"] = (effective_today - parsed_last_day).days if parsed_last_day else None

    per_lane = {
        strategy_id: {
            "strategy_id": strategy_id,
            "paper_trade_count": 0,
            "live_trade_count": 0,
            "execution_count": 0,
            "first_day": None,
            "last_day": None,
            "last_day_age_days": None,
            "paper_avg_r": None,
            "paper_sum_r": None,
            "live_avg_r": None,
            "live_sum_r": None,
        }
        for strategy_id in deployed_ids
    }
    live_journal_status = "not_checked"

    if deployed_ids:
        placeholders = ", ".join("?" for _ in deployed_ids)
        rows = con.execute(
            f"""
            SELECT
                strategy_id,
                COUNT(*) AS n,
                MIN(trading_day) AS first_day,
                MAX(trading_day) AS last_day,
                AVG(pnl_r) AS avg_r,
                SUM(pnl_r) AS sum_r
            FROM paper_trades
            WHERE strategy_id IN ({placeholders})
            GROUP BY strategy_id
            """,
            deployed_ids,
        ).fetchall()
        for strategy_id, n, first_day, last_day, avg_r, sum_r in rows:
            sid = str(strategy_id)
            row = per_lane[sid]
            row["paper_trade_count"] = int(n or 0)
            row["execution_count"] = int(row["execution_count"]) + int(n or 0)
            row["first_day"] = str(first_day) if first_day is not None else None
            row["last_day"] = str(last_day) if last_day is not None else None
            _set_last_day_age(row)
            row["paper_avg_r"] = round(float(avg_r), 6) if avg_r is not None else None
            row["paper_sum_r"] = round(float(sum_r), 6) if sum_r is not None else None

        if live_journal_path is not None:
            live_journal_status = "missing_file"
            if live_journal_path.exists():
                try:
                    import duckdb

                    live_con = duckdb.connect(str(live_journal_path), read_only=True)
                    try:
                        has_live_trades = bool(
                            live_con.execute(
                                """
                                SELECT COUNT(*)
                                FROM information_schema.tables
                                WHERE table_schema = 'main' AND table_name = 'live_trades'
                                """
                            ).fetchone()[0]
                        )
                        if has_live_trades:
                            live_rows = live_con.execute(
                                f"""
                                SELECT
                                    strategy_id,
                                    COUNT(*) AS n,
                                    MIN(trading_day) AS first_day,
                                    MAX(trading_day) AS last_day,
                                    AVG(actual_r) AS avg_r,
                                    SUM(actual_r) AS sum_r
                                FROM live_trades
                                WHERE strategy_id IN ({placeholders})
                                GROUP BY strategy_id
                                """,
                                deployed_ids,
                            ).fetchall()
                            live_journal_status = "ok"
                            for strategy_id, n, first_day, last_day, avg_r, sum_r in live_rows:
                                sid = str(strategy_id)
                                row = per_lane[sid]
                                first_day_text = str(first_day) if first_day is not None else None
                                last_day_text = str(last_day) if last_day is not None else None
                                row["live_trade_count"] = int(n or 0)
                                row["execution_count"] = int(row["execution_count"]) + int(n or 0)
                                row["first_day"] = min(
                                    [x for x in [row["first_day"], first_day_text] if x is not None],
                                    default=None,
                                )
                                row["last_day"] = max(
                                    [x for x in [row["last_day"], last_day_text] if x is not None],
                                    default=None,
                                )
                                _set_last_day_age(row)
                                row["live_avg_r"] = round(float(avg_r), 6) if avg_r is not None else None
                                row["live_sum_r"] = round(float(sum_r), 6) if sum_r is not None else None
                        else:
                            live_journal_status = "missing_table"
                    finally:
                        live_con.close()
                except Exception as exc:  # noqa: BLE001 - report journal evidence degradation, keep paper check
                    live_journal_status = f"unreadable:{type(exc).__name__}: {exc}"

    missing_ids = [sid for sid, row in per_lane.items() if int(row["execution_count"]) == 0]
    stale_ids = [
        sid
        for sid, row in per_lane.items()
        if int(row["execution_count"]) > 0
        and row["last_day_age_days"] is not None
        and int(row["last_day_age_days"]) > EXECUTION_STALE_AFTER_DAYS
    ]
    return {
        "profile_id": profile_id,
        "source_tables": ["paper_trades", "live_journal.live_trades"],
        "live_journal_path": str(live_journal_path) if live_journal_path is not None else None,
        "live_journal_status": live_journal_status,
        "deployed_count": len(deployed_ids),
        "covered_count": len(deployed_ids) - len(missing_ids),
        "missing_count": len(missing_ids),
        "missing_execution_strategy_ids": missing_ids,
        "stale_after_days": EXECUTION_STALE_AFTER_DAYS,
        "stale_execution_strategy_ids": stale_ids,
        "per_lane": [per_lane[sid] for sid in deployed_ids],
    }


def collect_deployment_state(
    db_path: Path,
    *,
    live_journal_path: Path | None = LIVE_JOURNAL_DB_PATH,
) -> tuple[dict | None, list[PulseItem]]:
    """Summarize deployed-live vs validated-active truth."""
    summary: dict | None = None
    items: list[PulseItem] = []
    if not db_path.exists():
        return summary, items

    try:
        import duckdb

        from trading_app.prop_profiles import get_profile_lane_definitions, resolve_profile_id

        profile_id = resolve_profile_id()
        deployed_lanes = get_profile_lane_definitions(profile_id)
        deployed_ids = [str(lane["strategy_id"]) for lane in deployed_lanes]
        execution_evidence = None

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            shelf_relation = deployable_validated_relation(con)
            validated_rows = con.execute(
                f"""
                SELECT strategy_id
                FROM {shelf_relation}
                ORDER BY strategy_id
                """
            ).fetchall()
            validated_ids = [str(r[0]) for r in validated_rows]
            try:
                execution_evidence = _collect_execution_evidence(
                    con,
                    profile_id=profile_id,
                    deployed_ids=deployed_ids,
                    live_journal_path=live_journal_path,
                )
            except Exception as exc:  # noqa: BLE001 - missing attribution source is a capital blocker
                items.append(
                    PulseItem(
                        category="broken",
                        severity="high",
                        source="execution",
                        summary=f"{profile_id}: execution evidence unavailable from paper_trades",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
        finally:
            con.close()

        deployed_set = set(deployed_ids)
        validated_set = set(validated_ids)
        deployed_not_validated = sorted(deployed_set - validated_set)
        validated_not_deployed = sorted(validated_set - deployed_set)

        summary = {
            "profile_id": profile_id,
            "deployed_count": len(deployed_ids),
            "validated_active_count": len(validated_ids),
            "deployed_not_validated": deployed_not_validated,
            "validated_not_deployed": validated_not_deployed,
            "deployed_strategy_ids": deployed_ids,
            "execution_evidence": execution_evidence,
        }

        if not deployed_ids:
            items.append(
                PulseItem(
                    category="broken",
                    severity="high",
                    source="deployment",
                    summary=f"{profile_id}: 0 deployed daily lanes",
                )
            )
        if deployed_not_validated:
            items.append(
                PulseItem(
                    category="broken",
                    severity="high",
                    source="deployment",
                    summary=(
                        f"{profile_id}: {len(deployed_not_validated)} deployed lane(s) not in active validated_setups"
                    ),
                    detail="\n".join(deployed_not_validated[:5]),
                )
            )
        if validated_not_deployed:
            items.append(
                PulseItem(
                    category="ready",
                    severity="low",
                    source="deployment",
                    summary=(f"{profile_id}: {len(validated_not_deployed)} active validated lane(s) not deployed-live"),
                    detail="\n".join(validated_not_deployed[:5]),
                    action="/trade-book",
                )
            )
        if execution_evidence:
            missing_ids = execution_evidence.get("missing_execution_strategy_ids", [])
            stale_ids = execution_evidence.get("stale_execution_strategy_ids", [])
            live_status = str(execution_evidence.get("live_journal_status") or "")
            if live_status.startswith("unreadable:"):
                items.append(
                    PulseItem(
                        category="broken",
                        severity="high",
                        source="execution",
                        summary=f"{profile_id}: live journal execution evidence unavailable",
                        detail=live_status,
                        action="scripts/tools/stop_live.ps1",
                    )
                )
            if missing_ids:
                items.append(
                    PulseItem(
                        category="broken",
                        severity="high",
                        source="execution",
                        summary=f"{profile_id}: {len(missing_ids)} deployed lane(s) have zero execution rows",
                        detail="\n".join(str(x) for x in missing_ids[:10]),
                        action=f"python -m trading_app.paper_trade_logger --profile {profile_id} --sync --dry-run",
                    )
                )
            if stale_ids:
                items.append(
                    PulseItem(
                        category="decaying",
                        severity="medium",
                        source="execution",
                        summary=(f"{profile_id}: {len(stale_ids)} deployed lane(s) have stale execution attribution"),
                        detail="\n".join(str(x) for x in stale_ids[:10]),
                        action=f"python -m trading_app.paper_trade_logger --profile {profile_id} --sync --dry-run",
                    )
                )
    except Exception as e:
        if _is_db_locked(e):
            items.append(
                PulseItem(
                    category="paused",
                    severity="low",
                    source="deployment",
                    summary="DB locked — deployment state check skipped",
                )
            )
        else:
            items.append(
                PulseItem(
                    category="broken",
                    severity="medium",
                    source="deployment",
                    summary=f"Deployment state error: {type(e).__name__}",
                )
            )

    return summary, items


def _collect_control_items_from_lifecycle(
    lifecycle: dict,
) -> tuple[dict | None, dict | None, dict | None, list[PulseItem]]:
    """Project unified lifecycle truth into pulse summaries and actionable items."""
    items: list[PulseItem] = []
    survival_summary = lifecycle.get("criterion11")
    sr_summary = lifecycle.get("criterion12")
    pause_summary = lifecycle.get("pauses")
    strategy_states = lifecycle.get("strategy_states", {})

    if isinstance(survival_summary, dict):
        if not survival_summary.get("gate_ok"):
            items.append(
                PulseItem(
                    category="broken",
                    severity="high",
                    source="criterion11",
                    summary=str(survival_summary.get("gate_msg")),
                    action=(
                        f"python scripts/tools/refresh_control_state.py --profile {survival_summary.get('profile_id')}"
                    ),
                )
            )
        elif survival_summary.get("report_age_days") is not None and int(survival_summary["report_age_days"]) >= 14:
            items.append(
                PulseItem(
                    category="decaying",
                    severity="low",
                    source="criterion11",
                    summary=(
                        f"Criterion 11 report is {survival_summary['report_age_days']}d old — "
                        "refresh before it hard-blocks"
                    ),
                )
            )

    if isinstance(sr_summary, dict):
        reviewed_watch_ids = sorted(
            strategy_id
            for strategy_id, state in strategy_states.items()
            if isinstance(state, dict)
            and state.get("sr_status") == "ALARM"
            and state.get("sr_review_outcome") == "watch"
        )
        unresolved_alarm_ids = sorted(
            strategy_id
            for strategy_id, state in strategy_states.items()
            if isinstance(state, dict)
            and state.get("sr_status") == "ALARM"
            and state.get("sr_review_outcome") is None
            and not state.get("paused")
        )
        sr_summary = dict(sr_summary)
        sr_summary["reviewed_watch_strategy_ids"] = reviewed_watch_ids
        sr_summary["reviewed_watch_count"] = len(reviewed_watch_ids)
        sr_summary["unresolved_alarm_strategy_ids"] = unresolved_alarm_ids
        sr_summary["unresolved_alarm_count"] = len(unresolved_alarm_ids)

        if not sr_summary.get("available"):
            items.append(
                PulseItem(
                    category="broken",
                    severity="high",
                    source="sr_monitor",
                    summary="Criterion 12 SR state missing — refresh control state",
                    action=(f"python scripts/tools/refresh_control_state.py --profile {sr_summary.get('profile_id')}"),
                )
            )
            sr_summary = None
        elif not sr_summary.get("valid"):
            reason = sr_summary.get("reason")
            is_stale = isinstance(reason, str) and reason.startswith("stale")
            items.append(
                PulseItem(
                    category="broken",
                    severity="high",
                    source="sr_monitor",
                    summary=(
                        "Criterion 12 SR state is stale — refresh control state"
                        if is_stale
                        else "Criterion 12 SR state mismatched/legacy — refresh control state"
                    ),
                    detail=str(reason) if reason is not None else None,
                    action=(f"python scripts/tools/refresh_control_state.py --profile {sr_summary.get('profile_id')}"),
                )
            )
            sr_summary = None
        else:
            alarms = int(sr_summary.get("counts", {}).get("ALARM", 0))
            unresolved_alarms = int(sr_summary.get("unresolved_alarm_count", 0))
            if unresolved_alarms > 0:
                items.append(
                    PulseItem(
                        category="decaying",
                        severity="high",
                        source="sr_monitor",
                        summary=f"Criterion 12 SR has {unresolved_alarms} unresolved ALARM lane(s)",
                        action="python -m trading_app.sr_monitor --apply-pauses",
                    )
                )
            elif (
                alarms == 0 and sr_summary.get("state_age_days") is not None and int(sr_summary["state_age_days"]) >= 2
            ):
                items.append(
                    PulseItem(
                        category="decaying",
                        severity="low",
                        source="sr_monitor",
                        summary=f"Criterion 12 SR state is {sr_summary['state_age_days']}d old",
                    )
                )

    if isinstance(pause_summary, dict) and pause_summary.get("paused_strategy_ids"):
        paused_ids = [str(x) for x in pause_summary.get("paused_strategy_ids", [])]
        items.append(
            PulseItem(
                category="paused",
                severity="low",
                source="pauses",
                summary=f"{pause_summary['profile_id']}: {pause_summary['paused_count']} lane(s) currently paused",
                detail="\n".join(paused_ids[:5]),
            )
        )

    return survival_summary, sr_summary, pause_summary, items


def collect_lifecycle_control(db_path: Path) -> tuple[dict | None, dict | None, dict | None, list[PulseItem]]:
    """Read unified lifecycle truth once, then project pulse control summaries from it."""
    try:
        from trading_app.lifecycle_state import read_lifecycle_state

        lifecycle = read_lifecycle_state(db_path=db_path)
    except Exception as e:
        return (
            None,
            None,
            None,
            [
                PulseItem(
                    category="broken",
                    severity="medium",
                    source="lifecycle",
                    summary=f"Lifecycle state error: {type(e).__name__}: {e}",
                )
            ],
        )

    return _collect_control_items_from_lifecycle(lifecycle)


def collect_survival_state() -> tuple[dict | None, list[PulseItem]]:
    """Compatibility wrapper for tests/consumers expecting the C11-only collector."""
    summary, _sr, _pauses, items = collect_lifecycle_control(GOLD_DB_PATH)
    return summary, [i for i in items if i.source == "criterion11" or i.source == "lifecycle"]


def collect_sr_state(db_path: Path) -> tuple[dict | None, list[PulseItem]]:
    """Compatibility wrapper for tests/consumers expecting the C12-only collector."""
    try:
        from trading_app.lifecycle_state import read_criterion12_state

        read_criterion12_state(db_path=db_path)
    except Exception as e:
        return (
            None,
            [
                PulseItem(
                    category="broken",
                    severity="medium",
                    source="sr_monitor",
                    summary=f"Criterion 12 state error: {type(e).__name__}: {e}",
                )
            ],
        )

    _survival, lifecycle_summary, _pauses, items = collect_lifecycle_control(db_path)
    if lifecycle_summary is None:
        return None, [i for i in items if i.source == "sr_monitor" or i.source == "lifecycle"]
    return lifecycle_summary, [i for i in items if i.source == "sr_monitor" or i.source == "lifecycle"]


def collect_pause_state() -> tuple[dict | None, list[PulseItem]]:
    """Compatibility wrapper for tests/consumers expecting the pause-only collector."""
    _survival, _sr, summary, items = collect_lifecycle_control(GOLD_DB_PATH)
    return summary, [i for i in items if i.source == "pauses" or i.source == "lifecycle"]


def collect_capital_packet(root: Path) -> tuple[dict | None, list[PulseItem]]:
    """Read the existing report-only Fast Lane capital packet summary."""
    items: list[PulseItem] = []
    path = root / "docs" / "runtime" / "fast_lane_capital_packet.json"
    if not path.exists():
        return None, items
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        summary = {
            "path": str(path.relative_to(root)),
            "available": True,
            "valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        items.append(
            PulseItem(
                category="broken",
                severity="high",
                source="capital_packet",
                summary="Fast Lane capital packet unreadable",
                detail=summary["error"],
            )
        )
        return summary, items
    diff = payload.get("rebalance_dry_run_diff", {})
    valid = True
    error = None
    if not isinstance(diff, dict):
        valid = False
        error = "rebalance_dry_run_diff is not an object"
        diff = {}
    summary = {
        "path": str(path.relative_to(root)),
        "available": True,
        "valid": valid,
        "error": error,
        "capital_boundary": payload.get("capital_boundary"),
        "generated_at": payload.get("generated_at"),
        "would_add": [str(x) for x in diff.get("would_add", [])],
        "would_remove": [str(x) for x in diff.get("would_remove", [])],
        "would_keep": [str(x) for x in diff.get("would_keep", [])],
        "blocked_change_reason": diff.get("blocked_change_reason"),
    }
    if not valid:
        items.append(
            PulseItem(
                category="broken",
                severity="high",
                source="capital_packet",
                summary="Fast Lane capital packet schema invalid",
                detail=error,
            )
        )
    return summary, items


def collect_live_readiness(profile_id: str = "topstep_50k_mnq_auto") -> tuple[dict | None, list[PulseItem]]:
    """Read the live-readiness cockpit and project it into pulse actions."""
    try:
        from scripts.tools.live_readiness_report import build_live_readiness_report

        report = build_live_readiness_report(profile_id=profile_id, db_path=GOLD_DB_PATH)
    except Exception as e:
        return (
            None,
            [
                PulseItem(
                    category="broken",
                    severity="medium",
                    source="live_readiness",
                    summary=f"Live readiness check error: {type(e).__name__}",
                    detail=str(e),
                    action=f"python scripts/tools/live_readiness_report.py --profile {profile_id} --format text",
                )
            ],
        )

    summary = {
        "profile_id": report.get("profile_id"),
        "runtime_root": report.get("runtime_root"),
        "strict_zero_warn": report.get("strict_zero_warn"),
        "automation_health": report.get("automation_health"),
        "telemetry_maturity": report.get("telemetry_maturity"),
        "profile_launch": report.get("profile_launch"),
    }
    items: list[PulseItem] = []
    strict = report.get("strict_zero_warn") or {}
    blockers = [str(blocker) for blocker in strict.get("blockers", [])]
    if blockers:
        items.append(
            PulseItem(
                category="broken",
                severity="high",
                source="live_readiness",
                summary=f"{summary['profile_id']}: {len(blockers)} live-readiness blocker(s)",
                detail="\n".join(blockers[:8]),
                action=f"python scripts/tools/live_readiness_report.py --profile {summary['profile_id']} --strict-zero-warn",
            )
        )

    automation = report.get("automation_health") or {}
    if automation.get("available") is True and automation.get("overall") not in (None, "OK"):
        task_lines = [
            f"{task.get('task_name')}={task.get('status')}"
            for task in automation.get("tasks", [])
            if task.get("status") != "OK"
        ]
        items.append(
            PulseItem(
                category="decaying",
                severity="medium",
                source="automation_health",
                summary=f"Automation health {automation.get('overall')}",
                detail="\n".join(task_lines),
                action=f"python scripts/tools/live_readiness_report.py --profile {summary['profile_id']} --format text",
            )
        )

    return summary, items


def collect_system_identity(root: Path, canonical: Path, db_path: Path) -> tuple[dict | None, list[PulseItem]]:
    """Expose the repo's core identity from linked canonical registries."""
    items: list[PulseItem] = []
    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from pipeline.system_authority import SYSTEM_AUTHORITY_BACKBONE_MODULES
        from pipeline.system_context import build_system_context, evaluate_system_policy, infer_context_name

        context_name = infer_context_name(root, Path(sys.executable))

        snapshot = build_system_context(
            root,
            context_name=context_name,
            active_mode="read-only",
            db_path=db_path,
        )
        decision = evaluate_system_policy(snapshot, "orientation")
        active_orb_instruments = snapshot.authority.active_orb_instruments or sorted(ACTIVE_ORB_INSTRUMENTS)
        backbone_modules = snapshot.authority.backbone_modules or list(SYSTEM_AUTHORITY_BACKBONE_MODULES)
        summary = {
            "canonical_repo_root": snapshot.git.canonical_root,
            "selected_repo_root": snapshot.git.selected_root,
            "canonical_db_path": snapshot.db.canonical_db_path,
            "selected_db_path": snapshot.db.selected_db_path,
            "db_override_active": snapshot.db.db_override_active,
            "live_journal_db_path": snapshot.db.live_journal_db_path,
            "active_orb_instruments": active_orb_instruments,
            "authority_map_doc": snapshot.authority.authority_map_doc,
            "doctrine_docs": snapshot.authority.doctrine_docs,
            "backbone_modules": backbone_modules,
            "published_relations": snapshot.authority.published_relations,
            "interpreter": {
                "context": snapshot.interpreter.context,
                "current_python": snapshot.interpreter.current_python,
                "current_prefix": snapshot.interpreter.current_prefix,
                "expected_python": snapshot.interpreter.expected_python,
                "expected_prefix": snapshot.interpreter.expected_prefix,
                "matches_expected": snapshot.interpreter.matches_expected,
            },
            "git": {
                "branch": snapshot.git.branch,
                "head_sha": snapshot.git.head_sha,
                "dirty_count": snapshot.git.dirty_count,
                "in_linked_worktree": snapshot.git.in_linked_worktree,
            },
            "active_stages": [
                {
                    "path": stage.path,
                    "task": stage.task,
                    "mode": stage.mode,
                    "agent": stage.agent,
                }
                for stage in snapshot.active_stages
            ],
            "fresh_claims": [
                {
                    "tool": claim.tool,
                    "branch": claim.branch,
                    "mode": claim.mode,
                    "root": claim.root,
                }
                for claim in snapshot.claims
            ],
            "work_queue": {
                "exists": snapshot.work_queue.exists,
                "open_count": snapshot.work_queue.open_count,
                "close_first_open_count": snapshot.work_queue.close_first_open_count,
                "stale_count": snapshot.work_queue.stale_count,
                "top_items": [item.model_dump(mode="json") for item in snapshot.work_queue.top_items],
                "close_first_items": [item.model_dump(mode="json") for item in snapshot.work_queue.close_first_items],
                "handoff_matches_rendered": snapshot.work_queue.handoff_matches_rendered,
            },
            "policy": {
                "allowed": decision.allowed,
                "warnings": [issue.message for issue in decision.warnings],
                "controls": decision.applicable_controls,
            },
        }
        for issue in decision.warnings:
            if issue.code in {
                "wrong_interpreter",
                "handoff_queue_mismatch",
                "precommit_hook_inactive",
                "stale_queue_items",
                "close_first_carryover",
                "queue_item_conflict",
            }:
                items.append(
                    PulseItem(
                        category="decaying",
                        severity="medium",
                        source="system_identity",
                        summary=issue.message,
                        detail=issue.detail,
                    )
                )
        return summary, items
    except Exception as exc:
        items.append(
            PulseItem(
                category="broken",
                severity="low",
                source="system_identity",
                summary=f"System identity error: {type(exc).__name__}: {exc}",
            )
        )
        return None, items


def collect_work_capsule(root: Path) -> tuple[dict | None, list[PulseItem]]:
    items: list[PulseItem] = []
    try:
        from pipeline.work_capsule import evaluate_current_capsule, read_worktree_metadata
    except Exception as exc:
        items.append(
            PulseItem(
                category="broken",
                severity="low",
                source="work_capsule",
                summary=f"Work capsule read error: {type(exc).__name__}: {exc}",
            )
        )
        return None, items

    managed = read_worktree_metadata(root) is not None
    summary, issues = evaluate_current_capsule(root)
    if summary is None and not managed and not issues:
        return None, items

    for issue in issues:
        if issue.level == "info":
            continue
        detail = issue.detail
        if issue.level == "blocker":
            items.append(
                PulseItem(
                    category="broken",
                    severity="high",
                    source="work_capsule",
                    summary=issue.message,
                    detail=detail,
                )
            )
        else:
            items.append(
                PulseItem(
                    category="decaying"
                    if issue.code not in {"capsule_missing_scope", "capsule_missing_verification"}
                    else "unactioned",
                    severity="medium"
                    if issue.code not in {"capsule_missing_scope", "capsule_missing_verification"}
                    else "low",
                    source="work_capsule",
                    summary=issue.message,
                    detail=detail,
                    action="python scripts/tools/work_capsule.py",
                )
            )
    return summary, items


def collect_system_brief(root: Path, db_path: Path, tool_name: str) -> tuple[dict | None, list[PulseItem]]:
    items: list[PulseItem] = []
    try:
        from pipeline.system_brief import build_system_brief

        payload = build_system_brief(
            root,
            briefing_level="read_only",
            context_name="codex-wsl" if tool_name == "codex" else "generic",
            active_tool=tool_name,
            active_mode="read-only",
            db_path=db_path,
        )
    except Exception as exc:
        items.append(
            PulseItem(
                category="broken",
                severity="low",
                source="system_brief",
                summary=f"System brief read error: {type(exc).__name__}: {exc}",
            )
        )
        return None, items

    summary = {
        "task_id": payload["task_id"],
        "briefing_level": payload["briefing_level"],
        "verification_profile": payload["verification_profile"],
        "work_capsule_ref": payload["work_capsule_ref"],
        "blocker_count": len(payload["blocking_issues"]),
        "warning_count": len(payload["warning_issues"]),
        "required_live_view_count": len(payload["required_live_views"]),
        "canonical_owner_count": len(payload["canonical_owners"]),
    }
    if payload["blocking_issues"]:
        items.append(
            PulseItem(
                category="broken",
                severity="medium",
                source="system_brief",
                summary=f"System brief has {len(payload['blocking_issues'])} blocker(s)",
                detail=", ".join(issue["message"] for issue in payload["blocking_issues"][:3]),
            )
        )
    elif payload["warning_issues"]:
        items.append(
            PulseItem(
                category="decaying",
                severity="low",
                source="system_brief",
                summary=f"System brief has {len(payload['warning_issues'])} warning(s)",
                detail=", ".join(issue["message"] for issue in payload["warning_issues"][:3]),
            )
        )
    return {"summary": summary, "payload": payload}, items


def collect_fitness_deep(db_path: Path) -> tuple[dict, list[PulseItem]]:
    """Full fitness: compute rolling FIT/WATCH/DECAY per instrument."""
    summary: dict = {}
    items: list[PulseItem] = []
    if not db_path.exists():
        return summary, items

    try:
        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from trading_app.strategy_fitness import compute_portfolio_fitness

        for inst in ACTIVE_ORB_INSTRUMENTS:
            report = compute_portfolio_fitness(db_path=db_path, instrument=inst)
            summary[inst] = report.summary
            decay_count = report.summary.get("decay", 0)
            watch_count = report.summary.get("watch", 0)
            if decay_count > 0:
                items.append(
                    PulseItem(
                        category="decaying",
                        severity="high" if decay_count >= 5 else "medium",
                        source="fitness",
                        summary=f"{inst}: {decay_count} DECAY, {watch_count} WATCH strategies",
                    )
                )
            elif watch_count > 0:
                items.append(
                    PulseItem(
                        category="decaying",
                        severity="low",
                        source="fitness",
                        summary=f"{inst}: {watch_count} WATCH strategies",
                    )
                )
    except Exception as e:
        if _is_db_locked(e):
            items.append(
                PulseItem(
                    category="paused",
                    severity="low",
                    source="fitness",
                    summary="DB locked — deep fitness check skipped",
                )
            )
        else:
            items.append(
                PulseItem(
                    category="broken",
                    severity="low",
                    source="fitness",
                    summary=f"Deep fitness error: {type(e).__name__}",
                )
            )

    return summary, items


def _worktree_metadata(canonical: Path, *, authoritative: bool) -> list[dict]:
    """Read managed worktree metadata without recursive repo crawls."""
    worktree_root = canonical / ".worktrees"
    if not worktree_root.exists():
        return []

    try:
        from scripts.tools.worktree_manager import WORKTREE_META, list_worktrees, read_metadata
    except Exception:
        return []

    canonical_resolved = canonical.resolve()
    worktrees_by_path: dict[Path, dict] = {}

    if authoritative:
        try:
            for info in list_worktrees(canonical):
                path = Path(info.path).resolve()
                if path == canonical_resolved:
                    continue
                meta_path = path / WORKTREE_META
                if not meta_path.exists():
                    continue
                data = read_metadata(path)
                if data:
                    worktrees_by_path[path] = data
        except Exception:
            pass

    shallow_patterns = (
        f"*/{WORKTREE_META}",
        f"*/*/{WORKTREE_META}",
        f"*/*/*/{WORKTREE_META}",
    )
    for pattern in shallow_patterns:
        for meta_path in worktree_root.glob(pattern):
            path = meta_path.parent.resolve()
            if path == canonical_resolved or path in worktrees_by_path:
                continue
            data = read_metadata(path)
            if data:
                worktrees_by_path[path] = data

    return list(worktrees_by_path.values())


def collect_worktrees(canonical: Path, *, fast: bool = False) -> list[PulseItem]:
    """Detect open managed worktrees. Summarizes when >3 to reduce noise."""
    items: list[PulseItem] = []
    worktrees = _worktree_metadata(canonical, authoritative=not fast)

    if not worktrees:
        return items

    if len(worktrees) > 3:
        tools: dict[str, int] = {}
        for wt in worktrees:
            t = wt.get("tool", "unknown")
            tools[t] = tools.get(t, 0) + 1
        breakdown = ", ".join(f"{n} {t}" for t, n in sorted(tools.items()))
        items.append(
            PulseItem(
                category="paused",
                severity="low",
                source="worktrees",
                summary=f"{len(worktrees)} open worktrees ({breakdown})",
            )
        )
        return items

    for data in worktrees:
        name = data.get("name", "?")
        tool = data.get("tool", "unknown")
        if fast:
            items.append(
                PulseItem(
                    category="paused",
                    severity="low",
                    source="worktrees",
                    summary=f"{name} ({tool}) — metadata present",
                )
            )
            continue
        momentum = _workstream_momentum(data, canonical)
        stalled = "STALLED" in momentum
        items.append(
            PulseItem(
                category="paused",
                severity="medium" if stalled else "low",
                source="worktrees",
                summary=f"{name} ({tool}) — {momentum}",
            )
        )

    return items


def collect_session_claims(root: Path) -> list[PulseItem]:
    """Summarize fresh active session claims without dumping full claim state."""
    items: list[PulseItem] = []
    try:
        from session_preflight import list_claims
    except Exception:
        return items

    claims = list_claims(fresh_only=True)
    if len(claims) <= 1:
        return items

    mutating_by_branch: dict[str, set[str]] = {}
    for claim in claims:
        if claim.mode != "mutating":
            continue
        mutating_by_branch.setdefault(claim.branch, set()).add(claim.tool)

    dangerous = {branch: tools for branch, tools in mutating_by_branch.items() if len(tools) > 1}
    if dangerous:
        branch, tools = sorted(dangerous.items())[0]
        items.append(
            PulseItem(
                category="decaying",
                severity="high",
                source="session_claims",
                summary=f"Active sessions: dangerous same-branch mutating claims on {branch} ({', '.join(sorted(tools))})",
            )
        )
        return items

    branches = {claim.branch for claim in claims}
    items.append(
        PulseItem(
            category="paused",
            severity="low",
            source="session_claims",
            summary=f"Active sessions: {len(claims)} fresh claims across {len(branches)} branch(es) — parallel appears isolated",
        )
    )
    return items


def collect_action_queue(canonical: Path, *, now: datetime | None = None) -> list[PulseItem]:
    """Parse the canonical active-work queue.

    `now` overrides the reference clock for the staleness sweep. Default uses
    wall clock; tests pass an explicit datetime to make staleness assertions
    deterministic.
    """
    items: list[PulseItem] = []
    queue_path = canonical / "docs" / "runtime" / "action-queue.yaml"
    if not queue_path.exists():
        return items

    queue = load_queue(canonical)
    stale_ids = {item.id for item in queue_stale_items(queue, now=now)}
    for item in queue.items:
        if item.status in {"closed", "superseded"}:
            continue

        category = "ready"
        severity = "low"
        if item.status == "blocked":
            category = "decaying"
            severity = "medium"
        elif item.status in {"waiting_observation", "parked"}:
            category = "paused"
            severity = "low"
        elif item.priority == "P1":
            severity = "medium"

        if item.id in stale_ids:
            category = "decaying"
            severity = "medium" if item.priority != "P3" else "low"

        summary = f"{item.id}: {item.title}"
        if len(summary) > 100:
            summary = summary[:97] + "..."
        detail_parts = [f"status={item.status}", f"priority={item.priority}"]
        if item.close_before_new_work:
            detail_parts.append("close-first")
        items.append(
            PulseItem(
                category=category,
                severity=severity,
                source="action_queue",
                summary=summary,
                detail=", ".join(detail_parts),
            )
        )

    return items


def collect_debt_ledger(root: Path) -> list[PulseItem]:
    """Surface open debt bullets as explicit pulse follow-up items."""
    debt_path = root / "docs" / "runtime" / "debt-ledger.md"
    if not debt_path.exists():
        return []

    try:
        lines = debt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    items: list[PulseItem] = []
    in_open_debt = False
    for line in lines:
        stripped = line.strip()
        if stripped == "## Open Debt":
            in_open_debt = True
            continue
        if in_open_debt and stripped.startswith("## "):
            break
        if not in_open_debt or not line.startswith("- "):
            continue
        if stripped.startswith("- ~~"):
            continue

        text = stripped.removeprefix("- ").strip()
        match = re.match(r"`([^`]+)`\s*(?:â€”|—|–|:|-)\s*(.*)", text)
        debt_id = match.group(1) if match else None
        description = match.group(2) if match else text
        summary = f"{debt_id}: {description}" if debt_id else description
        if len(summary) > 140:
            summary = summary[:137] + "..."
        items.append(
            PulseItem(
                category="unactioned",
                severity="low",
                source="debt_ledger",
                summary=summary,
                detail="docs/runtime/debt-ledger.md",
            )
        )

    return items


def _normalize_coverage_text(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _coverage_tokens(text: str) -> list[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "into",
        "item",
        "work",
        "fix",
        "run",
        "check",
        "pulse",
        "finding",
        "findings",
        "generic",
        "review",
    }
    tokens = [token for token in _normalize_coverage_text(text).split() if len(token) >= 4 and token not in stop]
    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped[:8]


def _coverage_needles(*needles: str) -> list[tuple[str, list[str]]]:
    candidates: list[tuple[str, list[str]]] = []
    for needle in needles:
        normalized = _normalize_coverage_text(needle)
        if not normalized:
            continue
        tokens = _coverage_tokens(needle)
        if (
            len(tokens) >= 2
            or len(normalized) >= 16
            or (len(normalized) >= 6 and any(ch.isdigit() for ch in normalized))
        ):
            candidates.append((normalized, tokens))
    return candidates


def _queue_coverage_blobs(root: Path) -> list[str]:
    queue_path = root / "docs" / "runtime" / "action-queue.yaml"
    if not queue_path.exists():
        return []
    try:
        queue = load_queue(root)
    except Exception:
        return []

    blobs: list[str] = []
    for item in queue.items:
        if item.status in {"closed", "superseded"}:
            continue
        parts = [
            item.id,
            item.title,
            item.status,
            item.next_action,
            item.exit_criteria,
            item.notes_ref or "",
            item.override_note or "",
            " ".join(item.decision_refs),
            " ".join(item.evidence_refs),
        ]
        blobs.append(_normalize_coverage_text(" ".join(parts)))
    return blobs


def _is_covered_by_queue(root: Path, *needles: str) -> bool:
    blobs = _queue_coverage_blobs(root)
    if not blobs:
        return False

    candidates = _coverage_needles(*needles)
    for blob in blobs:
        if any(normalized and normalized in blob for normalized, _tokens in candidates):
            return True
        for _normalized, tokens in candidates:
            if len(tokens) >= 2 and sum(1 for token in tokens if token in blob) >= min(3, len(tokens)):
                return True
    return False


def _parse_debt_entries(root: Path) -> list[DebtEntry]:
    debt_path = root / "docs" / "runtime" / "debt-ledger.md"
    if not debt_path.exists():
        return []
    try:
        lines = debt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    entries: list[DebtEntry] = []
    in_open_debt = False
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped == "## Open Debt":
            in_open_debt = True
            continue
        if in_open_debt and stripped.startswith("## "):
            break
        if not in_open_debt or not line.startswith("- "):
            continue
        if stripped.startswith("- ~~"):
            continue

        text = stripped.removeprefix("- ").strip()
        id_match = re.match(r"`([^`]+)`", text)
        debt_id = id_match.group(1) if id_match else None
        description = text[id_match.end() :].lstrip(" :-") if id_match else text
        lowered = text.lower()
        entries.append(
            DebtEntry(
                debt_id=debt_id,
                text=description or text,
                line_no=line_no,
                has_owner=bool(re.search(r"\bowner\s*[:=]", lowered)),
                has_status=bool(re.search(r"\bstatus\s*[:=]", lowered)),
                has_parking_decision=any(
                    token in lowered for token in ("parking_lot", "parking lot", "parked", "wont_do")
                ),
            )
        )
    return entries


def collect_debt_reconciliation(root: Path) -> list[PulseItem]:
    """Flag open debt that lacks queue coverage or an explicit parking decision."""
    items: list[PulseItem] = []
    for entry in _parse_debt_entries(root):
        debt_key = entry.debt_id or entry.text
        covered = _is_covered_by_queue(root, debt_key, entry.text)
        has_complete_metadata = entry.has_owner and entry.has_status
        intentionally_parked = entry.has_parking_decision and has_complete_metadata
        if covered or intentionally_parked:
            continue

        missing = []
        if not covered:
            missing.append("queue item")
        if not entry.has_owner:
            missing.append("owner")
        if not entry.has_status:
            missing.append("status")
        if not entry.has_parking_decision:
            missing.append("parking decision")
        items.append(
            PulseItem(
                category="unactioned",
                severity="medium",
                source="debt_reconciliation",
                summary=f"Open debt lacks follow-up coverage: {debt_key}",
                detail=f"docs/runtime/debt-ledger.md:{entry.line_no}; missing={', '.join(missing)}",
                action="Add an action-queue item or mark owner/status/parking_lot in docs/runtime/debt-ledger.md.",
            )
        )
    return items


def collect_queue_reconciliation(canonical: Path, items: list[PulseItem]) -> list[PulseItem]:
    """Report actionable pulse findings that are not represented in action-queue.yaml."""
    missing: list[PulseItem] = []
    ignored_sources = {
        "action_queue",
        "queue_reconciliation",
        "followup_coverage",
        "debt_ledger",
        "debt_reconciliation",
        "plan_reconciliation",
        "git",
        "worktree",
        "worktrees",
        "session_delta",
    }
    for item in items:
        if item.source in ignored_sources:
            continue
        actionable = item.category in {"broken", "decaying", "unactioned"}
        if not actionable:
            continue
        if _is_covered_by_queue(canonical, item.summary, item.detail or ""):
            continue
        missing.append(
            PulseItem(
                category="unactioned",
                severity="high" if item.category == "broken" else "medium",
                source="queue_reconciliation",
                summary=f"Pulse finding has no queue coverage: {item.summary}",
                detail=f"source={item.source}; severity={item.severity}",
                action="Add or intentionally park an action-queue.yaml item for this pulse finding.",
            )
        )
    return missing


def _parse_plan_frontmatter(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}
    data: dict[str, str] = {}
    for raw_line in text[4:end].splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data


def _plan_title(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("# "):
                return line.removeprefix("# ").strip()
    except OSError:
        pass
    return path.stem


def _active_plan_entries(root: Path) -> list[PlanEntry]:
    active_root = root / "docs" / "plans" / "active"
    if not active_root.exists():
        return []
    entries: list[PlanEntry] = []
    for path in sorted(active_root.rglob("*.md")):
        meta = _parse_plan_frontmatter(path)
        if meta.get("status") != "active":
            continue
        entries.append(
            PlanEntry(
                path=path.relative_to(root),
                title=_plan_title(path),
                owner=meta.get("owner", ""),
                status=meta.get("status", ""),
                last_reviewed=meta.get("last_reviewed", ""),
            )
        )
    return entries


def collect_plan_reconciliation(root: Path) -> list[PulseItem]:
    """Flag active design plans that have not been routed into the queue."""
    items: list[PulseItem] = []
    for entry in _active_plan_entries(root):
        rel = entry.path.as_posix()
        if _is_covered_by_queue(root, rel, entry.title):
            continue
        owner = entry.owner or "unowned"
        items.append(
            PulseItem(
                category="unactioned",
                severity="medium",
                source="plan_reconciliation",
                summary=f"Active plan has no queue coverage: {entry.title}",
                detail=f"{rel}; owner={owner}; last_reviewed={entry.last_reviewed or 'missing'}",
                action="Add a queue item referencing this docs/plans/active file, or archive/park the plan.",
            )
        )
    return items


def collect_followup_coverage(canonical: Path, items: list[PulseItem]) -> list[PulseItem]:
    """Flag actionable pulse findings when the canonical queue has no open work."""
    queue_path = canonical / "docs" / "runtime" / "action-queue.yaml"
    if not queue_path.exists():
        return []

    try:
        queue = load_queue(canonical)
    except Exception:
        return []
    if any(item.is_open for item in queue.items):
        return []

    actionable = [
        item
        for item in items
        if item.source != "followup_coverage"
        and (item.category == "broken" or (item.category == "decaying" and item.severity == "high"))
    ]
    if not actionable:
        return []

    return [
        PulseItem(
            category="unactioned",
            severity="high",
            source="followup_coverage",
            summary=(f"Action queue has 0 open items but pulse found {len(actionable)} broken/high-decay finding(s)"),
            detail=", ".join(f"{item.source}:{item.summary}" for item in actionable[:5]),
            action="Add or intentionally park queue coverage before treating follow-up as handled.",
        )
    ]


def collect_ralph_deferred(root: Path) -> list[PulseItem]:
    """Parse open deferred findings from ralph deferred-findings.md."""
    items: list[PulseItem] = []
    deferred_path = root / "docs" / "ralph-loop" / "deferred-findings.md"
    if not deferred_path.exists():
        return items

    try:
        text = deferred_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return items

    in_open = False
    header_seen = False
    for line in text.splitlines():
        if "## Open Findings" in line:
            in_open = True
            continue
        if in_open:
            if line.startswith("## "):
                break
            if line.startswith("|") and "ID" in line and "Severity" in line:
                header_seen = True
                continue
            if line.startswith("|--") or line.startswith("|-"):
                continue
            if header_seen and line.startswith("|"):
                cols = [c.strip() for c in line.split("|")]
                cols = [c for c in cols if c]
                if len(cols) >= 5:
                    finding_id = cols[0]
                    severity = cols[2].upper()
                    target = cols[3]
                    desc = cols[4]
                    items.append(
                        PulseItem(
                            category="unactioned",
                            severity=severity.lower() if severity in ("HIGH", "MEDIUM", "LOW") else "low",
                            source="ralph",
                            summary=f"{finding_id}: {desc}",
                            detail=f"Target: {target}",
                        )
                    )

    return items


def collect_upcoming_sessions(db_path: Path) -> list[dict]:
    """Find trading sessions starting in the next 6 hours with strategy counts."""
    sessions: list[dict] = []
    try:
        from datetime import time, timedelta
        from zoneinfo import ZoneInfo

        import duckdb

        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS
        from pipeline.dst import SESSION_CATALOG

        brisbane = ZoneInfo("Australia/Brisbane")
        now = datetime.now(brisbane).replace(tzinfo=None)  # naive Brisbane time
        today = now.date()

        for label, entry in SESSION_CATALOG.items():
            if entry.get("type") != "dynamic":
                continue
            try:
                h, m = entry["resolver"](today)
            except Exception:
                continue
            session_dt = datetime.combine(today, time(h, m))
            if session_dt < now:
                tomorrow = today + timedelta(days=1)
                try:
                    h2, m2 = entry["resolver"](tomorrow)
                    session_dt = datetime.combine(tomorrow, time(h2, m2))
                except Exception:
                    continue
            hours_away = (session_dt - now).total_seconds() / 3600
            if 0 <= hours_away <= 6:
                info: dict = {
                    "label": label,
                    "brisbane_time": session_dt.strftime("%H:%M"),
                    "hours_away": round(hours_away, 1),
                    "instruments": {},
                }
                if db_path.exists():
                    try:
                        con = duckdb.connect(str(db_path), read_only=True)
                        try:
                            shelf_relation = deployable_validated_relation(con)
                            for inst in ACTIVE_ORB_INSTRUMENTS:
                                row = con.execute(
                                    f"SELECT COUNT(*) FROM {shelf_relation} WHERE instrument = ? AND orb_label = ?",
                                    [inst, label],
                                ).fetchone()
                                if row and row[0] > 0:
                                    info["instruments"][inst] = row[0]
                        finally:
                            con.close()
                    except Exception:
                        pass
                sessions.append(info)
        sessions.sort(key=lambda s: s["hours_away"])
    except Exception:
        pass
    return sessions


def collect_worktree_conflicts(canonical: Path) -> list[PulseItem]:
    """Detect file overlap between active worktrees (merge conflict radar)."""
    items: list[PulseItem] = []
    worktree_files: dict[str, set[str]] = {}
    for data in _worktree_metadata(canonical, authoritative=True):
        branch = data.get("branch", "")
        name = data.get("name", "?")
        if not branch:
            continue
        r = _run_git(canonical, "diff", "--name-only", f"main...{branch}")
        if r and r.returncode == 0:
            files = {f.strip() for f in r.stdout.splitlines() if f.strip()}
            if files:
                worktree_files[name] = files

    # Find overlaps
    names = list(worktree_files.keys())
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            overlap = worktree_files[name_a] & worktree_files[name_b]
            if overlap:
                files_str = ", ".join(sorted(overlap)[:3])
                extra = f" +{len(overlap) - 3} more" if len(overlap) > 3 else ""
                items.append(
                    PulseItem(
                        category="decaying",
                        severity="medium",
                        source="conflicts",
                        summary=f"Merge risk: '{name_a}' and '{name_b}' both touch {files_str}{extra}",
                    )
                )

    return items


def collect_session_delta(root: Path, canonical: Path, *, tool_name: str = "unknown") -> list[str]:
    """What changed since THIS tool's last session (session continuity fingerprint)."""
    lines: list[str] = []
    # Read the last-session marker
    marker_path = root / ".pulse_last_session.json"
    current_head = _git_head(root)

    if marker_path.exists():
        try:
            data = json.loads(marker_path.read_text(encoding="utf-8"))
            last_head = data.get("head", "")
            last_tool = data.get("tool", "unknown")
            last_at = data.get("at", "")

            if last_head and last_head != current_head:
                r = _run_git(root, "log", "--oneline", f"{last_head}..{current_head}")
                if r and r.returncode == 0:
                    commits = [ln for ln in r.stdout.splitlines() if ln.strip()]
                    if commits:
                        lines.append(f"Since last session ({last_tool}, {last_at[:10] if last_at else '?'}):")
                        for c in commits[:5]:
                            lines.append(f"  {c}")
                        if len(commits) > 5:
                            lines.append(f"  ... +{len(commits) - 5} more commits")
        except (json.JSONDecodeError, OSError):
            pass

    # Write current marker
    try:
        marker_path.write_text(
            json.dumps({"head": current_head, "tool": tool_name, "at": datetime.now(UTC).isoformat()}, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass

    return lines


def _workstream_momentum(data: dict, canonical: Path) -> str:
    """Derive momentum label for a worktree from git history."""
    branch = data.get("branch", "")
    created = data.get("created_at", "")
    if not branch:
        return "unknown"

    days_old = 0
    if created:
        try:
            created_dt = datetime.fromisoformat(created)
            days_old = max(0, (datetime.now(UTC) - created_dt).days)
        except (ValueError, TypeError):
            pass

    r = _run_git(canonical, "rev-list", "--count", f"main..{branch}")
    commits = 0
    if r and r.returncode == 0:
        try:
            commits = int(r.stdout.strip())
        except ValueError:
            pass

    if days_old <= 1:
        return f"new, {commits} commit(s)"
    if commits == 0:
        return f"{days_old}d old, 0 commits — STALLED"
    if days_old > 5 and commits < 3:
        return f"{days_old}d old, {commits} commit(s) — slow"
    return f"{days_old}d old, {commits} commit(s)"


def _compute_recommendation(report: PulseReport) -> str:
    """Pick the single most impactful next action."""
    if report.broken:
        top = report.broken[0]
        action = top.action or "fix the issue"
        return f"Fix: {top.summary} → {action}"

    runtime_snapshot_items = [item for item in report.paused if item.source == "runtime_snapshot"]
    if runtime_snapshot_items:
        top = runtime_snapshot_items[0]
        action = top.action or "refresh the fast runtime snapshot"
        return f"Refresh: {top.summary} → {action}"

    if report.upcoming_sessions:
        s = report.upcoming_sessions[0]
        strats = sum(s.get("instruments", {}).values())
        if strats > 0:
            return f"Prep: {s['label']} in {s['hours_away']}h ({strats} strategies) → /trade-book"

    if report.decaying:
        top = report.decaying[0]
        action = top.action or "investigate"
        return f"Check: {top.summary} → {action}"

    ready = report.ready
    if ready:
        return f"Next: {ready[0].summary}"

    return "All clear — start new work or /orient --full for deep check"


def _compute_capital_recommendation(report: PulseReport) -> str:
    """Return the capital cockpit recommendation token plus concise reason."""
    blocked = [
        item
        for item in report.broken
        if item.source in {"execution", "criterion11", "sr_monitor", "lifecycle", "deployment"}
    ]
    if blocked:
        return f"NO_CHANGE: capital evidence blocked - {blocked[0].summary}"

    if report.pause_summary and report.pause_summary.get("paused_count", 0):
        return f"PAUSE: {report.pause_summary.get('paused_count')} lane(s) already paused"

    packet = report.capital_packet_summary or {}
    if packet and not packet.get("valid", True):
        return f"NO_CHANGE: capital packet invalid - {packet.get('error') or 'schema/read failure'}"

    would_add = packet.get("would_add", []) if isinstance(packet, dict) else []
    would_remove = packet.get("would_remove", []) if isinstance(packet, dict) else []
    if would_add or would_remove:
        boundary = str(packet.get("capital_boundary") or "")
        if boundary == REPORT_ONLY_CAPITAL_BOUNDARY:
            return f"SHADOW: packet has add={len(would_add)} remove={len(would_remove)} but boundary is {boundary}"
        if boundary in AUTHORIZED_PROMOTE_CAPITAL_BOUNDARIES:
            return f"PROMOTE: packet proposes add={len(would_add)} remove={len(would_remove)} and gates are clear"
        return f"NO_CHANGE: capital packet boundary {boundary or '<missing>'} is not recognized deployment authority"

    return "NO_CHANGE: no capital add/remove and evidence gates are clear"


def _update_time_since_green(root: Path, is_green: bool) -> str | None:
    """Track and return how long since the system was fully clean."""
    cache_path = root / CACHE_FILE
    try:
        data = {}
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        data = {}

    if is_green:
        data["last_green"] = datetime.now(UTC).isoformat()
        try:
            cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass
        return "now"

    last_green = data.get("last_green")
    if last_green:
        try:
            dt = datetime.fromisoformat(last_green)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            hours = (datetime.now(UTC) - dt).total_seconds() / 3600
            if hours < 1:
                return f"{int(hours * 60)}m ago"
            if hours < 48:
                return f"{int(hours)}h ago"
            return f"{int(hours / 24)}d ago"
        except (ValueError, TypeError):
            pass

    return None


def _attach_skill_suggestions(items: list[PulseItem]) -> None:
    """Attach actionable skill/command suggestions to pulse items."""
    for item in items:
        if item.action:
            continue
        template = SKILL_SUGGESTIONS.get(item.source)
        if template:
            # Substitute {inst} if present in summary
            inst_match = re.match(r"^(M\w+):", item.summary)
            if inst_match and "{inst}" in template:
                item.action = template.format(inst=inst_match.group(1))
            else:
                item.action = template.replace(" {inst}", "")


def _queue_id_from_summary(summary: str) -> str | None:
    match = re.match(r"^([A-Za-z0-9_.-]+):", summary)
    return match.group(1) if match else None


def _queue_claim_command(queue_id: str, *, platform_name: str | None = None) -> str:
    repo_python = _preferred_repo_python()
    python_cmd = str(repo_python) if repo_python is not None else "python"
    platform = platform_name or os.name
    script_path = "scripts\\tools\\work_queue.py" if platform == "nt" else "scripts/tools/work_queue.py"
    return f"{python_cmd} {script_path} claim --item {queue_id} --tool codex"


def _build_next_actions(items: list[PulseItem], *, limit: int = 5) -> list[NextAction]:
    """Return compact operator actions backed by queue IDs or exact commands."""
    actions: list[NextAction] = []
    seen: set[str] = set()
    priority = {"broken": 0, "decaying": 1, "unactioned": 2, "ready": 3, "paused": 4}
    severity = {"high": 0, "medium": 1, "low": 2}

    for item in sorted(
        items, key=lambda i: (priority.get(i.category, 9), severity.get(i.severity, 9), i.source, i.summary)
    ):
        if item.source == "action_queue":
            queue_id = _queue_id_from_summary(item.summary)
            if not queue_id:
                continue
            key = f"queue:{queue_id}"
            if key in seen:
                continue
            seen.add(key)
            actions.append(
                NextAction(
                    kind="queue",
                    label=item.summary,
                    queue_id=queue_id,
                    command=_queue_claim_command(queue_id),
                    source=item.source,
                )
            )
        elif item.action:
            key = f"action:{item.action}:{item.summary}"
            if key in seen:
                continue
            seen.add(key)
            actions.append(
                NextAction(
                    kind="command" if item.action.startswith(("python", ".\\", "uv ")) else "reconcile",
                    label=item.summary,
                    command=item.action,
                    source=item.source,
                )
            )
        if len(actions) >= limit:
            break
    return actions


# ---------------------------------------------------------------------------
# Cache — only caches expensive collectors (drift + tests) keyed on HEAD.
# Cheap collectors (<500ms total) always run fresh.
# ---------------------------------------------------------------------------

CACHE_FILE = ".pulse_cache.json"


def _read_expensive_cache(root: Path, head: str) -> dict | None:
    """Read cached drift/test results. Returns None on miss or HEAD mismatch."""
    cache_path = root / CACHE_FILE
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("head") != head:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def _write_expensive_cache(
    root: Path,
    head: str,
    drift_items: list[PulseItem] | None,
    test_items: list[PulseItem] | None,
    existing: dict | None,
) -> None:
    """Write expensive collector results to cache. Merges with existing data for partial runs.

    None = collector was skipped this run, preserve existing cached value.
    [] = collector ran and found nothing (clean).
    [items] = collector ran and found issues.
    """
    data = {
        "head": head,
        "cached_at": datetime.now(UTC).isoformat(),
        "drift_items": (
            [asdict(i) for i in drift_items] if drift_items is not None else (existing or {}).get("drift_items")
        ),
        "test_items": (
            [asdict(i) for i in test_items] if test_items is not None else (existing or {}).get("test_items")
        ),
        "fitness_deep": (existing or {}).get("fitness_deep"),
    }
    try:
        (root / CACHE_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def _fitness_items_from_summary(summary: dict) -> list[PulseItem]:
    """Derive PulseItems from a cached deep fitness summary dict."""
    items: list[PulseItem] = []
    for inst, data in summary.items():
        if not isinstance(data, dict):
            continue
        decay = data.get("decay", 0)
        watch = data.get("watch", 0)
        if decay > 0:
            items.append(
                PulseItem(
                    category="decaying",
                    severity="high" if decay >= 5 else "medium",
                    source="fitness",
                    summary=f"{inst}: {decay} DECAY, {watch} WATCH strategies",
                )
            )
        elif watch > 0:
            items.append(
                PulseItem(
                    category="decaying",
                    severity="low",
                    source="fitness",
                    summary=f"{inst}: {watch} WATCH strategies",
                )
            )
    return items


def _resolve_db_path(root: Path, canonical: Path) -> Path:
    """Find gold.db — canonical root first, then GOLD_DB_PATH, then worktree."""
    candidate = canonical / "gold.db"
    if candidate.exists():
        return candidate
    try:
        from pipeline.paths import GOLD_DB_PATH

        return GOLD_DB_PATH
    except ImportError:
        return root / "gold.db"


def build_pulse(
    root: Path,
    db_path: Path | None = None,
    no_cache: bool = False,
    deep: bool = False,
    fast: bool = False,
    skip_drift: bool = False,
    skip_tests: bool = False,
    tool_name: str = "unknown",
) -> PulseReport:
    """Build a full pulse report from all collectors.

    Modes:
      default: run everything, cache expensive results
      --fast:  serve drift/tests/deep-fitness from cache if available, skip if not
      --deep:  run full fitness computation (slow), cache result
      --skip-drift/--skip-tests: hard skip, never run or serve from cache
    """
    # Resolve canonical root once — used by worktrees, memory, DB resolution
    canonical = _canonical_repo_root(root)

    if db_path is None:
        db_path = _resolve_db_path(root, canonical)

    head = _git_head(root)

    # --- Expensive collectors: drift + tests (cached by HEAD) ---
    cache = _read_expensive_cache(root, head) if not no_cache else None
    used_cache = False

    if skip_drift:
        drift_items: list[PulseItem] = []
    elif cache and cache.get("drift_items") is not None:
        drift_items = [PulseItem(**i) for i in cache["drift_items"]]
        used_cache = True
    elif fast:
        drift_items = []  # fast mode, no cache: skip (don't run 30s check)
    else:
        drift_items = collect_drift(root)

    if skip_tests:
        test_items: list[PulseItem] = []
    elif cache and cache.get("test_items") is not None:
        test_items = [PulseItem(**i) for i in cache["test_items"]]
        used_cache = True
    elif fast:
        test_items = []  # fast mode, no cache: skip
    else:
        test_items = collect_tests(root)

    # Write cache (merge: preserve existing values for skipped collectors)
    if not skip_drift or not skip_tests:
        _write_expensive_cache(
            root,
            head,
            drift_items if not skip_drift and not fast else None,
            test_items if not skip_tests and not fast else None,
            cache,
        )

    # --- Cheap collectors: always fresh ---
    system_identity, identity_items = collect_system_identity(root, canonical, db_path)
    work_capsule_summary, work_capsule_items = collect_work_capsule(root)
    if fast:
        system_brief_bundle = None
        system_brief_items = []
    else:
        system_brief_bundle, system_brief_items = collect_system_brief(root, db_path, tool_name or "unknown")
    system_brief_summary = system_brief_bundle["summary"] if system_brief_bundle else None
    system_brief_payload = system_brief_bundle["payload"] if system_brief_bundle else None
    staleness_items = collect_staleness(root, db_path)

    # Fitness: deep > cached deep > fast proxy
    if deep:
        fitness_summary, fitness_items = collect_fitness_deep(db_path)
        # Cache the deep result for fast mode
        if cache is not None or not no_cache:
            existing = _read_expensive_cache(root, head)
            if existing is not None:
                existing["fitness_deep"] = fitness_summary
                try:
                    (root / CACHE_FILE).write_text(json.dumps(existing, indent=2), encoding="utf-8")
                except OSError:
                    pass
    elif cache and cache.get("fitness_deep"):
        # Serve cached deep result (fast mode benefits from prior --deep run)
        fitness_summary = cache["fitness_deep"]
        fitness_items = _fitness_items_from_summary(fitness_summary)
    else:
        fitness_summary, fitness_items = collect_fitness_fast(db_path)

    deployment_summary, deployment_items = collect_deployment_state(db_path)
    execution_summary = deployment_summary.get("execution_evidence") if isinstance(deployment_summary, dict) else None
    survival_summary, sr_summary, pause_summary, lifecycle_items = collect_lifecycle_control(db_path)
    capital_packet_summary, capital_packet_items = collect_capital_packet(root)
    live_readiness_summary, live_readiness_items = collect_live_readiness()
    handoff_context, handoff_items = collect_handoff(root)
    worktree_items = collect_worktrees(canonical, fast=fast)
    claim_items = collect_session_claims(root)
    conflict_items = [] if fast else collect_worktree_conflicts(canonical)
    git_items = collect_git_state(root)
    action_items = collect_action_queue(canonical)
    debt_items = collect_debt_ledger(root)
    debt_reconciliation_items = collect_debt_reconciliation(root)
    plan_reconciliation_items = collect_plan_reconciliation(root)
    ralph_items = collect_ralph_deferred(root)
    session_delta = collect_session_delta(root, canonical, tool_name=tool_name)
    upcoming = collect_upcoming_sessions(db_path)

    # Flag stale handoff (>2 days old)
    handoff_date_str = handoff_context.get("date")
    if handoff_date_str:
        try:
            from datetime import date

            handoff_date = date.fromisoformat(handoff_date_str)
            days_old = (date.today() - handoff_date).days
            if days_old >= 2:
                handoff_items.append(
                    PulseItem(
                        category="decaying",
                        severity="medium" if days_old >= 5 else "low",
                        source="handoff",
                        summary=f"Handoff context is {days_old} days old — consider updating",
                    )
                )
        except ValueError:
            pass

    # Collect all items
    all_items = (
        identity_items
        + work_capsule_items
        + system_brief_items
        + drift_items
        + test_items
        + staleness_items
        + fitness_items
        + deployment_items
        + lifecycle_items
        + capital_packet_items
        + live_readiness_items
        + claim_items
        + conflict_items
        + handoff_items
        + worktree_items
        + git_items
        + action_items
        + debt_items
        + debt_reconciliation_items
        + plan_reconciliation_items
        + ralph_items
    )
    all_items += collect_queue_reconciliation(canonical, all_items)
    all_items += collect_followup_coverage(canonical, all_items)

    # Attach skill invocation suggestions
    _attach_skill_suggestions(all_items)
    next_actions = _build_next_actions(all_items)

    # Check system health for time-since-green
    is_green = not any(i.category in ("broken", "decaying") for i in all_items)
    time_since_green = _update_time_since_green(root, is_green)

    # --- Assemble report ---
    report = PulseReport(
        generated_at=datetime.now(UTC).isoformat(),
        cache_hit=used_cache,
        git_head=head,
        git_branch=_git_branch(root),
        items=all_items,
        system_identity=system_identity,
        handoff_tool=handoff_context.get("tool"),
        handoff_date=handoff_context.get("date"),
        handoff_summary=handoff_context.get("summary"),
        handoff_next_steps=handoff_context.get("next_steps", []),
        fitness_summary=fitness_summary,
        deployment_summary=deployment_summary,
        survival_summary=survival_summary,
        sr_summary=sr_summary,
        pause_summary=pause_summary,
        execution_summary=execution_summary,
        capital_packet_summary=capital_packet_summary,
        live_readiness_summary=live_readiness_summary,
        upcoming_sessions=upcoming,
        time_since_green=time_since_green,
        session_delta=session_delta,
        work_capsule_summary=work_capsule_summary,
        system_brief_summary=system_brief_summary,
        history_debt_summary={
            "decision_ref_count": len(system_brief_payload["decision_refs"]) if system_brief_payload else 0,
            "debt_ref_count": len(system_brief_payload["debt_refs"]) if system_brief_payload else 0,
        },
        startup_latency_ms=system_brief_payload["startup_latency_ms"] if system_brief_payload else None,
        orientation_cost_budget=system_brief_payload["orientation_cost_budget"] if system_brief_payload else None,
        next_actions=next_actions,
    )

    # Compute single recommendation after full report is assembled
    report.recommendation = _compute_recommendation(report)
    report.capital_recommendation = _compute_capital_recommendation(report)

    return report


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

CATEGORY_LABELS = {
    "broken": "FIX NOW",
    "decaying": "ACT SOON",
    "ready": "ON DECK",
    "unactioned": "DEBT",
    "paused": "PAUSED",
}

SEVERITY_ICONS = {"high": "!", "medium": "~", "low": " "}


def _next_action_get(action: NextAction | dict, key: str) -> str | None:
    if isinstance(action, dict):
        value = action.get(key)
    else:
        value = getattr(action, key)
    return str(value) if value is not None else None


def format_text(report: PulseReport) -> str:
    """Concise terminal output for humans."""
    lines: list[str] = []
    meta_parts = [f"Branch: {report.git_branch}", f"HEAD: {report.git_head}"]
    if report.cache_hit:
        meta_parts.append("(cached)")
    if report.time_since_green:
        meta_parts.append(f"Green: {report.time_since_green}")
    lines.append("=" * 60)
    lines.append(f"PROJECT PULSE | {'  '.join(meta_parts)}")
    lines.append("=" * 60)
    lines.append("")

    # Session delta (what changed since last session)
    if report.session_delta:
        for dl in report.session_delta:
            lines.append(dl)
        lines.append("")

    if report.handoff_summary:
        lines.append(f"Last: {report.handoff_tool or '?'} ({report.handoff_date or '?'})")
        lines.append(f"  {report.handoff_summary}")
        lines.append("")

    if report.system_identity:
        identity = report.system_identity
        relations = identity.get("published_relations", {})
        doctrine_count = len(identity.get("doctrine_docs", []))
        backbone_count = len(identity.get("backbone_modules", []))
        lines.append("System identity:")
        lines.append(f"  Root: {identity.get('canonical_repo_root')} | DB: {identity.get('canonical_db_path')}")
        if identity.get("db_override_active"):
            lines.append(f"  Active DB override: {identity.get('selected_db_path')}")
        lines.append(f"  Active ORB instruments: {', '.join(identity.get('active_orb_instruments', [])) or 'none'}")
        lines.append(
            f"  Shelf: {relations.get('active', '?')}, {relations.get('deployable', '?')} | "
            f"Authority: {identity.get('authority_map_doc')} ({doctrine_count} doctrine, {backbone_count} backbone)"
        )
        lines.append("")

    if report.system_brief_summary:
        brief = report.system_brief_summary
        lines.append("System brief:")
        lines.append(
            "  "
            f"Route {brief.get('task_id')} [{brief.get('briefing_level')}] | "
            f"owners {brief.get('canonical_owner_count')} | "
            f"views {brief.get('required_live_view_count')} | "
            f"blockers {brief.get('blocker_count')} | warnings {brief.get('warning_count')}"
        )
        if report.work_capsule_summary:
            lines.append(f"  Capsule: {report.work_capsule_summary.get('path')}")
        if report.startup_latency_ms is not None and report.orientation_cost_budget:
            lines.append(
                f"  Latency {report.startup_latency_ms}ms / {report.orientation_cost_budget.get('budget_ms')}ms budget"
            )
        lines.append("")

    if report.handoff_next_steps:
        lines.append("Next steps:")
        for step in report.handoff_next_steps[:5]:
            lines.append(f"  {step}")
        if len(report.handoff_next_steps) > 5:
            lines.append(f"  ... +{len(report.handoff_next_steps) - 5} more")
        lines.append("")

    if report.next_actions:
        lines.append("Next actions:")
        for action in report.next_actions[:5]:
            label = _next_action_get(action, "label") or "action"
            command = _next_action_get(action, "command")
            queue_id = _next_action_get(action, "queue_id")
            if queue_id and command:
                lines.append(f"  [queue:{queue_id}] {command}")
            elif command:
                lines.append(f"  {command}  # {label}")
            else:
                lines.append(f"  {label}")
        lines.append("")

    if report.deployment_summary:
        ds = report.deployment_summary
        lines.append("Live control:")
        lines.append(
            "  "
            f"Profile {ds.get('profile_id')} | deployed {ds.get('deployed_count', '?')} | "
            f"active validated {ds.get('validated_active_count', '?')} | "
            f"validated-only {len(ds.get('validated_not_deployed', []))}"
        )
        if report.execution_summary:
            es = report.execution_summary
            lines.append(
                "  "
                f"Execution evidence {es.get('covered_count', 0)}/{es.get('deployed_count', 0)} lanes | "
                f"missing {es.get('missing_count', 0)} | stale {len(es.get('stale_execution_strategy_ids', []))}"
            )
            lines.append(f"  Execution sources: paper_trades + live_journal ({es.get('live_journal_status')})")
            missing = es.get("missing_execution_strategy_ids", [])
            if missing:
                lines.append(f"  Missing execution: {', '.join(str(x) for x in missing[:3])}")
        if report.survival_summary:
            ss = report.survival_summary
            op = ss.get("operational_pass_probability")
            op_str = f"{float(op):.1%}" if isinstance(op, float | int) else "?"
            lines.append(
                "  "
                f"C11 {('PASS' if ss.get('gate_ok') else 'BLOCK')} {op_str} | "
                f"as_of {ss.get('as_of_date') or '?'} | age {ss.get('report_age_days') if ss.get('report_age_days') is not None else '?'}d"
            )
        if report.sr_summary:
            sr = report.sr_summary
            counts = sr.get("counts", {})
            streams = sr.get("stream_counts", {})
            stream_suffix = ""
            if streams:
                stream_parts = [f"{k}:{v}" for k, v in sorted(streams.items())]
                stream_suffix = f" | streams {', '.join(stream_parts)}"
            lines.append(
                "  "
                f"C12 SR continue={counts.get('CONTINUE', 0)} alarm={counts.get('ALARM', 0)} "
                f"no_data={counts.get('NO_DATA', 0)} | age {sr.get('state_age_days') if sr.get('state_age_days') is not None else '?'}d"
                f"{stream_suffix}"
            )
            if sr.get("reviewed_watch_count"):
                lines.append(f"  C12 reviewed WATCH alarms: {sr.get('reviewed_watch_count')}")
        else:
            lines.append("  C12 SR BLOCK missing/invalid")
        if report.pause_summary:
            ps = report.pause_summary
            lines.append(f"  Paused lanes: {ps.get('paused_count', 0)}")
        if report.capital_packet_summary:
            cps = report.capital_packet_summary
            lines.append(
                "  "
                f"Capital packet {cps.get('generated_at') or '?'} | "
                f"add {len(cps.get('would_add', []))} | remove {len(cps.get('would_remove', []))} | "
                f"boundary {cps.get('capital_boundary') or '?'}"
            )
        if report.capital_recommendation:
            lines.append(f"  Capital recommendation: {report.capital_recommendation}")
        if report.live_readiness_summary:
            live = report.live_readiness_summary
            strict = live.get("strict_zero_warn") or {}
            automation = live.get("automation_health") or {}
            telemetry = live.get("telemetry_maturity") or {}
            lines.append(
                "  "
                f"Live readiness green={bool(strict.get('green'))} "
                f"blockers={len(strict.get('blockers', []))} "
                f"automation={automation.get('overall')} "
                f"telemetry_days={telemetry.get('n_unique_trading_days')}/{telemetry.get('min_required')}"
            )
        lines.append("")

    # Cap display items per category to keep output scannable
    MAX_DISPLAY = {"decaying": 5, "ready": 5, "unactioned": 3, "paused": 3}
    for cat in CATEGORIES:
        cat_items = [i for i in report.items if i.category == cat]
        if not cat_items:
            continue
        label = CATEGORY_LABELS[cat]
        lines.append(f"[{label}] ({len(cat_items)})")
        limit = MAX_DISPLAY.get(cat, len(cat_items))
        for item in cat_items[:limit]:
            icon = SEVERITY_ICONS.get(item.severity, " ")
            action_hint = f"  → {item.action}" if item.action else ""
            lines.append(f"  {icon} {item.summary}{action_hint}")
        if len(cat_items) > limit:
            lines.append(f"  ... +{len(cat_items) - limit} more")
        lines.append("")

    # Upcoming sessions
    if report.upcoming_sessions:
        lines.append("Upcoming sessions:")
        for s in report.upcoming_sessions[:3]:
            insts = ", ".join(f"{i}:{n}" for i, n in sorted(s.get("instruments", {}).items()))
            inst_str = f" — {insts}" if insts else ""
            lines.append(f"  {s['label']} in {s['hours_away']}h ({s['brisbane_time']} AEST){inst_str}")
        lines.append("")

    if report.fitness_summary:
        lines.append("Strategy fitness:")
        for inst, data in sorted(report.fitness_summary.items()):
            if isinstance(data, dict) and "active_strategies" in data:
                lines.append(f"  {inst}: {data['active_strategies']} active")
            elif isinstance(data, dict):
                parts = [f"{k}={v}" for k, v in data.items() if v]
                lines.append(f"  {inst}: {', '.join(parts)}")
        lines.append("")

    # Single recommendation — the most valuable line
    if report.recommendation:
        lines.append(f">>> {report.recommendation} <<<")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_json(report: PulseReport) -> str:
    """JSON output for /orient skill consumption."""
    data = {
        "generated_at": report.generated_at,
        "cache_hit": report.cache_hit,
        "git_head": report.git_head,
        "git_branch": report.git_branch,
        "system_identity": report.system_identity,
        "handoff": {
            "tool": report.handoff_tool,
            "date": report.handoff_date,
            "summary": report.handoff_summary,
            "next_steps": report.handoff_next_steps,
        },
        "fitness_summary": report.fitness_summary,
        "deployment_summary": report.deployment_summary,
        "survival_summary": report.survival_summary,
        "sr_summary": report.sr_summary,
        "pause_summary": report.pause_summary,
        "execution_summary": report.execution_summary,
        "capital_packet_summary": report.capital_packet_summary,
        "capital_recommendation": report.capital_recommendation,
        "live_readiness_summary": report.live_readiness_summary,
        "work_capsule_summary": report.work_capsule_summary,
        "system_brief_summary": report.system_brief_summary,
        "history_debt_summary": report.history_debt_summary,
        "startup_latency_ms": report.startup_latency_ms,
        "orientation_cost_budget": report.orientation_cost_budget,
        "upcoming_sessions": report.upcoming_sessions,
        "recommendation": report.recommendation,
        "next_actions": [asdict(action) if not isinstance(action, dict) else action for action in report.next_actions],
        "time_since_green": report.time_since_green,
        "session_delta": report.session_delta,
        "counts": {cat: len([i for i in report.items if i.category == cat]) for cat in CATEGORIES},
        "items": [asdict(i) for i in report.items],
    }
    return json.dumps(data, indent=2, default=str)


def format_markdown(report: PulseReport) -> str:
    """Markdown output for PROJECT_PULSE.md (Codex consumption)."""
    lines: list[str] = []
    lines.append("# Project Pulse")
    lines.append("")
    lines.append(f"*Generated: {report.generated_at} | Branch: {report.git_branch} | HEAD: {report.git_head}*")
    lines.append("")

    if report.system_identity:
        identity = report.system_identity
        relations = identity.get("published_relations", {})
        lines.append("## System Identity")
        lines.append(f"- **Canonical repo root**: `{identity.get('canonical_repo_root')}`")
        lines.append(f"- **Canonical DB**: `{identity.get('canonical_db_path')}`")
        if identity.get("db_override_active"):
            lines.append(f"- **Active DB override**: `{identity.get('selected_db_path')}`")
        lines.append(f"- **Active ORB instruments**: {', '.join(identity.get('active_orb_instruments', [])) or 'none'}")
        lines.append(
            f"- **Published shelf relations**: `{relations.get('active', '?')}`, `{relations.get('deployable', '?')}`"
        )
        lines.append(f"- **Authority map**: `{identity.get('authority_map_doc')}`")
        lines.append(f"- **Doctrine docs**: {', '.join(f'`{doc}`' for doc in identity.get('doctrine_docs', []))}")
        lines.append(f"- **Backbone modules**: {', '.join(f'`{mod}`' for mod in identity.get('backbone_modules', []))}")
        lines.append("")

    if report.handoff_summary:
        lines.append("## Last Session")
        lines.append(f"**{report.handoff_tool or '?'}** ({report.handoff_date or '?'}): {report.handoff_summary}")
        lines.append("")

    if report.system_brief_summary:
        brief = report.system_brief_summary
        lines.append("## System Brief")
        lines.append(
            f"- **Route**: `{brief.get('task_id')}` [{brief.get('briefing_level')}] | "
            f"owners={brief.get('canonical_owner_count')} | views={brief.get('required_live_view_count')} | "
            f"blockers={brief.get('blocker_count')} | warnings={brief.get('warning_count')}"
        )
        if report.work_capsule_summary:
            lines.append(f"- **Work capsule**: `{report.work_capsule_summary.get('path')}`")
        if report.startup_latency_ms is not None and report.orientation_cost_budget:
            lines.append(
                f"- **Startup latency**: {report.startup_latency_ms}ms / "
                f"{report.orientation_cost_budget.get('budget_ms')}ms budget"
            )
        lines.append("")

    if report.handoff_next_steps:
        lines.append("## Next Steps")
        for step in report.handoff_next_steps:
            lines.append(f"- {step}")
        lines.append("")

    if report.deployment_summary:
        lines.append("## Live Control")
        ds = report.deployment_summary
        lines.append(
            f"- **Profile**: {ds.get('profile_id')} | deployed={ds.get('deployed_count')} | "
            f"active_validated={ds.get('validated_active_count')} | "
            f"validated_only={len(ds.get('validated_not_deployed', []))}"
        )
        if report.execution_summary:
            es = report.execution_summary
            lines.append(
                f"- **Execution evidence**: {es.get('covered_count', 0)}/{es.get('deployed_count', 0)} lanes | "
                f"missing={es.get('missing_count', 0)} | "
                f"stale={len(es.get('stale_execution_strategy_ids', []))}"
            )
            lines.append(f"- **Execution sources**: `paper_trades` + `live_journal` ({es.get('live_journal_status')})")
            missing = es.get("missing_execution_strategy_ids", [])
            if missing:
                lines.append(f"- **Missing execution IDs**: {', '.join(f'`{x}`' for x in missing[:5])}")
        if report.survival_summary:
            ss = report.survival_summary
            op = ss.get("operational_pass_probability")
            op_str = f"{float(op):.1%}" if isinstance(op, float | int) else "?"
            lines.append(
                f"- **Criterion 11**: {'PASS' if ss.get('gate_ok') else 'BLOCK'} {op_str} | "
                f"as_of={ss.get('as_of_date')} | age={ss.get('report_age_days')}d"
            )
        if report.sr_summary:
            sr = report.sr_summary
            counts = sr.get("counts", {})
            lines.append(
                f"- **Criterion 12 SR**: continue={counts.get('CONTINUE', 0)} "
                f"alarm={counts.get('ALARM', 0)} no_data={counts.get('NO_DATA', 0)} "
                f"| age={sr.get('state_age_days')}d"
            )
            if sr.get("reviewed_watch_count"):
                lines.append(f"- **C12 reviewed WATCH alarms**: {sr.get('reviewed_watch_count')}")
        else:
            lines.append("- **Criterion 12 SR**: BLOCK missing/invalid")
        if report.pause_summary:
            lines.append(f"- **Paused lanes**: {report.pause_summary.get('paused_count', 0)}")
        if report.capital_packet_summary:
            cps = report.capital_packet_summary
            lines.append(
                f"- **Capital packet**: generated={cps.get('generated_at')} | "
                f"add={len(cps.get('would_add', []))} | remove={len(cps.get('would_remove', []))} | "
                f"boundary={cps.get('capital_boundary')}"
            )
        if report.capital_recommendation:
            lines.append(f"- **Capital recommendation**: {report.capital_recommendation}")
        lines.append("")

    for cat in CATEGORIES:
        cat_items = [i for i in report.items if i.category == cat]
        if not cat_items:
            continue
        lines.append(f"## {CATEGORY_LABELS[cat]} ({len(cat_items)})")
        for item in cat_items:
            severity_tag = f"[{item.severity.upper()}]" if item.severity != "low" else ""
            lines.append(f"- {severity_tag} {item.summary}".strip())
        lines.append("")

    if report.fitness_summary:
        lines.append("## Strategy Fitness")
        for inst, data in sorted(report.fitness_summary.items()):
            if isinstance(data, dict) and "active_strategies" in data:
                lines.append(f"- **{inst}**: {data['active_strategies']} active strategies")
            elif isinstance(data, dict):
                parts = [f"{k}={v}" for k, v in data.items() if v]
                lines.append(f"- **{inst}**: {', '.join(parts)}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project pulse — synthesize project state")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text")
    parser.add_argument("--no-cache", action="store_true", help="Force re-run drift+tests")
    parser.add_argument("--deep", action="store_true", help="Full fitness computation (slow)")
    parser.add_argument("--fast", action="store_true", help="Serve cached drift/tests, skip if uncached (~3s)")
    parser.add_argument("--skip-drift", action="store_true", help="Skip drift check")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test suite")
    parser.add_argument("--tool", default="unknown", help="Name of the tool/session writing pulse continuity")
    parser.add_argument("--out", default=None, help="Write output to file instead of stdout")
    parser.add_argument("--root", default=None, help="Override project root")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else PROJECT_ROOT

    report = build_pulse(
        root=root,
        no_cache=args.no_cache,
        deep=args.deep,
        fast=args.fast,
        skip_drift=args.skip_drift,
        skip_tests=args.skip_tests,
        tool_name=args.tool,
    )

    if args.format == "json":
        output = format_json(report)
    elif args.format == "markdown":
        output = format_markdown(report)
    else:
        output = format_text(report)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(output)

    return 1 if report.broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
