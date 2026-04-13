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
from datetime import UTC, datetime
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

from pipeline.paths import GOLD_DB_PATH
from trading_app.validated_shelf import deployable_validated_relation

# staleness_engine lives in scripts/tools/ (same dir as this file)
_SCRIPTS_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_TOOLS_DIR)

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


_UPDATE_HEADER_RE = re.compile(r"^## Update \((\d{4}-\d{2}-\d{2})(?:[^)]*?\s+[—-]\s+(.*?))?\)$")


def _strip_handoff_markup(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = text.replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


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
    if not handoff_path.exists():
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
        return [
            PulseItem(
                category="broken",
                severity="high",
                source="tests",
                summary="Test suite timed out (>2m)",
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


def collect_deployment_state(db_path: Path) -> tuple[dict | None, list[PulseItem]]:
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
                    category="decaying",
                    severity="low",
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
                    category="decaying",
                    severity="low" if is_stale else "medium",
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
            "policy": {
                "allowed": decision.allowed,
                "warnings": [issue.message for issue in decision.warnings],
                "controls": decision.applicable_controls,
            },
        }
        for issue in decision.warnings:
            if issue.code == "wrong_interpreter":
                items.append(
                    PulseItem(
                        category="decaying",
                        severity="medium",
                        source="system_identity",
                        summary="Interpreter mismatch for repo-managed context",
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


def collect_worktrees(canonical: Path) -> list[PulseItem]:
    """Detect open managed worktrees. Summarizes when >3 to reduce noise."""
    items: list[PulseItem] = []
    wt_base = canonical / ".worktrees"

    if not wt_base.exists():
        return items

    worktrees: list[dict] = []
    try:
        meta_files = list(wt_base.rglob(".canompx3-worktree.json"))
    except OSError:
        # rglob can fail on Windows with broken symlinks/junctions in worktrees
        return items
    for meta_file in meta_files:
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            worktrees.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    if not worktrees:
        return items

    # Summarize when many worktrees to avoid noise
    if len(worktrees) > 3:
        tools = {}
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


def collect_action_queue(canonical: Path) -> list[PulseItem]:
    """Parse ACTION QUEUE from Claude auto-memory MEMORY.md."""
    items: list[PulseItem] = []
    memory_path = _find_memory_md(canonical)
    if memory_path is None:
        return items

    try:
        text = memory_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return items

    in_queue = False
    for line in text.splitlines():
        if "## ACTION QUEUE" in line:
            in_queue = True
            continue
        if in_queue:
            if line.startswith("## "):
                break
            m = re.match(r"^\d+\.\s+(.+)", line.strip())
            if m:
                content = m.group(1)
                if "~~" in content:
                    continue
                clean = re.sub(r"\*\*(.+?)\*\*", r"\1", content).strip()
                short = re.split(r"\s*—\s*|\.\s", clean, maxsplit=1)[0].strip()
                if len(short) > 80:
                    short = short[:77] + "..."
                items.append(
                    PulseItem(
                        category="ready",
                        severity="low",
                        source="action_queue",
                        summary=short,
                    )
                )

    return items


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
    wt_base = canonical / ".worktrees"
    if not wt_base.exists():
        return items

    # Collect modified files per worktree branch
    worktree_files: dict[str, set[str]] = {}
    try:
        _meta_files = list(wt_base.rglob(".canompx3-worktree.json"))
    except OSError:
        return items
    for meta_file in _meta_files:
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            branch = data.get("branch", "")
            name = data.get("name", "?")
            if not branch:
                continue
            r = _run_git(canonical, "diff", "--name-only", f"main...{branch}")
            if r and r.returncode == 0:
                files = {f.strip() for f in r.stdout.splitlines() if f.strip()}
                if files:
                    worktree_files[name] = files
        except (json.JSONDecodeError, OSError):
            continue

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
    survival_summary, sr_summary, pause_summary, lifecycle_items = collect_lifecycle_control(db_path)
    handoff_context, handoff_items = collect_handoff(root)
    worktree_items = collect_worktrees(canonical)
    claim_items = collect_session_claims(root)
    conflict_items = collect_worktree_conflicts(canonical)
    git_items = collect_git_state(root)
    action_items = collect_action_queue(canonical)
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
        + claim_items
        + conflict_items
        + handoff_items
        + worktree_items
        + git_items
        + action_items
        + ralph_items
    )

    # Attach skill invocation suggestions
    _attach_skill_suggestions(all_items)

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
    )

    # Compute single recommendation after full report is assembled
    report.recommendation = _compute_recommendation(report)

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


def format_text(report: PulseReport) -> str:
    """Concise terminal output for humans."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("PROJECT PULSE")
    lines.append("=" * 60)
    meta_parts = [f"Branch: {report.git_branch}", f"HEAD: {report.git_head}"]
    if report.cache_hit:
        meta_parts.append("(cached)")
    if report.time_since_green:
        meta_parts.append(f"Green: {report.time_since_green}")
    lines.append("  ".join(meta_parts))
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
        doctrine = ", ".join(identity.get("doctrine_docs", []))
        backbone = ", ".join(identity.get("backbone_modules", []))
        lines.append("System identity:")
        lines.append(f"  Root: {identity.get('canonical_repo_root')}")
        lines.append(f"  Canonical DB: {identity.get('canonical_db_path')}")
        if identity.get("db_override_active"):
            lines.append(f"  Active DB override: {identity.get('selected_db_path')}")
        lines.append(f"  Active ORB instruments: {', '.join(identity.get('active_orb_instruments', [])) or 'none'}")
        lines.append(f"  Shelf contracts: {relations.get('active', '?')}, {relations.get('deployable', '?')}")
        lines.append(f"  Authority map: {identity.get('authority_map_doc')}")
        lines.append(f"  Doctrine: {doctrine}")
        lines.append(f"  Backbone: {backbone}")
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

    if report.deployment_summary:
        ds = report.deployment_summary
        lines.append("Live control:")
        lines.append(
            "  "
            f"Profile {ds.get('profile_id')} | deployed {ds.get('deployed_count', '?')} | "
            f"active validated {ds.get('validated_active_count', '?')} | "
            f"validated-only {len(ds.get('validated_not_deployed', []))}"
        )
        if report.survival_summary:
            ss = report.survival_summary
            op = ss.get("operational_pass_probability")
            op_str = f"{float(op):.1%}" if isinstance(op, (float, int)) else "?"
            lines.append(
                "  "
                f"C11 {('PASS' if ss.get('gate_ok') else 'BLOCK')} {op_str} | "
                f"as_of {ss.get('as_of_date') or '?'} | age {ss.get('report_age_days') if ss.get('report_age_days') is not None else '?'}d"
            )
        if report.sr_summary:
            sr = report.sr_summary
            counts = sr.get("counts", {})
            streams = sr.get("stream_counts", {})
            lines.append(
                "  "
                f"C12 SR continue={counts.get('CONTINUE', 0)} alarm={counts.get('ALARM', 0)} "
                f"no_data={counts.get('NO_DATA', 0)} | age {sr.get('state_age_days') if sr.get('state_age_days') is not None else '?'}d"
            )
            if sr.get("reviewed_watch_count"):
                lines.append(f"  C12 reviewed WATCH alarms: {sr.get('reviewed_watch_count')}")
            if streams:
                stream_parts = [f"{k}:{v}" for k, v in sorted(streams.items())]
                lines.append(f"  SR streams: {', '.join(stream_parts)}")
        if report.pause_summary:
            ps = report.pause_summary
            lines.append(f"  Paused lanes: {ps.get('paused_count', 0)}")
        lines.append("")

    # Cap display items per category to keep output scannable
    MAX_DISPLAY = {"ready": 5, "unactioned": 5, "paused": 5}
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
        "work_capsule_summary": report.work_capsule_summary,
        "system_brief_summary": report.system_brief_summary,
        "history_debt_summary": report.history_debt_summary,
        "startup_latency_ms": report.startup_latency_ms,
        "orientation_cost_budget": report.orientation_cost_budget,
        "upcoming_sessions": report.upcoming_sessions,
        "recommendation": report.recommendation,
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
        if report.survival_summary:
            ss = report.survival_summary
            op = ss.get("operational_pass_probability")
            op_str = f"{float(op):.1%}" if isinstance(op, (float, int)) else "?"
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
        if report.pause_summary:
            lines.append(f"- **Paused lanes**: {report.pause_summary.get('paused_count', 0)}")
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
