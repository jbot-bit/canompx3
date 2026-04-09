#!/usr/bin/env python3
"""Project pulse — synthesize project state from 9 signal sources.

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
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    next_steps: list[str] = []
    blocker_keywords = {"failure", "broken", "missing", "error", "cannot", "blocked"}

    section: str | None = None
    for line in text.splitlines():
        # Section headers
        if line.startswith("## Last Session"):
            section = "metadata"
            continue
        if re.match(r"^## Next Steps", line):
            section = "next_steps"
            continue
        if line.startswith("## Blockers") or line.startswith("## Blockers / Warnings"):
            section = "blockers"
            continue
        if line.startswith("## "):
            section = None
            continue

        # Content within sections
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
        elif section == "blockers":
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

    context["next_steps"] = next_steps
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

        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            for inst in ACTIVE_ORB_INSTRUMENTS:
                status = staleness_engine(con, inst)
                stale = status.get("stale_steps", [])
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
    """Fast proxy: count active validated strategies per instrument."""
    summary: dict = {}
    items: list[PulseItem] = []
    if not db_path.exists():
        return summary, items

    try:
        import duckdb

        from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute(
                "SELECT instrument, COUNT(*) as n FROM validated_setups "
                "WHERE LOWER(status) = 'active' GROUP BY instrument ORDER BY instrument"
            ).fetchall()
            for inst, n in rows:
                summary[inst] = {"active_strategies": n}
            for inst in ACTIVE_ORB_INSTRUMENTS:
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
            validated_rows = con.execute(
                """
                SELECT strategy_id
                FROM validated_setups
                WHERE LOWER(status) = 'active'
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
                    summary=(
                        f"{profile_id}: {len(validated_not_deployed)} active validated lane(s) not deployed-live"
                    ),
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


def collect_survival_state() -> tuple[dict | None, list[PulseItem]]:
    """Read Criterion 11 gate truth from the canonical helper + persisted report."""
    summary: dict | None = None
    items: list[PulseItem] = []

    try:
        from trading_app.account_survival import check_survival_report_gate, get_survival_report_path
        from trading_app.prop_profiles import resolve_profile_id

        profile_id = resolve_profile_id()
        gate_ok, gate_msg = check_survival_report_gate(profile_id=profile_id)
        report_path = get_survival_report_path(profile_id)

        report_data = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        summary_block = report_data.get("summary", {})
        generated_at = summary_block.get("generated_at_utc")
        report_age_days = None
        if generated_at:
            try:
                dt = datetime.fromisoformat(str(generated_at))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                report_age_days = max(0, (datetime.now(UTC) - dt).days)
            except ValueError:
                report_age_days = None

        summary = {
            "profile_id": profile_id,
            "gate_ok": gate_ok,
            "gate_msg": gate_msg,
            "as_of_date": summary_block.get("as_of_date"),
            "generated_at_utc": generated_at,
            "report_age_days": report_age_days,
            "operational_pass_probability": summary_block.get("operational_pass_probability"),
            "n_paths": summary_block.get("n_paths"),
            "horizon_days": summary_block.get("horizon_days"),
            "gate_pass": summary_block.get("gate_pass"),
        }

        if not gate_ok:
            items.append(
                PulseItem(
                    category="broken",
                    severity="high",
                    source="criterion11",
                    summary=gate_msg,
                )
            )
        elif report_age_days is not None and report_age_days >= 14:
            items.append(
                PulseItem(
                    category="decaying",
                    severity="low",
                    source="criterion11",
                    summary=f"Criterion 11 report is {report_age_days}d old — refresh before it hard-blocks",
                )
            )
    except Exception as e:
        items.append(
            PulseItem(
                category="broken",
                severity="medium",
                source="criterion11",
                summary=f"Criterion 11 state error: {type(e).__name__}: {e}",
            )
        )

    return summary, items


def collect_sr_state() -> tuple[dict | None, list[PulseItem]]:
    """Read Criterion 12 SR monitor state from the persisted state file."""
    summary: dict | None = None
    items: list[PulseItem] = []

    try:
        from trading_app.prop_profiles import resolve_profile_id

        profile_id = resolve_profile_id()
        state_path = PROJECT_ROOT / "data" / "state" / "sr_state.json"
        if not state_path.exists():
            items.append(
                PulseItem(
                    category="decaying",
                    severity="low",
                    source="sr_monitor",
                    summary="Criterion 12 SR state missing — run `python -m trading_app.sr_monitor`",
                )
            )
            return None, items

        data = json.loads(state_path.read_text(encoding="utf-8"))
        state_date = data.get("date")
        state_profile_id = data.get("profile_id")
        results = data.get("results", [])

        counts: dict[str, int] = {}
        stream_counts: dict[str, int] = {}
        for row in results:
            status = str(row.get("status", "UNKNOWN"))
            stream = str(row.get("stream_source", "unknown"))
            counts[status] = counts.get(status, 0) + 1
            stream_counts[stream] = stream_counts.get(stream, 0) + 1

        state_age_days = None
        if state_date:
            try:
                state_age_days = max(0, (date.today() - date.fromisoformat(str(state_date))).days)
            except ValueError:
                state_age_days = None

        summary = {
            "profile_id": state_profile_id,
            "state_date": state_date,
            "state_age_days": state_age_days,
            "counts": counts,
            "stream_counts": stream_counts,
            "apply_pauses": bool(data.get("apply_pauses")),
        }

        if state_profile_id != profile_id:
            items.append(
                PulseItem(
                    category="broken",
                    severity="medium",
                    source="sr_monitor",
                    summary=f"SR state is for {state_profile_id}, expected {profile_id}",
                )
            )
            return summary, items

        alarms = counts.get("ALARM", 0)
        if alarms > 0:
            items.append(
                PulseItem(
                    category="decaying",
                    severity="high",
                    source="sr_monitor",
                    summary=f"Criterion 12 SR has {alarms} ALARM lane(s)",
                    action="python -m trading_app.sr_monitor --apply-pauses",
                )
            )
        elif state_age_days is not None and state_age_days >= 2:
            items.append(
                PulseItem(
                    category="decaying",
                    severity="low",
                    source="sr_monitor",
                    summary=f"Criterion 12 SR state is {state_age_days}d old",
                )
            )
    except Exception as e:
        items.append(
            PulseItem(
                category="broken",
                severity="medium",
                source="sr_monitor",
                summary=f"SR state error: {type(e).__name__}: {e}",
            )
        )

    return summary, items


def collect_pause_state() -> tuple[dict | None, list[PulseItem]]:
    """Read current lane pause overrides."""
    summary: dict | None = None
    items: list[PulseItem] = []

    try:
        from trading_app.lane_ctl import get_paused_strategy_ids
        from trading_app.prop_profiles import resolve_profile_id

        profile_id = resolve_profile_id()
        paused = sorted(get_paused_strategy_ids(profile_id))
        summary = {
            "profile_id": profile_id,
            "paused_count": len(paused),
            "paused_strategy_ids": paused,
        }
        if paused:
            items.append(
                PulseItem(
                    category="paused",
                    severity="low",
                    source="pauses",
                    summary=f"{profile_id}: {len(paused)} lane(s) currently paused",
                    detail="\n".join(paused[:5]),
                )
            )
    except Exception as e:
        items.append(
            PulseItem(
                category="broken",
                severity="low",
                source="pauses",
                summary=f"Pause state error: {type(e).__name__}",
            )
        )

    return summary, items


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
                            for inst in ACTIVE_ORB_INSTRUMENTS:
                                row = con.execute(
                                    "SELECT COUNT(*) FROM validated_setups WHERE instrument = ? AND orb_label = ? AND LOWER(status) = 'active'",
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
    survival_summary, survival_items = collect_survival_state()
    sr_summary, sr_items = collect_sr_state()
    pause_summary, pause_items = collect_pause_state()
    handoff_context, handoff_items = collect_handoff(root)
    worktree_items = collect_worktrees(canonical)
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
        drift_items
        + test_items
        + staleness_items
        + fitness_items
        + deployment_items
        + survival_items
        + sr_items
        + pause_items
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

    if report.handoff_summary:
        lines.append("## Last Session")
        lines.append(f"**{report.handoff_tool or '?'}** ({report.handoff_date or '?'}): {report.handoff_summary}")
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
