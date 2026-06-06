#!/usr/bin/env python3
"""active_plan.py — read/write the durable plan anchor at docs/runtime/active_plan.md.

ROOT CAUSE #3 (plan/intent drift): a non-trivial plan is approved, work starts,
context compacts or `/clear` fires, and the ORIGINAL deliverable is forgotten —
the thread drifts off-plan. The auto-memory-capture probe proved that SessionStart
`additionalContext` survives `/clear`; this module is the data half of that fix:
a single, durable, machine-readable anchor that a SessionStart hook (Stage 4) can
re-surface every startup/resume/clear so the goal cannot be lost.

STAGE 1 SCOPE: this module + the artifact only. The resurfacing HOOK and the
drift-arrest cue are Stage 4 — out of scope here. Nothing imports this yet, so it
changes no behavior; it is the read/write substrate the later hook will consume.

Format is intentionally simple: YAML-ish frontmatter the hook can parse without a
yaml dependency (the hooks run on bare python), plus a free-form body. We keep the
parser hand-rolled and tolerant so a half-written anchor degrades to "no anchor"
rather than crashing a SessionStart hook (fail-open is mandatory there).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ACTIVE_PLAN_PATH = PROJECT_ROOT / "docs" / "runtime" / "active_plan.md"

# Recognised single-value frontmatter keys. `stages` is a count; `current_stage`
# the 1-based index of the stage in flight; everything else is free text.
_SCALAR_KEYS = ("goal", "plan_file", "current_stage", "stages", "status", "updated")


@dataclass
class ActivePlan:
    """The current approved plan, as anchored for cross-session survival."""

    goal: str = ""
    plan_file: str = ""
    current_stage: int = 0
    stages: int = 0
    status: str = ""
    updated: str = ""
    unfinished: list[str] = field(default_factory=list)
    body: str = ""

    @property
    def exists(self) -> bool:
        return bool(self.goal or self.plan_file)


def _parse(text: str) -> ActivePlan:
    """Tolerant parse of the anchor file. Never raises — returns an empty
    ActivePlan on any malformed input (fail-open for the SessionStart hook)."""
    plan = ActivePlan()
    if not text.strip():
        return plan

    lines = text.splitlines()
    # Optional frontmatter delimited by a leading '---' ... '---'.
    body_start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                body_start = i + 1
                break
            raw = lines[i]
            key, sep, value = raw.partition(":")
            if not sep:
                continue
            key = key.strip().lower()
            value = value.strip()
            if key in _SCALAR_KEYS:
                if key in ("current_stage", "stages"):
                    try:
                        setattr(plan, key, int(value))
                    except ValueError:
                        pass
                else:
                    setattr(plan, key, value)
            elif key == "unfinished":
                # Inline list `[a, b]` or empty; multi-line bullets handled in body.
                inner = value.strip().strip("[]")
                if inner:
                    plan.unfinished = [x.strip() for x in inner.split(",") if x.strip()]

    plan.body = "\n".join(lines[body_start:]).strip()
    return plan


def read_active_plan(path: Path = ACTIVE_PLAN_PATH) -> ActivePlan:
    """Read the anchor. Returns an empty (exists=False) ActivePlan if absent or
    unreadable — callers (hooks) must treat that as 'no active plan', never error."""
    try:
        if not path.is_file():
            return ActivePlan()
        return _parse(path.read_text(encoding="utf-8"))
    except OSError:
        return ActivePlan()


def write_active_plan(plan: ActivePlan, path: Path = ACTIVE_PLAN_PATH) -> None:
    """Persist the anchor. Overwrites atomically-enough for a single-writer
    artifact (one anchor per session). Creates the parent dir if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    front = ["---"]
    front.append(f"goal: {plan.goal}")
    front.append(f"plan_file: {plan.plan_file}")
    front.append(f"current_stage: {plan.current_stage}")
    front.append(f"stages: {plan.stages}")
    front.append(f"status: {plan.status}")
    front.append(f"updated: {plan.updated}")
    if plan.unfinished:
        joined = ", ".join(plan.unfinished)
        front.append(f"unfinished: [{joined}]")
    front.append("---")
    content = "\n".join(front) + "\n\n" + (plan.body.strip() + "\n" if plan.body.strip() else "")
    path.write_text(content, encoding="utf-8")


def summary_line(plan: ActivePlan) -> str:
    """One-line resurfacing string for a SessionStart cue (Stage 4 consumes this).

    Empty string when there is no active plan — the hook emits nothing then.
    """
    if not plan.exists:
        return ""
    parts = [f"ACTIVE PLAN: {plan.goal}" if plan.goal else "ACTIVE PLAN"]
    if plan.stages:
        parts.append(f"stage {plan.current_stage}/{plan.stages}")
    if plan.status:
        parts.append(plan.status)
    if plan.unfinished:
        parts.append("unfinished: " + "; ".join(plan.unfinished))
    return " — ".join(parts)
