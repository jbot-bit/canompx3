#!/usr/bin/env python3
"""Canonical operational-churn path predicate — the intended SINGLE source.

An *operational-churn* path is one whose dirtiness in a worktree is NOT
work-at-risk: it is rewritten constantly by the running bot, the session hooks,
or generated tooling. A worktree dirty ONLY on churn paths has no human work to
lose — for fleet-state / reaper purposes it is effectively clean.

WHY THIS MODULE EXISTS — the re-encoded-list class (institutional-rigor §4):
`fleet_state.py` originally inlined its own `_CHURN_PATHS` tuple. That is the
exact "parallel list that silently drifts" anti-pattern the rule forbids. There
is, today, NO single canonical churn list to import — the repo carries several
*overlapping but semantically distinct* path lists:

  - `scripts/run_live_session.py` `_DRIFT_IGNORE_SUFFIXES = ("live_journal.db",
    "HANDOFF.md")` — the TRUEST sibling: a live-drift-gate churn-ignore list.
  - `scripts/infra/check_root_hygiene.py` `ALLOWED_FILES` — a root-clutter
    allowlist (different question: "may this file sit at repo root?").
  - `scripts/tools/check_referenced_paths.py` known-root prefixes — path-resolution
    roots (different question: "is this a real repo path?").
  - `scripts/tools/checkpoint_guard.py` `DURABLE_ROOTS` — artifacts to PRESERVE
    (the inverse concern, not churn-to-ignore).

This module is canonical-BY-DESIGN: `fleet_state.py` imports it instead of
re-encoding. Adoption by the genuine sibling (`run_live_session.py`'s
`_DRIFT_IGNORE_SUFFIXES`) is tracked as explicit FOLLOW-UP DEBT in
`docs/runtime/active_plan.md` — NOT silently skipped. It is a separate, larger
stage (one copy lives in a capital path), so it is deferred deliberately, not by
oversight. The next reader should know the canonicalization is in progress by
design.

READ-ONLY pure helper. No git calls, no IO. The match is PATH-SEGMENT-aware
(exact path OR a trailing `/<churn>` segment) — NOT a bare substring. A bare
substring falsely matches a real source path that merely CONTAINS a churn name,
e.g. `tests/test_live_journal.db_helpers.py` would match `live_journal.db` and a
genuine uncommitted test edit would be silently dropped from the work-at-risk
count (adversarial-audit finding, 2026-06-06). Segment matching keeps the true
cases (`live_journal.db`, `docs/runtime/active_plan.md`, a nested
`x/live_journal.db`) while rejecting the contains-but-isn't case.
"""

from __future__ import annotations

# The canonical operational-churn paths. A tree dirty ONLY on these carries no
# work-at-risk. Mirrors the live-drift-gate exclusion intent (fb76e8cf) and the
# `feedback_stale_cleanup_plan` lesson; superset of run_live_session.py's
# 2-path `_DRIFT_IGNORE_SUFFIXES` (which migrates onto this module as tracked
# follow-up debt). Frozen tuple — consumers import the predicate, never a mutable
# copy of the list, so the definition cannot fork.
OPERATIONAL_CHURN_PATHS: tuple[str, ...] = (
    "HANDOFF.md",
    "REPO_MAP.md",
    "live_journal.db",
    "bot_state.json",
    "planned_launch.json",
    "docs/runtime/active_plan.md",
)


def is_churn_path(path: str) -> bool:
    """True iff ``path`` is an operational-churn path (no work-at-risk).

    PATH-SEGMENT match against :data:`OPERATIONAL_CHURN_PATHS`: the normalized
    path must EQUAL a churn entry or END WITH ``/<churn-entry>``. This rejects a
    real source path that merely contains a churn name as a substring (e.g.
    ``tests/test_live_journal.db_helpers.py`` is NOT churn; ``live_journal.db``
    and ``logs/live_journal.db`` ARE). An empty/blank path is never churn.
    """
    if not path or not path.strip():
        return False
    norm = path.strip().replace("\\", "/")
    return any(norm == churn or norm.endswith("/" + churn) for churn in OPERATIONAL_CHURN_PATHS)
