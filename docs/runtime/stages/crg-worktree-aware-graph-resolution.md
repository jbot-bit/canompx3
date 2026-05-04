---
task: crg-worktree-aware-graph-resolution
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift_crg_helpers.py
  - .githooks/pre-commit
  - tests/test_pipeline/test_check_drift_crg.py
blast_radius: pipeline/check_drift_crg_helpers.py adds _resolve_repo_root() honoring CRG_REPO_ROOT env + canonical-sibling fallback when local graph has <50 files; affects D1-D5 drift checks (Check 128-132) when run from a worktree, canonical runs unchanged. .githooks/pre-commit exports CRG_REPO_ROOT for worktree commits. tests/test_pipeline/test_check_drift_crg.py adds 4 targeted tests for env override, empty-graph fallback, file-count probe, candidate sibling resolution.
agent: claude
updated: 2026-04-30
---

# CRG worktree-aware graph resolution

**Status:** IMPLEMENTATION (1/1)
**Date:** 2026-04-30
**Worktree:** `canompx3-data-drift-restamp` on `session/joshd-data-drift-restamp`

## Why this stage exists

The CRG drift checks (D1–D5, Check 128–132) currently run against per-worktree
graph DBs. Because each worktree's pre-commit hook only does an *incremental*
`code-review-graph update --base origin/main`, worktrees end up with
graphs containing only the 4–10 files the recent commit touched. The drift
checks `pipeline/check_drift_crg_helpers.py:crg_is_available()` returns True
(because `.code-review-graph/graph.db` exists), then queries return findings
restricted to that tiny graph — looks like real CRG coverage but is actually
4-of-1052 files.

Meanwhile the canonical repo at `C:/Users/joshd/canompx3` has a full
13,912-node, 1,052-file graph built fresh on 2026-04-30. The right answer
is to point worktrees at that canonical graph via CRG's existing
`CRG_REPO_ROOT` env-var pattern (see `code_review_graph.tools._common.find_project_root`).

## Approach

1. Replace the hardcoded `_PROJECT_ROOT = Path(__file__).resolve().parent.parent`
   in `check_drift_crg_helpers.py` with a function `_get_project_root()` that
   delegates to `code_review_graph.tools._common.find_project_root()`. This
   honors `CRG_REPO_ROOT` env var and the standard CRG resolution order.
2. Add an empty-graph fallback: when the resolved project's graph has fewer
   than `_MIN_REAL_GRAPH_FILES` files (sentinel = 50), prefer the canonical
   sibling graph if one exists with more files. Avoids accidentally querying
   stale fragments.
3. Update `.githooks/pre-commit` to export `CRG_REPO_ROOT` to the canonical
   sibling when the worktree itself is not the canonical project. Same
   sibling-detection logic as the dev-deps probe (`PARENT_DIR/CANONICAL_BASENAME`).
4. Add 2 targeted tests in `test_check_drift_crg.py`: env-override path and
   empty-graph fallback path.

## Acceptance criteria

- `python pipeline/check_drift.py` from this worktree runs CRG D1–D5 against
  the canonical 1052-file graph (verified by D4 finding multiple 200+ line
  functions across the codebase, not just files I edited)
- `code-review-graph update` invoked from worktree pre-commit writes to the
  canonical graph (verified by `code-review-graph status` showing 1052+ files
  after the commit)
- Tests pass (env-override + empty-graph fallback)
- Drift check `python pipeline/check_drift.py` exit 0
