---
mode: IMPLEMENTATION
slug: fix-check-37-honor-duckdb-path
stage: 1/1
started: 2026-04-19
task: "Drift check 37 must honor DUCKDB_PATH env var (canonical pipeline.paths.GOLD_DB_PATH delegation)"
---

# Stage 1 — Check 37 honors DUCKDB_PATH; pip install anthropic for check 16

## Context
Running `python pipeline/check_drift.py` from a git worktree (e.g., f5)
fails check 37 ("Canonical DB not found at project root") because the
worktree directory has no local `gold.db`. The canonical DB lives at
`canompx3/gold.db` and is reachable via `DUCKDB_PATH=...` env var per
CLAUDE.md "DB path | pipeline.paths.GOLD_DB_PATH" rule.

But check 37 hardcodes `PROJECT_ROOT / "gold.db"` rather than delegating
to `pipeline.paths.GOLD_DB_PATH` — violation of integrity-guardian.md
rule 2 (canonical-source delegation). The existing helper
`pipeline.check_drift._get_db_path()` already does the right delegation;
check 37 just doesn't call it.

Also check 16 fails because `anthropic` SDK isn't installed in the
active venv. That's `pip install anthropic`, no code change.

## Approach
- TDD: RED test (DUCKDB_PATH-honoring scenario) → GREEN (delegate to
  `_get_db_path()`) → re-verify existing tests still pass.
- Update existing TestStaleScratchDb tests to patch via the
  GOLD_DB_PATH_FOR_CHECKS hook (the already-provided test override),
  not via PROJECT_ROOT.

## Scope Lock
- pipeline/check_drift.py
- tests/test_pipeline/test_check_drift_ws2.py

Notes:
- check_drift.py — change check 37 to use _get_db_path(); +1 doc line
- test_check_drift_ws2.py — add DUCKDB_PATH-honoring test; switch
  existing PROJECT_ROOT-patching tests to GOLD_DB_PATH_FOR_CHECKS hook

## Acceptance criteria
1. New RED test fails on current code (proves the bug exists).
2. GREEN: minimum change in check_drift.py makes the new test pass.
3. Existing TestStaleScratchDb tests still pass after refactor.
4. Full pytest suite passes.
5. `python pipeline/check_drift.py` (with DUCKDB_PATH set on f5) reports
   check 37 PASSED.
6. After `pip install anthropic`, check 16 passes too.
7. Final state on f5: drift = 0 violations across all checks (modulo
   any pre-existing main-branch failures, which would be separate).

## Blast Radius
- check_drift.py is core production code but the change is a 1-line
  delegation to the existing `_get_db_path()` helper.
- check 37's purpose unchanged (still detects missing canonical DB).
- All non-worktree callers (main repo, CI) unaffected — GOLD_DB_PATH
  resolves to PROJECT_ROOT/gold.db when no DUCKDB_PATH is set.
- Worktree callers benefit (drift now passes when DUCKDB_PATH is set).
