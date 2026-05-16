# EHR PASS 2 — Handoff for Next Session

**Last updated:** 2026-05-17, end of Stage 1.
**Worktree:** `C:/Users/joshd/canompx3/.worktrees/ehr-validation-mode`
**Branch:** `session/joshd-ehr-validation-mode`
**Head commit:** `d910d6f7 ehr(stage-1): add EARLY_HOLDOUT_BOUNDARY constant + EHR helpers`

## Where we are in the 8-stage plan

| # | Stage | Status |
|---|-------|--------|
| 1 | Holdout constants + EHR helpers | **DONE** — committed `d910d6f7`, 30 tests green, drift baseline 296 unchanged |
| 2 | `validated_setups` schema additive (5 columns) | **PENDING** — stage file written (`docs/runtime/stages/ehr-stage-2-validated-setups-schema.md`), implementation not started |
| 3 | Validator hard-fail on EHR verdict | pending |
| 4 | Discovery `--validation-mode` flag + anti-rescue | pending |
| 5 | Allocator EHR isolation + trade_book RESEARCH_QUEUE | pending |
| 6 | 7 drift checks + kill-threshold byte-equality + injection probes | pending |
| 7 | Pre-reg + result-MD templates | pending |
| 8 | MNQ EHR acceptance batch + GREEN/RED verdict | pending |

## Stage 1 — what shipped (`d910d6f7`)

**Files changed (3, +286 lines):**
- `trading_app/holdout_policy.py` — additive: `EARLY_HOLDOUT_BOUNDARY = date(2025, 1, 1)`,
  `EHR_MODE_LABEL = "EARLY_HOLDOUT_REDISCOVERY"`, `STANDARD_MODE_LABEL = "STANDARD"`,
  `is_ehr_mode(mode: str | None) -> bool` (strict equality), `enforce_early_holdout_date()`
  (raises ValueError on post-boundary, no override token by design). `__all__` extended.
- `tests/test_trading_app/test_holdout_policy_ehr.py` — 7 tests covering: boundary
  constant, Mode A regression guard (`test_mode_a_sacred_unchanged`), boundary ordering
  (`test_ehr_boundary_strictly_before_mode_a`), reject-post-boundary, accept-pre-or-on,
  predicate strictness (8 negative cases), label distinctness.
- `docs/runtime/stages/ehr-stage-1-holdout-constants.md` — stage file.

**Verification at commit time:**
- `pytest tests/test_trading_app/test_holdout_policy.py tests/test_trading_app/test_holdout_policy_ehr.py -v` — 30/30 pass
- `python pipeline/check_drift.py` — 296 violations (identical to baseline on `main`)
- `git grep` confirmed no production caller imports the new symbols yet — they go live
  in Stages 3/4/5/6

**Plan invariants enforced this stage:** #1 (Mode A unmodified), #5 (hard-fail at code level).

## Stage 2 — ready to resume

Stage file already written at `docs/runtime/stages/ehr-stage-2-validated-setups-schema.md`.
The plan in it:
- Insert additive migration block in `trading_app/db_manager.py` after line 624
  (the `validation_pathway`/`c8_oos_status` block, the most recent additive migration).
- 5 new columns on `validated_setups`:
  - `validation_mode TEXT DEFAULT 'STANDARD'`
  - `pseudo_oos_window_start DATE`
  - `pseudo_oos_window_end DATE`
  - `verdict_ceiling TEXT`
  - `cumulative_search_count INTEGER`
- Import `EHR_MODE_LABEL` + `STANDARD_MODE_LABEL` from `trading_app.holdout_policy` and add
  a runtime assert that the DEFAULT literal `'STANDARD'` equals `STANDARD_MODE_LABEL`
  (defense in depth — canonical-source delegation per `integrity-guardian.md` § 2).
- New test file: `tests/test_trading_app/test_db_manager_ehr_schema.py` with 4 tests:
  schema-introspection, default-value, nullability, round-trip.

**Anchor code site:** db_manager.py line 624 (end of validation_pathway/c8_oos_status
migration block). New code drops in immediately after that block.

**Acceptance criteria:** 9 items in the Stage 2 stage file. The non-obvious one is
acceptance #8: SQL DEFAULT cannot reference a Python constant, so we must use the literal
string `'STANDARD'` in the SQL — but add a Python-side `assert STANDARD_MODE_LABEL == "STANDARD"`
adjacent so any drift surfaces loudly.

## Self-review gotchas observed in Stage 1 (lessons for Stage 2)

1. **PowerShell `@'...'@` here-string syntax leaks into bash.** Stage 1's first commit had
   a literal `@` prefix on the subject line. Amend was required. For Stage 2, write the
   commit message to `C:/Users/joshd/.git/worktrees/ehr-validation-mode/COMMIT_MSG_*.txt`
   (the worktree's gitdir, NOT `.worktrees/ehr-validation-mode/.git/` which is a file
   pointer not a directory) and `git commit -F <file>`.

2. **Self-review caught a real test bug.** Initial `test_enforce_early_holdout_rejects_post_boundary`
   asserted the boundary date itself raises — but the helper contract (parallel to
   `enforce_holdout_date`) is that callers use the returned date as the EXCLUSIVE upper
   bound for `WHERE trading_day < <returned>`, so the boundary itself must be a permitted
   return value. Fixed before commit. Stage 2 round-trip tests should similarly check the
   SQL DEFAULT contract, not assert away from it.

3. **Pyright false positives expected.** The IDE indexer roots at `C:/Users/joshd/canompx3/`
   (main), not the worktree. New symbols added in the worktree appear "unknown" in main's
   index but are real in the worktree. Run actual tests, don't trust Pyright in worktree
   context.

## Plan items still flagged as needing attention later

1. **MCP allow-list (Stage 7 territory).** Plan Stage 7 adds `validation_mode` to
   `_ALLOWED_PARAMS` in `trading_app/mcp_server.py` and `trading_app/ai/sql_adapter.py`.
   Until then, the new column is invisible to MCP queries — acceptable interim state.

2. **`docs/institutional/pre_registered_criteria.md` Amendment 2.9** — plan says deferred
   until after Stage 8 GREEN. Don't write it preemptively.

3. **Drift count baseline.** 296 violations on main as of 2026-05-17. Every stage must
   keep watching this — my additions cannot quietly raise it. Stage 6 adds 7 NEW drift
   checks, which will increase the check count but should not increase the violation
   count if implementation is correct.

## Quick resume command sequence

```bash
# Confirm we are in the right worktree on the right branch
git -C C:/Users/joshd/canompx3/.worktrees/ehr-validation-mode rev-parse --abbrev-ref HEAD
# Expect: session/joshd-ehr-validation-mode

# Confirm Stage 1 is HEAD
git -C C:/Users/joshd/canompx3/.worktrees/ehr-validation-mode log -1 --format="%h %s"
# Expect: d910d6f7 ehr(stage-1): add EARLY_HOLDOUT_BOUNDARY constant + EHR helpers

# Read the Stage 2 stage file
cat C:/Users/joshd/canompx3/.worktrees/ehr-validation-mode/docs/runtime/stages/ehr-stage-2-validated-setups-schema.md
```

Then proceed with Stage 2 implementation per the stage file's scope_lock and 9 acceptance criteria.
