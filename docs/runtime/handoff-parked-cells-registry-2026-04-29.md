# Handoff — Parked-Cells Registry Build (2026-04-29)

**Status:** PARTIAL — Tasks 1-3 done, Tasks 4-5 NOT done.
**Branch:** `main`
**HEAD:** `29b4c8ae` (D6 verdict commit; pre-dates this session's working-tree changes)
**Origin:** User asked "do we have an automatic system that keeps track of the parked trades, so we don't lose them randomly over the repo." Answer: partially yes (4 disjoint surfaces), no code-enforced invariant. This session is building the structural fix.

---

## What is in the working tree right now (uncommitted)

```
M  pipeline/check_drift.py                                                        (+~150 lines)
?? docs/runtime/parked-cells.yaml                                                 (new — 5 cells)
?? <user-memory>/feedback_concordance_is_not_equivalence.md                       (new)
?? <user-memory>/feedback_branch_state_first_check.md                             (new)
M  <user-memory>/MEMORY.md                                                        (2 index lines added)
```

`<user-memory>` = `C:\Users\joshd\.claude\projects\C--Users-joshd-canompx3\memory\`

`git diff --stat` and `git status --short` from a fresh shell will confirm exactly the above on `main`.

---

## What was DONE (verified)

### Task 1 — feedback files + MEMORY.md index — COMPLETED

Two new feedback files written:

1. `<user-memory>/feedback_concordance_is_not_equivalence.md` — captures the 2026-04-29 plan-mode lesson that 100% concordance on a built cohort is NOT informational equivalence (the `break_dir` near-miss). Cites `outcome_builder.py:865-870`, `entry_rules.py:157-216`, `build_daily_features.py:288-346` as the source of the build-pipeline tautology.

2. `<user-memory>/feedback_branch_state_first_check.md` — captures the same-session branch-switch mistake; mandates `git rev-parse --abbrev-ref HEAD && git status --short` as first action of any multi-file working session.

`MEMORY.md` index updated with two 1-line entries (under "Statistical rigor" and "Git / workflow" sections respectively, ≤200 chars each per the index contract).

### Task 2 — `docs/runtime/parked-cells.yaml` — COMPLETED

New file. Source-grounded (no invented entries). Contains 5 cells:

| cell_id | status | source pre-reg | source result |
|---|---|---|---|
| `d4-mnq-comex-settle-garch70` | PARK_PENDING_OOS_POWER | `2026-04-28-mnq-comex-settle-garch-pathway-b-v1.yaml` | `2026-04-28-mnq-comex-settle-pathway-b-v1-result.md` |
| `d2-mes-europe-flow-ovnrng80` | PARK_PENDING_OOS_POWER | `2026-04-28-mes-europe-flow-ovn-range-pathway-b-v1.yaml` | `2026-04-28-mes-europe-flow-pathway-b-v1-result.md` |
| `d5-mnq-comex-settle-garch70-conditional-half-size` | KILL | `2026-04-28-mnq-comex-settle-garch-d5-both-sides-pathway-b-v1.yaml` | `2026-04-28-mnq-comex-settle-d5-both-sides-pathway-b-v1-result.md` |
| `d6-mnq-comex-settle-garch70-overlay` | PARK_CONDITIONAL_DEPLOY_RETAINED | `2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1.yaml` | `2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1-result.md` |
| `phase-d-d0-v2-garch-clean-rederivation` | PARK_ABSOLUTE_FLOOR_FAIL | `2026-04-28-phase-d-d0-v2-garch-clean-rederivation.yaml` | `2026-04-28-phase-d-d0-v2-garch-backtest.md` |

Schema includes `events: []` (append-only state-transition log, currently empty) and a header block defining canonical status vocabulary. Each cell has `cell_id, title, status, amendment, instrument, session, orb_minutes, rr_target, entry_model, confirm_bars, direction, base_filter, gate_predicate, pre_reg, result, decision_ledger_slug, predecessor/successor_cell_id, notes`.

Verdict tokens were extracted from each result file's `## Decision rule outcome` block (D2, D4, D5, D6) or `## Verdict` block (D-0 v2). Each token verbatim-matches the source file.

### Task 3 — drift check `check_parked_cells_registry_completeness` — COMPLETED

Added to `pipeline/check_drift.py` immediately before `check_stage_file_landed_drift` (around line 6637). Registered in CHECKS as **Check 124, BLOCKING (not advisory)**.

The check enforces three invariants:

1. **Surfaced result files have matching registry entries.** Globs `docs/audit/results/*pathway-b*-result.md` and `docs/audit/results/*-d0-v2-*backtest.md`. Any file containing a verdict token from `{PARK_PENDING_OOS_POWER, PARK_CONDITIONAL_DEPLOY_RETAINED, PARK_ABSOLUTE_FLOOR_FAIL, KILL, CANDIDATE_READY_FOR_PHASE_2, KEEP_PARKED_INDEFINITELY}` MUST have a `parked-cells.yaml` cell entry whose `result:` field points at it. Failure: "result file X contains a Pathway B verdict but has NO matching cell entry."

2. **Registry paths exist on disk.** Every cell's `pre_reg:` and `result:` paths are checked for existence. Failure: "cell X result=… does NOT exist on disk."

3. **Status token appears in result file.** Every cell's `status:` field must be a substring of its referenced result file's body. Catches the failure mode where the registry says PARK but the result was later moved to KILL via a successor pre-reg without registry update. Failure: "cell X status=… does NOT appear in result file Y (registry-result divergence)."

Authority anchors: `docs/runtime/parked-cells.yaml` schema_version=1; rule cited in commit message.

**Verified passing:** `python pipeline/check_drift.py --fast` returns 96 PASSED + 9 advisory + 0 violations. Check 124 = "Parked-cells registry completeness …" emitted PASSED [OK].

The check uses a permissive flat-block YAML reader (no `pyyaml` dep) — same approach as the existing `check_stage_file_landed_drift`. Reads only the `cell_id`, `status`, `pre_reg`, `result` keys per cell.

---

## What is NOT YET DONE (resume here)

### Task 4 — Pressure-test the drift check with an injected violation — PENDING

**Why mandatory:** `backtesting-methodology.md` § RULE 13 — every new detector must be pressure-tested with a known-bad input before trust. A silent-pass detector is worse than no detector.

**Concrete steps to execute (resume sequence):**

1. **Verify clean baseline:**
   ```
   PYTHONIOENCODING=utf-8 python pipeline/check_drift.py --fast 2>&1 | grep -E "Parked-cells|NO DRIFT|DRIFT DETECTED"
   ```
   Expected: `Check 124: ... PASSED [OK]` and `NO DRIFT DETECTED`.

2. **Inject violation A** (registry-stale failure mode): temporarily delete the D6 cell block from `parked-cells.yaml`. Re-run drift. Expected: Check 124 reports a violation on `2026-04-29-mnq-comex-settle-garch-d6-sizing-overlay-pathway-b-v1-result.md` ("contains a Pathway B verdict but has NO matching cell entry"). Restore the deleted block (use `git restore docs/runtime/parked-cells.yaml`).

3. **Inject violation B** (path-doesn't-exist failure mode): temporarily change one cell's `result:` field to a fake path like `docs/audit/results/does-not-exist.md`. Re-run drift. Expected: Check 124 reports "cell X result=… does NOT exist on disk." Restore.

4. **Inject violation C** (status-text mismatch): temporarily change one cell's `status:` field to `KILL` for the D6 cell (which is actually PARK_CONDITIONAL_DEPLOY_RETAINED). Re-run drift. Expected: Check 124 reports "cell X status=KILL does NOT appear in result file …" Restore.

If any of A/B/C silently passes, the check is broken and needs a fix BEFORE the commit. Do not skip this step.

After all three injections fire correctly, restore the file (a single `git restore docs/runtime/parked-cells.yaml` resets everything if you used `git stash` or careful manual reverts).

### Task 5 — Commit — PENDING

Single `[judgment]` commit covering all changes. Suggested commit message structure already drafted; key points:

- Parked-cells registry created at `docs/runtime/parked-cells.yaml` with 5 source-verified cells (D2, D4, D5, D6, D-0 v2).
- `check_parked_cells_registry_completeness` added as Check 124 BLOCKING in `pipeline/check_drift.py`.
- Pressure-tested with three injected-violation scenarios (A/B/C) — all fired correctly. (Add this only after Task 4 actually runs and confirms.)
- Two `feedback_*.md` memory files capturing 2026-04-29 lessons.
- `MEMORY.md` index updated.
- Decision-ledger entry slug: `parked-cells-registry-landed`.

The pre-commit hook will require a durable closeout surface (it blocked the D6 commit earlier in this session for the same reason). Solution: append a decision-ledger entry to `docs/runtime/decision-ledger.md` before commit. Slug + 1-paragraph entry recording the registry creation, the drift check addition, and the pressure-test results.

Drift gate must pass on the final commit (it does today; pressure-test only modifies state temporarily and restores). All 96 PASSED + 9 advisory + 0 violations is the locked end-state.

---

## How to resume in fresh context

Drop this prompt into the new session:

```
Resume the parked-cells registry build. Read docs/runtime/handoff-parked-cells-registry-2026-04-29.md
in full. Verify branch state (git rev-parse --abbrev-ref HEAD; git status --short) — should be on
main with the uncommitted state listed in the handoff. Tasks 1-3 are done. Execute Task 4 (pressure-
test with injected violations A/B/C, restore between each), then Task 5 (decision-ledger entry +
single judgment commit). No production-code mutation in pipeline/ or trading_app/ beyond the drift
check addition. RISK TIER: high — every step verified before next step.
```

That prompt + this handoff is sufficient. Fresh-context Claude does NOT need to re-derive any
verdicts, re-verify any paths, or re-author any of Tasks 1-3.

---

## Why this work matters (one-paragraph summary so resume agent has the why)

Before this session, parked Phase D research cells (PARK_*, KILL, NOT_OOS_CONFIRMABLE) lived
implicitly across 4 disjoint surfaces: `validated_setups` (DB, only carries `active`/`retired`),
`experimental_strategies` (DB, doesn't track Pathway B confirmatory), `docs/audit/results/*-result.md`
(prose, no enum), `docs/runtime/decision-ledger.md` (append-only, no schema). Convention bound
them; no code-level invariant did. A cell could go missing or drift between surfaces and the only
detection was human review at re-discovery time. The fix is a single durable registry
(`docs/runtime/parked-cells.yaml`) + a BLOCKING drift check enforcing three invariants. This is
the structural answer to the user's question "do we have an automatic system that keeps track of
the parked trades, so we don't lose them randomly over the repo." After this work lands, the
answer is unambiguously YES.

---

## Files in play (paths from project root unless noted)

**Created in this session:**
- `docs/runtime/parked-cells.yaml`
- `docs/runtime/handoff-parked-cells-registry-2026-04-29.md` (this file)
- `<user-memory>/feedback_concordance_is_not_equivalence.md`
- `<user-memory>/feedback_branch_state_first_check.md`

**Modified in this session:**
- `pipeline/check_drift.py` — added function `check_parked_cells_registry_completeness` near line 6637 + registered in CHECKS list near line 7185 (BLOCKING, not advisory)
- `<user-memory>/MEMORY.md` — 2 new index lines (under "Statistical rigor" and "Git / workflow" sections)

**Read but not modified:**
- 5 pre-reg yaml files + 5 result md files (verdict-token extraction; verbatim source-grounding)
- `pipeline/check_drift.py` lines 6502-6634 (`check_e2_lookahead_research_contamination` as template)
- `pipeline/check_drift.py` lines 6637-6740 (`check_stage_file_landed_drift` for permissive-YAML-reader idiom)

---

## Session conduct (so resume agent doesn't repeat my mistakes)

- I made a branch-state mistake earlier (silently switched to `feature/crg-integration-2026-04-29`,
  thought a file disappeared, recovered cleanly via `git stash; git checkout main`). Lesson captured
  in `feedback_branch_state_first_check.md`. Resume agent: ALWAYS run `git rev-parse --abbrev-ref HEAD`
  as first Bash call.
- I almost shipped a wrong fix on `break_dir` informational equivalence based on a 100% concordance
  query that turned out to be a build-pipeline tautology. Caught in plan-mode self-audit. Lesson
  captured in `feedback_concordance_is_not_equivalence.md`. Resume agent: ALWAYS read cohort-build
  source code before treating empirical concordance as structural identity.
- TaskCreate/TaskUpdate were used to track Tasks 1-5; Tasks 1-3 marked completed, 4 in_progress, 5
  pending. Resume agent: read the in-session task list and continue from there.

End of handoff.
