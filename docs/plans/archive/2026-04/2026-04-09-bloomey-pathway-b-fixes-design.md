---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Pathway B / Amendment 3.0 Bloomey-Findings Fix Plan

**Date locked:** 2026-04-09
**Status:** DESIGN — awaiting approval before implementation
**Author:** Claude Code session (post-bloomey-review of commits ce450fc → 39b8304)
**Triggering review:** /code-review verdict Grade C with 1 CRITICAL + 4 HIGH findings
**Authority chain:** `docs/institutional/pre_registered_criteria.md` Amendment 3.0 (locked text) overrides any "convenience" relaxations in implementation.

---

## Turn 1 — Orient

### What we are fixing

The Bloomey review of commits `ce450fc` (Amendment 3.0 locked text) → `c8efb20` (Pathway B validator branch) → `ea18c61` (Criterion 8 minimum-OOS pass-through) → `149f9d0` / `39b8304` (test fixes + handoff correction) found four enforcement gaps and one repo-hygiene blocker. The framework documents are sound. The IMPLEMENTATION does not honour what the documents lock down.

The five findings, restated in order of remediation priority:

1. **D-1 (CRITICAL):** Commit `03e9c9c` accidentally added two DuckDB snapshot files totalling roughly 4.89 gigabytes (`gold.db.pre-e2-fix.bak` at 2.46 GB and `gold_snap.db` at 2.43 GB). Both files are still tracked. Pushing to origin will almost certainly fail under GitHub's pack-size and per-blob limits. The repo history is permanently bloated until the blobs are filter-repo'd out.

2. **A-1 (HIGH):** The Pathway B branch in the validator only checks raw p-value and Sharpe direction. It does NOT enforce the Pardo walk-forward-efficiency floor of 0.50 that Pathway A applies on top of the worker's pass flag. Amendment 3.0 condition four says Criterion 6 is mandatory and non-waivable for Pathway B. The implementation silently waives it.

3. **A-2 (HIGH):** Commit `ea18c61` added an N-OOS-less-than-thirty pass-through inside Criterion 8 that returns "no rejection, no marker" and only logs at info level. Amendment 3.0 condition four also explicitly forbids "insufficient OOS data exemptions" for Pathway B. The pass-through fires in the shared pre-flight gate that runs for both pathways, so Pathway B inherits a waiver that Amendment 3.0 just locked out.

4. **A-3 (MEDIUM, promoted):** The Pathway B branch writes the raw bootstrap p-value into the `fdr_adjusted_p` column of `validated_setups`. A downstream auditor reading that column has no way to tell whether they are looking at a real Benjamini-Hochberg-adjusted value or a Pathway B raw value. There is no `validation_pathway` column on `validated_setups` to disambiguate. The audit trail is silently ambiguous.

5. **D-2 (HIGH):** The sixty-line Pathway B branch has zero integration test coverage. The four new tests in `test_hypothesis_loader.py` cover the loader-side `testing_mode` field surface. None exercise `run_validation(testing_mode="individual")`. The gate semantics, the database update chain, and the rejection-and-demote flow are all untested.

### Project context (canonical sources read for this design)

- `trading_app/strategy_validator.py` lines 760-1170 (pre-flight gate stack, Criterion 1, 2, 8, 9 helpers, grandfathering predicate) and lines 1260-2070 (the run-validation main loop, Phase A serial cull, Phase B walk-forward worker dispatch, Phase C database write block, Pathway A FDR branch, the new Pathway B branch, deflated Sharpe informational block, Phase D rejection counters).
- `trading_app/hypothesis_loader.py` (full) — read-side loader, MinBTL canonical enforcement, scope predicate, Mode A consistency check, the new top-level `testing_mode` surface added by Amendment 3.0.
- `trading_app/walkforward.py` lines 270-380 — confirms the worker's pass flag is built from valid-window count, percent-positive windows, aggregate OOS expectancy positive, and total-OOS-trades floor. The walk-forward-efficiency value is COMPUTED but is NOT part of the worker's pass logic.
- `trading_app/db_manager.py` lines 170-580 — `validated_setups` schema base plus the chronological migration ladder. New columns are added via try-except ALTER TABLE blocks. Adding `validation_pathway` and `c8_oos_status` follows the same pattern.
- `docs/institutional/pre_registered_criteria.md` Amendment 3.0 (locked) — verbatim text "Criteria 6, 8, 9 are ALL mandatory for Pathway B. No regime waivers, no insufficient OOS data exemptions."
- `docs/institutional/literature/harvey_liu_2015_backtesting.md` Exhibit 4 (page 22) — single-test profitability hurdles by observation count, used to ground the discussion of whether the N-equals-thirty threshold is literature-derived or folklore.
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` Theorem 1 — confirms MinBTL at K equals one is zero, leaving Pathway B with no Bailey-derived finite-sample protection, which is exactly why downstream gates must be enforced rigorously.
- `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md` Section 1.7.3 — verbatim "data scarcity is not an excuse to abandon rigor. It is a constraint that shapes which techniques we apply." This directly forbids the approach of "we have less than thirty OOS trades, therefore we silently pass-through."
- `.gitignore` — already covers `gold.db` and `gold.db.bak*` but the glob does NOT match `gold.db.pre-e2-fix.bak` because the pattern requires `.bak` immediately after `gold.db.`, with nothing in between. `gold_snap.db` is not in the ignore file at all.

### Why the literature does not bless N-equals-thirty as the right cutoff

Harvey-Liu Exhibit 4 gives concrete profitability hurdles only at one hundred twenty observations and above. The exhibit is silent below one hundred twenty. The N-equals-thirty heuristic is folklore from the Central Limit Theorem applied to roughly-normal distributions. Our R-multiple distributions are highly non-normal, so even thirty trades has marginal coverage. The honest framing for THIS plan is: thirty is a CLT placeholder, not a Bailey or Harvey-Liu prescription, and a future work item is to replace it with a power-grounded threshold derived from each strategy's in-sample effect size. THIS plan does not relitigate the threshold value. THIS plan reconciles the existing threshold with Amendment 3.0's "no insufficient OOS data exemptions" clause.

### Why this matters

If we ship the current code as-is and run a Pathway B discovery against the only individual-mode hypothesis file currently in the registry (`2026-04-09-mnq-rr10-individual.yaml`), any survivor will appear in `validated_setups` with the implication that it cleared all Amendment 3.0 conditions when in fact two of the three "non-waivable" downstream gates (Criterion 6 walk-forward-efficiency and Criterion 8 OOS sample size) were silently bypassed. Bailey's False Strategy Theorem tells us that any random walk produces significant t-statistics given enough degrees of freedom. With Criterion 6 and Criterion 8 cracked, Pathway B has no remaining structural defence beyond the pre-registered theory citation — and a single citation is a thin shield.

The CRITICAL D-1 finding is independent of the Pathway B work. It blocks ANY push to origin, so it must be addressed before any other commits move out.

---

## Turn 2 — Design

### Three takes on the C-8 question (the load-bearing decision)

I deliberated three approaches for reconciling the N-OOS-less-than-thirty pass-through with Amendment 3.0. Each has trade-offs.

**Take one: Pathway-asymmetric strict mode.** Keep the existing pass-through for Pathway A (legacy permissive behaviour, test-fixture compatibility) but make Pathway B reject hard at N-OOS less than thirty. The C8 helper takes a new flag indicating strict mode; the validator passes the flag based on the run-level `testing_mode` argument. Smallest blast radius. Honours Amendment 3.0 literally for Pathway B. Leaves Pathway A unchanged so we do not have to retroactively reject 124 grandfathered strategies. The honest weakness is that Pathway A's silent pass-through still happens, just now flagged in the audit trail via a new column rather than repaired.

**Take two: Universal strict mode with deferred-state for both pathways.** Both pathways move to: when N-OOS is less than thirty, mark the strategy as `PENDING_OOS_DATA` instead of promoting. The strategy stays in `experimental_strategies` and gets re-evaluated next time the validator runs. This is the most institutionally pure option — no silent passes, no asymmetry, full re-runnability. The honest weakness is that it adds a new validation status string that must be threaded through five places (the main loop, the database write block, the phase-counts dictionary, the logging counters, and the test-fixture protection branch). Larger blast radius. It also introduces an asymmetry with the existing N-equals-zero pass-through which exists to protect synthetic test databases — we would have to keep that exception while removing the one-to-twenty-nine exception, which is logically weird.

**Take three: Statistically grounded power gate.** Compute, at validation time, the minimum OOS sample needed to detect the in-sample effect size at five percent significance with eighty percent power (Cohen's d framework). Compare actual N-OOS to that derived minimum, not to a folklore thirty. The N-equals-thirty constant disappears entirely; each strategy has its own personalized minimum. Most rigorous. Largest blast radius — requires implementing a power formula module, mapping our R-multiple distribution to a normal-ish framework, and deciding what to do about strategies whose in-sample effect size is so small that no realistic OOS sample could detect it.

**Recommendation: Take one for THIS plan.** Take three is the right institutional answer but it is a research project of its own — Harvey-Liu Exhibit 4 only covers one hundred twenty observations and above, our R-multiple distributions are non-normal, and the power-formula choice itself involves modelling decisions that need their own design pass. Take two adds blast radius without commensurate institutional gain (the asymmetry between N-equals-zero and N-in-one-to-twenty-nine remains). Take one is the minimum-blast-radius fix that fully honours Amendment 3.0's literal text for Pathway B and adds an audit-trail column so Pathway A's pass-through becomes visible to future audits. We add a deferred-work item for Take three in the plan's "explicitly-out-of-scope" section.

### Three takes on the WFE gate (A-1)

**Take one: Mirror the Pathway A line at line 1906 inside the Pathway B branch.** Add a query for the strategy's stored `wfe` value, fail-closed to zero on missing, reject if less than the canonical `MIN_WFE` constant. Two new lines of code, identical pattern to Pathway A. Zero new abstractions. Trivial test.

**Take two: Push the WFE check DOWN into the walk-forward worker so both pathways inherit it for free.** Modify `walkforward.py` to add walk-forward-efficiency-greater-than-or-equal-to-MIN-WFE to the four-condition pass rule. Then the worker's pass flag enforces the gate, Pathway A's line 1906 becomes redundant (delete it), and Pathway B inherits the gate without any new code. This is the institutional-rigor rule four move ("delegate to canonical sources, never re-encode") — the duplicated WFE check between Pathway A and the worker becomes a single check in the worker. The honest weakness is that it changes the worker's pass semantics for the entire validator, which has implications beyond Pathway B. Some strategies that currently pass walk-forward-without-WFE would start failing, which would shift the count of strategies that reach the FDR gate, which would shift FDR adjusted p-values. That is a behaviour change for Pathway A, not just Pathway B.

**Take three: Add a new dedicated function `_check_criterion_6_wfe(row_dict, con)` that mirrors the structure of the other criterion helpers, and call it from both Pathway A and Pathway B branches.** This is the cleanest abstraction but requires changing the call site in Pathway A as well, which expands scope.

**Recommendation: Take one for THIS plan.** Take two is the right long-term refactor (and I will note it as a deferred item) but it changes Pathway A behaviour, which is out of scope for "fix the Pathway B findings." Take one is the minimum-change fix that honours Amendment 3.0 literally and matches the existing Pathway A pattern exactly.

### One take on the audit-trail column (A-3)

There is no real ambiguity here. We add a new `validation_pathway` column to `validated_setups` (and to `experimental_strategies` for symmetry, since the legacy code does sometimes carry pathway-style markers there too). The column takes one of three values: family, individual, grandfathered. Pathway A writes "family", Pathway B writes "individual", and the grandfathered code path that pre-dates Amendment 3.0 writes "grandfathered" (or null). One ALTER TABLE migration in `db_manager.py`. Both Pathway A and Pathway B database UPDATE statements gain the new column. Future auditors filter by this column to disambiguate. It also makes the `fdr_adjusted_p` semantic drift visible — when `validation_pathway` is "individual" the column should be read as "raw bootstrap p", when it is "family" the column is the real BH-adjusted p.

Optionally we add a second column `c8_oos_status` taking values "passed", "pass_through_no_data", "pass_through_insufficient_n", null. This makes the C8 pass-through explicitly visible in `validated_setups` for Pathway A strategies that go through the relaxed path. I will include this in the plan since it is the minimum-additional-work way to address the "silent failure" half of finding A-2 for the Pathway A side.

### One take on integration test coverage (D-2)

We need at least one end-to-end integration test that runs `run_validation` with `testing_mode="individual"` against a small synthetic DuckDB and asserts that the four critical decision points all fire correctly: raw p less than five percent passes, raw p greater than five percent rejects, sharpe-greater-than-zero passes, sharpe-less-than-zero rejects, walk-forward-efficiency-greater-than-or-equal-to-zero-point-five passes, walk-forward-efficiency-less-than-zero-point-five rejects, N-OOS-greater-than-or-equal-to-thirty under strict mode passes, N-OOS-less-than-thirty under strict mode rejects. That gives us a baseline integration test that exercises the new Pathway B code path end-to-end.

In addition, we add focused unit tests for each new gate (one for the WFE check, two for the strict-vs-permissive C8 mode, one for the new column population in both pathways).

### Layer placement

All the code changes live inside `trading_app/`. No `pipeline/` files are touched. The one-way dependency rule is preserved.

### Two-take check on phasing

The four findings can be addressed in three or four stages depending on how aggressively we bundle. I considered two phasings.

**Phasing A: One big stage covering everything except D-1.** A single commit refactors the Pathway B branch, adds the WFE gate, adds the strict C8 mode, adds the schema migration, and adds the integration tests. Pros: coherent context, single review pass. Cons: large blast radius for a single stage, harder to roll back any single piece without rolling back all of them.

**Phasing B: Four small focused stages.** D-1 first, then A-1 alone, then A-2 alone, then A-3 plus D-2 together. Pros: each stage has crisp acceptance criteria, each is independently rollable, each is independently reviewable. Cons: more ceremony, more commits.

**Recommendation: Phasing B.** The institutional-rigor rule "review own work before claiming done" gets stronger when each stage is small and can be reviewed in isolation. The repo-hygiene stage (D-1) MUST come first because a destructive history rewrite invalidates downstream commits. Stages 2 and 3 (A-1 and A-2) are completely independent and can ship in either order. Stage 4 (A-3 + D-2) depends on both A-1 and A-2 being in place because the integration tests need to verify both gates fire correctly.

---

## Turn 3 — Detail

### Stage 1 — Repo hygiene (D-1)

**Purpose:** Remove the 4.89 gigabytes of accidentally-committed binary blobs from git's tracked state and from the history of the local commits-ahead-of-origin so the branch becomes pushable again.

**Files touched:**
- `.gitignore` — add explicit entries for `gold.db.pre-e2-fix.bak` and `gold_snap.db`. The existing `gold.db.bak*` glob does NOT match the actual filename so we need an explicit entry.
- `gold.db.pre-e2-fix.bak` — remove from tracking via `git rm --cached`. The file remains on disk for the user to keep or delete manually.
- `gold_snap.db` — same.

**Step-by-step (executed by the implementation agent, not by me unilaterally):**

1. Confirm with the user that destructive history rewrite is acceptable. The fourteen commits ahead of origin will get NEW SHAs. If anyone else has cloned this branch (unlikely on a solo project but worth confirming), they will need to re-clone.
2. Add the two filenames to `.gitignore`.
3. Run `git rm --cached gold.db.pre-e2-fix.bak gold_snap.db`. The files remain on disk.
4. Commit the gitignore + untracked-state change with message "chore: untrack accidentally-committed gold.db snapshot blobs".
5. Use `git filter-repo --invert-paths --path gold.db.pre-e2-fix.bak --path gold_snap.db --refs HEAD~15..HEAD` to remove the blobs from history. If `git filter-repo` is not installed on Windows, fall back to `git filter-branch --index-filter` with the equivalent syntax (slower but bundled with git). NOTE: filter-repo or filter-branch rewrites SHAs of every commit from `03e9c9c` forward.
6. Verify with `git ls-files | grep -E "(gold.db.pre|gold_snap)"` returning empty AND `git log --all --oneline -- gold.db.pre-e2-fix.bak gold_snap.db` returning empty AND `du -sh .git` showing a substantial size reduction (the .git directory should drop by approximately 4.89 GB minus delta-compression savings).
7. Repository is now pushable.

**Acceptance:**
- `git ls-files` does not list either filename.
- `git log --all --oneline -- gold.db.pre-e2-fix.bak gold_snap.db` returns no commits.
- The `.git` directory size on disk is significantly smaller than before.
- `git push origin main` (when eventually run by the user) succeeds without GitHub size errors.

**Rollback:** If anything goes wrong during filter-repo, the original branch state is preserved in `.git/refs/original/refs/heads/main` (filter-branch) or in `git reflog` (filter-repo). A `git reset --hard HEAD@{1}` recovers the prior state. The filter-repo workflow is the standard one documented in the git-filter-repo manual.

**Why this comes first:** Destructive history rewrite invalidates SHAs of every later commit. If we did this LAST, every subsequent stage would need to be re-applied to the new history. Doing it first means stages two through four work against the cleaned history.

### Stage 2 — Pathway B walk-forward-efficiency enforcement (A-1)

**Purpose:** Add the Criterion 6 walk-forward-efficiency floor enforcement to the Pathway B branch. Mirrors the existing Pathway A enforcement at line 1906 of `strategy_validator.py`.

**Files touched:**
- `trading_app/strategy_validator.py` — inside the Pathway B branch, after the raw-p-and-Sharpe check and before the database UPDATE, add a query that fetches the strategy's stored walk-forward-efficiency value from `validated_setups`, fail-closed to zero on missing, and rejects when the value is below the canonical `MIN_WFE` constant (currently zero point five zero, sourced from `trading_app.config`).
- `tests/test_trading_app/test_strategy_validator.py` — add focused tests for the new gate.

**Step-by-step:**

1. Locate the Pathway B branch in `strategy_validator.py` (currently lines 1783 through 1843).
2. Inside the per-strategy loop, after the existing raw-p-less-than-five-percent and sharpe-greater-than-zero check, add a sub-step that queries `validated_setups` for the strategy's `wfe` value (the column already exists, populated by Phase B walk-forward).
3. If the value is missing or null, treat as zero (fail-closed). If the value is less than `MIN_WFE`, transition the strategy to the rejected branch with a reason string "criterion_6_pathway_b: wfe equals X, less than MIN_WFE equals zero point five zero (Amendment 3.0 condition four, walk-forward gate non-waivable)".
4. The reject path uses the same database UPDATE pattern as the existing raw-p reject (DELETE from `validated_setups`, UPDATE `experimental_strategies` with `validation_status` REJECTED and rejection reason).
5. Mirror the counters: `n_pathway_b_rejected` increments, `pathway_b_rejected_ids` appends.
6. Add three new tests in `test_strategy_validator.py`:
   - Test name "test pathway b walk forward efficiency gate passes at min wfe": fixture strategy with raw p of zero point zero three, sharpe of one point two, wfe of zero point five zero exactly, expects promotion.
   - Test name "test pathway b walk forward efficiency gate rejects below min wfe": same fixture but wfe of zero point four nine, expects rejection with the criterion-6-pathway-b reason string.
   - Test name "test pathway b walk forward efficiency gate fail closed on missing": same fixture but wfe is null, expects rejection (fail-closed).

**Acceptance:**
- The three new unit tests pass.
- The full validator test suite still passes (currently one hundred fifty eight, expected one hundred sixty one after this stage).
- A grep for `MIN_WFE` inside the Pathway B branch confirms the gate is wired in.
- `python pipeline/check_drift.py` still passes.

**Rollback:** Single-commit revert. The change is purely additive to the Pathway B branch. No schema changes, no signature changes, no caller-side updates needed.

### Stage 3 — Criterion 8 strict mode for Pathway B (A-2)

**Purpose:** Reconcile the N-OOS-less-than-thirty pass-through with Amendment 3.0's "no insufficient OOS data exemptions" clause for Pathway B. Pathway A retains the legacy permissive behaviour for backward compatibility with the one-hundred-twenty-four grandfathered strategies, but Pathway B rejects hard.

**Files touched:**
- `trading_app/strategy_validator.py` — three signature changes:
  - The C8 helper gains a new keyword parameter named `strict_oos_n` defaulting to false. When false, the existing N-less-than-thirty pass-through fires (preserving Pathway A behaviour). When true, the helper returns a rejection with reason "criterion_8: N OOS equals X, less than thirty, Pathway B requires sufficient OOS sample (Amendment 3.0 condition four, no insufficient OOS data exemptions)". The thirty constant gets renamed to `_OOS_MIN_TRADES_CLT_HEURISTIC` and gains a docstring noting that thirty is a Central Limit Theorem heuristic, not a Bailey or Harvey-Liu prescription, with a deferred-work pointer.
  - The pre-flight gates dispatcher gains a new keyword parameter `testing_mode` defaulting to "family". When the value is "individual", it passes `strict_oos_n=true` to the C8 helper.
  - The `run_validation` main loop passes its own `testing_mode` argument through to the pre-flight gates dispatcher.
- `tests/test_trading_app/test_strategy_validator.py` — add three new C8 tests covering the strict-vs-permissive split.

**Step-by-step:**

1. In `strategy_validator.py`, rename the existing `_OOS_MIN_TRADES` local constant to `_OOS_MIN_TRADES_CLT_HEURISTIC` and add a multi-line docstring explaining the heuristic basis and the deferred work item.
2. Add a new parameter to the C8 helper signature: `strict_oos_n: bool = False`. Default false preserves existing behaviour for all callers that do not pass the new flag.
3. Inside the C8 helper, the existing N-equals-zero pass-through stays unchanged (test-fixture compatibility). The N-in-one-to-twenty-nine branch gains a new condition: when `strict_oos_n` is true, return ("REJECTED", strict-mode reason string). When `strict_oos_n` is false, return the existing pass-through with an additional logger warning (institutional-rigor rule six: no silent failures, log the pass-through at warning level not info level).
4. Add the same `strict_oos_n` parameter to the pre-flight gates dispatcher. When called with the flag set, it passes the flag through to its internal C8 helper call.
5. In `run_validation`, the per-row pre-flight call gains an extra argument: pass `strict_oos_n=(testing_mode == "individual")`. This is the only place where `testing_mode` influences C8 behaviour — everything downstream is governed by the flag set here.
6. Add three new tests:
   - Test name "test criterion 8 permissive mode passes through low n": existing test renamed for clarity, asserts that the default `strict_oos_n=False` continues to pass-through at N less than thirty.
   - Test name "test criterion 8 strict mode rejects low n": new test, calls the helper with `strict_oos_n=True` and asserts rejection with the expected reason string.
   - Test name "test criterion 8 strict mode passes at n equals thirty": boundary test, calls the helper with `strict_oos_n=True` and N exactly thirty, asserts pass.

**Acceptance:**
- The three new C8 tests pass.
- The full test suite passes.
- A grep confirms `_OOS_MIN_TRADES` is gone and `_OOS_MIN_TRADES_CLT_HEURISTIC` is the only constant name.
- A grep confirms `strict_oos_n` flows from `run_validation` through the pre-flight dispatcher into the C8 helper.
- `python pipeline/check_drift.py` still passes.
- The pass-through log level is now warning, not info, so future audit log scans see the event.

**Rollback:** Single-commit revert. The change is purely a new keyword parameter with a backward-compatible default; all existing callers continue to work unchanged.

### Stage 4 — Audit trail column plus integration tests (A-3 and D-2)

**Purpose:** Make Pathway A versus Pathway B distinguishable in the validated_setups audit trail, AND add the comprehensive integration tests that exercise the full Pathway B branch end-to-end on a synthetic database.

**Files touched:**
- `trading_app/db_manager.py` — add a new ALTER TABLE migration that adds `validation_pathway VARCHAR` to both `experimental_strategies` and `validated_setups`. Add a second migration for `c8_oos_status VARCHAR` on `validated_setups` only.
- `trading_app/strategy_validator.py` — both Pathway A and Pathway B branches' database UPDATE statements gain the new column write. Pathway A writes "family". Pathway B writes "individual". The grandfathered legacy path leaves the column null. The C8 pass-through path (for Pathway A) populates `c8_oos_status` with "pass_through_insufficient_n" or "pass_through_no_data" as appropriate; strict-mode passes set "passed".
- `tests/test_trading_app/test_strategy_validator.py` — add the integration tests.
- `pipeline/check_drift.py` — verify the existing "validation_pathway not null for new rows" sort of check is not violated by old grandfathered rows. May not need any change; depends on whether drift checks already discriminate by created-at.

**Step-by-step:**

1. In `db_manager.py`, add ALTER TABLE migrations following the existing pattern (try-except CatalogException). Place them in the chronological migration block near the other 2026 migrations.
2. In `strategy_validator.py` Pathway A branch (the FDR path, around line 1910-1929), the existing UPDATE statement on `validated_setups` already has many fields. Add `validation_pathway = 'family'` to the SET clause.
3. In Pathway B branch (around lines 1801-1812), the analogous UPDATE statement currently sets `fdr_significant`, `fdr_adjusted_p`, `p_value`, and conditionally `discovery_date`. Add `validation_pathway = 'individual'` to the SET clause. ALSO populate `discovery_k = 1` (Pathway B is K-equals-one by Amendment 3.0 framing).
4. In the C8 helper, when the helper takes the strict-mode-or-permissive branches, make it record an out parameter or return value indicating the c8 status. The simplest pattern is to leave the existing two-tuple return shape alone and have the validator's main loop populate `c8_oos_status` based on its own knowledge of which return path fired. For permissive mode passing strategies that hit the N-less-than-thirty branch, the value is "pass_through_insufficient_n". For permissive mode passing strategies that hit the N-equals-zero branch, the value is "pass_through_no_data". For strategies that pass the real sign-and-ratio gate, the value is "passed". This requires the validator to call a slightly more informative version of the C8 helper, OR to inspect the helper's logger output. The cleanest approach is to make the C8 helper return a three-tuple instead of two-tuple, with the third element being the status string. ALL CALLERS update accordingly.
5. Add the integration tests:
   - Test name "test pathway b individual mode promotes valid strategy end to end": builds a synthetic DB with one experimental strategy that has raw p of zero point zero three, sharpe of one point two, wfe of zero point seven, thirty OOS trades all positive. Calls `run_validation(testing_mode="individual")`. Asserts the strategy ends up in `validated_setups` with `validation_pathway` equals "individual", `fdr_adjusted_p` equals zero point zero three, `c8_oos_status` equals "passed", `discovery_k` equals one.
   - Test name "test pathway b rejects on raw p above threshold": same fixture but raw p of zero point zero seven. Asserts rejection with criterion-3-pathway-b reason and the strategy is NOT in `validated_setups`.
   - Test name "test pathway b rejects on negative sharpe direction": same fixture but sharpe of negative one point two. Asserts rejection.
   - Test name "test pathway b rejects on walk forward efficiency below threshold": same fixture but wfe of zero point three. Asserts rejection with criterion-6-pathway-b reason.
   - Test name "test pathway b rejects on insufficient oos sample under strict mode": same fixture but only fifteen OOS trades. Asserts rejection with criterion-8 strict-mode reason.
   - Test name "test pathway a family mode populates validation pathway field": baseline test, runs `run_validation()` with default `testing_mode="family"` against a synthetic DB and asserts the surviving strategy has `validation_pathway` equals "family".
6. Verify drift checks still pass.

**Acceptance:**
- All six new integration tests pass.
- The full test suite passes (expected count approximately one hundred sixty seven).
- A query against any test DB after a Pathway A run shows `validation_pathway = 'family'` for survivors. A query after a Pathway B run shows `validation_pathway = 'individual'`.
- A query for `WHERE c8_oos_status != 'passed' AND validation_pathway = 'family'` returns the audit trail of Pathway A pass-through strategies — the silent failure is now visible.
- `python pipeline/check_drift.py` still passes.

**Rollback:** Two-commit revert (the schema migration and the validator UPDATE changes). Roll back in reverse order: first the validator UPDATE writes, then the schema migration. Tests get rolled back with their respective changes.

---

## Turn 4 — Validate

### Failure modes

1. **Filter-repo on Windows.** The standard `git filter-repo` tool is a Python script not bundled with stock git on Windows. The fallback is `git filter-branch` which is bundled but slower and more error-prone. Mitigation: try `pip install git-filter-repo` first; if that fails, the user can run filter-branch manually with our exact syntax. If neither works, a worst-case fallback is to abandon the Stage-1 history rewrite and instead push a brand-new branch starting from `origin/main` with the four desired stages cherry-picked on top, leaving the history-with-blobs as a dead local branch. This is uglier but always works.

2. **The Pathway B branch already inserted into validated_setups before the gate runs.** Pathway B's reject path explicitly DELETEs the strategy from `validated_setups`. This is correct for the existing flow. After our changes the reject path needs to fire BEFORE OR DURING the same UPDATE-and-DELETE block. We need to make sure we add the WFE check and the strict-C8 rejects in the right order so that ALL three rejection conditions (raw p, sharpe, WFE, strict-C8) cleanly converge on the existing reject branch.

3. **The strict-C8 enforcement runs in the pre-flight gate, BEFORE the legacy validator and BEFORE Phase B walk-forward.** This means strict-C8 rejection happens early, the strategy never reaches Pathway B's per-strategy loop. Good. But the C8 helper queries the OOS data per row, which is expensive. Pre-flight already accepts this cost. Confirmed no regression.

4. **Three-tuple return from the C8 helper changes its API surface.** All existing callers must be updated. Grep for the helper name to find them all. The current internal call sites are: the pre-flight dispatcher (one place), the legacy permissive path (zero direct calls — the helper is only called from the pre-flight dispatcher). Plus tests. So the API change is local.

5. **The integration tests need fixtures that closely mimic real Pathway B input.** Walk-forward results, OOS trades, hypothesis files with `testing_mode: individual`, the SHA stamping, the grandfathering predicate. Risk: the fixtures get out of sync with the real runtime. Mitigation: build the fixtures from the existing `_phase_4_row` helper used by C8 tests today, augment with hypothesis-file SHA stamping. The cost is one fixture-builder helper that gets reused across the six integration tests.

6. **Drift check 94 (the hypothesis-file integrity check from Stage 4.1) may interact with our schema migration.** The check validates that every post-cutoff experimental row carries a non-null `hypothesis_file_sha`. It does not yet know about `validation_pathway`. Adding the column should not break the existing drift check. If we want a NEW drift check that asserts "every Pathway B row has `validation_pathway = 'individual'`" we add it as a separate concern in a follow-up stage, not here.

7. **The `n_fdr_rejected` variable name is reused for Pathway B rejections at the existing line 1838.** This is misleading and was flagged as Low-A-4 in the review. Phasing-wise this is so trivial we can fold it into Stage 4 as a one-line rename. Or leave it as-is and clean up later. Not load-bearing.

### Tests that prove correctness

- **Stage 1 (D-1):** `git ls-files | grep gold_snap` returns empty. `git log --all --oneline -- gold.db.pre-e2-fix.bak` returns empty. `du -sh .git` shows substantial reduction.
- **Stage 2 (A-1):** Three new unit tests in `test_strategy_validator.py` proving the WFE gate passes at exactly MIN_WFE, rejects below MIN_WFE, and fails closed when WFE is null.
- **Stage 3 (A-2):** Three new unit tests proving the strict-mode C8 rejects at N less than thirty, passes at N equals thirty, and the permissive-mode C8 still passes through (backward compat).
- **Stage 4 (A-3 + D-2):** Six end-to-end integration tests exercising the full Pathway B branch through `run_validation`, plus one Pathway A regression test that confirms the new `validation_pathway` column is populated as "family".
- **Across all stages:** Full validator test suite (one hundred sixty seven expected after Stage 4) passes. `python pipeline/check_drift.py` shows no new failures.

### Rollback plan

Each stage is a single commit. If any stage's tests fail post-implementation, `git revert <stage-commit>` cleanly backs out the stage. The stages are independent in execution order (Stage 2 and Stage 3 do not depend on each other; Stage 4 depends on both being in place but its own changes are additive). Only Stage 1 (the history rewrite) is destructive and requires reflog recovery rather than revert.

### Guardian prompts

No new guardian prompts needed. Existing institutional-rigor rules cover the scope. The integrity-guardian rules four ("delegate to canonical sources, never re-encode") and six ("no silent failures") and seven ("never trust metadata, always verify") are the load-bearing rules for this work.

### Out-of-scope (deferred work, named explicitly so we do not forget)

- **Replace the N-equals-thirty CLT heuristic with a power-grounded threshold derived from each strategy's in-sample effect size.** This requires implementing a Cohen's-d-style power calculator and choosing a target detection power. Literature reference for the eventual fix: Harvey-Liu Exhibit 4 single-test hurdles; Cohen 1988 power formulas. Tracking item.
- **Push the WFE check down into the walk-forward worker and remove the duplicated check at line 1906 of the validator.** This is the institutional-rigor rule four refactor. It changes Pathway A behaviour (some strategies that currently pass would start failing) so it needs its own design pass and is not included in this plan. Tracking item.
- **Add a drift check that asserts every post-cutoff `experimental_strategies` row with `testing_mode = 'individual'` ends up with `validation_pathway = 'individual'` in `validated_setups` (or REJECTED in `experimental_strategies`).** Keeps the audit trail consistent. Tracking item.
- **Per-hypothesis-file consistency check between the hypothesis-loader's `testing_mode` field and the validator's `--testing-mode` argument.** Currently nothing enforces that they agree. Tracking item.
- **A future Amendment 3.1 to the locked criteria file that codifies the N-equals-thirty heuristic explicitly (or replaces it with the power formula).** Currently the N-equals-thirty constant is in code only — the locked criteria file says "no insufficient OOS data exemptions" without defining "insufficient". A future amendment should define the threshold in the locked text so it cannot drift. Tracking item.

---

## Acceptance gate (cross-stage)

After all four stages ship:

1. Full test suite: approximately one hundred sixty seven validator tests pass plus all loader tests pass (expected total approximately five hundred seventy three project-wide).
2. `python pipeline/check_drift.py` passes with no new failures.
3. `git ls-files` does not contain either binary blob.
4. A dry-run of `python -m trading_app.strategy_validator --instrument MNQ --testing-mode individual --dry-run` against the existing `2026-04-09-mnq-rr10-individual.yaml` hypothesis file completes without errors and emits log lines for the WFE gate and the strict-C8 gate firing.
5. `git push origin main` succeeds.

---

## Turn 5 — Iteration gaps filled (self-review pass)

After writing the plan I did a self-review and found seven concrete gaps that I am closing now before implementation.

**Gap one: the "fdr_adjusted_p semantic drift" claim needed downstream-consumer verification.** On grep I found `scripts/tools/audit_fdr_integrity.py` actively correlates `e.p_value` against `v.fdr_adjusted_p` and asserts "no active strategy should have fdr_adjusted_p greater than alpha" (line 394). For Pathway B strategies this audit will fire misleadingly because the "fdr_adjusted_p" value is actually the raw bootstrap p, not adjusted. Confirmed A-3 is a REAL bug not a theoretical one. The `validation_pathway` column added in Stage 4 lets the audit tool filter to family-only strategies when running FDR-specific checks. A follow-up will update audit_fdr_integrity.py to be pathway-aware — noting this as tracking item, not in Stage 4 scope.

**Gap two: the C8 helper three-tuple API change breaks five existing test call sites.** On grep I found `_check_criterion_8_oos` is called from one production site (`_check_phase_4_pre_flight_gates` at line 1166) and five test sites in `test_strategy_validator.py` (lines 1368, 1380, 1394, 1403, 1415). Stage 4 must update all five test unpacking patterns from two-value to three-value. Adding this as an explicit Stage 4 sub-step.

**Gap three: pre-flight-gates callers also need update.** The `_check_phase_4_pre_flight_gates` function is called from one production site (line 1312) and three test sites (1468, 1480, 1490). Adding `testing_mode` as a keyword with default "family" preserves backward compatibility for test calls. Tests continue to work unchanged. Drift check at pipeline/check_drift.py line 4087 just greps for the function name, does not care about signature.

**Gap four: how the integration tests supply walk-forward results.** My original plan said "use a fixture DB" without specifying. The real answer is `monkeypatch.setattr(strategy_validator, '_walkforward_worker', fake_worker)` in the integration tests. The fake worker returns a dict with a `wf_result` sub-dict containing a known `passed` flag and a known `wfe` value in the `as_dict` sub-sub-dict. That way the tests exercise the full `run_validation` path including the Pathway B branch without needing a realistic walk-forward fixture. Locking this into Stage 4.

**Gap five: integration test rows should be grandfathered (hypothesis_file_sha null) to sidestep pre-flight gate complexity.** My plan vaguely said "synthetic DB fixture" but did not specify whether the fixtures go through the pre-flight gates. The cleanest approach is: Pathway B branch tests use grandfathered rows (null SHA) so pre-flight gates return (None, None) early and the test focuses on the Pathway B branch logic. Then ONE separate test covers the non-grandfathered strict-C8 plumbing end-to-end. This is the test pyramid: small unit tests for each gate, one integration test for the plumbing.

**Gap six: Stage 1 destructive-rewrite sequencing.** My original plan said "Stage 1 must come first because history rewrite invalidates SHAs of later commits." On reflection the correct sequencing is actually: do the NON-destructive part of Stage 1 (gitignore + `git rm --cached`) first, then do Stages 2, 3, 4 as normal commits, then run `git filter-repo` at the END to scrub the blobs from the `03e9c9c` commit's history. The filter-repo pass rewrites all ahead-of-origin commits' SHAs in one shot, which is cleaner than doing it first and then rebuilding. The user explicitly confirms the filter-repo step before it runs.

**Gap seven: the thirty-trade constant needs a docstring that does not lie.** I am renaming `_OOS_MIN_TRADES` to `_OOS_MIN_TRADES_CLT_HEURISTIC` in Stage 3 and adding a docstring explicitly noting that thirty is CLT folklore, NOT a Bailey or Harvey-Liu prescription, and that a future Stage would replace it with a power-grounded threshold. Institutional rigor rule eight: verify-before-claiming, and also do not encode fake pedigree.

**Gap eight: L-1 variable-name reuse cleanup.** The review flagged `n_fdr_rejected = n_pathway_b_rejected` at line 1838 as a Low-severity naming confusion. Folding a one-line rename into Stage 2 (add a clarifying comment) since it lives right next to the Pathway B branch I am already editing. Trivial.

### Sequencing recap post-iteration

1. Write ONE stage file (`docs/runtime/stages/bloomey-pathway-b-fix.md`) covering the bundled work with scope_lock listing all four files that will be touched.
2. Execute Stage 1 non-destructive part (gitignore + git rm --cached + commit) — safe, no history rewrite.
3. Execute Stage 2 (WFE gate in Pathway B + tests + commit).
4. Execute Stage 3 (C8 strict mode plumbing + tests + commit).
5. Execute Stage 4 (validation_pathway column + c8_oos_status column + integration tests + tests update for 3-tuple return + commit).
6. Run full validator test suite, run `python pipeline/check_drift.py`, verify clean.
7. Report results.
8. STOP before running `git filter-repo` — ask user for explicit confirmation of the history rewrite since that step is destructive. Exact command to run will be written into the stage file.

### Confidence check

I have read every file I will touch. I have traced every caller of every function I will change. I have confirmed that the downstream consumer `audit_fdr_integrity.py` will misfire on Pathway B strategies without the `validation_pathway` column, so A-3 is load-bearing not theoretical. I have confirmed the walk-forward worker is module-level and monkey-patchable. The plan is implementation-ready.

Proceeding to Stage 1 (non-destructive part) immediately.
