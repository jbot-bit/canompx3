---
mode: IMPLEMENTATION
slug: bloomey-pathway-b-fix
task: Fix 5 bloomey-review findings on Amendment 3.0 Pathway B validator work (D-1 binary blobs, A-1 WFE gate, A-2 C8 strict mode, A-3 audit column, D-2 integration tests)
started: 2026-04-09
current_step: 1 of 5
blast_radius: .gitignore, trading_app/strategy_validator.py, trading_app/db_manager.py, tests/test_trading_app/test_strategy_validator.py
---

## Blast Radius

- `.gitignore` — add explicit entries for `gold.db.pre-e2-fix.bak` and `gold_snap.db` (Step 1)
- `trading_app/strategy_validator.py` — Pathway B branch gets WFE gate (Step 2), C8 helper gets `strict_oos_n` kwarg and becomes 3-tuple return (Steps 3+5), pre-flight dispatcher gets `testing_mode` kwarg (Step 3), both pathway branches' UPDATE statements get new column writes (Step 5)
- `trading_app/db_manager.py` — new ALTER TABLE migrations for `validation_pathway` and `c8_oos_status` (Step 5)
- `tests/test_trading_app/test_strategy_validator.py` — new unit tests for WFE gate (Step 2), new C8 tests for strict vs permissive mode (Step 3), new integration tests for Pathway B end-to-end (Step 5), update existing C8 test call sites for 3-tuple return (Step 5)

## Scope Lock

```yaml
scope_lock:
  - .gitignore
  - trading_app/strategy_validator.py
  - trading_app/db_manager.py
  - tests/test_trading_app/test_strategy_validator.py
```

Files NOT in scope_lock (must not be touched):
- `pipeline/check_drift.py` — read-only for verification; new drift checks are deferred work
- `trading_app/hypothesis_loader.py` — loader surface was already done in commit c8efb20
- `scripts/tools/audit_fdr_integrity.py` — downstream consumer; update is deferred work
- Any hypothesis yaml file under `docs/audit/hypotheses/`

## Task context

Post-review fix bundle for the Pathway B / Amendment 3.0 work that shipped in commits ce450fc → c8efb20 → ea18c61 → 149f9d0 → 39b8304. Bloomey review graded the stack C with 1 CRITICAL (binary blobs) + 4 HIGH findings (WFE waiver, C8 silent waiver, fdr_adjusted_p semantic drift, zero integration test coverage). Full design in `docs/plans/2026-04-09-bloomey-pathway-b-fixes-design.md`.

Authority chain: `docs/institutional/pre_registered_criteria.md` Amendment 3.0 (locked text) overrides any convenience relaxations in implementation.

## Acceptance criteria

After all 5 steps complete, before marking stage done:

1. `git ls-files` does not list `gold.db.pre-e2-fix.bak` or `gold_snap.db` (verifies Step 1).
2. Three new Pathway B WFE gate tests pass (Step 2 acceptance).
3. Three new C8 strict/permissive mode tests pass (Step 3 acceptance).
4. Six new Pathway B integration tests pass plus one Pathway A regression test (Step 5 acceptance).
5. Five existing C8 test call sites updated for 3-tuple return and still pass (Step 5 acceptance).
6. Full `test_strategy_validator.py` suite passes at expected count (approximately 171, was 158).
7. `python pipeline/check_drift.py` shows no new failures versus baseline of 86/1/7.
8. A grep for `fdr_adjusted_p =` in `strategy_validator.py` shows both Pathway A and Pathway B UPDATE statements include `validation_pathway` in their SET clauses.
9. `git filter-repo --invert-paths --path gold.db.pre-e2-fix.bak --path gold_snap.db` is STAGED BUT NOT EXECUTED — user must explicitly confirm before this runs because history rewrite is destructive. Exact command is in Step 6 of this stage file.

## Steps

### Step 1 — Untrack the binary blobs (non-destructive Stage 1 part)

- Add `gold.db.pre-e2-fix.bak` and `gold_snap.db` to `.gitignore`
- Run `git rm --cached gold.db.pre-e2-fix.bak gold_snap.db`
- Commit with "chore: untrack accidentally-committed gold.db snapshot blobs"
- Files remain on disk — the user can delete them manually later

### Step 2 — Pathway B WFE enforcement (A-1)

- In `strategy_validator.py` Pathway B branch, add a WFE query mirroring Pathway A line 1906
- Reject with `criterion_6_pathway_b` reason when `wfe < MIN_WFE`
- Fail-closed to 0.0 on missing or null WFE
- Add three unit tests
- Commit

### Step 3 — Criterion 8 strict mode plumbing (A-2)

- Rename `_OOS_MIN_TRADES` constant to `_OOS_MIN_TRADES_CLT_HEURISTIC` with an honest docstring
- Add `strict_oos_n: bool = False` keyword parameter to `_check_criterion_8_oos`
- Add `testing_mode: str = "family"` keyword parameter to `_check_phase_4_pre_flight_gates`
- Thread `testing_mode` from `run_validation` through the dispatcher to C8 helper
- Upgrade the N-less-than-30 pass-through log from `logger.info` to `logger.warning`
- Add three unit tests
- Commit

### Step 4 — (placeholder, merged into Step 5)

Step 4 and Step 5 are merged since the audit-trail column population and the integration tests are tightly coupled.

### Step 5 — Audit column + 3-tuple return + integration tests (A-3 + D-2)

- In `db_manager.py`, add ALTER TABLE migrations for:
  - `validation_pathway VARCHAR` on both `experimental_strategies` and `validated_setups`
  - `c8_oos_status VARCHAR` on `validated_setups`
- In `strategy_validator.py`:
  - Change `_check_criterion_8_oos` return to 3-tuple `(status, reason, c8_status)`
  - Update the ONE production caller in `_check_phase_4_pre_flight_gates` to unpack the 3-tuple and propagate `c8_status` back
  - Dispatcher returns its own 3-tuple, main-loop consumes it and attaches `c8_status` to the serial_results entry
  - Phase C DB write block reads `c8_status` from serial_results and writes it to `validated_setups.c8_oos_status`
  - Pathway A UPDATE at line 1910-1929 adds `validation_pathway = 'family'`
  - Pathway B UPDATE at line 1802-1811 adds `validation_pathway = 'individual'` and `discovery_k = 1` with CASE-WHEN-NULL pattern
- In `test_strategy_validator.py`:
  - Update 5 existing `_check_criterion_8_oos` call sites to unpack 3-tuple `(status, reason, c8_status)`
  - Add 6 new Pathway B integration tests using `monkeypatch` to fake the walk-forward worker
  - Add 1 Pathway A regression test that confirms `validation_pathway = 'family'` on survivors
- Commit

### Step 6 — Destructive Stage 1 part (history rewrite) — REQUIRES USER CONFIRMATION

Do NOT run this without the user saying "yes, run filter-repo" explicitly.

```
git filter-repo --invert-paths --path gold.db.pre-e2-fix.bak --path gold_snap.db
```

On Windows, this may require `pip install git-filter-repo` first. If filter-repo is unavailable, fallback to `git filter-branch --index-filter` with the equivalent arguments (slower).

Verification after the rewrite:
- `git ls-files | grep -E "(gold.db.pre|gold_snap)"` returns empty
- `git log --all --oneline -- gold.db.pre-e2-fix.bak gold_snap.db` returns empty
- `du -sh .git` shows a substantial reduction (approximately 4.89 GB minus delta-compression savings)

## Rollback

Each step is a single commit. Stages 2-5 can be reverted individually with `git revert <stage-commit>`. Step 1 (untrack) can be reverted but leaves the blobs tracked again. Step 6 (filter-repo) can be recovered via `.git/filter-repo/` backup or reflog.
