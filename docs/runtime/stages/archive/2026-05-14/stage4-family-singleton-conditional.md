---
task: Stage 4 — family_singleton conditional downgrade. Replace unconditional hard blocker at trading_app/deployability.py:539 with criteria-gated emit. When SINGLETON-status row clears the locked binding criteria from pre_registered_criteria.md (C3+C4+C6+C7+C9+C10) plus C5 dsr computed-and-reported, downgrade to warning + route to CONTROLLED_LIVE_PILOT_CANDIDATE. Otherwise keep hard. Truth-layer code change; adversarial-audit gate required.
mode: IMPLEMENTATION
scope_lock:
  - docs/runtime/stages/stage4-family-singleton-conditional.md
  - docs/audit/results/2026-05-11-stage4-family-singleton-conditional-impl.md
  - trading_app/deployability.py
  - tests/test_trading_app/test_deployability.py

## Blast Radius

- `trading_app/deployability.py` — modifies:
  - SELECT at line 169-179: add `vs.sharpe_ratio, vs.era_dependent` (column expansion, no row impact).
  - Line 103 RETIRE_OR_PURGE_ISSUES: REMOVE `family_singleton`. `family_purged` stays.
  - Lines 535-540 family branch: split SINGLETON path from PURGED path. SINGLETON emits HARD if any binding criterion fails OR WARNING + adds to controlled-pilot allowlist if all pass.
  - Lines 618-621 controlled_warning_ids: lift to module-level constant `CONTROLLED_PILOT_WARNINGS`; add `"family_singleton"` to it.
- `tests/test_trading_app/test_deployability.py` — adds 4 fixtures: SINGLETON+fail→hard, SINGLETON+all-pass→warning+CONTROLLED_LIVE_PILOT_CANDIDATE, SINGLETON+partial→hard, promotion-bucket routing test.
- Reads: gold.db read-only; canonical Chordia helper `trading_app.chordia.chordia_verdict_label`; canonical filter-era helpers `trading_app.config.ALL_FILTERS` + `pipeline.data_era.is_micro`.
- Writes: none. No DB mutation, no `lane_allocation.json` mutation, no schema change.
- Lane allocator (`scripts/tools/rebalance_lanes.py:109`) reads `validated_setups.status` NOT `StrategyDeployability.verdict` per Stage 4 blast-radius audit; capital allocation is NOT affected by this change. Reporting-only.
- Downstream callers: `scripts/tools/adversarial_stress_gate.py`, `scripts/tools/full_shelf_deployability_audit.py` — both consume the verdict surface unchanged; new CONTROLLED_LIVE_PILOT_CANDIDATE verdict routing is already supported (existing pattern for slippage_event_tail_pending + sr_alarm_watch_reviewed).
- Drift checks: zero existing drift checks key off `family_singleton` string or `BLOCKED_FAMILY_FRAGILE` verdict. No drift check needs update.

## Decisions Locked (user, 2026-05-11)

1. **Verdict for singleton-passing-criteria:** CONTROLLED_LIVE_PILOT_CANDIDATE. Honours Bailey/Carver family-corroboration asymmetry; matches existing pilot pattern.
2. **RETIRE_OR_PURGE_ISSUES set:** remove `family_singleton`. PURGED stays.
3. **PR sizing:** single PR with adversarial-audit gate.

## Binding criteria (from Stage 3 § 4 locked spec)

| # | Criterion | Predicate | validated_setups column |
|---|---|---|---|
| C3 | BH FDR | `fdr_significant = TRUE AND fdr_adjusted_p < 0.05` | `fdr_significant`, `fdr_adjusted_p` |
| C4 | Chordia banded | `chordia_verdict_label(sharpe_ratio, sample_size, has_theory)` in `{PASS_CHORDIA, PASS_PROTOCOL_A}` — DELEGATE to `trading_app.chordia` | `sharpe_ratio`, `sample_size` |
| C5 | DSR cross-check | `dsr_score IS NOT NULL` — reported, NOT gating | `dsr_score` |
| C6 | WFE | `wfe IS NOT NULL AND wfe >= 0.50` | `wfe` |
| C7 | Sample size | `sample_size >= 100` | `sample_size` |
| C9 | Era stability | `era_dependent = FALSE` | `era_dependent` |
| C10 | MICRO-only volume filter compat | if `ALL_FILTERS[filter_type].requires_micro_data`, then `is_micro(instrument) == True` | `filter_type`, `instrument` |

For C4 `has_theory`: SINGLETON candidates are by construction Pathway A (family-pathway promotion); Stage 3 audit confirmed all 5 MES candidates have `validation_pathway = family`. So `has_theory = False` is the default. The C4 helper still distinguishes BAND A (t >= 3.79, passes regardless of theory) from BAND B (3.00 <= t < 3.79, needs theory). Without theory, BAND C and below fail.

## Done criteria

1. SQL SELECT expanded; existing tests still pass (no behaviour change from column addition alone).
2. Conditional-downgrade helper `_singleton_clears_binding_criteria(row)` added; pure function with no IO.
3. `family` branch at lines 535-540 emits hard or warning per helper.
4. `CONTROLLED_PILOT_WARNINGS` module-level constant introduced; `family_singleton` included.
5. `RETIRE_OR_PURGE_ISSUES` updated; `family_singleton` removed.
6. 4 new test fixtures in `test_deployability.py`; existing tests untouched.
7. Empirical regression on real DB: 5 MES Stage-2 candidates evaluated; result documented.
8. Result MD with Bloomberg-grade provenance.
9. `python pipeline/check_drift.py` — full pass.
10. `pytest tests/test_trading_app/test_deployability.py -q` — full pass.
11. Adversarial-audit gate dispatched on the committed change.
12. HANDOFF updated.

## Mid-implementation discipline

If empirical regression surfaces unexpected SINGLETON candidates flipping to warning (i.e. some MNQ row in the 254-row universe unexpectedly clears all criteria + lane-correlation), document and surface to user BEFORE adopting the result as "expected." Stage 4 is reporting-only on capital, but a surprise list of newly-warning candidates is doctrine-relevant and merits review.

## Sibling worktrees

- `stage1/generalize-tbbo-slippage-inference` — PR #258 awaiting merge.
- `stage2/family-singleton-doctrine` — Disposition C + 3 self-audit passes committed.
- `stage3/c5-doctrine-resolution` — floor spec + 3 decisions locked.
- `stage3.5/shelf-wide-c8-backfill` — APPLIED (`c91377fc`).
- `stage4/family-singleton-conditional` — THIS BRANCH.
