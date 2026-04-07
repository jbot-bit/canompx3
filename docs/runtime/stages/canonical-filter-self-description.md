---
mode: IMPLEMENTATION
slug: canonical-filter-self-description
task: Canonical filter self-description — make StrategyFilter classes self-describing, delete parallel eligibility decomposition, thin adapter builder
created: 2026-04-07
updated: 2026-04-07
stage: 1
of: 1
scope_lock:
  - trading_app/config.py
  - trading_app/eligibility/builder.py
  - trading_app/eligibility/types.py
  - trading_app/eligibility/decomposition.py
  - trading_app/eligibility/__init__.py
  - tests/test_trading_app/test_config.py
  - tests/test_trading_app/test_eligibility_builder.py
  - tests/test_trading_app/test_eligibility_decomposition.py
  - tests/test_trading_app/test_eligibility_types.py
  - pipeline/check_drift.py
blast_radius: Adds AtomDescription dataclass + describe() method to StrategyFilter base class in trading_app/config.py (NEVER_TRIVIAL protected, additive only — no changes to matches_row/matches_df). Overrides describe() in ~17 filter subclasses. Rewrites trading_app/eligibility/builder.py as thin adapter (~200 lines, was 700). DELETES trading_app/eligibility/decomposition.py (500 lines of parallel model). DELETES tests/test_trading_app/test_eligibility_decomposition.py. Adds describe() tests to test_config.py. Rewrites test_eligibility_builder.py to test the thin adapter. Adds drift check to pipeline/check_drift.py asserting every ALL_FILTERS entry produces >=1 atom from a sample row. Consumers of StrategyFilter (strategy_discovery.py, strategy_validator.py, outcome_builder.py) unchanged — describe() is purely additive. Existing matches_row/matches_df tests unchanged.
---

# Stage: Canonical Filter Self-Description

## Purpose

Eliminate the parallel-model drift pattern in `trading_app/eligibility/`. Move filter decomposition logic into the filter classes themselves via a new `describe()` method, making filters self-describing. The eligibility builder becomes a thin adapter over the canonical filter classes, with zero re-encoded logic.

**This is a structural fix, not a patch.** Self-code-review of Phase 0+1 and its hardening pass identified the parallel-model architecture as the root cause of accumulating silent failures. Each iteration found new divergences because the architecture made divergence easy. This stage eliminates that possibility mechanically: filter logic has one canonical location, and decomposition lives alongside it.

Full design rationale: `docs/plans/2026-04-07-canonical-filter-self-description-design.md`

## Why this is not a band-aid

The institutional-rigor rule (`.claude/rules/institutional-rigor.md`) forbids offering "just ship it" as a realistic option. This stage applies that rule: instead of fixing the four new bugs from the hardening pass (NaN silent FAIL, ATR None divergence, FAST+E2 latent, size_multiplier footgun), we fix the architecture that made them possible.

After this stage:
- New filters automatically get eligibility support — add to `ALL_FILTERS`, implement `describe()`, done.
- No regex parsing. No hardcoded `validated_for` tuples. No re-encoded comparisons.
- Drift check #N asserts every filter in `ALL_FILTERS` is self-describing — regression to parallel model is mechanically blocked.
- Review cycles focus on per-filter correctness, not cross-module divergence.

## Scope

### Files touched (in order of implementation)

1. **`trading_app/config.py`** — Add `AtomDescription` dataclass. Add `describe()` method to `StrategyFilter` base class with a working default. Override `describe()` on each concrete subclass (~17 classes): `NoFilter`, `OrbSizeFilter`, `OrbVolumeFilter`, `CrossAssetATRFilter`, `OwnATRPercentileFilter`, `OvernightRangeFilter`, `OvernightRangeAbsFilter`, `PrevDayRangeNormFilter`, `GapNormFilter`, `DirectionFilter`, `DayOfWeekSkipFilter`, `ATRVelocityFilter`, `BreakSpeedFilter`, `BreakBarContinuesFilter`, `PitRangeFilter`, `CompositeFilter`, `VolumeFilter`, `CombinedATRVolumeFilter`, `CostRatioFilter`.
2. **`trading_app/eligibility/types.py`** — Keep `EligibilityReport`, `ConditionRecord`, `ConditionStatus`, `FreshnessStatus`, `OverallStatus`. Import `AtomDescription` from config.py for the builder's use. Remove `ConditionCategory`/`ResolvesAt` re-exports (they become strings on AtomDescription).
3. **`trading_app/eligibility/builder.py`** — Rewrite. Thin adapter calling `filter_instance.describe()` and overlay canonical functions. ~200 lines.
4. **`trading_app/eligibility/decomposition.py`** — DELETE.
5. **`trading_app/eligibility/__init__.py`** — Update exports.
6. **`tests/test_trading_app/test_config.py`** — Add `describe()` tests per filter subclass.
7. **`tests/test_trading_app/test_eligibility_builder.py`** — Rewrite. Test the thin adapter against fixture filter instances.
8. **`tests/test_trading_app/test_eligibility_decomposition.py`** — DELETE.
9. **`tests/test_trading_app/test_eligibility_types.py`** — Minor cleanup for any removed imports.
10. **`pipeline/check_drift.py`** — Add drift check: every `ALL_FILTERS` entry produces ≥1 atom from a sample row via `describe()`.

### Files NOT touched (canonical discipline)

- `strategy_discovery.py` — uses matches_row/matches_df, unchanged
- `strategy_validator.py` — same
- `outcome_builder.py` — same
- `init_db.py` — no schema changes
- `build_daily_features.py` — no feature computation changes
- `prop_profiles.py` — no lane changes
- `live/` — Phase 3 dashboard integration waits for this stage to land
- `scripts/tools/generate_trade_sheet.py` — Phase 2 trade sheet integration waits

## Canonical compliance checks

- [ ] `matches_row()` and `matches_df()` signatures unchanged (existing tests still pass)
- [ ] `describe()` is additive on the base class with a working default
- [ ] No changes to `ALL_FILTERS` keys or structure
- [ ] No re-encoded filter logic anywhere outside `trading_app/config.py`
- [ ] Drift check ensures future filters cannot regress

## Acceptance Criteria

1. **`AtomDescription` class defined** in `trading_app/config.py` with fields: name, category, resolves_at, feature_column, observed_value, threshold, comparator, passes, is_data_missing, is_not_applicable, not_applicable_reason, last_revalidated, explanation.
2. **`StrategyFilter.describe()` default implementation** returns one `AtomDescription` derived from `matches_row()` result with sensible defaults.
3. **Every concrete filter subclass has a working `describe()` override** (or uses the default correctly for trivial filters).
4. **`trading_app/eligibility/builder.py` rewritten** as thin adapter — no regex, no decomposition logic, no re-encoded comparisons. Max 250 lines.
5. **`trading_app/eligibility/decomposition.py` deleted.**
6. **`tests/test_trading_app/test_eligibility_decomposition.py` deleted.**
7. **`test_config.py` has tests for `describe()` on each non-trivial filter** — covering PASS/FAIL/DATA_MISSING/NOT_APPLICABLE paths.
8. **`test_eligibility_builder.py` rewritten** — tests the thin adapter with fixture filter instances.
9. **All existing `matches_row`/`matches_df` tests still pass** (no regression to canonical filter evaluation).
10. **NaN values in feature rows surface as DATA_MISSING, not FAIL** — verified by a new test in `test_config.py` using a filter that sees NaN.
11. **ATR velocity with `atr_vel_regime=None` matches canonical behavior** (warm-up fail-open = PASS) — verified by a new test.
12. **CONT and FAST atoms on E2 strategies both return `is_not_applicable=True` with `not_applicable_reason` mentioning entry model** — verified by tests.
13. **HALF_SIZE calendar action produces `passes=True` with `size_multiplier=0.5`** (still separate from pass/fail gate).
14. **`pipeline/check_drift.py`** has a new check that iterates `ALL_FILTERS`, calls `describe()` on each with a sample row, and asserts ≥1 atom returned. Drift check passes.
15. **Full test suite passes.** `PYTHONPATH=. python -m pytest tests/test_trading_app/test_config.py tests/test_trading_app/test_eligibility_*.py -v` → all green.
16. **Self-review** (code-review skill) on the committed refactor produces no new HIGH findings. Any findings are addressed before closing the stage.
17. **No production code outside `trading_app/config.py` + `trading_app/eligibility/` + `pipeline/check_drift.py` modified.** Verified via `git diff HEAD --stat -- trading_app/ pipeline/ scripts/`.

## Commit plan

Logical chunks for reviewability:

1. `feat(config): add AtomDescription + StrategyFilter.describe() base class method`
2. `feat(config): override describe() for simple filters (NoFilter, OrbSizeFilter, DirectionFilter)`
3. `feat(config): override describe() for pre-session filters (PitRange, PrevDayRangeNorm, GapNorm, Overnight, ATRPercentile, CrossAssetATR)`
4. `feat(config): override describe() for intra-session filters (BreakSpeed, BreakBarContinues, CostRatio, OrbVolume, Volume, CombinedATRVolume)`
5. `feat(config): override describe() for overlays and composites (DayOfWeekSkip, ATRVelocity, Composite)`
6. `refactor(eligibility): rewrite builder as thin canonical-delegation adapter`
7. `chore(eligibility): delete decomposition.py and its tests (parallel model eliminated)`
8. `test: add filter describe() coverage in test_config.py`
9. `test: rewrite test_eligibility_builder.py for thin adapter`
10. `feat(pipeline): add drift check #N for ALL_FILTERS describe() coverage`

Each commit must pass the drift check and the relevant test subset.

## Rollback

If the refactor reveals a deeper issue:
1. Revert the commit series
2. Restore `trading_app/eligibility/decomposition.py` from git history
3. Document the blocker in `docs/runtime/blockers/` and return to the band-aid approach with a new tracking plan

No schema changes, no data migration, no drift in validated_setups. Rollback is pure git revert.

## Out of scope

- Phase 2 (trade sheet integration) — waits for this stage
- Phase 3 (dashboard integration) — waits for this stage
- Phase 4 (routing expansion audit) — unaffected, deferred
- Modifying `matches_row` / `matches_df` behavior — this stage is PURELY additive on the canonical path
- Extending `describe()` to cover "what-if" queries (Phase 2 concern)
- Filter confidence tier overhaul — the current `@research-source` annotations stay as-is

## Commit message template (final merge)

```
refactor: canonical filter self-description — eliminate parallel eligibility model

Moves filter decomposition logic from trading_app/eligibility/decomposition.py
into trading_app/config.py as a new StrategyFilter.describe() method. Every
filter class owns its own atomic decomposition, eliminating the parallel-model
drift pattern identified in self-code-review of 046e80b and 7ead764.

Before:
- trading_app/eligibility/decomposition.py had regex parsers and hardcoded
  validated_for tuples duplicating logic from config.py
- trading_app/eligibility/builder.py re-encoded comparisons with its own
  _compare() / _resolve_pdr / _resolve_gap functions
- Each review cycle found new divergences (HALF_SIZE->FAIL, NaN->silent FAIL,
  ATR None divergence, FAST+E2 latent, CONT+E2 label lie)

After:
- StrategyFilter base class has describe(row, orb_label, entry_model) method
- Each concrete filter subclass overrides to return AtomDescription[] that
  accurately reflects its matches_row() semantics
- Eligibility builder is a thin adapter: calls filter.describe() and wraps
  results in ConditionRecord for the report model
- Zero re-encoded logic; divergence from canonical is mechanically impossible
- Drift check #N enforces every ALL_FILTERS entry is self-describing

Files DELETED: trading_app/eligibility/decomposition.py (500 lines),
tests/test_trading_app/test_eligibility_decomposition.py
Files rewritten: builder.py (700->200 lines), test_eligibility_builder.py

NaN silent FAIL: FIXED (filter's own NaN handling now propagates)
ATR None divergence: FIXED (describe() delegates to matches_row directly)
FAST/CONT+E2 label: FIXED (describe() marks is_not_applicable correctly)

Design: docs/plans/2026-04-07-canonical-filter-self-description-design.md
Stage: docs/runtime/stages/canonical-filter-self-description.md
```
