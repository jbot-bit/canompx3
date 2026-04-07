# Canonical Filter Self-Description — Design

**Date:** 2026-04-07
**Status:** APPROVED — implementation proceeding
**Supersedes:** `trading_app/eligibility/decomposition.py` parallel-model approach
**Trigger:** Self-code-review of Phase 0+1 hardening revealed systemic drift pattern.

---

## Why this exists

The first iteration built `trading_app/eligibility/` as a parallel model of filter logic. The builder hand-decomposed composite filters into atoms via `decomposition.py`, maintained its own `validated_for` tuples, its own confidence tiers, its own comparison logic. Each review cycle found new divergences from canonical:

- v1 review (commit 046e80b): HALF_SIZE→FAIL bug, TypeError crash, silent `_compare`, CONT+E2 label lie, dead enum, dead `build_errors` field. Seven findings.
- v2 review (commit 7ead764 hardening): ATR velocity None divergence, NaN silent FAIL, FAST+E2 latent bug, size_multiplier footgun. Four new findings introduced BY the hardening.

The pattern is the architecture, not the bugs. As long as eligibility re-encodes filter logic separately from `trading_app.config.ALL_FILTERS`, every canonical change creates drift opportunity and every review finds new divergences.

## The fix: make filters self-describing

Move the decomposition logic INTO the filter classes. The eligibility builder becomes a thin adapter that calls `filter_instance.describe(row, session, entry_model)` and wraps the result.

**Architectural principle:** if there is one canonical definition of filter logic (in `trading_app/config.py`), then the decomposition of that logic for display MUST also live there. Anywhere else is drift bait.

## The new interface

Add to `trading_app/config.py`:

```
AtomDescription (frozen dataclass)
    name: str                    # "ORB size >= 5 pts"
    category: str                # "PRE_SESSION" | "INTRA_SESSION" | "OVERLAY" | "DIRECTIONAL"
    resolves_at: str             # "STARTUP" | "ORB_FORMATION" | "BREAK_DETECTED" | "CONFIRM_COMPLETE" | "TRADE_ENTERED"
    feature_column: str | None   # already session-resolved, None for overlay atoms
    observed_value: Any          # today's actual value, None if missing/pending
    threshold: Any               # the comparison value
    comparator: str              # ">=", "<", "==", etc.
    passes: bool | None          # True/False/None for PENDING/MISSING
    is_data_missing: bool        # distinguishes missing from FAIL
    is_not_applicable: bool      # filter doesn't apply here
    not_applicable_reason: str   # "entry_model=E2 look-ahead" etc.
    last_revalidated: date | None
    explanation: str             # plain-English
```

Add method to `StrategyFilter` base class:

```
def describe(
    self,
    row: dict,
    orb_label: str,
    entry_model: str,
) -> list[AtomDescription]:
    """Return a list of atomic descriptions for this filter evaluated against
    the given row. Default implementation returns one atom derived from
    matches_row(). Composite filters override to return multiple atoms.

    The filter is responsible for:
    - Determining if each atom is PRE_SESSION / INTRA_SESSION / OVERLAY / DIRECTIONAL
    - Resolving feature columns against the session
    - Checking for missing data (None, NaN, pd.NA, NaT)
    - Evaluating the condition against today's value
    - Marking NOT_APPLICABLE when entry_model excludes the filter
    """
```

Subclasses override when they have multi-atom structure:
- `OrbSizeFilter` → 1 atom (min_size check, optionally max_size atom for bands)
- `CostRatioFilter` → 1 atom (cost ratio calculation)
- `PitRangeFilter` → 1 atom (pit_range_atr check)
- `PrevDayRangeNormFilter` → 1 atom (prev_day_range/atr_20 check)
- `GapNormFilter` → 1 atom (abs(gap)/atr_20 check)
- `OrbVolumeFilter` → 1 atom (orb volume check)
- `OvernightRangeAbsFilter` → 1 atom (overnight_range check)
- `OwnATRPercentileFilter` → 1 atom (atr_20_pct check)
- `CrossAssetATRFilter` → 1 atom (cross_atr_*_pct check)
- `BreakSpeedFilter` → 1 atom (break_delay_min check) — E2 returns NOT_APPLICABLE
- `BreakBarContinuesFilter` → 1 atom — E2 returns NOT_APPLICABLE
- `DirectionFilter` → 1 atom (DIRECTIONAL category)
- `CombinedATRVolumeFilter` → 2 atoms (ATR pct + rel_vol)
- `ATRVelocityFilter` → 1 atom (calls own matches_row)
- `DayOfWeekSkipFilter` → 1 atom per skip day
- `CompositeFilter` → atoms from each sub-filter
- `NoFilter` → zero atoms

## The new eligibility builder

`trading_app/eligibility/builder.py` becomes:

```
def build_eligibility_report(
    strategy_id,
    trading_day,
    feature_row=None,
    db_path=None,
) -> EligibilityReport:
    # 1. Parse strategy_id
    # 2. Look up ALL_FILTERS[filter_type]
    # 3. Fetch feature_row if not provided
    # 4. Resolve freshness (FRESH / PRIOR_DAY / STALE / NO_DATA)
    # 5. Call filter_instance.describe(row, session, entry_model)
    # 6. Convert AtomDescription[] to ConditionRecord[]
    # 7. Add overlay conditions via CalendarOverlay + ATRVelocityFilter.describe()
    # 8. Return immutable EligibilityReport
```

No regex parsing. No decomposition registry. No re-encoded filter logic. Just delegation.

## What this eliminates

| Current file | Fate |
|--------------|------|
| `trading_app/eligibility/decomposition.py` | DELETED (~500 lines) |
| Regex patterns (`_ORB_G_PATTERN`, `_FAST_PATTERN`, etc.) | DELETED |
| Hardcoded `validated_for` tuples in decomposition.py | DELETED — filters know their own validity |
| Hand-assigned confidence tiers | DELETED — derived from filter's `@revalidated-for` annotation |
| `_compare()` type-error handling | DELETED — filters handle their own comparisons |
| `_resolve_pdr` / `_resolve_gap` / `_resolve_dow` derived-value resolvers | DELETED — each filter's `describe()` computes its own derived values |
| Silent divergence risk on new filters | ELIMINATED — new filters get eligibility support by implementing `describe()` |
| NaN silent FAIL bug | FIXED — filters already use `pd.notna()` in their `matches_df()`; the new `describe()` handles NaN consistently |
| Tests of re-encoded logic | REPLACED by tests of `filter.describe()` per class |

## What survives

- `trading_app/eligibility/types.py` — `EligibilityReport`, `ConditionRecord` data classes (these are consumer contracts, not logic)
- `trading_app/eligibility/__init__.py` — re-export
- Status enum (PASS, FAIL, PENDING, DATA_MISSING, NOT_APPLICABLE_*, RULES_NOT_LOADED, STALE_VALIDATION) — these describe outputs, not inputs
- Freshness detection logic (STALE, PRIOR_DAY, NO_DATA) — operates on the report, not on filter values
- Calendar RULES_NOT_LOADED detection — operates on file state, not filter logic

## Blast radius

Files touched:

| File | Change | Reason |
|------|--------|--------|
| `trading_app/config.py` | Add `AtomDescription` type + `describe()` method on `StrategyFilter` base + overrides on ~17 filter subclasses | Canonical source of filter logic. NEVER_TRIVIAL protected. |
| `trading_app/eligibility/builder.py` | Rewrite as thin adapter | Was ~700 lines, becomes ~200 |
| `trading_app/eligibility/decomposition.py` | DELETE | Parallel-model source of drift |
| `trading_app/eligibility/types.py` | Minor cleanup — `ResolvesAt` and `ConditionCategory` become re-exports from config.py or their own location | Consumer contract, logic lives elsewhere |
| `tests/test_trading_app/test_eligibility_decomposition.py` | DELETE | Tested the parallel model |
| `tests/test_trading_app/test_eligibility_builder.py` | Rewrite fixtures to use filter instances | Thin adapter has a smaller surface to test |
| `tests/test_trading_app/test_config.py` | ADD tests for `describe()` per filter class | Each override needs coverage |
| `pipeline/check_drift.py` | ADD check #N — every `ALL_FILTERS` entry must produce ≥1 atom from a sample row | Prevents regression to parallel model |

**Non-touched:** strategy_discovery.py, strategy_validator.py, outcome_builder.py — they use `matches_row()` / `matches_df()`, unchanged. `describe()` is additive.

## Risks

1. **Touching config.py is NEVER_TRIVIAL.** Full stage gate applies. Each filter override is a chance to introduce a regression in `matches_row()` if I'm not careful.
   - **Mitigation:** do not modify `matches_row()` / `matches_df()` — only ADD `describe()` as a new method. Existing test suite for strategy discovery will catch any regression.

2. **Silent default implementations.** If a subclass forgets to override `describe()` but has composite behavior, the base class default returns one atom. That's wrong for a composite but not obviously broken.
   - **Mitigation:** drift check #N iterates `ALL_FILTERS` and ensures every entry returns a non-empty atom list. A follow-up could verify the atom count matches expected for composites.

3. **Backward compatibility.** Phase 1 eligibility tests pass against the current hand-coded model. Rewriting breaks those tests.
   - **Mitigation:** accept the breakage — those tests were testing the parallel model. New tests target the filter-class `describe()` method directly.

4. **Blast radius of config.py.** Every Python file that imports config.py indirectly depends on the new method. If the method signature is wrong, many tests fail.
   - **Mitigation:** default implementation on base class means additive change — no existing caller breaks.

## What I will NOT do (scope discipline)

- Touch `strategy_discovery.py` / `strategy_validator.py` / `outcome_builder.py` — their use of `matches_row()` is unchanged
- Refactor the filter class hierarchy — just add methods
- Change `ALL_FILTERS` structure or keys
- Touch any schema (init_db.py, daily_features, validated_setups)
- Implement Phase 2 (trade sheet) or Phase 3 (dashboard) — those wait until this refactor lands

## Acceptance criteria

1. `trading_app/config.py` has an `AtomDescription` class and a `describe()` method on `StrategyFilter` base with a working default.
2. Every filter subclass in `ALL_FILTERS` has either a working override or uses the default correctly.
3. `trading_app/eligibility/builder.py` calls only `filter_instance.describe()` and canonical overlay functions — no re-encoded comparison logic.
4. `trading_app/eligibility/decomposition.py` is deleted.
5. `tests/test_trading_app/test_eligibility_decomposition.py` is deleted.
6. All existing filter tests in `test_config.py` still pass (no regression to `matches_row()`).
7. New tests cover `filter.describe()` for each non-trivial subclass.
8. New tests cover the thin-adapter builder.
9. Drift check asserts every `ALL_FILTERS` entry produces ≥1 atom from a sample row.
10. NaN values in feature rows surface as DATA_MISSING (not FAIL) — verified by test.
11. ATR velocity with `atr_vel_regime=None` matches canonical behavior (warm-up fail-open) — verified by test.
12. CONT and FAST atoms on E2 strategies both return NOT_APPLICABLE_ENTRY_MODEL — verified by test.
13. `pipeline/check_drift.py` passes.
14. Self-review produces no new HIGH findings.

## Why this is "proper" and not "just another iteration"

- **Single source of truth.** Filter logic has ONE location: the filter class. No parallel model.
- **New filters get eligibility for free.** Adding a filter to `ALL_FILTERS` requires implementing `describe()` — this is the ONE and ONLY place you add decomposition logic.
- **Canonical drift is mechanically impossible.** The eligibility builder cannot diverge from `matches_row()` because it no longer re-encodes anything.
- **Dead code cannot accumulate.** There's no decomposition registry to go stale. No regex patterns to maintain. No hardcoded `validated_for` tuples.
- **Review cycles shrink.** Self-review of future work cannot find re-encoding divergences because there is no re-encoding. Reviews focus on per-filter `describe()` correctness, a scoped and local question.
- **Grounded in the codebase's existing architecture.** StrategyFilter already has `matches_row()` and `matches_df()`. Adding `describe()` follows the same pattern. It is not a new abstraction — it is the completion of the existing one.

This is the long-term survivable design. The current `decomposition.py` approach was the wrong factoring, identified in review, corrected at the root.
