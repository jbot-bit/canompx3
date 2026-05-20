---
task: filter_utils silent-fail-closed parity bug — add required_columns() contract to StrategyFilter base + wrapper pre-validation
mode: IMPLEMENTATION
slug: filter-utils-required-columns-contract
created: 2026-05-20
session: session/joshd-filter-utils-parity-bug
---

## Scope Lock

- trading_app/config.py
- research/filter_utils.py
- tests/test_research/test_filter_utils.py

## Per-file Plan

trading_app/config.py: add StrategyFilter.required_columns() base method + overrides on CostRatioFilter, CrossAssetATRFilter, VolumeFilter. Default returns empty set (no behavioral change for 23/26 filter classes).

research/filter_utils.py: wrapper calls filt.required_columns(orb_label) before matches_df; raises ValueError listing missing columns when caller's DataFrame is incomplete.

tests/test_research/test_filter_utils.py: new equivalence tests for COST_LT12 + matching missing-column ValueError tests. Asserts the prior-session CHECK 3 repro (0/N silent fail) now raises.

## Blast Radius

- trading_app/config.py — touches StrategyFilter base class (matches_df-class filters inherit). New method has a safe default `return set()` so 23 of 26 filter classes (those without cross-row prerequisites) need no override. Three classes get explicit overrides: CostRatioFilter (needs `symbol` + `orb_{label}_size`), VolumeFilter (needs `rel_vol_{label}`), CrossAssetATRFilter (needs `symbol` + sibling-instrument `atr_20`). Zero behavioral change to matches_df/matches_row itself; the new method is purely declarative.
- research/filter_utils.py — wrapper becomes STRICTER: callers that previously got `0/N` silent failures now get ValueError. Reviewed all 47 wrapper callers (`grep -rln "filter_signal" research/ tests/`); the only active COST_LT* caller (`research/yordanov_crossmiss_triage_v1.py:163`) passes a DataFrame derived from `daily_features.*` which always includes `symbol` via the triple-join. Will not break.
- tests/test_research/test_filter_utils.py — additive only (new test classes), no edits to existing 13 tests.
- Reads: gold.db (read-only by tests). Writes: none.
- Production callers (`trading_app/`, `pipeline/`, `scripts/`): ZERO. Verified via `grep -rln "filter_signal" trading_app/ pipeline/ scripts/` → empty.

## Done criteria

1. Repro from prior-session CHECK 3 now raises ValueError instead of returning all-False.
2. `pytest tests/test_research/test_filter_utils.py` green.
3. `python pipeline/check_drift.py` green.
4. yordanov_crossmiss_triage_v1.py still works (smoke test or static read confirms `symbol` is present).
5. Memory feedback file added documenting this as n=1 silent-fail-closed wrapper-contract bug; class threshold per n3-doctrine.

## Origin

Prior session CHECK 3 (commit `b3744787` neighborhood): `filter_signal(df, 'COST_LT12', 'NYSE_OPEN')` returned `0/1718 pass` vs inline reference `1695/1718 pass`. Root cause: CostRatioFilter.matches_df (trading_app/config.py:644) silently returns `pd.Series(False, ...)` when the input DataFrame is missing the `symbol` column it needs for cross-row cost lookup. The wrapper faithfully propagates the silent failure. Violates institutional-rigor rule 6 (no silent failures). Same class affects CrossAssetATRFilter and VolumeFilter.
