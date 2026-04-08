---
task: Phase 3a — data_era foundation (canonical-data-redownload plan)
mode: IMPLEMENTATION
slug: phase-3a-data-era
agent: claude-code
created: 2026-04-08T04:15:00Z
updated: 2026-04-08T04:15:00Z
plan: docs/plans/2026-04-07-canonical-data-redownload.md
parent_branch: main @ 6d7345d
---

## Purpose

Phase 3 of the canonical-data-redownload plan rebuilds downstream layers
(orb_outcomes, daily_features) on the clean real-micro `bars_1m` data that
Phase 2 ingested (commit `82e8b60`). Phase 3a is the **foundation stage** —
adds the canonical helpers that 3b/3c/3d will consume to distinguish PARENT
(proxy) vs MICRO (real) data for each instrument + trading day.

Zero behavior change at the pipeline level. Zero data writes. Zero
discovery runs. Zero consumer wire-up. Pure foundation: new config field
+ new pure-function module + tests.

## What Phase 3a delivers

### 1. `parent_symbol` field on ASSET_CONFIGS

Add `parent_symbol: str | None` to every entry in
`pipeline.asset_configs.ASSET_CONFIGS`:

| Instrument | parent_symbol | Reason |
|---|---|---|
| MGC, MNQ, MES | `"GC"`, `"NQ"`, `"ES"` | Active micros with canonical parent |
| M2K, MBT, M6E, MCL, SIL | `"RTY"`, `"BTC"`, `"6E"`, `"CL"`, `"SI"` | Dead micros that use parent data |
| NQ, ES, GC | `None` | Parent contracts — they ARE the root |
| 2YY, ZT | `None` | Research-only native contracts |

This eliminates the "is this a micro?" derivation ambiguity. Explicit over
implicit. Single source of truth.

### 2. `pipeline/data_era.py` — new module (~80 lines)

Five pure public functions:

```python
DataEra = Literal["PARENT", "MICRO"]

def is_micro(instrument: str) -> bool:
    """True iff instrument has a non-None parent_symbol."""

def parent_for(instrument: str) -> str | None:
    """Parent symbol, or None if instrument is a parent/native itself."""

def micro_launch_day(instrument: str) -> date:
    """minimum_start_date. Raises ValueError if not a micro."""

def era_for_source_symbol(instrument: str, source_symbol: str) -> DataEra:
    """Classify a bars_1m row. Matches against instrument's outright_pattern
    first (MICRO if instrument is micro, PARENT otherwise); falls back to
    parent's outright_pattern (always PARENT). Raises on no match — catches
    post-Phase-2 corruption that got past relabeling."""

def era_for_trading_day(instrument: str, trading_day: date) -> DataEra:
    """Classify an orb_outcomes/daily_features row by trading_day >=
    micro_launch_day. Raises ValueError if instrument is not a micro
    (caller should is_micro() first)."""
```

Single source of truth: `pipeline.asset_configs.ASSET_CONFIGS`. Reuses
`get_outright_root()` from Move C where possible.

Fail-closed on every failure mode:
- Unknown instrument → ValueError
- `source_symbol` is None/empty → ValueError
- source_symbol matches neither instrument's own pattern nor parent's pattern → ValueError
- `micro_launch_day` called on non-micro → ValueError
- `era_for_trading_day` called on non-micro → ValueError

### 3. Tests

- `tests/test_pipeline/test_data_era.py` (new, ~150 lines):
  - `test_is_micro_for_all_instruments` — coverage matrix for 13 configs
  - `test_parent_for_all_instruments` — coverage matrix
  - `test_micro_launch_day_matches_minimum_start_date` — derived correctly
  - `test_era_for_source_symbol_happy_path` — 13 known-good mappings
  - `test_era_for_source_symbol_detects_phase_2_corruption` — MNQ config with source="NQH4" → PARENT (the exact corruption Phase 2 fixed)
  - `test_era_for_source_symbol_rejects_unrelated` — MNQ config with source="ESH4" → ValueError
  - `test_era_for_source_symbol_rejects_null_empty` → ValueError × 2
  - `test_era_for_trading_day_boundaries` — day-before/day-of/day-after launch
  - `test_era_for_trading_day_non_micro_raises` → ValueError for NQ/ES/GC/2YY/ZT
  - `test_micro_launch_day_non_micro_raises`
  - `test_unknown_instrument_raises` × all 5 public functions

- `tests/test_pipeline/test_asset_configs.py` — add `TestParentSymbol` class:
  - `test_all_configs_have_parent_symbol_field` — structural
  - `test_active_micros_map_to_parent` — MNQ→NQ, MES→ES, MGC→GC
  - `test_dead_micros_map_to_parent` — M2K→RTY, MBT→BTC, M6E→6E, MCL→CL, SIL→SI
  - `test_parents_and_research_only_have_none` — NQ/ES/GC/2YY/ZT → None

## Scope Lock

- pipeline/asset_configs.py
- pipeline/data_era.py

(Test files NOT in scope_lock — `tests/` is in `SAFE_DIRS` per stage-gate-guard.)

## Blast Radius

**Direct edits:** 2 production files (1 existing, 1 new).

**Additive only — no existing behavior changes:**
- New `parent_symbol` field on `ASSET_CONFIGS`: 87 files reference
  `ASSET_CONFIGS[` or `get_asset_config(` per grep, **none iterate over
  fixed key sets**; all use specific key access. New key is invisible to
  existing readers.
- New module `pipeline/data_era.py`: zero current consumers. Wire-up is
  Stage 3b/3c/3d's job.

**Drift check coverage:** 85 checks currently passing. None inspect
`parent_symbol` field or `pipeline/data_era.py`. Expected post-edit state:
85/0/7 unchanged.

**Companion test files** (in `tests/`, not scope-gated):
- `tests/test_pipeline/test_data_era.py` (NEW)
- `tests/test_pipeline/test_asset_configs.py` (extended)

**Mode A holdout integrity:** Phase 3a is pure code + tests. Zero data
writes, zero `validated_setups` touches, zero discovery runs, zero
`strategy_discovery` invocation. `HOLDOUT_SACRED_FROM` unchanged. Mode A
not at risk.

**Out of scope (explicit — deferred to later stages):**
- Stage 3b: `StrategyFilter.requires_micro_data` attribute in
  `trading_app/config.py` + wire-up
- Stage 3c: DELETE + rebuild `orb_outcomes` and `daily_features` from new
  `bars_1m` (DB write operation, gated separately)
- Stage 3d: New drift check enforcing era discipline on `validated_setups`
- Phase 4: Clean rediscovery with `--holdout-date 2026-01-01` + pre-registered
  hypothesis file
- Phase 5: Deploy decision re-classify lanes against 12 Phase-0 criteria

## TDD Sequence

1. **Red:** write `test_data_era.py` with all test vectors → run → all
   fail (module import error)
2. **Green:** implement `pipeline/data_era.py` minimally → run →
   all `test_data_era` pass
3. **Red/extend:** add `TestParentSymbol` class to `test_asset_configs.py`
   → run → all fail (field missing on configs)
4. **Green:** add `parent_symbol` field to 13 entries in `asset_configs.py`
   → run → `TestParentSymbol` passes
5. **Re-verify:** full `pytest tests/test_pipeline/` → all pass (1170+
   from merge baseline + new 3a tests); `python pipeline/check_drift.py`
   → 85/0/7 unchanged; `python scripts/tools/audit_behavioral.py` → 7/7
6. **Self-review:** Bloomey-style diff read. Confirm zero canonical
   re-encoding, zero dead code, zero silent failures.
7. **Commit:** one commit covering field + module + tests. Descriptive
   message citing canonical-data-redownload plan.

## Done Criteria

- [ ] `pipeline/data_era.py` exists with 5 public functions
- [ ] `tests/test_pipeline/test_data_era.py` exists with ≥17 passing tests
- [ ] `pipeline/asset_configs.py` has `parent_symbol: str | None` on all
  13 configs
- [ ] `tests/test_pipeline/test_asset_configs.py::TestParentSymbol` passes
- [ ] `pytest tests/test_pipeline/` → all pass
- [ ] `python pipeline/check_drift.py` → 85/0/7 unchanged
- [ ] `python scripts/tools/audit_behavioral.py` → 7/7 clean
- [ ] `git diff` self-review finds no regressions
- [ ] One commit with descriptive message
