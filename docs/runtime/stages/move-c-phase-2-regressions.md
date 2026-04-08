---
task: Move C — Phase 2 regression sweep (canonical re-encoding cleanup)
mode: IMPLEMENTATION
slug: move-c-phase-2-regressions
agent: claude-code-terminal-b
created: 2026-04-08T02:50:00Z
updated: 2026-04-08T02:50:00Z
---

## Purpose

Phase 2 of canonical-data-redownload (commit `82e8b60`) replaced parent-futures
data with real-micro data in `bars_1m` and updated `pipeline/asset_configs.py`
patterns. **Three downstream consumers were not swept:**

1. `tests/test_pipeline/test_gc_mgc_mapping.py` — 4 stale assertions encoding
   the pre-Phase-2 invariants (verified failing via `pytest`):
   - `test_mgc_pattern_matches_gc_contracts` (asserts MGC pattern matches `GCM4`)
   - `test_mgc_pattern_rejects_mgc_contracts` (asserts MGC pattern does NOT match `MGCM4`)
   - `test_mgc_prefix_len_is_2` (asserts prefix_len == 2; now 3)
   - `test_es_stores_as_mes_symbol` (asserts ES.symbol == "MES"; now "ES")
2. `scripts/tools/refresh_data.py:35-42` — `DOWNLOAD_SYMBOLS = {"MGC": "GC", ...}`
   hardcoded dict. **Silent re-corruption hazard:** running
   `refresh_data.py --instrument MGC` would download GC parent contracts and
   ingest them as MGC, undoing Phase 2's fix.
3. `scripts/databento_daily.py:64-70` — `_DATABENTO_SYMBOLS = {"MGC": "GC.FUT", ...}`
   parallel dict. Cron-triggered. Pollutes MGC research archive with parent
   contract data on every daily run.

Both scripts violate **institutional rigor rule 4** (no canonical re-encoding) —
the parallel dicts duplicate state already present in
`pipeline.asset_configs.ASSET_CONFIGS[instrument].outright_pattern`.

## What Move C delivers

### 1. Canonical helper (in canonical source)

Add `get_outright_root(instrument: str) -> str` to
`pipeline/asset_configs.py`. Pure function. Single source of truth. Derives
the bare contract root prefix (e.g., `MGC`, `NQ`, `RTY`) from the existing
`outright_pattern` regex via `re.match(r"^\^(\w+?)\[", pattern)`.

Fail-closed: raises `ValueError` for unknown instruments OR non-canonical
patterns.

### 2. Replace both parallel dicts with helper consumers

- `scripts/tools/refresh_data.py`: delete `DOWNLOAD_SYMBOLS`. Both call sites
  (`download_dbn` line 84 + `main` line 312) call `get_outright_root(inst)`
  inside try/except for soft-skip semantics on unknown instruments.
- `scripts/databento_daily.py`: delete `_DATABENTO_SYMBOLS`. Build the
  `instrument → 'ROOT.FUT'` mapping at startup via dict comprehension calling
  the helper. Preserve fail-closed guard against missing
  `ACTIVE_ORB_INSTRUMENTS`.

### 3. Test cleanup

- `tests/test_pipeline/test_gc_mgc_mapping.py`: rewrite the 4 stale tests to
  reflect post-Phase-2 reality. Update class docstrings.
- `tests/test_pipeline/test_asset_configs.py`: add `TestGetOutrightRoot` class
  with assertions for all 13 configured instruments + failure cases.

## Scope Lock

- pipeline/asset_configs.py
- scripts/tools/refresh_data.py
- scripts/databento_daily.py

(Test files not listed — `tests/` is in `SAFE_DIRS` per stage-gate-guard, no
scope_lock entry needed.)

## Blast Radius

Direct edits: 3 production files. Two are non-core scripts. One is
`pipeline/asset_configs.py` (NEVER_TRIVIAL — adding a function only,
zero behavior change to existing readers).

Downstream consumer audit: 87 files reference `ASSET_CONFIGS[` or
`get_asset_config(` per grep. None iterate over keys expecting a fixed set;
all use specific key access. Adding a new top-level function is invisible to
existing readers.

Companion test files (in `tests/`, not gated):
- `tests/test_pipeline/test_gc_mgc_mapping.py` (4 stale tests fixed)
- `tests/test_pipeline/test_asset_configs.py` (new TestGetOutrightRoot class)

Drift checks: 84 currently passing. The new function is additive; no
existing drift check inspects function names or signatures of asset_configs.
Expected post-edit state: 84/0/7 unchanged.

Out of scope (deferred to other stages):
- `pipeline/data_era.py` — Stage 3a foundation, separate stage
- `parent_symbol` field on AssetConfig — Stage 3a
- `pipeline/ingest_dbn_mgc.py` legacy module reconciliation
- TopStep audit findings (other terminal owns)
- HANDOFF.md test backfill items 1+2 (other terminal's queue)

## TDD Sequence

1. **Red:** add `TestGetOutrightRoot` class with all assertions in
   `test_asset_configs.py` → run → all fail (helper doesn't exist).
2. **Green:** implement `get_outright_root` in `asset_configs.py` → run →
   all `TestGetOutrightRoot` pass.
3. **Red:** fix the 4 stale tests in `test_gc_mgc_mapping.py` to assert
   post-Phase-2 reality → run → all pass (production already matches).
4. **Green:** edit `refresh_data.py` to use helper, delete `DOWNLOAD_SYMBOLS`.
   Manually verify imports and behavior via `python -c "from scripts.tools
   import refresh_data; ..."`.
5. **Green:** edit `databento_daily.py` to use helper, delete
   `_DATABENTO_SYMBOLS`. Verify import + fail-closed guard still fires.
6. **Verify all:** `pytest tests/test_pipeline/ -v` → all green;
   `python pipeline/check_drift.py` → 84/0/7 unchanged.
7. **Self-review:** Bloomey-style read of `git diff`. Confirm zero new
   canonical re-encoding, zero dead code, zero silent failures.
8. **Commit:** descriptive message citing Phase 2 regression source and the
   3 fixed consumers.

## Done Criteria

- [ ] `tests/test_pipeline/test_gc_mgc_mapping.py` — all 22 tests pass
- [ ] `tests/test_pipeline/test_asset_configs.py` — `TestGetOutrightRoot`
  added, all pass
- [ ] `pipeline/asset_configs.py` — `get_outright_root` exposed, no other
  changes (no field additions, no behavior change)
- [ ] `scripts/tools/refresh_data.py` — `DOWNLOAD_SYMBOLS` removed, helper
  used at both call sites, fail-closed semantics preserved
- [ ] `scripts/databento_daily.py` — `_DATABENTO_SYMBOLS` removed, derived
  at module load, fail-closed guard against missing actives still fires
- [ ] `pytest tests/test_pipeline/` — full pipeline suite green
- [ ] `python pipeline/check_drift.py` — 84/0/7 unchanged
- [ ] `git diff` self-review — no new band-aids, no silent failures
- [ ] One descriptive commit
