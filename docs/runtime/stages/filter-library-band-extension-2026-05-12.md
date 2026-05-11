---
task: Extend OwnATRPercentileFilter + OvernightRangeFilter to support upper-bound bands (ATR_P30_75, OVNRNG_PCT_BAND_20_55)
mode: IMPLEMENTATION
slug: filter-library-band-extension-2026-05-12
created: 2026-05-12
updated: 2026-05-12
scope_lock:
  - trading_app/config.py
  - tests/test_config_filters.py
acceptance:
  - "OwnATRPercentileFilter accepts optional max_pct; existing instances unchanged (min_pct-only behaviour preserved)."
  - "OvernightRangeFilter accepts optional max_pct; existing instances unchanged."
  - "ALL_FILTERS gains ATR_P30_75 (min_pct=30, max_pct=75) and OVNRNG_PCT_BAND_20_55 (min_pct=20, max_pct=55)."
  - "describe() emits 2 atoms (min + max) when both bounds set; 1 atom when only min set (backward compat)."
  - "Companion tests cover: boundary inclusivity, NULL handling, backward compat for min_pct-only, atom-count for band vs single."
  - "python pipeline/check_drift.py passes."
  - "Self-review pass before commit (per institutional-rigor rule 1)."
---

## Blast Radius

- **trading_app/config.py** — extend 2 existing filter classes with optional `max_pct` field; register 2 new ALL_FILTERS entries. Backward-compat preserved because all existing instances pass only `min_pct` (positional or keyword) and the new field defaults to `None`.
- **tests/test_config_filters.py** — new test cases for band semantics; existing tests must continue to pass.
- **No schema change.** Both filters read existing daily_features columns (`atr_20_pct`, `overnight_range_pct`).
- **No pipeline change.** No build_daily_features.py edits.
- **No drift-check change.** No new check needed — the filter's `matches_row` is the canonical predicate; new fields are read transparently.
- **Reads:** daily_features (read-only via filter consumers). **Writes:** none.
- **Downstream consumers** of these filter classes (per Grep):
  - `trading_app/eligibility/*.py` — calls `StrategyFilter.matches_row()` and `describe()` polymorphically; new atoms surface naturally
  - `trading_app/strategy_discovery.py` — iterates ALL_FILTERS; new entries become testable filter cells
  - `trading_app/strategy_validator.py` — same; new filter_type strings become valid for validator gate
  - The hypothesis-loader / prereg validator (post `56f69f51`) — reads filter_type strings against ALL_FILTERS keys; new entries unblock the future preregs that reference them

## Why this scope (and not 5 filters)

External trading AI's framework specified 5 compound filters. Audit against canonical sources (queried `daily_features` columns 2026-05-12):

- `ATR_P30_75` — atr_20_pct exists. Code-only band extension to OwnATRPercentileFilter. **In scope.**
- `OVNRNG_PCT_BAND_20_55` — overnight_range_pct exists. Code-only band extension to OvernightRangeFilter. **In scope.**
- `CLEAN_AIR_R125_ATR035` — "current price" predicate ambiguous (entry bar = E2 look-ahead risk). **Deferred — needs external-AI disambiguation.**
- `OPEN_VS_PDMID_LONG/SHORT` — "session open" predicate ambiguous (entry bar = E2 look-ahead risk); `prev_day_mid` derivable inline. **Deferred — needs external-AI disambiguation.**
- `OVN_DRIFT_LT_06ATR` — requires `overnight_open` + `overnight_close` columns that don't exist in daily_features. **Deferred — separate pipeline stage to extend build_daily_features.py.**

Per institutional-rigor rule 8 (verify before claiming), shipping the 2 unambiguous filters tonight, queuing the 3 ambiguous ones as a follow-up question to the external AI, is the honest path. The MGC LONDON_METALS framework lane (the highest-priority of the 3 framework lanes) only needs the 2 band filters per the original spec — A1 unblocks that prereg.

## Implementation plan

### Step 1 — Extend OwnATRPercentileFilter
- Add `max_pct: float | None = None` (after `min_pct`)
- `matches_row`: if `max_pct is not None and val > max_pct: return False` (upper bound exclusive — symmetric with OrbSizeFilter's `<` upper bound)
- `matches_df`: same logic, vectorised
- `describe`: emit a second atom when `max_pct is not None` (comparator `"<="` vs `">"` decision matches matches_row exactly — review at code time)
- CONFIDENCE_TIER stays PLAUSIBLE

### Step 2 — Extend OvernightRangeFilter
- Same shape as Step 1
- Existing `strict_gt` field continues to govern the LOWER bound only
- Upper bound default semantic: `<=` (inclusive) — confirm at code time matches OrbSizeFilter precedent

### Step 3 — Register new ALL_FILTERS entries
- `ATR_P30_75` — `OwnATRPercentileFilter(filter_type="ATR_P30_75", description="Own ATR(20) percentile in [30, 75]", min_pct=30.0, max_pct=75.0)`
- `OVNRNG_PCT_BAND_20_55` — `OvernightRangeFilter(filter_type="OVNRNG_PCT_BAND_20_55", description="Overnight range percentile in [20, 55]", min_pct=20.0, max_pct=55.0)`
- Place in the section the existing single-bound entries occupy (MGC_VOLUME_FILTERS-adjacent block at line ~3136-3151 for ATR; OvernightRangeFilter location TBD at code time)

### Step 4 — Tests
- `tests/test_config_filters.py` (new file or append) covering:
  1. `ATR_P30_75` accepts row with `atr_20_pct=50.0` → True
  2. `ATR_P30_75` rejects row with `atr_20_pct=29.9` → False
  3. `ATR_P30_75` rejects row with `atr_20_pct=75.1` → False
  4. `ATR_P30_75` rejects row with `atr_20_pct=None` → False (fail-closed)
  5. `OVNRNG_PCT_BAND_20_55` mirror cases
  6. Existing `ATR_P30` (min_pct=30, max_pct=None) still behaves identically — regression
  7. `describe()` for band filter returns 2 atoms; `describe()` for single-bound returns 1 atom
  8. `matches_df` vectorised matches `matches_row` row-by-row for synthetic frame

### Step 5 — Self-review + drift + commit
- Read the diff end-to-end
- Run `pytest tests/test_config_filters.py -xvs`
- Run `pytest -k "ATR_P30 or OvernightRange or composite"` to catch downstream regression
- Run `python pipeline/check_drift.py`
- Commit on `feat/filter-library-extension-5-compound`

## Forbidden in this stage

- Adding new filter classes for band semantics (use existing classes — canonical-delegation rule 4)
- Modifying `pipeline/build_daily_features.py` (out of scope; that's the OVN_DRIFT follow-up)
- Adding `prev_day_mid` / `overnight_open` / `overnight_close` columns (out of scope)
- Touching `experimental_strategies` or `validated_setups` rows (out of scope)
- Writing preregs that reference the new filters (the 3 framework preregs are a follow-up after the deferred-3 disambiguation lands)

## Honest framing — what this stage does and does NOT prove

This stage adds two **testable predicates** to the filter library. It does NOT validate that they predict positive expectancy in live trading.

The framework's mechanism prior (compression-release setups need mid-vol regime, neither quiet nor extreme) is **plausible but not literature-grounded in `docs/institutional/literature/`**. The local extracts (Fitschen Ch 6, Chan 2013 Ch 7, Carver 2015 Ch 9-10) ground percentile-style ATR gates in general, but do not specifically validate "inverted U" / "mid-vol-band" semantics over "monotonic high-vol-only" or "monotonic low-vol-only" gates.

Path to live-trade value (per `pre_registered_criteria.md`):

1. ✅ Filters exist in ALL_FILTERS (this stage)
2. ⏳ External AI writes prereg with these filters + literature-grounded mechanism
3. ⏳ Prereg passes Criterion 1 (filter exists — now true)
4. ⏳ Pathway-A discovery scan with `--holdout-date 2026-01-01` (Mode A, sacred)
5. ⏳ Result must pass Criterion 8 OOS gate against strict Mode A baselines (per LLM trio audit lesson: do NOT compare against Mode B `validated_setups.oos_exp_r`)
6. ⏳ Lane promoted to `validated_setups` only on strict gate pass
7. ⏳ Lane-allocation review for deployment

**Lesson carried from LLM-prereg trio audit (2026-05-12):** 3/3 prior preregs REJECTED on Criterion 8 OOS despite Mode B baselines suggesting they would pass. The institutional pipeline worked as designed. This stage does not predict survival of the future band-filter preregs — it only unlocks the test.

## Cross-references

- Framework spec: external trading AI message in prior session (3-lane MGC/MNQ framework)
- Validator gate: `trading_app/strategy_validator.py::_check_prereg_present` (post `56f69f51`)
- Authority: `docs/institutional/pre_registered_criteria.md` Criterion 1 (filter must exist in ALL_FILTERS before prereg is testable)
- Existing patterns: `OrbSizeFilter` (band pattern at lines 535-614) — reference for upper-bound atom emission
- Sibling stage (deferred 3 filters): `docs/runtime/stages/filter-library-deferred-3-disambiguation-2026-05-12.md`
- Pre-validation discipline anchor: `docs/audit/results/2026-05-12-llm-prereg-trio-audit.md` (Mode B baseline contamination lesson)
