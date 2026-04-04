# Design: Adding a New Filter Type to the Pipeline

**Date:** 2026-04-04
**Mode:** plan
**Status:** Awaiting approval

---

## Turn 1: ORIENT

### What This Is About

A "filter type" in this codebase is a named, immutable data container defined in
`trading_app/config.py` that:
1. Has a unique `filter_type` string key
2. Implements `matches_row()` — per-row check against a `daily_features` dict
3. Implements `matches_df()` — vectorized check against a DataFrame
4. Is registered in `ALL_FILTERS` — the single source of truth for all known filter types

**PURPOSE:** Without a properly registered filter, any strategy using that filter type will
silently fail discovery and validation. Drift check #29 (`check_validated_filters_registered`)
catches unknown `filter_type` strings in `validated_setups`. Adding a filter correctly means
it participates in discovery, is recognized at validation time, is persisted safely in the
database, and is applied correctly at fitness/paper-trading time.

### Authority Check

- `TRADING_RULES.md` governs all filter logic. ORB size filters (G4+) are the primary edge gate.
  No filter can reference unbuilt features or look-ahead data.
- `docs/STRATEGY_BLUEPRINT.md` §3 Gate 2: before adding a filter to the discovery grid, a
  research basis must exist (tested ≥3 values, BH FDR survived).
- No existing spec in `docs/specs/` covers a generic "new filter" procedure. The closest is
  `docs/specs/presession-volatility-filters.md` — a template of what a completed filter looks like.
- One-way dependency is enforced: filter definitions live in `trading_app/config.py`.
  `pipeline/` code does not import from `trading_app/` at ingest time.

### Canonical Sources Touched

- `trading_app/config.py` — all filter containers and `ALL_FILTERS` registry
- `pipeline/build_daily_features.py` — any new column the filter reads must be pre-computed here
- `pipeline/init_db.py` — any new column requires a schema addition here
- `pipeline/check_drift.py` — checks #22, #28, #29 all verify filter registry integrity
- `trading_app/strategy_fitness.py` — consumes `ALL_FILTERS`; has `isinstance` guard for
  `VolumeFilter` subclasses triggering `bars_1m` enrichment
- `trading_app/strategy_discovery.py` — consumes `get_filters_for_grid()`, calls `matches_df()`
- `tests/test_trading_app/test_config.py` — behavioral tests for filter containers

### Blast Radius

| File | Why Touched |
|---|---|
| `trading_app/config.py` | New container + ALL_FILTERS entry + get_filters_for_grid() routing |
| `tests/test_trading_app/test_config.py` | Behavioral tests for the new container |
| `pipeline/init_db.py` | Only if new column needed in daily_features schema |
| `pipeline/build_daily_features.py` | Only if new column needed (computation) |

Files that auto-validate (no changes, must pass after):
- `pipeline/check_drift.py` drift checks #22, #28, #29

---

## Turn 2: DESIGN — Multi-Take Deliberation

### What Has Gone Wrong Before

- **E0 purge (Feb 2026):** Entry model left in registry after purge caused silent discovery
  corruption. Lesson: registry completeness and drift check coverage matter.
- **NODBL filter removal:** `NO_DBL_BREAK` was in `ALL_FILTERS` with strategies referencing it.
  Removing it from the grid still required keeping the registry entry for DB row compatibility.
  Lesson: once a filter type is in `validated_setups`, the registry entry cannot be removed.
- **`double_break` look-ahead:** A `daily_features` column was used as a filter before a
  look-ahead audit caught that it relied on data from after session open. 6 validated strategies
  were artifacts. Lesson: every new column needs a look-ahead audit before grid inclusion.
- **Silent trade drops:** `strategy-awareness.md` explicitly states: "filter_type must match
  ALL_FILTERS exactly. Unknown strings = silent trade drops."

### Take 1 — Filter Reading an Existing Column (minimal blast radius)

Add only to `trading_app/config.py` and tests. Correct when the column already exists in
`daily_features` (e.g., a new threshold on `atr_20_pct`, `overnight_range`, `prev_day_range`).
This is the right approach for most new threshold variants on validated signals.

### Take 2 — Filter Requiring a New Column

Blast radius expands to `pipeline/init_db.py` + `pipeline/build_daily_features.py` + full
`daily_features` rebuild for all instruments. Required for genuinely new signals. Triggers the
look-ahead audit requirement before any research runs.

### Take 3 — Composite Filter on Existing Filters

Some filters combine two signals (e.g., `CombinedATRVolumeFilter` extends `VolumeFilter`).
Correct when the signal is the intersection of two already-validated signals. Inheritance
creates an `isinstance` guard dependency in `strategy_fitness.py`.

### Recommendation

**Take 1 (minimal, column already exists) is the right default.** The steps below cover the
general pattern with explicit branch points for the "new column required" case.

### Data Flow

```
daily_features column (existing or newly computed)
  -> row dict at discovery/fitness time
    -> new filter's matches_row() / matches_df()
      -> boolean: trade eligible or not
        -> ALL_FILTERS registry (key == filter_type string)
          -> get_filters_for_grid(instrument, session) routing
            -> strategy_discovery.py grid iteration
              -> validated_setups rows (filter_type stored as string)
                -> strategy_fitness.py (looks up by key from ALL_FILTERS)
```

### One-Way Dependency Verification

- New container definition: `trading_app/config.py` (correct layer)
- `pipeline/build_daily_features.py` does NOT import from `trading_app/`
- `pipeline/init_db.py` does NOT import from `trading_app/`
- `trading_app/strategy_fitness.py` imports from `trading_app/config.py` — correct direction
- No violation of the one-way rule

---

## Turn 3: DETAIL — Ordered Implementation Steps

### Step 1: Verify the source column exists in `daily_features`

Before writing any code, confirm the exact column name by running a schema query against
`gold.db`. Do not assume from docs. Verify the column is non-null at an acceptable rate for
the target instrument/session combinations.

### Step 2: Add the immutable data container to `trading_app/config.py`

- Place after the most semantically related existing container
- The container must be frozen (immutable) with at minimum: `filter_type` (string, exactly
  matching the future registry key), `description` (string), and threshold parameter(s)
- Implement `matches_row()`: look up the column from the row dict, fail-closed if None
  (missing data = ineligible day, return False), apply threshold logic, return bool
- Implement `matches_df()`: look up the DataFrame column, fail-closed if column absent
  (return all-False Series), apply vectorized threshold, return boolean Series
- Add `@research-source`, `@entry-models`, `@revalidated-for` annotations in the docstring
  (drift check #45 enforces these for any filter with research backing)

### Step 3: Register in `ALL_FILTERS` in `trading_app/config.py`

- Add the new filter instance to the `ALL_FILTERS` dict
- Dict key must exactly equal the `filter_type` field — drift check #22 verifies this
- Group with semantically related filters

### Step 4: Route in `get_filters_for_grid()` in `trading_app/config.py`

- Identify which instruments and sessions have a research basis for this filter
- Add a conditional block that adds the new filter key(s) only for those instrument/session
  combinations
- Do NOT add to `BASE_GRID_FILTERS` unless valid for all sessions and all instruments
- For filters with look-ahead risk on certain sessions, document which sessions are excluded
  and why (see `OvernightRangeAbsFilter` docstring as a pattern)

### Step 5 (only if new column required): Update `pipeline/init_db.py`

- Add the new column to the `daily_features` table schema
- Place after related columns with appropriate NULL handling
- This is a schema change — triggers PIPELINE_DATA_GUARDIAN review

### Step 6 (only if new column required): Update `pipeline/build_daily_features.py`

- Add computation in Stage 6 (pre-session stats and derived features)
- Audit look-ahead: the column value must be strictly knowable before the earliest session
  this filter will be applied to
- Computation must be idempotent (DELETE+INSERT pattern)

### Step 7: Add behavioral tests in `tests/test_trading_app/test_config.py`

- New test group following the pattern of existing groups (e.g., `TestOrbSizeFilter`)
- Required tests: matches when condition met, no match when condition not met, fail-closed
  on None, fail-closed on missing DataFrame column, frozen, hashable, JSON round-trip
- Also verify the new key is present in `ALL_FILTERS`
- Verify `get_filters_for_grid()` includes the key for the target instrument/session

### Step 8 (only if new column required): Rebuild `daily_features` for all active instruments

- Run `build_daily_features.py` for all instruments (MNQ, MGC, MES) and all ORB apertures
  (5, 15, 30 minutes)
- Verify row counts before and after
- Verify new column populated at expected non-null rate

### File Summary

| File | Change | Conditional |
|---|---|---|
| `trading_app/config.py` | New container + ALL_FILTERS + get_filters_for_grid routing | Always |
| `tests/test_trading_app/test_config.py` | Behavioral tests | Always |
| `pipeline/init_db.py` | New column in schema | Only if new column needed |
| `pipeline/build_daily_features.py` | Column computation | Only if new column needed |

---

## Turn 4: VALIDATE

### Failure Modes and Risks

1. **filter_type key mismatch** — dict key in `ALL_FILTERS` not equal to `filter_type` on the
   instance. Drift check #22 fails. Filter invisible to portfolio lookups and fitness scoring.
   Mitigation: test JSON round-trip and key equality.

2. **Look-ahead contamination** — filter reads a column computed using data after session open.
   Strategies appear to work in-sample, fail OOS. How the `double_break` artifact happened.
   Mitigation: explicit per-session look-ahead audit before routing in `get_filters_for_grid()`.

3. **Missing fail-closed logic** — `matches_row()` returns True when column is NULL. Strategies
   trade on days with no signal data, inflating trade count and distorting metrics.
   Mitigation: explicit None check returning False in every filter. Verified by tests.

4. **Registry orphan after removal** — once strategies referencing a filter type are in
   `validated_setups`, removing the key from `ALL_FILTERS` triggers drift check #29 failures
   indefinitely. Never remove — deprecate and remove from `get_filters_for_grid()` routing only.

5. **`isinstance` guard miss** — `strategy_fitness.py` uses `isinstance(filter, VolumeFilter)`
   to decide whether to trigger `bars_1m` enrichment. A filter reading runtime-computed data
   (not stored in `daily_features`) needs enrichment logic, either via subclassing or a new
   guard. Mitigation: if the filter reads only stored `daily_features` columns, no enrichment
   needed. If it needs runtime computation, review `strategy_fitness.py` enrichment paths.

6. **Session routing mistakes** — adding a look-ahead-risky filter to sessions where it uses
   future data. Mitigation: document the look-ahead window explicitly in the container docstring
   and only route to safe sessions in `get_filters_for_grid()`.

### Tests That Prove Correctness

1. `test_new_filter_matches_when_above_threshold()` — returns True when column value meets condition
2. `test_new_filter_no_match_when_below_threshold()` — returns False when condition not met
3. `test_new_filter_fail_closed_on_none()` — `matches_row({}, "SESSION")` returns False, no crash
4. `test_new_filter_fail_closed_df_missing_column()` — `matches_df(df_without_column, "SESSION")` returns all-False
5. `test_new_filter_frozen()` — setting any attribute raises AttributeError
6. `test_new_filter_hashable()` — two instances with same params have equal hash, deduplicate in set
7. `test_new_filter_json_roundtrip()` — `json.loads(f.to_json())["filter_type"]` equals expected key
8. `test_new_filter_in_all_filters()` — key is present in `ALL_FILTERS`
9. `test_get_filters_for_grid_routes_correctly()` — new key present for target instrument/session,
   absent for sessions where filter should not apply

### Rollback Plan

- Before any strategies are discovered with the new filter: removing the container and the
  `ALL_FILTERS` entry from `config.py` is zero-impact.
- After strategies are promoted to `validated_setups`: keep the registry entry, remove the
  filter from `get_filters_for_grid()` to stop new discovery, add a deprecation comment.
- If a schema column was added: dropping the column requires ALTER TABLE and a full rebuild.
  Plan schema additions carefully — they are the hardest to rollback.

### Guardian Prompts Needed

- **PIPELINE_DATA_GUARDIAN** — required if `pipeline/init_db.py` or
  `pipeline/build_daily_features.py` are touched
- **ENTRY_MODEL_GUARDIAN** — not needed (filter addition only, no entry model changes)

---

## Approval Checkpoint

The above is a plan only. No code has been written.

Approve with "go", "looks good", "do it", or "approved" to proceed to implementation.
On approval, `STAGE_STATE.md` will be written with IMPLEMENTATION mode and scope_lock
from Turn 3 Step 7 (config.py + test file for the minimal case).
