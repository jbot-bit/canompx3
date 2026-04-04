# Design: Adding a New Filter Type to the Pipeline

**Date:** 2026-04-04
**Mode:** plan
**Status:** Awaiting approval

---

## Turn 1: ORIENT

### Blueprint Routing
Blueprint §3 (Research Test Sequence) governs: any new filter must clear Gate 1 (mechanism),
Gate 2 (baseline viability), Gate 3 (BH FDR significance), and Gate 4 (walk-forward) before
entering the discovery grid. This is a trading logic change — authority is TRADING_RULES.md.

### NO-GO Registry Check
No blanket NO-GO on new filter types. Prior sessions that killed specific filter families:
L-filters (negative ExpR), ML filters (0/12 BH), blanket calendar skip (disproven),
DOW NOFRI/NOTUE (LIKELY NOISE, removed from grid). None prohibit adding a new, research-backed filter class.

### Existing Spec
docs/specs/presession-volatility-filters.md is the canonical reference pattern (Apr 2 2026).
No generic spec exists — this design establishes the pattern.

### Files and Roles

| File | Role |
|------|------|
| trading_app/config.py | Canonical filter authority: StrategyFilter base, ALL_FILTERS registry, get_filters_for_grid() |
| pipeline/build_daily_features.py | Computes daily_features columns that filters read |
| pipeline/init_db.py | DB schema for daily_features — new column migrations go here |
| pipeline/check_drift.py | Drift checks: check_config_filter_sync(), check_validated_filters_registered() (#29), check_discovery_session_aware_filters() (#28) |
| trading_app/strategy_discovery.py | Iterates get_filters_for_grid(), calls matches_row() |
| trading_app/setup_detector.py | Live pre-session: calls matches_row() for trade gating |
| trading_app/execution_engine.py | Live execution: looks up ALL_FILTERS.get(filter_type) — fail-closed if unknown |
| tests/test_trading_app/test_config.py | Behavioral tests for filter classes |

### One-Way Dependency
pipeline/ → trading_app/ only. Filter classes live in trading_app/config.py.
Pipeline computes data columns, never imports filter logic. Rule respected.

### Purpose
Without a defined pattern, adding a filter type is error-prone and causes silent failures:
- Unknown filter_type in ALL_FILTERS → fail-closed trade drop in execution_engine.py
- Filter reading a missing daily_features column → all-False, zero-trade strategies
- Key/filter_type mismatch → check_config_filter_sync() drift failure
- Filter added to discovery without research basis → inflates FDR family count K

---

## Turn 2: DESIGN

### Prior Failures (hard_lessons.md + presession spec)

1. NODBL look-ahead bug (Feb 2026): double_break column is look-ahead. 6 strategies were
   artifacts. Lesson: verify every column is strictly prior-day or pre-entry data.

2. DOW NOFRI/NOTUE noise (Mar 2026): removed from discovery grid but kept in ALL_FILTERS
   for DB row compatibility. Lesson: grid removal != registry removal. Decouple these.

3. OvernightRangeAbsFilter session routing (Mar 2026): overnight range is 09:00-17:00
   Brisbane. Applying to Asian sessions = look-ahead. Must be routed via get_filters_for_grid()
   only, never in BASE_GRID_FILTERS. Lesson: session routing is a required safety valve.

4. Vacuous absolute thresholds (filter audit Apr 3): 53% of absolute filters had thresholds
   that passed so many days they were functionally NO_FILTER. Lesson: calibrate pass rate
   before deployment, target 40-70%.

### Three Approaches

**Take 1 — Minimal (no new column)**
Filter reads existing daily_features columns only. Blast radius: config.py + 1 test file.
Suitable for: new threshold on existing column (ATR percentile cutoff, ORB size band).

**Take 2 — Standard (new column + filter class)**
Filter needs a new daily_features column. Blast radius: init_db.py + build_daily_features.py
+ config.py + 2 test files. Suitable for: genuinely new signal.

**Take 3 — Composite (CompositeFilter assembly)**
Combine two existing filters via the existing CompositeFilter class. Blast radius: config.py
only. No new column, no new class. Suitable for: AND/OR combinations of existing filters.

**Recommendation: design for Take 2 as the canonical pattern.** Takes 1 and 3 are subsets.

### Six-Layer Architecture

- Layer 1: Research gate (before any code) — mechanism + BH FDR + walk-forward
- Layer 2: Column layer (pipeline, if needed) — init_db.py schema + build_daily_features.py computation
- Layer 3: Filter class layer (config.py) — frozen dataclass, matches_row(), matches_df()
- Layer 4: Registry layer (config.py) — add to ALL_FILTERS with key == filter_type
- Layer 5: Grid routing layer (config.py) — session-specific routing in get_filters_for_grid()
- Layer 6: Test and drift layer — test_config.py + check_drift.py passes

---

## Turn 3: DETAIL

### Ordered Implementation Steps

**Step 0 — Research complete (precondition)**
Confirm mechanism, baseline viability (≥3 variable combinations), BH FDR p < 0.05,
walk-forward WFE > 0.5. Document in docs/specs/YYYY-MM-DD-<filter-name>.md.

**Step 1 — Schema (only if new column needed)**
File: pipeline/init_db.py
- Add column definition in daily_features CREATE TABLE block with doc comment
  (look-ahead status, research source)
- Add forward migration: ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS <col> <type>

**Step 2 — Computation (only if new column needed)**
File: pipeline/build_daily_features.py
- Add computation in Stage 6 (pre-session stats)
- Use only prior-day or pre-session data
- Write NULL during warm-up period (insufficient lookback)
- Use idempotent DELETE+INSERT pattern

**Step 3 — Filter class definition**
File: trading_app/config.py
- Define frozen dataclass subclassing StrategyFilter (before line ~856, after last existing class)
- matches_row(row, orb_label): row.get(col), return False if None (fail-closed), threshold logic
- matches_df(df, orb_label): vectorized, return False Series if column absent

**Step 4 — Instances and registry**
File: trading_app/config.py
- Add named instances in predefined filter sets section (~line 975) with siblings
- Add @research-source, @entry-models, @revalidated-for annotations as comments
- Add all instances to ALL_FILTERS with key == filter_type
- Do NOT add to BASE_GRID_FILTERS unless safe for every session/instrument

**Step 5 — Grid routing**
File: trading_app/config.py
- Inside get_filters_for_grid(), add routing block for validated (instrument, session) pairs
- Comment: research source + look-ahead safety status
- Pattern: check if (instrument, session) in _validated_pairs: add keys to filters dict

**Step 6 — Tests**
File: tests/test_trading_app/test_config.py
- New test class for the filter type
- Required: (a) pass case, (b) fail case, (c) None column → False, (d) missing key → False,
  (e) boundary value correct side, (f) key matches ALL_FILTERS key, (g) frozen mutation raises

**Step 7 — Drift check**
Run: python pipeline/check_drift.py
Confirm: check_config_filter_sync() passes, check_validated_filters_registered() passes,
check_discovery_session_aware_filters() passes.

**Step 8 — Pipeline rebuild (only if new column)**
Run build_daily_features.py for all affected instruments and date ranges.
Confirm: column non-NULL for expected range, NULL during warm-up period.

**Step 9 — Spec file**
Create/update docs/specs/YYYY-MM-DD-<filter-name>.md with: research origin, validated
sessions/instruments, filter instances and pass rates, what was NOT validated,
drift check impact, remaining action items.

### Execution Order
Steps 1-2 before Step 3 (class reads the column).
Steps 3-4 before Step 5 (routing references instances).
Step 4 before Step 6 (tests import from config.py).
Step 6 before Step 7 (tests are part of drift-pass definition).

### Rebuild Impact
- New column: daily_features rebuild required for all instruments from earliest date.
- No new column: no rebuild needed.
- orb_outcomes: never needs rebuild (filters applied post-hoc in discovery, not baked in).

---

## Turn 4: VALIDATE

### Failure Modes

| Risk | Guard |
|------|-------|
| Look-ahead contamination | Verify column uses only prior-day or pre-session open data. Check stage assignment in build_daily_features.py. |
| Silent all-false filter | Fail-closed in matches_row() + test case (d). |
| Key/filter_type mismatch | check_config_filter_sync() drift check + test case (f). |
| Vacuous filter (>90% pass rate) | Query pass rate from daily_features before deployment. Target 40-70%. |
| FDR family inflation | Document K increase in spec file. Re-run FDR with new K. |
| Overly broad grid routing | Never add session-restricted filters to BASE_GRID_FILTERS. |
| Column migration failure | Always add both CREATE TABLE column AND ALTER TABLE migration. |
| E2 exclusion interaction | Check new filter prefix does not match E2_EXCLUDED_FILTER_PREFIXES. |

### Tests That Prove Correctness

1. Fail-closed on missing column: omit column key from row dict, assert matches_row() → False.
2. Boundary precision: at exactly threshold value, assert correct side.
3. Drift check programmatic: check_config_filter_sync() returns zero violations.
4. Grid routing coverage: get_filters_for_grid() for validated pair has key; non-validated pair does not.
5. Strategy ID stability: new filter_type produces valid strategy ID format (no disallowed chars).
6. No look-ahead regression: permutation shuffle of column → ExpR collapses to near-zero.

### Rollback Plan

1. Remove filter instances from ALL_FILTERS and routing from get_filters_for_grid().
2. Filter class definition can remain (inert if not in ALL_FILTERS).
3. If discovery run was done: delete experimental_strategies rows for the new filter_type.
4. If new daily_features column was added: leave as inert column (no rebuild needed).
5. No DB rebuild required — only config.py change + fresh discovery run.

### Guardian Prompts

PIPELINE_DATA_GUARDIAN: applies if new daily_features column is added.
Full gate: read current schema → verify no look-ahead → add column + migration → rebuild → verify.

ENTRY_MODEL_GUARDIAN: applies if filter interacts differently with E1 vs E2.
E1 unfiltered is negative everywhere. A filter that appears to rescue E1 is almost certainly an artifact.
