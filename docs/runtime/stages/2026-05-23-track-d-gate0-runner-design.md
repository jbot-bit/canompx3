---
task: |
  DESIGN — Track D MNQ COMEX_SETTLE Gate 0 runner.
  Design the Databento top-of-book table schema and bounded research runner
  needed to execute the DESIGN_ONLY prereg at
  docs/audit/hypotheses/2026-04-23-mnq-comex-settle-gate0-microstructure-v1.yaml.
  Three required_before_execution gates must be satisfied:
    1. Databento top-of-book event tables implemented + populated (DB schema)
    2. A bounded runner written and reviewed (research/run_gate0_comex_settle.py)
    3. No 2026 OOS threshold tuning (holdout 2026-01-01 remains sacred)
  This stage is DESIGN ONLY — no code is written until design is approved.
mode: DESIGN
updated: 2026-05-23T14:00Z
scope_lock:
  - pipeline/db_contracts.py
  - pipeline/migrations/add_microstructure_tables.sql
  - research/gate0_comex_settle.py
  - research/gate0_features.py
  - tests/test_pipeline/test_microstructure_tables.py
  - tests/test_research/test_gate0_comex_settle.py
---

## Prior Art

Archived design doc: `docs/plans/archive/2026-04/2026-04-23-microstructure-gate0-design.md` (status: archived, not superseded).
Existing TBBO module: `research/databento_microstructure.py` (slippage-oriented, TBBO schema, MGC/GC only — different purpose).
Pre-reg: `docs/audit/hypotheses/2026-04-23-mnq-comex-settle-gate0-microstructure-v1.yaml` (K=4, IS-only, holdout 2026-01-01).

## Design Proposal

### What and Why

Gate 0 tests whether pre-break top-of-book imbalance predicts COMEX_SETTLE breakout follow-through.
The pre-reg is locked (K=4, frozen family). It cannot run because:
- No `microstructure_windows` manifest table exists in gold.db
- No `micro_mbp1_events` / `micro_tbbo_events` tables exist
- No feature extract (`micro_gate0_features`) table exists
- No runner that pulls Databento MBP-1 windows, computes OFI/TBI/QI, and runs the K=4 family exists

### Files to Touch

1. **`pipeline/db_contracts.py`** — add DDL strings for 4 new tables:
   - `microstructure_windows` (window manifest, idempotent by `window_id`)
   - `micro_tbbo_events` (TBBO ticks linked to window)
   - `micro_mbp1_events` (MBP-1 ticks linked to window)
   - `micro_gate0_features` (frozen feature extracts per candidate)

2. **`pipeline/migrations/add_microstructure_tables.sql`** — NEW: one-time DDL migration (CREATE TABLE IF NOT EXISTS).

3. **`research/gate0_features.py`** — NEW: pure feature computation from raw tick DataFrames.
   - `compute_ofi_60s(mbp1_df, touch_ts, direction)` → signed OFI
   - `compute_tbi_60s(tbbo_df, touch_ts, direction)` → signed TBI
   - `compute_qi_1s(mbp1_df, touch_ts, direction)` → signed queue imbalance
   No Databento imports here — pure pandas/numpy over already-fetched DataFrames.

4. **`research/gate0_comex_settle.py`** — NEW: bounded runner for the Gate 0 prereg.
   - `--estimate-cost` flag (no data download, just prints expected Databento pull cost)
   - `--pull-windows` flag (downloads MBP-1 + TBBO for IS candidate windows, caches locally)
   - `--compute-features` flag (reads cached ticks, writes `micro_gate0_features`)
   - `--run-falsification` flag (runs K=4 family vs parent, prints IS results, no OOS touch)
   - Hard abort if `trading_day >= 2026-01-01` is detected in any feature computation
   - Reads `orb_outcomes` for candidate windows; reads `daily_features` for context
   - REPORT_ONLY: does not write to `validated_setups`, `experimental_strategies`, or any live path

5. **`tests/test_pipeline/test_microstructure_tables.py`** — schema contract tests (DDL round-trip).
6. **`tests/test_research/test_gate0_comex_settle.py`** — mock-tick unit tests for OFI/TBI/QI compute functions.

### Blast Radius

- `pipeline/db_contracts.py` — additive only (4 new table DDL strings). Existing tables untouched.
- `pipeline/migrations/add_microstructure_tables.sql` — NEW file, no callers yet. Run manually once.
- `research/gate0_features.py` — NEW, no callers in production code.
- `research/gate0_comex_settle.py` — NEW, no callers in production code. Research-only.
- `tests/` — NEW test files. Additive.
- No `pipeline/`, `trading_app/`, or `scripts/` production logic touched.
- No DB schema mutation to existing tables.
- Capital class: NONE.

### Approach

1. Derive candidate windows from canonical `orb_outcomes` (IS only: `trading_day < 2026-01-01`).
   Filter: `symbol='MNQ'`, `session='COMEX_SETTLE'`, `orb_minutes=5`, `entry_model='E2'`, `rr_target=1.5`.
   Window per candidate: `[touch_ts_utc - 60s, touch_ts_utc + 5s]` for pre-touch features.

2. Databento pull: use `mbp-1` schema for OFI and QI; use `tbbo` schema for TBI.
   Both use `GLBX.MDP3` dataset, `MNQ.FUT` symbol, `parent` stype. Cache `.dbn.zst` locally under
   `research/data/gate0_cache/` (mirroring the existing tbbo_pilot pattern).

3. Feature computation (IS-calibrated thresholds):
   - `signed_ofi_60s_high`: top-quartile signed OFI over the IS sample
   - `signed_tbi_60s_high`: top-quartile signed TBI over the IS sample
   - `signed_qi_last_1s_high`: top-quartile signed QI (last 1s before touch) over the IS sample
   - `signed_ofi_60s_high AND signed_qi_last_1s_high`: intersection of the above two

4. Falsification: compare filtered vs parent on `pnl_r` (from `orb_outcomes`).
   Metrics: `policy_ev_per_opportunity_r` (parent N × mean_pnl vs filtered N × mean_pnl),
   WR, mean_pnl_r, N. IS only. OOS gate deferred until IS shows signal worth testing.

### Open Questions Before Implementation

A. **Cost envelope**: how many IS candidate windows exist for MNQ COMEX_SETTLE O5 E2 RR1.5?
   This determines pull cost. Need to query `orb_outcomes` to get N before coding the runner.
   (Can run `--estimate-cost` without any Databento API call.)

B. **MBP-1 vs TBBO availability for MNQ**: The existing pilot only covers MGC/GC.
   Need to confirm Databento has MBP-1 for MNQ/NQ for the full IS window (2019-2025).

C. **`touch_ts_utc` precision**: Does `orb_outcomes` have millisecond-level touch timestamps,
   or only bar-resolution (1-minute)? If bar-resolution only, the 60s lookback window may overlap
   with the break bar itself — need to define the window boundary precisely.

## Acceptance (for implementation stage)

- `python pipeline/migrations/add_microstructure_tables.sql` runs idempotently
- `python research/gate0_comex_settle.py --estimate-cost` prints N windows + Databento cost estimate
- All 4 tables creatable from `db_contracts.py` DDL strings
- Tests green: `pytest tests/test_pipeline/test_microstructure_tables.py tests/test_research/test_gate0_comex_settle.py`
- `python pipeline/check_drift.py` passes
- Feature computation unit-tested against synthetic tick data with known OFI/TBI/QI values

## Out of Scope

- Actual Databento data download (requires API key + budget approval)
- `micro_gate0_labels` table (feature labels from `orb_outcomes` — trivial join, deferred)
- `micro_mbo_events` table (MBO escalation layer — only if Gate 0 shows signal)
- Any write to `validated_setups`, `experimental_strategies`, or `live_config`
- OOS falsification run
