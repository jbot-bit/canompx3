---
status: accepted_design
owner: codex
created: 2026-05-29
class: research_design
queue_item: track_d_mnq_comex_settle_gate0_runner_design
---

# Track D Gate 0 Runner And Table Contract

## Decision

Track D Gate 0 is unblocked at the design-contract level.

The repo should implement a bounded, no-broad-scan runner for the locked prereg
`docs/audit/hypotheses/2026-04-23-mnq-comex-settle-gate0-microstructure-v1.yaml`.
It must test only the exact `MNQ COMEX_SETTLE O5 E2 RR1.5 CB1` parent with the
four locked top-of-book features in that prereg. It must not mutate
`validated_setups`, `lane_allocation.json`, broker state, live config, or
`paper_trades`.

This is still a research falsification stage, not a deployment or sizing stage.

## Freshness Checks

Checked on 2026-05-29 before authoring:

- `git fetch origin --prune` completed.
- `origin/main` = `f9a0b24e`.
- local `HEAD` = `8f62274b`.
- local checkout was ahead of origin by 4 commits and had unrelated dirty work.
- `scripts/tools/work_queue.py status` showed exactly one open queue item:
  `track_d_mnq_comex_settle_gate0_runner_design`.
- `gold.db` was readable at `C:/Users/joshd/canompx3/gold.db`.
- Databento Python package present: `0.73.0`.
- `DATABENTO_API_KEY` was present; only metadata/cost probes were run.

## Current Candidate Universe

Read-only DuckDB measurement against canonical `orb_outcomes`:

```sql
SELECT COUNT(*), COUNT(DISTINCT trading_day), MIN(trading_day), MAX(trading_day)
FROM orb_outcomes
WHERE symbol = 'MNQ'
  AND orb_label = 'COMEX_SETTLE'
  AND orb_minutes = 5
  AND entry_model = 'E2'
  AND rr_target = 1.5
  AND confirm_bars = 1
  AND pnl_r IS NOT NULL
  AND entry_ts IS NOT NULL;
```

Measured:

- Total candidate windows: 1,741.
- IS candidate windows before `2026-01-01`: 1,658.
- OOS candidate windows from `2026-01-01`: 83.
- Date range: `2019-05-06` through `2026-05-22`.
- Direction split from `daily_features.orb_COMEX_SETTLE_break_dir`:
  - long: 921 total, 42 OOS.
  - short: 820 total, 41 OOS.
- Every measured `entry_ts` had zero seconds, so the current canonical outcome
  layer is minute-resolution. The runner must pull a buffered event window
  around the minute-level touch time and must never infer sub-minute touch order
  from the 1-minute bar itself.

## Official Databento Grounding

Official Databento docs checked on 2026-05-29:

- MBP-1 is the L1 market-by-price schema for every event that updates the best
  bid/offer, including trades and BBO depth changes:
  `https://databento.com/docs/schemas-and-data-formats/mbp-1`.
- TBBO is the L1 BBO-on-trade schema and is trade-conditioned; it is useful for
  trade-space pressure but cannot replace MBP-1 for quote-update OFI:
  `https://databento.com/docs/schemas-and-data-formats`.
- GLBX.MDP3 supports futures parent/continuous symbology workflows:
  `https://databento.com/docs/examples/futures/futures-introduction/special-conventions-for-futures-on-databento`.

Live metadata probe:

- `GLBX.MDP3` `tbbo` range covers the full required window:
  `2010-06-06T00:00:00Z` to `2026-05-29T02:50:00Z`.
- `GLBX.MDP3` `mbp-1` range covers the full required window:
  `2010-06-06T00:00:00Z` to `2026-05-29T02:50:00Z`.
- `mbo` exists from `2017-05-21`, but Gate 0 must not use it unless top-of-book
  passes and leaves a specific explanatory gap.

Representative metadata-only cost probes on five 76-second `MNQ.FUT` parent
windows:

- `tbbo`: total estimated cost `0.002799630165` USD.
- `mbp-1`: total estimated cost `0.007555380464` USD.

This is not the full budget. The implementation runner must run a full
metadata-cost pass over the 1,741 frozen windows and require an explicit
`--max-cost-usd` before any download.

## Minimal Table Contract

These are proposed DuckDB tables. Implementation should put DDL in one
canonical module, not inline strings spread across scripts.

### `micro_gate0_windows`

Purpose: immutable manifest of every candidate window and every paid pull.

Required columns:

- `window_id TEXT PRIMARY KEY`
- `hypothesis_slug TEXT NOT NULL`
- `trading_day DATE NOT NULL`
- `symbol TEXT NOT NULL`
- `databento_symbol TEXT NOT NULL`
- `stype_in TEXT NOT NULL`
- `schema_used TEXT NOT NULL`
- `orb_label TEXT NOT NULL`
- `orb_minutes INTEGER NOT NULL`
- `entry_model TEXT NOT NULL`
- `rr_target DOUBLE NOT NULL`
- `confirm_bars INTEGER NOT NULL`
- `break_dir TEXT NOT NULL`
- `entry_ts TIMESTAMPTZ NOT NULL`
- `window_start_utc TIMESTAMPTZ NOT NULL`
- `window_end_utc TIMESTAMPTZ NOT NULL`
- `lookback_seconds INTEGER NOT NULL`
- `post_touch_buffer_seconds INTEGER NOT NULL`
- `source_dbn_path TEXT`
- `metadata_cost_usd DOUBLE`
- `downloaded_at_utc TIMESTAMPTZ`
- `git_sha TEXT NOT NULL`
- `created_at_utc TIMESTAMPTZ NOT NULL`

Contract:

- Unique key must cover `(hypothesis_slug, trading_day, schema_used)`.
- `window_start_utc < entry_ts < window_end_utc`.
- `lookback_seconds = 60` for Gate 0 unless prereg is amended before execution.
- Use `MNQ.FUT` with `stype_in='parent'` unless a later symbology audit proves a
  better point-in-time contract mapping.

### `micro_mbp1_events`

Purpose: raw top-of-book update events for OFI and queue imbalance.

Required columns:

- `window_id TEXT NOT NULL`
- `ts_event TIMESTAMPTZ NOT NULL`
- `ts_recv TIMESTAMPTZ`
- `instrument_id BIGINT`
- `symbol TEXT`
- `action TEXT`
- `side TEXT`
- `price DOUBLE`
- `size DOUBLE`
- `bid_px_00 DOUBLE`
- `ask_px_00 DOUBLE`
- `bid_sz_00 DOUBLE`
- `ask_sz_00 DOUBLE`
- `bid_ct_00 DOUBLE`
- `ask_ct_00 DOUBLE`
- `sequence BIGINT`
- `flags BIGINT`

Contract:

- Store only events for the resolved front outright, not spreads.
- Preserve event order by `(ts_event, sequence)`.
- Do not aggregate before raw rows are persisted.

### `micro_tbbo_events`

Purpose: raw trade-conditioned BBO events for trade-space imbalance.

Required columns:

- `window_id TEXT NOT NULL`
- `ts_event TIMESTAMPTZ NOT NULL`
- `ts_recv TIMESTAMPTZ`
- `instrument_id BIGINT`
- `symbol TEXT`
- `price DOUBLE`
- `size DOUBLE`
- `side TEXT`
- `bid_px_00 DOUBLE`
- `ask_px_00 DOUBLE`
- `bid_sz_00 DOUBLE`
- `ask_sz_00 DOUBLE`
- `sequence BIGINT`
- `flags BIGINT`

Contract:

- TBBO may compute `signed_tbi_60s`; it must not compute quote-update OFI.
- Missing `side` rows are excluded from TBI numerator and denominator, and the
  exclusion count is surfaced in quality output.

### `micro_gate0_features`

Purpose: one frozen feature row per candidate window per schema.

Required columns:

- `feature_row_id TEXT PRIMARY KEY`
- `window_id TEXT NOT NULL`
- `trading_day DATE NOT NULL`
- `symbol TEXT NOT NULL`
- `orb_label TEXT NOT NULL`
- `orb_minutes INTEGER NOT NULL`
- `entry_model TEXT NOT NULL`
- `rr_target DOUBLE NOT NULL`
- `confirm_bars INTEGER NOT NULL`
- `break_dir TEXT NOT NULL`
- `entry_ts TIMESTAMPTZ NOT NULL`
- `lookback_seconds INTEGER NOT NULL`
- `signed_ofi_60s DOUBLE`
- `signed_tbi_60s DOUBLE`
- `signed_qi_last_1s DOUBLE`
- `signed_qi_mean_10s DOUBLE`
- `spread_mean_ticks_10s DOUBLE`
- `spread_max_ticks_10s DOUBLE`
- `event_count_mbp1 INTEGER`
- `event_count_tbbo INTEGER`
- `feature_version TEXT NOT NULL`

Contract:

- Feature timestamps must be `>= entry_ts - 60 seconds` and `< entry_ts`.
- Thresholds are fit on IS only and written as a frozen sidecar before OOS
  evaluation.
- No 2026 threshold tuning.

### `micro_gate0_results`

Purpose: frozen family decision rows.

Required columns:

- `result_id TEXT PRIMARY KEY`
- `hypothesis_slug TEXT NOT NULL`
- `feature_id TEXT NOT NULL`
- `sample_split TEXT NOT NULL`
- `n_parent INTEGER NOT NULL`
- `n_selected INTEGER NOT NULL`
- `parent_policy_ev_r DOUBLE NOT NULL`
- `selected_policy_ev_r DOUBLE NOT NULL`
- `selected_trade_mean_r DOUBLE`
- `delta_policy_ev_r DOUBLE`
- `threshold_value DOUBLE`
- `decision TEXT NOT NULL`
- `created_at_utc TIMESTAMPTZ NOT NULL`
- `git_sha TEXT NOT NULL`

Contract:

- Exactly four feature rows for the locked family:
  `signed_ofi_60s_high`, `signed_tbi_60s_high`, `signed_qi_last_1s_high`,
  `signed_ofi_60s_high_AND_signed_qi_last_1s_high`.
- `K=4` remains binding.
- OOS rows are descriptive unless the prereg is amended to promote a specific
  OOS decision rule before any OOS look.

## Runner Contract

Implement one bounded runner, suggested path:

`research/track_d_gate0_microstructure.py`

Required modes:

1. `--build-manifest`
   - Reads canonical `orb_outcomes` plus `daily_features`.
   - Writes deterministic manifest rows for the exact candidate universe.
   - No Databento calls.

2. `--estimate-cost`
   - Calls Databento metadata only.
   - Writes per-window cost estimates for `tbbo` and `mbp-1`.
   - Refuses to proceed if any window is outside Databento availability.

3. `--pull --schema {tbbo,mbp-1} --max-cost-usd X --yes`
   - Pulls only manifest windows whose metadata cost has already been recorded.
   - Refuses if total pending cost exceeds `X`.
   - Writes DBN files under `research/data/track_d_gate0/{schema}/`.

4. `--ingest --schema {tbbo,mbp-1}`
   - Loads cached DBN only.
   - Filters to front outright; spread rows are rejected with counts.
   - Inserts raw rows into the matching micro table.

5. `--features`
   - Computes only pre-touch features.
   - Fails closed if any feature requires events at or after `entry_ts`.

6. `--evaluate`
   - Fits thresholds on IS only.
   - Emits `micro_gate0_results` plus a markdown result doc under
     `docs/audit/results/`.
   - Does not write `validated_setups`, allocation files, or live state.

## Acceptance Gates

Implementation is acceptable only if all are true:

- Full cost estimate exists before any data pull.
- Download mode requires explicit `--max-cost-usd` and `--yes`.
- Manifest count remains 1,741 unless the runner explains a newer DB horizon.
- IS count remains 1,658 and OOS count remains 83 unless the DB has advanced and
  the result doc states the new horizon.
- Feature code proves no post-entry event usage.
- The runner has a `--dry-run` or metadata-only path that can be verified without
  Databento spend.
- The result doc reports parent policy EV and selected policy EV, not just
  selected-trade mean.

## Not In Scope

- No MBO pull.
- No live routing.
- No lane allocation.
- No validated setup promotion.
- No paper-trade carrier.
- No threshold expansion beyond K=4.

## New Queue State

The old queue item `track_d_mnq_comex_settle_gate0_runner_design` can close:
the missing table/runner contract now exists and the Databento availability/cost
blocker has a measured metadata-only path. A future implementation item should
be explicit and separate, for example:

`track_d_mnq_comex_settle_gate0_runner_implementation`

That future item should start with `--build-manifest` and `--estimate-cost`, not
with data download or schema mutation.
