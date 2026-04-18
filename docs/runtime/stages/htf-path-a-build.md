---
mode: IMPLEMENTATION
slug: htf-path-a-build
stage: 1/1
started: 2026-04-18
task: "Canonical HTF feature build — prev_week_* and prev_month_* in daily_features"
---

# Stage 1 — HTF Path A canonical feature build

## Scope
Add 12 price-safe HTF level fields to `daily_features`:
- `prev_week_{high, low, open, close, range, mid}`
- `prev_month_{high, low, open, close, range, mid}`

All fields are post-pass derivations from existing `daily_{open, high, low, close}`.
Monday-anchor week via DuckDB `DATE_TRUNC('week', ...)` semantics; Python
post-pass uses equivalent `weekday()`-based anchor. Calendar-month anchor.
Prior-period-only — no look-ahead.

## Scope Lock
- pipeline/init_db.py
- pipeline/build_daily_features.py
- pipeline/check_drift.py
- tests/test_pipeline/test_build_daily_features.py
- scripts/backfill_htf_levels.py
- research/verify_htf_fire_rate.py

## Blast Radius
blast_radius: |
  Schema change adds 12 DOUBLE NULLABLE columns to daily_features — fully
  backwards-compatible with all existing readers (validator, discovery,
  fitness, dashboard, audit tooling). No live bot impact (live_journal.db
  is independent). New post-pass in build_daily_features.py runs O(n) via
  running per-(symbol,week_key)/(symbol,month_key) dicts. New drift check
  in check_drift.py raises count from 87 passing to 88 passing. Tests
  extended in test_build_daily_features.py. Optional one-shot backfill
  script at scripts/backfill_htf_levels.py. Research smoke test at
  research/verify_htf_fire_rate.py. Downstream consumers of daily_features
  (strategy_discovery, strategy_validator, fitness_tracker, dashboards)
  all use SELECT * or explicit column lists; new optional columns do not
  break any of them. Research layer: enables follow-up HTF Path A v1-index
  pre-reg file and validation pass — not executed in this stage.

## Acceptance criteria
1. `python pipeline/init_db.py` applies migration without error
2. `python -m pytest tests/test_pipeline/test_build_daily_features.py -q` passes including new HTF tests
3. `python pipeline/check_drift.py` exits 0 with count 88/0/7
4. `python scripts/backfill_htf_levels.py` populates existing canonical DB rows idempotently
5. Drift check re-run confirms no divergence between stored values and canonical SQL aggregation
6. Fire-rate smoke script reports fire_rate per v1-index cell (MNQ+MES × {TOKYO_OPEN,EUROPE_FLOW,NYSE_OPEN} × {long,short})
7. One focused commit with message `feat(daily_features): add prev-week and prev-month HTF level fields`

## Non-goals for this stage
- Running the HTF discovery scan
- Committing a pre-reg YAML
- First-touch flag fields (`week_took_pwh_to_date` etc.) — deferred as scan-time derivation per prompt guidance
- Wiring an HTF filter into `trading_app/config.py`
- Any F5 / allocator / live-routing work
