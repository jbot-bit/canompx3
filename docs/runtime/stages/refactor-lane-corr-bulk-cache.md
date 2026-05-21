---
task: Refactor compute_pairwise_correlation to bulk-load daily_features once per (instrument, orb_minutes); fix rebalancer segfault (exit 139) on N=762 candidate pool.
mode: IMPLEMENTATION
scope_lock:
  - trading_app/lane_allocator.py
  - trading_app/lane_correlation.py
  - trading_app/strategy_fitness.py
  - tests/test_trading_app/test_lane_correlation_cache.py
---

## Blast Radius

- trading_app/strategy_fitness.py — extract canonical filter+enrichment helper (`_filter_outcomes_with_features`) shared with bulk path; original `_load_strategy_outcomes` delegates to it; no behavior change for existing callers (`compute_portfolio_fitness`, `live_health`, etc.).
- trading_app/lane_correlation.py — preserve `_load_lane_daily_pnl(con, lane)` signature unchanged; ADD cache-aware `_load_lane_daily_pnl_cached(con, lane, outcomes_cache, features_cache, applied_enrichments)`.
- trading_app/lane_allocator.py — refactor `compute_pairwise_correlation` internals only; same public signature (4 production callers depend on it: `rebalance_lanes.py`, `generate_profile_lanes.py`, `allocator_gate_audit.py`, `audit_allocator_rho_excluded.py`, `allocator_scored_tail_audit.py`). Build outcomes_cache + features_cache up-front, loop with cache-aware helper.
- tests/test_trading_app/test_lane_correlation_cache.py — NEW. Byte-identical equivalence test against ≥5 candidates spanning NO_FILTER + ordinary filter + VolumeFilter + CrossAssetATRFilter + ≥2 distinct (instrument, orb_minutes).
- Reads: gold.db (orb_outcomes, daily_features) read-only.
- Writes: none in DB; no canonical config or schema change.
- Canonical reuse: ALL_FILTERS[key].matches_row(), DST_AFFECTED_SESSIONS, is_winter_for_session — single implementation path.
- Mechanism fix: collapses 762 × 4K × 289-col dict materializations into one bulk-load per (instrument, orb_minutes) reused across all candidates sharing that pair — mirrors `_bulk_load_all_features` precedent at trading_app/strategy_fitness.py:477.

## Why

Rebalancer crashed exit 139 (SIGSEGV) on 2026-05-21 inside `compute_pairwise_correlation` at N=762 candidates. Root cause: per-candidate `_load_strategy_outcomes` materializes ~4K rows × 289 cols × dict per candidate; 762 calls allocate ~8.5×10⁸ Python objects, fragmenting the heap until malloc fails. Mechanism is structural (F-10 fix widened SELECT to all columns), not a recent regression.

## Approach

Mirror the existing `_bulk_load_all_features` + cache-feeding pattern that `compute_portfolio_fitness` already uses (proven in production for the same data layer). New helper `_load_strategy_outcomes_from_caches` accepts pre-loaded outcome and feature caches. `compute_pairwise_correlation` loads each `(instrument, orb_minutes)` pair once and reuses the cache across all candidates sharing that pair.

Single-candidate path (`check_candidate_correlation` in `live_health`) preserves original `_load_lane_daily_pnl(con, lane)` signature unchanged.

## Verification

1. Numerical equivalence test: 5-candidate subset, pre-refactor vs post-refactor pairs identical to float-precision.
2. Full rebalance: `python scripts/tools/rebalance_lanes.py` completes without segfault; produces allocation for MNQ profile.
3. `python pipeline/check_drift.py` passes.
4. `tests/test_trading_app/test_lane_correlation_cache.py` passes — equivalence test fixture-driven.
