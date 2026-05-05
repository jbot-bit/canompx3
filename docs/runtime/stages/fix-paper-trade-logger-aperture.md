---
task: Fix paper_trade_logger.py:167 aperture-mismatch (PR #189 class bug recurrence)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/paper_trade_logger.py
  - tests/test_trading_app/test_paper_trade_logger.py
---

## Blast Radius

- `trading_app/paper_trade_logger.py` — modify `_load_features()` and `_inject_cross_asset_atrs()` to accept `orb_minutes: int`; modify `backfill()` callsite to pass `lane.orb_minutes` (per-lane, not per-instrument).
- `tests/test_trading_app/test_paper_trade_logger.py` — add regression test that an O15 lane reads orb_minutes=15 rows (asserts session-column value matches the o15 row, not o5).
- Reads: `gold.db` (read-only) for daily_features rows.
- Writes: `paper_trades` table (no schema change; only the rows that pass filter shift because the filter now sees correct per-aperture session data).
- Affects: paper-trade backfill output for any lane where O15/O30 rows differ from O5 rows on the same (day, symbol, session) — i.e. all non-O5 lanes in `lane_allocation.json`.
- Does NOT affect: live execution path (paper_trade_logger writes paper_trades, not live trades). Live allocator already correct post-PR #189.
- Callers: `scripts/infra/signal_alert.py` imports `build_lanes` only — unaffected. Test file imports updated.

## Why

PR #189 (2026-04-30, commit `d73bab28`) fixed the same class bug in `lane_allocator.py` (5 query sites hardcoded `orb_minutes=5`). The sweep did not extend to `paper_trade_logger.py`. Independent evidence-auditor confirmed via gold.db query (MNQ 2026-04-02 NYSE_OPEN_size = 56.25/141.0/229.75 across o5/o15/o30; break_dir flips short→long at o15) that the 3 orb_minutes rows carry materially different per-session values, not duplicates. The CTE-Guard rule (`=5` for non-aperture-specific columns) does NOT apply to a `SELECT *` whose downstream consumes aperture-sensitive columns.

## Acceptance

- `paper_trade_logger.py` _load_features and _inject_cross_asset_atrs accept `orb_minutes` param.
- `backfill()` passes `lane.orb_minutes` to both helpers per-lane.
- New regression test: load_features for O15 lane returns rows where `orb_NYSE_OPEN_size` matches the o15 row, not o5.
- `python pipeline/check_drift.py` passes.
- Existing tests in `tests/test_trading_app/test_paper_trade_logger.py` still pass.
