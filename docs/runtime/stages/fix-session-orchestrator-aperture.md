---
task: Fix session_orchestrator.py LIVE-PATH aperture hardcode (PR #232 auditor finding)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_session_orchestrator.py
  - scripts/run_live_session.py
---

## Blast Radius

- `trading_app/live/session_orchestrator.py` — `_build_daily_features_row()`:
  - Remove default `orb_minutes: int = 5` from signature (line 986). Make required.
  - Parameterize median_atr_20 SQL `WHERE ... AND orb_minutes = 5` → `... = ?` at line 1053, bind `orb_minutes`.
  - Parameterize cross-asset atr_20_pct SQL `WHERE ... AND orb_minutes = 5` → `... = ?` at line 1074, bind `orb_minutes`.
- `tests/test_trading_app/test_session_orchestrator.py` line 758 — single non-keyword caller; update to pass `orb_minutes=5` explicitly. Add new TestApertureRouting class with: (a) signature test asserts no default, (b) BEHAVIORAL test mocking duckdb, asserting bind params include `orb_minutes=15` when called with 15 (auditor flagged this gap on PR #232 — signature alone can miss literal-5 reversion in SQL).
- `scripts/run_live_session.py` line 134 — preflight caller; update to pass `orb_minutes=5` explicitly (preflight is deliberately an O5 reference check).
- Reads: `gold.db` (read-only).
- Writes: none. Affects bot_state daily_features_row dict for any non-O5 lane in the live profile (currently only MNQ_NYSE_OPEN_O5 active per memory 2026-05-01 PM rebalance, so live impact is zero TODAY but the F-1 kill-gate could surface non-O5 lanes on the next rebalance).
- Companion to: PR #231 (paper_trade_logger), PR #232 (paper_trader). Same class as PR #189 (lane_allocator).

## Why

PR #232 evidence-auditor (independent context) surfaced two CRITICAL recurrences in the LIVE execution code path:
- Line 1053: `median_atr_20` query — atr_20 may be aperture-invariant in practice (architecture-consistent inference) but the literal is still drift-bait.
- Line 1074: cross-asset `atr_20_pct` query — auditor verified per-aperture atr_20_pct percentile rank may differ across rows; this is the live-money equivalent of the bug fixed in PR #232.

Per institutional-rigor rule "after any fix, review the fix" — the PR #232 audit pass has no exemption: live-path recurrence must be closed in same audit cycle.

Per institutional-rigor rule 5 (no dead/drift-bait parameters) — the `orb_minutes: int = 5` default at line 986 must be removed even though current callers all override it. The default is the failure-mode entry point.

## Acceptance

- `_build_daily_features_row(trading_day, instrument, orb_minutes)` — orb_minutes required (no default).
- Both queries (median + cross-asset) use parameterized `orb_minutes`, no literal `5`.
- All 3 callers updated: live (already correct), test fixture (line 758), preflight (scripts/run_live_session.py:134).
- New `TestApertureRouting` class: signature test + behavioral test (mock duckdb, capture bind params, assert orb_minutes value matches argument).
- Existing test_session_orchestrator.py tests still pass (~1900 lines, many use `patch.object` on the method so signature compat preserved).
- `python pipeline/check_drift.py` passes.
