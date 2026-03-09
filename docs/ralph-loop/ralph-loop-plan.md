# Ralph Loop — Current Iteration Plan

> Updated each iteration by the Architect agent.
> Previous plans are preserved in `ralph-loop-history.md`.

## Status: IN PROGRESS

## Iteration: 1

## Targets: Finding 1 (HIGH) + Finding 2 (HIGH)

---

### Finding 1: Tradovate positions "BUY"/"SELL" vs "long"/"short"
- **File:** `trading_app/live/tradovate/positions.py:36`
- **Why top priority:** Infrastructure gate failure — 1 pytest failure blocking CI. Also a crash recovery bug: `_emergency_flatten` and all caller code expects "long"/"short" (matching ProjectX pattern). "BUY"/"SELL" would silently misroute.
- **Blast radius:** `session_orchestrator.py` lines 136, 1147 (orphan detection, EOD reconciliation), `_emergency_flatten` line 873 (direction matching).
- **Fix:** Working tree already contains the correct 1-line fix: `"BUY" -> "long"`, `"SELL" -> "short"`. Verify and confirm.
- **Must NOT change:** Test expectations, ProjectX positions, broker_base interface.

### Finding 2: Slippage under-reporting in _record_exit
- **File:** `trading_app/live/session_orchestrator.py:531-534`
- **Why second priority:** Silent data quality issue. Comment says "entry + exit" but only exit slippage captured. CUSUM drift detection sees artificially low slippage. Entry slippage is already computed and stored in `PositionRecord.entry_slippage` (position_tracker.py:144) but never read by `_record_exit`.
- **Blast radius:** `TradeRecord.slippage_pts` -> `PerformanceMonitor.record_trade()` -> CUSUM alerts. `actual_r` is NOT affected (computed from fill prices directly).
- **Fix:** Add `entry_slippage` parameter to `_record_exit`. At each call site, retrieve entry_slippage from the position record before it's deleted, and pass it through. Add entry_slippage to the slippage_pts sum.
- **Must NOT change:** `PositionTracker` internals, `TradeRecord` schema, `PerformanceMonitor` interface, actual_r computation.

## Constraints

- Minimal diffs only
- No schema changes without PIPELINE_DATA_GUARDIAN
- No entry model changes without ENTRY_MODEL_GUARDIAN
- Audit before fixing
- Evidence before assertions
- If uncertain, mark as HYPOTHESIS
