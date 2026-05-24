---
task: F1 — add retry-with-jitter wrapper around the two `duckdb.connect()` write-phase calls in `trading_app/strategy_validator.py` (Phase A ~line 1381, Phase C ~line 1730). Peer process briefly holding gold.db at the write-back moment currently throws away ~750s of WF compute. No schema changes, no trading-logic changes, no criteria changes.
mode: CLOSED
closed_date: 2026-05-24
closed_note: |
  Shipped. Helper `_open_writer_with_retry` added at line 92; all 3 writer-phase
  `duckdb.connect(str(db_path))` calls (Phase A line 1408, Phase C line 1757,
  validation_run_log line 2433) routed through it. 6 attempts × exponential
  jitter (1s/2s/4s/8s/16s/30s cap), ~61s ceiling. Non-lock IOExceptions
  re-raise unchanged. Validator run on MNQ (`--allow-legacy-prereg`) exit 0:
  768 PASSED, 77787 REJECTED of 205491. validated_setups MNQ:
  789 → 793 total, 783 → 787 active, max_last_trade_day 2025-12-31 → 2026-05-19,
  max_promoted_at 2026-05-10 → 2026-05-24 05:03 Brisbane. Resolved the n=3
  write-lock-race failure pattern from this session's earlier attempts.
original_mode: IMPLEMENTATION
scope_lock:
  - trading_app/strategy_validator.py
agent: claude (opus 4.7)
---

## Blast Radius
- WRITES: trading_app/strategy_validator.py only (+30/-3). Helper at line 92; three writer-phase connect sites switched (1408, 1757, 2433).
- READS: gold.db (validator's existing write phases only).
- LIVE-IMPACT: validator promotion pathway resilience only. No criteria/threshold/classification change.
- Idempotency: identical to current.
- Rollback: revert the single-file edit.
