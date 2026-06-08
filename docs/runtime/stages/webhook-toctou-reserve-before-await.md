task: Close webhook TOCTOU — reserve dedup + position slot BEFORE the _place_order await, roll back on failure, so concurrent duplicate triggers can't both submit (cap overshoot).
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/webhook_server.py
  - tests/test_trading_app/test_webhook_server.py

## Blast Radius
- trading_app/live/webhook_server.py — modifies the /trade handler ordering: moves position-counter increment + a dedup reservation BEFORE the `await loop.run_in_executor(None, _place_order, ...)`, adds rollback on contract-resolution / order-placement failure. `_OPEN_POSITIONS`, `_DEDUP_CACHE`, `_check_dedup`, `_check_position_limit`, `_cache_response` are all module-internal (grep: zero external callers). No signature changes. Behavior change is on a capital path (when cap state commits) — operator-approved (AskUserQuestion, this session).
- tests/test_trading_app/test_webhook_server.py — flips the TOCTOU char-test from asserting the unsafe behavior (both submit) to asserting the safe behavior (second is deduplicated/blocked, counter == 1). 16 existing tests must stay green.
- Reads: env vars (WEBHOOK_MAX_POSITIONS, DEDUP_WINDOW). Writes: none to DB. No live arm.

## Acceptance
- Concurrent identical entries → exactly ONE submit, _OPEN_POSITIONS[instr] == 1 (not 2).
- A failed order placement rolls back the reservation (counter returns to pre-request value, no phantom dedup block).
- All prior webhook tests green. drift check passes. Independent evidence-auditor pass (adversarial-audit-gate) after.
