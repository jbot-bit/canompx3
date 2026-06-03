task: Restore strict-zero-warn parity on the dashboard live-launch path — a preflight WARN (incl. SKIPPED) must block a mode=live launch, matching the retired start_topstep_live_pilot.py --strict-zero-warn gate. Signal/demo unchanged.
mode: IMPLEMENTATION

## Scope Lock
- trading_app/live/bot_dashboard.py
- tests/test_trading_app/test_bot_dashboard.py

## Blast Radius
- trading_app/live/bot_dashboard.py — modifies action_start() prep_status gate (line ~2520). For mode="live" only, adds "warn" to the blocking-status set so a readiness WARN/SKIPPED blocks the real-money launch. mode in {signal, demo} unchanged (warn stays advisory — no live orders). No signature change, returns the existing {"status":"blocked"} shape the UI already renders. Reads: preflight cache via _prepare_profile_for_start (read-only). Writes: none.
- tests/test_trading_app/test_bot_dashboard.py — adds regression test asserting live blocks on warn while signal/demo proceed.
- Capital path: tightens an existing gate (more restrictive), never loosens. Final capital gate (UI HOLD TO GO LIVE → --auto-confirm) unchanged.

## Acceptance
- New test passes; full bot_dashboard test file green.
- check_drift.py passes.
- Independent adversarial-audit gate (evidence-auditor) before close — capital path, per .claude/rules/adversarial-audit-gate.md.
