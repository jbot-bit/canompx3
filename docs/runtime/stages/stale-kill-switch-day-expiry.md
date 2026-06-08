# Stage: Stale kill-switch day-expiry + START_BOT mode menu

task: Day-gate the persisted kill-switch restore so a prior-day kill switch auto-expires on launch (preserving same-day crash-restart protection), and add an interactive mode menu + stale-state preflight to START_BOT.bat.
mode: CLOSED

## Scope Lock
- trading_app/live/session_safety_state.py
- trading_app/live/session_orchestrator.py
- tests/test_trading_app/test_session_safety_state.py
- START_BOT.bat

## Blast Radius
- session_orchestrator.py `__init__` (~line 755) — read by EVERY live/demo/signal launch. Change NARROWS a halt condition (clears a stale kill switch from a prior closed trading day). Dangerous failure direction = bot trades when it should halt, so the same-day-PRESERVE branch + its test are load-bearing. Mirrors the existing daily-P&L day-gate at line 764 (proven pattern). Adversarial-audit gate applies (CRIT/HIGH live path).
- START_BOT.bat — launcher only, no capital logic. Adds interactive [1]SIGNAL/[2]DEMO/[3]LIVE menu on bare double-click (existing `START_BOT.bat live` arg path unchanged) + a 1-line stale-safety-state preflight warning.
- Reads/writes data/state/session_safety_*.json on expiry. Idempotent.
- Root cause: kill_switch_fired restored unconditionally (line 755) while daily_pnl is correctly day-gated (line 764). n=1 repro 2026-06-09: 2026-06-08 kill switch halted the 2026-06-09 launch.
