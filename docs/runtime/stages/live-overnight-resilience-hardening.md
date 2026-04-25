# Stage: live-overnight-resilience-hardening

mode: IMPLEMENTATION
date: 2026-04-25
scope_lock:
  - trading_app/live/session_orchestrator.py
  - trading_app/live/projectx/positions.py
  - scripts/run_live_session.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_projectx_positions.py
  - docs/runtime/stages/live-overnight-resilience-hardening.md
  - HANDOFF.md

# Scope expanded 2026-04-25 02:35 Brisbane during live launch attempt.
# 3 launch-blocking bugs discovered in sequence:
#   (a) F2 regression: self._notify in F2 fix fires before self._stats init in __init__,
#       causing AttributeError that fails-closed the entire orchestrator startup.
#   (b) --account-id ignored when copies>1: scripts/run_live_session.py:542
#       slices all_accounts[:n_copies] BEFORE checking user's account-id, so the
#       user's choice can fall outside the candidate set and silently route to
#       the wrong account.
#   (c) ProjectX query_equity returns None for $0-balance accounts: positions.py:68
#       uses `acct.get("balance") or acct.get("cashBalance")` — 0.0 is falsy, so
#       day-1 XFA accounts (which legitimately start at $0) report "not found"
#       and break F-1 EOD seeding for the live XFA path.
# All 3 fixes are contiguous: same launch attempt, same failure cascade, same root
# cause class (silent state-degradation). One commit, one verification pass.

## Why

User is launching `python scripts/run_live_session.py --profile topstep_50k_mnq_auto --instrument MNQ --demo` for unattended ~24h walkaway. Two parallel scout agents (general-purpose + blast-radius) found 5 small-scope silent-failure bugs that would each invisibly degrade the overnight demo session. Fixing them now keeps the run honest as a demo-to-live promotion gate.

## Findings being fixed (one stage, one commit)

### F8 — Orphan bracket cleanup failure should HALT, not warn
`session_orchestrator.py:507-508` swallows the exception with `log.warning(...)` and continues. A failed cleanup means prior-session AutoBracket orders may still be live at the broker; on Monday's open they could fire and create ghost positions with no PositionTracker entry, no HWM accounting, and no software exit path. **CRITICAL** for unattended.

The fix mirrors the `force_orphans` bypass pattern at L482-491 (30 lines above) which is already in use for the orphan-detection step: log critical + notify + raise unless `--force-orphans`.

### R2 — Telegram `urlopen` blocks the async event loop
`scripts/infra/telegram_feed.py:41` calls `urllib.request.urlopen(req, timeout=10)` synchronously. Called from the orchestrator's `_notify()` inside async `_on_bar`. With Telegram down for 6h, every notify burns 10s of event-loop time. Bursts (heartbeat warnings, stale-order alerts, kill switch) stack to 60-120s blockages, which then falsely triggers the bar-heartbeat watchdog and may trip the engine circuit breaker. **CRITICAL** for unattended.

The fix wraps `_notify` body in `loop.run_in_executor` so the blocking call runs on a thread without stalling the event loop. The `_notifications_broken` short-circuit already exists; this just keeps it from blocking.

### F2 — F-1 None equity at startup silently blocks all entries
`session_orchestrator.py:636-637` logs `log.warning("HWM tracker: broker equity unavailable at startup — will init on first poll")` but does NOT notify. The HWM intraday poll runs every 10 bars (~10min). Until that fires with non-None equity, `risk_manager._topstep_xfa_eod_balance` stays None, F-1 fail-closes every entry. The TOKYO_OPEN ORB at 10:00 Brisbane could fire entirely inside this silent-block window. **HIGH** — caller has no signal.

The fix adds a `self._notify(...)` alongside the warning so Telegram surfaces the silent-block state.

### F5 — HWM equity poll exception bypasses the 3-strike halt
`session_orchestrator.py:1404-1405` `except Exception as e: log.warning("HWM equity poll failed: %s", e)` swallows the exception and never calls `tracker.update_equity(None)`. The tracker's `_consecutive_poll_failures` counter (3 → halt at `account_hwm_tracker.py:314`) requires the `update_equity(None)` call to increment. A persistent broker fault (auth token expiry mid-session) produces warning spam forever and never halts. **HIGH** — DD protection silently degrades.

The fix routes the exception through `update_equity(None)` so the 3-consecutive-failure halt mechanism fires as designed.

### F6 — Journal unhealthy in demo is silent
`session_orchestrator.py:519-525` raises in live mode but only `log.warning(...)` in demo/signal-only. No `_notify`. Operator wakes up to zero trade records with no indication the journal was broken — invalidates the demo as a promotion gate. **MED.**

The fix adds `_notify` alongside the warning.

## Blast Radius

- `trading_app/live/session_orchestrator.py`:
  - L488-491 (F8): wrap orphan-cancel `except` in `force_orphans` bypass pattern (~10 lines).
  - L519-525 (F6): add `_notify` next to the warning (~2 lines).
  - L636-637 (F2): add `_notify` next to the warning (~2 lines).
  - L1077-1108 (R2): wrap `_notify` body in `loop.run_in_executor` if event loop available (~6 lines net).
  - L1404-1405 (F5): route exception through `update_equity(None)` + halt path (~12 lines).
- `tests/test_trading_app/test_session_orchestrator.py`:
  - 5 new tests, one per finding. All use existing `build_orchestrator(FakeBrokerComponents)` fixture pattern.
- HANDOFF.md: tick the 5 findings into "Verified clean this session"; surface 6 deferred findings as new "Next Steps — Active" items with explicit severity.

No production behaviour change for the happy path. All fixes are silent-failure → loud-failure conversions or async-correctness improvements. No new public API. No new dependencies.

## Out of scope (deferred, logged in HANDOFF)

- **F4 (CRITICAL)** — Bracket submit failure post-fill leaves position naked. Touches `_submit_bracket` + execution_engine.py; needs careful design re: vs the existing safety-gate at L1952 (pre-submit) and the emergency-flatten path. Multi-stage refactor.
- **F7 (HIGH)** — Fill poller stuck PENDING consumes lane concurrency slot indefinitely. Needs poller-level retry-then-give-up logic with explicit stale-order timeout.
- **R1 (CRITICAL)** — Trading-day rollover only fires from `_on_bar`. If the feed is dead at 09:00 Brisbane, rollover is delayed. Needs a wall-clock task scheduled independently of bars.
- **R3 (HIGH)** — `ORCHESTRATOR_MAX_RECONNECTS=5` is too low for 24h demo. Easy bump but should also persist last-success and resume.
- **R4 (HIGH)** — `live_signals.jsonl` unbounded growth. Needs daily-suffix or RotatingFileHandler.
- **R5 (HIGH)** — Engine circuit breaker silent until rollover. Needs periodic re-notify in heartbeat task.

## Verification

1. `pytest tests/test_trading_app/test_session_orchestrator.py -k "OrphanCleanup or Notify or HWMPoll or JournalHealth or AsyncNotify" -q` — 5 new tests pass.
2. `pytest tests/test_trading_app/test_session_orchestrator.py tests/test_trading_app/test_risk_manager.py tests/test_trading_app/test_notifications.py -q` — full sweep no regression.
3. `PYTHONPATH=. python pipeline/check_drift.py` — 107/107 + 6 advisory.
4. `PYTHONPATH=. python scripts/run_live_session.py --profile topstep_50k_mnq_auto --instrument MNQ --demo --preflight` — preflight 5/5.
5. Mutation probe: revert each fix individually and confirm the corresponding test fails.

## Commit

`fix(live): close 5 silent-failure paths before unattended demo run`
