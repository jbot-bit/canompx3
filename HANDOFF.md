# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-04-25
- **Commit:** 6c9b8bea — [mechanical] fix: Ralph Loop iter 177 — audit docs + ledger rebuild
- **Files changed:** 4 files
  - `HANDOFF.md`
  - `docs/ralph-loop/ralph-ledger.json`
  - `docs/ralph-loop/ralph-loop-audit.md`
  - `docs/ralph-loop/ralph-loop-history.md`

## Next Steps — Active

1. **LIVE XFA RUNNING (PID 56464)** — launched 2026-04-25 ~10:33 Brisbane. `--profile topstep_50k_mnq_auto --instrument MNQ --live --account-id 21944866 --copies 1 --auto-confirm`. Account 21944866 is the EXPRESS-V2 simulated XFA ($0 balance, F-1 active and seeded $0.00 day-1). Markets reopen Mon 07:00 Brisbane (CME), Tokyo ORB at 10:00, then 5 more lanes through the day. Log: `logs/live/live_20260425_1033.log.err`. Operator monitoring: `tail -f logs/live/live_*.err` + `grep SIGNAL_ENTRY live_signals.jsonl`. **DO NOT touch this process** unless intentional; killing it stops overnight smoke.

2. **Deferred audit findings (NOT blocking the running session)** — institutional audit 2026-04-25 surfaced these. F4 closed by Ralph 174 (`87dffa38`). Remaining for next-session work:
   - **R1 (CRITICAL):** trading-day rollover only fires from `_on_bar`. If feed dies at 08:55 Brisbane and reconnects at 09:30, the 09:00 rollover is delayed → ORB windows for new day's Tokyo/Singapore could be miscomputed against stale `trading_day`. Fix: schedule a wall-clock task in `run()` that calls `_check_trading_day_rollover(datetime.now(UTC))` every 60s independent of bars.
   - **F7 (HIGH):** fill poller stuck PENDING consumes lane concurrency slot indefinitely. `session_orchestrator.py:2432-2434` swallows per-order exceptions and continues. Needs per-order retry-with-give-up + stale-order timeout.
   - **R3 (HIGH):** `ORCHESTRATOR_MAX_RECONNECTS=5` × 20 inner = 100 attempts ~27 min outer cycle. For 24h+ unattended, single multi-hour broker outage = permadeath. Raise to 20+ for `--demo`/`--live`, or persist last-success and resume.
   - **R4 (HIGH):** `live_signals.jsonl` unbounded growth at `session_orchestrator.py:194,1195`. Daily-suffix or `RotatingFileHandler`.
   - **R5 (HIGH):** engine circuit-breaker silent after the one notification at threshold. If Telegram is down at the firing moment, operator never knows. Needs periodic re-notify in heartbeat task.
   - **Falsy-or pattern leak (LOW-MED):** SAME bug class that broke `query_equity` exists in 7 sites — `auth.py:115`, `contract_resolver.py:75`, `positions.py:66/105`, `data_feed.py:78`, `bot_dashboard.py:1519`, `session_orchestrator.py:797`. Most lower-risk (values not naturally 0/empty) but pattern is unsafe. Recommend lint rule: flag `\.get\([^)]+\)\s*or\s*[a-zA-Z_]+\.get` in production code.
   - **Stale `data/state/account_hwm_20092334.json` (LOW):** Apr 6 2026 file with HWM=$103,034.09 and `daily_start_equity=None` for the user's intended real-money XFA target (per memory). If a future session connects to that account ID, the tracker loads this stale state — first daily-DD calc may be bogus. Audit/repair before any real-money launch on 20092334.

3. **O-SR debt** — `trading_app/live/cusum_monitor.py` implements CUSUM Eq 3, not Shiryaev-Roberts Eq 10 per `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`. Multi-stage; not autonomous.

4. **Codex parallel-session WIP drain** — 6 Python files still dirty in working tree. LF-pinned via `.gitattributes`; current CRLF state preserved until Codex commits/drops. Don't `git add --renormalize "*.py"` until Codex is clear.

5. **`test_multi_instrument_locks` skipped** — fails environmentally because PID 56464 holds `bot_MNQ.lock`. Re-run after live session ends.

## Blockers / Warnings

- Memory entry calling F-1 "dormant (F-1 hard gate dormant)" is FACTUALLY WRONG — F-1 is active and fail-closing in signal-only PRE-fix; post-fix it enforces against the canonical day-1 $0 cap. Correct before next research/deployment decision.
- `build_live_portfolio()` is DEPRECATED — `--all --signal-only` without `--profile` hard-fails all 3 orchestrators. Only the `--profile` path works for live. This is surfaced at `multi_runner.py:79-81` as a log-and-drop silence (D2 in audit).
- Drift check pre-existing: #4 (`work_queue.py` schema parser false-positive on `table 'the'`) and #59 (**MGC** — corrected from earlier "MNQ" — 1 day with `!= 3 daily_features` rows). Neither blocked today's work.
- stash@{0} `canompx3-pre-pr100-ff-2026-04-24` — DROPPED 2026-04-25 after this-session review verified contents were superseded (HANDOFF text older than HEAD; test_bot_dashboard.py whitespace collapses already absorbed by PR #100).

## Verified clean this session

- `is_pid_alive` public in both `trading_app/live/instance_lock.py` and `pipeline/db_lock.py` (via PR #100).
- `PerformanceMonitor` + `CUSUMMonitor` wired at `session_orchestrator.py:474, 1481` (v3 audit false-positive corrected).
- `CircuitBreaker` wired at `session_orchestrator.py:648` + 6 call sites (v3 audit false-positive corrected).
- Signal path end-to-end: bar feed → ORB build → break detect → filter check → F-1 gate. **B6 closed + empirically verified:** signal-only branch now seeds F-1 EOD balance with $0.0 (canonical day-1 XFA per `topstep_scaling_plan.py:51-53`). E2E probe this session — real `SessionOrchestrator(signal_only=True, profile=topstep_50k_mnq_auto, broker=projectx)` construction → `risk_mgr.limits.topstep_xfa_account_size = $50,000` (F-1 ON), `_topstep_xfa_eod_balance = $0.00` (seed applied), all 6 deployed MNQ lanes (`MNQ_EUROPE_FLOW`, `_SINGAPORE_OPEN`, `_COMEX_SETTLE`, `_NYSE_OPEN`, `_TOKYO_OPEN`, `_US_DATA_1000`) ALLOW through `risk_mgr.can_enter()`. Pre-fix: all 6 would REJECT with "EOD XFA balance unknown".
- B2 notifications: now returns `bool`; preflight check 5 detects a broken Telegram loudly.
- Post-codex-audit review (Claude /code-review): PR #100 + 5be52bdc + dirty WIP + stash@0 — A- grade, full sweep clean. Findings doc not written; B6 fix landed instead.
- **F2/F6/F8 source-marker tests REPLACED with behavior tests (this commit).** 3 inline branches in `__init__` extracted as helpers (`_cleanup_orphan_brackets`, `_notify_journal_unhealthy_demo`, `_notify_f1_silent_block_if_active`) + 8 new behavior tests. Mutation-proof now.
- **F5 mock-only test SUPERSEDED by real-tracker integration test (this commit).** New `test_f5_real_tracker_three_strikes_fires_halt` uses real `AccountHWMTracker`, calls `update_equity(None)` 3 times, asserts canonical halt at `account_hwm_tracker.py:314`. Plus a reset-on-success counterpart.
- **`--account-id` validation + `query_equity` $0 fix now have tests** (audit gap closure). `tests/test_scripts/test_run_live_session_account_selection.py` (7 tests) + 4 new tests in `test_projectx_positions.py`.
- **F4 (CRITICAL) bracket-naked-position CLOSED by Ralph 174 (`87dffa38`).** Deferred-list F4 is no longer outstanding.
- **Live XFA session up:** PID 56464 connected to ProjectX Market Hub, F-1 active+seeded ($0 day-1 = 2-lot bottom-tier cap), all 3 self-tests pass. Markets reopen Mon 07:00 Brisbane.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
- `docs/runtime/stages/live-b2-notifications-bool-return.md` (B2 stage doc)
- `docs/runtime/stages/live-b6-f1-signal-only-seed.md` (B6 stage doc — landed)
- `C:/Users/joshd/.claude/plans/i-wnat-live-going-squishy-sparkle.md` (v4 plan + checkpoint log)
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` (O-SR grounding)
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` (ORB premise)
