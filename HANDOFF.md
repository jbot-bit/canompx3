# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-04-26
- **Commit:** 45720109 — [judgment] fix(live): HWM Stage 1 fix-up — STAGE1-GAP-1 None-reason contract-drift visibility
- **Files changed:** 5 files
  - `HANDOFF.md`
  - `docs/ralph-loop/deferred-findings.md`
  - `docs/runtime/stages/hwm-stage1-gap1-none-reason-contract-guard.md`
  - `tests/test_trading_app/test_session_orchestrator.py`
  - `trading_app/live/session_orchestrator.py`

## Next Steps — Active

1. **HWM persistence integrity hardening — IN PROGRESS, Stage 1 of 4 LANDED.** Parent design: `docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md` (v3 — passed two design audits, 19 + 8 revisions applied). Stage 1 commits this session:
   - `5b172a44` — feat(live) Stage 1: DD warning tier (50%/75%) reaches operator via Telegram. Wires `self._notify` into the elif-WARN branch at `session_orchestrator.py:1601`. Closes design audit CRITICAL-2.
   - `68c63482` — fix(live) Stage 1 audit-gate fix-up: None-guard the elif (was `"WARN" in reason`, now `reason is not None and "WARN" in reason`). Closes audit-gate CRITICAL-1 (TypeError on None reason silently swallowed by bare except → recreated the very silent-failure mode Stage 1 was meant to close).
   - 8 mutation-proof tests in `TestHWMWarningTierNotifyDispatch` (50/75/generic-WARN/OK/halt-order-pinning + None-guard/update-equity-ordering/check-halt-raises). 174/174 full suite passes. Drift 107/107 + 6 advisory.
   - Stage 1 audit-gate re-fired on `68c63482` per `.claude/rules/adversarial-audit-gate.md`. **Result: CONDITIONAL — Stage 2 CLEARED to proceed.** All 4 prior audit findings CLOSED (CRITICAL-1 None-guard + 3 mutation-proof tests verified by independent code-trace). One new LOW silent gap surfaced (GAP-1 below) — auditor says "may be addressed as the first item in Stage 2 or logged as a deferred finding... must not be silently dropped." Logged to `docs/ralph-loop/deferred-findings.md` as `STAGE1-GAP-1`. Stage 2 will close it as item-zero before any other Stage 2 work.

2. **Stage 2-4 NOT YET STARTED.** Per design v3 § 5-7 and the audit-gate per-commit cadence:
   - **Stage 2** — tracker integrity package (`trading_app/account_hwm_tracker.py` + tests). Adds `notify_callback` constructor param, 30-day stale-state fail-closed raise (UNGROUNDED operational heuristic per design § 2 — figure borrowed from TopStep inactivity rule, not derived), 24-hour soft warning, corrupt-state notify, poll-recovery notify, persist-IO notify, UNGROUNDED labels on 4 existing constants + 2 new ones. Plus 5 integration scenarios in new file.
   - **Stage 3** — orchestrator integration + pre-session shared reader. **v3 audit added `weekly_review.py:37,46` to scope_lock** as third hidden consumer of `account_hwm_*.json`. Behavior change: corrupt-state message format unification across 3 callers (boolean already correct, message format diverges today).
   - **Stage 4** — `check_topstep_inactivity_window` in `pre_session_check.py` + escalation docs (SILENT-6 S2/S3/S4 undefined, UNSUPPORTED-5/6 R3 numeric values ungrounded — escalate to Ralph's iter 178 audit pass).

3. **Code-review LOW findings on Stage 1 (NOT blockers):** auto-staged HANDOFF (process-doc gap, hook behavior); 6-line in-code comment block at `session_orchestrator.py:1602-1607` is verbose. Cosmetic only — defer to optional cleanup commit (rejected as Stage 1 fix-up scope per option-(c)-was-wrong audit).

4. **Parallel session activity (DO NOT TOUCH):**
   - **Ralph Loop v5.2 burndown plan** at `docs/plans/2026-04-25-ralph-crit-high-burndown-v5.md` is active. Ralph closed F4 (`87dffa38`), R1 (`6dafda10`), R3 (`64d0952d`), C1 (`f8f993b7`) since I last saw HANDOFF. Remaining Ralph iters: 178 (audit), 179 (Pass-one shutdown helper), 181 (R4 live-signals rotation), 183 (R5 heartbeat re-notify), 185 (F7 fill poller), 187 (Pass-three magic numbers drift check), 188 (silent-gap S2 cleanup). Ralph's untracked stage doc `docs/runtime/stages/ralph-iter-179-pass-one-hardening.md` is in working tree.
   - **Codex parallel-session WIP** — 6 Python files dirty in working tree (CRLF-pinned via `.gitattributes`). Includes `scripts/tools/context_views.py`, `tests/test_tools/test_*.py`, `tests/test_trading_app/test_phase_4_discovery_gates.py`, `trading_app/phase_4_discovery_gates.py`, `trading_app/strategy_discovery.py`. Don't `git add --renormalize "*.py"` until Codex is clear. Plus untracked `docs/plans/2026-04-25-claude-code-global-hardening-design.md`.

5. **Deferred audit findings (still pending — partly absorbed by Ralph):**
   - **R1/R3** — Ralph closed both. Verify in his commits.
   - **F7/R4/R5** — queued in Ralph's plan; do not duplicate.
   - **Falsy-or pattern leak (7 sites)** — still mine. `auth.py:115`, `contract_resolver.py:75`, `positions.py:66/105`, `data_feed.py:78`, `bot_dashboard.py:1519`, `session_orchestrator.py:797`. Recommend a drift check; not in current Stage 1-4 scope.
   - **Stale `data/state/account_hwm_20092334.json`** — 19 days old at design date (verify with `ls -la` post-compact). Stage 2's 30-day fail-closed raise will trigger on this file once Stage 2 lands. Operator must archive (`.STALE_<YYYYMMDD>.json` suffix) or delete before Stage 2 deploys against account 20092334.

6. **O-SR debt** — `trading_app/live/cusum_monitor.py` implements CUSUM Eq 3, not Shiryaev-Roberts Eq 10 per `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`. Multi-stage; not autonomous; not in current scope.

7. **Live XFA process from prior session is DEAD.** PID 56464 died Saturday ~12:29 Brisbane (correct behavior — feed went stale on weekend close, reconnect loop continued, then process exited). Markets reopen Mon 07:00 Brisbane. No live process running as of compact.

8. **`test_multi_instrument_locks`** — re-runnable now that PID 56464 is gone.

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
