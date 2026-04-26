# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-04-26
- **Commit:** da2c4dfb — [judgment] audit: Ralph Loop iter 178 — adversarial audit R3+C1 PASS, Stage 2 cleared
- **Files changed:** 5 files
  - `HANDOFF.md`
  - `docs/ralph-loop/ralph-ledger.json`
  - `docs/ralph-loop/ralph-loop-audit.md`
  - `docs/ralph-loop/ralph-loop-history.md`
  - `docs/ralph-loop/ralph-loop-plan.md`

## Next Steps — Active

1. **HWM persistence integrity hardening — ALL 4 STAGES LANDED.** Parent design: `docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md` (v3 — passed two design audits, 19 + 8 revisions applied). All four stages commit-and-audit-gate complete this session.

   **Stage 1 — DD warning tier reaches operator (LANDED + audit-gate CONDITIONAL → closed):**
   - `5b172a44` — feat: wires `self._notify` into elif-WARN branch at `session_orchestrator.py:1601`. Closes design audit CRITICAL-2.
   - `68c63482` — fix-up: None-guard the elif. Closes audit-gate CRITICAL-1.
   - `45720109` — fix-up: STAGE1-GAP-1 None-reason contract-drift visibility.
   - `df00589b` — docs: replace placeholder commit hash with literal `45720109`.

   **Stage 2 — tracker integrity package (LANDED + audit-gate CONDITIONAL → closed):**
   - `1f29009b` — feat: tracker integrity package (`notify_callback` ctor param, 30-day stale-state raise, 24h soft warning, corrupt-state notify, poll-recovery notify, persist-IO notify, UNGROUNDED+Rationale annotations on 6 constants, `state_file_age_days` pure helper, `@canonical-source` block on class docstring).
   - `e67f46f6` — Stage 2 audit-gate fix-up: closes SG1 (null-timestamp gate bypass), SG3 (post-halt RECOVERY misleading), SG4 (NaN equity bypasses halt).
   - `c5be3453` — SG-NEW-1 fix: reject `bool` in `_is_finite_equity` (Python `True is int`).

   **Stage 3 — orchestrator wiring + EOD dispatch + shared reader (LANDED + audit-gate CONDITIONAL → closed):**
   - `1de0f17f` — feat: `notify_callback=self._notify` wired at construction; signal-only authority comment; EOD silent-skip resolved on BOTH branches (`end_equity is None` AND `Exception`) with kill-switch suppression; new `read_state_file(path)` shared helper; pre_session × 2 + weekly_review converted to delegate; unified `BLOCKED <filename>:` message format.
   - `cc20f52c` — Stage 3 audit-gate fix-up: closes C-1 (Scenario 6 added to `test_account_hwm_tracker_integration.py`) + TM-2 (path-in-warning assertions added to all 5 read_state_file None-return tests). C-2 (HANDOFF closure) deferred to Stage 4 per design v3 § 7 — closed by this commit.

   **Stage 4 — TopStep inactivity-window pre-flight + HANDOFF closure (LANDED — see this commit):**
   - This commit — feat: `check_topstep_inactivity_window` in `pre_session_check.py` delegating to Stage 2's `state_file_age_days` (single source of truth). Warn ≥25 days, block ≥30 days. 12 mutation-proof tests. Wires into `run_checks`. Three deferred-findings rows added (HWM-SIL6, HWM-UNS5, HWM-UNS6). HANDOFF closure (this update). Stage 3 audit-gate finding C-2 closed.

   **Audit-gate verdict closures:**
   - Stage 1: CONDITIONAL CRITICAL-1 + 3 mutation-tests → CLOSED (`68c63482`).
   - Stage 1 GAP-1 (None-reason contract drift, LOW) → CLOSED (`45720109`).
   - Stage 2: CONDITIONAL SG1/SG3/SG4 + SG-NEW-1 → CLOSED (`e67f46f6`, `c5be3453`).
   - Stage 3: CONDITIONAL C-1/TM-2 → CLOSED (`cc20f52c`); C-2 deferred to Stage 4 → CLOSED (this commit).
   - SILENT-6, UNSUPPORTED-5, UNSUPPORTED-6 → DEFERRED, see `docs/ralph-loop/deferred-findings.md` rows HWM-SIL6, HWM-UNS5, HWM-UNS6.

2. **HWM hardening — POST-LANDING NEXT STEPS:**
   - Dispatch `evidence-auditor` on this Stage 4 commit per `.claude/rules/adversarial-audit-gate.md`. Final stage of the design v3 plan.
   - Operator action on `data/state/account_hwm_20092334.json` (~20 days old): inside the WARN band but well clear of the 30-day BLOCK boundary. No immediate action; the new check will warn at next pre_session run.
   - After Stage 4 audit-gate PASS: design v3 work complete. Move to next priority.

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
