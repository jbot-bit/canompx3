# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-04-25
- **Commit:** e02c529d — fix(live): close 5 silent-failure paths before unattended demo run
- **Files changed:** 4 files
  - `HANDOFF.md`
  - `docs/runtime/stages/live-overnight-resilience-hardening.md`
  - `tests/test_trading_app/test_session_orchestrator.py`
  - `trading_app/live/session_orchestrator.py`

## Next Steps — Active

1. **DEMO LAUNCH READY (next market session)** — `--demo` preflight 5/5 against `topstep_50k_mnq_auto`. AC sleep disabled. 5 silent-failure paths just hardened (commit `e02c529d`). Launch when markets reopen Mon 07:00 Brisbane (CME futures) → fires Tokyo/Singapore/Europe/NYSE_Open/COMEX/US_Data_1000 across the day. Command: `cd C:/Users/joshd/canompx3 && mkdir -p logs/live && PYTHONPATH=. python scripts/run_live_session.py --profile topstep_50k_mnq_auto --instrument MNQ --demo > logs/live/demo_$(date +%Y%m%d_%H%M).log 2>&1`. **Note:** demo account is a Trading Combine (50KTC-V2-451890-20372221), not XFA — F-1 auto-disables on broker-reality check, other gates active (max_daily_loss=-5R/~$285, max_concurrent=3, max_dd≈-35R/$2000).
2. **Deferred resilience findings (not blocking demo)** — 6 issues found by parallel scout agents, deferred per stage `live-overnight-resilience-hardening.md`:
   - **F4 (CRITICAL):** bracket submit failure post-fill leaves position naked. Touches execution path; multi-stage fix.
   - **F7 (HIGH):** fill poller stuck PENDING consumes lane concurrency slot indefinitely. Needs poller-level retry-with-give-up.
   - **R1 (CRITICAL):** trading-day rollover only fires from `_on_bar`. Feed-down at 09:00 Brisbane = rollover delayed; new wall-clock task needed.
   - **R3 (HIGH):** `ORCHESTRATOR_MAX_RECONNECTS=5` too low for 24h demo. Easy bump but should also persist last-success.
   - **R4 (HIGH):** `live_signals.jsonl` unbounded growth. Daily-suffix or RotatingFileHandler.
   - **R5 (HIGH):** engine circuit-breaker silent until rollover. Needs periodic re-notify in heartbeat.
3. **O-SR debt** — `trading_app/live/cusum_monitor.py` implements CUSUM Eq 3, not Shiryaev-Roberts Eq 10 per `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`. Multi-stage; not autonomous.
4. **Codex parallel-session WIP drain** — 6 Python files still dirty in working tree. LF-pinned via `.gitattributes`; current CRLF state preserved until Codex commits/drops. Don't `git add --renormalize "*.py"` until Codex is clear.

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
