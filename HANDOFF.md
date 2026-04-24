# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-04-25
- **Commit:** 56233e35 — docs(handoff): record B6 e2e empirical verification (6 lanes ALLOW post-fix)
- **Files changed:** 1 files
  - `HANDOFF.md`

## Next Steps — Active

1. **Live-feed smoke (next market session)** — B6 itself is **empirically verified offline** this session: real `SessionOrchestrator(signal_only=True, profile=topstep_50k_mnq_auto, broker=projectx)` construction confirmed all 6 deployed MNQ lanes ALLOW through the F-1 gate (vs pre-fix REJECT-all). Logged in `Verified clean this session` below. The remaining live-feed value is integration-only: WebSocket bar feed health, ORB build + break detect on real bars, `live_signals.jsonl` SIGNAL_ENTRY write. Run during NYSE_OPEN window when markets are open; weekend = blocked.
2. **O-SR debt** — `trading_app/live/cusum_monitor.py` implements CUSUM Eq 3, not Shiryaev-Roberts Eq 10 per `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`. Paper argues SR is strictly better for live drift detection. Upgrade requires new class + threshold `A` calibration (literature gap G2). Multi-stage; not autonomous.
3. **Codex parallel-session WIP drain** — 6 Python files still dirty in working tree (`scripts/tools/context_views.py`, 4× test files, `trading_app/phase_4_discovery_gates.py`, `trading_app/strategy_discovery.py`). Now LF-pinned via `.gitattributes` so future writes won't re-introduce CRLF. The current dirty diff is the saved-as-CRLF state from another terminal. Coordinate with Codex before resetting/renormalizing — `git add --renormalize "*.py"` would conflict.
4. **One-shot full renormalization (deferred)** — once Codex's WIP is committed/dropped, run `git add --renormalize "*.py" && git commit -m "chore(repo): one-shot LF normalization of *.py"` to flush all historical CRLF in one commit. Until then, individual files normalize on next save.

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
