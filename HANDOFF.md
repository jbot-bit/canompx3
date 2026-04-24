# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-04-25
- **Commits:**
  - `5be52bdc` fix(live): notify() returns bool so preflight self-test can detect broken Telegram
  - (fast-forwarded from `b660502b` → `43a123ad` via `git pull --ff-only` to pick up PR #100 symmetric `is_pid_alive` rename + test teeth)
- **Plan:** `C:/Users/joshd/.claude/plans/i-wnat-live-going-squishy-sparkle.md` — v4 with gates P−1..P4, checkpoint log appended through P4.
- **Signal-only smoke test:** launched `--profile topstep_50k_mnq_auto --signal-only` at 18:06:21 Brisbane, ran ~6h continuous, clean Ctrl-C at ~00:07 Brisbane (2026-04-25). 330+ bars received at 1/min, heartbeat cadence 30-min clean, B3 canary negative. Bars flushed via `post_session()` to `bars_1m` MAX(ts_utc)=`2026-04-25 00:07:00+10:00`.

## Next Steps — Active

1. **B6 — F-1 EOD balance never seeded in signal-only mode (REAL BLOCKER for full entry-path validation).** Discovered this session from `live_signals.jsonl` REJECT history: every entry since 2026-04-15 rejected with `"risk_rejected: topstep_scaling_plan: EOD XFA balance unknown — refusing entry"`. NYSE_OPEN 23:40 Brisbane 2026-04-24 confirmed live. Root cause at `trading_app/live/session_orchestrator.py:568-599`: `_apply_broker_reality_check` (which seeds F-1 EOD balance) is gated on `initial_equity is not None` from `positions.query_equity(account_id=0)` — returns None in signal-only because `order_router=None`. "will init on first poll" comment on L598 is false — the poll path is the same. Memory said F-1 dormant; actually fully wired + fail-closing. Candidate fixes (pick via stage-gate): (a) seed from profile `account_size` when broker equity is None + `topstep_xfa_account_size` is set; (b) disable_f1 automatically in signal-only. (a) preserves F-1 semantics for demo/live; (b) is a signal-only-specific bypass.
2. **Push `5be52bdc`** — currently 1 commit ahead of `origin/main`. No conflicts expected.
3. **O-CRLF debt** — `autocrlf=true` (global) vs `autocrlf=false` (local) → 6 files recurring CRLF-only diff (`scripts/tools/context_views.py`, 4× test files, `trading_app/phase_4_discovery_gates.py`, `trading_app/strategy_discovery.py`). Pin with `.gitattributes text eol=lf` on `*.py`. Low priority cosmetic.
4. **O-SR debt** — `trading_app/live/cusum_monitor.py` implements CUSUM Eq 3, not Shiryaev-Roberts Eq 10 per `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`. Paper argues SR is strictly better for live drift detection. Upgrade requires new class + threshold `A` calibration (literature gap G2).
5. **stash@{0} `canompx3-pre-pr100-ff-2026-04-24`** — parked reversible safety net before ff-pull. Contents verified to be 7× CRLF drift + 1× autoformatter cosmetic collapses (0 semantic lines via `git diff -w`). Safe to drop when convenient.

## Blockers / Warnings

- Memory entry calling F-1 "dormant (F-1 hard gate dormant)" is FACTUALLY WRONG — F-1 is active and fail-closing. Correct before next research/deployment decision.
- `build_live_portfolio()` is DEPRECATED — `--all --signal-only` without `--profile` hard-fails all 3 orchestrators. Only the `--profile` path works for live. This is surfaced at `multi_runner.py:79-81` as a log-and-drop silence (D2 in audit).
- Drift check pre-existing: #4 (`work_queue.py` schema parser false-positive on `table 'the'`) and #59 (MNQ 1 day with `!= 3 daily_features` rows) — neither blocked today's work.

## Verified clean this session

- `is_pid_alive` public in both `trading_app/live/instance_lock.py` and `pipeline/db_lock.py` (via PR #100).
- `PerformanceMonitor` + `CUSUMMonitor` wired at `session_orchestrator.py:474, 1481` (v3 audit false-positive corrected).
- `CircuitBreaker` wired at `session_orchestrator.py:648` + 6 call sites (v3 audit false-positive corrected).
- Signal path end-to-end: bar feed → ORB build → break detect → filter check → F-1 gate (rejects, but pipeline intact). `live_signals.jsonl` schema observed: `SESSION_START`, `SIGNAL_ENTRY`, `SIGNAL_EXIT`, `REJECT` events.
- B2 notifications: now returns `bool`; preflight check 5 detects a broken Telegram loudly.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
- `docs/runtime/stages/live-b2-notifications-bool-return.md` (B2 stage doc)
- `C:/Users/joshd/.claude/plans/i-wnat-live-going-squishy-sparkle.md` (v4 plan + checkpoint log)
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` (O-SR grounding)
- `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` (ORB premise)
