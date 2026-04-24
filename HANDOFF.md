# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-04-25
- **Commit:** ca363e1a — fix(live): seed F-1 XFA EOD balance to $0 in signal-only (closes B6)
- **Files changed:** 4 files
  - `HANDOFF.md`
  - `docs/runtime/stages/live-b6-f1-signal-only-seed.md`
  - `tests/test_trading_app/test_session_orchestrator.py`
  - `trading_app/live/session_orchestrator.py`

## Next Steps — Active

1. **Push current branch** — was 1 commit ahead of `origin/main` (`5be52bdc`); now adds the B6 fix + this baton. No conflicts expected.
2. **O-CRLF debt** — `autocrlf=true` (global) vs `autocrlf=false` (local) → 6 files recurring CRLF-only diff (`scripts/tools/context_views.py`, 4× test files, `trading_app/phase_4_discovery_gates.py`, `trading_app/strategy_discovery.py`). Pin with `.gitattributes text eol=lf` on `*.py`. Low priority cosmetic. **NOTE:** the 6 dirty CRLF-only files are Codex parallel-session WIP — do NOT reset them; coordinate with Codex first.
3. **O-SR debt** — `trading_app/live/cusum_monitor.py` implements CUSUM Eq 3, not Shiryaev-Roberts Eq 10 per `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`. Paper argues SR is strictly better for live drift detection. Upgrade requires new class + threshold `A` calibration (literature gap G2).
4. **Re-run the 6h signal-only smoke test** — with B6 fixed, expect `SIGNAL_ENTRY` events instead of `REJECT` events. Diff `live_signals.jsonl` against the prior run to confirm the fix unblocks entries.

## Blockers / Warnings

- Memory entry calling F-1 "dormant (F-1 hard gate dormant)" is FACTUALLY WRONG — F-1 is active and fail-closing in signal-only PRE-fix; post-fix it enforces against the canonical day-1 $0 cap. Correct before next research/deployment decision.
- `build_live_portfolio()` is DEPRECATED — `--all --signal-only` without `--profile` hard-fails all 3 orchestrators. Only the `--profile` path works for live. This is surfaced at `multi_runner.py:79-81` as a log-and-drop silence (D2 in audit).
- Drift check pre-existing: #4 (`work_queue.py` schema parser false-positive on `table 'the'`) and #59 (**MGC** — corrected from earlier "MNQ" — 1 day with `!= 3 daily_features` rows). Neither blocked today's work.
- stash@{0} `canompx3-pre-pr100-ff-2026-04-24` — DROPPED 2026-04-25 after this-session review verified contents were superseded (HANDOFF text older than HEAD; test_bot_dashboard.py whitespace collapses already absorbed by PR #100).

## Verified clean this session

- `is_pid_alive` public in both `trading_app/live/instance_lock.py` and `pipeline/db_lock.py` (via PR #100).
- `PerformanceMonitor` + `CUSUMMonitor` wired at `session_orchestrator.py:474, 1481` (v3 audit false-positive corrected).
- `CircuitBreaker` wired at `session_orchestrator.py:648` + 6 call sites (v3 audit false-positive corrected).
- Signal path end-to-end: bar feed → ORB build → break detect → filter check → F-1 gate. **B6 closed:** signal-only branch now seeds F-1 EOD balance with $0.0 (canonical day-1 XFA per `topstep_scaling_plan.py:51-53`); F-1 still enforces against bottom-tier cap (2 lots for 50K).
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
