# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-13
- **Commit:** d67444b7 — handoff: 2026-05-13 PM — Sonnet review + Opus audit + blast-radius corrections (3 commits)
- **Files changed:** 1 files
  - `HANDOFF.md`

## This Session (2026-05-13 PM)
- Token-efficient code review (Sonnet) found a LOW `BrokerDispatcher.supports_sequential_bracket_ids()` delegation gap — committed `a6e79c6b`. Also refreshed 316 `validated_setups.last_trade_day` rows (2026-05-07 → 2026-05-12) via inline python (Sonnet violated integrity-guardian § 2; canonical migration `scripts/migrations/backfill_validated_trade_windows.py` reproduces identical state; `--dry-run` shows `drifted=0`).
- Adversarial audit (Opus) on the Sonnet commit caught: (a) stale class docstring claiming `BrokerDispatcher` is wired as top-level router (zero production construction sites — Stage 2 multi-broker plan never wired), (b) zero companion tests for either delegation method, (c) inline-python integrity violation. Fixed in `aef0cf2e` (docstring + 4 mutation-proof tests).
- Blast-radius audit on `aef0cf2e` found three more same-class API-parity bugs on `BrokerDispatcher`: `is_degraded` was `@property` but base/orchestrator call it as method (`TypeError` would fire if dispatcher ever wired), `degraded_accounts()` missing override (would silently return `{}`), `verify_bracket_legs()` missing override (would misread as MISSING legs CRITICAL alarm). Fixed in `612bf331` (property→method + 2 overrides + 5 mutation-proof tests).
- Net: 9 new tradovate tests (63→72), all 126 drift checks pass, 247 sibling tests (`session_orchestrator` + `copy_order_router`) green. Class now fully API-aligned with `BrokerRouter` base + `CopyOrderRouter` peer. Production unaffected — `BrokerDispatcher` has zero live callsites.
- Memory: `feedback_code_review_dead_class_detection.md` added (grep for `ClassName\(` construction sites before grading dead-code severity).

## Current Session Addendum
- **Tool:** Codex (WSL)
- **Date:** 2026-05-13
- **Summary:** Implemented a shadow-only opportunity awareness overlay that ranks active profile lanes as `PRIME_SHADOW`, `WATCH`, `BLOCKED`, or `NORMAL` from lifecycle state, lane allocation, regime, Chordia/protocol gates, and trailing expression. Wired it into lifecycle state, pre-session checks, the bot dashboard, deployability audit rendering, and session-start lifecycle logging without changing broker, allocator, DB, sizing, or execution behavior. Follow-up review added lane names/reasons to operator detail and removed the module import-time state-directory write.
- **Status:** Focused tests and lint pass. CLI smoke for `topstep_50k_mnq_auto` on 2026-05-13 reported 2 `PRIME_SHADOW`, 1 `WATCH`, 0 `BLOCKED`; the watch lane was provisional allocation status.
- **Verification:** `pytest` focused overlay/lifecycle/pre-session/dashboard/deployability suite passed (`83 passed`), targeted session-orchestrator lifecycle block tests passed, ruff check/format passed, py_compile passed, behavioral audit passed, integrity audit passed, and full `pipeline/check_drift.py` passed with no blocking drift (`126 checks passed`, `20 advisory`). CodeRabbit CLI review could not run because the CLI was missing and its installer requires `unzip`; installing `unzip` via apt required an interactive sudo password.

## Current Session Addendum — Token Hygiene
- **Tool:** Codex (WSL)
- **Date:** 2026-05-14
- **Summary:** Reduced Claude prompt-hook overhead using official Claude Code cost/hook guidance. Restored `.claude/hooks/prompt-broker.py`, consolidated `UserPromptSubmit` from four hooks to one capped JSON broker, added context-budget rules for `/clear`, `/compact`, subagents, and cheap default effort, and extended `scripts/tools/token_hygiene_report.py` to detect prompt-hook drift/broker absence.
- **Verification:** `python3 -m json.tool .claude/settings.json`, `pytest .claude/hooks/tests/test_prompt_broker.py -q` (`14 passed`), `python3 -m py_compile .claude/hooks/prompt-broker.py scripts/tools/token_hygiene_report.py`, and `python3 scripts/tools/token_hygiene_report.py` all passed. Report now shows `UserPromptSubmit hook commands: 1`.

## Current Session Addendum — Stage File Hygiene
- **Tool:** Codex (WSL)
- **Date:** 2026-05-14
- **Summary:** Conservatively archived 8 closed/completed stage docs to `docs/runtime/stages/archive/2026-05-14/` and aligned token/startup stage reporting with `pipeline.system_context.list_active_stages()`. Token hygiene and WSL doctor now agree on 22 active stage files, down from the raw 36 markdown files.
- **Verification:** `pytest tests/test_pipeline/test_system_context.py tests/test_tools/test_claude_superpower_brief.py tests/test_tools/test_token_hygiene_report.py -q` (`25 passed`), `python3 -m py_compile pipeline/system_context.py scripts/tools/token_hygiene_report.py scripts/tools/claude_superpower_brief.py .claude/hooks/stage-awareness.py .claude/hooks/session-start.py`, `python3 scripts/tools/token_hygiene_report.py`, `python3 scripts/infra/codex_local_env.py doctor --platform wsl`, and `python3 pipeline/check_drift.py` (`126 checks passed`, `20 advisory`) passed.
- **Follow-up cleanup:** Routed `.claude/hooks/stage-awareness.py` and `.claude/hooks/session-start.py` legacy fallback through the canonical active-stage classifier, archived 5 more completed/malformed-top-level stage docs into `docs/runtime/stages/archive/2026-05-14/`, and added valid frontmatter to the still-pending `feat-live-app-ux-smoke.md` operator template. Top-level stage markdown now equals active-stage classifier output (`23` files, `0` hidden).
- **Follow-up verification:** Red/green hook tests added for closed/loose stage filtering. `pytest --import-mode=importlib .claude/hooks/tests/test_stage_awareness.py .claude/hooks/tests/test_session_start_stage_lines.py .claude/hooks/tests/test_prompt_broker.py .claude/hooks/tests/test_main_ci_preflight.py tests/test_pipeline/test_system_context.py tests/test_tools/test_claude_superpower_brief.py tests/test_tools/test_token_hygiene_report.py -q` (`51 passed`), py_compile, `git diff --check`, `python3 scripts/tools/token_hygiene_report.py`, `python3 scripts/infra/codex_local_env.py doctor --platform wsl`, and `python3 pipeline/check_drift.py` (`126 checks passed`, `20 advisory`) passed.

## Current Session Addendum — UI QA Tooling
- **Tool:** Codex (WSL)
- **Date:** 2026-05-16
- **Summary:** Installed the Codex `playwright` and `screenshot` skills into `/home/joshd/.codex/skills/` for browser-based dashboard UI review. Playwright Firefox was downloaded to the user cache; Chrome install was blocked by sudo, so Firefox is the working browser path. Missing `libasound2t64` was handled without sudo by downloading the deb into `/tmp` and extracting the library to `/tmp/canompx3-playwright-libs/`.
- **Artifacts:** Baseline dashboard screenshots saved in `output/playwright/` for desktop and narrow viewports. Internal `.playwright-cli/` scratch output was moved to `/tmp/canompx3-playwright-cli-20260516-baseline`.
- **Continue after `/clear`:** `docs/plans/2026-05-16-ui-redesign-continuation.md`.
- **Status:** Dashboard browser smoke loaded `http://127.0.0.1:18082/` with title `ORB Bot Dashboard`; Playwright console reported `0` errors and `0` warnings. Pre-existing uncommitted edits remain in `scripts/run_live_session.py`, `tests/test_scripts/test_run_live_session_preflight.py`, `tests/test_trading_app/test_bot_dashboard.py`, `trading_app/live/bot_dashboard.py`, and `trading_app/live/bot_dashboard.html`.

## Current Session Addendum — Codex Launch + Dashboard Redesign
- **Tool:** Codex (WSL)
- **Date:** 2026-05-16
- **Summary:** Diagnosed `codex.bat` launch warnings. `.venv-wsl` and the WSL-home repo are present; normal launcher now pins the repo `canompx3` profile so `codex.bat` opens on the stable repo model instead of falling through to global `gpt-5.5`. Removed stale local Gmail/Calendar MCP entries from `/home/joshd/.codex/config.toml`; they pointed at a missing Windows Organisation path and caused the broken-pipe startup failures.
- **Dashboard:** Continued the `START_BOT.bat` app/dashboard redesign in `trading_app/live/bot_dashboard.html`. The pass flattens the theme away from glass/neon/free-template styling, removes duplicated broker-blocked CTAs, makes broker readiness one action surface, improves mobile control order, and preserves route/API behavior.
- **Artifacts:** Final screenshots saved in `output/playwright/`: `bot-dashboard-redesign-final-desktop-1440x1100.png` and `bot-dashboard-redesign-final-narrow-390x900.png`. Internal `.playwright-cli/` scratch output moved to `/tmp/canompx3-playwright-cli-20260516-redesign`.
- **Verification:** `pytest tests/test_tools/test_codex_launcher_scripts.py tests/test_trading_app/test_bot_dashboard.py -q` (`26 passed`), `git diff --check`, Playwright console (`0` errors, `0` warnings), and `python3 scripts/infra/codex_local_env.py doctor --platform wsl` ran. Doctor still warns that global `/home/joshd/.codex` defaults to `gpt-5.5`, but managed `codex.bat` normal sessions now pass the repo profile.

## Next Steps — Active
1. **MGC LONDON_METALS — DO NOT RE-LITIGATE.** Verdict frozen at `docs/audit/results/2026-05-12-mgc-london-metals-mode-a-k1-revalidation.md`. Reopen only if new evidence clears one of the prereg kill criteria (K1 t_IS≥3.00 with theory grant, or K3 N_IS_on≥100). Do not re-run Phase A on alternative apertures as a back-door — that pattern is the trap.
2. **Highest-EV next is MNQ.** Live: 2 deployed MNQ E2 RR1.5 lanes (COMEX_SETTLE OVNRNG_100 N=150 annual_r=36.2 + US_DATA_1000 VWAP_MID_ALIGNED_O15 N=112 annual_r=27.1) per `docs/runtime/lane_allocation.json`. L1 NYSE_OPEN_E2_RR1.5_CB1_COST_LT12 paused (PR #271). Concrete candidates: (a) rank-3 AUDIT_GAP_ONLY VWAP_MID_ALIGNED_O30 pre-reg authoring per Chordia v2 readouts, (b) trade-book drift check (MEMORY index lists 3 deployed; canonical lane_allocation.json shows 2 — reconcile).
3. **Pre-existing carry-over (still open):** Track D MNQ COMEX_SETTLE Gate 0 runner design (Databento top-of-book table + bounded runner for DESIGN_ONLY prereg); deployment-coverage decision on 78 ROUTABLE_DORMANT strategies (`docs/audit/results/2026-05-12-deployment-coverage-orphans.md`).

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
- `docs/plans/2026-05-16-ui-redesign-continuation.md`
