# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-14
- **Commit:** 32c2065d — harden: verdict-vocab drift check — fix runner-crash class + intersection-order + layout-lock
- **Files changed:** 2 files
  - `pipeline/check_drift.py`
  - `tests/test_pipeline/test_check_drift.py`

## This Session (2026-05-13 PM)
- Token-efficient code review (Sonnet) found a LOW `BrokerDispatcher.supports_sequential_bracket_ids()` delegation gap — committed `a6e79c6b`. Also refreshed 316 `validated_setups.last_trade_day` rows (2026-05-07 → 2026-05-12) via inline python (Sonnet violated integrity-guardian § 2; canonical migration `scripts/migrations/backfill_validated_trade_windows.py` reproduces identical state; `--dry-run` shows `drifted=0`).
- Adversarial audit (Opus) on the Sonnet commit caught: (a) stale class docstring claiming `BrokerDispatcher` is wired as top-level router (zero production construction sites — Stage 2 multi-broker plan never wired), (b) zero companion tests for either delegation method, (c) inline-python integrity violation. Fixed in `aef0cf2e` (docstring + 4 mutation-proof tests).
- Blast-radius audit on `aef0cf2e` found three more same-class API-parity bugs on `BrokerDispatcher`: `is_degraded` was `@property` but base/orchestrator call it as method (`TypeError` would fire if dispatcher ever wired), `degraded_accounts()` missing override (would silently return `{}`), `verify_bracket_legs()` missing override (would misread as MISSING legs CRITICAL alarm). Fixed in `612bf331` (property→method + 2 overrides + 5 mutation-proof tests).
- Net: 9 new tradovate tests (63→72), all 126 drift checks pass, 247 sibling tests (`session_orchestrator` + `copy_order_router`) green. Class now fully API-aligned with `BrokerRouter` base + `CopyOrderRouter` peer. Production unaffected — `BrokerDispatcher` has zero live callsites.
- Memory: `feedback_code_review_dead_class_detection.md` added (grep for `ClassName\(` construction sites before grading dead-code severity).

## Next Steps — Active
1. **MGC LONDON_METALS — DO NOT RE-LITIGATE.** Verdict frozen at `docs/audit/results/2026-05-12-mgc-london-metals-mode-a-k1-revalidation.md`. Reopen only if new evidence clears one of the prereg kill criteria (K1 t_IS≥3.00 with theory grant, or K3 N_IS_on≥100). Do not re-run Phase A on alternative apertures as a back-door — that pattern is the trap.
2. **Highest-EV next is MNQ.** Live: 2 deployed MNQ E2 RR1.5 lanes (COMEX_SETTLE OVNRNG_100 N=150 annual_r=36.2 + US_DATA_1000 VWAP_MID_ALIGNED_O15 N=112 annual_r=27.1) per `docs/runtime/lane_allocation.json`. L1 NYSE_OPEN_E2_RR1.5_CB1_COST_LT12 paused (PR #271). Concrete candidates: (a) rank-3 AUDIT_GAP_ONLY VWAP_MID_ALIGNED_O30 pre-reg authoring per Chordia v2 readouts, (b) trade-book drift check (MEMORY index lists 3 deployed; canonical lane_allocation.json shows 2 — reconcile).
3. **Pre-existing carry-over (still open):** Track D MNQ COMEX_SETTLE Gate 0 runner design (Databento top-of-book table + bounded runner for DESIGN_ONLY prereg); deployment-coverage decision on 78 ROUTABLE_DORMANT strategies (`docs/audit/results/2026-05-12-deployment-coverage-orphans.md`).
4. **NUGGET 5 PARKED 2026-05-13.** Agent-control-plane evaluation (Paperclip / amux / Cogpit / OctoAlly / LONA / reasoning sidecar) marked PARKED in `docs/plans/2026-05-12-agent-control-plane-evaluation.md`. Reopen only if worktree/branch/PR cleanup exceeds 2 hrs/week for two consecutive weeks. Existing worktree-manager + 5 MCPs + 11 subagents + 27 skills + 17 hooks already constitutes a control plane; NUGGET 4 (commit `b90c6291`) addressed the actual bottleneck (session-start context load). Do not re-evaluate without the reopen trigger firing.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
