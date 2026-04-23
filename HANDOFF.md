# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Codex
- **Date:** 2026-04-23
- **Summary:** Closed stale control-state and MES/MGC pipeline debt, implemented PP-167, compacted the baton, fixed trader-logic holdout recompute drift, brought health_check back to truthful green end-to-end, executed the ranked PR48 frozen rel-vol sizing replay (`MGC` candidate, `MES` not ready), explicitly closed the already-landed MGC 5-minute payoff-compression audit as actioned work with a narrow exit-shape-only follow-up scope, restored the missing L1 EUROPE_FLOW frozen prereg and ran it to an honest `KILL`, then executed the MNQ NYSE_CLOSE RR1.0 governance follow-up to an honest `CONTINUE with narrow prereg` and froze the exact next step as `MNQ NYSE_CLOSE ORB_G8 RR1.0`.
 - **Update:** The exact `MNQ NYSE_CLOSE ORB_G8 RR1.0` prereg has now also been executed and closed `KILL`. The strongest native filter path is dead on era stability, but the broad RR1.0 NYSE_CLOSE family is still not dead. The remaining honest branch is a role audit, not another filter rescue.
 - **Update:** The `MNQ NYSE_CLOSE RR1.0` role audit is now also executed and closed into a durable result doc. Current truth: the free-slot additive allocator framing is alive (`annualized R +9.5`, honest Sharpe `+0.136`, candidate-to-book daily corr `+0.023` on the common IS window), so the next honest move is policy / allocator follow-through, not more NYSE_CLOSE filter hunting.
 - **Update:** Removed the advisory M2.5 staged-file scan from `.githooks/pre-commit`; commit-time hooks now stop at the real local gates (lint/format/drift/tests/behavioral/syntax) instead of hanging on an extra token-heavy second-opinion step.
 - **Update:** Added a lightweight anti-bias / grounding upgrade. Claude project settings now run a compact `UserPromptSubmit` guard for research-review-deploy prompts, the shared agent layer now includes `.claude/agents/evidence-auditor.md` for separate-context claim scrutiny, Codex project routing now mirrors the same `CLAIMS not evidence`, `MEASURED/INFERRED/UNSUPPORTED`, and repo-local/primary-source grounding defaults, and GitHub now has a compact PR template requiring `Evidence`, `Claims`, `Disconfirming Checks`, and `Grounding` sections.
 - **Update:** Simplified that anti-bias layer so it is cheaper and more durable: the Claude reminder now runs from tracked `scripts/tools/bias_grounding_guard.py` as a single-line prompt-time guard, there is now a tracked `scripts/tools/check_claim_hygiene.py` reused by CI and local hooks, pre-commit now adds a fast staged-result-doc claim-hygiene gate, and the compact institutional-review rubric lives in `docs/prompts/INSTITUTIONAL_RESEARCH_REVIEW_MINI.md` so operators can reuse it without re-pasting large prompts.
 - **Update:** Ran a last-two-days action-gap closeout. Meaningful research from the 2026-04-21..2026-04-23 window is now either closed into durable repo surfaces or saved as an explicit bounded stage; it should not be treated as stranded branch-local knowledge anymore.

## Next Steps — Active
1. Move to the next ranked open queue item: `docs/runtime/stages/prior-day-pathway-b-hot-cell-prereg.md`.
2. Keep pulse/ralph/handoff surfaces aligned as each thread closes so finished work does not linger as fake backlog.
3. Do not treat PR48 as a pooled `MES/MGC` promotion story anymore; use the 2026-04-23 result doc as the current truth.
4. Do not reopen broad GC proxy exploration from the MGC payoff-compression result; if revisited, keep it to a narrow MGC exit-shape prereg.
5. Do not reopen the L1 EUROPE_FLOW pre-break path with banned `break_*` or ATR-normalized replacement variants; the restored frozen `K=2` family is now a documented `KILL`.
6. Do not reopen NYSE_CLOSE filter shopping. Current truth is narrower and already actioned: `ORB_G8` is dead as a filter, while raw `NYSE_CLOSE RR1.0` is alive specifically as a free-slot additive allocator candidate.
7. Prior-day bridge work is no longer missing locks. The next honest move on that branch is execution / triage among already-locked hypotheses, not another broad prior-day prereg-writing pass.

## Blockers / Warnings
- Worktree remains intentionally dirty with unrelated in-flight threads; do not revert them blindly.

## Durable References
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
- `docs/plans/2026-04-22-recent-pr-followthrough-queue.md`
- `docs/plans/2026-04-19-pp167-per-session-instrument-cap.md`
- `docs/plans/2026-04-21-post-stale-lock-action-queue.md`
- `docs/runtime/stages/pr48-mes-mgc-sizer-rule-backtest.md`
- `docs/audit/results/2026-04-23-pr48-mes-mgc-sizer-rule-backtest-v1.md`
- `docs/runtime/stages/mgc-5m-payoff-compression-audit.md`
- `docs/audit/results/2026-04-19-mgc-payoff-compression-audit.md`
- `docs/runtime/stages/l1-europe-flow-pre-break-context-scan.md`
- `docs/audit/hypotheses/2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml`
- `docs/audit/results/2026-04-21-l1-europe-flow-pre-break-context-prereg.md`
- `docs/runtime/stages/mnq-nyse-close-rr10-followup.md`
- `docs/audit/results/2026-04-23-mnq-nyse-close-rr10-followup.md`
- `docs/audit/hypotheses/2026-04-23-mnq-nyse-close-orbg8-rr10-prereg.yaml`
- `docs/runtime/stages/mnq-nyse-close-orbg8-rr10-prereg.md`
- `docs/audit/results/2026-04-23-mnq-nyse-close-orbg8-rr10-prereg.md`
- `docs/runtime/stages/mnq-nyse-close-rr10-role-audit.md`
- `docs/audit/results/2026-04-23-mnq-nyse-close-rr10-role-audit.md`
- `docs/plans/2026-04-23-last-two-days-action-gap-audit.md`
- `docs/handoffs/archived/2026-04-23-root-handoff-archive-4.md`
