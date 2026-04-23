# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Gemini
- **Date:** 2026-04-23
- **Commit:** PENDING
- **Summary:** Implemented the "Gemini Capability and Orchestration Plan" (docs/plans/2026-04-23-gemini-capability-plan.md). Locked the MNQ COMEX_SETTLE Gate 0 microstructure prereg. Implemented PR48 Phase 2 native conditional-role surface (RoleResolver) in execution_engine.py and paper_trader.py for durable shadow logging.
- **Audit Addendum:** No regressions in execution engine or paper trader tests. Drift check passed.

## Next Steps — Active
1. Do not reopen `mnq_parent_structure_shadow_buckets_v1`. Exact-parent structure shadow buckets for these MNQ lanes are now closed `KILL` and should not be rescued under renamed score language.
2. PR48 is no longer a pooled promotion story. Current truth is narrower: `MGC:cont_exec` is still the strongest live branch, but it now has one exact next move only: `shadow_only` overlay contract first; `MES:q45_exec` still needs a bridge; `DUO` and `MNQ:shadow_addon` remain shadow-only.
3. PR48 Phase 2 implemented (RoleResolver). The next move is operator observation of the MGC shadow context in live logs/dashboard.
4. Keep pulse/ralph/handoff surfaces aligned as each thread closes so finished work does not linger as fake backlog.
5. Do not reopen broad GC proxy exploration from the MGC payoff-compression result; if revisited, keep it to a narrow MGC exit-shape prereg.
6. Do not reopen the L1 EUROPE_FLOW pre-break path with banned `break_*` or ATR-normalized replacement variants; the restored frozen `K=2` family is now a documented `KILL`.
7. Do not reopen NYSE_CLOSE filter shopping. Current truth is narrower and already actioned: `ORB_G8` is dead as a filter, while raw `NYSE_CLOSE RR1.0` is alive specifically as a free-slot additive allocator candidate.
8. Treat the `.4R` MES participation note as descriptive only. The exact 20-cell family is killed; do not let sub-threshold OOS deltas sneak back in as pseudo-survivors.
9. Discovery routing is now supposed to flow through `docs/prompts/INSTITUTIONAL_DISCOVERY_PROTOCOL.md` via the shared `discover` skill, Codex research workflow notes, and the context resolver `research_discovery` route. If discovery starts drifting back into ad hoc scans or stale skill text, that wiring regressed.
10. Prior-day bridge work is no longer missing locks. The next honest move on that branch is execution / triage among already-locked hypotheses, not another broad prior-day prereg-writing pass.
11. Do not describe `lane_allocation.json` as unconditional live truth. For audit claims, pair it with allocator replay / SR-liveness context.
12. Do not cite `docs/audit/results/2026-04-21-ovnrng-allocator-routing.md` without the rolling-CV retraction. Current truth is the router `KILL`, not the earlier single-fold positive.
13. Track D MNQ COMEX_SETTLE Gate 0 prereg is locked. The next move is execution of `research/research_mnq_e2_microstructure_pilot.py` (once written/adapted) against the locked hypothesis.
14. Shadow-bucket re-audit reproduced cleanly from canonical data. Keep the verdict narrow: exact-parent `MNQ COMEX_SETTLE RR1.5 long PD_CLEAR_LONG` and `MNQ US_DATA_1000 O15 RR1.5` prior-day structure score buckets are `KILL` for the tested `shadow_only` parent-value role, but that is not a global kill of the broader prior-day geometry family. The checked-in result doc was refreshed only to fix stale prereg provenance metadata after the later stamp commit.

## Blockers / Warnings
- Worktree remains intentionally dirty with unrelated in-flight threads; do not revert them blindly.
- Codex push path is healthy over HTTPS (`origin` + `gh auth status`); SSH does not need to be part of this workflow.

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
- `docs/audit/results/2026-04-23-htf-prior-structure-confluence-repo-audit.md`
- `docs/audit/results/2026-04-21-ovnrng-router-rolling-cv.md`
- `docs/audit/hypotheses/2026-04-23-mnq-parent-structure-shadow-buckets-v1.yaml`
- `docs/audit/results/2026-04-23-mnq-parent-structure-shadow-buckets-v1.md`
- `research/mnq_parent_structure_shadow_buckets_v1.py`
- `docs/plans/2026-04-23-microstructure-gate0-design.md`
- `docs/plans/2026-04-23-research-pipeline-sync.md`
