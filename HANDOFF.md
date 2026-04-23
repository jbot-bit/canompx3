# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Codex
- **Date:** 2026-04-23
- **Commit:** `pending publish` — `docs(research): design pr48 mgc shadow overlay contract`
- **Summary:** Froze the next exact PR48 move after the redesign verdict. `MGC:cont_exec` should enter the repo first as a `shadow_only` profile-local conditional overlay, not as a lane and not as live sizing. The chosen contract is a checked-in static overlay spec plus a daily derived-state envelope with pre-session/dashboard visibility only.
- **Audit Addendum:** Do not route this through `validated_setups`, `lane_allocator`, `paper_trades`, or the current execution-time `size_multiplier` hook in Phase 1. The active stage is now `docs/runtime/stages/pr48-mgc-shadow-only-overlay-contract.md`.

## Next Steps — Active
1. Do not reopen `mnq_parent_structure_shadow_buckets_v1`. Exact-parent structure shadow buckets for these MNQ lanes are now closed `KILL` and should not be rescued under renamed score language.
2. PR48 is no longer a pooled promotion story. Current truth is narrower: `MGC:cont_exec` is still the strongest live branch, but it now has one exact next move only: `shadow_only` overlay contract first; `MES:q45_exec` still needs a bridge; `DUO` and `MNQ:shadow_addon` remain shadow-only.
3. Do not reopen generic PR48 confluence discovery. The exact next PR48 move is `docs/runtime/stages/pr48-mgc-shadow-only-overlay-contract.md`; do not jump straight to runtime sizing or schema-wide conditional rebuild.
4. Keep pulse/ralph/handoff surfaces aligned as each thread closes so finished work does not linger as fake backlog.
5. Do not reopen broad GC proxy exploration from the MGC payoff-compression result; if revisited, keep it to a narrow MGC exit-shape prereg.
6. Do not reopen the L1 EUROPE_FLOW pre-break path with banned `break_*` or ATR-normalized replacement variants; the restored frozen `K=2` family is now a documented `KILL`.
7. Do not reopen NYSE_CLOSE filter shopping. Current truth is narrower and already actioned: `ORB_G8` is dead as a filter, while raw `NYSE_CLOSE RR1.0` is alive specifically as a free-slot additive allocator candidate.
8. Treat the `.4R` MES participation note as descriptive only. The exact 20-cell family is killed; do not let sub-threshold OOS deltas sneak back in as pseudo-survivors.
9. Discovery routing is now supposed to flow through `docs/prompts/INSTITUTIONAL_DISCOVERY_PROTOCOL.md` via the shared `discover` skill, Codex research workflow notes, and the context resolver `research_discovery` route. If discovery starts drifting back into ad hoc scans or stale skill text, that wiring regressed.
10. Prior-day bridge work is no longer missing locks. The next honest move on that branch is execution / triage among already-locked hypotheses, not another broad prior-day prereg-writing pass.
11. Do not describe `lane_allocation.json` as unconditional live truth. For audit claims, pair it with allocator replay / SR-liveness context.
12. Do not cite `docs/audit/results/2026-04-21-ovnrng-allocator-routing.md` without the rolling-CV retraction. Current truth is the router `KILL`, not the earlier single-fold positive.
13. Track D is now a documented future system-upgrade branch, not an immediate excuse to abandon current-stack open work. Use `docs/plans/2026-04-23-microstructure-gate0-design.md` for the design truth: start with a cheapest top-of-book Gate 0 on one exact lane, escalate to MBO only if L1/TBBO features prove signal, and do not describe it as “OHLCV exhausted” or as an HFT build.

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
