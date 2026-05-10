# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Codex (WSL)
- **Date:** 2026-05-11
- **Commit:** saved in latest git history as `fix(deployability): ground MNQ live readiness gates`
- **Files changed:** 20 files
  - `HANDOFF.md`
  - `docs/audit/hypotheses/2026-05-10-mnq-comexsettle-orbg5-rr10-chordia-unlock-v1.yaml`
  - `docs/audit/hypotheses/2026-05-10-mnq-comexsettle-pdclearlong-rr10-chordia-unlock-v1.yaml`
  - `docs/audit/hypotheses/2026-05-10-mnq-nyseopen-xmesatr60-rr10-chordia-unlock-v1.yaml`
  - `docs/audit/hypotheses/2026-05-10-mnq-usdata1000-pdgolong-rr10-chordia-unlock-v1.yaml`
  - `docs/audit/results/2026-05-10-clean-long-stop-mnq-comex-settle-e2-rr1.0-cb1-pd-clear-long.csv`
  - `docs/audit/results/2026-05-10-clean-long-stop-mnq-comex-settle-e2-rr1.0-cb1-pd-clear-long.md`
  - `docs/audit/results/2026-05-10-clean-long-stop-mnq-us-data-1000-e2-rr1.0-cb1-pd-go-long.csv`
  - `docs/audit/results/2026-05-10-clean-long-stop-mnq-us-data-1000-e2-rr1.0-cb1-pd-go-long.md`
  - `docs/audit/results/2026-05-10-mnq-comexsettle-orbg5-rr10-chordia-unlock-v1.csv`
  - `docs/audit/results/2026-05-10-mnq-comexsettle-orbg5-rr10-chordia-unlock-v1.md`
  - `docs/audit/results/2026-05-10-mnq-comexsettle-pdclearlong-rr10-chordia-unlock-v1.csv`
  - `docs/audit/results/2026-05-10-mnq-comexsettle-pdclearlong-rr10-chordia-unlock-v1.md`
  - `docs/audit/results/2026-05-10-mnq-nyseopen-xmesatr60-rr10-chordia-unlock-v1.csv`
  - `docs/audit/results/2026-05-10-mnq-nyseopen-xmesatr60-rr10-chordia-unlock-v1.md`
  - ... and 5 more

## Next Steps — Active
1. Convert MNQ controlled-pilot shelf pool into a profile-safe paper/sandbox proposal — Use `docs/audit/results/2026-05-11-mnq-all-active-deployability.json` as
the candidate input, not a fresh signal search. The all-active audit now
finds 177 `CONTROLLED_LIVE_PILOT_CANDIDATE` MNQ rows after the routine
MNQ E2 TBBO slippage metadata gap was fixed. This queue item is the
bounded production-readiness translation layer:

1. Deduplicate same-family / same-session / same-filter variants.
2. Run add/replace/correlation against the current selected
   `topstep_50k_mnq_auto` profile.
3. Enforce account-risk, one-position/session, SR, allocator Chordia,
   replay, OOS, and execution constraints.
4. Emit a paper/sandbox-only proposed profile change set, or an explicit
   no-change verdict.

Do not mutate `docs/runtime/lane_allocation.json`, broker/live state,
schema, or deployment DB state in this item. The 177 rows are not direct
live routes.

2. PR48 MES q45_exec bridge — Define the honest bridge from the alive MES q45_exec research branch into a bounded runtime surface.
3. PR48 MGC shadow-only observation closeout — Observe the MGC shadow-only context in dashboard and live logs and record whether the visibility path behaves as designed.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
