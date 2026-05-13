# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-13
- **Commit:** 6a9ed048 — hooks: auto-sync Pinecone on knowledge-base commits
- **Files changed:** 2 files
  - `.githooks/post-commit`
  - `HANDOFF.md`

## Next Steps — Active
1. **MGC LONDON_METALS — DO NOT RE-LITIGATE.** Verdict frozen at `docs/audit/results/2026-05-12-mgc-london-metals-mode-a-k1-revalidation.md`. Reopen only if new evidence clears one of the prereg kill criteria (K1 t_IS≥3.00 with theory grant, or K3 N_IS_on≥100). Do not re-run Phase A on alternative apertures as a back-door — that pattern is the trap.
2. **Highest-EV next is MNQ.** Live: 2 deployed MNQ E2 RR1.5 lanes (COMEX_SETTLE OVNRNG_100 N=150 annual_r=36.2 + US_DATA_1000 VWAP_MID_ALIGNED_O15 N=112 annual_r=27.1) per `docs/runtime/lane_allocation.json`. L1 NYSE_OPEN_E2_RR1.5_CB1_COST_LT12 paused (PR #271). Concrete candidates: (a) rank-3 AUDIT_GAP_ONLY VWAP_MID_ALIGNED_O30 pre-reg authoring per Chordia v2 readouts, (b) trade-book drift check (MEMORY index lists 3 deployed; canonical lane_allocation.json shows 2 — reconcile).
3. **Pre-existing carry-over (still open):** Track D MNQ COMEX_SETTLE Gate 0 runner design (Databento top-of-book table + bounded runner for DESIGN_ONLY prereg); deployment-coverage decision on 78 ROUTABLE_DORMANT strategies (`docs/audit/results/2026-05-12-deployment-coverage-orphans.md`).

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
