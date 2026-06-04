# NEXT: C11 Equity-Drawdown Throttle — Validation (read-only resume point)

**Status 2026-06-04 (UPDATED):** VALIDATION DONE — **VERDICT: PASS**. WF-stable band
{600,800,1000}@factor=0.5 (width 3) clears all draft-YAML gates; chosen cell
trigger=$800/factor=0.5/recover=$400 (full max90dDD $1,459, edge 89.3%, WF 5/5,
breach 0). Result: `docs/audit/results/2026-06-04-c11-throttle-validation.md`.
Harness: `C:\Users\joshd\c11_matrix\throttle_validate.py`. NOT implemented (Tier B,
capital path — needs operator approval + throttle-aware MC + gate/live parity).
All work read-only — nothing committed to main.

**Earlier status:** Solution matrix DONE, root cause found, fix identified,
pre-reg drafted. This was the one-shot resume point.

## What's settled (don't re-litigate)
- C11 fails on the strict-DD leg only: max 90d DD **$2,038.84** (reproduced exact),
  fails $1,600 AND the real $2,000 MLL (by $39).
- **Root cause = 2022 regime grind** (Jul-Oct 2022): all 3 lanes WR 43%->25%,
  ExpR +0.18->-0.20, 78-day leg. NOT cap/stop/lane/correlation.
- Gate is CORRECT (Topstep MLL = EOD trailing, verified live vs help.topstep.com).
- Caps/stops/lane-removal/de-correlation all REFUTED by evidence.
- R-multiples: US (0.182) ~ TOKYO (0.179) tied-best; "$-proxy said US best" was a
  position-size illusion.
- **FIX = causal equity drawdown throttle.** Provisional best: trigger=$800,
  factor=0.5, recover=$400 -> DD $1,459 (clears both budgets), ~11% edge loss,
  engages in 5 historical episodes (general, not 2022-fit), no-lookahead.

## Deliverables on disk (this worktree, uncommitted)
- `docs/audit/results/2026-06-04-c11-solution-matrix.md` (claim-hygiene PASS)
- `docs/audit/hypotheses/drafts/2026-06-04-c11-equity-drawdown-throttle.draft.yaml`
- Analysis scripts (outside repo, read-only): `C:\Users\joshd\c11_matrix\*.py`
  (`repro_baseline`, `test_daily_loss_halt_in_dd`, `locate_worst_window`,
  `reconcile_and_rmultiple`, `throttle_test`)

## Next session — validation (still READ-ONLY until operator approves implementation)
1. **k-budget gate FIRST:** research-catalog MCP `estimate_k_budget` with K=8,
   horizon 6.66yr. Must clear Bailey MinBTL before touching the grid.
2. **Extend `throttle_test.py`** to anchored walk-forward (IS 2019-2024) + frozen
   holdout (2025-2026 touched ONCE). Select the WF-STABLE cell, not the IS-best
   cell. If 800/0.5 isn't WF-stable, it's overfit -> disqualify.
3. Apply pass/fail criteria from the draft YAML verbatim (edge floor >=85%, C11
   <=$1,600 AND <=$2,000 holdout-inclusive, band >=2 wide, FDR q=0.05).
4. If it survives: promote draft YAML -> `docs/audit/hypotheses/`, lock date,
   then present implementation plan (NOT before).

## Implementation guardrail (when/if approved — Tier B, capital path)
Single canonical throttle param source consumed by BOTH `account_survival` (gate)
AND `session_orchestrator` (live); reuse `account_hwm_tracker.py` HWM state.
Adversarial-audit-gated. Don't split the source or the gate-vs-live divergence returns.

## Git note
Worktree `canompx3-c11-attribution` @ branch `c11-attribution-analysis` (de1f9089).
Main has live peers — DO NOT touch main. Everything here is uncommitted/read-only.
