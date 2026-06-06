---
task: Clear the C11 live-trading gate for topstep_50k_mnq_auto PROPERLY (no gate weakening, no fake readiness) or prove it cannot be cleared safely. C11 regenerated fresh and FAILS strict deployability: worst 90d DD $2,788 > $1,600 budget, 7 historical daily-loss breach days, at 1 micro (minimum size). 12-phase institutional audit: source-of-truth trace, calc-bug hunt, breach-day attribution, book decomposition, read-only scenario testing (lane reduction/swap/overlay/risk-control/account/contract), MTC control, ranked clearance paths, ONE recommendation. Read-only until operator approval.
mode: IMPLEMENTATION
updated: 2026-06-03
acceptance:
  - C11 calc audited; any bug fixed with regression test + rerun
  - Breach-day attribution table (7 days) produced
  - Book decomposition + scenario matrix (families A-F) produced read-only
  - ONE ranked clearance-path recommendation with evidence labels
  - No live_config / lane / contract / threshold mutation before approval
---

## Scope Lock
- docs/runtime/stages/2026-06-03-c11-c12-live-gate-regen.md
- docs/audit/results/2026-06-03-c11-clearance-audit.md (new report)
- scripts/tools/ (read-only scenario runner if needed; new file only)
- trading_app/account_survival.py (ONLY if a real calc bug is found; smallest diff + test)

## Blast Radius
- Read-only by default. The audit READS gold.db (read-only), account_survival, prop_profiles, lane_allocation, cost_model, dst.
- Writes ONLY: this stage file, a new audit report MD, optionally a read-only scenario-runner script under scripts/tools/.
- NO live capital, broker, session-orchestrator, live_config, lane activation, or contract-size mutation before explicit operator approval (Tier B).
- C11/C12 lifecycle state regeneration already done (account_survival ran). That is offline survival evidence, not capital exposure.
- If a genuine calc bug is found in account_survival.py: smallest diff + regression test + rerun + adversarial self-review (truth-layer path).

## Decision class: LIVE_CAPITAL_RISK — stop at patch preview, require approval before any exposure-altering action.
