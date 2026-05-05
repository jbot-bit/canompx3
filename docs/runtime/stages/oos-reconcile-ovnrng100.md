---
task: OOS reconciliation diligence on deployed lane MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 — three OOS values exist (validated_setups.oos_exp_r=+0.2029, strict-unlock CSV +0.166, ad-hoc raw query +0.069). Lane is currently DEPLOY in 2026-05-03 rebalance. Capital-at-risk diligence; no canonical writes.
mode: IMPLEMENTATION
---

## Scope Lock

- research/oos_reconcile_ovnrng100.py
- docs/audit/hypotheses/2026-05-05-oos-reconcile-ovnrng100.yaml
- docs/audit/results/2026-05-05-oos-reconcile-ovnrng100.md
- docs/runtime/stages/oos-reconcile-ovnrng100.md

## Blast Radius

- research/oos_reconcile_ovnrng100.py — new file, zero callers, imports from canonical modules only (pipeline.paths, trading_app.holdout_policy, trading_app.config, research.filter_utils, research.chordia_strict_unlock_v1 helpers).
- docs/audit/hypotheses/2026-05-05-oos-reconcile-ovnrng100.yaml — new pre-reg stub for K=1 confirmatory diligence; no new discovery.
- docs/audit/results/2026-05-05-oos-reconcile-ovnrng100.md — new audit-result document; not a pooled finding.
- Reads: gold.db (read-only via duckdb read_only=True), validated_setups (read-only one row).
- Writes: none to canonical layers; no validated_setups update; no allocator change; no experimental_strategies row; no chordia_audit_log.yaml entry.

## Acceptance criteria

1. Post-filter N_IS in runner output equals strict-unlock CSV N_IS_fired = 522.
2. Post-filter N_OOS in runner output equals strict-unlock CSV N_OOS_fired = 66.
3. Runner OOS_ExpR matches strict-unlock CSV +0.1658 within 0.001.
4. Runner IS_ExpR matches validated_setups.expectancy_r=+0.2151 within 0.001.
5. python pipeline/check_drift.py exits 0.
6. Reconciliation report cites three lit extracts verbatim from docs/institutional/literature/.
7. No raw column comparison on overnight_range_pct — every fire decision must route through research.filter_utils.filter_signal.
8. No inline date(2026,1,1); HOLDOUT_SACRED_FROM imported from trading_app.holdout_policy.
9. No inline 2020-01-01; WF_START_OVERRIDE imported from trading_app.config.
