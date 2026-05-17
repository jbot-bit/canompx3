---
task: "Refresh lane_allocation.json — 2026-05-18 rebalance (truthful 3-lane MNQ allocation, C8 gate enforces drop of OVNRNG_25)"
mode: TRIVIAL
scope_lock:
  - docs/runtime/lane_allocation.json
---

## Blast Radius
- docs/runtime/lane_allocation.json — regenerated via canonical `scripts/tools/rebalance_lanes.py --profile topstep_50k_mnq_auto`. Old rebalance_date 2026-05-14 → new 2026-05-18 (4d fresher trailing data).
- Net lanes: 4 → 3. Removed: ORB_VOL_2K (correlation displaced ρ=1.000 by OVNRNG_100), VWAP_MID_RR1.0_O15 (correlation displaced ρ=0.852 by RR1.5 sibling), OVNRNG_25 (Criterion 8 OOS gate `c8_oos_status='FAILED_RATIO'` — new gate enforcement). Added: OVNRNG_100 (ExpR 0.2159, N=155, ann_r 30.9), VWAP_MID_RR1.5_O15 (ExpR 0.2416, N=112, ann_r 27.1).
- Allocation produced by deterministic rerun of canonical allocator on current `validated_setups` + `chordia_audit_log.yaml` + `daily_features` regime data. No parameter changes, no gate relaxations, no manual edits.
- Drift verification: `python pipeline/check_drift.py` → 133/133 PASS, including Check 148 (Chordia gate) + Check 149 (C8 OOS-status gate) + Check 150 (displaced[] integrity).
- Why this matters for Monday: old allocation contained OVNRNG_25 which now fails the C8 OOS deployment gate. Trading the stale file Monday would route capital at a silently-failed lane. Fresh allocation is the truthful eligibility set as of 2026-05-18.
- Writes: docs/runtime/lane_allocation.json ONLY. No prop_profiles, no validated_setups, no chordia_audit_log, no test files.
