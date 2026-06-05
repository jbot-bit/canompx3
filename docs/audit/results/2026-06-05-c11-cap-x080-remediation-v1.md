# C11 cap remediation — cap×0.75 result (topstep_50k_mnq_auto)

Prereg: `docs/audit/hypotheses/2026-06-05-c11-cap-x080-remediation-v1.yaml` (K=2; cap_x0.75 = robustness sibling).
Baseline commit (canonical loader): `9429c540`. As-of 2026-06-05. Read-only, parity-by-construction (canonical `_load_lane_trade_paths`).

## Verdict: cap×0.75 CLEARS C11. All locked acceptance gates PASS.

Budget = $1,800 (express 0.90 × $2,000 Topstep MLL). Stop UNCHANGED at 0.75 (no stop lever used).

### Survival gate (account_survival, write_state=False)
| config | 90d max DD | budget | breaches | op-survival | strict gate | C11 |
|---|---|---|---|---|---|---|
| baseline (current, uncapped) | $2,038.84 | $1,800 | 0 | 0.9975 | FAIL | NO-GO |
| **cap×0.75 (all 3 lanes)** | **$1,535.22** | $1,800 | 0 | 0.9997 | PASS | **GO** |

### Locked robustness gates
| gate | threshold | result | verdict |
|---|---|---|---|
| edge retained | ≥ 0.85 | 0.9180 (91.8%) | PASS |
| deflated Sharpe (Bailey-LdP, daily, sr_0=0) | DSR ≥ 0.95 | 1.0000 | PASS |
| walk-forward eras | no negative-edge era | 0/8 years negative | PASS |

Capped daily Sharpe 0.1500→0.1615 (trimmed tail was negative-EV). Edge cost = 8.2% ($1,938 of $23,625 over 6.65yr book). 2026 holdout takes the largest cut (−$1,113) but stays net +$1,628.

### Exact cap values (cap_x0.75; p90 × 0.75, source-traced from get_profile_lane_definitions @ 9429c540)
| lane | p90_orb_pts (UNCHANGED) | risk_cap_pts (cap×0.75) |
|---|---|---|
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | 49.8 | 37.35 |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 | 143.2 | 107.40 |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08 | 44.2 | 33.15 |

## Implementation decision (operator-approved 2026-06-05): Option 1 — honest override field

Store cap as a NEW `risk_cap_pts` field; preserve `p90_orb_pts` truth. NOT pursued: overwriting p90 (lying field; rebalancer reverts it). Stop change REJECTED (not EV-tested; not needed — cap alone clears).

### Files to change (NOT YET APPLIED — deferred at CTX PRE-CLEAR boundary)
1. `docs/runtime/lane_allocation/topstep_50k_mnq_auto.json` — add `risk_cap_pts` to 3 lanes (p90 untouched).
2. `trading_app/prop_profiles.py` (load_allocation_lanes ~1519-1522) — `max_orb = entry.get("risk_cap_pts") or entry.get("p90_orb_pts") or _P90_ORB_PTS.get(inst)`.
3. `scripts/tools/rebalance_lanes.py` / `save_allocation` — PRESERVE `risk_cap_pts` across regen (else silently stripped → C11 silently re-fails).
4. `pipeline/check_drift.py` — new check: if `risk_cap_pts` present → `0 < risk_cap_pts <= p90_orb_pts` (honesty invariant).
5. Tests: loader prefers override; rebalancer preserves; gate↔live parity.

### Verified architecture facts (executable-grounded this session)
- Cap source = `p90_orb_pts` in per-profile `lane_allocation/<profile>.json`, mapped to `max_orb_size_pts` at prop_profiles.py:1520.
- Gate↔live parity holds: BOTH account_survival AND live/session_orchestrator.py:395 read via `get_profile_lane_definitions` → same value. Live enforcement = ORB_CAP_SKIP at session_orchestrator.py:2451-2453 (risk_points ≥ cap).
- Stale baton corrected: orchestrator `self._orb_caps` registry EXISTS (line 370) and auto-populates from JSON — there is no separate "wire the value" step.
- rebalance_lanes.py `save_allocation` regenerates the file → WILL strip a hand-added field unless taught to preserve (durability gap; mandatory fix #3).

## Remaining blockers to live (after the above lands)
1. Implementation #1-5 above (Tier-B capital + schema; needs stage file + full verify).
2. Adversarial-audit gate: bracket-parity `9b3fc530` closed CONDITIONAL, still owes an independent reviewer.
3. account_survival write_state regen + live_readiness_report --strict-zero-warn green.
4. Separate operator GO to arm --live.

## Path-preservation (operator priority 3)
100k/150k uncapped recovers the 8.2% edge: current $2,038.84 DD clears $2,700 (100k) / $4,050 (150k) budgets with zero cap. Requires CREATING topstep_100k_mnq_auto (no such profile exists). Tracked as deferred follow-up.

Nothing armed. --live disarmed. active unchanged. No production code edited this session (this result doc + the prereg pull are the only writes).
