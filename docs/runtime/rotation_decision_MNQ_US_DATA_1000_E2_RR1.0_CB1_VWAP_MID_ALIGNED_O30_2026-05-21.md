# Rotation Decision Report

**Candidate:** `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30`
**Incumbent:** `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15`
**Profile:** `topstep_50k_mnq_auto`
**Rebalance date:** 2026-05-21
**Generated:** 2026-05-21T13:00:11+00:00
**Tool:** `scripts/research/displaced_rotation_analyzer.py --diagnostic`

---

## Canonical correlation gate

| Metric | Value |
|---|---|
| Fresh trade-level Pearson ρ | 0.728 |
| Cached ρ in `lane_allocation.json` | 0.7278902992239927 |
| Canonical gate threshold | 0.7 |
| Gate pass | False |

Source: `trading_app.lane_correlation.check_candidate_correlation` (delegated, not re-encoded).

## Heavyweight Chordia provenance

| Field | Value |
|---|---|
| Chordia verdict | PASS_CHORDIA |
| Audit date | 2026-05-14 |

Source: `docs/runtime/chordia_audit_log.yaml`.

## Day-by-day outcome decomposition

| Bucket | Count |
|---|---|
| Incumbent trading days (IS+OOS) | 936 |
| Candidate trading days (IS+OOS) | 998 |
| Overlap (both fire same day) | 756 |
| Incumbent-only days | 180 |
| Candidate-only days | 242 |
| Both win | 436 |
| Both loss | 209 |
| Incumbent W / Candidate L | 104 |
| Incumbent L / Candidate W | 7 |
| Zero P&L either side | 0 |
| Days with different pnl_r (>0.01) | 373 (49.3% of overlap) |
| Outcome swaps (W↔L) | 111 (14.7% of overlap) |

## Interpretation

The lane correlation engine measures trade-level Pearson ρ on daily P&L summed within each session. A high ρ between two same-session lanes can mean EITHER (a) they are mechanically the same fill series (subset duplicates), OR (b) they trade the same session under correlated regimes but with mechanically-separable fills (e.g., different ORB minutes → different entry/exit prices on the same day).

**The decomposition table above distinguishes these two cases.**

- If `different pnl_r %` is low (< 10%) → mechanically redundant → STAY.
- If `different pnl_r %` is high (>= 30%) → real edge co-movement on separable fills → PARALLEL_DEPLOY_CANDIDATE worth manual evaluation.
- The canonical auto-allocator gate at ρ > 0.7 stays in place regardless — this report is for operator manual override only.

## Operator options

### Option A — Stay (no action)
- Pros: Auto-allocator decision unchanged. No DD-budget reshuffle. Auditable.
- Cons: Forfeits the 49% of overlap days where the candidate trades a separable signal.

### Option B — Rotate (replace incumbent with candidate)
- Pros: Auto-allocator picks up the rotation on next rebalance via `rebalance_lanes.py`.
- Cons: Lose incumbent's track record. Only viable if fresh ρ passes the gate; here it does NOT, so this requires manual override doctrine which we do not have.
- Status: **NOT AVAILABLE** under current canonical rules (gate did not pass).

### Option C — Parallel-deploy (add candidate as a SECOND lane in the same session)
- Pros: Captures the 111-day outcome swap divergence. Combined lane exposure may improve session-level Sharpe.
- Cons: Doubles per-session position count. DD-budget must be hand-checked. Auto-allocator will resist on next rebalance — requires either a profile edit that grandfathers both lanes, OR a gate exception entry.
- Capital-class change: requires `topstep_50k_mnq_auto` profile review + adversarial-audit gate dispatch.

## Recommendation

PARALLEL_DEPLOY_CANDIDATE — gate-fail on ρ but outcomes diverge on 49% of overlap days. Operator may add as a second lane in the same session via manual profile edit (bypasses auto-allocator). DD-budget impact must be hand-computed.

## Doctrine references

- `trading_app/lane_correlation.py:24` — canonical `RHO_REJECT_THRESHOLD = 0.70`.
- `trading_app/lane_allocator.py:938` — greedy-correlation selection gate.
- `feedback_high_r_inventory_comes_from_chordia_not_raw_expr.md` — the "golden nug" doctrine (this candidate IS one).
- `feedback_max_profit_grow_chordia_inventory_not_force_slots.md` — parallel-deploy must satisfy Chordia inventory rules.
- `.claude/rules/institutional-rigor.md` § 3 — do not patch the gate; surface the diagnostic for operator decision.

## Files NOT modified by this report

- `docs/runtime/lane_allocation.json`
- `docs/runtime/chordia_audit_log.yaml`
- `trading_app/prop_profiles.py`
- `gold.db`

Operator must take explicit action via `rebalance_lanes.py` or a profile edit to materialize any decision.
