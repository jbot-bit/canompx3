---
audit_type: decision record + offensive workstream queue
companion_to: docs/audit/results/2026-05-12-l1-replacement-candidate-scan.md + PR #271
mutates: nothing (decision artifact only)
date: 2026-05-12
decision: OPTION_A_SKIP
risk_tier: critical
---

# L1 Decision + Chordia Audit Queue — 2026-05-12

## Scope / question

This MD records two things:

1. **Decision** on the L1 slot after the 2026-05-12 SR-alarm diagnostic
   paused L1 (`MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12`) and the companion
   candidate-scan MD verdict was `NO_QUALIFIED_REPLACEMENT_CANDIDATE`.
2. **Forward workstream** to address the underlying structural bottleneck
   surfaced by the scan: the Chordia-audited universe is exactly 3 strategies,
   so any defensive verdict shrinks the live set monotonically.

Question answered: *should we replace L1 from the current pool, and if not,
how do we grow the qualified inventory so future defensive verdicts don't
strand the system?*

Out-of-scope of this MD: L2 / L3 changes (untouched per user scope) and
the enumeration of the queue itself (defined here, executed in the next
workstream).

## Decision

**Option A — Skip. Keep L1 paused and the slot empty. No bootstrap
replacement.**

Rationale: the L1 slot has zero Chordia-passing non-absolute-threshold
candidates. Forcing a replacement from the MISSING-Chordia pool or from an
absolute-threshold sibling re-litigates the COST_LT12 mechanism-falsification
failure class. The expected value of filling the slot from a weak pool is
strictly negative once governance + scale-drift risk is priced in.

Allocator state preserved as: 2 deployed (L2 COMEX_SETTLE OVNRNG_100,
L3 US_DATA_1000 VWAP_MID_ALIGNED_O15) + 1 paused (L1 NYSE_OPEN COST_LT12)
+ 52 other paused + 0 stale.

## Next workstream: Chordia-audit queue (not just L1)

The 2026-05-12 candidate scan exposed the structural bottleneck: the
Chordia-audited universe is **3 strategies total** out of 54 in the allocator
corpus and 786 in the deployability batch. Every defensive verdict shrinks
the live set; the only way to grow it back is to expand Chordia coverage.

Build a Chordia-audit queue across **MNQ + MES + MGC**, not just L1's slot.

### Ranking criteria (in order, hard-to-soft)

1. **Chordia readiness.** Strategy has enough trade history + clean replay +
   stable family classification for a Chordia audit to even be runnable. This
   is the binding gate; candidates that can't be audited drop here, before any
   performance ranking.
2. **OOS power.** Carver C8 OOS status must be achievable (`PASSED` or
   plausibly-reachable with an OOS window extension). `NO_OOS_DATA` candidates
   move to a separate sub-queue.
3. **Non-degenerate filter.** Filter family must NOT be absolute-threshold
   (`COST_LT*`, `OVNRNG_*`, `ORB_VOL_*`, raw-points cutoffs) per
   `feedback_absolute_threshold_scale_audit.md`. Cross-asset percentile
   (`X_MGC_ATR*`, `X_MES_ATR*`), intra-asset percentile (`ATR_P*`), and
   directional filters preferred.
4. **Correlation / additive EV.** Candidate must add EV beyond the existing
   2-lane portfolio. Per `feedback_per_lane_breakdown_required.md`, a
   pooled-universe ExpR claim doesn't establish a per-lane lift; need a
   correlation-vs-L2/L3 check and an incremental-Sharpe estimate.
5. **Live deployability.** Slippage status, broker routing compatibility,
   profile lane-count headroom. `topstep_50k_mnq_auto` has 5 free slots
   (3/7 → 5/7 capacity); MES and MGC route to separate profiles.

### Hard constraints (apply before ranking)

- **No allocator mutation in this workstream.** `lane_allocation.json`
  remains canonical-frozen until a Chordia audit completes AND a separate
  promotion proposal is reviewed.
- **No live routing.** Queue is paper/audit-only. No `--bootstrap-runtime-control`
  invocation. No new `PROVISIONAL` rows in `lanes[]`. Per
  `feedback_provisional_not_paused_rr_variant_drift.md`, PROVISIONAL is
  live-routable so the gate must hold here.
- **No bootstrap of audit-shortcut.** Each Chordia audit must be authored
  separately from any promotion proposal it eventually feeds, per
  `feedback_bootstrap_disclosure_not_separation_of_duties.md`.

### X_MGC_ATR70 status

Strongest single ExpR candidate in the L1 slot (0.197, N=413, ROBUST,
fitness FIT) but currently `BLOCKED_OOS_UNDERPOWERED` with
`c8_oos_status: NO_OOS_DATA`. It is **not** auto-promoted to "first
candidate to audit." Per criterion ordering above:

- Criterion 1 (Chordia readiness): unknown — never been audited; depends on
  whether the clean trade-day count (500 vs validated_setups N=413) reflects
  a deeper data structure issue.
- Criterion 2 (OOS power): currently FAILS (NO_OOS_DATA hard block).
  Possibly remediable via extended OOS window or sample-shrinkage exemption
  pre-reg — not free.
- Criterion 3 (non-degenerate filter): PASSES (cross-asset percentile,
  Harris-grounded mechanism).
- Criterion 4 (correlation/EV): UNTESTED.
- Criterion 5 (deployability): currently FALSE per 786-batch row.

X_MGC_ATR70 enters the queue at its actual rank after queue-wide audit,
not by default. If other MNQ/MES/MGC candidates rank higher on criteria
1+2, they go first.

## Out of scope

- L1 replacement decision (resolved: skip).
- L2 / L3 changes (untouched per user scope).
- New pre-reg drafting in this MD (queue scope, ranking criteria are
  pre-reg-lite; full pre-reg yaml comes per-candidate when ranking surfaces a
  top pick).
- Cross-session adjacency scan for L1 slot (Option C in candidate-scan MD —
  superseded by this queue approach which is broader and discipline-bounded).

## Reproduction

Decision is an artifact, not a query. No reproduction step. The supporting
candidate-scan reproduction is in
`docs/audit/results/2026-05-12-l1-replacement-candidate-scan.md` §Reproduction.

## Caveats and limitations

- **Queue is not yet enumerated.** This MD locks the decision and the
  criteria; the actual candidate list is the next workstream and has not run.
  Decision is durable; queue contents are forward-looking.
- **Criterion 4 (correlation/EV) requires data not in deployability JSON.**
  Per-lane trade-time-series correlation against L2/L3 needs a fresh query
  against `orb_outcomes` for each candidate. Defer to per-candidate audit MD.
- **Criterion 1 (Chordia readiness) is a meta-gate without a precise spec.**
  The deployability JSON marks `chordia_missing` as a block but does not
  distinguish "ready to audit" from "structurally un-auditable." The first
  queue iteration will need to refine criterion 1 into a checkable predicate
  before it can rank candidates.
- **No statistical bootstrap of the "max profit comes from inventory not
  forcing one slot" claim.** That is the user's design hypothesis, recorded
  here as the framing for the workstream; not falsified, not validated.
- **Time-bound.** Allocator state (2 lanes + L1 paused) is the as-of
  2026-05-12 baseline. Any rebalance or independent paused-lane action by a
  parallel session invalidates the "5 free slots" capacity figure; re-check
  `lane_allocation.json` before queue-driven promotion.

## Disconfirming evidence the decision would update on

- A future audit reveals a Chordia-passing non-absolute-threshold candidate
  in the L1 slot with passing C8 OOS → revisit Option B (audit + promote).
- The 2-lane portfolio underperforms expectations during the
  L1-empty period → revisit at next rebalance.
- The user authorizes a slot-forcing move under explicit deferred-pre-reg
  conditions → re-evaluate against the rejected (d) option in the
  candidate-scan MD.

## Sources

- `docs/audit/results/2026-05-12-l1-replacement-candidate-scan.md` (candidate scan, NO_QUALIFIED_REPLACEMENT verdict)
- `docs/audit/results/2026-05-12-sr-alarm-3lane-summary.md` (originating diagnostic)
- `docs/runtime/lane_allocation.json` (allocator state baseline)
- `feedback_absolute_threshold_scale_audit.md` (criterion 3 rationale)
- `feedback_provisional_not_paused_rr_variant_drift.md` (no-live-routing rationale)
- `feedback_bootstrap_disclosure_not_separation_of_duties.md` (separated-author rationale)
- `feedback_per_lane_breakdown_required.md` (criterion 4 rationale)
- User decision recorded verbatim, 2026-05-12.
