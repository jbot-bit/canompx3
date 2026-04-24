# Cross-Asset / Cross-Session Chronology Spec

**Date:** 2026-04-25
**Status:** `READY_FOR_PREREG`
**Scope:** governance contract for any future prereg that proposes to use
an earlier session's facts as input to a later-ORB entry decision, on the
same instrument or on a different instrument.
**Non-goals:** this document does NOT author a scan, propose a factor,
claim a signal exists, or authorize deployment. It defines what honest
testability looks like in this space so that a future preregistration can
be cited against it and so that a reviewer can reject a prereg that
violates any rule.

## 1. Purpose

The recent audit window closed most near-term cross-asset and cross-session
narratives: pooled averages were shown to hide opposite-sign per-lane
cells; break-bar columns were shown to be E2-unsafe; proxy-revival paths
were closed honestly. One plausible-but-easy-to-fake idea remained open:
whether an earlier session can honestly contribute admissibility or quality
information to a later ORB entry without leakage, heterogeneity masking,
or revival of a closed NO-GO.

This spec freezes what is known when. It is the gating artifact for any
future preregistration in this space. Without it, such a scan drifts into
look-ahead leakage or pooled narrative fishing. With it, the scan either
has a bounded, honest question to test, or is explicitly parked.

## 2. Canonical Time Sources

All timing claims in a downstream prereg MUST resolve through the repo's
canonical sources. Never inline values.

- Session boundaries: `pipeline.dst.SESSION_CATALOG` is the only
  authoritative source of each session's open and close in UTC. Brisbane
  trading-day alignment is handled inside the canonical helpers.
- Target ORB window: `pipeline.dst.orb_utc_window(trading_day, orb_label,
  orb_minutes)` is the only authoritative boundary for ORB start and ORB
  end in UTC.
- Brisbane trading day: 09:00 local to next 09:00 local. Bars before 09:00
  belong to the previous trading day.

Any prereg that writes its own timestamp string for a session boundary or
an ORB boundary is in violation of this spec.

## 3. Admissibility Contract

A candidate source fact `F` about an earlier session `S` is admissible as
input to a later-ORB entry on target lane `L`, trading day `D`, aperture
`M`, if and only if all six of the following hold.

1. `F` is derivable entirely from bars whose bar-end UTC is at or before
   the earlier session `S`'s canonical session-end UTC, resolved via
   `SESSION_CATALOG`.
2. `S`'s canonical session-end UTC is at or before the target ORB's end
   UTC, resolved via `orb_utc_window(D, L, M)`.
3. `F` does not use the target lane's break bar or any price observed
   after that bar, for any asset.
4. `F` is computed from fully-closed bars only. Intra-bar state is never
   admissible.
5. The admissibility boundary is defined against `orb_utc_window`, never
   against `break_ts`, `break_delay_min`, or any post-break quantity.
6. Cross-Brisbane-trading-day sourcing is allowed only when the source
   trading day differs from the target trading day by exactly one, and
   rule 1 still holds. The canonical Brisbane-day alignment applies.

Symmetry note: rules 1–6 make no distinction between same-instrument and
cross-instrument sourcing. A MNQ-earlier-session source and a MGC-earlier
-session source are governed identically.

## 4. Banned Constructions

Each banned construction below is tied to a prior closed audit or to a
registered NO-GO. A future prereg that contains any of these is rejected
at review, not at runtime.

- Break-bar-derived source features on any asset. Source: PR #67 doctrine
  correction and its postmortem; `.claude/rules/backtesting-methodology.md`
  §RULE 6.1 after the 2026-04-21 amendment.
- Post-break price for any asset, under any name. Source: PR #67 plus
  `docs/postmortems/2026-04-07-e2-canonical-window-fix.md`.
- Same-Brisbane-day later sessions as "source". This is trivially
  look-ahead and is called out here because the spec's cross-day clause
  could otherwise be read as permissive.
- Pooled cross-asset or cross-session averages used as the primary
  evidence. Source: `feedback_pooled_not_lane_specific.md` and
  `feedback_per_lane_breakdown_required.md`.
- Continuous transfer-function fitting over multiple source sessions,
  under any relabeling. Source: the existing cross-asset lead-lag NO-GO
  registered in `MEMORY.md`.
- `prev_day_range` as a standalone source fact without explicit per-lane
  justification. Source: the existing `prev_day_range` standalone NO-GO.
- `prev_close_position` as a source fact. Source: the existing standalone
  NO-GO.
- Absolute-points thresholds on earlier-session facts without a
  scale-stability audit (fire-rate and lift-by-year). Source:
  `feedback_absolute_threshold_scale_audit.md`.

## 5. Heterogeneity Discipline

Every preregistration in this space MUST satisfy all of the following.

- Declare, before running, the exact target lane set: instrument, session,
  ORB-minutes, filter, direction.
- Per-lane effect is the primary evidence. Pooled statistics are
  supplementary only.
- If, across the declared lane set, at least a quarter of cells flip sign
  relative to the pooled effect, the result is classified as a
  heterogeneity artefact, not an edge.
- Any pooled claim that is proposed as the basis of a deployment change
  on a specific lane MUST be followed by a lane-specific Pathway-B K=1
  verification under Mode A on the 2026-01-01 holdout.

## 6. Power Floor

- OOS verdict requires OOS power against the IS effect size to reach at
  least 50 percent. Below this floor the verdict is `UNVERIFIED`, never
  `DEAD` and never `ALIVE`.
- A strong IS signal MUST NOT be killed by an underpowered OOS cell.
  Source: `feedback_oos_power_floor.md`.
- MinBTL discipline: no brute-force cell search greater than 300 trials.
  Source: the Phase 0 institutional grounding in
  `docs/institutional/pre_registered_criteria.md` and the Bailey–López de
  Prado grounding in `docs/institutional/literature/`.

## 7. Authoring Checklist

A preregistration authored in this space is valid only if every item
below is satisfied and cited by section in the prereg itself.

- Cites this spec by filename and section.
- Names exactly one target lane, or names a pre-registered lane set with
  per-lane reporting committed up front.
- States the source session's canonical end UTC and the target ORB's end
  UTC, in each case resolved via `SESSION_CATALOG` and `orb_utc_window`
  rather than inlined.
- Declares the expected IS fire-rate and includes a scale-stability
  check if any threshold is in absolute points.
- Declares K, the multiple-testing correction, and the Mode A holdout
  date `2026-01-01`.
- Contains an explicit kill criterion stated before execution.
- Contains an explicit UNVERIFIED criterion matching the power floor.

## 8. Worked Examples

### 8.1 Admissible — sketch

Target lane: MNQ NYSE_OPEN 15m ORB with filter G5 and direction long.
Source session: the same Brisbane trading day's earlier EUROPE_FLOW
session, which has a canonical session-end UTC strictly before the
target ORB's end UTC. Candidate source fact: a closed-bar summary
statistic computed from EUROPE_FLOW bars whose bar-end UTC falls at or
before the EUROPE_FLOW session-end UTC.

- Rule 1: satisfied — fact uses only bars at or before EUROPE_FLOW end.
- Rule 2: satisfied — EUROPE_FLOW end is before NYSE_OPEN ORB end on
  the same Brisbane trading day.
- Rule 3: satisfied — no break bar is referenced.
- Rule 4: satisfied — all bars are fully closed.
- Rule 5: satisfied — boundary uses `orb_utc_window`.
- Rule 6: not triggered — same-day.

This sketch describes an admissible shape. It does NOT claim a signal
exists. A prereg that adopts this shape still has to pass sections 5–7.

### 8.2 Admissible — cross-day sketch

Target lane: MGC COMEX_SETTLE 15m ORB with filter ORB_G5 and direction
short. Source session: the previous Brisbane trading day's NYSE_OPEN.
Candidate source fact: a closed-bar summary from the prior trading day's
NYSE_OPEN whose bar-end UTC is at or before that session's canonical end.

- Rule 1: satisfied.
- Rule 2: satisfied — prior day's NYSE_OPEN end is before current day's
  COMEX_SETTLE ORB end.
- Rule 3: satisfied.
- Rule 4: satisfied.
- Rule 5: satisfied.
- Rule 6: satisfied — source trading day and target trading day differ
  by exactly one.

Again, admissible shape only. The prereg must still pass sections 5–7.

### 8.3 Banned — same-day later-session "source"

Proposal: "condition NYSE_OPEN MNQ entry on an NYSE_CLOSE summary fact
from the same Brisbane trading day."

- Fails rule 2 — NYSE_CLOSE end is AFTER NYSE_OPEN ORB end on the same
  Brisbane trading day, so the "source" cannot be observed in time.
- Also banned by section 4 explicit clause on same-Brisbane-day later
  sessions.

### 8.4 Banned — break-bar-derived source

Proposal: "use MNQ EUROPE_FLOW break-bar range as a source input to
MNQ NYSE_OPEN entry."

- Fails rule 3 — source feature is derived from a break bar.
- Also banned by section 4 first bullet (PR #67 correction).

### 8.5 Banned — pooled primacy

Proposal: "compute a single pooled coefficient across MNQ, MES, MGC
target lanes using an earlier NYSE_OPEN source fact, deploy if the
pooled coefficient is significant."

- Fails section 5 — pooled primacy is explicitly disallowed.
- Also fails section 5 bullet 3 if per-lane cells show sign
  heterogeneity at or above 25 percent.

### 8.6 Banned — continuous transfer function over many sessions

Proposal: "fit a kernel-smoothed transfer function from the last N
sessions' features onto the next ORB's direction."

- Fails the NO-GO clause in section 4 (cross-asset and cross-session
  lead-lag revival in new clothing).
- Also fails MinBTL discipline in section 6 once N and the feature set
  are expanded.

## 9. Verdict

`READY_FOR_PREREG`.

The admissibility contract in section 3 is narrow enough that a future
author can cite it. Sections 4–7 are concrete enough that a reviewer can
reject a prereg that violates any rule. The spec does not claim an edge;
it defines honest testability.

## 10. Out of Scope

- Not a cross-asset discovery scan.
- Not a new factor family.
- Not a deployment or allocator change.
- Not a revival path for any registered NO-GO.

## 11. References

- `pipeline/dst.py` — `SESSION_CATALOG`, `orb_utc_window`.
- `.claude/rules/backtesting-methodology.md` — §RULE 6.1 and subsequent
  amendments.
- `docs/postmortems/2026-04-07-e2-canonical-window-fix.md`.
- `docs/audit/results/2026-04-21-recent-claims-skeptical-reaudit-v1.md`.
- `docs/audit/results/2026-04-21-pr48-participation-shape-oos-replication-v1.md`.
- `docs/audit/results/2026-04-19-gc-mgc-translation-audit.md`.
- `docs/institutional/pre_registered_criteria.md`.
- `docs/institutional/literature/` (Bailey–López de Prado, Harvey–Liu).
- Memory: `feedback_pooled_not_lane_specific.md`,
  `feedback_per_lane_breakdown_required.md`,
  `feedback_oos_power_floor.md`,
  `feedback_absolute_threshold_scale_audit.md`,
  `feedback_default_cross_session_scope.md`.
- `docs/runtime/stages/cross-asset-session-chronology-spec.md` — stage
  closed on commit of this spec.
