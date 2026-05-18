---
date: 2026-05-18
scope: fast_lane_v5_template_calibration
status: BLOCKED_ON_PREREQUISITES
verdict: TEMPLATE_INERT_PENDING_PREREQS
template_path: docs/audit/hypotheses/TEMPLATE-fast-lane-v5.yaml
---

# FAST_LANE v5 Template — Calibration BLOCKED on Prerequisites

## Outcome

`TEMPLATE-fast-lane-v5.yaml` is committed as an **inert reference template**.
Its `post_promote_action.calibration_required.status` field is `NOT_RUN`. No
FAST_LANE pre-reg may be instantiated from this template for live screening
until the calibration backtest runs against a balanced cohort. Per v5 plan
G-A4 + B-A2, the "5-10× throughput, 30-40% FP rate" bargain is an assumption,
not a derived number.

## Why calibration was not run on 2026-05-18

Two infrastructure prerequisites for valid calibration were undeclared
dependencies in the v5 plan. Surfacing both before running calibration
prevents the calibration itself from being biased.

### Prerequisite A — Runner emissions gap

`research/chordia_strict_unlock_v1.py` was claimed in the v5 plan as
sufficient ("current runner works as-is; FAST_LANE only consumes its
existing outputs"). Verification 2026-05-18 shows it emits **only 1 of 6**
fields the FAST_LANE template requires:

| Required emission | Currently emitted by `chordia_strict_unlock_v1.py` |
|---|---|
| `t_IS` | yes (pre-existing) |
| `ExpR_IS` | yes (pre-existing) |
| `N_IS_on` | yes (pre-existing) |
| `fire_rate` | YES (line :307) |
| `long_ExpR` | **NO** |
| `short_ExpR` | **NO** |
| `max_IS_trading_day` | **NO** |
| `min_OOS_trading_day` | **NO** |
| `holdout_boundary_proof` | NO (derived from prior two) |

Without `long_ExpR` + `short_ExpR`, the template's G-A2 sign-check rule
(`sign(long_ExpR) == sign(short_ExpR) else KILL`) cannot fire on a
`direction: pooled` cell. Without `max_IS_trading_day` + `min_OOS_trading_day`,
the holdout boundary proof cannot be asserted in the result MD. Running
calibration against runner output that lacks these fields would force the
calibrator to plug stub values, which would make the calibration measure
*runner-stack drift* rather than *severity-threshold quality*. Invalid signal.

### Prerequisite B — No labeled cohort exists

The plan's calibration method requires "a balanced sample (known-PROMOTE +
known-KILL + known-NEEDS-MORE) across instruments" (template line ~159,
`calibration_required.method`). G-A1 explicitly forbids survivor-only
sampling.

Verification of `docs/audit/results/` on 2026-05-18:

- 354 result MDs total
- **10** carry a top-level structured verdict field (`verdict:`, `outcome:`,
  `pooled_finding:`, `status:`)
- **344** record their verdict in prose only

Sampling exclusively from the 10 structured-verdict files is the textbook
survivor population G-A1 forbids. Building a labeling heuristic across the
344 prose-only files introduces exactly the labeling bias the v5 plan flagged
as the dominant calibration risk (B-A1, B-A2). Neither path produces a
defensible cohort.

## What this means

The v5 template is correctly built and committed. It can be *read* and
*referenced*. It **cannot** be *instantiated for live screening* until
both prerequisites land. Downstream consumers must respect the
`is_triage_screen: true` flag and the `calibration_required.status: NOT_RUN`
field — neither is enforced by drift check in v5 (explicit S-A3
acknowledgment).

## Tracked work — separate from the template commit

These are NOT auto-spawned. Each requires its own design-gate pass before
implementation:

1. **PR-A — runner emissions upgrade.** Add `long_ExpR`, `short_ExpR`,
   `max_IS_trading_day`, `min_OOS_trading_day`, and derived
   `holdout_boundary_proof` to `chordia_strict_unlock_v1.py` output.
   Scope: research/ only, single file, backward-compatible (new columns,
   not renamed existing ones). Companion test must inject a known
   long-only or short-only cell and confirm the per-direction ExpR
   matches the pooled ExpR for the active direction and equals NaN for
   the silent direction.

2. **PR-B — explicit verdict field requirement on future heavyweight
   result MDs.** Doctrine addition (not enforcement yet): every
   heavyweight Chordia pre-reg landing 2026-05-18 or later must produce
   a result MD with a top-level structured `verdict:` field drawn from
   the locked taxonomy (`PASS_REPLACE | PROVISIONAL | UNVERIFIED | PARK
   | KILL`). Existing result MDs are NOT retroactively re-labeled
   (would inject post-hoc bias). Drift-check enforcement is v6+.

3. **PR-C — calibration backtest itself.** Runs ONLY after PR-A merges
   AND ≥30 heavyweight result MDs with structured verdict fields exist
   (PR-B accumulation). Uses balanced sampling across the
   structured-verdict cohort: equal PROMOTE/KILL/NEEDS-MORE counts,
   stratified across MNQ/MES/MGC. Discordance >25% → tune
   `screen.promote_threshold` (currently 2.5) before any FAST_LANE
   instantiation. Date estimate: contingent on heavyweight run cadence;
   not before mid-2026 at current pace.

## Re-screen condition

This BLOCKED status auto-clears when PR-A and PR-C both land. PR-B
accumulation is a continuous prerequisite (calibration needs ≥30 labeled
cells, not a single PR landing). Until then, any agent or operator who
proposes instantiating FAST_LANE for live screening must be redirected to
this doc.

## Files committed in the same change-set

- `docs/audit/hypotheses/TEMPLATE-fast-lane-v5.yaml` — the inert template
- `docs/audit/results/2026-05-18-fast-lane-v5-calibration-blocked.md` — this file

No other files in this commit. The two tracked-work PRs (A, B, C) are
separate change-sets pending design-gate approval.
