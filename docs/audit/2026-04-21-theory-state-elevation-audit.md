# 2026-04-21 Theory-State Elevation Audit

Scope: PCC-3 additive posture-clearing evidence for the Fork D orthogonal hunt.

Question:
- Can any of the current live six be elevated from `without_theory` to `with_theory` for Chordia banding without retrofitting the mechanism after the fact?

## Governing standard

From `docs/institutional/pre_registered_criteria.md`:
- Criterion 4 requires `t >= 3.00` only for strategies with strong pre-registered economic theory support.
- The same section warns that the current `Harvey-Liu-Zhu` grounding is still indirect/stub-level and that direct grounding must be promoted before accepting a `3.00 <= t < 3.79` candidate on theory alone.

From `docs/institutional/mechanism_priors.md`:
- Chordia `t >= 3.79` still binds unless a literature extract says otherwise.
- The priors document is explicitly not permission to loosen the t-stat bar after seeing results.

## Lane audit

| Lane | Filter | Existing mechanism prior | Pre-specified direct theory support for this exact lane? | Elevate to `with_theory`? | Reason |
| --- | --- | --- | --- | --- | --- |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `ORB_G5` | Generic ORB-size / friction mechanism exists | No | No | Generic size/friction prior exists, but no direct pre-specified lane-level theory file or literature extract scoped to `MNQ × EUROPE_FLOW × ORB_G5 × RR1.5`. |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | `ATR_P50` | Generic ATR-percentile regime concept exists | No | No | No lane-specific pre-result theory support found for `SINGAPORE_OPEN O15 ATR_P50`; would be retrospective storytelling. |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | `ORB_G5` | Generic ORB-size / friction mechanism exists | No | No | Same issue as EUROPE_FLOW: broad cost/size theory exists, but not a direct pre-specified lane theory that clears the stricter with-theory standard. |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | `COST_LT12` | Generic minimum-viable-trade-size / friction theory exists | No | No | `COST_LT12` is a normalized cost screen, but no direct lane-level pre-spec was found proving `MNQ × NYSE_OPEN × RR1.0` should qualify for 3.00 banding. |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | `COST_LT12` | Generic minimum-viable-trade-size / friction theory exists | No | No | Same as NYSE_OPEN: economically plausible, but not directly pre-specified enough to justify retrospective elevation. |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | `ORB_G5` | Generic ORB-size / friction mechanism exists | No | No | Session-specific macro-release momentum literature exists in the repo, but not as a pre-specified direct mechanism for this exact `ORB_G5 O15 RR1.5` lane. |

## Diagnosis

There is **some** broad mechanism grounding for the live-six family:
- ORB size / friction logic
- minimum-viable-trade-size logic
- regime screening as a concept

But that is not enough for retrospective theory-state elevation under the repo's current standard. The missing piece is a **direct, pre-result, lane-scoped mechanism statement** tied to the exact lane configuration before the empirical result existed.

## Verdict

`NO_THEORY_STATE_ELEVATION`

Implication:
- The live six should remain on the `without_theory` Chordia path for current posture purposes.
- Any future move to `with_theory` must happen in a new clean pre-registration, not as a rescue of already-known results.
