# US_DATA_1000 Long Paired Role Design

**Input family:** `docs/audit/results/2026-04-20-usdata1000-long-f5-f6-paired-v1.md`
**Purpose:** translate the verified `F5 TAKE / F6 AVOID` pair into the most honest ORB usage role.

## Grounding

- `resources/Algorithmic_Trading_Chan.pdf`: bounded executable routes are valid strategy units.
- `resources/Robert Carver - Systematic Trading.pdf`: signals can be conditioners and size/priority modifiers, not only standalone systems.
- `RESEARCH_RULES.md`: do not overclaim discovery from a design synthesis; this note translates an already-verified pair.
- `docs/institutional/mechanism_priors.md`: inside-range opens should hurt breakouts, while washed-out below-PDL opens can help long quality.

## Canonical role metrics

### RR 1.0

- IS baseline: N=881 ExpR=+0.0409
- IS F5 TAKE-only: N=136 ExpR=+0.3258
- IS F6 AVOID-state: N=513 ExpR=-0.0434
- IS neutral-only: N=232 ExpR=+0.0601
- IS OFF_F6 filtered parent: N=368 ExpR=+0.1583
- OOS F5 TAKE-only: N=8 ExpR=-0.0243
- OOS OFF_F6 filtered parent: N=13 ExpR=-0.0982

### RR 1.5

- IS baseline: N=866 ExpR=+0.0606
- IS F5 TAKE-only: N=135 ExpR=+0.4022
- IS F6 AVOID-state: N=503 ExpR=-0.0588
- IS neutral-only: N=228 ExpR=+0.1219
- IS OFF_F6 filtered parent: N=363 ExpR=+0.2261
- OOS F5 TAKE-only: N=8 ExpR=-0.0878
- OOS OFF_F6 filtered parent: N=13 ExpR=-0.2509

## Decision

- The verified pair is **not best handled as a standalone F5-only lane**.
- The most honest ORB integration is a **binary `NOT_F6_INSIDE_PDR` filter candidate**, with `F5_BELOW_PDL` treated as a higher-quality sub-state inside that allowed set.
- This means the pair belongs in ORB as a conditioner / lane-design route, not as a separate standalone trade family.

## Why

- RR 1.0: prefer `NOT_F6` as the primary ORB integration route. It keeps 368 IS trades at ExpR +0.1583, which is more practical than F5-only (136 trades at +0.3258). The three-state ordering `F5 > neutral > F6` also holds in-sample.
- RR 1.5: prefer `NOT_F6` as the primary ORB integration route. It keeps 363 IS trades at ExpR +0.2261, which is more practical than F5-only (135 trades at +0.4022). The three-state ordering `F5 > neutral > F6` also holds in-sample.

## Next bounded action

- Preserve a candidate-lane validation contract for `MNQ US_DATA_1000 O5 E2 long NOT_F6_INSIDE_PDR`.
- Treat `F5_BELOW_PDL` as a secondary priority / upsize descriptor inside that route, not the primary gate.
- Do not claim live-readiness from this design note; the next step is a bounded candidate-lane validation or shadow design.

## Artefacts

- CSV: `research/output/usdata1000_long_paired_role_design_v1.csv`
- Script: `research/usdata1000_long_paired_role_design_v1.py`
