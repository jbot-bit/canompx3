# ML Clean-Room Reset Plan

**Date:** 2026-04-21
**Status:** DESIGN ONLY
**Branch:** `research/ml-sizing-v1`
**Scope:** Replace "just retrain the old ML" with a clean-room institutional workflow grounded in canonical repo evidence.

## 1. Why This Reset Exists

The right question is not "can we retrain the old ML stack?" The right question is:

> Can any strictly pre-2026 research process produce a decision layer that improves a valid ORB baseline without leakage, negative-baseline contamination, or thin-forward self-deception?

The answer is currently **unproven**. The prior ML lineage already provides enough repo evidence to reject blind retraining:

- `docs/plans/2026-03-21-ml-zero-context-audit.md`
  - negative-baseline trap
  - non-deterministic config selection
  - split drift risk
  - threshold-selection artifacts
- `docs/audit/hypotheses/2026-04-11-ml-v3-pooled-confluence-postmortem.md`
  - pooled RR-stratified meta-label result: **DEAD (0/3 survived)**
  - holdout lift failed despite positive train ExpR
- `docs/plans/ml-v3-research-design.md`
  - simple filters often dominated ML
  - pooled ML added complexity without robust incremental lift
- `RESEARCH_RULES.md`
  - derived layers banned for truth-finding
  - 2026-01-01 onwards sacred
  - holdout cannot be used for model selection
- `TRADING_RULES.md`
  - MNQ `TOKYO_OPEN` `NO_FILTER` is a valid positive baseline candidate
  - `TOKYO_OPEN` is long-only in canonical live doctrine

## 2. Truth Status

### Proven from canonical data

- A dense, positive, homogeneous family exists for ML development:
  - `MNQ TOKYO_OPEN E2 RR1.5 CB1 O5 LONG-ONLY NO_FILTER`
  - pre-2026 win/loss rows: `859`
  - wins: `421`
  - losses: `438`
  - pre-2026 ExpR: `+0.071158R`
- The current 2026 forward slice for that family is only `36` rows.
- `2026+` is too thin for ML sign-off and must remain forward-monitor only.

### Not proven

- That Random Forest or HistGradientBoosting adds incremental edge.
- That binary `pnl_r > 0` is the correct target.
- That trade-level meta-labeling is the best layer for this problem.
- That an ML allocator beats a simpler allocator or scorecard.

## 3. What Gets Wiped vs Preserved

### Wipe

Treat the old ML program as invalidated state. Do not inherit:

- old model bundles
- old thresholds
- old calibrators
- old survivor session lists
- old pooled universes
- old threshold or feature choices that were justified by contaminated or weak evidence

### Preserve

Keep the lessons that are now repo-grounded:

- meta-labeling on negative baselines is invalid
- break-bar features are banned for pre-entry sizing
- pooled cross-session convenience models are contamination-prone
- 2026 holdout is sacred and too short for final ML validation
- full Kelly is unacceptable under model uncertainty
- simple filters/allocators may dominate ML and must be included in any honest bakeoff

## 4. Alternative Framings That Must Be Kept Alive

Do not pigeonhole this into one trade-level classifier. At least four roles exist:

1. **Trade-level allocator**
   - input: pre-trade features
   - output: continuous risk scalar
   - status: scaffolded, not validated
2. **Trade-level filter**
   - input: pre-trade features
   - output: take/skip
   - status: not yet tested in this clean-room lineage
3. **Day/session-level regime gate**
   - input: one row per day/session
   - output: enable/disable ORB family
   - status: explicitly not tested here
4. **Portfolio allocator across already-valid lanes**
   - input: lane-level signals and context
   - output: capital allocation across lanes
   - status: not tested here

The current scaffold only addresses role 1. That is a valid start, not a proven best framing.

## 5. Clean-Room Research Protocol

### Phase A — Data and framing lock

1. Use canonical layers only: `daily_features` + `orb_outcomes`.
2. Use exactly one homogeneous family:
   - `MNQ TOKYO_OPEN E2 RR1.5 CB1 O5 LONG-ONLY NO_FILTER`
3. Exclude scratches from the binary target.
4. Freeze `2026-01-01+` as forward-monitor only.
5. Keep the current fail-closed feature contract unless a new feature is proven temporally safe.

### Phase B — Baseline bakeoff before ML

Every candidate must beat the same raw baseline under the same pre-2026 walk-forward protocol:

1. **Baseline**
   - static sizing
   - all-take
2. **Simple allocator**
   - monotonic binning or scorecard using 1-3 pre-trade features
   - no trees, no hidden interactions
3. **ML allocator**
   - `RandomForestClassifier`
   - `HistGradientBoostingClassifier`
   - isotonic calibration
   - bounded quarter-Kelly map capped at `2.0x`, floored at `0.0x`

If simple matches or beats ML, ML loses by default.

### Phase C — Development-only model selection

All model selection must live entirely inside pre-2026:

- anchored walk-forward or expanding-window validation
- no random shuffles across time
- no use of 2026 for thresholds, features, or model choice
- report uncertainty, not just point estimates

### Phase D — Forward monitoring only

After a model is frozen from pre-2026 work:

- compute 2026 paper-forward performance
- do not promote on that slice alone
- treat it as monitoring, not proof

## 6. What Counts As Real Success

The overlay is alive only if it shows **incremental** value over the raw baseline, not merely positive standalone returns.

Minimum institutional tests:

- positive lift vs baseline in pre-2026 walk-forward ExpR
- improved or at least not-worse drawdown profile
- lift survives honest bootstrap uncertainty
- no local destruction of the family mechanism
- no dependence on execution-unsafe features

Nice-looking but insufficient:

- AUC near 0.52 without economic lift
- isolated threshold wins
- one good split
- 2026 anecdotal improvement on a 36-row slice

## 7. Preferred Research Order

Highest-EV order from here:

1. keep the current scaffold as the clean-room dataset source
2. add a **simple monotonic allocator baseline**
3. run strictly pre-2026 walk-forward comparison:
   - static baseline
   - simple allocator
   - RF allocator
   - HGB allocator
4. if ML does not clearly beat simple, declare ML unnecessary for this family
5. if trade-level framing fails, pivot upward:
   - day/session regime gate
   - portfolio allocator

## 8. Kill Criteria

Kill the trade-level ML allocator immediately if any of the following occurs:

- any feature fails temporal safety
- pre-2026 family ExpR is non-positive
- simple allocator matches or beats ML
- lift disappears under walk-forward or bootstrap uncertainty
- the result depends on 2026 tuning
- the case for ML is complexity-first rather than mechanism-first

## 9. Desk Verdict

The correct institutional action is **reset, not blind retrain**.

- Old ML state should be treated as invalidated.
- Repo-grounded lessons survive.
- The current `MNQ TOKYO_OPEN` family is a valid substrate for a clean-room test.
- The next decision should be made by a pre-2026 bakeoff against a simple allocator, not by faith in Random Forests.
