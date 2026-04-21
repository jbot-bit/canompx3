# 2026-04-21 PCC escalation memo

## Why this memo exists

This session did **not** produce a new orthogonal deployable candidate. The
highest-EV output is additive posture-clearing evidence for the live-six path.

Use this memo as a re-scope input for the baseline / ONC work, not as a
replacement for it.

## What the session established

### 1. The Phase B gate stack is not intrinsically incapable of `KEEP`

Source:
- `docs/audit/2026-04-21-phase-b-calibration-probe.md`

Verified result:
- a synthetic `clean_strong_signal` routed through the same Phase B logic
  returns `KEEP`
- the same signal with dirty provenance flips to `DEGRADE`
- the same strong signal in SR `ALARM` flips to `PAUSE-PENDING-REVIEW`

Implication:
- the current live-book bottleneck is not “framework can never keep anything”
- the binding issue remains provenance / posture, not a gate-stack calibration bug

### 2. Grandfathering is only a provisional deployability carve-out

Source:
- `docs/audit/2026-04-21-grandfathering-eligibility-audit.md`

Verified result:
- `6/6` live lanes fit the current `research-provisional + operationally deployable` carve-out
- `0/6` are currently Mode A clean

Implication:
- no live lane has an existing evidence path to production-grade status without
  clean rediscovery and fresh OOS scoring
- validator-native provenance does not rescue post-holdout discovery timing

### 3. The shortest clean rediscovery queue is already visible

Source:
- `docs/audit/2026-04-21-mode-a-readiness-scan.md`

Verified ranking:
1. `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12`
2. `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`
3. `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`

Reason:
- these are the nearest lanes to a credible Mode A rerun path once provenance
  is fixed
- `NYSE_OPEN` is currently subordinated by live SR `ALARM`

Implication:
- if the objective is the minimum honest route to first live-ready strategy,
  these three should be the first clean-rediscovery targets once the baseline
  posture path is open

## What this memo rules out

1. Do **not** spend cycles proving the Phase B framework is fundamentally biased.
   - PCC-1 already falsified that.

2. Do **not** treat grandfathering as a path to production-grade status.
   - it is only a provisional operational carve-out.

3. Do **not** divert into the orthogonal hunt as a faster alternate route.
   - this session's orthogonal families produced zero GOLD and zero SILVER
     candidates

## Recommended immediate action

If the question is “what is the minimum honest path to first live strategy?”:

1. keep ONC / baseline-cleanliness as the main line
2. use this memo to narrow the first clean Mode A rediscovery queue to:
   - `TOKYO_OPEN`
   - `SINGAPORE_OPEN`
   - `US_DATA_1000`
3. defer `NYSE_OPEN` until the current SR alarm path is resolved

## Net

This session did not find a hedge that bypasses the live-six posture blocker.
It did reduce uncertainty about the blocker itself and about where the clean
rediscovery effort should start once Terminal 1 completes the posture work.
