# 2026-04-21 baseline posture routing decision

## Decision

Route the project's next baseline effort to **clean Mode A rediscovery planning**
for the live-six book, not to further filter surgery and not to another
orthogonal hunt expansion.

This decision is driven by verified evidence from the orthogonal hunt PCC
artifacts plus the committed Phase B institutional re-evaluation.

## Evidence base

### 1. The current blocker is posture, not a dead empirical book

Source:
- `docs/audit/2026-04-21-phase-b-institutional-reeval.md`

Verified state on canonical rows:
- the live six are not empirically dead
- the binding problem is that all six are still provenance-blocked /
  research-provisional rather than Mode A clean

### 2. The Phase B framework is not intrinsically calibration-biased

Source:
- `docs/audit/2026-04-21-phase-b-calibration-probe.md`

Verified result:
- the same gate logic can return `KEEP` for a strong clean-holdout candidate
- therefore the live-six outcome is not explained by “framework can never keep”

### 3. Grandfathering is not a production-grade escape hatch

Source:
- `docs/audit/2026-04-21-grandfathering-eligibility-audit.md`

Verified result:
- `6/6` lanes fit the provisional grandfathering carve-out
- `0/6` lanes are currently Mode A clean

### 4. The clean rediscovery queue is already ranked

Source:
- `docs/audit/2026-04-21-mode-a-readiness-scan.md`

Verified priority order:
1. `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12`
2. `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`
3. `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`

`NYSE_OPEN` is explicitly not first-line because of the current SR alarm.

### 5. The orthogonal hedge path did not produce a faster route

Sources:
- `docs/audit/2026-04-21-session-multiple-testing-reconciliation.md`
- `docs/audit/2026-04-21-golden-egg-report.md`

Verified result:
- no GOLD candidates
- no SILVER candidates
- no empirically surviving alternate route that is waiting only on posture

## Routing consequence

### Priority 1

Keep ONC / baseline-cleanliness as the main line.

### Priority 2

When baseline posture work is ready to convert into concrete rediscovery
execution, queue the first clean Mode A rediscovery targets in this order:

1. `TOKYO_OPEN`
2. `SINGAPORE_OPEN`
3. `US_DATA_1000`

### Deferred

Defer these until the baseline posture path is clarified:

- ORB_G5 degeneracy work (`C1`)
- additional orthogonal hunt expansion
- any argument that grandfathering alone justifies production-grade labeling

### Explicitly subordinated

`NYSE_OPEN` remains subordinated until its SR alarm path is resolved.

## What this decision is not

- It is **not** a claim that the live six are ready for production.
- It is **not** a claim that ONC no longer matters.
- It is **not** a claim that the orthogonal hunt was wasted.

It is a routing decision: the orthogonal hunt reduced uncertainty enough to say
where the main line should go next and where it should not.

## Ordered next actions

1. Finish / consume the ONC and baseline-cleanliness work as the posture
   decision layer.
2. Use this routing order for the first clean Mode A rediscovery queue:
   `TOKYO_OPEN`, then `SINGAPORE_OPEN`, then `US_DATA_1000`.
3. Do not spend the next cycle on ORB_G5 or another orthogonal scan unless the
   baseline posture path stalls again.

## Verdict

**APPROVED:** baseline posture-first routing, with clean rediscovery priority
ordered as `TOKYO_OPEN -> SINGAPORE_OPEN -> US_DATA_1000`.
