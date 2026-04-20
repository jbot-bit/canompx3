---
slug: l1-europe-flow-filter-diagnostic
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: L1 EUROPE_FLOW ORB_G5 diagnostic — does ATR-normalized threshold recover pre-2022 selectivity?
---

# Stage: L1 EUROPE_FLOW filter diagnostic

## Question

L1 `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` filter fires 57%(2019)→100%(2026).
PR #57 confirmed: unfilt IS t=+1.61 (marginal), filt IS t=+2.28, Welch
fire-vs-non-fire p=0.001 — the filter historically discriminated but
100% fire in 2026 means selection is operationally gone.

Question: can an ATR-normalized equivalent (orb_size / atr_20 threshold)
restore pre-2022 selectivity (~57% fire) at current MNQ price scale
AND retain or improve the historical +0.021R IS lift over unfiltered
geometry?

This is a diagnostic (read-only) to answer: is a Pathway-B pre-reg
worth writing for an ORB_G5 replacement on L1?

## Scope Lock

- `research/audit_l1_europe_flow_filter_diagnostic.py` (new)
- `docs/audit/results/2026-04-21-l1-europe-flow-filter-diagnostic.md` (new)

## Blast Radius

- Read-only research. Zero production-code touch.
- Canonical data: `orb_outcomes`, `daily_features`.
- No new filters registered. No config changes.
- No pre-reg created (this is diagnostic, not hypothesis test).

## Approach

1. Load L1 universe (MNQ EUROPE_FLOW E2 RR1.5 CB1 orb_minutes=5).
2. Compute `orb_size / atr_20` ratio per trade.
3. Per-year distribution of orb_size/atr_20, fire rate of ORB_G5, fire
   rate of candidate thresholds (ratio ≥ 0.10, 0.15, 0.20, 0.25, 0.30).
4. For each candidate ratio threshold: per-year ExpR of fire-vs-non-fire,
   Welch t/p, delta.
5. Compare the candidate that best recovers ~57% fire rate in 2026 vs
   the legacy ORB_G5:
   - Does it show Welch p<0.05 discrimination? (like the original did
     on pre-2022 data)
   - Does it add per-year stability (vs ORB_G5's 57→100% drift)?
6. If a candidate clears the bar, note operational implication: "Pathway-B
   pre-reg is worth writing." If not, close the diagnostic.

## Acceptance criteria

1. Script runs without exceptions.
2. MD contains the per-year table for ORB_G5 baseline + 3-5 ATR-normalized
   candidates, with Welch stats.
3. MD states operational conclusion: does the question "is Pathway-B
   worth it?" have answer YES / NO / AMBIGUOUS.
4. `python pipeline/check_drift.py` passes.
5. No production-code change.

## Non-goals

- Not writing the Pathway-B pre-reg (separate turn).
- Not testing the candidate outside L1 (per-lane diagnostic).
- Not proposing any deployment change.
