---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# ML V2 Config Selection

**Generated:** 2026-03-27T03:27:22.236194+00:00
**Instrument:** MNQ
**Pre-registration:** `docs/pre-registrations/ml-v2-preregistration.md`

Committed BEFORE bootstrap. Selection uses CPCV AUC on train split only.
Test set is never consulted for selection (honest OOS is reported but not used).

## Selected Configs for Bootstrap

| # | Session | Mode | Aperture | RR | CPCV AUC | Test AUC | OOS dR | Skip% |
|---|---------|------|----------|----|----------|----------|--------|-------|
| 1 | US_DATA_1000 | per_aperture | O30 | 1.0 | 0.502 | 0.524 | +0.0 | 0.5% |
| 2 | NYSE_CLOSE | per_aperture | O5 | 1.0 | 0.538 | 0.557 | +3.9 | 12.4% |

**Total selected:** 2/12 sessions

## Bootstrap Parameters (from pre-registration)

- Permutations: 5000 (Phipson & Smyth 2010)
- Family unit: session (K=12)
- BH FDR: q=0.05 at K=12 (promotion), K=108 reported as footnote
- Kill gate: 0=DEAD, 1=CONDITIONAL, >=2=ALIVE

## Configs NOT Selected (for audit trail)

Sessions with no ML model across all 6 combos are omitted (negative baseline,
insufficient data, or CPCV below random in ALL configs).

