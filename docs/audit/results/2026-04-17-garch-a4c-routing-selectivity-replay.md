# Garch A4c Routing-Selectivity Replay

**Date:** 2026-04-18
**As-of trading day:** 2026-04-16
**Pre-registration:** `docs/audit/hypotheses/2026-04-17-garch-a4c-routing-selectivity.yaml`
**Design:** `docs/plans/2026-04-17-garch-a4c-routing-selectivity-design.md`
**Framing commit:** `1a721e92`
**Dual-surface verdict:** **NULL**

## Binding preflight (re-verified in-harness)

| Surface | Max slots | Binds | Total | Ratio | ≥80% gate | Mean supply | Median |
|---|---:|---:|---:|---:|:---:|---:|---:|
| A_raw_slots | 5 | 68 | 72 | 0.944 | PASS | 26.36 | 29 |
| B_rho_survivor_slots | 3 | 67 | 72 | 0.931 | PASS | 7.04 | 7 |

## Harness-sanity gate (positive control vs primary null)

**Rule:** R-per-fill(positive_control) − R-per-fill(primary_null) ≥ 0.01 on BOTH surfaces.
**Gate outcome:** PASS

| Surface | R/fill primary null | R/fill positive control | Lift | Required | Pass |
|---|---:|---:|---:|---:|:---:|
| A_raw_slots | 0.062250 | 0.084921 | 0.022671 | 0.010000 | PASS |
| B_rho_survivor_slots | 0.046496 | 0.089080 | 0.042584 | 0.010000 | PASS |

## IS replay — R per filled slot-day by surface and policy

| Surface | Policy | Trading days | Total R | Fills | R/fill | Sharpe | DD | Ann R | Hit-rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_raw_slots | RANDOM_UNIFORM_UNDER_BINDING | 1755 | +291.21 | 4678 | +0.062250 | +1.410 | 43.37 | +41.81 | 2.685 |
| A_raw_slots | TRAILING_SHARPE | 1755 | +366.08 | 4984 | +0.073450 | +1.769 | 32.09 | +52.56 | 2.846 |
| A_raw_slots | POSITIVE_CONTROL_TRAILING_EXPR | 1755 | +303.51 | 3574 | +0.084921 | +1.651 | 42.32 | +43.58 | 2.064 |
| A_raw_slots | CANDIDATE_GARCH | 1755 | +278.86 | 4044 | +0.068955 | +1.420 | 48.09 | +40.04 | 2.287 |
| A_raw_slots | DESTRUCTION_SHUFFLE | 1755 | +313.90 | 5111 | +0.061417 | +1.349 | 48.09 | +45.07 | 2.915 |
| B_rho_survivor_slots | RANDOM_UNIFORM_UNDER_BINDING | 1755 | +139.81 | 3007 | +0.046496 | +0.949 | 41.35 | +20.08 | 1.725 |
| B_rho_survivor_slots | TRAILING_SHARPE | 1755 | +174.34 | 2844 | +0.061301 | +1.071 | 36.08 | +25.03 | 1.629 |
| B_rho_survivor_slots | POSITIVE_CONTROL_TRAILING_EXPR | 1755 | +180.39 | 2025 | +0.089080 | +1.411 | 19.77 | +25.90 | 1.175 |
| B_rho_survivor_slots | CANDIDATE_GARCH | 1755 | +179.45 | 2279 | +0.078740 | +1.238 | 24.03 | +25.77 | 1.292 |
| B_rho_survivor_slots | DESTRUCTION_SHUFFLE | 1755 | +179.87 | 3108 | +0.057873 | +1.066 | 34.77 | +25.83 | 1.779 |

## Candidate primary evaluation (per surface)

Primary pass rule (all must hold):
- R/fill candidate − R/fill primary_null ≥ 0.01
- Sharpe candidate − Sharpe secondary_comparator ≥ 0.05
- DD candidate / DD primary_null ≤ 1.2
- Selection churn (jaccard) ≤ 0.5

| Surface | ΔR/fill | R/fill pass | ΔSharpe | Sharpe pass | DD ratio | DD pass | Churn | Churn pass | Primary verdict |
|---|---:|:---:|---:|:---:|---:|:---:|---:|:---:|:---:|
| A_raw_slots | +0.006705 | FAIL | -0.349 | FAIL | 1.109 | PASS | 0.842 | FAIL | **FAIL** |
| B_rho_survivor_slots | +0.032244 | PASS | +0.166 | PASS | 0.581 | PASS | 0.956 | FAIL | **FAIL** |

## Destruction shuffle control (must FAIL primary on both surfaces)

| Surface | ΔR/fill | Primary rule verdict | Pass (i.e. failed primary rule) |
|---|---:|:---:|:---:|
| A_raw_slots | -0.000833 | FAIL | PASS (shuffle failed as required) |
| B_rho_survivor_slots | +0.011377 | FAIL | PASS (shuffle failed as required) |

## 2026 OOS descriptive (per surface, candidate vs primary null)

Effect ratio ≥ 0.40 AND direction match required.

| Surface | IS Δ | OOS Δ | Direction match | Effect ratio | Effect ≥ 0.40 | OOS days | OOS fills cand | OOS fills null |
|---|---:|---:|:---:|---:|:---:|---:|---:|---:|
| A_raw_slots | +0.006705 | -0.054992 | FAIL | -8.2016 | FAIL | 67 | 259 | 236 |
| B_rho_survivor_slots | +0.032244 | -0.057822 | FAIL | -1.7933 | FAIL | 67 | 139 | 135 |

## Dual-surface verdict

**NULL**

Verdict semantics (locked in hypothesis file):
- STRONG_PASS: routing edge on BOTH raw supply AND operative mechanic
- STANDARD_PASS: raw-supply edge only (A passes, B fails)
- STRUCTURAL_PASS: operative-mechanic edge only (B passes, A fails)
- NULL: both surfaces fail primary rule — no rescue
- NULL_DATA_MINED: destruction shuffle passed primary — candidate is data-mined
- ABORT: harness-sanity gate failed — harness bug, not candidate verdict

