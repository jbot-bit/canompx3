# MGC Native Low-R v1

Date: 2026-04-19

## Scope

Native follow-through on the prior MGC payoff-compression diagnostics.
This pass treats conservative low-R exits as candidate native MGC
families and tests them under honest global BH across the full locked matrix.

Locked matrix:

- 8 family rows (`3` broad + `5` warm/filtered)
- 2 target variants each (`0.5R`, `0.75R`)
- total K = 16 with global BH at q=0.10
- pre-2026 for selection; 2026 held back as diagnostic OOS only

## Executive Verdict

5 families survive the locked matrix.

- survivor split by target: `0.5R=4`, `0.75R=1`
- survivor split by family kind: `broad=2`, `warm=3`

Interpretation:

- if broad rows survive too, the unresolved issue is broader native MGC
  target shape, not just translated warm rows
- if only 0.5R survives, the compression problem is still tight enough that
  even modestly higher targets lose the edge
- 2026 remains diagnostic only and must not rescue or kill survivors by itself

## Full Matrix

| Family | Kind | Variant | N IS | Avg IS | p IS | BH | Primary | N OOS | Avg OOS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| NYSE_OPEN_OVNRNG_50_RR1 | warm | LR075 | 52 | +0.2226 | 0.0291 | Y | Y | 56 | +0.3519 |
| US_DATA_1000_ATR_P70_RR1 | warm | LR05 | 414 | +0.0710 | 0.0056 | Y | Y | 67 | +0.0982 |
| US_DATA_1000_OVNRNG_10_RR1 | warm | LR05 | 565 | +0.0685 | 0.0019 | Y | Y | 70 | +0.0921 |
| US_DATA_1000_BROAD_RR1 | broad | LR05 | 869 | +0.0488 | 0.0049 | Y | Y | 70 | +0.0921 |
| NYSE_OPEN_BROAD_RR1 | broad | LR05 | 913 | +0.0380 | 0.0235 | Y | Y | 71 | +0.1976 |
| NYSE_OPEN_OVNRNG_50_RR1 | warm | LR05 | 52 | +0.1293 | 0.1012 | N | N | 56 | +0.2622 |
| EUROPE_FLOW_OVNRNG_50_RR1 | warm | LR05 | 52 | +0.0896 | 0.2368 | N | N | 56 | +0.0485 |
| US_DATA_1000_ORB_G5_RR1 | warm | LR075 | 292 | +0.0669 | 0.1395 | N | N | 70 | +0.0808 |
| EUROPE_FLOW_OVNRNG_50_RR1 | warm | LR075 | 52 | +0.0604 | 0.5634 | N | N | 56 | +0.1375 |
| US_DATA_1000_ORB_G5_RR1 | warm | LR05 | 292 | +0.0579 | 0.0980 | N | N | 70 | +0.0921 |
| US_DATA_1000_OVNRNG_10_RR1 | warm | LR075 | 565 | +0.0558 | 0.0655 | N | N | 70 | +0.0808 |
| US_DATA_1000_ATR_P70_RR1 | warm | LR075 | 414 | +0.0484 | 0.1716 | N | N | 67 | +0.0792 |
| NYSE_OPEN_BROAD_RR1 | broad | LR075 | 913 | +0.0390 | 0.0891 | N | N | 71 | +0.3020 |
| US_DATA_1000_BROAD_RR1 | broad | LR075 | 869 | +0.0342 | 0.1522 | N | N | 70 | +0.0808 |
| EUROPE_FLOW_BROAD_RR1 | broad | LR05 | 917 | -0.0095 | 0.4518 | N | N | 71 | +0.0744 |
| EUROPE_FLOW_BROAD_RR1 | broad | LR075 | 917 | -0.0132 | 0.4865 | N | N | 71 | +0.0761 |

## Guardrails

- These exits are still diagnostic rewrites, not live-ready execution rules.
- Ambiguous losses remain fail-closed.
- This pass does not revive the retired GC shelf or reopen proxy discovery.

## Outputs

- `research/output/mgc_native_low_r_v1_matrix.csv`
