# MGC Path-Accurate Sub-R v1

Date: 2026-04-19

## Scope

Hard-proof rebuild of the 5 native MGC low-R survivors from actual 1-minute
price path. This replaces the earlier MFE-based rewrite approximation with
canonical fill-bar + post-entry target/stop sequencing.

Locked matrix:

- 5 families carried forward from native low-R v1
- K = 5 with global BH at q=0.10
- same-bar target/stop conflicts fail closed as losses
- scratches remain scratches if neither target nor stop is reached by trading-day end

## Executive Verdict

No families survive after path-accurate sub-R reconstruction.

That kills the current native low-R MGC path as a validated edge claim.

## Matrix

| Family | Target | N IS | Avg IS | p IS | BH | Primary | N OOS | Avg OOS | Scratch | Ambig |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NYSE_OPEN_OVNRNG_50_RR1 | 0.75 | 52 | +0.1280 | 0.2308 | N | N | 56 | +0.3219 | 0.0% | 0.0% |
| US_DATA_1000_OVNRNG_10_RR1 | 0.50 | 575 | -0.0126 | 0.6026 | N | N | 71 | +0.0776 | 2.1% | 0.0% |
| US_DATA_1000_ATR_P70_RR1 | 0.50 | 420 | -0.0316 | 0.2723 | N | N | 68 | +0.0829 | 1.6% | 0.0% |
| US_DATA_1000_BROAD_RR1 | 0.50 | 893 | -0.0417 | 0.0307 | Y | N | 71 | +0.0776 | 2.5% | 0.0% |
| NYSE_OPEN_BROAD_RR1 | 0.50 | 917 | -0.0623 | 0.0011 | Y | N | 71 | +0.1976 | 0.1% | 0.1% |

## Guardrails

- This still does not reopen broad GC proxy discovery.
- This still does not promote live deployment by itself.
- But this is the correct proof layer for sub-1R target claims.

## Outputs

- `research/output/mgc_path_accurate_subr_v1_summary.csv`
- `research/output/mgc_path_accurate_subr_v1_trade_matrix.csv`
