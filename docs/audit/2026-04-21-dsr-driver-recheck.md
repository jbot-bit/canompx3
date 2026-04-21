# 2026-04-21 DSR Driver Recheck

Scope: PCC-2 additive posture-clearing evidence for the Fork D orthogonal hunt.

Question:
- Do the DSR driver numbers cited in the Phase B lineage reproduce from the current canonical DB and `trading_app.dsr`, or is there arithmetic drift that should be escalated before ONC work completes?

## Method

- Canonical DB: `/mnt/c/Users/joshd/canompx3/gold.db` read-only
- Helper: `trading_app.dsr`
- Inputs per lane: `sample_size`, `sharpe_ratio`, `n_trials_at_discovery`, `skewness`, `kurtosis_excess` from `active_validated_setups`
- Cross-sectional variance: `estimate_var_sr_from_db(..., min_sample=30)`
- Bracket: `rho_hat ∈ {0.3, 0.5, 0.7}`

## Recomputed drivers

- `var_sr = 0.006712097170233147`

| Lane | `sample_size` | `SR_hat` | `M` | `SR_0(rho=0.7)` | `DSR(rho=0.7)` | Drift vs Phase B narrative |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | `1459` | `0.0973` | `35616` | `0.317625` | `0.0000000000` | none |
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | `1485` | `0.0656` | `35616` | `0.317625` | `0.0000000000` | none |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | `1508` | `0.0904` | `35616` | `0.317625` | `0.0000000000` | none |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | `968` | `0.0941` | `35700` | `0.317672` | `0.000000000001` | none beyond rounding |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | `918` | `0.1122` | `35616` | `0.317625` | `0.000000000225` | none beyond rounding |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | `1348` | `0.0771` | `35700` | `0.317672` | `0.0000000000` | none |

## Interpretation

1. No arithmetic drift was found.
   - The recomputed driver stack matches the Phase B lineage story:
     - `M ≈ 35.6k / 35.7k`
     - `var_sr ≈ 0.006712`
     - `SR_0 ≈ 0.318` at the conservative `rho=0.7` bound
     - `DSR` effectively collapses to zero for all six

2. The DSR problem remains structural, not arithmetic.
   - The live six are being compared against a very large trial family count with modest observed per-trade Sharpe.
   - That is exactly why Amendment 2.1 downgraded DSR to a cross-check pending proper ONC / `N_eff`.

3. This result does not resolve the ONC question.
   - It only says the existing pre-ONC DSR arithmetic is internally consistent.

## Verdict

`NO_DSR_DRIVER_DRIFT`

No escalation is warranted from arithmetic mismatch. The unresolved issue remains the ONC / effective-trial-count question, not a broken current recomputation.
