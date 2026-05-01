# Chordia strict unlock audit — MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12

**Prereq file:** `docs/audit/hypotheses/2026-05-02-mnq-comex-costlt12-chordia-unlock-v1.yaml`
**Result CSV:** `docs/audit/results/2026-05-02-mnq-comex-costlt12-chordia-unlock-v1.csv`
**Canonical DB:** `/mnt/c/Users/joshd/canompx3/gold.db`

## Verdict

**MEASURED verdict:** `PASS_CHORDIA`

IS clears strict threshold 3.79 with N=1281 and ExpR=0.1073; OOS sign matches at N_OOS=70.

**MEASURED theory mode:** `NO_THEORY_GRANT`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1658 | 1281 | 77.26% | 7 | 0 | 0.1073 | 0.0829 | 0.1174 | 4.202 | 0.00003 |
| OOS | 72 | 70 | 97.22% | 1 | 0 | 0.0795 | 0.0773 | 0.0839 | 0.702 | 0.48272 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 672 | 0.1034 | 2.933 | 609 | 0.1116 | 3.011 |
| OOS | 35 | 0.0163 | 0.101 | 35 | 0.1427 | 0.891 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Canonical filter delegation: `filter_signal(..., 'COST_LT12', 'COMEX_SETTLE')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.
