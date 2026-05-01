# Chordia strict unlock audit — MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15

**Prereq file:** `docs/audit/hypotheses/2026-05-02-mnq-usdata1000-vwapmid-o15-chordia-unlock-v1.yaml`
**Result CSV:** `docs/audit/results/2026-05-02-mnq-usdata1000-vwapmid-o15-chordia-unlock-v1.csv`
**Canonical DB:** `/mnt/c/Users/joshd/canompx3/gold.db`

## Verdict

**MEASURED verdict:** `PASS_CHORDIA`

IS clears strict threshold 3.79 with N=889 and ExpR=0.2113; OOS sign matches at N_OOS=47.

**MEASURED theory mode:** `NO_THEORY_GRANT`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1717 | 889 | 51.78% | 117 | 0 | 0.2113 | 0.1094 | 0.1860 | 5.547 | 0.00000 |
| OOS | 72 | 47 | 65.28% | 7 | 0 | 0.2709 | 0.1768 | 0.2355 | 1.614 | 0.10646 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 497 | 0.1919 | 3.762 | 392 | 0.2358 | 4.114 |
| OOS | 26 | 0.3386 | 1.527 | 21 | 0.1870 | 0.716 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Canonical filter delegation: `filter_signal(..., 'VWAP_MID_ALIGNED', 'US_DATA_1000')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.
