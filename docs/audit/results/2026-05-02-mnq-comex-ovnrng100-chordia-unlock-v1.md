# Chordia strict unlock audit — MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100

**Prereq file:** `docs/audit/hypotheses/2026-05-02-mnq-comex-ovnrng100-chordia-unlock-v1.yaml`
**Result CSV:** `docs/audit/results/2026-05-02-mnq-comex-ovnrng100-chordia-unlock-v1.csv`
**Canonical DB:** `/mnt/c/Users/joshd/canompx3/gold.db`

## Verdict

**MEASURED verdict:** `PASS_CHORDIA`

IS clears strict threshold 3.79 with N=529 and ExpR=0.1742; OOS sign matches at N_OOS=66.

**MEASURED theory mode:** `CLASS_GROUNDED_ONLY`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1658 | 529 | 31.91% | 2 | 0 | 0.1742 | 0.0556 | 0.1919 | 4.414 | 0.00001 |
| OOS | 72 | 66 | 91.67% | 1 | 0 | 0.1673 | 0.1534 | 0.1809 | 1.469 | 0.14173 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 283 | 0.1844 | 3.424 | 246 | 0.1624 | 2.796 |
| OOS | 33 | 0.1228 | 0.757 | 33 | 0.2119 | 1.308 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Canonical filter delegation: `filter_signal(..., 'OVNRNG_100', 'COMEX_SETTLE')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.
