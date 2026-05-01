# Chordia strict unlock audit — MNQ_CME_PRECLOSE_E2_RR1.0_CB1_X_MES_ATR60

**Prereq file:** `docs/audit/hypotheses/2026-05-02-mnq-cmepreclose-xmesatr60-chordia-unlock-v1.yaml`
**Result CSV:** `docs/audit/results/2026-05-02-mnq-cmepreclose-xmesatr60-chordia-unlock-v1.csv`
**Canonical DB:** `/mnt/c/Users/joshd/canompx3/gold.db`

## Verdict

**MEASURED verdict:** `FAIL_STRICT_CHORDIA`

IS t=3.716 < 3.79.

**MEASURED theory mode:** `UNSUPPORTED`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1643 | 700 | 42.60% | 73 | 0 | 0.1230 | 0.0524 | 0.1405 | 3.716 | 0.00020 |
| OOS | 71 | 49 | 69.01% | 1 | 0 | -0.0229 | -0.0158 | -0.0241 | -0.169 | 0.86582 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 340 | 0.1910 | 4.105 | 360 | 0.0588 | 1.256 |
| OOS | 26 | 0.1118 | 0.610 | 23 | -0.1752 | -0.874 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Canonical filter delegation: `filter_signal(..., 'X_MES_ATR60', 'CME_PRECLOSE')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.
