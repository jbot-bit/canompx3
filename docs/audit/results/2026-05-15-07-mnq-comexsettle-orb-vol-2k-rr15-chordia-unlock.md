# Chordia strict unlock audit — MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K

**Prereq file:** `docs\audit\hypotheses\2026-05-15-07-mnq-comexsettle-orb-vol-2k-rr15-chordia-unlock.yaml`
**Result CSV:** `docs\audit\results\2026-05-15-07-mnq-comexsettle-orb-vol-2k-rr15-chordia-unlock.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_VOL_2K`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.79, has_theory=False) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `PASS_CHORDIA`

IS clears strict threshold 3.79 with N=1437 and ExpR=0.1197; OOS sign matches at N_OOS=76.

**MEASURED theory mode:** `UNSUPPORTED`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1494 | 1437 | 96.18% | 19 | 0 | 0.1197 | 0.1151 | 0.1044 | 3.956 | 0.00008 |
| OOS | 76 | 76 | 100.00% | 3 | 0 | 0.0427 | 0.0427 | 0.0366 | 0.319 | 0.74937 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 758 | 0.1257 | 3.020 | 679 | 0.1130 | 2.564 |
| OOS | 39 | -0.0052 | -0.028 | 37 | 0.0931 | 0.471 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'ORB_VOL_2K', 'COMEX_SETTLE')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\2026-05-15-07-mnq-comexsettle-orb-vol-2k-rr15-chordia-unlock.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-15-07-mnq-comexsettle-orb-vol-2k-rr15-chordia-unlock.md`
- `docs\audit\results\2026-05-15-07-mnq-comexsettle-orb-vol-2k-rr15-chordia-unlock.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.
