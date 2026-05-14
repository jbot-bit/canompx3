# Chordia strict unlock audit — MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4

**Prereq file:** `docs\audit\hypotheses\2026-05-15-06-mnq-cmepreclose-orb-g4-rr10-chordia-unlock.yaml`
**Result CSV:** `docs\audit\results\2026-05-15-06-mnq-cmepreclose-orb-g4-rr10-chordia-unlock.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.79, has_theory=False) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `PARK`

IS gates clear but OOS sign opposes IS once N_OOS=75 >= 30.

**MEASURED theory mode:** `UNSUPPORTED`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1479 | 1478 | 99.93% | 180 | 0 | 0.1049 | 0.1048 | 0.1217 | 4.677 | 0.00000 |
| OOS | 75 | 75 | 100.00% | 3 | 0 | -0.0625 | -0.0625 | -0.0674 | -0.583 | 0.55963 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 749 | 0.1444 | 4.620 | 729 | 0.0643 | 2.000 |
| OOS | 35 | 0.0403 | 0.253 | 40 | -0.1524 | -1.051 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'ORB_G4', 'CME_PRECLOSE')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\2026-05-15-06-mnq-cmepreclose-orb-g4-rr10-chordia-unlock.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-15-06-mnq-cmepreclose-orb-g4-rr10-chordia-unlock.md`
- `docs\audit\results\2026-05-15-06-mnq-cmepreclose-orb-g4-rr10-chordia-unlock.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.
