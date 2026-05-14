# Chordia strict unlock audit — MNQ_NYSE_OPEN_E2_RR2.5_CB1_X_MGC_ATR70

**Prereq file:** `docs\audit\hypotheses\2026-05-15-16-mnq-nyseopen-x-mgc-atr70-rr25-chordia-unlock.yaml`
**Result CSV:** `docs\audit\results\2026-05-15-16-mnq-nyseopen-x-mgc-atr70-rr25-chordia-unlock.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MNQ_NYSE_OPEN_E2_RR2.5_CB1_X_MGC_ATR70`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.79, has_theory=False) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `FAIL_STRICT_CHORDIA`

IS t=3.424 < 3.79.

**MEASURED theory mode:** `UNSUPPORTED`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1548 | 427 | 27.58% | 49 | 0 | 0.2579 | 0.0711 | 0.1657 | 3.424 | 0.00062 |
| OOS | 79 | 73 | 92.41% | 12 | 0 | 0.1978 | 0.1828 | 0.1307 | 1.117 | 0.26414 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 223 | 0.3393 | 3.232 | 204 | 0.1689 | 1.564 |
| OOS | 32 | 0.2684 | 1.027 | 41 | 0.1427 | 0.587 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'X_MGC_ATR70', 'NYSE_OPEN')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\2026-05-15-16-mnq-nyseopen-x-mgc-atr70-rr25-chordia-unlock.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-15-16-mnq-nyseopen-x-mgc-atr70-rr25-chordia-unlock.md`
- `docs\audit\results\2026-05-15-16-mnq-nyseopen-x-mgc-atr70-rr25-chordia-unlock.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.
