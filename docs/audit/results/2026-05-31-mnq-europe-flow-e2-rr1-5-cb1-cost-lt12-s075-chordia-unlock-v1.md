# Chordia strict unlock audit — MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12_S075

**Prereq file:** `docs\audit\hypotheses\2026-05-31-mnq-europe-flow-e2-rr1-5-cb1-cost-lt12-s075-chordia-unlock-v1.yaml`
**Result CSV:** `docs\audit\results\2026-05-31-mnq-europe-flow-e2-rr1-5-cb1-cost-lt12-s075-chordia-unlock-v1.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12_S075`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.79, has_theory=False) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `FAIL_STRICT_CHORDIA`

IS t=2.653 < 3.79.

**MEASURED theory mode:** `UNSUPPORTED`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1547 | 1024 | 66.19% | 0 | 0 | 0.0840 | 0.0556 | 0.0829 | 2.653 | 0.00797 |
| OOS | 87 | 83 | 95.40% | 0 | 0 | 0.1631 | 0.1556 | 0.1553 | 1.415 | 0.15719 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 509 | 0.0604 | 1.352 | 515 | 0.1074 | 2.392 |
| OOS | 40 | 0.3069 | 1.812 | 43 | 0.0293 | 0.188 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'COST_LT12', 'EUROPE_FLOW')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\2026-05-31-mnq-europe-flow-e2-rr1-5-cb1-cost-lt12-s075-chordia-unlock-v1.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-31-mnq-europe-flow-e2-rr1-5-cb1-cost-lt12-s075-chordia-unlock-v1.md`
- `docs\audit\results\2026-05-31-mnq-europe-flow-e2-rr1-5-cb1-cost-lt12-s075-chordia-unlock-v1.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.
