# Chordia strict unlock audit — MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12

**Prereq file:** `docs\audit\hypotheses\2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.yaml`
**Result CSV:** `docs\audit\results\2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.00, has_theory=True) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `PASS_PROTOCOL_A`

IS clears theory threshold 3.00 with N=1532 and ExpR=0.1092. Verdict rests on IS gates only.

**MEASURED theory mode:** `NO_THEORY_GRANT`
**MEASURED threshold applied:** `3.00`
**MEASURED loader has_theory:** `True`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1548 | 1532 | 98.97% | 60 | 0 | 0.1092 | 0.1081 | 0.0920 | 3.600 | 0.00032 |
| OOS | 77 | 77 | 100.00% | 3 | 0 | 0.0743 | 0.0743 | 0.0616 | 0.541 | 0.58860 |

**OOS power:** 0.088 (STATISTICALLY_USELESS per `research/oos_power.py`; n_for_80pct=1,853). OOS sign is positive but cannot be interpreted as confirmatory — the sample is too small to distinguish signal-alive from noise. The PASS_PROTOCOL_A verdict is determined by IS gates alone.

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 778 | 0.1097 | 2.591 | 754 | 0.1088 | 2.498 |
| OOS | 35 | 0.2176 | 1.059 | 42 | -0.0451 | -0.244 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'COST_LT12', 'NYSE_OPEN')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.md`
- `docs\audit\results\2026-05-07-mnq-nyseopen-costlt12-rr15-chordia-unlock-v1.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.
