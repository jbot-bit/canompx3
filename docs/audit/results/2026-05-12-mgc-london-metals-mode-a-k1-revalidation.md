# Chordia strict unlock audit — MGC_LONDON_METALS_E1_RR1.5_CB1_ORB_VOL_8K_O30

**Prereq file:** `docs\audit\hypotheses\2026-05-12-mgc-london-metals-mode-a-k1-revalidation.yaml`
**Result CSV:** `docs\audit\results\2026-05-12-mgc-london-metals-mode-a-k1-revalidation.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MGC_LONDON_METALS_E1_RR1.5_CB1_ORB_VOL_8K_O30`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.00, has_theory=True) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `FAIL_STRICT_CHORDIA`

IS t=2.930 < 3.00.

**MEASURED theory mode:** `NO_THEORY_GRANT`
**MEASURED threshold applied:** `3.00`
**MEASURED loader has_theory:** `True`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 916 | 49 | 5.35% | 3 | 0 | 0.4826 | 0.0258 | 0.4186 | 2.930 | 0.00339 |
| OOS | 71 | 34 | 47.89% | 1 | 0 | 0.0359 | 0.0172 | 0.0297 | 0.173 | 0.86237 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 24 | 0.5611 | 2.391 | 25 | 0.4073 | 1.733 |
| OOS | 18 | -0.2651 | -1.007 | 16 | 0.3746 | 1.197 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MGC']=2022-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'ORB_VOL_8K', 'LONDON_METALS')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\2026-05-12-mgc-london-metals-mode-a-k1-revalidation.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-12-mgc-london-metals-mode-a-k1-revalidation.md`
- `docs\audit\results\2026-05-12-mgc-london-metals-mode-a-k1-revalidation.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.
