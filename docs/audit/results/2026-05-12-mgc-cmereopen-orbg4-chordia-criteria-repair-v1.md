# Chordia strict unlock audit — MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4

**Prereq file:** `docs/audit/hypotheses/2026-05-12-mgc-cmereopen-orbg4-chordia-criteria-repair-v1.yaml`
**Result CSV:** `docs/audit/results/2026-05-12-mgc-cmereopen-orbg4-chordia-criteria-repair-v1.csv`
**Canonical DB:** `/home/joshd/canompx3/.worktrees/tasks/codex/chordia-prereg-audit/gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.00, has_theory=True) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `FAIL_STRICT_CHORDIA`

IS t=2.163 < 3.00.

**Decision:** do not append this strategy to `docs/runtime/chordia_audit_log.yaml`.
Do not activate an MGC profile from this evidence. The exact-lane repair
failed the Chordia floor even after applying the entry-theory threshold.

**MEASURED theory mode:** `ENTRY_THEORY_ONLY`
**MEASURED threshold applied:** `3.00`
**MEASURED loader has_theory:** `True`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 788 | 168 | 21.32% | 82 | 0 | 0.1110 | 0.0237 | 0.1669 | 2.163 | 0.03056 |
| OOS | 68 | 68 | 100.00% | 20 | 0 | 0.0663 | 0.0663 | 0.0800 | 0.659 | 0.50959 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 91 | 0.1742 | 2.608 | 77 | 0.0362 | 0.459 |
| OOS | 36 | 0.1397 | 0.967 | 32 | -0.0163 | -0.117 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MGC']=2022-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'ORB_G4', 'CME_REOPEN')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs/audit/hypotheses/2026-05-12-mgc-cmereopen-orbg4-chordia-criteria-repair-v1.yaml
```

Outputs (overwritten in place):

- `docs/audit/results/2026-05-12-mgc-cmereopen-orbg4-chordia-criteria-repair-v1.md`
- `docs/audit/results/2026-05-12-mgc-cmereopen-orbg4-chordia-criteria-repair-v1.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.
