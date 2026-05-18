# Chordia strict unlock audit — MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15_O30

**Prereq file:** `docs\audit\hypotheses\2026-05-18-mes-cmepreclose-e2-rr10-cb1-costlt15-pooled-o30-fast-lane-v1.yaml`
**Result CSV:** `docs\audit\results\2026-05-18-mes-cmepreclose-e2-rr10-cb1-costlt15-pooled-o30-fast-lane-v1.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15_O30`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.79, has_theory=False) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `FAIL_STRICT_CHORDIA`

IS t=0.020 < 3.79.

**MEASURED theory mode:** `UNSUPPORTED`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 495 | 493 | 99.60% | 442 | 0 | 0.0003 | 0.0003 | 0.0009 | 0.020 | 0.98415 |
| OOS | 31 | 31 | 100.00% | 26 | 0 | 0.1527 | 0.1527 | 0.3934 | 2.190 | 0.02852 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 275 | -0.0092 | -0.432 | 218 | 0.0124 | 0.433 |
| OOS | 14 | 0.2042 | 1.733 | 17 | 0.1103 | 1.310 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MES']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'COST_LT15', 'CME_PRECLOSE')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\2026-05-18-mes-cmepreclose-e2-rr10-cb1-costlt15-pooled-o30-fast-lane-v1.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-18-mes-cmepreclose-e2-rr10-cb1-costlt15-pooled-o30-fast-lane-v1.md`
- `docs\audit\results\2026-05-18-mes-cmepreclose-e2-rr10-cb1-costlt15-pooled-o30-fast-lane-v1.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.

## FAST_LANE v5.1 verdict (automated)

**FAST_LANE verdict:** `KILL`

Fire-rate 0.9960 outside [0.05, 0.95] — degenerate filter, not a t-stat failure.

Computed by `_fast_lane_verdict_v5_1()` per `docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml` § screen + § outcomes. This block is automated; the heavyweight Chordia verdict above is independent and unchanged.

### Gate table

| # | Gate | Threshold | Observed | Pass |
|---|---|---|---|---|
| 1 | Holdout boundary proof | max_IS < 2026-01-01 ≤ min_OOS | max_IS=2025-12-23 < 2026-01-01 ≤ min_OOS=2026-01-07 | yes |
| 2 | Fire-rate band | 0.05 ≤ fire ≤ 0.95 | 0.9960 | no |
| 3 | ExpR_IS strict positive | > 0.00 | 0.0003 | not evaluated |
| 4 | N_IS_on triage min | ≥ 50 | 493 | not evaluated |
| 5 | Per-direction sign-check (pooled) | sign(long_ExpR) == sign(short_ExpR) AND both present | sign(long_ExpR=-0.0092)=-, sign(short_ExpR=0.0124)=+ | not evaluated |
| 6 | t-stat band | ≥ 3.0 PROMOTE / [2.5, 3.0) NEEDS-MORE / < 2.5 KILL | t=0.020 → KILL | not evaluated |

> **Diagnostic note:** the fire-rate gate decided this verdict, not the t-stat band. A fire rate outside [0.05, 0.95] indicates a degenerate filter (firing on nearly every session or almost never) rather than a weak edge. The heavyweight Chordia verdict above does not currently apply a fire-rate gate — operator should not treat its t-stat verdict as the diagnostic signal here.

### What PROMOTE authorizes

- Authoring a heavyweight Chordia pre-reg for this lane (theory grant + clustered SE at trading_day + OOS power floor + era-stability section + DSR + Harvey-Liu haircut).
- Nothing else.

### What PROMOTE does NOT authorize

- Capital allocation.
- Writing this cell into `chordia_audit_log.yaml` as `PASS_CHORDIA`.
- Sibling-cell rescue (other RR / CB / aperture / session variants need their own pre-reg).
- Treating the FAST_LANE verdict as a substitute for paper-trade + SR-monitor validation.

_Scope direction at screen: `'pooled'`. Pooled lanes require both per-direction ExpRs and same-sign to PROMOTE; single-direction lanes bypass that gate._
