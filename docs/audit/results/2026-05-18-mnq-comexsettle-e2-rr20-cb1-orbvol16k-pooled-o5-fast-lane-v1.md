# Chordia strict unlock audit — MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K

**Prereq file:** `docs\audit\hypotheses\2026-05-18-mnq-comexsettle-e2-rr20-cb1-orbvol16k-pooled-o5-fast-lane-v1.yaml`
**Result CSV:** `docs\audit\results\2026-05-18-mnq-comexsettle-e2-rr20-cb1-orbvol16k-pooled-o5-fast-lane-v1.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_VOL_16K`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.79, has_theory=False) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `FAIL_STRICT_CHORDIA`

IS t=3.300 < 3.79.

**MEASURED theory mode:** `UNSUPPORTED`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1494 | 97 | 6.49% | 10 | 0 | 0.4676 | 0.0304 | 0.3350 | 3.300 | 0.00097 |
| OOS | 77 | 18 | 23.38% | 2 | 0 | 0.4387 | 0.1025 | 0.3109 | 1.319 | 0.18710 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 47 | 0.4427 | 2.120 | 50 | 0.4909 | 2.525 |
| OOS | 9 | 0.2632 | 0.586 | 9 | 0.6141 | 1.203 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'ORB_VOL_16K', 'COMEX_SETTLE')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\2026-05-18-mnq-comexsettle-e2-rr20-cb1-orbvol16k-pooled-o5-fast-lane-v1.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-18-mnq-comexsettle-e2-rr20-cb1-orbvol16k-pooled-o5-fast-lane-v1.md`
- `docs\audit\results\2026-05-18-mnq-comexsettle-e2-rr20-cb1-orbvol16k-pooled-o5-fast-lane-v1.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.

## FAST_LANE v5.1 verdict (automated)

**FAST_LANE verdict:** `PROMOTE`

t=3.300 clears 3.0 PROMOTE band; all gates pass.

Computed by `_fast_lane_verdict_v5_1()` per `docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml` § screen + § outcomes. This block is automated; the heavyweight Chordia verdict above is independent and unchanged.

### Gate table

| # | Gate | Threshold | Observed | Pass |
|---|---|---|---|---|
| 1 | Holdout boundary proof | max_IS < 2026-01-01 ≤ min_OOS | max_IS=2025-12-31 < 2026-01-01 ≤ min_OOS=2026-01-02 | yes |
| 2 | Fire-rate band | 0.05 ≤ fire ≤ 0.95 | 0.0649 | yes |
| 3 | ExpR_IS strict positive | > 0.00 | 0.4676 | yes |
| 4 | N_IS_on triage min | ≥ 50 | 97 | yes |
| 5 | Per-direction sign-check (pooled) | sign(long_ExpR) == sign(short_ExpR) AND both present | sign(long_ExpR=0.4427)=+, sign(short_ExpR=0.4909)=+ | yes |
| 6 | t-stat band | ≥ 3.0 PROMOTE / [2.5, 3.0) NEEDS-MORE / < 2.5 KILL | t=3.300 → PROMOTE | PROMOTE |

### What PROMOTE authorizes

- Authoring a heavyweight Chordia pre-reg for this lane (theory grant + clustered SE at trading_day + OOS power floor + era-stability section + DSR + Harvey-Liu haircut).
- Nothing else.

### What PROMOTE does NOT authorize

- Capital allocation.
- Writing this cell into `chordia_audit_log.yaml` as `PASS_CHORDIA`.
- Sibling-cell rescue (other RR / CB / aperture / session variants need their own pre-reg).
- Treating the FAST_LANE verdict as a substitute for paper-trade + SR-monitor validation.

_Scope direction at screen: `'pooled'`. Pooled lanes require both per-direction ExpRs and same-sign to PROMOTE; single-direction lanes bypass that gate._
