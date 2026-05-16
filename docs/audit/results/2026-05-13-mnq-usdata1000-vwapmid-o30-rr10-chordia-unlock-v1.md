# Chordia strict unlock audit — MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30

**Prereq file:** `docs\audit\hypotheses\2026-05-13-mnq-usdata1000-vwapmid-o30-rr10-chordia-unlock-v1.yaml`
**Result CSV:** `docs\audit\results\2026-05-13-mnq-usdata1000-vwapmid-o30-rr10-chordia-unlock-v1.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O30`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.79, has_theory=False) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `PASS_CHORDIA`
**FINAL verdict (power-floor override):** `UNVERIFIED_INSUFFICIENT_POWER`

IS clears strict threshold 3.79 with N=866 and ExpR=0.1332; OOS sign matches at N_OOS=46.

**OOS power override (per `feedback_chordia_oos_park_vs_unverified_power_floor.md` + backtesting-methodology.md RULE 3.3):**
OOS N=46 (27 long + 19 short) is STATISTICALLY_USELESS tier. Power = 7.8% to detect the IS effect (Cohen's d = 0.151, N_per_group_needed_for_80pct = 688). A positive OOS sign-match at this N is noise-consistent — it cannot distinguish "IS edge alive", "IS edge dead", or "IS edge reversed". Binary OOS gate does NOT apply. Verdict overridden from PASS_CHORDIA to UNVERIFIED_INSUFFICIENT_POWER. Not PARK (which implies borderline evidence); UNVERIFIED means the OOS sample is too small to render any verdict. IS result stands (t=4.450, p=0.00001); OOS confirmation is deferred until N_OOS reaches ≥ 688 per group (current OOS accumulation rate: ~72 universe days per year, ~46 fired → would require ~15 years at current pace — pooled cross-lane aggregation is the practical path per RULE 3.3 option 1).

**MEASURED theory mode:** `NO_THEORY_GRANT`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1539 | 866 | 56.27% | 167 | 0 | 0.1332 | 0.0749 | 0.1512 | 4.450 | 0.00001 |
| OOS | 72 | 46 | 63.89% | 16 | 0 | 0.1049 | 0.0670 | 0.1268 | 0.860 | 0.38991 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 467 | 0.1360 | 3.335 | 399 | 0.1299 | 2.944 |
| OOS | 27 | 0.0457 | 0.285 | 19 | 0.1891 | 0.987 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'VWAP_MID_ALIGNED', 'US_DATA_1000')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\2026-05-13-mnq-usdata1000-vwapmid-o30-rr10-chordia-unlock-v1.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-13-mnq-usdata1000-vwapmid-o30-rr10-chordia-unlock-v1.md`
- `docs\audit\results\2026-05-13-mnq-usdata1000-vwapmid-o30-rr10-chordia-unlock-v1.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.
