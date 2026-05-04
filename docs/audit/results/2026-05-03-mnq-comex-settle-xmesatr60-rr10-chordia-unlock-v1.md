# Chordia strict unlock audit — MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60

**Prereq file:** `docs\audit\hypotheses\2026-05-03-mnq-comex-settle-xmesatr60-rr10-chordia-unlock-v1.yaml`
**Result CSV:** `docs\audit\results\2026-05-03-mnq-comex-settle-xmesatr60-rr10-chordia-unlock-v1.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_X_MES_ATR60`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.79, has_theory=False) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `PASS_CHORDIA`

IS clears strict threshold 3.79 with N=676 and ExpR=0.1519; OOS sign matches at N_OOS=49.

**MEASURED theory mode:** `UNSUPPORTED`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1494 | 676 | 45.25% | 3 | 0 | 0.1519 | 0.0688 | 0.1677 | 4.361 | 0.00001 |
| OOS | 72 | 49 | 68.06% | 1 | 0 | 0.1537 | 0.1046 | 0.1642 | 1.150 | 0.25025 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 371 | 0.2005 | 4.332 | 305 | 0.0929 | 1.761 |
| OOS | 25 | -0.0410 | -0.214 | 24 | 0.3565 | 1.964 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'X_MES_ATR60', 'COMEX_SETTLE')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\2026-05-03-mnq-comex-settle-xmesatr60-rr10-chordia-unlock-v1.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-03-mnq-comex-settle-xmesatr60-rr10-chordia-unlock-v1.md`
- `docs\audit\results\2026-05-03-mnq-comex-settle-xmesatr60-rr10-chordia-unlock-v1.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.

## Post-merge addenda (2026-05-04)

Added by `chordia/post-audit-addendum` after independent evidence-auditor review of the merged audit. Verdict (PASS_CHORDIA) unchanged; both qualifications below are audit-trail hygiene, not evidence revisions.

**Reproducibility note — IS t-statistic.** The cited `t=4.361` was computed in-memory by `chordia_strict_unlock_v1.py` from the pandas dataframe with full float64 precision. Independent recompute from the committed CSV (which serializes `pnl_r_effective` at 4 decimal places) yields `t=4.323`. Delta=0.038 is a CSV rounding artefact, not an evidence discrepancy — both clear the strict 3.79 hurdle (gap=0.57 vs gap=0.53). Anyone re-deriving the t-stat from this CSV alone should expect the lower value.

**OOS power floor — descriptive value of OOS sign-match.** Power to detect the IS-sized effect (`ExpR=+0.1506R`, `std=0.94`) at `N_OOS=49` is ≈0.30 (1-tailed, alpha=0.05). Per `memory/feedback_oos_power_floor.md`: at OOS power < 50%, a positive sign-match is uninformative — it neither confirms nor falsifies. The OOS gate as designed is a kill-on-sign-flip veto only; this lane does not trigger the veto, so the verdict stands on IS evidence alone. The original `## Verdict` and `## Split summary` framing implied OOS sign-match was a confirmatory data point — that framing overstated the OOS evidence and should be read together with this addendum.

**E2 look-ahead exposure (CLEARED, see registry row #30).** `X_MES_ATR60` was independently verified against `docs/audit/results/2026-04-28-e2-lookahead-contamination-registry.md` and `trading_app/config.py:3904-3912` (`E2_EXCLUDED_FILTER_PREFIXES` / `E2_EXCLUDED_FILTER_SUBSTRINGS`). The filter sources `daily_features.atr_20_pct` (rolling, prior-close, `resolves_at="STARTUP"`) on MES — not break-bar state. Not subject to the E2 break-bar contamination class. Recorded as registry row #30 `CLEARED`.
