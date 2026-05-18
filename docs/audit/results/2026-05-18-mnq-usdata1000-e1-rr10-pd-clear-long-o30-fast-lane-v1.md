# Chordia strict unlock audit — MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30

**Prereq file:** `docs\audit\hypotheses\2026-05-18-mnq-usdata1000-e1-rr10-pd-clear-long-o30-fast-lane-v1.yaml`
**Result CSV:** `docs\audit\results\2026-05-18-mnq-usdata1000-e1-rr10-pd-clear-long-o30-fast-lane-v1.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.79, has_theory=False) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict:** `FAIL_STRICT_CHORDIA`

IS t=3.064 < 3.79.

**MEASURED theory mode:** `UNSUPPORTED`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1539 | 226 | 14.68% | 60 | 5 | 0.1708 | 0.0251 | 0.2038 | 3.064 | 0.00218 |
| OOS | 72 | 14 | 19.44% | 4 | 0 | -0.0233 | -0.0045 | -0.0263 | -0.099 | 0.92151 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 221 | 0.1747 | 3.065 | 5 | 0.0000 | nan |
| OOS | 14 | -0.0233 | -0.099 | 0 | nan | nan |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'PD_CLEAR_LONG', 'US_DATA_1000')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\2026-05-18-mnq-usdata1000-e1-rr10-pd-clear-long-o30-fast-lane-v1.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-18-mnq-usdata1000-e1-rr10-pd-clear-long-o30-fast-lane-v1.md`
- `docs\audit\results\2026-05-18-mnq-usdata1000-e1-rr10-pd-clear-long-o30-fast-lane-v1.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.

## FAST_LANE v5.1 verdict (operator-applied)

The runner's `Verdict:` line above (`FAIL_STRICT_CHORDIA`) uses the heavyweight Chordia threshold (t >= 3.79) — that is the runner's native gate, not this pre-reg's gate.

This pre-reg is a **FAST_LANE v5.1 triage screen** (`docs/audit/hypotheses/TEMPLATE-fast-lane-v5.1.yaml`). v5.1 is a cheaper screen that promotes cells to heavyweight Chordia review, NOT to capital. The runner emits the v5.1 inputs in the sibling `.summary.csv`; this section applies the v5.1 mapping by hand because no automated post-processor exists yet (link gap tracked for follow-up).

### v5.1 gate table (IS row of `.summary.csv`)

| v5.1 gate | Threshold | Observed | Pass |
|---|---|---|---|
| Holdout boundary proof | `max_IS < 2026-01-01 <= min_OOS` | `2025-12-31 < 2026-01-01 <= 2026-01-02`, proof=True | yes |
| `t_IS >= promote_threshold (2.5) + needs_more_band (0.5) = 3.0` | t >= 3.0 for PROMOTE | t = 3.064 | yes (margin = 0.064) |
| `ExpR_IS > 0` | > 0 | 0.171 | yes |
| `N_IS_on >= 50` | >= 50 | n_fired = 226 | yes |
| `fire_rate in [0.05, 0.95]` | inside band | 0.147 | yes |
| Per-direction sign-check | `direction=long` lanes BYPASS per v5.1 `single_direction_lanes` rule | n/a (long_n=221, short_n=5 is structural artefact of long-only filter) | bypass |

### Verdict: **PROMOTE**

Per the v5.1 template: "worth heavyweight Chordia review — NOT a deploy verdict." The cell sits in the screen-PROMOTE / heavyweight-FAIL band (t = 3.064 clears the 3.0 fast-lane floor but not the 3.79 Chordia hurdle), which is exactly the design intent of the v5.1 triage screen — surface candidates for heavyweight scrutiny without paying the heavyweight cost upfront.

### What PROMOTE authorizes

- Authoring a heavyweight Chordia pre-reg for this lane (theory grant + clustered SE at trading_day + OOS power floor + era-stability section + DSR + Harvey-Liu haircut).
- Nothing else.

### What PROMOTE does NOT authorize

- Capital allocation.
- Writing this cell into `chordia_audit_log.yaml` as PASS_CHORDIA.
- Sibling-cell rescue (other RR / CB / aperture / session variants need their own pre-reg).
- Treating the fast-lane verdict as a substitute for paper-trade + SR-monitor validation.

### OOS row (descriptive only)

OOS `n_fired = 14`, `t = -0.099`. Per v5.1 § `outcomes.NEEDS-MORE` precedence, OOS power is well below the 30-cluster floor for clustered-SE interpretation; the OOS row cannot refute or confirm the IS finding. Holdout proof is TRUE — evidence boundary is intact, the OOS slice is simply too small to be informative yet.
