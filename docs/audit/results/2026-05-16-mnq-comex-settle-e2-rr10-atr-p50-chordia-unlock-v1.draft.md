# Chordia strict unlock audit — MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50

**Prereq file:** `docs\audit\hypotheses\drafts\2026-05-16-mnq-comex-settle-e2-rr10-atr-p50-chordia-unlock-v1.draft.yaml`
**Result CSV:** `docs\audit\results\2026-05-16-mnq-comex-settle-e2-rr10-atr-p50-chordia-unlock-v1.draft.csv`
**Canonical DB:** `C:\Users\joshd\canompx3\gold.db`

## Scope

Strict-Chordia unlock audit for the exact lane `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ATR_P50`. Tests whether the bounded canonical replay clears Chordia's strict t-stat hurdle (3.79, has_theory=False) on canonical IS data, with descriptive OOS sign-match as a secondary gate. Single-lane K=1 confirmatory replay; no parameter sweeps, no filter variants, no instrument extensions.

## Verdict

**MEASURED verdict (runner):** `PASS_CHORDIA`
**ROLE-DECISION verdict (RULE 3.3 power-floor override):** `UNVERIFIED_INSUFFICIENT_POWER`

IS clears strict threshold 3.79 with N=837 and ExpR=0.1451; OOS sign matches at N_OOS=59. **Role-decision override:** OOS one-sample power against the IS effect is 22.9% (pooled), 15.1% (long-only), 11.9% (short-only) — all in the `STATISTICALLY_USELESS` tier per `research.oos_power.power_verdict` and RULE 3.3 of `.claude/rules/backtesting-methodology.md`. Sign-match at this N is noise-consistent and cannot serve as a confirmatory gate. Memory anchor: `feedback_chordia_oos_park_vs_unverified_power_floor.md` mandates the override.

**MEASURED theory mode:** `NO_THEORY_GRANT`
**MEASURED threshold applied:** `3.79`
**MEASURED loader has_theory:** `False`

## OOS power audit (RULE 3.3)

Computed via `research.oos_power.oos_ttest_power` / `one_sample_power` on the result CSV at commit-time (no script-derived overrides).

| Framing | Cohen's d (IS) | N_OOS | Power @ α=0.05 | N per group for 80% | RULE 3.3 tier |
|---|---:|---:|---:|---:|---|
| OOS pooled ExpR vs 0 (one-sample) | 0.1608 | 59 | 22.9% | 306 | `STATISTICALLY_USELESS` |
| OOS long ExpR vs 0 (one-sample) | 0.1707 | 31 | 15.1% | 272 | `STATISTICALLY_USELESS` |
| OOS short ExpR vs 0 (one-sample) | 0.1492 | 28 | 11.9% | 355 | `STATISTICALLY_USELESS` |
| OOS long vs short (two-sample) | 0.020 | 31 / 28 | 5.1% | 37 913 | `STATISTICALLY_USELESS` |

Per RULE 3.3 tier table, `STATISTICALLY_USELESS` requires verdict `UNVERIFIED`, never `DEAD` and never auto-pass. The runner's `PASS_CHORDIA` label survives as the IS measurement; the role decision adds the power-floor gate the runner does not enforce.

## OOS long collapse — outlier check

The long-side OOS ExpR (0.0128) collapses ~12× from IS (0.1535). Diagnosed for outlier contamination via leave-out sensitivity on the 31 OOS long trades:

| Subset | N | ExpR |
|---|---:|---:|
| base | 31 |  0.0128 |
| drop top 1 | 30 | −0.0190 |
| drop top 2 | 29 | −0.0529 |
| drop bottom 1 | 30 |  0.0465 |
| drop bottom 2 | 29 |  0.0826 |
| trim 10% (1+1) | 29 |  0.0148 |

**Not outlier-driven.** The collapse is a WR shift, not a single-day artefact. IS long WR=62.2% (avg_win=0.852, avg_loss=−0.994); OOS long WR=54.8% (avg_win=0.847, avg_loss=−1.000). Win and loss magnitudes are essentially identical IS vs OOS — only hit-rate moved. At RR=1.0 this 7.4 pp WR drop fully explains the ExpR collapse. Whether this is regime drift or sampling noise cannot be distinguished at N_OOS_long=31 (power 15.1%).

## Role decision (no allocator action)

Per the pre-reg's `conditional_role` routing and the `forbidden claims` enumerated by the prereg front-door:

- This result MUST NOT be written to `experimental_strategies`.
- This result MUST NOT be auto-promoted to `validated_setups`.
- No `lane_allocation.json` mutation. No paper-trades pipeline activation.
- Verdict for any downstream consumer: `UNVERIFIED_INSUFFICIENT_POWER` (not `PASS_CHORDIA`, not `PARK`, not `DEAD`).

Path forward to legitimate role-decision:
1. Allow OOS to accumulate to N ≥ 306 (one-sample 80% power at d=0.16) before any binary OOS gate is reapplied. At current ~59 trades over 2026-01-09 → 2026-04-14 (~3.4 months), the floor lands ~early 2027.
2. Or replace the binary OOS gate with a Harvey-Liu 2015 Sharpe haircut (`docs/institutional/literature/harvey_liu_2015_backtesting.md`) treating OOS as discount multiplier rather than veto.
3. Or run CPCV (AFML 2018 Ch 12) on the combined sample to bypass the underpowered 2026 holdout slice. Requires explicit amendment to the prereg.

None of these are in scope of the current draft; this result is closed at `UNVERIFIED_INSUFFICIENT_POWER` pending a future role-decision pre-reg.

## Split summary

| Split | N_universe | N_fired | Fire% | Scratch | Null non-scratch | ExpR | Policy EV/opp | Sharpe | t | p_two |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 1494 | 837 | 56.02% | 4 | 0 | 0.1451 | 0.0813 | 0.1608 | 4.652 | 0.00000 |
| OOS | 77 | 59 | 76.62% | 1 | 0 | 0.1463 | 0.1121 | 0.1571 | 1.206 | 0.22764 |

## Directional breakdown

| Split | Long N | Long ExpR | Long t | Short N | Short ExpR | Short t |
|---|---:|---:|---:|---:|---:|---:|
| IS | 452 | 0.1535 | 3.628 | 385 | 0.1352 | 2.927 |
| OOS | 31 | 0.0128 | 0.075 | 28 | 0.2942 | 1.715 |

## Method notes

- Canonical source only: `orb_outcomes` joined to `daily_features` on `(trading_day, symbol, orb_minutes)`.
- Sacred holdout boundary: `trading_day < 2026-01-01` for IS, `>=` for descriptive OOS.
- Cohort lower bound: `WF_START_OVERRIDE['MNQ']=2020-01-01` applied to match canonical promoter (`trading_app/strategy_discovery._load_outcomes_bulk`).
- Canonical filter delegation: `filter_signal(..., 'ATR_P50', 'COMEX_SETTLE')`.
- Scratch handling: `pnl_r NULL -> 0.0` in the measured trade stream; scratch and null-non-scratch counts are reported separately.
- No writes to `experimental_strategies`, `validated_setups`, or `docs/runtime/chordia_audit_log.yaml`.

## Reproduction

```
python research/chordia_strict_unlock_v1.py --hypothesis-file docs\audit\hypotheses\drafts\2026-05-16-mnq-comex-settle-e2-rr10-atr-p50-chordia-unlock-v1.draft.yaml
```

Outputs (overwritten in place):

- `docs\audit\results\2026-05-16-mnq-comex-settle-e2-rr10-atr-p50-chordia-unlock-v1.draft.md`
- `docs\audit\results\2026-05-16-mnq-comex-settle-e2-rr10-atr-p50-chordia-unlock-v1.draft.csv`

## Caveats

- Single-lane K=1 confirmatory replay; the strict t-stat hurdle does not include a search-family multiple-comparison correction. Survivorship/multiple-testing risk is carried by the upstream pre-registration, not this replay.
- IS sample size in this audit reports `N_fired` (wins+losses+scratches with R=0). `validated_setups.sample_size` reports wins+losses only. Comparing the two t-stats directly is not like-for-like; reconcile via the scratch count reported above.
- OOS window is descriptive only. Sign-match at `N_OOS >= 30` is a confirmatory gate, not a deployment criterion. PARK on OOS sign-flip means insufficient confirmation, not falsification.
- Cross-asset enrichment (e.g., `cross_atr_MES_pct` for `X_MES_ATR60`) is computed in this runner from `daily_features.atr_20_pct` of the source instrument; verify the canonical promoter's enrichment path agrees before treating verdicts as directly comparable.
