# Phase D D5 — MNQ COMEX_SETTLE BOTH-SIDES Conditional Sizing — Result

**Pre-reg:** docs/audit/hypotheses/2026-04-28-mnq-comex-settle-garch-d5-both-sides-pathway-b-v1.yaml
**Pre-reg commit:** 416855cd96f3b4475ed2790b288caf9b66ec7311
**Run timestamp:** 2026-04-28
**DB freshness:** orb_outcomes max trading_day = 2026-04-26
**Holdout boundary (Mode A):** 2026-01-01

## Scope

Pathway B K=1 confirmatory test of the D5 conditional sizing rule on the deployed `MNQ COMEX_SETTLE O5 RR1.5 E2 CB1 BOTH-SIDES ORB_G5` cohort. Locked transformation: `pnl_r_d5 = pnl_r if garch_forecast_vol_pct > 70 else 0.5 * pnl_r`. PRIMARY METRIC: SR_ann diff (D5 minus flat-1.0x baseline). Hypothesis: halving exposure on the no-edge cohort improves risk-adjusted return via variance reduction (Carver Ch 9-10 vol-targeting framework). This is a RISK-OVERLAY test, not an alpha test.

## Locked schema (verbatim from pre-reg)

- instrument: MNQ
- session: COMEX_SETTLE
- orb_minutes: 5
- rr_target: 1.5
- entry_model: E2
- confirm_bars: 1
- direction_segmentation: ('long', 'short')
- feature: garch_forecast_vol_pct
- feature_threshold: 70.0
- feature_op: >
- orb_g5_min_size: 5.0
- on_branch_multiplier: 1.0
- off_branch_multiplier: 0.5
- year_no_flip_floor_sr_pt: 0.1
- scratch_policy: include-as-zero (commit 68ee35f8)

## Amendment 3.2 classification (locked at pre-reg time)

- sd_monthly_sharpe_diff_empirical: 0.0475
- expected_monthly_sharpe_diff: 0.05
- required_n_oos_months: 7
- min_trl_years: 0.59
- classification: STANDARD
- classification_revised_from: EXTENDED_PARK

## IS reproduction (Mode A, BOTH-SIDES combined)

| Metric | IS combined | IS long | IS short |
|---|---:|---:|---:|
| N | 1577 | 839 | 738 |
| ExpR flat (mean pnl_r) | +0.0941 | +0.0950 | +0.0930 |
| ExpR D5 (mean pnl_r_d5) | +0.0761 | +0.0788 | +0.0731 |
| SR_pt flat | +0.0825 | — | — |
| SR_pt D5 | +0.1005 | — | — |
| SR_ann flat | +1.2381 | — | — |
| SR_ann D5 | +1.5086 | — | — |
| **abs SR_ann diff** | **+0.2705** | — | — |
| **rel SR_ann uplift %** | **+21.85%** | — | — |

## Significance (paired test on per-trade differences)

- Paired t-stat (D5 vs flat per-trade R): -1.4473
- Paired p (two-tailed): 0.148019
- Block-bootstrap paired p (block=5, B=10000): 0.135986
- Mean diff per trade (D5 - flat): -0.0180

## Pre-reg expected values vs runner reproduction (KILL_BASELINE_SANITY)

| Metric | Expected (pre-reg) | Reproduced (runner) | abs diff |
|---|---:|---:|---:|
| SR_pt_flat_is | +0.0825 | +0.0825 | 0.000009 |
| SR_pt_d5_is | +0.1005 | +0.1005 | 0.000011 |
| SR_ann_flat_is | +1.2381 | +1.2381 | 0.000043 |
| SR_ann_d5_is | +1.5086 | +1.5086 | 0.000025 |

## Per-year IS breakdown (year, N, SR_pt flat, SR_pt D5, diff)

| Year | N | SR_pt flat | SR_pt D5 | diff |
|---:|---:|---:|---:|---:|
| 2019 | 99 | -0.2099 | -0.2099 | +0.0000 |
| 2020 | 236 | +0.1006 | +0.1109 | +0.0104 |
| 2021 | 248 | +0.0459 | +0.0977 | +0.0519 |
| 2022 | 250 | +0.0018 | +0.0149 | +0.0131 |
| 2023 | 248 | +0.1228 | +0.1058 | -0.0169 |
| 2024 | 249 | +0.1167 | +0.1399 | +0.0232 |
| 2025 | 247 | +0.2129 | +0.2401 | +0.0272 |

## C9 era stability (D5 variant)

- Verdict: **FAIL ([(2019, -0.10751464646464647)])**
- Worst era: 2019 ExpR_D5 = -0.1075
- Threshold: era ExpR_D5 must be >= -0.05 for any era with N >= 50

## OOS descriptive (Amendment 3.2 STANDARD: full C8 deferred to milestone)

- N_OOS_combined: 70 (long=35, short=35)
- SR_pt flat OOS: +0.0270
- SR_pt D5 OOS: +0.1151
- abs SR_ann diff OOS (descriptive): +0.7377
- OOS dir-match descriptive: DIR_MATCH

## Kill-criterion results (locked from pre-reg)

| ID | Threshold | Computed | Verdict |
|---|---|---|---|
| KILL_DIR | sign(SR_ann_D5 - SR_ann_flat) on IS == negative | sign=+1 | PASS |
| KILL_ABS_FLOOR | abs(SR_ann diff) >= 0.05 | +0.2705 | PASS |
| KILL_REL_FLOOR | rel uplift pct >= 15.0% | +21.85% | PASS |
| KILL_PAIRED_P | paired t p < 0.05 | p=0.1480 | FAIL_KILL_OR_PARK |
| KILL_YEAR_FLIP | max(SR_pt_flat - SR_pt_D5) for any IS year (N>=50) > 0.1 | worst_year=2023 flip=+0.0169 | PASS |
| KILL_N | N_IS_combined >= 100 | N=1577 | PASS |
| KILL_ERA | any IS era (N>=50) ExpR_D5 < -0.05 | worst_era=2019 ExpR_D5=-0.1075 | FAIL_KILL |
| KILL_BASELINE_SANITY | abs(reproduced sr_pt_flat - expected) <= 0.0001 | abs_diff=0.000009 | PASS |

## Decision rule outcome

**VERDICT: KILL**

At least one KILL criterion fired: KILL_ERA. Pre-reg locked-decision rule: any KILL fire → KILL verdict.

## Audit pressure-test (RULE 13)

- Pressure test: PASS

## Reproduction

```
DUCKDB_PATH=C:\Users\joshd\canompx3\gold.db python research/phase_d_d5_mnq_comex_settle_d5_both_sides.py
```

- DB: `pipeline.paths.GOLD_DB_PATH`
- Holdout: `trading_app.holdout_policy.HOLDOUT_SACRED_FROM` (2026-01-01)
- Pre-reg: locks all schema parameters and kill criteria
- Sizing rule LOCKED: 1.0x if garch>70 else 0.5x; threshold 70 inherited from D4

## Caveats and limitations

- D5 is a RISK-OVERLAY test (variance reduction), not an alpha test. The per-trade ExpR
  REDUCES under D5 (off-cohort half-sized → smaller per-trade R). The improvement is in
  Sharpe via variance reduction.
- 2019 IS year has N=99 and SR_pt diff = 0.0000 (likely all off-cohort or near-degenerate),
  flagged by the per-year breakdown. Not a flip but worth noting.
- Live execution at 1-contract minimum cannot literally express '0.5×' sizing — the research
  model is a portfolio-level weighting that requires either ≥2-contract lane scaffolding or a
  shadow/observation overlay. This pre-reg does NOT prescribe live execution; deployment
  requires Phase E + capital-review + execution-translation design.
- RULE 7 N/A by design (modifies existing deployed lane, doesn't add a slot).
- Cross-session BH-FDR at K=12 (D5 framing): best q=0.0707 — the GARCH effect is
  COMEX_SETTLE-specific, not universal. D5 is cell-specific risk overlay, not a universal mechanism.

## Not done by this run

- No write to validated_setups, edge_families, lane_allocation, live_config
- No paper trade simulation
- No live execution-translation design (separate Phase E step if D5 PASSES)
- No continuous-scaling DoF expansion (locked binary 1.0/0.5)
- No threshold sensitivity test (locked at 70)
- No capital deployment — requires Phase E + capital-review skill + explicit user GO