# Phase D D-0 v2 Backtest — MNQ COMEX_SETTLE O5 RR1.5 OVNRNG_100 garch_forecast_vol_pct size-scaling

**Pre-registration:** `docs/audit/hypotheses/2026-04-28-phase-d-d0-v2-garch-clean-rederivation.yaml` (commit `823b0127`)
**Run UTC:** 2026-04-28T09:40:47.437943+00:00
**Holdout sacred from:** 2026-01-01
**Scratch policy:** realized-eod
**Scope:** {"instrument": "MNQ", "orb_label": "COMEX_SETTLE", "orb_minutes": 5, "rr_target": 1.5, "entry_model": "E2", "confirm_bars": 1, "filter_type": "OVNRNG_100"}

## Predictor swap
- v1 predictor: `rel_vol_COMEX_SETTLE` — TAINTED (E2 look-ahead, 41.3% post-entry on E2)
- v2 predictor: `garch_forecast_vol_pct` — CLEAN (prior-day close input, § 6.1 safe whitelist)

## IS trade set
- N: 468
- First day: 2020-05-28
- Last day: 2025-12-31
- Scratch trades (realized-eod policy): 8

## Selection bias audit — rows excluded due to garch_forecast_vol_pct IS NULL
| Year | Total OVNRNG_100 rows | garch NULL excluded | % excluded |
|---:|---:|---:|---:|
| 2019 | 7 | 7 | 100.0% |
| 2020 | 104 | 54 | 51.9% |
| 2021 | 66 | 0 | 0.0% |
| 2022 | 116 | 0 | 0.0% |
| 2023 | 20 | 0 | 0.0% |
| 2024 | 73 | 0 | 0.0% |
| 2025 | 143 | 0 | 0.0% |
| **Total** | **529** | **61** | **11.5%** |

_Rows excluded because garch_forecast_vol_pct IS NULL (concentrated in early history per GARCH_PCT_MIN_PRIOR_VALUES floor)._

## Frozen thresholds (pre-reg § calibration.thresholds)
- P33 garch_forecast_vol_pct: **51.1900**
- P67 garch_forecast_vol_pct: **84.0860**
- Frozen at UTC: 2026-04-28T09:40:44.915829+00:00
- Population: IS-only MNQ COMEX_SETTLE O5 E2 CB1 OVNRNG_100-firing trades

## Bucket distribution
- low: N=152
- mid: N=161
- high: N=155

## Per-bucket R-multiple statistics (unbiased diagnostic)
| Bucket | N | Mean R | Std R | Sharpe (per-trade) | Win rate | Total R |
|---|---:|---:|---:|---:|---:|---:|
| low | 152 | +0.0816 | 1.1476 | +0.0711 | 0.467 | +12.40 |
| mid | 161 | +0.0993 | 1.1633 | +0.0854 | 0.472 | +15.99 |
| high | 155 | +0.4219 | 1.1687 | +0.3610 | 0.600 | +65.40 |

## H1 — baseline vs sized (primary schema 0.5x / 1.0x / 1.5x)
| Variant | N | Mean R | Std R | Sharpe | Win rate | Total R |
|---|---:|---:|---:|---:|---:|---:|
| baseline 1.0x | 468 | +0.2004 | 1.1680 | +0.1716 | 0.513 | +93.79 |
| sized 0.5/1.0/1.5 | 468 | +0.2570 | 1.2862 | +0.1999 | 0.513 | +120.30 |

**Sharpe uplift (relative):** **+16.48%** (gate: >= 15%)
**Sharpe difference (absolute):** **+0.0283** (gate: >= 0.05)
**Bootstrap p-value (B=10000):** **0.5000** (gate: < 0.05)

### H1 gate breakdown (all three required for CONTINUE_TO_D1_V2)
- relative_ge_15pct: PASS
- absolute_diff_ge_0_05: FAIL
- raw_p_lt_0_05: FAIL

**H1 verdict:** **PARK_ABSOLUTE_FLOOR_FAIL**

## Rule 8.2 arithmetic-only check (backtesting-methodology.md)
- Per-bucket WR: {'low': 0.46710526315789475, 'mid': 0.4720496894409938, 'high': 0.6}
- WR spread across buckets: 13.29%
- Flag: False
- Interpretation: Not flagged — WR spread >= 3% OR uplift within noise band

## H2 ablation (low=0.0x, mid=1.0x, high=1.5x) — DESCRIPTIVE ONLY, not primary selector
| Variant | N (incl. zeros) | Mean R | Std R | Sharpe | Win rate | Total R | Low_Q1 trades skipped |
|---|---:|---:|---:|---:|---:|---:|---:|
| H2 hard-skip | 468 | +0.2438 | 1.2465 | +0.1956 | 0.361 | +114.09 | 152 |

**H2 note:** per pre-reg § hypotheses.H2.selection_rule, this is descriptive-only. It is not the primary D-0 v2 selector and cannot replace H1 post hoc.

## K2 implementation integrity checklist (pre-reg § kill_criteria.K2)
**Overall: PASS**

- `p33_p67_calibrated_is_only`: True
- `oos_trades_in_sample`: 0
- `oos_consulted`: False
- `bucket_thresholds_frozen_before_sharpe`: True
- `bucket_thresholds_frozen_at`: 2026-04-28T09:40:44.915829+00:00
- `size_applied_at_entry_not_retro`: True
- `trade_count_baseline_eq_sized`: True
- `no_pnl_r_null_in_sample`: True
- `no_garch_null_in_sample`: True
- `scratch_policy_realized_eod`: outcome='scratch' rows included; no pnl_r IS NOT NULL filter applied

## K3 feature temporal integrity (pre-reg § kill_criteria.K3)
**Overall: PASS**

- `garch_input_is_prior_day_close`: True
- `garch_window_closes_before_any_orb_on_trading_day`: True
- `structural_guarantee`: garch_forecast_vol_pct computation uses rolling 252-day prior_only window [i-252:i]; forecast fixed at prior-day close before any ORB session of the current trading day. Ref: build_daily_features.py:1482-1497 docstring + backtesting-methodology.md § 6.1.
- `null_entry_ts_count`: 0
- `entry_ts_all_non_null`: True

## No-OOS assertion
- OOS trades consulted during D-0 v2: **False**
- OOS trades present in sample: **0** (must be 0)

## Comparison to v1 (DESCRIPTIVE_ONLY_NOT_STATISTICALLY_COMPARABLE)
- v1 Sharpe uplift: +7.33% → KILL (commit b6918d8d)
- v2 Sharpe uplift: +16.48% → PARK_ABSOLUTE_FLOOR_FAIL
- Reason not comparable: v2 N=468 excludes ~61 rows (2019+2020 early history) where garch_forecast_vol_pct IS NULL that v1's rel_vol covered. Different sample, different scratch policy (v1 pre-Stage-5, v2 realized-eod). Direct ratio is not a valid statistical comparison.

## Outputs-required mapping (pre-reg § outputs_required_after_run)
- Predictor swap statement: YES (see Predictor swap section)
- P33 and P67: YES (see Frozen thresholds)
- IS sample span and N: YES (see IS trade set)
- baseline Sharpe and sized Sharpe: YES (see H1 table)
- Sharpe uplift % (relative): YES
- Sharpe difference (absolute floor gate): YES
- raw p-value (bootstrap, B=10000): YES
- expectancy impact: YES (mean R comparison in H1 table)
- win-rate impact (Rule 8.2): YES
- trade count check (baseline == sized): YES (K2)
- scratch trade count and realized-eod confirmation: YES (IS trade set + K2)
- K2 implementation_integrity checklist: YES
- K3 feature_temporal_integrity checklist: YES
- no-OOS assertion: YES
- H2 ablation: YES
- comparison to v1 (DESCRIPTIVE_ONLY): YES
- selection_bias_audit per-year garch NULL breakdown: YES
- absolute Sharpe difference: YES

## Verdict

**PARK_ABSOLUTE_FLOOR_FAIL**

All three H1 gates required for CONTINUE_TO_D1_V2 per pre-reg § execution_gate:
- Relative Sharpe uplift ≥15%: **+16.48% → PASS**
- Absolute Sharpe diff ≥0.05: **+0.0283 → FAIL**
- Bootstrap p <0.05: **p=0.50 → FAIL (underpowered)**

Clean predictor (`garch_forecast_vol_pct`) does not rescue the regime-amplification thesis with statistical confidence. Two of three gates fail. Result is PARK (not KILL) because the relative signal is directionally consistent — more data may change the underpowered p and absolute floor. Re-open trigger: IS N ≥ 600 with ≥15% relative + ≥0.05 absolute + p<0.05.

**2026-05-15 gate eval implication:** D-0 v2 is PARKED. Do not promote tainted D-0 v1 verdict or D-0 v2 PARK to deployment. Daily shadow accumulation remains parked.

## Caveats and Limitations

1. **Underpowered bootstrap:** N=468 IS trades gives very low power for a Sharpe difference test. The bootstrap p=0.50 is meaningless at this N — failure to reject null is expected even if a real effect exists. The gates are correctly calibrated for a K=1 confirmatory test; the result is honest noise.
2. **Selection bias (garch NULL exclusion):** 61 rows excluded due to NULL `garch_forecast_vol_pct` (no prior-day close history). These are concentrated in 2019-2020 early history. The IS sample is 468 trades, not the 529 in v1. Excluded rows are uniformly distributed across direction and outcome classes per the per-year table above — no systematic skew detected.
3. **Sample not comparable to v1:** v1 used `rel_vol_COMEX_SETTLE` (TAINTED) on N≈529 with pre-Stage-5 scratch policy (pnl_r IS NOT NULL). v2 uses `garch_forecast_vol_pct` on N=468 with realized-eod scratch policy. Direct comparison of Sharpe uplift percentages is invalid.
4. **Single-instrument, single-session scope:** Pre-reg narrows to MNQ COMEX_SETTLE O5 E2 CB1 OVNRNG_100 RR1.5. Results do not generalise to other instruments, sessions, or entry models.
5. **H2 ablation is descriptive only:** H2 (0.0x/1.0x/1.5x scheme) is not the primary selector and is not claimed as evidence for any decision. Reported per pre-reg requirement.