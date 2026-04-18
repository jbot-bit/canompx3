# Phase D D-0 Backtest — MNQ COMEX_SETTLE O5 RR1.5 OVNRNG_100 rel_vol size-scaling

**Pre-registration:** `docs/audit/hypotheses/2026-04-18-phase-d-d0-rel-vol-sizing-mnq-comex-settle.yaml` (commit `b6918d8d`)
**Run UTC:** 2026-04-18T02:52:27.339072+00:00
**Holdout sacred from:** 2026-01-01
**Scope:** {"instrument": "MNQ", "orb_label": "COMEX_SETTLE", "orb_minutes": 5, "rr_target": 1.5, "entry_model": "E2", "confirm_bars": 1, "filter_type": "OVNRNG_100"}

## IS trade set
- N: 519
- First day: 2019-08-05
- Last day: 2025-12-31

## Frozen thresholds (pre-reg § calibration.thresholds)
- P33 rel_vol_COMEX_SETTLE: **1.0952**
- P67 rel_vol_COMEX_SETTLE: **1.9241**
- Frozen at UTC: 2026-04-18T02:52:27.338470+00:00

## Bucket distribution
- low: N=171
- mid: N=177
- high: N=171

## Per-bucket R-multiple statistics (unbiased diagnostic)
| Bucket | N | Mean R | Std R | Sharpe (per-trade) | Win rate | Total R |
|---|---:|---:|---:|---:|---:|---:|
| low | 171 | +0.1019 | 1.1537 | +0.0883 | 0.480 | +17.42 |
| mid | 177 | +0.1940 | 1.1786 | +0.1646 | 0.508 | +34.33 |
| high | 171 | +0.3178 | 1.1836 | +0.2685 | 0.556 | +54.34 |

## H1 — baseline vs sized (primary schema 0.5x / 1.0x / 1.5x)
| Variant | N | Mean R | Std R | Sharpe | Win rate | Total R |
|---|---:|---:|---:|---:|---:|---:|
| baseline 1.0x | 519 | +0.2044 | 1.1731 | +0.1743 | 0.514 | +106.10 |
| sized 0.5/1.0/1.5 | 519 | +0.2400 | 1.2832 | +0.1870 | 0.514 | +124.56 |

**Sharpe uplift:** **+7.33%**
**H1 verdict:** **KILL**

## Rule 8.2 arithmetic-only check (backtesting-methodology.md)
- Per-bucket WR: {'low': 0.47953216374269003, 'mid': 0.5084745762711864, 'high': 0.5555555555555556}
- WR spread across buckets: 7.60%
- Flag: False
- Interpretation: Not flagged — WR spread >= 3% OR uplift within noise band

## H2 ablation (low=0.0x, mid=1.0x, high=1.5x) — DESCRIPTIVE ONLY, not primary selector
| Variant | N (incl. zeros) | Mean R | Std R | Sharpe | Win rate | Total R | Low_Q1 trades skipped |
|---|---:|---:|---:|---:|---:|---:|---:|
| H2 hard-skip | 519 | +0.2232 | 1.2427 | +0.1796 | 0.356 | +115.85 | 171 |

**H2 note:** per pre-reg § hypotheses.H2.selection_rule, this is descriptive-only. It is not the primary D-0 selector and cannot replace H1 post hoc.

## K2 implementation integrity checklist (pre-reg § kill_criteria.K2)
**Overall: PASS**

- `p33_p67_calibrated_is_only`: True
- `oos_trades_in_sample`: 0
- `oos_consulted`: False
- `bucket_thresholds_frozen_before_sharpe`: True
- `bucket_thresholds_frozen_at`: 2026-04-18T02:52:27.338470+00:00
- `size_applied_at_entry_not_retro`: True
- `trade_count_baseline_eq_sized`: True

## No-OOS assertion
- OOS trades consulted during D-0: **False**
- OOS trades present in sample: **0** (must be 0)

## Outputs-required mapping (pre-reg § outputs_required_after_run)
- P33 and P67: YES (see thresholds)
- IS sample span and N: YES (see IS trade set)
- baseline Sharpe and sized Sharpe: YES (see H1 table)
- Sharpe uplift %: YES
- expectancy impact: YES (mean R comparison in H1 table)
- win-rate impact (Rule 8.2): YES
- trade count impact: YES (N_baseline == N_sized required check)
- H2 hard-skip ablation: YES
- no-OOS assertion: YES
- K2 checklist PASS/FAIL: YES