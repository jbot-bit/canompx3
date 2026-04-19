# MNQ NYSE_CLOSE LONG direction-locked validation v1 — result

**Generated:** 2026-04-19
**Pre-reg:** `docs/audit/hypotheses/2026-04-19-mnq-nyse-close-long-direction-locked-v1.yaml`
**Script:** `research/research_mnq_nyse_close_long_direction_locked_v1.py`
**Canonical sources:** `gold.db::orb_outcomes`, `pipeline.cost_model`, `deployable_validated_setups` (for correlation gate only)
**IS boundary:** `trading_day < 2026-01-01` (Mode A per `trading_app/holdout_policy.py`)
**K:** 3 apertures, BH q=0.05, t>=3.00 (with-theory per Criterion 4)

## Family verdict

**PROMOTE — 2/3 apertures pass all required gates strict.**

| Aperture | N | avg_r | t | BH | C4 | C6 | C7 | C8 | C9 | Parity | Year | Boot | Corr | **STRICT** |
|---:|---:|---:|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 5 | 740 | +0.1180 | 3.60 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | **PASS** |
| 15 | 546 | +0.1792 | 4.66 | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | **PASS** |
| 30 | 440 | +0.1915 | 4.50 | Y | Y | **N** | Y | Y | Y | Y | Y | Y | Y | **FAIL** (C6 only) |

## Per-aperture detail (canonical pre-2026)

| Aperture | N_long | avg_r | t | p (two-tailed) | bootstrap p (moving-block) | WR | median risk $ | per-trade $ | annual gross $ at 5×3 scale |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 740 | +0.1180 | 3.599 | 3.4e-4 | 0.0023 | 33.2% | 39.92 | 4.71 | 7,466 |
| 15 | 546 | +0.1792 | 4.656 | 4.1e-6 | 0.0029 | 19.4% | 47.92 | 8.58 | 10,044 |
| 30 | 440 | +0.1915 | 4.504 | 8.6e-6 | 0.0014 | 15.2% | 50.42 | 9.66 | 9,105 |

**Dollar-EV declaration (per pre-reg):** annual gross $ at scale = avg_r × median_risk_dollars × (N/7yrs) × 5 copies × 3 contracts. Apertures are correlated (same session, overlapping days); realistic combined EV is NOT the naive sum. Single-aperture deployment recommended pending correlation audit between O5 and O15 fire days.

## Direction parity (SHORT on same cells)

| Aperture | LONG t | SHORT t | gap σ | SHORT also significant? | **Parity verdict** |
|---:|---:|---:|---:|:--:|:--:|
| 5 | 3.60 | 1.46 | 2.14 | N | PASS |
| 15 | 4.66 | 1.15 | 3.51 | N | PASS |
| 30 | 4.50 | 1.52 | 2.98 | N | PASS |

Direction asymmetry confirmed: LONG t-stats are materially higher than SHORT on all apertures; SHORT does not clear t≥3.00. Hypothesis is not an artifact of direction-blind encoding.

## Per-year R (LONG only, pre-2026)

| Aperture | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | +0.179 | +0.217 | +0.065 | +0.093 | -0.022 | +0.188 | +0.075 |
| 15 | -0.028 | +0.360 | +0.307 | +0.128 | +0.116 | +0.212 | +0.129 |
| 30 | -0.028 | +0.360 | +0.343 | -0.029 | +0.325 | -0.081 | +0.263 |

**Single-year dominance:** max share is 2020 at 0.34 (O5), 0.34 (O15), 0.49 (O30). All ≤ 0.50 → pass. 2020 is the top contributor for all three apertures; pre-reg gate holds.

## Era stability (Criterion 9)

| Aperture | Era | N | avg_r | Exempt? | Pass? |
|---:|---|---:|---:|:--:|:--:|
| 5 | 2019-2022 | 423 | +0.143 | — | **Y** |
| 5 | 2023 | 102 | -0.022 | — | **Y** (>=-0.05) |
| 5 | 2024-2025 | 215 | +0.128 | — | **Y** |
| 15 | 2019-2022 | 309 | +0.206 | — | **Y** |
| 15 | 2023 | 82 | +0.116 | — | **Y** |
| 15 | 2024-2025 | 155 | +0.164 | — | **Y** |
| 30 | 2019-2022 | 282 | +0.187 | — | **Y** |
| 30 | 2023 | 59 | +0.325 | — | **Y** |
| 30 | 2024-2025 | 99 | +0.101 | — | **Y** |

No era with N≥50 violates the -0.05 floor. O5 2023 is the closest (-0.022) but still passes.

## Walk-forward efficiency (Criterion 6) — pre-2026 only

IS window: `2019-05-06..2023-12-31` | Pseudo-OOS: `2024-01-01..2025-12-31`

| Aperture | N_IS | N_OOS | IS Sharpe | OOS Sharpe | WFE | **Pass (>=0.50)** |
|---:|---:|---:|---:|---:|---:|:--:|
| 5 | 525 | 215 | 0.147 | 0.160 | **1.088** | Y |
| 15 | 391 | 155 | 0.246 | 0.210 | **0.853** | Y |
| 30 | 341 | 99 | 0.213 | 0.096 | **0.448** | **N** |

**O30 fails C6.** Pseudo-OOS (2024-2025) Sharpe collapses from IS (2019-2023). Interpretation: O30 LONG has real pre-2023 edge but materially weaker edge in recent years — likely a thinner-aperture / shorter-window tail-dependency pattern rather than a contamination artifact. Consistent with single-year table (2024 O30 is -0.081; 2025 recovers to +0.263).

## 2026 diagnostic (Criterion 8 — pre-registered gate)

| Aperture | N_2026 | avg_r 2026 | 2026/IS ratio | **Pass** |
|---:|---:|---:|---:|:--:|
| 5 | 29 | +0.459 | 3.89× | **Y** |
| 15 | 20 | +0.370 | 2.06× | **Y** |
| 30 | 14 | +0.920 | 4.80× | **Y** |

2026 is declared as **pre-registered gate**, not post-hoc tuning. All three apertures exceed the C8 0.40× ratio by wide margins. 2026 N per aperture is thin (<30) — treat as directional confirmation, not statistical confirmation.

## T8 cross-instrument diagnostic (informational)

MES NYSE_CLOSE LONG direction-locked, same cell spec pre-2026:

| Aperture | N | avg_r | t |
|---:|---:|---:|---:|
| 5 | 758 | +0.002 | +0.08 |
| 15 | 535 | -0.041 | -1.09 |
| 30 | 439 | -0.066 | -1.57 |

**MES is flat to slightly negative.** The MNQ LONG edge does NOT transfer to MES at direction-locked level. Mechanism (US cash-close directional flows per Fitschen 2013) is MNQ-specific; Nasdaq futures close alignment differs from S&P 500 (MES) close alignment in a way canonical data supports. Not a kill — direction-locked validation is instrument-specific by design. But framework-level claims about "US equity-index futures" cannot rely on MES here.

## Correlation vs 36 live MNQ deployable lanes

**0 pairs at Jaccard ≥ 0.50; 0 pairs at Jaccard ≥ 0.70.**

All three candidate apertures on MNQ NYSE_CLOSE show essentially zero overlap with any of the 36 current live MNQ deployable strategy_ids on a pre-2026 fire-day basis. NYSE_CLOSE is a new session for the allocator — genuinely additive capacity, not a disguised overlap. Correlation gate passes cleanly.

## Pre-registered gate table (summary)

| Criterion | Threshold | O5 | O15 | O30 |
|---|---|:--:|:--:|:--:|
| C1 pre-registration | file exists at run time | Y | Y | Y |
| C2 MinBTL | K=3 ≤ 28 @ E=1.0 on 6.65yr clean MNQ | Y | Y | Y |
| C3 BH-FDR | BH-adjusted p ≤ 0.05 (K=3) | Y | Y | Y |
| C4 t-stat (with theory) | t ≥ 3.00 | Y (3.60) | Y (4.66) | Y (4.50) |
| C5 DSR | informational cross-check only (Amendment 2.1) | — | — | — |
| C6 WFE | WFE ≥ 0.50 | Y (1.09) | Y (0.85) | **N (0.45)** |
| C7 N | N ≥ 100 | Y (740) | Y (546) | Y (440) |
| C8 2026 OOS | avg_r > 0 AND ratio ≥ 0.40 | Y | Y | Y |
| C9 era stability | no era ExpR < -0.05 (N≥50) | Y | Y | Y |
| C10 data era compat | not volume filter | N/A | N/A | N/A |
| Parity (adversarial) | LONG t − SHORT t ≥ 1.5σ AND SHORT not sig | Y | Y | Y |
| Single-year (adversarial) | ≤ 0.50 share | Y (0.34) | Y (0.34) | Y (0.49) |
| Bootstrap (adversarial) | moving-block p ≤ 0.05 | Y (0.0023) | Y (0.0029) | Y (0.0014) |
| Correlation (adversarial) | Jaccard < 0.70 vs all 36 live MNQ | Y | Y | Y |
| **STRICT pass** | all required | **Y** | **Y** | **N (C6)** |

## Verdict and recommended disposition

**O5** and **O15** apertures PROMOTE under all required gates. **O30** FAILS the C6 pre-registered walk-forward efficiency gate (WFE=0.448 < 0.50) and is NOT eligible for promotion despite passing every other gate. O30's weaker WFE is driven by 2024 (-0.081) and a concentrated single-year contribution from 2020; the pattern is real but does not meet the locked walk-forward threshold.

**Promotion artifact naming** (pre-reg recommendation): `MNQ_NYSE_CLOSE_E2_RR1.0_CB{5|15}_DIR_LONG` using the existing `DIR_LONG` filter from `trading_app/config.py:3050`. Follows the TOKYO_OPEN / MNQ SINGAPORE_OPEN live precedent of direction-locking at the filter layer rather than inventing a new strategy_id convention.

**DEPLOYABILITY IS GATED** on a separate code change: `trading_app/portfolio.py:633, :997, :1011` hardcode `exclude_sessions={"NYSE_CLOSE"}` in both raw-baseline and multi-RR builders with a direction-blind rationale (`docs/STRATEGY_BLUEPRINT.md:215` "Low WR 30.8%"). That rationale applies only to the direction-blind aggregate and not to the LONG-only subset validated here. Disposition = open a follow-up branch for `portfolio.py` annotation / re-scope before any allocator consumption of a new deployable_validated_setups row on NYSE_CLOSE.

## Next actions (post-validation)

1. **DO NOT** write to `validated_setups` or `deployable_validated_setups` from this worktree. That insertion requires canonical `trading_app/strategy_validator.py` pre-promotion gate flow, not a research script.
2. **Follow-up branch A** — `trading_app/portfolio.py` exclusion re-scope with `@canonical-source` annotation tying the carve-out to the direction-blind aggregate specifically. Paired blast-radius audit of `get_filters_for_grid()` DIR_LONG wiring for NYSE_CLOSE.
3. **Follow-up branch B** — once A lands, run the canonical promotion path for `MNQ_NYSE_CLOSE_E2_RR1.0_CB5_DIR_LONG` and `MNQ_NYSE_CLOSE_E2_RR1.0_CB15_DIR_LONG`. O30 stays graveyard-candidate pending a fresh pre-reg that addresses the WFE failure mode (or stays dead if no mechanism separates O30 from O5/O15).
4. **Do NOT** extend the mechanism claim to MES or MGC without a dedicated pre-reg — cross-instrument T8 is flat/negative.
5. **Adversarial audit queued:** bootstrap sensitivity to block_size (3/5/10) + a holdout-aware walk-forward with the first IS year shifted forward one year to check 2019-start dependency of the WFE margin.

## CSV outputs (this session)

- `research/output/mnq_nyse_close_long_direction_locked_v1_apertures.csv`
- `research/output/mnq_nyse_close_long_direction_locked_v1_years.csv`
- `research/output/mnq_nyse_close_long_direction_locked_v1_eras.csv`
- `research/output/mnq_nyse_close_long_direction_locked_v1_direction_parity.csv`
- `research/output/mnq_nyse_close_long_direction_locked_v1_correlation.csv`

Outputs are generated under `research/output/` which is `.gitignore`d per project convention; reproduce by running:
`DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/research_mnq_nyse_close_long_direction_locked_v1.py`
