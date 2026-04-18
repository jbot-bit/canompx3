# Allocator Rho Audit — Excluded Lanes

**Date:** 2026-04-18
**Profile:** `topstep_50k_mnq_auto` (topstep 50K, max_slots=7, copies=2)
**Rebalance date audited:** 2026-04-18
**Lanes scored by allocator:** 38
**Eligible (profile-filtered):** 30
**Selected (live-6):** 6
**Excluded candidates analysed:** 24

## Phase 1 audit output — methodology

- Reproduces the 2026-04-18 allocator rebalance end-to-end using canonical
  `trading_app.lane_allocator.{compute_lane_scores, compute_pairwise_correlation, build_allocation}`.
- Self-consistency check: reproduction must match `docs/runtime/lane_allocation.json` selected set exactly.
- Gate-order classification: rank → rho → DD (first-failing wins).
- Mode A/B labels: trailing_expr uses allocator's 12mo trailing window which INCLUDES post-2026-01-01 data (Mode A OOS already consumed by allocator rebalance; this audit re-reads that same window — no new OOS consumption).
- Bootstrap CI + Fisher-z p-value + BH-FDR at q=0.05 applied to "rho<0.70" claims.
- Literature footnote: rho<0.70 is the allocator's threshold. Markowitz 1952 suggests rho<~0.3 for material risk reduction — lenient threshold.

## Self-consistency check

**PASS.** Reproduction selected 6 strategy_ids, matches live-6 in lane_allocation.json.

## Verdict breakdown

| Verdict | Count |
|---|---:|
| TRUE_UNLOCK | 0 |
| BLOCKED_BY_RANKING | 20 |
| BLOCKED_BY_RHO | 4 |
| BLOCKED_BY_DD | 0 |

## Per-excluded-lane classification

Trailing-12mo data — mode label: **[TRAILING-12MO, includes 2026 Q1 already consumed by allocator]**

| Rank | strategy_id | eff_annual_r | max_rho | paired_with | dd_$ | dd_headroom_$ | fits_dd | fdr_rho<0.70 | verdict |
|---:|---|---:|---:|---|---:|---:|:---:|:---:|---|
| 3 | `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30` | 44.00 | +1.000 | `SINGAPORE_OPEN_RR1.5_ATR_P50_O` | 57 | 1418 | Y | . | **BLOCKED_BY_RHO** |
| 5 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_COST_LT12` | 40.60 | +1.000 | `EUROPE_FLOW_RR1.5_ORB_G5` | 58 | 1418 | Y | . | **BLOCKED_BY_RHO** |
| 6 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM` | 39.20 | +1.000 | `EUROPE_FLOW_RR1.5_ORB_G5` | 58 | 1418 | Y | . | **BLOCKED_BY_RHO** |
| 7 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_ORB_G5` | 38.10 | +0.861 | `EUROPE_FLOW_RR1.5_ORB_G5` | 58 | 1418 | Y | . | **BLOCKED_BY_RHO** |
| 8 | `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_OVNRNG_100` | 37.40 | +1.000 | `EUROPE_FLOW_RR1.5_ORB_G5` | 58 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 9 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_ORB_G5` | 34.50 | +0.799 | `EUROPE_FLOW_RR1.5_ORB_G5` | 58 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 10 | `MNQ_COMEX_SETTLE_E2_RR2.0_CB1_ORB_G5` | 34.20 | +0.849 | `COMEX_SETTLE_RR1.5_ORB_G5` | 79 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 11 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_COST_LT12` | 32.60 | +0.803 | `EUROPE_FLOW_RR1.5_ORB_G5` | 58 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 12 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_OVNRNG_100` | 30.80 | +0.804 | `COMEX_SETTLE_RR1.5_ORB_G5` | 79 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 14 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G5` | 27.40 | +0.805 | `COMEX_SETTLE_RR1.5_ORB_G5` | 79 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 15 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_OVNRNG_100` | 26.80 | +0.827 | `EUROPE_FLOW_RR1.5_ORB_G5` | 58 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 16 | `MNQ_EUROPE_FLOW_E2_RR2.0_CB1_CROSS_SGP_MOMENTUM` | 26.60 | +0.860 | `EUROPE_FLOW_RR1.5_ORB_G5` | 58 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 17 | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_COST_LT12` | 24.90 | +0.807 | `COMEX_SETTLE_RR1.5_ORB_G5` | 79 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 18 | `MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM` | 23.50 | +0.797 | `EUROPE_FLOW_RR1.5_ORB_G5` | 58 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 20 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | 20.05 | +1.000 | `COMEX_SETTLE_RR1.5_ORB_G5` | 79 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 21 | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5` | 19.10 | +1.000 | `TOKYO_OPEN_RR1.5_COST_LT12` | 68 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 23 | `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5` | 18.60 | +0.837 | `TOKYO_OPEN_RR1.5_COST_LT12` | 68 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 24 | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5` | 14.50 | +1.000 | `NYSE_OPEN_RR1.0_COST_LT12` | 177 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 25 | `MNQ_US_DATA_1000_E2_RR1.0_CB1_VWAP_MID_ALIGNED_O15` | 13.80 | +0.832 | `US_DATA_1000_RR1.5_ORB_G5_O15` | 142 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 26 | `MNQ_US_DATA_1000_E2_RR2.0_CB1_VWAP_MID_ALIGNED_O15` | 12.10 | +0.892 | `US_DATA_1000_RR1.5_ORB_G5_O15` | 142 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 27 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_COST_LT12` | 12.00 | +0.825 | `NYSE_OPEN_RR1.0_COST_LT12` | 177 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 28 | `MNQ_NYSE_OPEN_E2_RR1.5_CB1_ORB_G5` | 12.00 | +0.825 | `NYSE_OPEN_RR1.0_COST_LT12` | 177 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 29 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | 8.00 | +1.000 | `US_DATA_1000_RR1.5_ORB_G5_O15` | 142 | 1418 | Y | . | **BLOCKED_BY_RANKING** |
| 30 | `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_COST_LT12` | 7.10 | +0.815 | `TOKYO_OPEN_RR1.5_COST_LT12` | 68 | 1418 | Y | . | **BLOCKED_BY_RANKING** |

## TRUE_UNLOCK candidates

**NONE.** The allocator's correlation + DD + ranking gates correctly explain every excluded lane.

This verifies the adversarial audit's preliminary finding that the 32-lane "gap" is ALLOCATOR WORKING AS DESIGNED — not unfair exclusion.

## Live-6 internal rho summary

| live_lane | count_paired | min_rho | median_rho | max_rho |
|---|---:|---:|---:|---:|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | 5 | -0.017 | +0.001 | +0.042 |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 5 | -0.029 | -0.017 | -0.006 |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | 5 | -0.024 | +0.027 | +0.050 |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 5 | -0.025 | +0.011 | +0.040 |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | 5 | -0.060 | -0.025 | +0.042 |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 5 | -0.060 | +0.001 | +0.050 |

## Caveats (acknowledged limitations)

- Rho computed on FILTERED DAILY PNL overlap days per allocator's `_load_lane_daily_pnl` — reflects signal-correlation, not account-equity correlation. A rho<0.70 on signal days does not necessarily mean rho<0.70 on daily account P&L.
- Fisher z-transformation assumes Gaussian residuals. Fat-tailed trade returns make p-values conservative-biased.
- Bootstrap CI uses i.i.d. resample — adequate for rho (not serially autocorrelated at daily level).
- BH-FDR at q=0.05 applied to rho claims. Cites Benjamini-Hochberg 1995 + Harvey-Liu 2015 § multi-testing.
- rho<0.70 is the **allocator's** threshold. Markowitz 1952 would argue rho<0.30 for material diversification. This is a design-choice caveat, not an audit failure.
- Trailing-12mo window consumed by the allocator rebalance INCLUDES Mode A sacred-window data (post-2026-01-01). This audit READS the same data the allocator used — zero NEW OOS consumption.

## Outstanding questions (for follow-up)

1. L6 swap audit (VWAP_MID_ALIGNED → ORB_G5_O15 on 2026-04-18): this audit does NOT address it. Separate cycle.
2. Pre-2024 vs post-2024 rho stability: deferred to A2b (portfolio-optimization quality).
3. Ledoit-Wolf shrinkage covariance as alternative to pairwise Pearson: deferred to A2b post-literature-expansion.

## Supplementary artifact

- Full rho matrix (upper triangle, all eligible pairs): `2026-04-18-allocator-rho-matrix.csv`

## Commit trail

- Roadmap: `docs/plans/2026-04-18-multi-phase-audit-roadmap.md`
- Adversarial audit parent: `docs/audit/results/2026-04-18-portfolio-audit-adversarial-reopen.md`
