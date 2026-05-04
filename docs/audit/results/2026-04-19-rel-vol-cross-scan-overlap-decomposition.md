# rel_vol_HIGH_Q3 cross-lane overlap decomposition

**Generated:** 2026-04-18T15:58:49+00:00
**Script:** `research/rel_vol_cross_scan_overlap_decomposition.py`
**IS window:** `trading_day < 2026-01-01` (Mode A, imported from `trading_app.holdout_policy.HOLDOUT_SACRED_FROM`).

## Audited claim

`docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md` § BH-FDR Survivors — Global reports 13 cells surviving at K_global=14,261. Of those, 5 distinct (instrument, session, direction) lanes use the `rel_vol_HIGH_Q3` feature. The memory file `comprehensive_scan_apr15.md` and other downstream notes call this '5 independent BH-global survivors' and treat rel_vol_HIGH_Q3 as a universal volume-confirmation signal.

**Adversarial question:** are the 5 lanes independent draws, or are their per-day fires correlated enough to collapse effective-K? The Nyholt 2004 M-effective statistic answers the 'effective number of independent tests' question using correlation-eigenvalue structure.

## Per-cell fire counts (IS)

| ID | Instrument | Session | Dir | RR | Scan N_on | This-run fire-days | Match |
|---|---|---|---|---:|---:|---:|---|
| C1 | MES | TOKYO_OPEN | long | 1.0 | 276 | 276 | ✓ |
| C2 | MES | COMEX_SETTLE | short | 1.0 | 288 | 288 | ✓ |
| C3 | MNQ | COMEX_SETTLE | short | 1.0 | 298 | 298 | ✓ |
| C4 | MGC | LONDON_METALS | short | 1.0 | 148 | 148 | ✓ |
| C5 | MNQ | SINGAPORE_OPEN | short | 1.0 | 286 | 286 | ✓ |

Match ✓ means the script's fire-days count is within ±2 of the audited scan's `N_on`. Exact match is not expected — the scan counts trades (rows in `orb_outcomes` with non-null `pnl_r`) while this script reports unique trading_days, which can differ by 0-2 when an outcome row is missing.

## Pairwise overlap matrix

| Pair | A | B | \|A\| | \|B\| | \|A ∩ B\| | \|A ∪ B\| | Jaccard | % of min |
|---|---|---|---:|---:|---:|---:|---:|---:|
| C1 x C2 | MES TOKYO_OPEN long | MES COMEX_SETTLE short | 276 | 288 | 49 | 515 | 0.095 | 17.8 |
| C1 x C3 | MES TOKYO_OPEN long | MNQ COMEX_SETTLE short | 276 | 298 | 57 | 517 | 0.110 | 20.7 |
| C1 x C4 | MES TOKYO_OPEN long | MGC LONDON_METALS short | 276 | 148 | 28 | 396 | 0.071 | 18.9 |
| C1 x C5 | MES TOKYO_OPEN long | MNQ SINGAPORE_OPEN short | 276 | 286 | 43 | 519 | 0.083 | 15.6 |
| C2 x C3 | MES COMEX_SETTLE short | MNQ COMEX_SETTLE short | 288 | 298 | 193 | 393 | 0.491 | 67.0 |
| C2 x C4 | MES COMEX_SETTLE short | MGC LONDON_METALS short | 288 | 148 | 26 | 410 | 0.063 | 17.6 |
| C2 x C5 | MES COMEX_SETTLE short | MNQ SINGAPORE_OPEN short | 288 | 286 | 59 | 515 | 0.115 | 20.6 |
| C3 x C4 | MNQ COMEX_SETTLE short | MGC LONDON_METALS short | 298 | 148 | 24 | 422 | 0.057 | 16.2 |
| C3 x C5 | MNQ COMEX_SETTLE short | MNQ SINGAPORE_OPEN short | 298 | 286 | 65 | 519 | 0.125 | 22.7 |
| C4 x C5 | MGC LONDON_METALS short | MNQ SINGAPORE_OPEN short | 148 | 286 | 26 | 408 | 0.064 | 17.6 |

**Interpretation:**

- Jaccard > 0.30 ≈ two cells share ≥ 30% of their fire days → likely not independent hypotheses on the same underlying driver.
- Jaccard < 0.10 with overlap-of-min < 20% ≈ weak coupling → approximately independent.

## Simultaneous-fire distribution

| # cells firing | Trading-days with this count |
|---:|---:|
| 1 | 512 |
| 2 | 245 |
| 3 | 79 |
| 4 | 13 |
| 5 | 1 |
| **total fire-days** | **850** |

Sum of per-cell fire-day counts: 1296. Union of fire-days: 850. Redundancy = 1 − union/sum = 0.344. Redundancy 0 ≈ fully disjoint fire sets; 1 ≈ fully identical fire sets across all 5 cells.

## Pairwise Pearson correlation of fire indicators

Correlation computed on the union-trading_day grid (rows that are not in a given cell's fire-set contribute 0). This is conservative — cross-instrument cells may have structurally disjoint eligible-day sets (e.g., MGC LONDON_METALS fires on MGC days; MNQ SINGAPORE_OPEN fires on MNQ days) and the correlation captures both co-firing and joint-absence.

| | C1 | C2 | C3 | C4 | C5 |
|---|---:|---:|---:|---:|---:|
| C1 | 1.000 | -0.236 | -0.209 | -0.133 | -0.265 |
| C2 | -0.236 | 1.000 | 0.479 | -0.158 | -0.199 |
| C3 | -0.209 | 0.479 | 1.000 | -0.181 | -0.184 |
| C4 | -0.133 | -0.158 | -0.181 | 1.000 | -0.156 |
| C5 | -0.265 | -0.199 | -0.184 | -0.156 | 1.000 |

## Nyholt 2004 M-effective

**Meff = 4.817** (out of m=5 cells).

Eigenvalues of the correlation matrix: [0.408 0.52  1.111 1.264 1.696].

Meff heuristic: Meff ≈ m (all 5 cells here ≈ 5) means the 5 lanes are approximately independent draws on the alignment grid and K_effective for the rel_vol_HIGH_Q3 family is ≈ 5, not ≈ 1. Meff ≈ 1 means a single effective test. Values in-between interpolate.

## Verdict heuristic

- If Meff ≥ m − 0.5 AND max pairwise Jaccard < 0.20 → the '5 independent survivors' framing is supported; DSR / BH-FDR verdicts computed with K ≈ 5 are honest.
- If Meff is materially below m (e.g., < 0.7·m) OR max Jaccard ≥ 0.30 → the framing is overstated; effective K collapses toward Meff and downstream DSR/BH claims need re-computation with the reduced K.

Observed: Meff = 4.817 / m = 5; max pairwise Jaccard = 0.491.

**Heuristic verdict:** the '5 independent BH-global survivors' framing is NOT fully supported. Effective K is below 5. Downstream DSR / BH-FDR reports that used K=5 on the family survivors should be re-reviewed with K_effective ≈ 4.8 and the max-Jaccard pair flagged for possible co-firing driver.

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/rel_vol_cross_scan_overlap_decomposition.py
```

No randomness. Fire masks replicated from `research.comprehensive_deployed_lane_scan.bucket_high`. No writes to `validated_setups` / `experimental_strategies`.

## Inherited methodology caveat

The fire-mask definition here uses `bucket_high(rel_vol, 67)` which computes the 67th percentile over the full cell distribution (IS + OOS combined) and then applies it to gate IS rows. That is a subtle look-ahead — IS fires depend on OOS distribution. This audit DELIBERATELY replicates the audited scan's convention so the decomposition compares like-for-like fire counts (our N_on matches the 2026-04-15 scan's N_on to within ±2 rows per cell). It does not fix the underlying look-ahead.

A sensitivity check was run on 2026-04-19 (`research/rel_vol_cross_scan_overlap_decomposition.py --quantile-method is_only`) with the 67th percentile computed IS-only. Result: Meff = 4.817, max Jaccard = 0.491 / 0.490 — identical to 3 decimals. Union fire-days 850 vs 852 (2-day difference). **The specific K_eff ≈ 4 finding is ROBUST under look-ahead correction** — the MES/MNQ COMEX_SETTLE short pair's 0.49 Jaccard is structural, not an upstream quantile artifact.

See companion document `docs/audit/results/2026-04-19-rel-vol-cross-scan-overlap-decomposition-is-only-quantile.md` for the IS-only numbers. See `.claude/rules/backtesting-methodology.md` § Historical failure log ("Quantile-over-full-sample as feature look-ahead") for the class-level rule.

