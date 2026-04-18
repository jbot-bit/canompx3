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
| C3 | MNQ | COMEX_SETTLE | short | 1.0 | 298 | 299 | ✓ |
| C4 | MGC | LONDON_METALS | short | 1.0 | 148 | 149 | ✓ |
| C5 | MNQ | SINGAPORE_OPEN | short | 1.0 | 286 | 285 | ✓ |

Match ✓ means the script's fire-days count is within ±2 of the audited scan's `N_on`. Exact match is not expected — the scan counts trades (rows in `orb_outcomes` with non-null `pnl_r`) while this script reports unique trading_days, which can differ by 0-2 when an outcome row is missing.

## Pairwise overlap matrix

| Pair | A | B | \|A\| | \|B\| | \|A ∩ B\| | \|A ∪ B\| | Jaccard | % of min |
|---|---|---|---:|---:|---:|---:|---:|---:|
| C1 x C2 | MES TOKYO_OPEN long | MES COMEX_SETTLE short | 276 | 288 | 49 | 515 | 0.095 | 17.8 |
| C1 x C3 | MES TOKYO_OPEN long | MNQ COMEX_SETTLE short | 276 | 299 | 57 | 518 | 0.110 | 20.7 |
| C1 x C4 | MES TOKYO_OPEN long | MGC LONDON_METALS short | 276 | 149 | 28 | 397 | 0.071 | 18.8 |
| C1 x C5 | MES TOKYO_OPEN long | MNQ SINGAPORE_OPEN short | 276 | 285 | 42 | 519 | 0.081 | 15.2 |
| C2 x C3 | MES COMEX_SETTLE short | MNQ COMEX_SETTLE short | 288 | 299 | 193 | 394 | 0.490 | 67.0 |
| C2 x C4 | MES COMEX_SETTLE short | MGC LONDON_METALS short | 288 | 149 | 26 | 411 | 0.063 | 17.4 |
| C2 x C5 | MES COMEX_SETTLE short | MNQ SINGAPORE_OPEN short | 288 | 285 | 59 | 514 | 0.115 | 20.7 |
| C3 x C4 | MNQ COMEX_SETTLE short | MGC LONDON_METALS short | 299 | 149 | 24 | 424 | 0.057 | 16.1 |
| C3 x C5 | MNQ COMEX_SETTLE short | MNQ SINGAPORE_OPEN short | 299 | 285 | 65 | 519 | 0.125 | 22.8 |
| C4 x C5 | MGC LONDON_METALS short | MNQ SINGAPORE_OPEN short | 149 | 285 | 26 | 408 | 0.064 | 17.4 |

**Interpretation:**

- Jaccard > 0.30 ≈ two cells share ≥ 30% of their fire days → likely not independent hypotheses on the same underlying driver.
- Jaccard < 0.10 with overlap-of-min < 20% ≈ weak coupling → approximately independent.

## Simultaneous-fire distribution

| # cells firing | Trading-days with this count |
|---:|---:|
| 1 | 515 |
| 2 | 244 |
| 3 | 79 |
| 4 | 13 |
| 5 | 1 |
| **total fire-days** | **852** |

Sum of per-cell fire-day counts: 1297. Union of fire-days: 852. Redundancy = 1 − union/sum = 0.343. Redundancy 0 ≈ fully disjoint fire sets; 1 ≈ fully identical fire sets across all 5 cells.

## Pairwise Pearson correlation of fire indicators

Correlation computed on the union-trading_day grid (rows that are not in a given cell's fire-set contribute 0). This is conservative — cross-instrument cells may have structurally disjoint eligible-day sets (e.g., MGC LONDON_METALS fires on MGC days; MNQ SINGAPORE_OPEN fires on MNQ days) and the correlation captures both co-firing and joint-absence.

| | C1 | C2 | C3 | C4 | C5 |
|---|---:|---:|---:|---:|---:|
| C1 | 1.000 | -0.235 | -0.209 | -0.134 | -0.268 |
| C2 | -0.235 | 1.000 | 0.478 | -0.159 | -0.196 |
| C3 | -0.209 | 0.478 | 1.000 | -0.183 | -0.183 |
| C4 | -0.134 | -0.159 | -0.183 | 1.000 | -0.156 |
| C5 | -0.268 | -0.196 | -0.183 | -0.156 | 1.000 |

## Nyholt 2004 M-effective

**Meff = 4.817** (out of m=5 cells).

Eigenvalues of the correlation matrix: [0.408 0.522 1.111 1.266 1.694].

Meff heuristic: Meff ≈ m (all 5 cells here ≈ 5) means the 5 lanes are approximately independent draws on the alignment grid and K_effective for the rel_vol_HIGH_Q3 family is ≈ 5, not ≈ 1. Meff ≈ 1 means a single effective test. Values in-between interpolate.

## Verdict heuristic

- If Meff ≥ m − 0.5 AND max pairwise Jaccard < 0.20 → the '5 independent survivors' framing is supported; DSR / BH-FDR verdicts computed with K ≈ 5 are honest.
- If Meff is materially below m (e.g., < 0.7·m) OR max Jaccard ≥ 0.30 → the framing is overstated; effective K collapses toward Meff and downstream DSR/BH claims need re-computation with the reduced K.

Observed: Meff = 4.817 / m = 5; max pairwise Jaccard = 0.490.

**Heuristic verdict:** the '5 independent BH-global survivors' framing is NOT fully supported. Effective K is below 5. Downstream DSR / BH-FDR reports that used K=5 on the family survivors should be re-reviewed with K_effective ≈ 4.8 and the max-Jaccard pair flagged for possible co-firing driver.

## Reproduction

```
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python research/rel_vol_cross_scan_overlap_decomposition.py
```

No randomness. Fire masks replicated from `research.comprehensive_deployed_lane_scan.bucket_high`. No writes to `validated_setups` / `experimental_strategies`.

