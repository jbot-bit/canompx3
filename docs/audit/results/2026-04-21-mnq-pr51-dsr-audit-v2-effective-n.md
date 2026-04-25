# PR #51 DSR audit — v2 Bailey Exhibit 4 effective-N correction

**Authority:** Bailey-López de Prado 2014 Appendix A.3 + Exhibit 4 (Eq. 9: `N̂ = ρ̂ + (1 − ρ̂)·M`). See `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`.

**Phase 0 C5 threshold:** DSR >= 0.95.

**Follow-on to v1 (`2026-04-21-mnq-pr51-dsr-audit-v1.md`).** V1 computed DSR assuming 105 independent trials — upper-bound-conservative. This v2 applies the Bailey Eq. 9 correction for correlated trials to produce the honest effective-N DSR.

## Family + correlation

- Raw trials M: **105**
- Pairwise Pearson correlation (off-diagonal mean, min 30d overlap): **ρ̂ = +0.0578**
- Effective independent trials: **N̂ = ρ̂ + (1 − ρ̂)·M = 98.99**
- Family V[SR] (ddof=1): `0.00414`
- SR_0 at M=105 (raw): `+0.16396` (v1 value)
- **SR_0 at N̂ (corrected): `+0.16263`**

## Per-cell DSR — raw M vs corrected N̂

| Apt | RR | Session | Trade SR | DSR (M=105) | DSR (N̂) | Phase 0 C5 @ N̂ |
|---:|---:|---|---:|---:|---:|---|
| 5 | 1.0 | NYSE_OPEN | +0.08441 | 0.0006 | 0.0007 | **FAIL** |
| 5 | 1.5 | NYSE_OPEN | +0.07944 | 0.0003 | 0.0003 | **FAIL** |
| 15 | 1.0 | NYSE_OPEN | +0.10068 | 0.0070 | 0.0081 | **FAIL** |
| 15 | 1.0 | US_DATA_1000 | +0.10112 | 0.0067 | 0.0077 | **FAIL** |
| 15 | 1.5 | US_DATA_1000 | +0.08850 | 0.0016 | 0.0019 | **FAIL** |

## Summary

- CANDIDATEs tested: 5
- DSR PASS at N̂ (>= 0.95): **0**
- DSR FAIL at N̂: 5

## Interpretation

None of the 5 PR #51 cells pass Phase 0 C5 even after the Exhibit 4 effective-N correction (ρ̂ = +0.0578, N̂ = 98.99). The v1 finding hardens: these cells fail DSR under both raw-M and corrected-N̂ framings. Shadow-deployment remains institutionally blocked under Pathway A. A Pathway B K=1 pre-reg (theory-driven, single-cell) per pre_registered_criteria.md Amendment 3.0 is the remaining legitimate path.

## Methodology notes

- Pairwise correlation computed on per-day MEAN pnl_r per cell (aggregates multiple trades per day into one representative value). This maps each cell to a daily series so correlations are well-defined across cells that fire at different intra-day counts.
- `min_periods=30` on pairwise correlation to avoid spurious ρ̂ contributions from sparsely-overlapping cell pairs.
- ρ̂ is the MEAN of the off-diagonal upper-triangle finite entries of the correlation matrix, matching Bailey's 'average correlation between the trials'.
- Bailey Eq. 9 is an interpolation between ρ̂ → 0 (full independence, N̂ = M) and ρ̂ → 1 (full dependence, N̂ = 1). Linear in ρ̂.

## Not done by this result

- No writes to validated_setups / edge_families / lane_allocation / live_config.
- No capital action.
- Does NOT compute C11 (Monte Carlo account death) or C12 (live Shiryaev-Roberts).