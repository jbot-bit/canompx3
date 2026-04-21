# MNQ PR #51 5 CANDIDATE_READY cells — Deflated Sharpe Ratio audit v1

**Authority:** Bailey-López de Prado 2014 Eq. 2 (`docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`).

**Phase 0 C5 threshold:** DSR >= 0.95.

**Confirmatory audit** (no new discovery, no pre-reg required per research-truth-protocol.md § 10).

## Family context

- Family (re-derived from PR #51 axes): **K = 105** cells with N_IS >= 100
- Family mean trade SR: `+0.01858`
- Family variance V[SR]: `0.00414`
- Bailey SR_0 rejection threshold (trade-level): `+0.16396`

## DSR per CANDIDATE_READY cell

| Apt | RR | Session | N_IS | Trade SR | Skew (γ₃) | Kurt (γ₄ Pearson) | DSR | Phase 0 C5 |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| 5 | 1.0 | NYSE_OPEN | 1693 | +0.08441 | -0.242 | 1.062 | 0.0006 | **FAIL** |
| 5 | 1.5 | NYSE_OPEN | 1650 | +0.07944 | +0.184 | 1.036 | 0.0003 | **FAIL** |
| 15 | 1.0 | NYSE_OPEN | 1545 | +0.10068 | -0.253 | 1.064 | 0.0070 | **FAIL** |
| 15 | 1.0 | US_DATA_1000 | 1594 | +0.10112 | -0.272 | 1.079 | 0.0067 | **FAIL** |
| 15 | 1.5 | US_DATA_1000 | 1495 | +0.08850 | +0.168 | 1.031 | 0.0016 | **FAIL** |

## Summary

- CANDIDATEs tested: 5
- DSR PASS (>= 0.95): **0**
- DSR FAIL (< 0.95): 5

## Interpretation

NONE of the 5 PR #51 CANDIDATE_READY cells pass Phase 0 C5 (DSR < 0.95). These cells passed H1/C6/C8/C9 but fail C5 — Bailey-LdP 2014 DSR corrects for family selection bias + non-normality. Re-classify as RESEARCH_SURVIVOR pending C5. Deployment paused.

## Methodology notes

- Trade-level SR (not annualized). Bailey Eq. 2 is scale-invariant — SR and SR_0 in the same units cancel, so trade-level vs daily-level does not change DSR as long as both sides use the same convention.
- Skewness via `scipy.stats.skew(bias=False)` — the bias-adjusted Fisher-Pearson estimator, matches Bailey γ̂₃.
- Kurtosis via `scipy.stats.kurtosis(fisher=False, bias=False)` — Pearson (non-excess) kurtosis, matches Bailey γ̂₄.
- Family V[SR] computed with ddof=1 (bias-corrected sample variance).
- Independence assumption: treats the 105 cells as independent trials. This is conservative — per Bailey Exhibit 4, correlated trials yield a smaller effective N, which LOWERS SR_0, which RAISES DSR. If DSR passes at N=105 independent, it passes at any smaller effective N. If DSR fails at N=105, an independence correction would not rescue it (effective N would still be <= 105).
- Sanity check: Bailey-LdP 2014 worked example (pp 9-10) reproduced in `_sanity_bailey_worked_example()`.

## Not done by this result

- No writes to validated_setups / edge_families / lane_allocation / live_config.
- No deployment or capital action.
- Does NOT compute C11 (90-day account-death Monte Carlo) or C12 (live Shiryaev-Roberts).
- Does NOT re-test MES/MGC (PR #53 + PR #55 canonical for those).

## Canonical run output

See terminal output on commit; all per-cell stats logged.