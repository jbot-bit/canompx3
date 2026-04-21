# G3 — DSR + N̂ Certificate

**Candidate:** ________________________
**Hypothesis family:** ________________________
**Pre-reg:** `docs/audit/hypotheses/________.yaml`

---

## Purpose

Per Bailey-López de Prado 2014 (`docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`):
- Eq. 2: Deflated Sharpe Ratio formula using observed moments + family selection bias
- Appendix A.3 + Eq. 9: effective-N correction for correlated trials

Currently **cross-check only** per Amendment 2.1 of `pre_registered_criteria.md` until N̂ is formally resolved in-repo via ONC (Lopez de Prado 2020 Ch 4).

## Canonical DSR formula (Eq. 2)

```
DSR ≡ Φ( (ŜR - ŜR_0) · √(T-1) / √(1 - γ̂₃·ŜR + (γ̂₄-1)/4 · ŜR²) )
ŜR_0 = √V[{ŜR_n}] · ((1-γ)·Φ⁻¹[1 - 1/N̂] + γ·Φ⁻¹[1 - 1/(N̂·e)])
γ ≈ 0.5772156649 (Euler-Mascheroni)
```

Implementation must validate against Bailey's paper pp.9-10 worked example:
- Input: SR_ann=2.5, T=1250, V=0.5, skew=-3, kurt=10, N=100
- Expected: DSR = 0.9004

## Live computation

| Input | Value | Source / query |
|---|---|---|
| `T` (trade count) | | `SELECT COUNT(*) FROM orb_outcomes WHERE ...` |
| `ŜR` (candidate Sharpe) | | per-trade SR from same query |
| `V[{ŜR_n}]` (variance of SR across hypothesis family) | | pre-reg K cells' SRs |
| `γ̂₃` (skewness) | | `SELECT SKEWNESS(pnl_r)` |
| `γ̂₄` (kurtosis) | | `SELECT KURTOSIS(pnl_r)` (Pearson, unbiased) |
| `N̂` (effective-N) | | see Eq. 9 computation below |
| `DSR` | | result |

## N̂ computation (Eq. 9)

**Preferred (per Amendment 2.1 prerequisite): ONC clustering** per LdP 2020 Ch 4.
- Input: K×K correlation matrix across hypothesis family fire-patterns
- Output: optimal cluster count c, then N̂ = c

**Fallback if ONC not implemented: Bailey A.3 pairwise average**
- `N̂ = ρ̂ + (1 - ρ̂) · M` where M is raw K, ρ̂ is mean pairwise Pearson correlation

Evidence:
```
$ python <ONC or pairwise script>
<paste output: M, ρ̂ or cluster count, N̂>
```

## Sanity check (BEFORE trusting DSR)

- [ ] DSR implementation validated against Bailey paper pp.9-10 worked example (DSR = 0.9004)
- [ ] Evidence: `$ python -c "from <module> import dsr_bailey; print(dsr_bailey(...))"` output attached

## Verdict

- [ ] BINDING PASS — DSR ≥ 0.95 at resolved N̂. Candidate clears C5 for deploy-gate.
- [ ] BINDING FAIL — DSR < 0.95 at resolved N̂. Per Amendment 2.1 current state, this is CROSS-CHECK ONLY — does NOT block advancement unless N_eff has been formally resolved institutionally.
- [ ] CROSS-CHECK LOGGED — DSR computed and logged; per Amendment 2.1 does not act as hard gate.

**Current status:** until ONC workstream is complete (resolves Amendment 2.1's gating condition), DSR is CROSS-CHECK only regardless of value. Log the value; don't let it kill a candidate whose H1/C6/C8/C9 gates passed.

## Failure disposition

BINDING FAIL post-N_eff-resolution → candidate is research-grade only, cannot deploy.
CROSS-CHECK FAIL (current Amendment 2.1 regime) → log, do not block.

## Literature citation

- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` Eq. 2 + Appendix A.3 Eq. 9
- `docs/institutional/pre_registered_criteria.md` Amendment 2.1 (DSR downgraded to cross-check)
- `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md` Ch 4 (ONC)

## Authored by / committed

- Author: ____________________________
- Commit SHA of candidate's script: ________________
- Commit SHA of DSR helper: ________________
- Pinned `pre_registered_criteria.md` commit SHA at eval time: ________________
