# ONC N_eff Workstream — Plan (no execution)

**Date:** 2026-04-21
**Status:** SCHEDULED — execution deferred to v3 Phase G (research resumption)
**Authored by:** Claude terminal (6lane-baseline worktree)

---

## Citation

Lopez de Prado 2020, *Machine Learning for Asset Managers*, Ch 4 — Optimal Number of Clusters (ONC) algorithm. Local extract: `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md`. Pointer to Bailey-López de Prado 2014 Appendix A.3 Eq. 9 in `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` for the downstream N̂ substitution.

## Scope

ρ̂ estimation via ONC clustering for the deployed lane universe — delivers N̂ per Bailey-LdP Appendix A.3 Eq. 9, replacing Amendment 2.1's cross-check-only DSR status in `docs/institutional/pre_registered_criteria.md`. Output N_eff per hypothesis family becomes the canonical institutional value; DSR graduates from cross-check to binding gate at that resolved N_eff.

## Gate

Executes in v3 Phase G (research resumption), AFTER Terminal 2 Phase B rollup at origin AND AFTER Terminal 1 PR #48 re-audit concludes. Not during freeze. Freeze binding until both upstream conditions are met.

## Expected output

ONC-derived N_eff per hypothesis family, documented alongside the Bailey A.3 pairwise-correlation N_eff (computed in DSR audit v2, commit `4e545950`, reported ~99 on PR #51's 105-cell family). The conservative of {ONC, pairwise} becomes canonical. Retroactively sharpens every Phase B verdict issued under bracketed DSR — Terminal 2 Phase B KEEP/DEGRADE verdicts issued with ρ̂ ∈ {0.3, 0.5, 0.7} bracketing get re-evaluated against the concrete N_eff when this workstream lands.

## Blast radius

- New module (likely `research/onc_n_eff.py` or `pipeline/onc_neff.py`) implementing ONC per LdP 2020 Ch 4
- New drift check entry in `pipeline/check_drift.py` verifying N_eff freshness against feature-fire correlation matrix drift
- Update to `trading_app/strategy_validator.py:583-589` and `:1400-1453` DSR call sites (referenced in `pre_registered_criteria.md` Amendment 2.1) to consume the resolved N_eff as binding gate
- Amendment to `pre_registered_criteria.md` removing Amendment 2.1's gating condition once N_eff is institutionally established
- Zero freeze-period changes — all work deferred to v3 Phase G

## Non-actions (by this plan)

- Does NOT implement ONC — plan only
- Does NOT run any clustering now
- Does NOT modify DSR call sites now
- Does NOT amend `pre_registered_criteria.md` now
- Does NOT affect any Phase B verdict Terminal 2 issues in the interim — see `docs/handoff/2026-04-21-terminal-2-phase-b-unblock.md` (D5b) for the bracketed-DSR unblock path
