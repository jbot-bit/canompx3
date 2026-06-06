"""Canonical DSR (Deflated Sharpe Ratio) verdict policy — single source of truth.

Extracted from ``pipeline/check_drift.py`` (2026-06-04) so the DSR drift-cache
entry can declare a PRECISE dependency on the verdict-bearing constant. Before
this module existed, ``check_drift.py``'s DSR cache entry hashed only the
``docs/audit/hypotheses/*.yaml`` tree; the verdict ALSO depends on the allowed-
derivation set below, which lived in ``check_drift.py`` itself and was therefore
absent from the cache key. A commit that tightened the set computed the SAME key,
read the OLD cached PASS, and the blocking Bailey–López de Prado DSR/multiplicity
gate reported green on stale logic (Codex high-risk finding 2026-06-04, proven by
execution). Deping the cache entry on THIS module binds the key to the policy
without forcing a cold DSR run on every unrelated ``check_drift.py`` edit
(12.5%-commit cold-run tax of the rejected whole-file dep). Mirrors the
``trading_app/chordia.py`` T-threshold dep precedent (institutional-rigor §4 —
delegate to a canonical source, never re-encode).

Authority: ``docs/institutional/pre_registered_criteria.md`` Amendment 3.5
(2026-05-29, binding) — the DSR reference-universe lock.
"""

# Allowed values for criterion_5.effective_trials_derivation in any prereg
# claiming DSR-clearance.
# declared_K_conservative — N̂ = the prereg's declared K (Amendment 3.5 default,
#   strict: larger K -> higher SR_0 -> stricter gate).
# onc_clustered — N̂ from trading_app.dsr.estimate_n_eff_onc (López de Prado
#   Optimal Number of Clusters). Permitted now that the canonical helper exists.
ALLOWED_DSR_TRIALS_DERIVATION = {"declared_K_conservative", "onc_clustered"}
