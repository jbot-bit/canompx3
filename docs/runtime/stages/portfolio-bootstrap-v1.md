---
slug: portfolio-bootstrap-v1
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Portfolio-level block-bootstrap on IS — probabilistic OOS estimate for 6-lane deployed book
---

# Stage: Portfolio-level block-bootstrap (v1)

## Task

Generate probabilistic OOS evidence for the current 6-lane `topstep_50k_mnq_auto`
portfolio from IS (pre-2026-01-01) data, as the replacement for the CPCV path
killed by H3 on 2026-04-21. Measured per-lane IS pnl_r lag-1 autocorrelation is
ρ ∈ [−0.03, +0.03] (effectively zero), so plain bootstrap on daily portfolio R
is statistically valid — the CPCV-with-embargo machinery was solving a
contamination that empirically does not exist in this data.

## Scope Lock

- docs/audit/hypotheses/2026-04-21-portfolio-bootstrap-v1.yaml
- research/portfolio_bootstrap_v1.py
- docs/audit/results/2026-04-21-portfolio-bootstrap-v1.md

blast_radius: new research script research/portfolio_bootstrap_v1.py (one-off, no production callers); new pre-reg YAML at docs/audit/hypotheses/2026-04-21-portfolio-bootstrap-v1.yaml per research-truth-protocol Phase 0; new results MD at docs/audit/results/2026-04-21-portfolio-bootstrap-v1.md. Read-only on orb_outcomes + paper_trades + daily_features (canonical layers). No schema changes. No allocator changes. No validator changes. No changes to prop_profiles.py or lane_allocation.json. Parent authority: pre_registered_criteria.md Amendment 3.2 + research-truth-protocol.md Phase 0 + backtesting-methodology.md RULE 3.2 (tiered OOS).

## Acceptance criteria

1. Pre-reg YAML written with numeric thresholds, declared fixed seeds, pre-committed N_bootstrap, pre-committed window length matched to 2026 OOS depth, and a kill criterion for IS selection bias.
2. `research/portfolio_bootstrap_v1.py` runs deterministically (fixed seed), reads only canonical layers, computes:
   - IS daily portfolio R series (2020-2025) using only the 6 lanes in `docs/runtime/lane_allocation.json`
   - Bootstrap distribution of ExpR + Sharpe over rolling windows matched to 2026-OOS length
   - Percentile rank of observed 2026 shadow performance within the IS bootstrap distribution
3. Results MD reports: IS baseline stats, bootstrap CI of ExpR and Sharpe, 2026 shadow percentile rank, interpretation under Amendment 3.2 tiered framework.
4. All tests pass; drift check passes; no new production callers.

## Non-goals

- Not changing the holdout boundary.
- Not using 2026 data for selection or tuning — 2026 is the observed point compared against the bootstrap null.
- Not proposing a sizing overlay this stage (follow-on).
- Not running per-lane bootstrap (per-lane N too small for reliable bootstrap at Tier 1 spec; portfolio-level is the unit of analysis).
- Not promoting any strategy.
