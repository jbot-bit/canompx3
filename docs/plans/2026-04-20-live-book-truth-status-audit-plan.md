# Live Book Truth-Status Audit Plan

**Date:** 2026-04-20  
**Status:** LOCKED  
**Companion pre-reg:** [2026-04-20-live-book-truth-status-audit.yaml](/mnt/c/Users/joshd/canompx3/docs/audit/hypotheses/2026-04-20-live-book-truth-status-audit.yaml)  
**Mode:** audit-first, fail-closed  
**Scope:** audit the current diagnosis of the live prop book before any rollout, scale, or fix recommendation

## Non-Negotiable Procedure

1. Pre-flight on canonical layers only.
2. Decompose claims exactly.
3. Verify each claim from the proof hierarchy.
4. Classify each claim `VALID / ALIVE / DEAD / MISCLASSIFIED`.
5. Only after classification, produce a fix queue.

No implementation or capital-allocation recommendation is allowed before step 4.

## Proof Hierarchy

Trusted, in order:
1. Current code / current files
2. Direct command output
3. Canonical tables: `orb_outcomes`, `daily_features`, `bars_1m`, `paper_trades`
4. Verbatim local resource text
5. Docs only for intent / declared policy, never as proof when direct proof exists

Explicitly not trusted as evidence:
- Extract-file metadata and front matter
- Project-written “application to our project” sections in literature extracts
- Prior summaries
- Memory files
- Derived-layer narratives when canonical layers can answer directly

## Hard Claims Under Audit

| # | Claim | Type | Decision impact if TRUE | Decision impact if FALSE |
|---|---|---|---|---|
| 1 | The live book is the 6-lane MNQ allocator book on `topstep_50k_mnq_auto` | operational_fact | Defines current book | Entire diagnosis anchored to stale surface |
| 2 | The book is under-deployed, and scale is the highest-confidence lever | capital_allocation_claim | Supports `copies 2→5` as next move | Scale thesis blocked |
| 3 | Missing current-lane realized attribution is the biggest live-truth gap | operational_gap | Prioritizes attribution work | Another blocker is higher EV |
| 4 | `prop_portfolio.py` breakage and doctrine drift are real control-plane failures | operator_integrity_claim | Promotes control-plane repair | Reduces urgency of surface repair |
| 5 | Routine-day MNQ slippage is not the primary bottleneck | cost_realism_claim | De-prioritizes slippage as main leak | Slippage stays central blocker |
| 6 | Correlation / risk control is not ready for larger scale | risk_governance_claim | Blocks size increase until fixed | Scale less constrained than thought |
| 7 | Discovery is not the main issue | methodology_claim | Pushes attention away from rediscovery | Forces revalidation / rediscovery path |

## Pre-Flight Status Already Observed

Directly observed from canonical layers:
- `orb_outcomes` MNQ max trading day: `2026-04-16`
- `daily_features` MNQ max trading day: `2026-04-17`
- `bars_1m` MNQ max timestamp: `2026-04-17 09:59:00+10:00`
- Canonical schemas for `orb_outcomes` and `daily_features` present and queryable

This is sufficient to continue the audit. It is not permission to trust derived layers.

## Phase Plan

### Phase 1 — Operational Truth

Goal:
- prove the current live book, current lane set, attribution state, and operator-surface health

Required evidence:
- [prop_profiles.py](/mnt/c/Users/joshd/canompx3/trading_app/prop_profiles.py)
- [lane_allocation.json](/mnt/c/Users/joshd/canompx3/docs/runtime/lane_allocation.json)
- `paper_trades`
- direct compile/run check of [prop_portfolio.py](/mnt/c/Users/joshd/canompx3/trading_app/prop_portfolio.py)

Decision rules:
- if current-lane attribution is absent, claim 3 stays `VALID`
- if `prop_portfolio.py` fails, claim 4 stays `VALID`
- if current book description differs across code and docs, doctrine drift stays `VALID`

### Phase 2 — Evidence-Status / Methodology Truth

Goal:
- separate operational deployability from research proof

Required evidence:
- [RESEARCH_RULES.md](/mnt/c/Users/joshd/canompx3/RESEARCH_RULES.md)
- [pre_registered_criteria.md](/mnt/c/Users/joshd/canompx3/docs/institutional/pre_registered_criteria.md)
- direct `validated_setups` provenance fields

Decision rules:
- if deployed families are still research-provisional under Mode A, anti-rediscovery framing cannot be `VALID`
- if diagnosis collapses provisional shelf truth into clean evidence, claim 2 is downgraded and claim 7 becomes `MISCLASSIFIED`

### Phase 3 — Cost / Risk / Execution Bias

Goal:
- determine whether slippage, control-plane, or execution blindness is the actual blocker

Required evidence:
- [debt-ledger.md](/mnt/c/Users/joshd/canompx3/docs/runtime/debt-ledger.md)
- direct current-lane `paper_trades`
- direct risk-control surface checks

Decision rules:
- if slippage coverage is incomplete, slippage claim can only be `ALIVE`
- if correlation controls are not populated / proven for the current scale path, scale thesis cannot be `VALID`

### Phase 4 — Post-Result Sanity Pass

Run the repo’s post-result sanity pass on the audit verdict itself:
- reality
- tunnel vision
- fresh-eyes check

This pass exists to prevent a technically true local finding from being mislabeled as the main bottleneck.

## Resource-Grounded Constraints

These come from verbatim resource text only:

- Bailey et al. (2013): hidden trial counts and overfit backtests mislead capital allocation.
- Bailey & López de Prado (2014): backtests without search-control disclosure are worthless regardless of performance.
- Harvey & Liu (2015): OOS can fail to be truly OOS, especially under trial-and-error revision.
- Lopez de Prado (2020): backtests cannot prove a strategy is a true positive; theory must lead.
- Pepelyshev & Polunchenko (2015): live surveillance is required for on-the-go break detection.

Operational translation:
- derived winners do not become proof just because they are current
- modeled book arithmetic does not become realized edge
- absent live attribution blocks confidence
- absent clean rediscovery blocks institutional-grade language

## Current Bottom Line Before Full Classification

What is already safe to say:
- some operational facts in the diagnosis are true
- the evidence-status language is too strong
- scale is not yet admissible as a high-confidence conclusion

What is not yet admissible:
- rollout
- closure
- promotion
- scale recommendation
- “discovery is not the move” as a blanket statement

## Fix Queue Policy

No fix enters the queue unless the corresponding claim survives audit.

If claims survive, fixes will be ordered as:
1. truth-surface repair
2. live attribution restoration
3. scale-ready gate definition
4. missing cost / risk closure
5. clean Mode A rediscovery of active families

If a claim does not survive, its proposed fix is dropped or reframed.
