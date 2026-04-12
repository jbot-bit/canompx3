# Finite-Data ORB Re-Audit — Clean Worktree

**Date:** 2026-04-07
**Branch:** `wt-codex-finite-data-reaudit2`
**Purpose:** Re-audit the April 7 finite-data / institutional framework from a clean worktree so the result can be reviewed by Claude without the main repo's local noise.

---

## Scope and method

This re-audit is grounded in:
- canonical repo rules (`RESEARCH_RULES.md`)
- current code paths (`strategy_discovery.py`, `strategy_validator.py`, `config.py`, `execution_engine.py`, `check_drift.py`)
- current DB state in `gold.db`
- local literature extracts present in this worktree

This memo does **not** assume stale docs are true. Where docs, code, and DB disagree, code + canonical-layer facts win.

---

## Executive conclusion

The finite-data direction is correct, but the current institutional framework still overstates what is proven.

Three things are true at the same time:

1. A literal "20-year rule" is not the right standard for newly launched micro futures.
2. The real failure mode was excessive search relative to clean independent history.
3. The repo is **not yet entitled** to treat its new locked criteria as fully settled institutional law.

The unresolved issues are not cosmetic:
- holdout policy is internally inconsistent
- DSR is treated as binding in docs but informational in code
- one cited literature file is missing in this clean worktree
- operational overlays are still too easy to blur into discovery evidence

The correct stance is:

**Current ORB lanes may be operationally usable, but they remain research-provisional until the holdout state, `N_eff`, and overlay-vs-discovery boundary are made explicit and enforced.**

---

## Re-audit findings

### 1. No-lookahead mechanics are mostly in place

The current codebase does support a no-lookahead path for clean ORB research:

- `strategy_discovery.py` restricts outcomes to `trading_day < holdout_date`
- the same code caps feature windows at `holdout_date` to avoid leakage
- `config.py` explicitly excludes contaminated E2 filter families (`VOL_RV*`, `ATR70_VOL`, `_CONT`, `_FAST`)
- `execution_engine.py` keeps E2 timeout logic as an execution overlay rather than pretending it is a pre-break feature

That part of the methodology is real.

### 2. Parent/full-size proxy use is defensible for price-based filters

The repo explicitly extends:
- `MGC` through `GC` price history
- `MNQ` through `NQ` backfill

That supports using proxy history for price-based filters such as:
- `COST_LT10`
- `COST_LT12`
- `OVNRNG_*`
- `ORB_G*`

This does **not** solve the search-bias problem. It only answers whether the historical price series is mechanically comparable enough for those filters.

### 3. Holdout governance is still contradictory

The repo currently says two incompatible things:

- `RESEARCH_RULES.md` says 2026 holdout is sacred and must not influence selection.
- `docs/pre-registrations/2026-03-20-mnq-rr1-verified-sessions.md` says the holdout test completed and 2026 data is now included in discovery.

`pipeline/check_drift.py` currently resolves this contradiction badly:
- the contamination checker still exists
- but its declaration map is empty
- so the guard passes without actively checking any instrument

That means the current "holdout clean" story is not enforceable from the repo as checked out in this worktree.

### 4. DSR is still unresolved as a hard gate

The biggest policy gap is DSR.

The current code says:
- DSR is informational, not a gate
- reason: `N_eff` is uncertain
- prior adversarial review reached the same conclusion

But the new institutional docs say:
- `DSR > 0.95` is required for validation and deployment eligibility

Those two positions are incompatible.

Current DB state reinforces the problem:
- `validated_setups = 124`
- `edge_families = 61`
- `0 / 124` setups have `dsr_score > 0.95`
- max `dsr_score = 0.1198`

So the repo currently has:
- a documented hard threshold
- no validated strategies satisfying it
- code that still treats it as informational

That is not a resolved framework. It is an unresolved governance fork.

### 5. One cited institutional source is missing in this clean worktree

`docs/institutional/finite_data_framework.md` cites:
- `literature/lopez_de_prado_2020_ml_for_asset_managers.md`

That file is **not present** in this clean worktree.

The same document still says the extract is "pending."

Therefore the framework's theory-first section is not fully grounded locally in this branch. It may exist in another dirty workspace state, but not in the clean audit branch.

### 6. Some institutional claims are still stronger than the repo can justify

The current docs import several ideas correctly, but state them too strongly for this repo's present evidence:

- Chordia's `t >= 3.79` threshold is useful as a severity benchmark, but not yet fully justified as a universal hard gate for this ORB family structure without a stricter family definition.
- DSR is valuable as a warning light, but not yet as a lawful deploy / don't deploy switch.
- 2026 OOS criteria are only clean if the project commits back to a true holdout-clean state.

### 7. Trade count is not the core weakness for the current lane set

Current lane samples are not tiny in raw trade count:
- 591
- 766
- 849
- 978
- 1441
- 1593
- 1941

So the primary weakness is **not** "too few trades to say anything."

The primary weakness is:
- too many related hypotheses
- unclear effective independence
- inconsistent holdout policy
- too much narrative slippage between discovered filters and live overlays

---

## Improved plan

### Phase 1 — Freeze one evidence regime

Choose one and make it explicit:

1. **Holdout-clean regime**
   - discovery data ends 2025-12-31
   - 2026 not used for choosing sessions / thresholds / filters / RR
   - 2026 only used for later OOS reporting

2. **Post-holdout-monitoring regime**
   - 2026 has already been consumed
   - stop calling it clean OOS
   - all forward evidence from now on is paper/live forward evidence

Do not mix them.

### Phase 2 — Keep the next audit family narrow

The next honest finite-data audit should restrict itself to:
- E2 only
- price-based filters only
- fixed ORB aperture
- fixed pre-registered RR set
- session list declared in advance

Explicitly exclude:
- `VOL_RV*`
- `ATR70_VOL`
- `_FAST`
- `_CONT`
- composite or overlay logic not encoded in the discovery ID

This is the correct fix for "too many independent trials," not another full brute-force rerun.

### Phase 3 — Use honest K before discovery, not after

Before running discovery:
- define the hypothesis family
- define the exact sessions, filters, thresholds, RR targets, and entry model
- define the exact trial count
- define the kill criteria

Then run BH FDR on that pre-registered family.

Do not use global K when convenient to kill results and local K when convenient to rescue them.

### Phase 4 — Treat DSR as an analytical cross-check until `N_eff` is solved in-repo

Do not pretend this is already settled.

The repo should continue using DSR as:
- an analytical stress test
- a diagnostic on search fragility
- a documentation field in `validated_setups`

It should **not** yet be the sole eligibility gate unless:
- the `N_eff` estimator is formalized
- the implementation is verified
- the literature grounding is complete in the clean branch

### Phase 5 — Audit overlays separately from discovered filters

For each deployed lane, report two things separately:

1. **Discovery evidence**
   - the filter encoded in `strategy_id`
   - search family
   - holdout state
   - OOS / WFE / era evidence

2. **Operational overlays**
   - calendar skip / half-size
   - ATR velocity skip
   - E2 timeout
   - market-state gating

Otherwise the repo will keep narratively upgrading the evidence without actually upgrading the discovery proof.

### Phase 6 — Reclassify current lanes honestly

Current lanes should be described as:
- operationally deployable
- provisionally supported
- not production-grade proof

That wording matches the actual state of evidence better than "institutionally validated."

---

## Concrete gaps to close next

1. Decide and document one 2026 policy.
2. Make `check_holdout_contamination()` enforce that policy instead of checking nothing.
3. Complete or remove the missing LdP 2020 literature reference from the clean branch.
4. Add a repo-visible note that DSR remains informational until `N_eff` is formally solved.
5. Write one narrow pre-registered E2 price-filter hypothesis file and audit only that family.

---

## What Claude should audit next

Claude should review this branch for:
- whether the holdout contradiction is framed correctly
- whether the DSR policy conflict is described fairly
- whether the improved plan is narrow enough to avoid fresh data-snooping
- whether the separation between discovery evidence and execution overlays is stated clearly enough to govern future work

This branch is intentionally limited to audit artifacts so that review can focus on reasoning rather than runtime changes.
