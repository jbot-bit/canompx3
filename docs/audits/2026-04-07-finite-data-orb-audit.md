# Finite-Data ORB Audit — Grounded Path Without Lookahead

**Date:** 2026-04-07
**Status:** COMPLETE
**Scope:** Audit the new finite-data / institutional framework against current repo code, canonical research rules, and local literature extracts. This memo is not a replacement for `docs/institutional/`; it is a grounding check on what is solid, what is still unresolved, and how to proceed without bias or lookahead.

---

## Executive verdict

The new finite-data framing is **partly correct** and **partly overconfident**.

What is correct:
- A literal "20-year rule" is the wrong frame for newly launched micro futures.
- Parent/full-size proxy data can be valid for **price-based** ORB filters.
- The real problem is excessive search relative to clean independent history, not merely "short sample bad."
- Current deployed lanes should be treated as **provisional operational candidates**, not production-grade proof of durable edge.

What is not yet justified:
- Treating the current framework as fully "institutional-grade" while `N_eff` remains unresolved.
- Treating DSR as a binding requirement in policy while the repo still documents DSR as informational because the effective number of trials is unknown.
- Treating 2026 as both sacred holdout and already reincorporated into discovery.
- Treating deployment overlays and discovery filters as one clean evidence object.

The correct next move is **not** "relax thresholds until finite data passes" and **not** "revert to impossible 20-year standards." The correct move is:

1. Freeze a no-lookahead research state.
2. Separate proxy validity from selection validity.
3. Restrict the audited universe to mechanically safe ORB families.
4. Use honest, pre-registered K on canonical layers.
5. Treat current lanes as provisional until the above is complete.

---

## Grounded findings

### 1. No-lookahead controls exist in the current pipeline

Discovery code supports true temporal holdout:
- `strategy_discovery.py` loads outcomes only before `holdout_date`.
- It also caps feature windows at `holdout_date` to avoid feature leakage.

Relevant code:
- `trading_app/strategy_discovery.py:996-998`
- `trading_app/strategy_discovery.py:1062-1065`
- `trading_app/strategy_discovery.py:1092-1097`
- `trading_app/strategy_discovery.py:1117-1125`

The filter layer also explicitly distinguishes safe vs contaminated E2 filters:
- `VOL_RV*`, `ATR70_VOL`, `_CONT`, and `_FAST` are excluded for E2 because they depend on break-bar information unavailable at E2 decision time.

Relevant code:
- `trading_app/config.py:2808-2827`

There are additional no-lookahead fixes in supporting analytics:
- `strategy_fitness.py` passes `end_date=as_of_date` to prevent future outcomes leaking into rolling fitness calculations.

Relevant code:
- `trading_app/strategy_fitness.py:624-627`

### 2. Price-proxy validity is supported, but only solves one problem

The repo explicitly treats MGC and MNQ as proxy-extended instruments:
- MGC uses GC full-size source data, same price, micro cost model.
- MNQ minimum start date is extended via NQ futures backfill.

Relevant code:
- `pipeline/asset_configs.py:61-70`
- `pipeline/asset_configs.py:83-90`

That supports using proxy history for **price-based** filters like:
- `COST_LT10`
- `COST_LT12`
- `OVNRNG_100`
- `ORB_G*`

Relevant code:
- `trading_app/config.py:2280-2295`
- `trading_app/config.py:2483-2487`

This does **not** solve search-bias or holdout-contamination problems. It only answers whether the historical price series is mechanically comparable enough for those filters.

### 3. Execution overlays are a separate source of bias if not audited separately

The repo now openly documents that multiple overlays are applied at execution time:
- calendar skip / half-size
- ATR velocity skip
- E2 order timeout

These are not equivalent to the discovery filter encoded in `strategy_id`.

Relevant docs/code:
- `docs/plans/2026-04-07-eligibility-context-design.md:12`
- `trading_app/config.py:2600-2611`
- `trading_app/execution_engine.py:207-212`
- `trading_app/execution_engine.py:654-674`

This matters because a lane can be "clean" at discovery time and still have later operational logic that changes the actual live decision rule. Those overlays may be valid, but they must be evaluated as overlays, not smuggled into claims about the original discovery evidence.

### 4. The biggest unresolved issue is holdout governance, not filter mechanics

Current repo authority is internally inconsistent:
- `RESEARCH_RULES.md` says 2026 holdout is sacred.
- April rebuild instructions say discovery must run with `--holdout-date 2026-01-01`.
- A pre-registration file says the holdout test completed and 2026 data is now included in discovery.
- `pipeline/check_drift.py` contains a holdout contamination checker, but its declaration map is currently empty, so no active instrument is being guarded by that check.

Relevant files:
- `RESEARCH_RULES.md:26`
- `docs/plans/2026-04-02-16yr-pipeline-rebuild.md:79-83`
- `docs/plans/2026-04-02-16yr-pipeline-rebuild.md:110`
- `docs/pre-registrations/2026-03-20-mnq-rr1-verified-sessions.md:4`
- `pipeline/check_drift.py:3354-3405`

Until that is resolved, "clean 2026 OOS" claims are not safe.

### 5. Current policy and current code still say DSR is informational

The validator explicitly removed DSR/FST as hard gates:
- rationale: DSR and FST were misleading without a trustworthy `N_eff`.
- multiple testing is currently handled by BH FDR as the hard gate.

Relevant code:
- `trading_app/strategy_validator.py:583-589`
- `trading_app/strategy_validator.py:1400-1453`

The adversarial audit says the same thing:
- DSR analytical cross-check is useful
- ONC / better `N_eff` estimation is still TODO

Relevant doc:
- `docs/plans/2026-03-18-adversarial-review-findings.md:47-58`

Current DB state is aligned with that caution:
- `validated_setups = 124`
- `edge_families = 61`
- `0 / 124` have `dsr_score > 0.95`
- max `dsr_score = 0.1198`

So any document treating `DSR > 0.95` as an immediately binding deployment law is ahead of what the repo has actually solved.

### 6. Trade count is not the core weakness for the current lane set

Current validated lane samples are materially larger than "tiny sample" intuition suggests:
- `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`: N=591
- `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_COST_LT12`: N=766
- `MNQ_TOKYO_OPEN_E2_RR2.0_CB1_COST_LT10`: N=849
- `MNQ_EUROPE_FLOW_E2_RR3.0_CB1_COST_LT10`: N=978
- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_OVNRNG_50`: N=1441
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_COST_LT10`: N=1941
- `MES_NYSE_OPEN_E2_RR2.0_CB1_COST_LT12`: N=1593

But large raw trade count does not cancel:
- selection bias from thousands of tested variants
- regime-span weakness
- holdout contamination
- policy drift between discovery and live overlays

The failure mode here is not "too few trades to say anything." The failure mode is "too many related ways to say something positive."

---

## Where the new institutional docs are strong

The new `docs/institutional/` layer gets several important things right:
- literature extracts now exist locally instead of hand-waving from model memory
- brute-force discovery against finite clean history is correctly identified as the core statistical failure
- pre-registration and smaller search budgets are the right direction
- price-based proxy use is separated from volume-filter contamination
- current lanes are not treated as automatically production-grade

Relevant files:
- `docs/institutional/README.md`
- `docs/institutional/finite_data_framework.md`
- `docs/institutional/pre_registered_criteria.md`

---

## Where the new institutional docs still overreach

### 1. They jump from literature extraction to binding policy faster than the codebase supports

Examples:
- `finite_data_framework.md` makes `DSR > 0.95` a binding validation rule.
- `pre_registered_criteria.md` makes DSR a required deployment threshold.

But the live validator and prior adversarial audit still classify DSR as informational because `N_eff` remains unresolved.

This is not a small detail. It is the difference between:
- "DSR is an informative warning light"
- and "DSR is a lawful deploy / don't deploy switch"

The repo has only established the first.

### 2. Chordia / HLZ thresholds are being imported too directly

Using Chordia's `t >= 3.79` as a universal hard bar for this ORB futures search is too aggressive as stated.

What is grounded:
- finance multiple testing requires much higher bars than naive 1.96
- t-statistics around 2 are weak evidence in large test universes

What is not yet grounded:
- that the exact Chordia threshold should be transplanted unchanged into this repo's specific ORB family structure

The honest use is as a **severity benchmark**, not as a fully proven repo-native law.

### 3. The framework assumes a clean post-2026 process that the repo does not yet prove

`docs/institutional/pre_registered_criteria.md` includes 2026 OOS positivity as a locked criterion, but repo authority still conflicts on whether 2026 is still protected or already folded back into discovery.

That criterion is only valid after research state is frozen and lineage is explicit.

---

## Bias controls that must be treated as mandatory

Any future audit or rediscovery should enforce the following:

1. **Canonical layers only for truth-finding.**
   Use `bars_1m`, `daily_features`, and `orb_outcomes`. Treat docs, `validated_setups`, `prop_profiles`, and memory as potentially stale summaries.

2. **Hard no-lookahead filter set for E2.**
   Allowed E2 families are those whose inputs exist at decision time.
   Excluded for E2: `VOL_RV*`, `ATR70_VOL`, `_CONT`, `_FAST`, or any filter dependent on a completed break bar.

3. **Execution overlays audited separately from discovery filters.**
   Calendar rules, ATR velocity, market-state gating, and E2 timeouts must be reported as operational overlays, not retroactively folded into original evidence claims.

4. **One holdout policy at a time.**
   Either:
   - run a holdout-clean research state with 2026 excluded from discovery
   - or explicitly declare that 2026 is no longer a clean holdout and stop using it as clean OOS evidence

   The project cannot honestly do both.

5. **No post-hoc K switching.**
   The test family and K must be locked before re-running discovery. Do not use raw global K when it is convenient to kill results and local K when it is convenient to rescue them.

6. **No threshold relaxation in response to current lane performance.**
   If criteria are loosened, the change must be justified from source literature or a formally better repo-specific estimator, not because the best current lane "almost passes."

---

## Clean analysis plan from here

### Phase A — Freeze the research state

Pick one of two states and document it explicitly:

- **State 1: Holdout-clean audit**
  - discovery data ends 2025-12-31
  - 2026 not used for session, RR, filter, or lane choice
  - 2026 only used later for OOS reporting

- **State 2: Post-holdout monitoring state**
  - 2026 already consumed
  - no more claims of "clean 2026 holdout"
  - all forward evidence from here is live/paper-forward only

Do not mix the two.

### Phase B — Restrict the audited ORB universe

For the first clean finite-data audit, restrict to:
- E2 only
- price-based filters only
- sessions already known to be operationally relevant
- fixed ORB aperture
- fixed, pre-registered RR ranges

Explicitly exclude contaminated or policy-drifting families:
- break-bar-volume families
- `ATR70_VOL`
- `VOL_RV*`
- `_FAST`
- `_CONT`
- ad hoc overlay combinations not present in discovery IDs

This reduces bias far more effectively than inventing softer pass thresholds.

### Phase C — Use an honest K on the pre-registered family

Before running discovery:
- define the exact hypothesis family
- define the exact test count
- define the exact filters and thresholds
- define kill criteria

Then run BH FDR on that family.

This is the correct answer to "are we doing too many independent trials?" The answer is yes, but the fix is **narrower pre-registered search**, not blind re-running.

### Phase D — Treat DSR as a cross-check until `N_eff` is properly solved

Use DSR in the audit write-up, but do not make it the sole deploy/no-deploy switch until:
- `N_eff` estimation is formalized for this repo
- the estimation method is documented
- the calculation path is verified against the literature extract

Until then:
- BH FDR on honest K
- WFE
- era stability
- no-lookahead eligibility
- clean holdout / forward monitoring

should carry the burden of the decision.

### Phase E — Reclassify current lanes honestly

Current lanes should be described as:
- **operationally deployable**
- **research-provisional**
- **not yet production-grade proof**

That is materially different from calling them "validated" in the strong institutional sense.

---

## Final position

The correct grounded position is:

- The old brute-force discovery process was too loose.
- The new finite-data framework is directionally better.
- The repo is not yet entitled to claim that all of the new locked criteria are already solved.
- The most important unsolved problem is governance of holdout state and effective trial counting, not whether micro futures have 20 years of native history.
- The proper next step is a no-lookahead, canonical-layer, pre-registered re-audit on a restricted ORB family budget.

Until that is done, the honest stance is:

**Current ORB lanes may be useful and tradeable, but the evidence base is provisional and narrower than some recent docs imply.**

---

## Recommended immediate actions

1. Decide whether the project is in holdout-clean or post-holdout-monitoring mode.
2. Update `pipeline/check_drift.py` so holdout contamination checking reflects the chosen policy instead of silently checking nothing.
3. Add an audit note to `docs/institutional/` clarifying that DSR remains informational until `N_eff` is formally solved in-repo.
4. Write one pre-registered narrow ORB hypothesis file for a price-based E2 family and audit that end-to-end without overlays.
5. Re-label current live lanes in docs as **provisional** rather than strongly validated.
