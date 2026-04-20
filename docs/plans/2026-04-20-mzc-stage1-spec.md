# MZC Stage-1 Research Spec

**Date:** 2026-04-20
**Status:** Locked before data onboarding
**Scope:** Structural-feasibility and first research spec for Micro Corn (`MZC`) as the first agriculture diversification candidate

---

## 1. Why this exists

The asset-universe audit closed the parent/proxy cleanup branch enough to move
to the next candidate. Per the existing diversification program, agriculture is
the next highest-EV branch after rates, but only if it is approached with a
narrow mechanism-first design.

This document locks that design before any ag data onboarding.

It exists specifically to prevent:

- "download first, invent later" research
- 24-hour ORB cloning on a market with different structure
- broad parameter mining on a short-history micro contract

---

## 2. Grounding

Repo grounding used:

- `docs/plans/diversification-research-program.md`
- `docs/plans/diversification-candidate-shortlist.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/institutional/pre_registered_criteria.md`

The shortlist already states the correct direction:

- start with micro agriculture, but one crop at a time
- start with `MZC` before `MZS`
- do not begin with broad ORB
- begin with USDA-report response structure
- standard `ZC` can be used as the research proxy if micro-specific history is too short

This spec adopts that guidance as binding.

---

## 3. Stage-1 question

**Primary question:**

Does corn show a repeatable, economically coherent intraday response around
major USDA report windows that is distinct enough from the current gold/equity
ORB book to justify deeper research?

This is a structural-feasibility question, not a validation claim.

Allowed outcome labels:

- `GO_TO_STAGE_2`
- `NO_GO`
- `NOT_WORTH_DATA_BUDGET`

Not allowed at Stage 1:

- "validated edge"
- "deployable"
- generic "corn works"

---

## 4. Locked market / contract scope

### Primary execution contract

- `MZC` only

### Allowed research proxy

- `ZC` only, and only with explicit proxy disclosure

### Not in scope

- `MZS`
- `ZS`
- broad multi-grain sweep
- cross-commodity composite logic

Reason:

- the goal is to learn whether the **corn report-response mechanism** deserves
  more budget
- not whether "agriculture in general" can be made to backtest

---

## 5. Locked mechanism

The mechanism is **USDA information shock response**, not generic session-open
breakout.

Candidate driver set:

- USDA report releases
- crop-balance-sheet repricing
- acreage / yield / stocks surprise absorption
- day-session directional continuation or failed first move after the report

Why this may diversify:

- drivers are agricultural balance-sheet and weather / crop-cycle linked
- these are structurally different from equity/gold ORB drivers
- the event windows are discrete and mechanism-rich rather than "more of the
  same intraday breakout idea"

---

## 6. Allowed Stage-1 model family

Only one family is allowed:

- **event-window response around major USDA reports**

Allowed sub-structures:

1. immediate continuation after a genuine post-report directional break
2. failed first move / reversal only if the initial move and failure definition
   are fixed before testing

Not allowed:

- 24-hour ORB scan
- multi-session ORB sweep
- indicator families
- threshold-rich filter zoo
- model blending

This is deliberately narrower than the rates pilot because the micro contract is
newer and the temptation to overfit is higher.

---

## 7. Required verification gates before any serious backtest

### Gate A. Vendor / raw availability

Verify directly:

- `MZC.FUT` minute-bar availability
- `ZC.FUT` minute-bar availability
- billable size / cost for practical research windows

Do not assume availability from docs alone.

### Gate B. Session / event compatibility

Verify from actual minute bars:

- there is usable coverage around the chosen USDA release window
- the market is active enough in the target window to make event analysis honest
- shock/follow windows are not sparsely sampled

### Gate C. Sample-path realism

Before any Stage-2 work, estimate:

- number of relevant reports in the available horizon
- whether `MZC` alone can plausibly reach meaningful event count
- whether `ZC` proxy is required just to make Stage 2 statistically honest

### Gate D. Execution realism

At Stage 1, this is only a feasibility gate:

- micro product must be practically tradable
- event windows must not imply obviously impossible fill assumptions

If event windows are too violent for realistic micro execution, kill early.

---

## 8. Stage-1 outputs

Stage 1 must answer these, and only these:

1. Is the event window structurally tradable on the tape we can actually get?
2. Is there enough event history to justify a narrow Stage-2 scan?
3. Is `MZC` enough on its own, or is `ZC` proxy required?
4. Does the mechanism look genuinely different enough to matter for the
   portfolio if it survives later stages?

---

## 9. Kill criteria

Kill the branch immediately if any of the following hold:

1. usable event count is too small with no honest proxy path
2. minute-bar coverage around the report window is sparse or structurally poor
3. the best-looking result only appears after widening parameters
4. price action is too dominated by chaotic event noise to define a stable
   first-pass structure
5. execution assumptions for the micro contract would obviously dominate the
   signal
6. the candidate ends up being just another same-day directional macro-beta
   clone of existing book stress

---

## 10. Success criteria for Stage 1

Advance to Stage 2 only if all of the following are true:

1. raw data is obtainable and honest
2. report-window coverage is operationally usable
3. there is a fixed narrow model family to test
4. there is a plausible path to enough observations using `MZC` and, if
   required, explicitly-declared `ZC` proxy support
5. the mechanism remains distinct from the current ORB concentration story

---

## 11. Immediate next step after this spec

With this spec locked, the next action is:

1. run the vendor / coverage feasibility probe for `MZC` and `ZC`
2. decide `GO_TO_STAGE_2`, `NO_GO`, or `NOT_WORTH_DATA_BUDGET`

No ag data onboarding beyond that should happen until those gates are answered.

