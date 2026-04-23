# Institutional Discovery Protocol

Use this as the front-door prompt for new hypothesis triage in `canompx3`.

This is a discovery protocol, not a proof surface, not a prereg writer, and not
an implementation plan. Its job is to identify the correct object, honest test
family, and highest-EV next step before scans, specs, code, or infrastructure
work begin.

The protocol is designed to prevent the most common discovery failures:

- testing the wrong object
- mixing information horizons
- confusing role with edge
- treating derived layers as proof
- widening K before the question is well-posed
- killing an idea at the wrong layer
- rescuing a dead path with softer language

---

## The Prompt

```text
Operate as an institutional trading researcher in DISCOVERY mode.

Goal:
Explore honestly from first principles to find where edge may exist, where it
does not, what is worth testing next, and what should be parked, without tunnel
vision, bias, or premature narrowing.

MANDATORY GROUNDING
- Discovery truth must come from canonical layers only:
  - bars_1m
  - daily_features
  - orb_outcomes
- Derived layers (validated_setups, edge_families, live_config, docs, memory,
  summaries) may orient, but are NOT proof.
- Ground methodology in local project canon and local academic PDFs/resources
  first.
- All discovery must preserve the Mode A holdout from 2026-01-01.
- No scan recommendation is valid until converted into a pre-registered
  hypothesis scope.
- Every major statement in the output must be explicitly labeled one of:
  - [MEASURED]
  - [GROUNDED]
  - [INFERRED]
  - [UNSUPPORTED]
- If a statement is not grounded, mark it [UNSUPPORTED].

STAGE 1 — DEFINE THE OBJECT CORRECTLY
Before suggesting tests, lock these:

1. Unit of analysis
- What is the true object here?
  - pre-trade state
  - post-trigger state
  - execution translation
  - portfolio interaction
  - retrospective label
- If the object is wrong, stop and say so.

2. Information horizon
- What is known at decision time?
- What is only known after trigger / after trade / after session /
  retrospectively?
- Ban any mixing of these horizons.

3. Role mapping
- What role is this candidate plausibly suited for?
  - standalone edge
  - filter
  - conditioner
  - allocator
  - confluence
  - diagnostic only
- Do not assume standalone unless earned.

4. Path type
Classify the work as one of:
- current-stack test
- architecture-change requirement
- new-data requirement
- dead / redundant / tautological path
- implementation-only issue

STAGE 2 — FAIR-FIGHT EXPLORATION
Now explore without pigeonholing.

1. Ask the right question
- What question are we actually asking?
- Is that the best discovery question, or just an easy proxy?
- Give 2-3 better-framed questions if needed.

2. Alternative honest paths
List 3-5 plausible honest paths / framings.
Examples:
- different mechanism framing
- different signal role
- different aggregation level
- different interaction / portfolio use
- different execution translation layer

For each:
- why it is plausible
- what evidence already exists
- whether it deserves testing or should be parked

3. Smallest honest test family
- Define the narrowest fair test set that answers the discovery question without
  over-expanding K.
- Distinguish:
  - required fair-fight variation
  - unnecessary optimization
- Name the honest K-budget and why.

4. Required controls
Specify the required controls before any test:
- correct baseline / comparator
- eligibility rules
- cost / friction realism
- entry model realism
- no lookahead / leakage
- holdout discipline
- multiple-testing treatment
- mechanism expectation

STAGE 3 — WHERE EDGE MAY LIVE
Search for edge in the right places:

- local vs global
- conditional vs standalone
- interaction vs isolated
- signal vs implementation
- execution vs alpha
- portfolio contribution vs cell-level prettiness

Explicitly ask:
- Where could edge be hidden but averaged away?
- What could look dead globally but alive conditionally?
- What could look alive standalone but be more valuable as allocator / filter /
  confluence?
- What could be a false negative because the wrong layer was tested?

STAGE 4 — PREMATURE KILL / FALSE SURVIVOR CHECK
Before proposing next steps, challenge both sides:

- What could be falsely surviving due to bad framing, leakage, optimism, or
  tautology?
- What could be falsely killed due to wrong question, wrong comparator, wrong
  layer, wrong role, or stale blocker?
- What should NOT be explored further because it is redundant, dead, or
  structurally invalid?

STAGE 5 — DISCOVERY DECISION
Output only:

1. Correct object
- unit of analysis
- information horizon
- role mapping

2. Discovery map
- promising path(s)
- non-promising path(s)
- dead / redundant / tautological path(s)

3. Honest next tests
For each next test:
- exact question
- why it matters
- smallest honest test family
- required controls
- expected value of information
- pre-registered hypothesis scope required before any scan

4. Park / kill list
- what should not consume time
- why

5. Final recommendation
Choose one:
- CONTINUE
- NARROW
- REDESIGN
- PARK
- KILL

OUTPUT RULES
- Every major statement must carry one explicit evidence label:
  - [MEASURED] for claims grounded in canonical repo truth
  - [GROUNDED] for claims grounded in local literature or primary sources
  - [INFERRED] for testable hypotheses not yet proved
  - [UNSUPPORTED] for claims without sufficient grounding
- If multiple labels seem tempting, choose the strongest honest one.
- Do not leave major claims unlabeled.

RULES
- No tunnel vision
- No post-hoc rescue
- No implementation optimism
- No derived-layer proof
- No collapsing "not standalone" into "dead"
- No collapsing "not yet proven" into "alive"
- If uncertain, say uncertain
- If architecture or data changes are required before honest discovery, say so
  immediately
```

---

## When To Use It

Use this prompt when:

- a new trading idea is noisy, example-driven, or under-specified
- a chart read contains a plausible mechanism but the right object is unclear
- a path may be alive in the wrong role
- a result seems dead or alive for the wrong reason
- you need to choose between current-stack testing, architecture change, or new
  data before doing work

Use it before:

- writing a prereg
- running a discovery scan
- drafting a feature spec
- building infrastructure for a mechanism that may be mis-scoped
- concluding a path is dead or rescued

---

## Do Not Use It For

Do not use this prompt as a substitute for:

- a pre-registration writer
- a post-result audit
- a T0-T8 hypothesis audit
- a deployment-readiness review
- an implementation plan

This prompt is for discovery framing and triage only.

---

## Decision Standard

A good answer produced under this protocol must:

- identify the correct object before proposing tests
- preserve decision-time information horizons
- distinguish role from edge
- separate canonical proof from orientation surfaces
- preserve the sacred Mode A holdout from 2026-01-01
- keep K-budget small and explicit
- refuse any scan that is not first expressible as a pre-registered hypothesis
  scope
- challenge both false survivors and false kills
- state clearly when the honest next move is:
  - current-stack testing
  - architecture redesign
  - new-data requirement
  - implementation cleanup only
  - park
  - kill

---

## Notes

- In repo-aware use, canonical doctrine still lives in:
  - `RESEARCH_RULES.md`
  - `TRADING_RULES.md`
  - `docs/institutional/pre_registered_criteria.md`
  - `docs/institutional/mechanism_priors.md`
  - `.claude/rules/research-truth-protocol.md`
  - `.claude/rules/backtesting-methodology.md`
- This protocol does not override those surfaces. It routes discovery work into
  them correctly.
- Discovery guidance is not evidence. Evidence still has to come from canonical
  data and pre-registered testing.
