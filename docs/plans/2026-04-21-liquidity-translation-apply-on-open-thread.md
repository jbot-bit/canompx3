# Applying The Liquidity / Displacement Translator To The Open Research Thread

**Date:** 2026-04-21  
**Status:** DECISION SURFACE  
**Purpose:** apply `docs/prompts/LIQUIDITY_DISPLACEMENT_TRANSLATOR.md` to the
already-open HTF/LTF / retest / sweep-reclaim thread so the repo stops mixing
retail prompt language, dead ORB execution variants, and untested standalone
trade families.

## Grounding

Local authority only:

- `docs/prompts/LIQUIDITY_DISPLACEMENT_TRANSLATOR.md`
- `docs/institutional/mechanism_priors.md`
- `docs/specs/level_interaction_v1.md`
- `docs/institutional/pre_registered_criteria.md`
- `RESEARCH_RULES.md`
- `docs/institutional/edge-finding-playbook.md`
- `resources/Algorithmic_Trading_Chan.pdf`
- `resources/Robert Carver - Systematic Trading.pdf`
- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf`
- `resources/Two_Million_Trading_Strategies_FDR.pdf`

## 1. What the earlier work actually tested

### A. ORB retest pilot

Reference:

- `research/orb_retest_entry_pilot_v1.py`

Question actually tested:

- Does a very specific `ORB break -> later retest of ORB boundary ->
  continuation entry` beat canonical `E2`?

Translator classification:

- **Role:** execution variant on an existing ORB lane
- **Not:** standalone liquidity/displacement family

Verdict:

- Dead for that bounded execution route.
- Does **not** kill broader sweep / reclaim / displacement families.

### B. Sweep-reclaim v1

Reference:

- `docs/audit/hypotheses/2026-04-19-sweep-reclaim-v1.yaml`
- `docs/audit/results/2026-04-19-sweep-reclaim-v1.md`

Question actually tested:

- Do reclaimed PDH/PDL sweep events have positive signed next-2-bar event-study
  response in a narrow first-30-minute window?

Translator classification:

- **Role:** research-only event study
- **Mechanism:** closest current local analogue to a standalone sweep/reclaim
  idea
- **Not:** a trade strategy, not a deployable family, not an ORB filter

Verdict:

- Null at the locked `K=36` event-study scope.
- Useful as mechanism reconnaissance only.

## 2. Where the earlier thread was tunnel-visioned

### Tunnel 1 — treating all retest-style ideas as one thing

- `E_RETEST` continuation on an ORB boundary
- swept prior-day level reclaim reversal
- opening-drive pullback continuation

These are different mechanisms. Only the first was truly killed.

### Tunnel 2 — treating event studies as trade strategies

`sweep-reclaim-v1` measured short-horizon response after reclaim.

It did **not** define:

- entry price
- stop logic
- target logic
- fill realism
- trade-time costs

So the null result cannot be used to close the standalone family.

### Tunnel 3 — importing chart-language without translation

Terms like:

- displacement
- imbalance
- FVG
- inducement

were never the right unit of evidence. They needed translation into geometry
and timing first.

## 3. Honest role map after translation

### A. Standalone

Still open.

Best admissible family from the prior thread:

- `sweep -> reclaim reversal`

because it is already adjacent to `level_interaction_v1` and is genuinely
distinct from the dead ORB retest execution variant.

### B. Filter / Conditioner

Also open, but lower priority for this specific prompt-family.

The repo already has better live evidence on ORB conditioners from the prior-day
context thread. Liquidity/displacement prompts do not add much there yet.

### C. Allocator

Not the right first use.

Need a validated conditioner first.

### D. Confluence

Too early.

Going straight to `level state AND displacement state AND participation`
would create implementation drag and K inflation before a single base family is
validated.

## 4. Best opportunity

**Best opportunity:** a bounded standalone `sweep -> reclaim reversal` family.

Why this ranks first:

- it is structurally different from the dead ORB `E_RETEST` route
- it uses the existing local `level_interaction_v1` machinery
- it avoids importing FVG ontology
- it is the cleanest translation of the useful part of the external prompt

## 5. Biggest blocker

The previous branch asked the wrong question in the wrong form.

`sweep-reclaim-v1` asked:

- "is there a short-horizon event-study response after reclaim?"

The tradable question is:

- "does a bounded sweep-reclaim reversal trade with explicit entry, stop,
  target, and cost geometry survive local gates?"

Those are not the same.

## 6. Biggest miss

The repo had already built the right thin abstraction:

- `docs/specs/level_interaction_v1.md`

But the follow-on family stayed at event-study level and never translated into a
real trade definition. That left the whole mechanism in an awkward half-state:

- not dead
- not tradeable
- easy to overclaim

## 7. Highest-EV next test

Preserve a pre-reg starter for:

- `MNQ`, `MES`
- sessions: `NYSE_OPEN`, `EUROPE_FLOW`
- levels: `prev_day_high`, `prev_day_low`, `overnight_high`, `overnight_low`
- mechanism: swept `close_through` + reclaim
- role: **standalone reversal**
- geometry:
  - entry on reclaim close or first pullback to reclaimed level
  - stop beyond sweep extreme
  - target at next ex-ante liquidity pool or fixed `RR`

This is the smallest honest family that actually tests the translated idea.

## 8. What not to do next

Do not:

- reopen `E_RETEST`
- reopen broad FVG / IFVG ontology
- start with confluence
- call the old `sweep-reclaim-v1` null a kill on standalone sweep/reclaim
- broaden to 6 sessions × 3 instruments × 3 apertures on first pass

## Net decision

The translator changes the earlier thread this way:

- `ORB retest continuation` stays dead
- `sweep-reclaim v1` is reclassified as useful reconnaissance, not a kill
- the next honest family is a **standalone sweep-reclaim reversal starter**
