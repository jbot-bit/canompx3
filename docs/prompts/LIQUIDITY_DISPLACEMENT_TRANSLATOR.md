# Liquidity / Displacement Prompt Translator

**Purpose:** turn external "liquidity / sweep / displacement / FVG" material into
repo-safe research inputs for `canompx3` without importing retail chart-lore,
post-hoc bias, or unverifiable ontology.

**Use this when:**
- a user pastes ICT / SMC / liquidity-sweep / FVG style material
- a session wants new hypothesis ideas grounded in market-structure priors
- you need to convert discretionary language into a bounded pre-reg

**Do not use this as:**
- evidence that a setup works
- permission to skip `docs/institutional/pre_registered_criteria.md`
- a replacement for `docs/specs/level_interaction_v1.md`

---

## 1. Grounding rule

Treat external liquidity/displacement material as **ideation only**.

The admissible local grounding for this repo is:

- `docs/institutional/mechanism_priors.md`
- `docs/specs/level_interaction_v1.md`
- `docs/institutional/pre_registered_criteria.md`
- `RESEARCH_RULES.md`
- `docs/institutional/edge-finding-playbook.md`
- `resources/Algorithmic_Trading_Chan.pdf`
- `resources/Robert Carver - Systematic Trading.pdf`
- `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf`
- `resources/Two_Million_Trading_Strategies_FDR.pdf`

If an external prompt says:

- "this is the only real model"
- "institutional footprint"
- "works across all markets"
- "best markets right now"

that is **not** a proved claim in this repo. It is only a candidate prior.

---

## 2. What is actually usable

The useful content in liquidity/displacement prompts is usually one of these:

1. **Level interaction**
- obvious highs/lows
- prior-day and session levels
- sweep / reclaim / failed-break logic

2. **Expansion / displacement**
- fast move after interaction
- non-overlap / expansion
- shallow pullback after impulse

3. **Session timing**
- London / NY open
- avoid dead midday chop
- event-window sensitivity

4. **Role ideas**
- standalone reversal
- continuation after opening drive
- conditioner on an existing lane
- confluence / allocator role

These are all acceptable **research families** if translated into objective
definitions.

---

## 3. What is not usable as-is

Do not carry these forward verbatim:

- `FVG`, `IFVG`, `order block`, `inducement`, `absorption`
- "institutional" explanations without local support
- broad market rankings
- universal edge claims
- discretionary phrases like "clean structure", "strong impulse", "obvious range"

They may still be used, but only after conversion into explicit geometry,
timing, and threshold definitions.

---

## 4. Translation map

Use this table when converting prompt language into a pre-reg.

| Prompt term | Repo-safe translation |
|---|---|
| liquidity pool | `prev_day_high`, `prev_day_low`, `overnight_high`, `overnight_low`, session high/low, ORB high/low |
| sweep | `level_interaction_v1` sweep flag using explicit `sweep_epsilon` |
| reclaim | `level_interaction_v1` reclaim within locked lookahead bars |
| failed breakout | `wick_fail` or swept `close_through` followed by reclaim |
| displacement | explicit expansion metric: range multiple, close-to-close impulse, overlap constraint, or post-break travel threshold |
| imbalance / FVG | geometric gap / non-overlap rule only; never as ontology |
| micro pullback | first retest of level / impulse midpoint / opening-drive anchor within `N` bars |
| trend bias | explicit pre-trade state only: prior-day relation, ORB side, VWAP relation, gap sign, regime bucket |
| no-trade chop | low expansion, overlap-heavy sequence, or compressed ORB regime with locked threshold |
| inducement | pre-sweep fake move; only allowed if formally defined ex ante |
| absorption | not admissible unless mapped to a concrete pre-trade or event-time observable in canonical data |

---

## 5. Allowed role framings

Any liquidity/displacement idea must declare its role before testing:

### A. Standalone
- Example: `sweep -> reclaim reversal`
- Good when it defines a genuinely new trade species
- Must not be disguised ORB reuse

### B. Filter
- Example: take current ORB only when a pre-trade liquidity state holds
- Must use pre-entry knowable information only

### C. Conditioner
- Example: same lane, but expectancy differs when price opens below `PDL`
- Best use for local context edges

### D. Allocator
- Example: reduce exposure in inside-range days, upsize after washed-out opens
- Only after a conditioner is already proven

### E. Confluence
- Example: `level state AND participation state`
- High implementation drag; do not jump here first

---

## 6. Best-practice conversion rules

### Rule 1 — Separate geometry from story

Bad:
- "price leaves an institutional footprint"

Good:
- "after sweeping `prev_day_high`, price closes back below it within 2 bars"

### Rule 2 — Use ex-ante levels first

Prefer:
- prior-day high/low
- overnight high/low
- session highs/lows once chronologically safe
- ORB high/low

Avoid inventing new synthetic structures unless a local spec already exists.

### Rule 3 — Keep first family small

For any new liquidity/displacement family:

- one mechanism
- one or two sessions
- one or two instruments
- one or two apertures
- a small fixed `K`

Do not turn a prompt into a 200-cell scope bomb.

### Rule 4 — Test one role at a time

If the idea could be:
- standalone
- filter
- conditioner
- allocator

pick one. Do not test all four in one pre-reg.

### Rule 5 — Favor session-bounded execution

These prompts are usually strongest as:
- London open
- NY open
- first hour / opening-drive logic

That fits both:
- local mechanism priors
- the repo’s existing ORB/session architecture

### Rule 6 — Fail closed on ontology

If a term cannot be defined objectively from canonical data, either:
- replace it with an explicit measurable rule, or
- exclude it from the family

---

## 7. Highest-EV starter families

These are the best first conversions from liquidity/displacement prompts.

### 1. Sweep -> reclaim reversal

Why:
- already adjacent to `docs/specs/level_interaction_v1.md`
- likely to produce a genuinely distinct trade family
- does not require FVG ontology

First bounded scope:
- `MNQ`, `MES`
- `NYSE_OPEN`, `EUROPE_FLOW`
- `5m` or `15m`
- levels: `prev_day_high`, `prev_day_low`, `overnight_high`, `overnight_low`

### 2. Opening-drive pullback continuation

Why:
- closer to current ORB infrastructure
- cleaner execution than loose “breaker/FVG” language
- fits Chan-style intraday momentum framing

First bounded scope:
- opening expansion threshold
- first pullback only
- no midday
- no event-window overlap on first pass

### 3. Failure-of-sweep opposite-side trade

Why:
- often cleaner than raw continuation
- natural inversion of the same mechanism
- can be standalone or conditioner later

---

## 8. Prompt use policy for future sessions

When a user pastes a market-structure prompt, classify it immediately:

### `IDEATION_ONLY`
- broad edge claims
- discretionary language
- market rankings
- no objective definitions

### `PRE_REG_CANDIDATE`
- one mechanism
- one role
- clear session/instrument focus
- objective, testable definitions can be written

### `IMPLEMENTATION_CANDIDATE`
- only after a family already passed local gates

### `ALREADY_DISPROVEN_ANALOGUE`
- if it maps onto a dead branch already in repo history

### `NEEDS_SEPARATE_MARKET_STRUCTURE_VERIFICATION`
- if it depends on current external market conditions, venue structure, costs, or data coverage

---

## 9. Generator block

Use this block when converting a pasted prompt into a canompx3-ready pre-reg idea:

```text
Take the pasted liquidity/displacement material as IDEATION_ONLY.

Ground only in:
- docs/institutional/mechanism_priors.md
- docs/specs/level_interaction_v1.md
- docs/institutional/pre_registered_criteria.md
- RESEARCH_RULES.md
- resources/Algorithmic_Trading_Chan.pdf
- resources/Robert Carver - Systematic Trading.pdf
- resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf
- resources/Two_Million_Trading_Strategies_FDR.pdf

Tasks:
1. Strip out all universal claims, discretionary chart language, and ungrounded ontology.
2. Translate the idea into one role only: standalone, filter, conditioner, allocator, or confluence.
3. Convert every term into objective geometry/timing definitions.
4. Keep K small and pre-committed.
5. State what local files already cover the mechanism and what remains ungrounded.
6. Produce:
   - exact mechanism statement
   - role
   - bounded scope
   - hypothesis family
   - numeric kill criteria
   - non-actions

Forbidden:
- "institutional footprint" as evidence
- FVG/IFVG/order-block ontology without explicit geometry
- broad market ranking claims
- parameter sweeps beyond the declared family
- using 2026 OOS to design the family
```

---

## 10. Net rule

External liquidity/displacement prompts are useful **only** if they become:

- a prior-beliefs input
- an objective local spec
- a small pre-regged family

If they stay as rhetoric, they are noise.
