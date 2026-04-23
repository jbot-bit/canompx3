# Research Pipeline Contract

**Date:** 2026-04-23
**Authority:** complements `RESEARCH_RULES.md`, `docs/specs/research_modes_and_lineage.md`,
`docs/institutional/conditional-edge-framework.md`, and
`docs/prompts/INSTITUTIONAL_DISCOVERY_PROTOCOL.md`.

This file defines how discovery, validation, role classification, deployment,
and execution relate. It exists so agents do not confuse internal command
plumbing with the user-facing research workflow.

---

## 1. Human-facing contract

The human should be able to ask in natural language:

- "find edges in this area"
- "run discovery on this mechanism"
- "test whether this is a filter or allocator"
- "is this validated or deployable?"

The agent is responsible for routing that request through the correct project
pipeline. The human should not need to remember script names, Python modules,
or command flags.

Internal scripts such as `scripts/infra/prereg-loop.sh` are implementation
details. They are acceptable in operator logs and verification notes, but they
are not the product interface.

---

## 2. Status ladder

These states are separate. Do not collapse them.

| State | Meaning | Typical destination |
|---|---|---|
| `framed` | Correct object, role, horizon, K budget, and kill criteria are defined | prereg file |
| `discovered` | A prereg run found a candidate or useful conditional signal | result doc or `experimental_strategies` |
| `confirmed` | Candidate survived the role-appropriate validation / OOS / FDR checks | confirmation result |
| `validated` | Candidate is accepted as deployable research inventory for the role tested | `validated_setups` for standalone lanes; role-specific docs/contracts for non-standalone roles until native tables exist |
| `deployed` | Validated object is selected into an active profile, allocator, overlay, or runtime route | profile / lane / overlay config |
| `executed` | The live or shadow system generated actual trade records | `paper_trades` / journals |

`validated` does **not** require `deployed`.

`deployed` does **not** create research truth.

`paper_trades` are operational records, not discovery proof.

---

## 3. Branches

### Standalone edge branch

Use when the object is a complete tradeable lane.

Flow:

`DISCOVERY prereg -> canonical run -> experimental_strategies -> CONFIRMATION / validator -> validated_setups -> optional live routing -> paper_trades`

Rules:

- Discovery truth comes only from `bars_1m`, `daily_features`, and
  `orb_outcomes`.
- Discovery may write candidates to `experimental_strategies`.
- Confirmation and validation may promote to `validated_setups`.
- Nothing reaches live routing automatically.

### Conditional-role branch

Use when the object is a filter, conditioner, allocator, confluence, execution
modifier, or portfolio/routing input.

Flow:

`DISCOVERY / role prereg -> bounded canonical runner -> result doc -> role decision -> optional role-specific contract -> optional runtime design`

Rules:

- Do not force conditional objects into standalone validation.
- Do not write conditional-role results to `experimental_strategies` unless the
  prereg explicitly defines a complete standalone lane.
- Do not call a conditional signal dead because it failed as a standalone lane.
- Do not call a conditional signal validated unless its parent, comparator,
  metric, and role-specific promotion target were tested.

---

## 4. Validation is not deployment

A strategy does **not** need to pass live routing or create `paper_trades` to be
research-validated.

Research validation answers:

- Does this object survive the pre-registered statistical and mechanism tests?
- Is the result clean under the correct K / FDR / holdout discipline?
- Does it belong on the deployable shelf for the role actually tested?

Deployment answers:

- Should this validated object be selected into the current live book?
- Does it coexist with existing lanes, risk caps, sizing, broker constraints,
  and account rules?
- Can the runtime express it without logical leakage or implementation drag?

Execution answers:

- Did the live or shadow system actually generate trade records?
- Did monitoring behave as expected?

Failure at deployment can block use without invalidating the research edge.
Failure in research validation kills or parks the candidate before deployment is
considered.

---

## 5. Widen versus narrow

The correct research posture is:

- **Widen across mechanism families** when the current family is exhausted,
  redundant, or overly local.
- **Narrow within each prereg** so the K budget, parent population, role, and
  comparator remain honest.

Good widening:

- distinct structural mechanisms with separate theory citations
- different instruments only when the mechanism plausibly ports
- apertures only when the ORB horizon is part of the mechanism
- role variants only when they are declared up front as separate tests

Bad widening:

- one giant grid because it is easy to enumerate
- hidden thresholds, hidden role shifts, or unreported failed variants
- using 2026 OOS to choose the winning shape
- turning every weak signal into a translation audit

---

## 6. Agent obligation

For a natural-language discovery or role request, the agent must:

1. classify the mode: `DISCOVERY`, `CONFIRMATION`, `DEPLOYMENT_ANALYTICS`, or
   `OPERATIONS`
2. classify the role: standalone, filter, conditioner, allocator, confluence,
   execution, routing, or diagnostic
3. state whether the next move is discovery, confirmation, deployment audit, or
   no-go
4. create or inspect the prereg before any run
5. use the prereg front door internally to verify the branch
6. execute only the branch that matches the prereg
7. report where the result landed and what it is allowed to mean

The agent should not ask the human to run a command unless direct human control
is necessary. It should run the repo tooling itself, then summarize the result.

---

## 7. Grounding

This contract is grounded in local project doctrine:

- `RESEARCH_RULES.md` for canonical truth layers, holdout discipline, and
  statistical standards
- `docs/specs/research_modes_and_lineage.md` for research modes and write
  destinations
- `docs/institutional/conditional-edge-framework.md` for role-aware testing and
  promotion language
- `docs/institutional/pre_registered_criteria.md` for preregistration and
  validation criteria
- `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md`
  for theory-first research
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` and
  `docs/institutional/literature/harvey_liu_2015_backtesting.md` for hidden
  trials and multiple-testing discipline

