# Research Pipeline Contract

**Date:** 2026-04-23
**Authority:** complements `RESEARCH_RULES.md`, `docs/specs/research_modes_and_lineage.md`,
`docs/institutional/conditional-edge-framework.md`, and
`docs/prompts/INSTITUTIONAL_DISCOVERY_PROTOCOL.md`.

This file defines how discovery, validation, role classification, deployment,
and execution relate. It exists so agents do not confuse internal command
plumbing with the user-facing research workflow, or confuse research inventory
with the live book.

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

## 3. Supported route options

Every research or runtime question should land on exactly one primary route.
If a request spans routes, answer them in this order: research status first,
then deployment, then execution.

| Route | Use when | Allowed inputs | Output / destination | Forbidden claim |
|---|---|---|---|---|
| `standalone_discovery` | Finding better complete tradeable setups | canonical data + `standalone_edge` prereg | `experimental_strategies`, then validator / `validated_setups` if accepted | A candidate row is proof or deployable by itself |
| `conditional_role` | Testing a filter, conditioner, allocator, confluence, execution modifier, or diagnostic | canonical data + `conditional_role` prereg with role block | bounded result doc and explicit role decision / contract | It must become a standalone strategy to be useful |
| `confirmation` | Deciding whether a discovered standalone candidate becomes validated research inventory | candidate provenance + validation / OOS / FDR evidence | `validated_setups` for accepted standalone lanes | Live routing or `paper_trades` are required for validation |
| `deployment_readiness` | Asking whether validated research should enter the live or shadow book | `validated_setups`, role contracts, `deployment_scope`, profiles, lane caps, overlays, broker/account constraints | go/no-go deployment decision | Deployment creates research truth |
| `operations` | Asking whether a deployed route actually fired or behaved correctly | runtime config, journals, monitoring, broker state, `paper_trades` | execution / monitoring evidence | Execution records validate the original edge |

`scripts/tools/prereg_front_door.py` only inspects prereg-backed research
branches today: `standalone_edge` and `conditional_role`. Confirmation,
deployment-readiness, and operations questions are still first-class route
options for agents, but they use their existing repo surfaces rather than this
prereg tool.

---

## 4. Extra-Strategy Layer

`experimental_strategies` is the extra standalone-candidate inventory.

Use it for discovery outputs from complete tradeable lane candidates:

- one instrument / session / aperture / entry / target / filter shape
- pre-registered K accounting
- canonical source data only
- no deployment authority

Do **not** treat `experimental_strategies` as proof. It can show where improved
strategy candidates may live, but it is still a candidate shelf until the
validator / confirmation path accepts or rejects the object.

Do **not** use `experimental_strategies` as deployment truth. Runtime and
deployment analytics must use the validated and deployable surfaces described in
§ 6.

Conditional-role findings do not belong in `experimental_strategies` unless the
prereg explicitly defines a complete standalone lane. A conditioner, allocator,
confluence, execution modifier, or diagnostic variable may be useful without
being a standalone strategy.

---

## 5. Branches

### Standalone edge branch

Use when the object is a complete tradeable lane.

Flow:

`framed -> discovered -> confirmed -> validated -> deployed -> executed`

Operational path:

`prereg -> canonical discovery -> experimental_strategies -> validator / confirmation -> validated_setups -> optional deployment -> optional paper_trades`

Rules:

- Discovery truth comes only from `bars_1m`, `daily_features`, and
  `orb_outcomes`.
- Discovery may write candidates to `experimental_strategies`.
- Confirmation and validation may promote to `validated_setups`.
- `validated_setups` is the research validation shelf for standalone lanes.
- Nothing reaches live routing automatically.
- `paper_trades` are optional execution evidence after deployment, not a
  prerequisite for validation.

### Conditional-role branch

Use when the object is a filter, conditioner, allocator, confluence, execution
modifier, or portfolio/routing input.

Flow:

`framed -> discovered -> confirmed / role-tested -> validated for role -> optional deployment design -> optional execution`

Operational path:

`role prereg -> bounded canonical runner -> result doc -> role decision -> optional role contract -> optional runtime design`

Rules:

- Do not force conditional objects into standalone validation.
- Do not write conditional-role results to `experimental_strategies` unless the
  prereg explicitly defines a complete standalone lane.
- Do not call a conditional signal dead because it failed as a standalone lane.
- Do not call a conditional signal validated unless its parent, comparator,
  metric, and role-specific promotion target were tested.
- Until native conditional-role tables exist, the durable evidence is the result
  doc plus an explicit role contract, not a disguised standalone candidate row.

---

## 6. Validation is not deployment

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

`validated_setups` is not the live book. It is a validated research shelf. The
runtime must still check deployability and current account/profile selection,
including surfaces such as `deployment_scope`,
`trading_app.validated_shelf.deployable_validated_relation()`, prop profile
selection, lane caps, overlay contracts, broker constraints, and current
operator controls.

---

## 7. Widen versus narrow

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

## 8. Simulated use cases

### "Find better setups in MGC 5m"

Correct route:

1. Frame the mechanism and K budget.
2. Write or inspect a `standalone_edge` prereg.
3. Run canonical discovery into `experimental_strategies`.
4. Validate survivors through the validator / confirmation path.
5. Promote only accepted standalone lanes to `validated_setups`.

Allowed conclusion: "discovered candidate", "confirmed candidate", or
"validated standalone setup" depending on what passed.

Not allowed: "deploy it" merely because it appears in `experimental_strategies`.

### "Test whether MES participation is an allocator"

Correct route:

1. Frame the object as a `conditional_role`.
2. Define parent population, comparator, primary metric, and promotion target.
3. Run a bounded role-specific runner.
4. Write a result doc and role decision.
5. Design deployment only if the role decision survives.

Allowed conclusion: "allocator evidence passed / failed for the declared
parent and metric."

Not allowed: forcing the allocator result into `experimental_strategies` or
calling it dead because it is not standalone.

### "Is this live yet?"

Correct route:

1. Check research status first: discovered, confirmed, or validated.
2. If validated, check deployability: `deployment_scope`,
   `deployable_validated_relation()`, profile selection, and runtime route.
3. Check execution only after deployment: `paper_trades`, journals, monitoring.

Allowed conclusion: "validated but not deployed" or "deployed but not yet
executed."

Not allowed: requiring `paper_trades` before calling a standalone setup
validated.

### "Validate this discovered candidate"

Correct route:

1. Confirm the candidate came from an admissible discovery run.
2. Apply the validator / confirmation criteria for the declared family and
   holdout.
3. Promote only passing standalone lanes to `validated_setups`.
4. Stop before profile selection unless the user asked for deployment.

Allowed conclusion: "validated research shelf row" or "failed confirmation."

Not allowed: treating runtime deployment as a validation criterion.

### "Did the live/shadow path work?"

Correct route:

1. Confirm the object was deployed or shadow-routed first.
2. Inspect runtime config, journals, monitoring, broker state, and
   `paper_trades`.
3. Report execution behavior separately from research truth.

Allowed conclusion: "executed as expected", "did not fire", or "runtime issue."

Not allowed: using an execution record as proof that the edge is valid.

---

## 9. Agent obligation

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

## 10. Grounding

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
