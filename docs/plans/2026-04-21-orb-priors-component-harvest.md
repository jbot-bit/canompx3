# 2026-04-21 ORB Priors — Reusable Components Harvest Matrix (Revised)

## Objective

Produce an **optimal, repo-aligned** reusable-components plan for ORB priors with explicit classification:

- **Plug-In**: immediately reusable with existing primitives.
- **Adapt**: reusable but requires scoped build + verification.
- **Reject**: not acceptable in current posture.

This revision hardens grounding requirements by explicitly checking what is available in `/resources` and mapping any missing evidence to fail-closed decisions.

---

## Grounding Baseline (including `/resources`)

### What is currently available

- `/resources` currently contains only `resources/prop-firm-official-rules.md`, which is operationally useful but **not** a mechanism source for ORB component design.
- Mechanism and portfolio-design grounding therefore comes from the canonical institutional extracts in `docs/institutional/literature/` (Fitschen, Carver, Bailey/LdP, Chordia, Harvey-Liu) plus policy gates in `docs/institutional/pre_registered_criteria.md`.

### Hard grounding rule used in this plan

If a component lacks support from `/resources` **or** an accepted institutional literature extract already in-repo, it cannot be `Plug-In`; it is automatically `Adapt` or `Reject`.

---

## Optimization rubric (for “optimal plan”)

Priority order is set by:

1. **Expected utility lift per unit complexity**
2. **Evidence strength** (institutional + repo-local)
3. **Implementation blast radius**
4. **Failure mode severity** (lookahead, payoff-shape drift, hidden leverage)

This keeps high-EV/low-risk components first and forces high-risk geometry changes later.

---

## Reusable Components Matrix

| Item | Proxy (closest existing implementation) | Data needs | Test shape | Risk | Classification | Why for this repo now |
|---|---|---|---|---|---|---|
| **R1 FILTER (skip/take)** | `StrategyFilter` + `ALL_FILTERS` + grid routing | Existing `daily_features` + `orb_outcomes`; no schema change | Standard pre-reg + T0→T8 + OOS consistency + drift checks | **Low** | **Plug-In** | Highest EV/complexity ratio; already the project’s proven control surface with strong governance fit. |
| **R2 DIRECTION filter** | `DIR_LONG`/`DIR_SHORT` composed with signal filters | Same as R1; no extra schema | Same as R1 + directional asymmetry robustness | **Low** | **Plug-In** | Same low blast radius as R1 and directly compatible with current architecture. |
| **R3 POSITION-SIZE modifier** | `risk_manager.py` path + planned combiner surfaces | Continuous pre-ORB forecasts, lane-level sizing state, correlation-aware inputs | Additive-vs-baseline test + WF + drawdown convexity checks + anti-leverage guards | **Medium** | **Adapt** | Strong upside but requires controlled forecast stack and strict anti-overfit sizing discipline. |
| **R4 STOP modifier** | Existing stop-policy knobs in execution path | Pre-entry level-distance/signal-strength metadata + explicit stop-policy provenance | Counterfactual replay + slippage sensitivity + payoff-shape stability checks | **Medium-High** | **Adapt** | Useful but changes loss distribution directly; must prove no hidden fragility. |
| **R5 TARGET modifier** | Target computation path (currently fixed-RR centric) | Reliable next-level map, level quality metadata, path-dependent target logic | Full re-simulation + anti-lookahead proof + target fill realism audit | **High** | **Reject** | No sufficient `/resources`-level mechanism grounding for level-target doctrine; highest story-fit risk. |
| **R6 ENTRY-MODEL switch (E1/E2)** | Existing E2 conventions in schema/validator stack | Entry-mode-safe timing fields, schema support, friction-aware semantics | Paired fairness test (E1 vs E2) with matched costs and OOS survival | **High** | **Adapt** | Promising but easy to contaminate with timing leakage; requires schema and replay hardening. |
| **R7 CONFLUENCE score** | Composite feature build + forecast-combiner pathway | Harmonized pre-ORB features, calibrated forecast scale, correlation controls | Component validity → monotonicity → OOS utility vs simpler proxies | **Medium-High** | **Adapt** | Valuable only if it demonstrably beats simpler alternatives after correlation-aware controls. |
| **R8 PORTFOLIO allocator weight** | Existing lane allocation surfaces | Daily confluence summaries, lane utility estimates, correlation/turnover controls | Portfolio WF + turnover drag + concentration/correlation stress tests | **High** | **Adapt** | Should be downstream of proven R3/R7 primitives, not used to mask weak component signals. |

---

## Optimal sequencing (project-aligned)

1. **Phase A — Immediate Plug-In path:** R1, R2
2. **Phase B — Controlled adaptation path:** R3 + R7 together (shared combiner/sizing logic)
3. **Phase C — Execution-shape adaptations:** R4 then R6 (replay-first)
4. **Phase D — Portfolio adaptation:** R8 only after B/C survive
5. **Rejected in current cycle:** R5 until explicit new grounding is added to `/resources` or canonical institutional extracts

---

## Final classification snapshot

- **Plug-In:** R1, R2
- **Adapt:** R3, R4, R6, R7, R8
- **Reject:** R5

This is the highest-confidence ordering under current repo doctrine: fail-closed, pre-registered testing, and no deployment claims from ungrounded mechanism stories.
