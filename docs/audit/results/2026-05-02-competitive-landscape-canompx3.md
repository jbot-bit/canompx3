# canompx3 vs peer trading frameworks (external benchmark)

Date: 2026-05-02
Scope: Honest external comparison against actively used open-source trading frameworks.

## Outcome (plain English)

- You have **not** wasted your time.
- This is **salvageable and high-upside**, but only if you focus on a short list of compounding fixes.
- If you keep spreading effort across low-EV audits and infrastructure churn, expected ROI drops quickly.

## Decision (continue vs quit)

Use this as a hard gate:

- **Continue** if you will commit to the top 3 EV items below in sequence and time-box them.
- **Pause/quit** if you are unwilling to enforce those gates and keep operating in diffuse "analysis without closure" mode.

## Method

- Internal claims treated as **MEASURED** only when grounded in repository artifacts from this checkout.
- External claims treated as **MEASURED** when grounded in official project docs/repos fetched during this session.
- Any causal/explanatory interpretation labeled **INFERRED**.
- Anything not grounded in local docs + fetched primary sources labeled **UNSUPPORTED**.

## Peers sampled (external)

1. QuantConnect LEAN
2. NautilusTrader
3. Freqtrade
4. VectorBT
5. Backtrader

Why these: they collectively cover the main nearby design space (research-heavy engines, research-to-live engines, vectorized research stacks, and retail/live bot platforms).

## Snapshot of what peers emphasize (MEASURED)

- **LEAN**: event-driven, modular, built for backtesting and live trading, with CLI surface (`lean backtest`, `lean live`, `lean research`) and broad plugin model.
- **NautilusTrader**: deterministic event-driven architecture, explicit research-to-live parity, multi-venue/multi-asset support, and strong supply-chain/provenance messaging.
- **Freqtrade**: very broad retail operator surface (backtesting, hyperopt, webserver/web UI, exchange tooling), crypto-first workflow.
- **VectorBT**: very fast vectorized research and parameter sweeps (NumPy/Numba + optional Rust engine), excellent for idea throughput.
- **Backtrader**: long-standing event-driven + vectorized hybrid with live broker support and broad built-in analysis/indicator surface.

## canompx3 strengths (MEASURED)

1. **Research governance is unusually strict for an open repo**.
   - Pre-registered criteria, multiple-testing discipline, holdout enforcement, and doctrine-driven research boundaries are first-class citizens.
2. **Strong fail-closed architecture mindset**.
   - Core docs and code repeatedly enforce abort-on-invalid-state behavior rather than silently continuing.
3. **Microstructure- and instrument-aware design**.
   - Session logic, DST handling, contract-era boundaries, and explicit per-instrument cost specs are unusually explicit.
4. **Clear research/live separation doctrine**.
   - Canonical layers for truth-finding and explicit bans on derived-layer contamination reduce accidental self-deception.
5. **Operationally explicit live controls**.
   - Lane allocation, holdout policy, and runtime policy checks exist as concrete modules rather than tribal knowledge.

## canompx3 weaknesses (MEASURED + INFERRED)

1. **Operational fragility in local environment setup** (**MEASURED**).
   - Current session preflight reports missing `.venv-wsl` interpreter and pulse reports missing `gold.db`.
2. **High cognitive load / governance overhead** (**INFERRED**).
   - The doctrine depth is a strength, but onboarding and contributor throughput likely suffer without a thinner happy path.
3. **Single-repo dependency concentration** (**MEASURED**).
   - Important runtime and research state depend heavily on local artifacts and strict process; when one piece is missing, many checks degrade.
4. **Narrow ecosystem surface vs larger peers** (**INFERRED**).
   - Compared with LEAN/Freqtrade/Nautilus, there is less evidence of a large adapter/plugin/operator ecosystem for quick external adoption.
5. **Some institutional controls are declared but currently incomplete in this checkout** (**MEASURED**).
   - Example: pulse indicates Criterion 11 survival report missing for active profile.

## What appears missing vs top peers

### 1) First-class bootstrap + health contract (MEASURED need)

A single command that guarantees:
- correct interpreter,
- required DB presence,
- minimum viable local data snapshot,
- and green preflight.

Without this, the excellent research doctrine is harder to execute consistently.

### 2) Standardized external benchmark harness (INFERRED need)

You should have a repeatable benchmark pack that runs your core strategy families against:
- naive baseline,
- simple trend baseline,
- and at least one peer-style backtest implementation contract.

Right now, rigor is high internally, but external comparability is weaker than it could be.

### 3) “Thin mode” operator workflow (INFERRED need)

Your institutional mode is strong. You likely also need a constrained quick mode for:
- fast hypothesis triage,
- safe no-commit dry-runs,
- and simplified status dashboards.

### 4) External trust packaging (INFERRED need)

Peers win trust partly via ergonomics: easy install, clear quickstarts, polished docs, and visible stable APIs.
For canompx3, there is strong truth-protocol depth, but less obvious external productization.

### 5) Live-readiness evidence bundle (MEASURED/INFERRED need)

A standardized artifact bundle per deploy candidate:
- latest survival Monte Carlo,
- SR/decay monitor state,
- current lane gate reasons,
- and last known-good replay hash.

Pieces exist, but packaging is not yet as turnkey as it could be.

## Direct answer: “Is this a joke?”

**No. Not remotely.**

- **MEASURED:** The project has unusually serious research guardrails for a personal/open trading stack.
- **INFERRED:** The current pain is not lack of seriousness; it is integration friction and operational complexity.

If your goal is “institutional-grade truth over convenience,” this project is directionally strong.
If your goal is “fast, low-friction strategy iteration like mainstream retail frameworks,” you are currently paying a large complexity tax.

## Priority recommendations (ordered)

1. **Stabilize bootstrap reliability first** (env + DB + preflight green path).
2. **Automate live-readiness bundle generation per profile.**
3. **Add a compact benchmark harness for external comparability.**
4. **Create a thin/operator mode for fast safe iteration.**
5. **Publish a concise architecture + onboarding map for new contributors.**

## Highest-EV ROI items (do these or stop)

### EV-1 — Make startup deterministic (highest ROI)

**Why:** Every failed/misaligned session burns research quality and velocity.

**Definition of done:**
- single bootstrap command creates expected interpreter + validates required DB paths,
- `session_preflight` passes without mutating-session blockers,
- `project_pulse --fast` returns no broken-category startup blockers.

**If this is not done first, other work is discounted.**

### EV-2 — Ship a one-command live-readiness report

**Why:** You need deploy/no-deploy truth without reading scattered files.

**Definition of done:**
- one command emits profile-level bundle:
  - Criterion 11 survival status,
  - Criterion 12 SR monitor status,
  - active lanes + paused/stale reasons,
  - most recent canonical rebalance timestamp and provenance.

**If this is missing, you are flying blind operationally.**

### EV-3 — Build a bounded external benchmark harness

**Why:** This converts "opinion" into measurable competitiveness.

**Definition of done:**
- fixed baseline suite (naive + simple trend + your core lane family),
- fixed period split and evaluation rubric,
- output artifact committed under `docs/audit/results/` with pass/fail interpretation.

**If this is not present, comparison with peers stays rhetorical.**

## 30-day checkpoint (kill-switch)

At day 30:

- If EV-1 and EV-2 are complete and EV-3 is in progress with real artifacts: **continue**.
- If EV-1 is still unstable or EV-2 is absent: **stop/pause and simplify scope**.

## Faster decision cadence (do not wait 30 days)

## One-row execution plan (can be done in one concentrated sprint)

Yes — most of this can be done sequentially in one run instead of waiting.

### 48-hour sequence (institutionally grounded)

1. **Hour 0-6: EV-1 plumbing hardening**
   - fix interpreter + DB path + preflight blockers until green.
   - proof artifact: preflight + pulse outputs committed to `docs/audit/results/`.
2. **Hour 6-14: EV-2 live-readiness command**
   - implement one command that emits C11/C12 + lane-state + provenance bundle.
   - proof artifact: one reproducible report for active profile.
3. **Hour 14-24: EV-3 bounded benchmark harness**
   - implement fixed-baseline comparison harness (naive/trend/core lane family).
   - proof artifact: first benchmark report with fixed rubric.
4. **Hour 24-30: anti-bias pass**
   - run pre-reg/FDR/holdout/mechanism/sensitivity checklist on artifacts.
   - mark any unresolved claim as `UNSUPPORTED`.
5. **Hour 30-36: disconfirming audit**
   - force a "how this could be wrong" section per artifact.
   - include failed controls and negative findings.
6. **Hour 36-48: decision cut**
   - if EV-1+EV-2 are green and EV-3 produced artifact: continue.
   - else: reduce scope immediately (do not continue diffuse work).

This sequence is consistent with repo doctrine: finite-data discipline,
multiple-testing controls, holdout protection, and mechanism-first interpretation.

### Day 3 gate (plumbing truth)

- `session_preflight` mutating blockers = 0
- `project_pulse --fast` broken startup blockers = 0
- If either fails at day 3: pause all non-plumbing research work and fix EV-1 first.

### Day 7 gate (operational truth)

- one-command live-readiness bundle exists and is reproducible (EV-2)
- bundle includes C11/C12 status + lane pause/stale reasons + rebalance provenance
- if missing at day 7: no new strategy promotion work

### Day 14 gate (research truth)

- bounded external benchmark harness emits first artifact (EV-3)
- artifact includes naive/trend/core-lane baselines under fixed rubric
- if missing at day 14: freeze peer-comparison claims as UNSUPPORTED

## Anti-bias and silence-filling protocol (literature-grounded)

To avoid self-deception and "silent gaps," force every EV item through these controls:

1. **Pre-register hypotheses and trial count** before running new scans
   (Criterion 1/2; Bailey MinBTL discipline).
2. **Control multiple testing** with BH/FDR for family searches; use strict
   t-threshold logic per criterion pathway.
3. **Separate discovery from confirmation** (sacred holdout policy; no leakage
   from protected windows).
4. **Require mechanism statements** for promoted claims (theory-first, not just
   numeric lift).
5. **Publish disconfirming evidence** (what failed, not only what passed) in the
   same artifact as positive findings.
6. **Run sensitivity checks** (threshold perturbation / parameter drift) before
   accepting a filter as robust.

## Repository resources to ground execution

- `docs/institutional/pre_registered_criteria.md` (Criteria 1-12, locked gates)
- `docs/institutional/finite_data_framework.md` (finite-sample constraints)
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md`
- `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`
- `docs/institutional/literature/harvey_liu_2015_backtesting.md`
- `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md`
- `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md`
- `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md`
- `RESEARCH_RULES.md` (operating doctrine + anti-bias constraints)

## Institutional-grade iteration program (more planning + research + design)

If you want deeper iteration (your latest request), run this as 4 tight cycles:

### Cycle A — Research expansion (fetch + disconfirm first)

- Expand peer set to include one additional research engine and one production
  execution stack.
- For each peer: capture architecture, verification model, live controls,
  and known failure modes from primary docs.
- Start with disconfirming questions:
  - "Where is canompx3 weaker than this peer?"
  - "What claim in our current report fails under their model?"

### Cycle B — Design synthesis

- Translate findings into 3 concrete design proposals:
  1. bootstrap contract design,
  2. live-readiness bundle contract,
  3. benchmark harness contract.
- Each design proposal must include blast radius, dependencies, failure modes,
  and rollback path.

### Cycle C — Bounded implementation pass

- Implement only the smallest version of EV-1/EV-2/EV-3 that can produce
  measurable artifacts.
- No "big rewrite" allowed; every pass must end with a measurable report.

### Cycle D — Adversarial audit pass

- For every claimed improvement, run an explicit adversarial review:
  - what evidence would falsify this claim?
  - did we test that evidence?
  - if not tested, mark `UNSUPPORTED`.

## No-gaps / no-silences reporting contract

To avoid hidden bias or silent omissions, every cycle artifact must include:

1. **What was tested**
2. **What was not tested (and why)**
3. **What failed**
4. **What remains unsupported**
5. **What would change the decision**

If any section is missing, the artifact is incomplete.

## Concrete execution plan (agent-owned, not user-owned)

You asked for **us** to do the research and drive execution. This is the
operator plan to run now, in order.

### Phase 1 — Research capture (same day)

1. Fetch and pin primary-source evidence from peers (LEAN, NautilusTrader,
   Freqtrade, VectorBT, Backtrader) into one comparison appendix.
2. Add a "capability parity map" table: what they have vs what canompx3 lacks,
   with impact score and implementation complexity.
3. Write explicit disconfirming notes: where canompx3 is worse and why.

**Deliverable:** `docs/audit/results/2026-05-03-peer-capability-parity-map.md`

### Phase 2 — Design package (same day)

4. Draft three implementation specs (not just ideas):
   - bootstrap reliability contract (EV-1),
   - live-readiness bundle contract (EV-2),
   - bounded benchmark harness contract (EV-3).
5. Each spec must include:
   - inputs/outputs,
   - source-of-truth files,
   - failure modes,
   - verification commands,
   - rollback plan.

**Deliverable:** `docs/plans/2026-05-03-ev1-ev2-ev3-implementation-specs.md`

### Phase 3 — Build + verify (next 1-2 days)

6. Implement EV-1 minimal slice and verify with `session_preflight` +
   `project_pulse --fast`.
7. Implement EV-2 one-command bundle and generate first profile artifact.
8. Implement EV-3 bounded benchmark harness and generate first benchmark report.

**Deliverables:** code + artifacts in `docs/audit/results/` with pass/fail.

### Phase 4 — Adversarial closeout (same day as build completion)

9. Run anti-bias review across all three deliverables (pre-reg/FDR/holdout/
   mechanism/sensitivity/disconfirming evidence).
10. Produce one go/no-go memo with unresolved risks labeled `UNSUPPORTED`.

**Deliverable:** `docs/audit/results/2026-05-0X-ev-program-go-no-go.md`

## Immediate next command set (agent runbook)

1. Research pull + parity map
2. Spec writing for EV-1/2/3
3. EV-1 implementation + verification
4. EV-2 implementation + verification
5. EV-3 implementation + verification
6. Adversarial memo + decision cut

## Sources used (external primary sources)

- QuantConnect LEAN repo: https://github.com/QuantConnect/Lean
- NautilusTrader repo: https://github.com/nautechsystems/nautilus_trader
- NautilusTrader architecture note: https://nautilustrader.io/blog/why-nautilustrader-exists/
- Freqtrade repo: https://github.com/freqtrade/freqtrade
- VectorBT repo: https://github.com/polakowo/vectorbt
- Backtrader features/docs: https://www.backtrader.com/home/features/

## Internal evidence references

- `RESEARCH_RULES.md`
- `docs/institutional/pre_registered_criteria.md`
- `trading_app/holdout_policy.py`
- `pipeline/cost_model.py`
- `pipeline/asset_configs.py`
- `pipeline/dst.py`
- `scripts/tools/session_preflight.py` (run output in this session)
- `scripts/tools/project_pulse.py --fast --format json` (run output in this session)
