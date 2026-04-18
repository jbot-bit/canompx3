# Deferred phases — design docs requiring user approval before execution

**Generated:** 2026-04-19 (overnight session)
**Status:** DESIGN — not yet approved for execution. All four phases touch production code (`pipeline/`, `trading_app/`, `scripts/`) which per CLAUDE.md Design Proposal Gate requires user sign-off before any code edit.

Parent plan: the 14-phase overnight session remediation plan.

## Scope bundle

Phases 10, 11, 12, 13 from the original 14-phase plan are bundled here because they all share:
- Production-code impact (`pipeline/` or `trading_app/`)
- Non-trivial blast radius requiring stage-gate
- User-explicit-approval required by institutional-rigor.md Rule 1
- NOT shippable overnight without user awake to review

Each design is presented with: goal, scope, files, blast radius, approach, acceptance criteria, risk, effort estimate. No code was written for any of these phases.

---

## Phase 10 — E_RETEST entry model implementation

### Goal

Add a new `entry_model='E_RETEST'` to `orb_outcomes` + outcome builder. Entry fires on LIMIT-AT-LEVEL after a failed first break — a different mechanism than the stop-market-fade tested (and killed) on 2026-04-15 (`docs/audit/results/2026-04-15-t0-t8-adversarial-fade-audit.md`). Literature ground: Chan Ch 7 stop-cascade mechanism + Fitschen Ch 3 intraday trend-follow.

### Scope

- Parent pre-reg stub: `docs/audit/hypotheses/phase-c-e-retest-entry-model.md` (2-3 weeks infra per stub).
- New entry rule: after an ORB break in direction D at bar B, if price returns to the break level at any later bar within the session, fire a LIMIT order at the break level for direction D.
- Compatible with existing filter model (COST_LT, ORB_Gn, VWAP_MID_ALIGNED etc. apply unchanged).
- New pre-reg family scan testing E_RETEST across top validated MNQ/MES lanes.

### Files (production)

- `trading_app/outcome_builder.py` — new entry-model handler (writes rows to `orb_outcomes` with `entry_model='E_RETEST'`). CORE file.
- `trading_app/config.py` — register `E_RETEST` in `ENTRY_MODELS` constant (or equivalent).
- `pipeline/init_db.py` — NO schema change (entry_model column already `VARCHAR`, values not constrained).
- `tests/test_trading_app/test_outcome_builder.py` — new test cases covering E_RETEST entry firing rules.

### Files (research + docs)

- New pre-reg YAML locking E_RETEST family scan.
- New result doc after scan executes.

### Blast radius

**Up (what imports the touched files):**
- `pipeline/run_full_pipeline.py::step_build_outcomes` — unchanged (runs via subprocess).
- `trading_app/strategy_discovery.py` — expects `entry_model` values from `ENTRY_MODELS`; adding `E_RETEST` broadens the grid.
- `trading_app/strategy_validator.py` — same.
- Any script iterating `orb_outcomes` by `entry_model` — will see new rows tagged `E_RETEST` once outcome builder re-runs.

**Down (what the touched files depend on):**
- `bars_1m` — E_RETEST needs bar-level data beyond the break bar to detect retest. Already the bar source for E2; no new dependency.
- `pipeline.dst.orb_utc_window` — uses existing canonical window computation.

### Approach

4-stage implementation per stage-gate-protocol.md:

1. **Stage A — schema & config entries** (0.5 day): Add E_RETEST to ENTRY_MODELS. No DB schema change. Tests updated but no rows produced yet.
2. **Stage B — outcome builder E_RETEST handler** (1 week): Implement retest detection, limit-order-fill simulation at break level. Tests for (i) normal retest → fill → outcome, (ii) no retest in session → no row, (iii) retest but target/stop hit before next-bar confirm → correct outcome. Backfill for 1 test symbol × 1 session to validate shape.
3. **Stage C — full backfill** (1-2 days compute): Run outcome builder E_RETEST for all active instruments × all sessions × all RRs across full IS + OOS. ~200k new rows expected.
4. **Stage D — pre-reg and scan** (1 week): Draft, lock, run pre-reg family scan on E_RETEST across top MNQ/MES lanes (K=6-10). Pathway A Chan Ch 7 stop-cascade + Fitschen Ch 3 intraday-trend.

Total: ~3 weeks.

### Acceptance criteria (Stage B signoff)

- Unit tests pass (new + existing)
- Hand-traced example: pick 3 specific trading_days with known ORB patterns; verify E_RETEST outcomes match expected limit-at-level entries
- Drift check passes
- No changes to E0/E1/E2/E3 outcomes

### Risk

- **HIGH:** retest detection logic has edge cases (what if price wicks through level without tagging? what if retest + break-of-break on same bar?). Requires careful spec before code.
- **MEDIUM:** 200k new rows on `orb_outcomes` means DB size grows ~5%. Verify backup + check_drift handles.
- **LOW:** downstream scans that iterate `entry_model` may unintentionally include E_RETEST where they expect E2 — one-shot search-and-filter audit required.

### Effort

~3 weeks implementation + 1 week pre-reg + 1 week scan = **5 weeks total**. Not overnight-compatible.

### Not in this design

- Short-direction E_RETEST — deferred to after long-direction validated.
- Limit-with-slippage model variants — stub only, future pre-reg.
- E_RETEST on non-ORB sessions — not in scope.

---

## Phase 11 — Carver continuous-sizing pilot on GARCH forecast

### Goal

Test Carver Ch 9-10 continuous position sizing (Kelly-linked) on GARCH vol forecast as the sizing signal. GARCH-as-binary-filter is exhausted (A4a/A4b/A4c parked); Carver continuous sizing is a different hypothesis. Not deployment, not live capital — a research pilot.

### Scope

- Parent spec stub: `docs/audit/hypotheses/phase-d-carver-forecast-combiner.md`.
- Implement continuous multiplier = f(garch_forecast_vol_pct) per Carver Ch 10 Kelly-linked formulation.
- Backtest on MNQ TOKYO_OPEN (Carver pilot lane stub).
- Evaluate Sharpe vs. binary fixed-size baseline.

### Files (research, NOT production)

- New script `research/carver_garch_continuous_sizing_pilot.py`.
- New pre-reg YAML + result doc.

### Files (potentially production — DEFER)

If pilot shows material uplift, subsequent stage would touch `trading_app/lane_allocator.py` or a new `trading_app/forecast_combiner.py`. NOT in this pilot.

### Blast radius

**Research-only pilot:** zero production impact. No allocator change. No deployed capital.

### Approach

1. Draft pre-reg with explicit single-signal scope (GARCH only, no combined forecasts yet).
2. Lock pre-reg.
3. Run pilot on MNQ TOKYO_OPEN + compare to binary baseline.
4. Write result doc. If KILL → file as negative evidence. If CONTINUE → follow-up pre-reg combining 2+ signals (this is where Carver's forecast-combiner framework kicks in).

### Acceptance criteria

- Pilot result doc reports: Sharpe_continuous, Sharpe_binary, uplift, t-test on per-trade return differences.
- Pathway A citation: Carver Ch 9 (volatility-targeted sizing) + Ch 10 (forecast combination) + GARCH forecast literature from Harvey-Liu 2015 or equivalent.

### Risk

- **LOW (research-only):** any result is informative; no live capital at risk.
- **MEDIUM (methodology):** implementing Carver's exact Kelly-linked formula needs careful reading of Ch 10 (already extracted in `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md`). Easy to get the scaling wrong.

### Effort

- Pre-reg: 1-2 hours.
- Pilot script: 4-6 hours.
- Scan + result doc: 2-3 hours.
- Total: **1-2 days research effort**.

### Not in this design

- Combined forecast (GARCH + another signal) — separate pre-reg after single-signal Carver proves.
- Deployment to allocator — separate production-code phase after multi-signal Carver validates.

---

## Phase 12 — Allocator maximand review + lane-correlation-gate re-scope

### Goal

Audit whether `trading_app.lane_allocator` should keep its current objective (max total R across lanes) or switch to capital-efficiency-weighted (max per-R-$-efficiency or max Sharpe-weighted). Also revisit whether the uniform ρ > 0.80 lane-correlation gate should be scoped to same-session only vs. cross-session differently.

### Scope

DESIGN-ONLY at this stage. Produce a decision document comparing:
- Current: max Total R across lanes (implicit objective)
- Alternative A: max per-R-$-efficiency (favors capital-efficient lanes even if lower total R)
- Alternative B: max Sharpe-weighted score (favors risk-adjusted return)
- Alternative C: hybrid (Total R subject to per-R floor)

### Files (DESIGN — not editing yet)

- DESIGN DOC: `docs/plans/2026-04-XX-allocator-maximand-design.md` (this file's descendant)

### Files (IF implementation approved — separate stage)

- `trading_app/lane_allocator.py` — allocator selection function
- `trading_app/lane_correlation.py` — scope-aware correlation gate

### Blast radius (if implemented)

**Production-CORE-critical:** allocator selects which lanes deploy. Changing the maximand changes the live book. Every downstream (`ACCOUNT_PROFILES`, fitness tracker, shadow recorder) reads allocator output. Stage-gate required.

### Approach

1. **Step 1 (audit):** Read `lane_allocator.py` carefully. Document the current maximand implementation exactly.
2. **Step 2 (alternatives):** For each alternative (A/B/C), compute on the 38 active validated_setups what would be selected. Compare.
3. **Step 3 (decision):** Present tradeoffs. User decides.
4. **Step 4 (implementation, IF approved):** Stage-gated implementation per institutional-rigor.md. Test-first.

### Acceptance criteria (for the design doc)

- Current maximand explicitly documented (code lines cited)
- Each alternative simulated on the 38-lane surface
- Tradeoff table: which lanes get selected under each objective
- Risk section for each alternative
- User-decision template for committee vote

### Risk

- **HIGH if implemented:** live-book selection changes. Any error affects capital deployment.
- **LOW if only design doc:** pure planning; no production impact.

### Effort

- Design doc: **~1 week** (careful audit of existing code + simulation of alternatives).
- Implementation (if approved): **~1 week** stage-gated implementation + testing.

### Not in this phase

- Multi-firm-scaling-aware allocator (future work — depends on Phase 13 cost model first).
- Kelly-sized per-lane allocation (future work).

---

## Phase 13 — Self-funded 1 NQ mini cost model + portfolio translation

### Goal

Add a canonical cost spec for 1 NQ mini (E-mini NASDAQ 100) to `pipeline.cost_model.COST_SPECS`. At self-funded broker scale, 1 NQ mini commissions are ~77% lower than 10 MNQ (per `memory/topstep_scaling_corrected_apr15.md`). Validate this against live broker documentation. Model portfolio EV shift when the 38 active MNQ lanes are sized in NQ rather than MNQ terms.

### Scope

- Add `NQ` cost spec to `COST_SPECS` with canonical broker-commission sourcing
- Compute current portfolio EV at MNQ scale
- Compute alternative portfolio EV at 1-NQ-mini-per-10-MNQ scale (where possible — some lanes may have risk-$ too small for NQ)
- Research doc with the comparison

### Files (production)

- `pipeline/cost_model.py` — new entry `NQ` in `COST_SPECS` dict
- `tests/test_pipeline/test_cost_model.py` — test new entry parses correctly

### Files (research)

- `research/nq_self_funded_cost_translation.py` — computes per-lane EV at NQ scale
- `docs/audit/results/2026-04-XX-nq-self-funded-cost-translation.md`

### Blast radius

**Production cost-spec addition:**
- Downstream: any code reading `COST_SPECS` (outcome builder, cost-ratio filters like COST_LT08/12). Adding a new entry is additive — existing MNQ/MES/MGC unchanged. Safe.
- If any existing lane is re-classified as "NQ-compatible" — that triggers portfolio-translation; deferred to post-design user decision.

### Approach

1. Source canonical broker commissions: TradeStation, Interactive Brokers, AMP, NinjaTrader canonical commission tables. Cite each.
2. Document NQ spec: point value, min tick, tick value, total commission (round-trip).
3. Add to `COST_SPECS`.
4. Test: existing cost-ratio filters on NQ (e.g., COST_LT08 on NQ) should compute correctly given the new spec.
5. Research script: simulate portfolio EV on NQ-scale where applicable.
6. Result doc: comparison table MNQ vs NQ EV per lane + dollar impact at 1 live account.

### Acceptance criteria

- COST_SPECS["NQ"] populated with canonical values + broker-source citation
- `tests/test_pipeline/test_cost_model.py` passes (new test + existing)
- Research doc shows per-lane EV diff MNQ → NQ

### Risk

- **LOW-MEDIUM:** production `cost_model.py` edit. If values are wrong, downstream cost-ratio filters mis-fire. Canonical broker citation + test mitigates.

### Effort

- Broker commission research + documentation: **0.5 day**
- Code edit + test: **0.5 day**
- Portfolio translation script + result doc: **0.5-1 day**
- Total: **~2 days**.

### Not in this phase

- Actually migrating any lane to NQ-sized deployment (live-trading decision, requires committee)
- Multi-instrument cost-spec variants (ES, RTY, CL self-funded equivalents)

---

## Summary table

| Phase | Type | Effort | Prerequisites | Risk if executed |
|---|---|---|---|---|
| 10 E_RETEST | Production code (outcome builder + schema) | 5 weeks | Chan Ch 7 extract (done Phase 9) | HIGH — new entry model touches writing path |
| 11 Carver GARCH sizing | Research pilot (research/ only) | 1-2 days | Carver Ch 9-10 extract (exists) | LOW — research-only |
| 12 Allocator maximand | Production code (lane_allocator) | 2 weeks | Phase 3 Mode A baselines (done) | HIGH — live book selection affected |
| 13 NQ cost model | Production code (cost_model) + research | 2 days | Broker commission research | LOW-MEDIUM — additive cost spec |

## Recommended priority order (user decides)

If any of these proceed beyond design, suggested order:

1. **Phase 13 (NQ cost model)** — cheapest, lowest risk, material impl-EV upside. ~2 days.
2. **Phase 11 (Carver GARCH pilot)** — research-only, informative regardless of outcome. ~2 days.
3. **Phase 12 (allocator review)** — design doc first to see if alternatives actually improve things. Only proceed to implementation if design doc shows material uplift. ~1-2 weeks.
4. **Phase 10 (E_RETEST)** — biggest undertaking, highest ambiguity on pattern-validation likelihood. Last. ~5 weeks.

## User decision requested

For each phase, pick one:
- **GO**: execute per the plan above. Stage-gate discipline applies.
- **DESIGN-ONLY**: produce a more detailed design doc (one level deeper than this) but do NOT touch code.
- **DEFER**: leave for a later session; no immediate action.
- **KILL**: remove from roadmap permanently (requires justification).

All four phases are independent — user can GO on any subset, DESIGN-ONLY on others, DEFER the rest.

This bundle closes Phase 19 of the overnight session task list (task #19 in the overnight TaskList).
