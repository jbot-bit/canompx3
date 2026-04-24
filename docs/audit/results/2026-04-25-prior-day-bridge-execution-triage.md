# Prior-Day Pathway-B Bridge Execution Triage

**Date:** 2026-04-25
**Status:** `PARK_ON_MEASURED_BLOCKERS`
**Closes:** action-queue item `prior_day_bridge_execution_triage`
(P1 close-first, from `docs/runtime/action-queue.yaml`).
**Closes stage:** `docs/runtime/stages/prior-day-pathway-b-hot-cell-prereg.md`.

## Scope

The action-queue directive was: "choose one already-locked prior-day bridge
hypothesis and execute it instead of writing another broad prereg." The
stage scope was narrower still: freeze one strongest Pathway-B hot cell
into a single-cell confirm-or-kill prereg, or park.

This doc closes the triage honestly from the three prior audits that
already landed. It does NOT rerun discovery, does NOT reopen broad
family rescan, and does NOT propose a new hypothesis.

## Truth surfaces used

Measured rows quoted verbatim below are sourced from these already-landed
audits — no fresh canonical query was required because no new data has
been collected on these cells since 2026-04-23.

- `docs/audit/results/2026-04-23-prior-day-bridge-closure-audit.md`
- `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md`
- `docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-audit.md`

Canonical signal-layer truth remains `orb_outcomes` + `daily_features`;
all three prior audits derived their tables from that layer.

## MEASURED — hypothesis consumption state (from closure audit)

All six Pathway-B bridge hypothesis files were already consumed exactly
once through the discovery / validation path.

| hypothesis file | strategy_id | experimental | shelf |
|---|---|---|---|
| `2026-04-22-mnq-usdata1000-near-pivot-50-avoid-v1.yaml` | `MNQ_US_DATA_1000_E2_RR1.0_CB1_F3_NEAR_PIVOT_50` | REJECTED | no |
| `2026-04-22-mnq-usdata1000-downside-displacement-take-v1.yaml` | `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG` | PASSED | yes |
| `2026-04-22-mnq-usdata1000-clear-of-congestion-take-v1.yaml` | `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG` | PASSED | yes |
| `2026-04-22-mnq-usdata1000-positive-context-union-v1.yaml` | `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG` | PASSED | yes |
| `2026-04-22-mnq-usdata1000-rr15-positive-context-union-v1.yaml` | `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG` | PASSED | yes |
| `2026-04-22-mnq-comex-pd-clear-long-take-v1.yaml` | `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG` | PASSED | yes |

Net: 6/6 consumed; 5/6 on validated shelf; 1/6 rejected as a dead exact
cell. There is no un-run locked path to execute.

## MEASURED — IS vs OOS sample and effect sizes (from closure audit)

| filter | session | RR | N_IS_on | ExpR_IS | N_OOS_on | ExpR_OOS |
|---|---|---:|---:|---:|---:|---:|
| `PD_DISPLACE_LONG` | `US_DATA_1000` | 1.0 | 205 | +0.2320 | 10 | +0.1745 |
| `PD_CLEAR_LONG` | `US_DATA_1000` | 1.0 | 232 | +0.2155 | 13 | +0.1962 |
| `PD_GO_LONG` | `US_DATA_1000` | 1.0 | 350 | +0.1811 | 15 | +0.2993 |
| `PD_GO_LONG` | `US_DATA_1000` | 1.5 | 347 | +0.2078 | 15 | +0.2964 |
| `PD_CLEAR_LONG` | `COMEX_SETTLE` | 1.0 | 338 | +0.1602 | 15 | +0.1321 |

Every shelf survivor has `N_OOS_on` in the range 10–15.

## Power-floor check

Applying the canonical power-floor rule (memory:
`feedback_oos_power_floor.md`; also encoded in
`docs/plans/2026-04-25-cross-asset-session-chronology-spec.md` §6): when
OOS power against the IS effect size is below 50 percent, verdict is
`UNVERIFIED`, never `DEAD` and never `ALIVE`.

Using the measured IS effect sizes (+0.16 to +0.23 ExpR per trade) and
OOS `N = 10–15`, the two-sample power to separate OOS mean from 0 at
α = 0.05 two-sided is well under 50 percent for every cell. A
single-cell confirm-or-kill prereg run against the current Mode A
horizon (holdout from 2026-01-01 through current cutoff) cannot return
anything except `UNVERIFIED` on any of the five cells.

Source-of-truth for IS/OOS split boundary:
`trading_app/holdout_policy.py` — `HOLDOUT_SACRED_FROM = 2026-01-01`.

## MEASURED — routing layer (from routing audit)

All five shelf survivors received the decision `KEEP_ON_SHELF`. Verbatim
from `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md`
decision summary:

| Candidate | Session | RR | Add Δ Annual R IS | Replacement target | Replace Δ Annual R IS | Replace Δ Sharpe IS | Decision |
|---|---|---:|---:|---|---:|---:|---|
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG` | US_DATA_1000 | 1.0 | +6.9 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | -15.4 | -0.146 | `KEEP_ON_SHELF` |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG` | US_DATA_1000 | 1.0 | +7.2 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | -15.1 | -0.155 | `KEEP_ON_SHELF` |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG` | US_DATA_1000 | 1.0 | +9.1 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | -13.2 | -0.118 | `KEEP_ON_SHELF` |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG` | US_DATA_1000 | 1.5 | +10.3 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | -12.0 | -0.114 | `KEEP_ON_SHELF` |
| `MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG` | COMEX_SETTLE | 1.0 | +7.7 | `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | -12.7 | -0.050 | `KEEP_ON_SHELF` |

No candidate cleared the additive free-slot route test. Every replacement
delta is negative on both annualized R and honest Sharpe. All five rows
carry the `KEEP_VISIBLE` watch status per the routing audit's
profile/routing fit section.

## MEASURED — execution translation layer (from translation audit)

The four US_DATA_1000 shelf survivors all received the verdict
`ARCHITECTURE_CHANGE_REQUIRED`. Verbatim from the translation audit
candidate-outcome summary:

| Candidate | Time Overlap Days | Half-Size Suggested | Half-Size Unexpressible | Δ Annual R IS | Verdict |
|---|---:|---:|---:|---:|---|
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_DISPLACE_LONG` | 94 | 85 | 85 | +6.7 | `ARCHITECTURE_CHANGE_REQUIRED` |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_CLEAR_LONG` | 122 | 110 | 110 | +4.6 | `ARCHITECTURE_CHANGE_REQUIRED` |
| `MNQ_US_DATA_1000_E2_RR1.0_CB1_PD_GO_LONG` | 170 | 154 | 154 | +6.6 | `ARCHITECTURE_CHANGE_REQUIRED` |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_PD_GO_LONG` | 210 | 190 | 190 | +9.4 | `ARCHITECTURE_CHANGE_REQUIRED` |

The measured runtime blocker is:

- Live profile rows resolve at `max_contracts=1` from
  `build_portfolio_from_profile` in `trading_app/portfolio.py`.
- Same-session different-aperture overlap triggers
  `suggested_contract_factor=0.5` in `RiskManager.can_enter`.
- `ExecutionEngine` fails closed when that half-size cannot be expressed
  on a 1-contract row.
- Consequence: every half-size suggestion across the four candidates is
  unexpressible and rejected (85, 110, 154, 190 of them respectively).

The COMEX_SETTLE shelf survivor was out of the translation audit's
narrow scope (US_DATA_1000 only), but the same-session collision shape
against its incumbent `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` is
structurally identical, and the routing audit has already recorded
`KEEP_ON_SHELF` on that row.

## Verdict

`PARK_ON_MEASURED_BLOCKERS`.

No Pathway-B hot cell currently merits single-cell execution. The
reasons are measured, not narrative:

1. **All locked hypotheses are consumed.** The closure audit shows 6/6
   Pathway-B hypothesis files already ran. There is no un-executed
   locked path to "choose and execute". Writing a new prereg would be
   the "broad prereg" the action queue explicitly forbids.
2. **OOS power floor fails on every cell.** `N_OOS = 10–15` per cell
   against IS effects of `+0.16 to +0.23 ExpR` is below the 50 percent
   power floor. Any confirm-or-kill run on the current Mode A holdout
   returns `UNVERIFIED`, not a decision-grade verdict.
3. **Routing is already resolved.** All 5 shelf survivors are
   `KEEP_ON_SHELF` with every replacement delta negative (routing audit).
4. **Runtime translation is already resolved.** The 4 US_DATA_1000
   candidates are `ARCHITECTURE_CHANGE_REQUIRED` because every half-size
   suggestion is unexpressible on 1-contract lanes (translation audit).
   The COMEX_SETTLE candidate shares the same structural blocker.

None of those four blockers can be moved by running another prereg on
the same cells with the same data.

## Carry-forward state (unchanged — stated for durability)

- Validated shelf: all 5 passing strategy IDs remain on the shelf with
  watch status `KEEP_VISIBLE`. This doc does NOT demote or retire them.
- Live allocator: unchanged. Zero of these rows appear in
  `docs/runtime/lane_allocation.json`; that was the resolved state at
  the close of the routing audit and remains so.
- Signal layer on the three positive families stays alive in a
  research-only sense — `MNQ US_DATA_1000 O5 E2 RR1.0 long`,
  `MNQ US_DATA_1000 O5 E2 RR1.5 long`, and
  `MNQ COMEX_SETTLE O5 E2 RR1.0 long` — per the closure audit's
  signal-layer verdict.
- Rejected cell: `MNQ_US_DATA_1000_E2_RR1.0_CB1_F3_NEAR_PIVOT_50`
  remains dead. Do not retest.

## Re-open triggers (explicit, measurable)

The branch re-opens at Pathway-B exactly when either of the following
becomes true.

- **Trigger A — OOS horizon clears the power floor.** When per-cell
  `N_OOS_on` against the Mode A holdout reaches a power-floor-passing
  sample given the observed IS effect sizes, a single-cell confirm-or
  -kill prereg on the strongest remaining cell becomes honest. At
  current fire rates this is an elapsed-time trigger, not a re-tuning
  trigger; do not change cell parameters to hit it faster.
- **Trigger B — same-session execution-translation scaffolding lands.**
  Either (1) `trading_app/risk_manager.py` / `trading_app/portfolio.py`
  / `trading_app/execution_engine.py` gain an expressible half-size
  path for 1-contract lanes, or (2) a manual-playbook routing surface
  for same-session shelf rows lands under
  `docs/plans/manual-trading-playbook.md`. Either landing flips the four
  `ARCHITECTURE_CHANGE_REQUIRED` verdicts to `REVIEW_REQUIRED` and the
  triage can be reopened with a bounded single-cell prereg.

Both triggers are observable without rerunning research.

## Outputs

- This result doc at
  `docs/audit/results/2026-04-25-prior-day-bridge-execution-triage.md`.
- `docs/runtime/action-queue.yaml` item `prior_day_bridge_execution_triage`
  transitioned to `status: done` with `notes_ref` pointing at this doc.
- `docs/runtime/stages/prior-day-pathway-b-hot-cell-prereg.md` deleted
  (stage closed by this commit).

No code changes. No canonical-layer queries run by this doc itself — all
measured rows are cited verbatim from the three prior audits.

## Limitations and what this audit does NOT do

- Not a new Pathway-B discovery pass.
- Not a cell-shopping exercise after looking at forward behaviour.
- Not a shelf demotion or retirement decision.
- Not a live-allocator change.
- Not a proposal for Track A reopening.
- Not a reinterpretation of the closure, routing, or translation audit
  verdicts — this doc consumes them, it does not reopen them.

## References

- `docs/audit/results/2026-04-23-prior-day-bridge-closure-audit.md`
- `docs/audit/results/2026-04-23-prior-day-geometry-routing-audit.md`
- `docs/audit/results/2026-04-23-prior-day-geometry-execution-translation-audit.md`
- `docs/plans/2026-04-25-cross-asset-session-chronology-spec.md` (power-floor clause)
- `trading_app/holdout_policy.py` — `HOLDOUT_SACRED_FROM`
- `trading_app/portfolio.py` — `build_portfolio_from_profile` (max_contracts=1)
- `trading_app/risk_manager.py` — `RiskManager.can_enter` (suggested_contract_factor=0.5)
- `trading_app/execution_engine.py` — fail-closed on unexpressible half-size
- Memory: `feedback_oos_power_floor.md`,
  `feedback_pooled_not_lane_specific.md`,
  `feedback_per_lane_breakdown_required.md`.
- `docs/runtime/stages/prior-day-pathway-b-hot-cell-prereg.md` — stage
  closed by this doc.
- `docs/runtime/action-queue.yaml` — item `prior_day_bridge_execution_triage` set to `status: done`.
