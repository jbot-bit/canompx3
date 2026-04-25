# MES Event-Tail Slippage Realism — Debt Scope

**Date:** 2026-04-25
**Status:** `OPEN_DEBT_SCOPED`
**Closes:** action-queue item `mes_event_tail_slippage_realism` (P3 debt).

The action-queue exit criterion was: "Event-tail realism is either
measured directly or explicitly left open with a bounded debt note."
This doc takes the "left open with a bounded debt note" path, with
explicit scope, escalation triggers, and a pre-registered method for
the measurement when it becomes warranted.

## Scope

Define what "MES event-tail slippage realism" means, what it does NOT
mean, and what would have to change before measurement becomes the
right move. No new measurement is run by this doc.

## Truth surfaces

- `docs/audit/results/2026-04-24-mes-e2-slippage-pilot-v1.md` — the
  routine-tendency pilot (`N=40`, 4 sessions, all `<= 1 modeled tick`,
  median `0`, p95 `0`).
- `docs/runtime/debt-ledger.md` — `cost-realism-slippage-pilot` entry
  with the existing line "STILL OPEN: event-day tail NOT measured for
  MNQ/MES routine samples (known-unknown, not refuted concern)."
- `pipeline/cost_model.py` — the modeled MES friction the routine pilot
  checked against. Canonical queried value at commit time
  (`get_cost_spec("MES")`):
  `tick_size = 0.25`, `slippage = 1.25` points = **5 ticks per side**,
  `spread_doubled = 1.25` points, `friction_in_points = 0.784`,
  `total_friction = 3.92` points round-trip, `point_value = $5.00`,
  `commission_rt = $1.42`, `min_ticks_floor = 10`.
- `research/databento_microstructure.reprice_e2_entry` — the canonical
  repricer the routine pilot delegates to.
- `trading_app/holdout_policy.py::HOLDOUT_SACRED_FROM` — Mode A holdout.

No canonical query is required by this doc beyond the cost-model
inlining above. No new MES live exposure has been introduced since the
routine pilot, so the existing evidence does not understate or overstate
against any new live state.

## What "event-tail" means here (definition)

An "event-tail" day for MES routine-fill slippage is any day on which
fill quality at the canonical E2 entry differs **structurally** from a
typical liquid-session day. The categories below enumerate the cases
that the routine `N=40` pilot does NOT cover:

1. **Calendar-event days.** FOMC statements, NFP, CPI / PPI / retail
   sales, OPEX, Fed Chair speeches, holiday-shortened sessions,
   month-end / quarter-end roll days. Fill behaviour on these days is
   plausibly worse because the spread widens and the order book thins.

2. **Macro-shock days.** Surprise central-bank actions, geopolitical
   shocks, large overnight equity gaps, large overnight currency moves.
   These are unpredictable in advance but identifiable post hoc by
   range, gap, or volume Z-score.

3. **Liquidity-degraded days.** Databento-flagged degraded windows,
   heartbeat / feed gap recovery, exchange-side anomalies. The
   2026-04-24 pilot already saw one degraded-day warning (2025-11-28)
   that repriced cleanly — that is one observation, not a tail sample.

4. **Microstructure-burst windows.** Gap-down + reversal, fast-market
   stop runs, news-spike fade. These occur within otherwise normal
   sessions and are not isolatable by a daily filter.

The **routine pilot** measured a 40-day, all-clean, calendar-routine
window. Its evidence applies only to days drawn from that distribution.

## What this debt is NOT

- Not a claim that MES routine slippage is wrong. The routine pilot
  passed; modeled friction is conservative for routine days (measured
  median 0 ticks vs modeled 5 ticks/side).
- Not a claim that event-tail behaviour is necessarily worse. It is
  **unmeasured**, not **adverse**.
- Not a discovery question. Cost-realism does not change MES alpha
  evidence; it changes net-of-cost expectancy at deployment.
- Not a blocker on current live state. MES has zero live lanes in
  `docs/runtime/lane_allocation.json` (verified at commit time: only six
  MNQ lanes are routed). The debt is dormant for current exposure.

## When the debt becomes load-bearing (escalation triggers)

Any one of these makes the event-tail measurement load-bearing rather
than dormant. Until at least one fires, the debt remains scoped-open and
the routine pilot's modeled-conservative claim stands as the central
tendency answer:

- **Trigger A — MES enters the live allocator.** The first time
  `docs/runtime/lane_allocation.json` carries **any MES lane** — on any
  session, regardless of whether the session directly overlaps FOMC /
  NFP / CPI / OPEX release windows. Every MES session is exposed to
  overnight-gap carry from calendar-event and macro-shock days, so a
  session's own clock does not insulate it from the event-tail
  distribution. The moment a MES lane is live-routed, event-tail cost
  realism is load-bearing against capital.
- **Trigger B — MES cost model is changed.** Any edit to
  `pipeline/cost_model.py` that touches MES friction parameters
  (`tick_size`, `slippage`, `spread_doubled`, `friction_in_points`,
  `total_friction`, `commission_rt`, `point_value`, `min_ticks_floor`)
  invalidates the routine pilot's "modeled-conservative" claim and
  forces a re-measurement on the new model.
- **Trigger C — MES contract size scales beyond 1.** Routine-pilot
  evidence is at one-contract scale; book-size sensitivity to fill
  quality is a separate question. If a MES profile lifts
  `max_contracts > 1` or any sizer rule promotes MES beyond unit size,
  the event-tail question gains a queue-impact dimension that the
  one-contract pilot does not address.
- **Trigger D — A second live MES routine sample contradicts the pilot.**
  If any future MES routine fill (TBBO repriced, same canonical method)
  shows measured slippage `> 1 tick` for a calendar-routine day, the
  pilot's central-tendency claim itself is impeached and the event-tail
  question becomes secondary to a routine-tendency re-audit.

## Pre-registered method for the eventual measurement

When a trigger fires, the event-tail measurement follows this
methodology — set down here so the future author cannot drift the
question after looking at outcomes.

### Sample selection

- Universe: MES E2 trades on the deployable sessions in scope at the
  time of the trigger.
- Stratify into four cells by event class:
  1. Calendar-event-day (NFP, FOMC, CPI, PPI, OPEX, holiday-short,
     month-end / quarter-end).
  2. Macro-shock-day (post-hoc range Z `>= 2` OR overnight gap Z `>= 2`).
  3. Liquidity-degraded-day (Databento degraded warning fired).
  4. Microstructure-burst-window (entry within 5 minutes of a
     same-session range Z `>= 2` move).
- Sample size: pre-commit at the trigger to a minimum of `30` repriced
  fills per cell. This uses the project's exploratory-entry floor from
  `docs/institutional/pre_registered_criteria.md` Criterion 7
  ("`N >= 30` for exploratory discovery entry"); it is a stability
  floor for median / p95 estimation on a one-sided distribution, not a
  Bailey-2013 MinBTL bound (MinBTL bounds strategy selection over
  Sharpe, which is a different question from distribution estimation on
  a fixed process). If event density is too low to reach `30` per cell
  over the available window, report as power-floor `UNVERIFIED` for
  that cell, not as proof of "no event effect."

### Measurement

- Repricing: `research.databento_microstructure.reprice_e2_entry`
  exclusively. No bespoke repricer.
- Boundary: `pipeline.dst.orb_utc_window`. No fallback to `break_ts` or
  `break_delay_min`.
- Compare measured per-cell median, p95, and max slippage (in ticks)
  against the current `pipeline.cost_model.get_cost_spec("MES")`
  modeled-slippage baseline of **5 ticks per side** (1.25 points).
- Holdout discipline: `HOLDOUT_SACRED_FROM = 2026-01-01`. Event-tail
  evidence drawn from before the holdout is IS for any future
  cost-model change; OOS evidence is monitor-only.

### Verdict bar

All slippage figures are **measured slippage in ticks** (not
modeled-tick units). The "modeled-conservative" claim is preserved iff
**every cell** shows median `<= 1 tick` AND p95 `<= 2 ticks` — i.e.,
measured central tendency stays at least `~5x` below the modeled
5-ticks-per-side baseline and measured p95 stays at least `~2.5x`
below it.

Any cell breaching that bar triggers a cost-model re-derivation
proposal (separate stage, not this debt scope).

## Outputs

- This debt-scope doc at
  `docs/plans/2026-04-25-mes-event-tail-slippage-debt-scope.md`.
- `docs/runtime/debt-ledger.md` `cost-realism-slippage-pilot` entry —
  tightened: replaces the previous single-sentence "STILL OPEN:
  event-day tail NOT measured" line with a pointer to this doc and the
  four escalation triggers.
- `docs/runtime/action-queue.yaml` item `mes_event_tail_slippage_realism`
  -> `status: done` with `notes_ref` pointing here.

## Limitations

- This is a debt-scope doc, not a measurement. The pre-registered
  method above is a contract for the future author when a trigger
  fires; it has not been executed.
- The four event-class cells are a chosen partition, not an exhaustive
  one. The author at trigger time should review whether new event types
  (e.g., a new policy-rate cycle, exchange schedule changes) need to be
  added to the partition before sampling.
- The "modeled-conservative" preservation bar (median `<= 1 tick`, p95
  `<= 2 ticks`) is set against the current `cost_model` 5-ticks-per-side
  baseline. If the cost-model slippage assumption changes (Trigger B),
  the bar moves with it and must be re-derived at trigger time.
- Trigger A is deliberately session-agnostic rather than session-scoped.
  The cost of a false-positive trigger (running the measurement earlier
  than strictly necessary) is bounded; the cost of a false-negative
  trigger (MES live exposure without event-tail evidence when one was
  warranted) is not.
- This doc does not change MES live behaviour, alpha discovery, or any
  validated-shelf state. It is purely a contract on future debt closure.

## References

- `docs/audit/results/2026-04-24-mes-e2-slippage-pilot-v1.md` — routine pilot.
- `docs/runtime/debt-ledger.md` — `cost-realism-slippage-pilot` entry.
- `pipeline/cost_model.py` — MES friction parameters.
- `research/databento_microstructure.reprice_e2_entry` — canonical repricer.
- `trading_app/holdout_policy.py::HOLDOUT_SACRED_FROM`.
- `docs/institutional/pre_registered_criteria.md` — Criterion 7
  (`N >= 30` exploratory floor) used as the per-cell stability floor.
- `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md`
  — MinBTL theorem. **NOT cited as the per-cell bound for this
  measurement** (MinBTL bounds strategy selection, not
  distribution-estimation sample size). Included here only so future
  readers do not misapply it to this debt.
- `.claude/rules/backtesting-methodology.md` — RULE 1 (feature
  knowability), RULE 6 (trade-time knowability), RULE 9 (canonical
  layers only).
