---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Garch A4c Routing-Selectivity Allocator — Design

**Date:** 2026-04-17
**Status:** DESIGN LOCKED — hypothesis pre-registered
**Hypothesis file:** `docs/audit/hypotheses/2026-04-17-garch-a4c-routing-selectivity.yaml`
**Predecessor:** A4b (NULL_BY_CONSTRUCTION at 5-slot budget + throughput-mis-metric harness)
**Audit grounding:** `docs/audit/results/2026-04-17-allocator-scarcity-surface-audit.md`

## Authority chain

- `CLAUDE.md`
- `RESEARCH_RULES.md`
- `TRADING_RULES.md`
- `docs/plans/2026-04-16-garch-institutional-attack-plan.md`
- `docs/plans/2026-04-16-garch-a4b-binding-budget-design.md` (predecessor design)
- `docs/audit/results/2026-04-17-garch-a4b-binding-budget-replay.md` (predecessor result)
- `docs/audit/results/2026-04-17-allocator-scarcity-surface-audit.md` (audit grounding)
- `docs/institutional/pre_registered_criteria.md` (v2.7, Mode A holdout)

---

## 1. Why A4b is demoted

A4b executed 2026-04-17 with `NULL_BY_CONSTRUCTION` verdict. Two load-bearing failures:

1. **Non-binding scarcity surface after candidate-eligibility filter.** Bind ratio 50/72 = 0.694 vs locked 0.80, because the candidate's own `N_garch_high >= 20` per-lane filter thinned the deployable shelf below the binding threshold on 22 of 72 rebalance dates.
2. **Throughput-biased primary rule on a selectivity-style candidate.** A4b's primary metric was annualized total R on selected slots, which rewarded slot-firing volume over per-slot quality. Baseline fired 2.92 slots/day, candidate 2.29, positive control 2.06 — every non-baseline ranker lost total R by design, regardless of per-fill quality. **Positive control failed the primary rule**, which tells you the rule itself was mis-dimensioned.

A4b's own guidance:

> Do not tune weights, thresholds, or budget to rescue A4b. Do not interpret this as allocator utility evidence. Redesign the scarce-resource surface first.

A4c implements that redesign. It does not reinterpret A4b.

---

## 2. Corrected question

> Does the locked A4b pre-entry garch composite, re-scored on a dimension-neutral routing/selectivity metric, improve the canonical shelf allocator over a random-uniform null and a trailing-Sharpe comparator, on two independently binding scarcity surfaces?

This isolates two variables A4b conflated:

- **Scarcity surface:** now tested on TWO surfaces, both audit-verified binding before the replay runs.
- **Metric dimension:** now R per filled slot-day, normalized to scarce-resource consumption.

All other inputs (candidate identity, weights, correlation gate, hysteresis, deploy window, OOS boundary) are inherited verbatim from A4b.

---

## 3. Corrected scarcity surfaces

Both pre-verified binding via `research/allocator_scarcity_surface_audit.py` run on 2026-04-17.

| Surface | Budget | Bind ratio (72 IS dates) | Role |
|---|---:|---:|---|
| A. Raw slots | max_slots = 5 | 0.944 | Primary (apples-to-apples with A4b's nominal budget) |
| B. Rho-survivor slots | max_slots = 3 | 0.931 | Challenger (operative-mechanic; matches build_allocation's actual rho gate) |

Dropped at audit:

- Surface C (risk-R dollars at $2,500) — bind 0.167; not salvageable without remeasurement as daily firing-overlap.
- Surface D (contract-integer geometry) — demoted; identical to Surface A at 1-contract minimum.

**Key fix vs A4b:** binding is verified on RAW deployable supply BEFORE any candidate-specific filter. The candidate's `N_garch_high >= 20` filter is a candidate eligibility gate, not a binding surface definition.

---

## 4. Corrected baselines

| Role | Identity | Rationale |
|---|---|---|
| Primary null | `RANDOM_UNIFORM_UNDER_BINDING` (seeded) | Dimension-neutral; no ranking key; tests whether candidate's edge is real vs. noise under the same rho/DD/budget machinery |
| Secondary comparator | `TRAILING_SHARPE` (sharpe_ann_adj as ranking key) | Literature-grounded alternative to `_effective_annual_r`; audit showed 34% hit-rate divergence from lit-grounded on the last 12 rebalances (structural alternative, not a noise permutation) |

**A4b's baseline (`_effective_annual_r`) is not used in A4c.** It failed A4b's positive control and carries the dimension-mismatch bug. Any rescue rerun would re-import the defect.

Role allocation is unchanged from user's explicit call: audit data changed the *strength* of random-uniform as a null (gap only 13% vs baseline on hit-rate dimension), but not its *role*. Random-uniform remains primary null because its semantic contract is "does the candidate beat noise under binding constraints."

---

## 5. Corrected metric

**Primary: R per filled slot-day.**

```
R_per_fill =
  (sum of 1R-normalized pnl across all forward-window fires
   across all selected lanes across all rebalance months)
  /
  (total filled slot-day count)
```

"Filled slot-day" = one (selected_lane, forward_day) pair where the selected lane actually fired in the forward window.

Properties:

- Denominator grows with both slot budget AND firing frequency → dimension-neutral.
- A selective ranker firing fewer slots is NOT penalized on dimensional grounds.
- Scale-free relative to hit rate.

**Secondary robustness:** selected-slot Sharpe — annualized Sharpe of daily portfolio pnl across selected lanes over all forward windows.

**Descriptive (reported, not used for pass/fail):**

- Annualized total R (A4b's former primary — kept as audit trail)
- Integer-contract realized DD
- Selection churn (mean jaccard across rebalances)
- Mean slot-hit-rate per day
- Per-month R-per-fill time series
- 2026 OOS mirror of the same metrics per surface per ranker

---

## 6. Harness-sanity gate (MANDATORY, runs BEFORE candidate evaluation)

1. Run `POSITIVE_CONTROL_TRAILING_EXPR` = `trailing_expr` directly as the ranking key on BOTH surfaces A and B.
2. Check: `R_per_fill_positive_control >= R_per_fill_primary_null + 0.01` on BOTH surfaces.
3. **If NO on either surface → ABORT A4c.** The primary rule cannot discriminate; candidate math has no meaning. Do not rescue, do not re-run, do not tune. Return to framing.
4. **If YES on both → proceed to candidate evaluation.**

This directly fixes A4b's latent bug. A4b ran to completion with positive control failing and produced a result that could not be cleanly interpreted. A4c cannot repeat that failure mode.

---

## 7. Pre-declared constants (locked, no post-hoc tuning)

| Constant | Value | Origin |
|---|---:|---|
| `min_lift_per_fill` | 0.01 R/fill | Pre-registered 2026-04-17; ~15% relative uplift at implied baseline R/fill levels, dimension-matched to selectivity candidate |
| `min_sharpe_lift` | 0.05 | Inherited from A4b |
| `max_dd_inflation` | 1.20 | Inherited from A4b |
| `selection_churn_cap` | 0.50 jaccard | Inherited from A4b |
| `bind_pass_ratio_gate` | 0.80 per surface | Pre-registered 2026-04-17; met at audit |
| `K_budget` | 1 | Single locked composite vs single pair of baselines |

Any attempt to tune any of these after the first result is a hard kill criterion per the hypothesis file.

---

## 8. Dual-surface verdict matrix

| Surface A | Surface B | Verdict | Interpretation |
|---|---|---|---|
| Pass | Pass | **STRONG PASS** | Routing edge on raw supply AND operative mechanic |
| Pass | Fail | STANDARD PASS | Raw-supply edge only; flag — no rho-structure lift |
| Fail | Pass | STRUCTURAL PASS | Operative-mechanic edge only; investigate why A fails |
| Fail | Fail | **NULL** | No rescue; do not tune |
| Positive control fails on A or B | any | **ABORT** | Harness bug; not a candidate verdict |

Note: a STANDARD_PASS or STRUCTURAL_PASS is a legitimate result, not a partial failure. Each carries different deployment implications.

---

## 9. Controls carried from A4b

- **Destruction shuffle** on `garch_forecast_vol_pct` across days within instrument; shuffled version MUST fail primary on BOTH surfaces. Kill if it passes either.
- **Selection churn cap** jaccard ≤ 0.50 across rebalances on each surface.
- **OOS descriptive-only,** 2026-01-01 holdout sacred (Mode A per `pre_registered_criteria.md` v2.7).
- **OOS direction match** required per surface.
- **OOS effect ratio** ≥ 0.40 per surface that passed IS.
- **K = 1**, no sweeps.
- **No weight, threshold, budget, or min_lift tuning** after first result seen.

---

## 10. Scope hard boundaries

- No deployment claim (routing utility only).
- No profile/firm translation (that's A5).
- No new state variables beyond `garch_forecast_vol_pct` at the 70-pct threshold.
- No per-session tuning.
- No post-hoc weight search (Stage 2 pre-registered at ±50% sensitivity, reserved for Stage 1 pass only).
- No ad-hoc baseline invention.
- No revised metric after results are seen — R-per-fill is locked.
- Routing claim only — NOT a garch edge claim, NOT a throughput claim, NOT a capital-deployment claim.

---

## 11. Outputs (when Stage 1 replay runs)

- `docs/audit/results/2026-04-17-garch-a4c-routing-selectivity-replay.md`
- `research/output/garch_a4c_routing_selectivity_replay.json`
- `research/garch_a4c_routing_selectivity_replay.py` (committed to `research/` before execution)

---

## 12. Next order

1. Lock design (this doc) + hypothesis YAML — committed 2026-04-17.
2. **Replay script** `research/garch_a4c_routing_selectivity_replay.py` — **awaiting explicit "write the replay" call. Not authored by this design commit.**
3. Run positive-control gate on BOTH surfaces.
4. If gate passes: run candidate + controls on BOTH surfaces.
5. Write replay MD with dual-surface verdict.
6. If STRONG / STANDARD / STRUCTURAL pass: proceed to Stage 2 sensitivity (±50% on w1/w2) on passing surfaces only.
7. If NULL or ABORT: do not rescue. Park A4c. Return to framing for A4d only if user-approved.

---

## 13. Current doctrine

- **A4a demoted:** null by construction, active-profile slot cap surface did not bind.
- **A4b demoted:** null by construction at 5-slot budget AFTER candidate-eligibility filter, AND throughput-mis-metric primary rule that no selectivity ranker could clear.
- **A4c** is the corrected retry with: (1) two audit-verified binding scarcity surfaces measured on RAW supply, (2) dimension-neutral R-per-fill primary metric, (3) MANDATORY harness-sanity gate on positive control that aborts before any candidate math, (4) inherited A4b candidate for clean experimental separation.
- **A4c cannot be rescued if it fails.** NULL and ABORT are acceptable outcomes.
- **A4c does NOT claim or refute garch edge.** It tests routing utility of a locked composite.

---

## 14. Explicitly deferred, NOT killed

- **Candidate variations** (weight sweeps, new state vars, different garch thresholds) — reserved for A4d or later, pre-registered separately.
- **Profile/firm translation** (A5) — blocked on A4c passing on at least one surface.
- **Daily firing-overlap Surface C remeasurement** — reserved for A4e if routing value is established and risk-R becomes a live scaling question.
- **Non-garch routing candidates** (ATR-vel regime, overnight-range, etc.) — Phase B of the institutional attack plan, blocked on A4c completion.
