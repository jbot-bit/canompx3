# 6-Lane Unfiltered Baseline Stress-Test

**Date:** 2026-04-20
**Branch:** `research/6lane-unfiltered-baseline-stress`
**Script:** `research/audit_6lane_unfiltered_baseline.py`
**Raw JSON:** `docs/audit/results/2026-04-20-6lane-unfiltered-baseline-stress.json`
**Parent finding:** `docs/audit/results/2026-04-20-6lane-filter-vestigialness.md` (PR #47)

---

## Question

PR #47 showed 5 of 6 deployed-lane filters fire ≥75% in 2026. High fire rate
means the filters don't *subtract* in 2026, but does not by itself tell us
whether they *add* lift. The follow-up:

> For each DEPLOY lane, does the unfiltered session + RR + entry + CB geometry
> carry the edge alone, or is the filter doing real work?

---

## Power & scope discipline

**2026 OOS is not a valid classification base.** The 2026 slice is ~3.5 months,
n=54–72 per lane. At per-trade `pnl_r` std ≈ 0.9, the 95% CI on a 65-trade
mean is ±0.22R. We cannot distinguish a +0.0R filter from a +0.15R filter at
that sample size. The first draft of this audit classified on 2026 OOS and
got the answer wrong.

**This pass uses IS (2019–2025, n=1636–1794 per lane) for classification.**
2026 OOS is reported as monitoring-only context.

Classification is a 2×2 grid on IS:

| Axis A: unfiltered baseline edge? | Axis B: filter discriminates? | Verdict |
|---|---|---|
| YES (unf \|t\| > 2.0, mean > 0) | YES (Welch fire-vs-non-fire p < 0.05, t > 0) | BOTH_CONTRIBUTE |
| YES | NO | FILTER_VESTIGIAL |
| NO | YES | FILTER_IS_THE_EDGE |
| NO | NO | DEAD_LANE |

Welch two-sample t-test compares IS `pnl_r` on days the filter FIRED vs days
it did NOT. This is the correct test for "does the filter's selection rule
add information beyond geometry?" If the filter fires in IS on essentially
every day (n_non_fire < 10), axis B is untestable and we report on axis A
alone.

Method: reuse `parse_strategy_id` + `load_lane_universe` from PR #47's
`research/audit_6lane_scale_stability.py` (same orb_minutes=5, same
aperture-overlay handling — two audits are directly comparable).

---

## Results

### Portfolio rollup

| Classification | Lanes | Count |
|---|---|---|
| FILTER_IS_THE_EDGE | L1 EUROPE_FLOW, L2 SINGAPORE_OPEN | 2 |
| BOTH_CONTRIBUTE | L3 COMEX_SETTLE, L5 TOKYO_OPEN | 2 |
| FILTER_VESTIGIAL | L4 NYSE_OPEN, L6 US_DATA_1000 | 2 |

Two lanes depend on the filter. Two lanes have both components working.
Two lanes are filter-vestigial — their filter no longer selects but the
unfiltered baseline carries the edge regardless.

### Per-lane IS decomposition (n=1636–1794)

| Lane | unfilt t | unfilt ExpR | filter t | filter ExpR | Welch t | Welch p | IS Δ | Verdict |
|------|---------|-------------|----------|-------------|---------|---------|------|---------|
| L1 EUROPE_FLOW ORB_G5 | +1.61 | +0.043 | +2.28 | +0.064 | +3.29 | 0.001 | +0.021 | FILTER_IS_THE_EDGE |
| L2 SINGAPORE_OPEN ATR_P50 | −0.38 | −0.010 | +1.73 | +0.063 | +3.05 | 0.002 | +0.073 | FILTER_IS_THE_EDGE |
| L3 COMEX_SETTLE ORB_G5 | +2.37 | +0.067 | +3.15 | +0.092 | +5.12 | <0.001 | +0.025 | BOTH_CONTRIBUTE |
| L4 NYSE_OPEN COST_LT12 | +3.47 | +0.081 | +3.50 | +0.082 | +0.54 | 0.593 | +0.001 | FILTER_VESTIGIAL |
| L5 TOKYO_OPEN COST_LT12 | +2.64 | +0.070 | +3.27 | +0.122 | +2.21 | 0.028 | +0.052 | BOTH_CONTRIBUTE |
| L6 US_DATA_1000 ORB_G5 O15 | +3.20 | +0.093 | +3.30 | +0.096 | +1.53 | 0.147 | +0.003 | FILTER_VESTIGIAL |

### 2026 OOS (monitoring only — n=54–72, 95% CI ≈ ±0.22R per-trade)

| Lane | 2026 n_unf | 2026 unfilt ExpR | 2026 filter ExpR | 2026 Δ | fire% 2026 |
|------|-----------|-----------------|------------------|--------|------------|
| L1 EUROPE_FLOW | 72 | +0.293 | +0.293 | +0.000 | 100.0% |
| L2 SINGAPORE_OPEN | 72 | +0.192 | +0.216 | +0.024 | 75.0% |
| L3 COMEX_SETTLE | 66 | +0.006 | +0.006 | +0.000 | 100.0% |
| L4 NYSE_OPEN | 71 | +0.136 | +0.136 | +0.000 | 100.0% |
| L5 TOKYO_OPEN | 72 | +0.149 | +0.153 | +0.004 | 97.2% |
| L6 US_DATA_1000 | 68 | −0.034 | −0.034 | +0.000 | 100.0% |

None of the 2026 deltas are statistically distinguishable from zero.

---

## Interpretation per classification

### FILTER_IS_THE_EDGE (L1, L2) — filter-dependent lanes

**L1 MNQ_EUROPE_FLOW ORB_G5.** Unfiltered IS baseline is not significant
(t=+1.61, p=0.107). The filter's Welch test vs non-fire days is strongly
significant (t=+3.29, p=0.001). The +0.021R IS delta is small but the
selection is real — removing the filter would drop the lane below
deploy-viable IS power.

**L2 MNQ_SINGAPORE_OPEN ATR_P50.** This is the most filter-dependent lane.
Unfiltered IS is net-negative (ExpR −0.010, t=−0.38). The filter turns a
losing lane into a borderline-significant winner (filter ExpR +0.063,
t=+1.73). Welch p=0.002 confirms the selection is doing real work. If the
filter were removed, this lane would be deploy-killed outright.

**Operational implication:** L1 and L2 require their filters. Any proposal
to "simplify" by removing filters must not touch these two lanes.

### BOTH_CONTRIBUTE (L3, L5) — both components load-bearing

**L3 MNQ_COMEX_SETTLE ORB_G5.** Baseline alone hits significance (t=+2.37).
Filter adds meaningful selection (Welch p<0.001, IS Δ=+0.025). But the
100%-fire rate from 2022 onward (PR #47) means the selection was
historically useful and is now structurally 0pp per post-2021 trade. Edge
that survives 2026 is essentially baseline.

**L5 MNQ_TOKYO_OPEN COST_LT12.** Both load-bearing. Baseline t=+2.64,
filter adds +0.052R (Welch p=0.028). Filter is still meaningfully selecting
(fire rate 48–84% in recent years, PR #47) — this is the lane where the
filter-vs-geometry decomposition is cleanest.

**Operational implication:** these lanes benefit from both components.
Removing the filter reduces portfolio ExpR; so does removing the session
geometry. No change recommended.

### FILTER_VESTIGIAL (L4, L6) — baseline carries, filter is decorative

**L4 MNQ_NYSE_OPEN COST_LT12.** Baseline is the strongest of the six (t=+3.47).
Filter Welch p=0.593 — statistically indistinguishable from random
selection among fired/non-fired trades. The filter fires 95–100% IS and
100% in 2026. Operationally identical to "no filter."

**L6 MNQ_US_DATA_1000 ORB_G5 O15.** Baseline t=+3.20 — strong. Filter Welch
p=0.147 — not significant. Fire rate 95–100% IS, 100% in 2026.

**Operational implication:** removing these two filters would not change
deployed behavior (100% fire means the filter is a no-op). But removing
them risks introducing regressions in dispatch code with zero EV upside.
Leave them in place. Note that the `lane_allocation.json trailing_expr`
for these lanes is computed on the filtered universe, which is within
0.01R of the unfiltered baseline — no scoring adjustment needed.

---

## What this updates in our understanding

1. **PR #47 conclusion refined.** PR #47 said "5 of 6 lanes are
   filter-vestigial." The honest 2×2 reads: 2 of 6 are filter-dependent
   (L1, L2), 2 of 6 are both-contributing (L3, L5), 2 of 6 are truly
   vestigial (L4, L6). "Filter vestigial at the fire-rate level" is not
   the same as "filter contributes nothing to lane P&L" — the prior MD
   conflated these.

2. **L2 SINGAPORE_OPEN ATR_P50 is the lane most at risk from the
   ATR_P50 rolling-percentile instability flagged in PR #47.** That
   filter IS the edge on that lane. If the percentile's rolling window
   drifts, the lane's entire IS signal (t=+1.73) collapses to the
   unfiltered baseline (t=−0.38). L2 is the highest-priority filter to
   re-audit for instability.

3. **L1 EUROPE_FLOW ORB_G5 fire-rate drift (57%→100% yoy) eliminates
   a selection that was historically load-bearing.** Pre-2022 the filter
   selected ~60% of days with meaningful edge. Post-2022 it fires on
   ~all days. The IS Δ=+0.021R is pre-2022-heavy; 2026 Δ=0 because the
   filter is pass-through. In effect, L1 is *transitioning* from
   FILTER_IS_THE_EDGE (IS) to unfiltered-baseline-equivalent
   (post-scale-drift era). Worth diagnosing separately.

4. **L4/L6 filter vestigialness is confirmed and benign.** Baseline
   carries the edge. Filter is decorative. No capital at risk.

---

## Operational conclusions

1. **Portfolio posture UNCHANGED.** All 6 DEPLOY lanes continue.

2. **L2 SINGAPORE_OPEN ATR_P50 is the new-priority filter audit.** This
   lane's IS signal depends entirely on the filter. Run a separate
   pre-reg to stress-test the ATR_P50 rolling-percentile stability under
   SINGAPORE_OPEN's sparse session data before committing further capital.

3. **L1 EUROPE_FLOW ORB_G5 diagnostic.** Filter IS the edge historically
   but 2026 fire rate = 100% means the selection is no longer
   discriminating. If a refined filter can restore 2019-era selectivity
   at current scale, the lane's forward edge could be materially higher.
   Candidate: ATR-normalized absolute-threshold equivalent (per PR #47
   methodology).

4. **L4/L6 leave alone.** Filters cost nothing at 100% fire rate; removing
   them is dispatch-code churn with zero EV upside.

5. **L3/L5 leave alone.** Both components work. Don't touch.

6. **No new pre-regs required from this finding.** This is a descriptive
   stress-test, not a new hypothesis. The pre-regs it *suggests* (L2
   ATR_P50 stability audit, L1 EUROPE_FLOW ATR-normalized replacement)
   are separate work items.

---

## Next-best tests (not pursued in this turn)

1. **L2 ATR_P50 rolling-percentile stability audit.** Highest priority
   given L2's full filter-dependence. Read-only, ~1h. Would ship as its
   own pre-reg.

2. **L1 EUROPE_FLOW ATR-normalized ORB_G5 replacement.** If an
   ATR-normalized grid threshold restores the pre-2022 57% fire rate
   at current scale, L1 forward edge materially improves. Pathway-B K=1.

3. **L6 US_DATA_1000 2026 breakdown.** Unfiltered 2026 ExpR = −0.034R.
   Diagnostic turn to check per-release-type decomposition. Separate work.

---

## Provenance & discipline

- Canonical data: `orb_outcomes`, `daily_features` (`gold.db`, UTC).
- Filter delegation: `research.filter_utils.filter_signal` (calls
  `trading_app.config.ALL_FILTERS[key].matches_df`).
- Lane universe: `docs/runtime/lane_allocation.json`
  (DEPLOY lanes as of `rebalance_date: 2026-04-18`).
- Holdout: `2026-01-01` (Mode A sacred, per
  `institutional_phase0_grounding.md`).
- Reuse: `parse_strategy_id`, `load_lane_universe` from
  `research/audit_6lane_scale_stability.py` (PR #47 `007ecd6b`).
- No production code touched. No new filters registered. No config changes.

**Power caveat honored:** classifications operate on IS (n=1636–1794 per
lane); 2026 OOS (n=54–72) is monitoring commentary only, per
`feedback_oos_power_floor.md`.
