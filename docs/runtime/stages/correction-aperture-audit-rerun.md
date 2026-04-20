---
slug: correction-aperture-audit-rerun
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Correct PR #52/#54/L6 — aperture overlay (_O15) discarded; L2/L6 re-run on canonical orb_minutes=15
---

# Stage: Correction — aperture overlay discard

## Bug summary

PR #52 (`audit_6lane_unfiltered_baseline.py`) and PR #54
(`audit_l2_atr_p50_stability.py`) and the uncommitted L6 diagnostic
all hardcoded `orb_minutes = 5` in their SQL. This was inherited from
`research/audit_6lane_scale_stability.py` (PR #47) which parses the
`_O15` aperture suffix but discards it.

The canonical parser at `trading_app/eligibility/builder.py:122-125`
extracts orb_minutes from the `_O*` suffix:
  - L2 `MNQ_SINGAPORE_OPEN_..._ATR_P50_O15` → orb_minutes=**15**
  - L6 `MNQ_US_DATA_1000_..._ORB_G5_O15` → orb_minutes=**15**

2 of 6 deployed lanes were backtested at the wrong aperture.

## Blast radius (material differences confirmed)

| Lane | Wrong (orb_min=5) ExpR full | Correct (orb_min=15) ExpR full |
|------|------------------------------|--------------------------------|
| L2   | −0.002R (n=1794)             | **+0.052R (n=1786)**           |
| L6   | +0.088R (n=1742)             | +0.108R (n=1550)               |

L2's shift from net-negative to net-positive on the baseline completely
overturns PR #52's `FILTER_IS_THE_EDGE` classification for L2 and
invalidates PR #54's premise (that the lane is fully filter-dependent).
L6's shift is smaller but still wrong data.

L1, L3, L4, L5 do not carry `_O` suffix → orb_minutes=5 is correct
for those lanes. Their PR #52 numbers are unaffected.

## Code-review findings to apply (from `superpowers:code-reviewer`)

- **H1** (fixed here): use canonical `parse_strategy_id` from
  `trading_app.eligibility.builder`, pass orb_minutes per lane.
- **H2**: L6 bootstrap reframe — primary verdict = one-sample t + power,
  bootstrap as descriptive null band only.
- **H3**: L2 "FILTER_IS_THE_EDGE" soften — "filter's day-selection
  correlates with edge; mechanism is vol-regime gate, not
  session-specific."
- **M1**: L2 KS claim reframe — per-year Δμ table is honest evidence,
  not KS p=1.0 on pooled.
- **M2**: MD phrasing: "deploy-killed" → "deploy-not-selected";
  `t=-0.38` = zero-consistent, not net-negative.
- **M3**: IS/OOS split — label as trade-count median, not calendar
  midpoint.
- **M4**: Add K=6 multiple-testing note to 6-lane audit.

## Scope Lock

- `research/audit_lane_baseline_decomposition_v2.py` (new — consolidated,
  canonical-delegated)
- `docs/audit/results/2026-04-21-correction-aperture-audit-rerun.md` (new —
  supersedes PR #52, PR #54, and stashed L6 diagnostic)

## Blast Radius

- Read-only research. Zero production-code touch.
- Canonical data only: `orb_outcomes`, `daily_features`.
- Imports from `trading_app.eligibility.builder` (canonical parser).
- No new filters registered. No config changes.
- Supersedes: PR #52 (L2/L6 rows), PR #54 (entirely).
- Preserved: PR #52 L1/L3/L4/L5 numbers, PR #47 6-lane scale stability
  audit (which has its own use of orb_minutes=5 — worth flagging but not
  fixed here as out of scope).

## Approach

1. Import canonical `parse_strategy_id` from `trading_app.eligibility.builder`.
2. For each DEPLOY lane, use canonical spec to load universe.
3. Three analyses in one script (shared universe loading):
   - (A) 2×2 baseline-vs-filter decomposition (IS-based, with K=6 note)
   - (B) L2 stability deep-dive (per-year, early/late halves, rolling 3y,
     per-year atr_pct distribution)
   - (C) L6 2026 diagnostic (one-sample t + bootstrap null band + calendar
     + vol-regime decomp)
4. Consolidated MD with:
   - Section 1: bug + scope + what changed
   - Section 2: corrected 2×2 for all 6 lanes
   - Section 3: corrected L2 deep-dive with softened language
   - Section 4: corrected L6 diagnostic with reframed bootstrap
   - Section 5: what stands from prior PRs
   - Section 6: revised operational implications

## Acceptance criteria

1. Script runs without exceptions on current `gold.db`.
2. MD contains all 6 lanes with correct orb_minutes per lane.
3. L2 classification recomputed — likely changes from
   `FILTER_IS_THE_EDGE` to a different label.
4. L6 primary verdict is one-sample t + power calc, bootstrap as
   descriptive context.
5. L2 KS claim dropped or replaced with per-year distribution table.
6. MD explicitly documents what supersedes what in PR #52/#54.
7. `python pipeline/check_drift.py` passes.
8. `git diff --stat` scope matches scope_lock.

## Non-goals

- Not fixing `research/audit_6lane_scale_stability.py` (PR #47 artifact
  — noted as follow-up in correction MD).
- Not fixing `research/audit_6lane_unfiltered_baseline.py` (PR #52
  artifact — superseded by this audit).
- Not re-running L1/L3/L4/L5 decompositions (unaffected by the bug).
- Not proposing deployment changes (even if L2 classification flips).
