---
slug: 6lane-unfiltered-baseline-stress
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-20
task: 6-lane unfiltered baseline stress-test — does edge survive without filter?
---

# Stage: 6-lane unfiltered baseline stress-test

## Task

For each of the 6 currently-DEPLOYED lanes (per `docs/runtime/lane_allocation.json`),
quantify how much of the lane's 2026 OOS edge comes from the FILTER versus the
UNFILTERED baseline (session + RR + direction + entry_model + CB geometry alone).

Motivation: PR #47 6-lane audit showed 5 of 6 deployed filters ≥75% fire rate
in 2026 — largely vestigial. If the unfiltered baseline carries the edge,
the portfolio is simpler than claimed. If the baseline is weak, the portfolio
is quietly weaker (relying on selection that no longer fires selectively).

This test is READ-ONLY on canonical data (`orb_outcomes` JOIN `daily_features`).
No production code touched. Output: one audit script + one results MD.

## Scope Lock

- `research/audit_6lane_unfiltered_baseline.py` (new)
- `docs/audit/results/2026-04-20-6lane-unfiltered-baseline-stress.md` (new)

## Blast Radius

- Read-only scripts/docs. Zero production-code touch.
- Reads `docs/runtime/lane_allocation.json` (treated as input, not edited).
- Canonical data only (`orb_outcomes` + `daily_features`).
- No new filters registered. No config changes.
- No downstream consumers.

## Approach

1. Parse each DEPLOYED lane's strategy_id into (inst, session, orb_minutes,
   entry, rr, cb, filter) — reuse `parse_strategy_id()` from
   `research/audit_6lane_scale_stability.py` pattern.
2. For each lane, fetch canonical universe via `orb_outcomes` JOIN
   `daily_features` (inst, session, orb_minutes, entry, rr, cb filtered).
3. Compute UNFILTERED (no filter applied) metrics:
   - Per-year: N, ExpR, Sharpe, win rate
   - Full sample: N, ExpR, Sharpe, annualized Sharpe, t-stat vs null
   - 2026 OOS: N, ExpR, Sharpe + one-sample t-test
4. Compute FILTERED (canonical filter applied via `filter_signal()`) for
   same breakdown.
5. Compute DELTA (filter − unfiltered) per-year and on 2026 OOS.
6. Classify each lane as one of:
   - `BASELINE_CARRIES_EDGE` — unfiltered 2026 ExpR ≥ filter ExpR (filter
     adds nothing meaningful)
   - `FILTER_ADDS_EDGE` — filter 2026 ExpR > unfiltered + 0.05R
   - `FILTER_HURTS` — filter ExpR < unfiltered
   - `UNDERPOWERED` — 2026 N too small for either arm to be decisive
     (tie to the OOS-power-floor feedback rule).
7. Report in MD with per-lane tables + portfolio-level rollup.

## Acceptance criteria

1. `research/audit_6lane_unfiltered_baseline.py` runs to completion on
   current `gold.db` without exceptions. Exit 0.
2. All 6 DEPLOYED lanes classified (status field in MD). No lane marked
   "error" or "skipped".
3. MD contains per-lane: 2019-2026 year table with N/ExpR/Sharpe
   (unfiltered and filtered), delta column, 2026 OOS row, classification
   verdict.
4. Portfolio rollup: count of BASELINE_CARRIES_EDGE vs FILTER_ADDS_EDGE
   vs FILTER_HURTS vs UNDERPOWERED. Implication statement for portfolio
   posture.
5. `python pipeline/check_drift.py` passes (expected: no change — we only
   add two files, touch no production code).
6. `git diff --stat` scope matches scope_lock above.

## Non-goals (explicit)

- Not touching production code.
- Not registering new filters.
- Not re-running the PR #47 scale-artifact audit (already done).
- Not proposing deploy/pause decisions — research output only; deploy
  decisions need a separate design turn.
- Not a DSR computation — unfiltered baseline here is pre-registered
  (the 6 DEPLOYED lane geometries), not discovered. DSR is meaningless
  when K=1 per lane by construction.
