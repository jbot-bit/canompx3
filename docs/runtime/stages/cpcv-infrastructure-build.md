---
slug: cpcv-infrastructure-build
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Build trading_app/cpcv.py module + run H1/H2/H3 calibration per 2026-04-21-cpcv-infrastructure-v1.yaml pre-reg
---

# Stage: CPCV infrastructure build + calibration

## Task

Execute the build scoped by `docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1.yaml`
(committed `0e33df2d`).

Build Combinatorial Purged Cross-Validation with embargo per Lopez de Prado
2020 ML for Asset Managers Ch 7.  Run the 3 pre-registered calibration
hypotheses (H1 known-null, H2 known-edge, H3 embargo sensitivity) against
synthetic known-state data to validate the implementation.

Only proceed to integration (wire into `strategy_validator._check_criterion_8_oos`
as opt-in `cpcv_fallback` kwarg) if all 3 hypotheses pass their pre-registered
thresholds.  Any kill trigger stops the build — do not relax thresholds.

## Scope Lock

- trading_app/cpcv.py
- tests/test_trading_app/test_cpcv.py
- research/cpcv_calibration_v1.py
- docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1-postmortem.md

blast_radius: new module trading_app/cpcv.py (no existing callers to break); new test file tests/test_trading_app/test_cpcv.py (no existing tests affected); new research script research/cpcv_calibration_v1.py (one-off evaluation, no production path); new postmortem docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1-postmortem.md (pre-reg companion, write-only). No changes to strategy_validator.py this stage — integration of cpcv_fallback kwarg is gated on calibration pass and will be a follow-on stage if earned. No schema changes. No allocator or deployment-gate consumers touched. Parent authority: pre-reg file 2026-04-21-cpcv-infrastructure-v1.yaml (commit 0e33df2d); Amendment 3.2 in pre_registered_criteria.md.

## Acceptance criteria

1. `trading_app/cpcv.py` exists with `cpcv_splits`, `cpcv_evaluate`, `cpcv_fold_t_statistic` — each with docstring citing the pre-reg.
2. `tests/test_trading_app/test_cpcv.py` passes — index-disjointness, embargo correctness, deterministic partitioning.
3. `research/cpcv_calibration_v1.py` runs H1, H2, H3 with fixed seeds and writes a postmortem markdown reporting numeric pass/fail against pre-registered thresholds.
4. All 3 calibration hypotheses pass OR the postmortem documents exactly which kill criterion triggered and infrastructure is parked (no integration commit).
5. Drift check passes; no dead code; canonical-delegation: no re-encoding of filter logic.

## Non-goals

- Not wiring `cpcv_fallback` into `strategy_validator` this stage (deferred to next stage if calibration passes).
- Not running CPCV on any deployed-lane real data (reserved for a separate pre-reg, since that would cross into sizing-overlay experimentation).
- Not changing the holdout boundary.
- Not touching `lane_allocator.py` or `portfolio.py`.
