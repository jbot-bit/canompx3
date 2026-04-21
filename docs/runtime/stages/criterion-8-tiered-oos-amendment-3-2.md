---
slug: criterion-8-tiered-oos-amendment-3-2
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
task: Amendment 3.2 — tiered Criterion 8 OOS gate + opt-in power-floor; CPCV infra pre-reg
---

# Stage: Amendment 3.2 tiered Criterion 8 OOS gate

## Task

Resolve the authority-document conflict between `docs/institutional/pre_registered_criteria.md`
Criterion 8 (binary pass/fail at N ≥ 30) and `.claude/rules/backtesting-methodology.md`
RULE 3.2 (N < 30 directional-only). Codify a 3-tier OOS gate with opt-in
power-floor enforcement per Harvey-Liu 2015 p.17 OOS caveat.

Motivation: 2026 holdout depth is 79 trading days → filtered OOS N per lane
is 15–30, well below power to detect +0.04–0.11 R IS effects (MDE ~0.6–0.8 R).
Current binary Criterion 8 false-kills true edges and false-passes noise at
this sample size.

## Scope Lock

- docs/institutional/pre_registered_criteria.md
- trading_app/strategy_validator.py
- tests/test_trading_app/test_strategy_validator.py
- docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1.yaml

## Scope rationale

- `docs/institutional/pre_registered_criteria.md` — amend Criterion 8 body and add Amendment 3.2 section (version history + full spec at end of file).
- `trading_app/strategy_validator.py` — add `_estimate_oos_power` helper; extend `_check_criterion_8_oos` with tier classification, power logging, and opt-in `require_power_floor` reject gate.
- `tests/test_trading_app/test_strategy_validator.py` — add `TestCriterion8PowerFloor` class; update `test_fails_with_oos_below_ratio` to reflect Tier 2 semantics (ratio gate SKIPPED at 30 ≤ N < 100).
- `docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1.yaml` — pre-reg stub for CPCV infrastructure build.

blast_radius: trading_app/strategy_validator.py (add _estimate_oos_power helper; tier + power logic in _check_criterion_8_oos; new kwarg require_power_floor default False); tests/test_trading_app/test_strategy_validator.py (new TestCriterion8PowerFloor class; updated test_fails_with_oos_below_ratio to Tier 2 semantics); docs/institutional/pre_registered_criteria.md (Criterion 8 body + Amendment 3.2); docs/audit/hypotheses/2026-04-21-cpcv-infrastructure-v1.yaml (new pre-reg stub). Downstream consumers of _check_criterion_8_oos: single caller at trading_app/strategy_validator.py:1229 (run_validation) — signature backward-compatible, new kwargs default preserve behavior. Pathway A callers: behavior change at 30 ≤ N < 100 — ratio gate SKIPPED (was applied); sign gate unchanged. Pathway B callers (strict_oos_n=True): no behavior change this amendment. No schema changes. No allocator or deployment-gate consumers touched.

## Blast radius

- Pathway A callers of `_check_criterion_8_oos`: behavior changes at Tier 2 (30 ≤ N < 100) from "apply ratio gate" to "sign gate only". This relaxes one specific rejection path at low N where the ratio estimate is not reliably confirmatory. At Tier 1 (N ≥ 100) behavior is verbatim identical. At Tier 3 (N < 30) behavior is verbatim identical.
- Pathway B callers (`strict_oos_n=True`): no behavior change at this amendment. Pathway-B-specific tightening (require Tier 1) reserved for a follow-on amendment when candidates actually have N ≥ 100 OOS.
- `require_power_floor` is a new opt-in kwarg; default False preserves all existing callers.
- Downstream validators / allocator: no API break.
- Tests: one assertion updated (`test_fails_with_oos_below_ratio` → now expects PASS at N=30 low-ratio case, per Amendment 3.2); ratio-gate coverage moved to a new N=100 Tier 1 test to preserve the gate's test coverage.

## Acceptance criteria

1. All pre-existing tests in `TestCriterion8OOSPositive` and `TestCriterion8StrictMode` still pass (with one updated assertion in `test_fails_with_oos_below_ratio` per Amendment 3.2).
2. New tests in `TestCriterion8PowerFloor` pass — power computation sanity + opt-in gate behavior.
3. `python pipeline/check_drift.py` passes.
4. No dead code introduced; no hardcoded stats.
5. Self-review confirms: delegated to canonical power calc (scipy.stats.nct), no silent failures in power computation (returns None on degenerate cases), no re-encoding of existing logic.

## Non-goals

- Not changing the holdout boundary (immovable per criteria.md:5).
- Not tightening Pathway B beyond current strict-mode semantics.
- Not implementing CPCV itself (this stage writes only the pre-reg stub).
- Not changing the deployment promotion gate (tier info becomes available for future consumption).
