---
task: "Fix HIGH defect in baf99cfe capital gates — _check_survival_report / _check_sr_state SKIP (pass) when signal_only=False + profile_id=None, routing orders with zero C11/C12 evidence validated"
mode: IMPLEMENTATION
slug: baf99cfe-capital-gate-profile-none-failclosed
scope_lock:
  - scripts/run_live_session.py
  - tests/test_scripts/test_run_live_session_preflight.py
  - docs/runtime/stages/baf99cfe-capital-gate-profile-none-failclosed.md
---

## Plan Reference

Track 1 of the two-track capital/canonical review (this session). `baf99cfe`
("Harden live capital evidence gates") landed on main without the mandatory
adversarial-audit gate (`institutional-rigor.md` §2, capital path). Retrospective
independent-context `evidence-auditor` audit (this session) returned SEVERITY HIGH:
3/4 hypotheses CONFIRM fail-closed; H2 found a real defect.

## Defect (REPRO confirmed by execution)

`_check_survival_report` (Criterion 11) and `_check_sr_state` (Criterion 12),
both at `scripts/run_live_session.py:264-306`, SKIP (return CheckResult(True))
when `ctx.profile_id is None`. But a `--demo` or `--live` single-instrument
session launched WITHOUT `--profile` runs with `signal_only=False, profile_id=None`
— order routing is ACTIVE (orchestrator initializes order_router when
`signal_only=False`, `session_orchestrator.py:641`) yet both capital-evidence
gates pass without reading any C11/C12 evidence (the lifecycle state is
profile-keyed; with profile_id=None there is nothing to read, so the gate that
exists to block unverified capital is silently bypassed).

REPRO matrix (executed):
```
signal_only=True  profile_id=None    -> SKIP(pass)   [correct: no routing]
signal_only=False profile_id=None    -> SKIP(pass)   [DEFECT: routes orders, no evidence]
signal_only=False profile_id=topstep -> validates    [correct]
```

## Purpose

Make the SKIP precise: skip only when no order routing occurs (`signal_only=True`).
When routing is active (`signal_only=False`) AND `profile_id is None`, FAIL-CLOSED
— there is no profile-keyed C11/C12 evidence to validate, which is exactly the
unverified-capital state the gate exists to block.

## Approach

In both `_check_survival_report` and `_check_sr_state`, reorder/refine the guard:
1. `if ctx.signal_only:` → SKIP (unchanged — signal-only routes no orders;
   verified at `session_orchestrator.py:641` order_router=None when signal_only).
2. `if ctx.profile_id is None:` → **CheckResult(False, FAILED)** with a message
   directing the operator to launch with `--profile` (capital evidence gates
   require a profile-keyed lifecycle). Replaces the prior SKIP(pass).
3. Remaining lifecycle-read logic unchanged.

## Caller safety (verified, institutional-rigor §4)

Only sanctioned demo/live launch path is `START_BOT.bat:85` which ALWAYS passes
`--profile %ACTIVE_PROFILE%`. Dashboard "start" resolves the active profile from
bot_state. No sanctioned `--demo`/`--live` launch omits `--profile`. The fail-closed
branch is only reachable via raw ad-hoc CLI — exactly the unverified path to block.
FAIL-CLOSED breaks zero legitimate workflow.

## Blast Radius

- scripts/run_live_session.py — two preflight check functions; tighten the
  profile_id-None branch from SKIP(True) to FAILED(False) for routing sessions.
  No change to signal_only SKIP, no change to the lifecycle-read happy path, no
  change to PREFLIGHT_CHECKS ordering or count.
- tests/test_scripts/test_run_live_session_preflight.py — add 2 regression tests
  (the untested defect corner): survival + sr-state FAIL-CLOSED when
  signal_only=False + profile_id=None. Existing 50+ tests must still pass.
- Reads: none new. Writes: none. No schema, no DB.
- Reversibility: git revert; pure control-flow refinement.

## Acceptance

1. `_check_survival_report(ctx)` with `signal_only=False, profile_id=None` →
   `passed is False`, message names the missing-profile cause.
2. `_check_sr_state(ctx)` same input → `passed is False`.
3. `signal_only=True, profile_id=None` → still SKIP(True) (no regression on the
   raw-baseline signal-only path).
4. `signal_only=False, profile_id="topstep_50k_mnq_auto"` → still validates
   lifecycle (existing tests `test_survival_report_check_blocks_capital_profile`,
   `test_sr_state_check_blocks_missing_state` unchanged).
5. `test_capital_state_checks_skip_signal_only` still passes.
6. Full preflight suite green; `python pipeline/check_drift.py` count unchanged
   from baseline.

## Self-review gotchas observed before coding

1. The lifecycle state is profile-keyed — a hard FAIL on profile_id=None for
   signal-only would break the legitimate raw-baseline evidence-accumulation
   path. The signal_only SKIP MUST be checked FIRST and remain a pass.
2. Both functions have identical guard structure — apply the same fix to both,
   keep messages distinct (C11 vs C12) for operator diagnosis.
3. `PREFLIGHT_CHECKS` order/count unchanged — `test_preflight_checks_is_an_ordered_list`
   and `test_no_hardcoded_checks_total_constant` are regression tripwires.
