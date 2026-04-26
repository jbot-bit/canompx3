---
slug: hwm-stage1-gap1-none-reason-contract-guard
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1.5
of: 4
created: 2026-04-26
updated: 2026-04-26
parent_design: docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md (v3)
deferred_finding: docs/ralph-loop/deferred-findings.md — STAGE1-GAP-1
audit_basis: evidence-auditor pass on commit 68c63482 (verdict CONDITIONAL — Stage 2 cleared with this LOW gap deferred to be closed as item-zero of Stage 2)
audit_gate: `.claude/rules/adversarial-audit-gate.md` — small judgment commit on truth-layer path; audit fires after this commit before Stage 2 tracker work begins
task: HWM Stage 1 fix-up — close STAGE1-GAP-1 silent contract-drift path. After Stage 1's None-guard added `reason is not None and "WARN" in reason`, the elif short-circuits to False on `(False, None)` with zero log/notify visibility. Add defensive `elif reason is None:` branch that logs a contract-drift warning + matching test in `TestHWMWarningTierNotifyDispatch`.
---

# Stage: hwm-stage1-gap1-none-reason-contract-guard

mode: IMPLEMENTATION
date: 2026-04-26

scope_lock:
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_session_orchestrator.py
  - docs/ralph-loop/deferred-findings.md
  - docs/runtime/stages/hwm-stage1-gap1-none-reason-contract-guard.md

## Why

Stage 1 audit-gate on commit `68c63482` returned CONDITIONAL with one residual LOW silent-failure path:

> When `check_halt()` returns `(False, None)`, the elif `reason is not None and "WARN" in reason` short-circuits to False, the if-elif block exits, and there is no log/notify — operator has zero visibility into a tracker contract drift.

LOW because `account_hwm_tracker.py:392-435` `check_halt()` never returns `(False, None)` in any current code path. The risk is contract drift on a future tracker refactor or mock misconfiguration in integration tests. The auditor's exact recommendation:

> Add a defensive `else if reason is None: log.warning("HWM check_halt returned (False, None) — tracker contract drift; expected non-None reason string")` plus a matching test.

Closing this as item-zero before Stage 2 (per the deferred-findings ledger entry) preserves the audit-gate cadence: each judgment commit on a truth-layer path gets its own audit pass, and Stage 2's tracker work does not co-mingle with Stage 1 fix-up.

Grounding: `.claude/rules/institutional-rigor.md` § 6 (no silent failures); `.claude/rules/adversarial-audit-gate.md` (deferred audit findings must be closed or explicitly deferred — not silently dropped).

## Behavior change

`session_orchestrator.py:1601` adds a third branch after the existing `elif reason is not None and "WARN" in reason` branch:

```
elif reason is None:
    log.warning(
        "HWM check_halt returned (False, None) — tracker contract drift; "
        "expected non-None reason string"
    )
```

No notify dispatch — this is a contract-drift visibility log, not an operator alert. Per auditor's exact phrasing, log-only is sufficient because the path is theoretically unreachable in production code today.

The existing test `test_hwm_check_halt_none_reason_does_not_raise` still passes — it asserts no spurious `_notify` and no `log.error` from "HWM tracker update/check raised". The new log entry is at WARNING level with a different substring, so the existing assertions hold.

## Acceptance criteria

1. Edit limited to one new `elif` branch on `session_orchestrator.py` after the existing `"WARN" in reason` elif. Diff size: ≤6 lines added.
2. New branch logs at WARNING level with the exact phrase `tracker contract drift` so future readers grep-find this guard.
3. New branch does NOT dispatch `_notify` (auditor explicit — contract drift is not an operator-actionable event).
4. New test `test_hwm_check_halt_false_none_logs_contract_drift_warning` in `TestHWMWarningTierNotifyDispatch` asserts the log line is emitted, no notify is dispatched, no exception propagates.
5. Existing `test_hwm_check_halt_none_reason_does_not_raise` continues to pass unchanged (the test was the original Stage 1 audit-gate fix-up test for the None-guard; this new branch must not regress it).
6. Existing `test_hwm_check_halt_raises_is_caught_silently` continues to pass (mutation guard — the new elif must not perturb the bare-except behavior).
7. `docs/ralph-loop/deferred-findings.md` STAGE1-GAP-1 row moves from "Open Findings" to "Resolved Findings" with this commit's hash.
8. Drift check passes at full count.
9. Adversarial-audit pass returns PASS or all CONDITIONAL findings closed.

## Tests required (mutation-proof)

| Name | What it pins |
|---|---|
| `test_hwm_check_halt_false_none_logs_contract_drift_warning` | New behavior: when `check_halt() == (False, None)`, the elif branch logs WARNING containing the substring `tracker contract drift`, asserts `_notify.call_count == 0`, asserts no exception. Mutation: removing the new elif → no log captured → assertion fails. |

## Blast radius

Single elif branch in the orchestrator's 10-bar HWM-poll block. No tracker change, no test rewrite. Companion test added in the same `TestHWMWarningTierNotifyDispatch` class. Drift count unchanged.

Other call paths into `_notify`, the kill-switch, the emergency-flatten, and the bare-except are untouched.

## Rollback

`git revert` of the single judgment commit restores prior behavior. No persisted state changed.

## Audit-gate hand-off

After this commit lands, dispatch `evidence-auditor` per `.claude/rules/adversarial-audit-gate.md` before Stage 2 tracker work begins. Auditor scope:
- This commit only (one elif branch + one test + ledger row update).
- Verify the new branch does not perturb halt-path or warn-path call ordering.
- Verify the deferred-findings row move is honest (hash matches the commit that fixes it).
