---
stage: 2026-06-05-c11-s1-bracket-parity-audit
mode: VERIFICATION
status: IN_PROGRESS
verdict: PENDING-INDEPENDENT-AUDIT
verdict_note: "IMPLEMENTER SELF-REVIEW ONLY (not gate-clearing): parity fix appears sound; guard defect at session_orchestrator.py:2166 FIXED this pass (fail-closed on present-but-<=0 event.risk_points; RED->GREEN test added; 17+22 green). Independent evidence-auditor (Stage 1c) NOT yet run — gate remains OPEN. Earlier 'DONE/CONDITIONAL-RESOLVED' framing + 'independent evidence-auditor dispatched' claim in the closeout were CORRECTED 2026-06-05: no independent subagent had run."
scope_lock_extension: "operator GO 2026-06-05 to fix guard in-stage: trading_app/live/session_orchestrator.py helper + tests/test_trading_app/test_session_orchestrator.py (new test only)."
worktree: C:/Users/joshd/canompx3-c11-s1-audit
branch: session/joshd-c11-s1-bracket-audit
base: origin/main f680772b
audit_target: 9b3fc530
date: 2026-06-05
---

# Stage 1 — Close adversarial-audit gate on bracket-parity fix (9b3fc530)

## Gate authority
`.claude/rules/adversarial-audit-gate.md` — non-negotiable for a judgment-classified
CRITICAL change touching `trading_app/live/`. Self-review is necessary but NOT
sufficient. The audit gate for 9b3fc530 is recorded OPEN in
`docs/audit/results/2026-06-03-bracket-risk-parity-closeout.md` (Limitations).

## Why this gates C11
The bracket-parity fix governs the per-trade effective stop distance. The C11 survival
sim and the live order path BOTH depend on that distance. If the parity math is wrong,
the $2,038.84 baseline and the "$1,594 clears $1,800" cap claim are computed on broken
math. Stage 1 must PASS before Stage 2 (cap pre-registration) writes acceptance gates
against those numbers.

## scope_lock (this stage edits ONLY these)
- `docs/runtime/stages/2026-06-05-c11-s1-bracket-parity-audit.md` (this file)
- `docs/audit/results/2026-06-03-bracket-risk-parity-closeout.md` (record verdict only)

NO production code. NO prop_profiles.py. NO cap wiring. NO arming.

## Audit target diff (read-only)
Commit 9b3fc530 — 4 files, +257/-23:
- `trading_app/live/session_orchestrator.py` (live order surface)
- `trading_app/account_survival.py` (replay/sim side)
- `tests/test_trading_app/test_session_orchestrator.py`
- `tests/test_trading_app/test_account_survival.py`

## Eight falsification points for the independent auditor
Scope = combined post-fix live-path state (shipped 9b3fc530 parity logic + this pass's
present-but-<=0 fail-closed guard). The auditor must NOT pre-accept the self-review.
1. Both live bracket-construction paths route through the single stop-distance helper
   (callers :2206/:2646); prove no third bypassing path.
2. event.risk_points present -> used raw; only median present -> stop_multiplier once;
   neither present -> fail closed (flatten, not a guessed bracket). Confirm the
   `is not None` + `>0` split closes the present-but-zero hole.
3. Replay side (account_survival.py) and live side compute the SAME effective stop
   distance (bidirectional parity — the point of 9b3fc530).
4. No scope leak into exit-record actual-R, journal dollar fallback, naked-position
   fail-closed behavior.
5. A bar with TWO entry events yields correct stop distance for BOTH (canonical
   multi-event counterexample from the gate rule's proof case).
6. Tight-stop handling runs before the helper reads event.risk_points, so the trusted
   value is already the effective post-tight-stop distance.
7. Cross-path consistency: bracket path now FLATTENS on present-but-zero while the
   exit-record/journal path (:2086, :2121) COALESCES zero->median. Prove these are
   intentionally different decision points (placement vs accounting), not a latent bug.
8. Absent-vs-zero reachability: prove whether risk_points=None (absent) is reachable
   through ExecutionEngine. If reachable, "absent -> median fallback" is still a live
   guess and the fix is only partial.

## Required artifact fields (per gate rule)
Verdict (PASS/CONDITIONAL/FAIL); critical issues with file:line; silent gaps;
unsupported assumptions; missing tests; do-not-touch; single highest-priority fix.

## Acceptance
- PASS -> record verdict into closeout doc; Stage 3 (cap wiring) unblocks.
- CONDITIONAL/FAIL -> HARD STOP. Findings route to a fix iteration; no Stage 2, no
  wiring, until closed or written-deferred.

## Baseline anchor (re-run before dispatch)
tests/test_trading_app/test_session_orchestrator.py::TestBracketOrders
tests/test_trading_app/test_session_orchestrator.py::TestF4BracketNakedPosition
