# Stage: hwm-warning-tier-notify-dispatch

mode: IMPLEMENTATION
date: 2026-04-25
parent_design: docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md (v3)
audit_basis: evidence-auditor pass 2026-04-25 — verdict CONDITIONAL with this stage in DO-NOT-CHANGE list (Stage 1 design sound). v3 audit ratified Stage 1 unchanged.
audit_gate: `.claude/rules/adversarial-audit-gate.md` — fires after this judgment commit before Stage 2 dispatches.

scope_lock:
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_session_orchestrator.py
  - docs/runtime/stages/hwm-warning-tier-notify-dispatch.md

## Why

The HWM tracker's `check_halt()` returns a string containing the substring `WARN` whenever dollar drawdown crosses 50% or 75% of the account DD limit but has not yet reached the halt threshold. The 10-bar equity-poll path in the orchestrator already dispatches `_notify` on the halt branch (`session_orchestrator.py:1598`) but logs only on the warning branch (`:1601-1602`).

On a 24-hour unattended session, drawdown can cross 75% with the operator unaware until the full halt fires at 100%. The 50/75 warning tiers exist precisely to give Telegram-visible advance notice before the kill switch — they are operationally dead today.

Auditor named this the highest-priority fix: smallest diff, highest safety-margin impact, no architectural change.

Grounding: `.claude/rules/institutional-rigor.md` § 6 (no silent failures).

## Behavior change

When `check_halt()` returns a non-halted result whose reason string contains the warning marker, the orchestrator dispatches an operator-visible message via the existing notification helper and continues to log at warning level. When the reason string is the OK case, behavior is unchanged.

The single-line edit at `session_orchestrator.py:1601-1602`:

- Before: `elif "WARN" in reason: log.warning("HWM: %s", reason)`
- After: `elif "WARN" in reason: log.warning("HWM: %s", reason); self._notify(f"HWM WARNING: {reason}")`

(Exact phrasing of the notify message and code style is at the implementer's discretion within the acceptance criteria; the test suite pins the dispatch behavior, not the formatting.)

## Acceptance criteria (verbatim from design § 4)

1. Warning at 50% triggers operator dispatch exactly once per poll, with the full reason string including dollar amounts.
2. Warning at 75% triggers operator dispatch exactly once per poll.
3. OK case dispatches nothing.
4. Halt branch is unchanged (existing dispatch and kill-switch path verified intact by mutation test). Call ORDER preserved: `_notify` THEN `_fire_kill_switch` THEN `_emergency_flatten`.
5. Drift check passes at full count.
6. Adversarial-audit pass returns PASS or all CONDITIONAL findings closed.

## Tests required (mutation-proof — strengthened per v3 audit)

| Name | What it pins |
|---|---|
| `test_hwm_warning_50_dispatches_notify` | Inject a 50% warning reason into a stub tracker on the 10-bar poll path; assert exact `_notify` call with exact reason substring including dollar amounts |
| `test_hwm_warning_75_dispatches_notify` | Same, at 75% |
| `test_hwm_warning_generic_substring_match_not_literal` | Mutation guard: assert that a future `WARNING_60` tier (containing substring `WARN`) also dispatches. Test the dispatch logic is generic (`"WARN" in reason`) not literal (`"WARNING_50" in reason`) |
| `test_hwm_ok_does_not_dispatch_notify` | Mutation guard: ensure OK case is silent on Telegram |
| `test_hwm_halt_path_unchanged_by_warning_wiring` | Regression: halt branch still notifies, fires kill switch, calls emergency flatten — and the call ORDER is `_notify` THEN `_fire_kill_switch` THEN `_emergency_flatten` (assert via `mock.call_args_list` index ordering, not just presence). Strengthened per v3 design audit |

## Blast radius

Limited to a single block (lines 1601-1602) in the orchestrator's 10-bar HWM-poll path. Other call paths into `_notify` are unaffected. The kill-switch and emergency-flatten paths are not touched. The risk manager, broker layer, and tracker itself are not touched.

Ralph collision check: v5.2 plan iters 181 (R4 live-signals jsonl), 183 (R5 heartbeat re-notify), 185 (F7 fill poller), 187 (Pass-three magic numbers), 188 (silent-gap cleanup) — none touch line 1601 region. Clear.

## Risks acknowledged

- 6-Telegram-messages-per-hour spam during DD warning band. UNGROUNDED operationally accepted; documented in design § 2 honest-grounding-gaps. If observed as too noisy in operation, rate-limiting is a follow-up stage; not in scope here (would expand Stage 1 blast radius).

## Done criteria

- All 5 tests above pass — output captured in commit body
- All existing `test_session_orchestrator.py` tests still pass
- `python pipeline/check_drift.py` returns full count clean
- Self-review per `.claude/rules/institutional-rigor.md` § 1 captured in commit body
- Adversarial-audit pass via `evidence-auditor` subagent on the commit before Stage 2 dispatches
- This stage doc deleted on close, with the closure note in HANDOFF.md
