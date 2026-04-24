---
slug: harden-dashboard-action-coordination
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-24
updated: 2026-04-24
task: Harden commit 45f50916 (fix(live) dashboard action coordination) — apply the 8 items approved in the plan at ~/.claude/plans/inspoect-repoi-instpect-resource-imperative-clarke.md.
---

# Stage: Harden Dashboard Action Coordination

## Goal

Apply the 8 hardening items F1-F8 approved in the plan at
`~/.claude/plans/inspoect-repoi-instpect-resource-imperative-clarke.md`
so commit `45f50916` moves from B+ to A/A-. Fixes cover dead branches,
a preflight honesty gap, unlocked module-global state machine, an unused
parameter, a cross-module private import, a duplicated hardcoded threshold,
and missing regression/state-machine tests.

## Scope Lock

- scripts/run_live_session.py
- trading_app/live/bot_dashboard.py
- trading_app/live/instance_lock.py
- tests/test_trading_app/test_instance_lock.py
- tests/test_trading_app/test_bot_dashboard.py

## Blast Radius

All edits are local to a single commit's surface. `_is_pid_alive` rename
covers 4 call sites across 3 files (grep-verified). `_handoff_state` /
`_preflight_cache` lock change covers 7 call sites all in
`bot_dashboard.py`; no external importers. Dead-branch removal in
`run_live_session.py` has 1 internal caller. Constant extraction covers
2 call sites, same file. New tests in `tests/test_trading_app/test_bot_dashboard.py`
are purely additive. No pipeline logic, no trading logic, no DB schema, no
canonical source changes. Out-of-scope (NOT touched): session_orchestrator.py,
bot_dashboard.html, HANDOFF.md, pre-existing smells (broker-status silent
except at line 697, hardcoded port "8080", fallback profile string).

## Approach

Execute in plan's prescribed order (low-risk first):
F4 → F5 → F6 → F7 → F1 → F2 → F3 → F8.

Verify after each item where practical. Run full test suite + drift check +
dead-code sweep at the end.

## Acceptance Criteria

1. All 8 findings resolved.
2. `pytest -xvs tests/test_trading_app/test_bot_dashboard.py tests/test_trading_app/test_instance_lock.py` — all pass including 2 new tests.
3. `python pipeline/check_drift.py` — all checks pass.
4. Dead-code sweep `grep -rn "_is_pid_alive" trading_app/ scripts/ tests/` returns zero hits after F6.
5. Preflight smoke test prints the honest message.
6. code-review self-review on the resulting diff finds no new CRITICAL/HIGH issues.

## References

- Plan file: `C:\Users\joshd\.claude\plans\inspoect-repoi-instpect-resource-imperative-clarke.md`
- Source commit: `45f50916 fix(live): harden dashboard action coordination`
