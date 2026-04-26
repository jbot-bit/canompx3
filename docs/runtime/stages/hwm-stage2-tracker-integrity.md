---
slug: hwm-stage2-tracker-integrity
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 2
of: 4
created: 2026-04-26
updated: 2026-04-26
parent_design: docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md (v3)
prior_stage_audit: commit 45720109 audit-gate verdict PASS (2026-04-26) — Stage 2 cleared to dispatch
audit_gate: `.claude/rules/adversarial-audit-gate.md` — fires after this judgment commit before Stage 3 dispatches
task: HWM Stage 2 — tracker integrity package on `trading_app/account_hwm_tracker.py`. Adds optional `notify_callback`, fail-closed 30-day stale-state raise, 24h soft warning, corrupt-state notify, poll-recovery notify, persist-IO notify, UNGROUNDED labels on 6 constants, @canonical-source annotation block on the eod_trailing class docstring, and a shared `state_file_age_days(path)` pure helper. 16+ mutation-proof unit tests + 5 integration scenarios.
---

# Stage: hwm-stage2-tracker-integrity

mode: IMPLEMENTATION
date: 2026-04-26

scope_lock:
  - trading_app/account_hwm_tracker.py
  - tests/test_trading_app/test_account_hwm_tracker.py
  - tests/test_trading_app/test_account_hwm_tracker_integration.py
  - docs/runtime/stages/hwm-stage2-tracker-integrity.md

## Why

The HWM tracker has no operator-visible channel — its only output is the Python logger. When state is stale, corrupt, or persistence fails, the operator sees only a log file unless someone is actively watching it. Three CRITICAL audit findings and four SILENT findings collapse into one architectural primitive: an optional `notify_callback` the tracker uses to escalate state-integrity events. Plus six UNGROUNDED constants get explicit honest annotations.

Grounding (per design v3 § 2 — verified against `docs/research-input/topstep/` at design audit + this stage's pre-execution sanity check 2026-04-26):

- `topstep_xfa_parameters.txt:39` — XFA day-1 starting balance is $0 (verbatim verified)
- `topstep_xfa_parameters.txt:289` — "If you break a rule, your Express Funded Account will be liquidated immediately" (verbatim verified)
- `topstep_xfa_parameters.txt:349-351` — 30-day inactivity-closure rule (verbatim verified). The 30-day FIGURE is borrowed by analogy for state-file staleness; the design v3 honesty disclaimer (these are distinct concepts: TopStep's rule fires on no broker trades; this threshold fires on no bot equity polls) is preserved verbatim.
- `topstep_mll_article.md` — EOD trailing mechanic anchor for the class docstring annotation
- `.claude/rules/institutional-rigor.md` § 6 (no silent failures), § 7 (UNGROUNDED labelling)
- `.claude/rules/integrity-guardian.md` § 2 (canonical sources — never re-encode), § 3 (fail-closed)

## Pre-execution sanity-check improvements over design v3

Three additions baked into this stage doc beyond design v3, identified by the 2026-04-26 sanity-check:

1. **Pass Three forward-compatibility (NEW IMPROVEMENT):** `docs/runtime/stages/pass-three-magic-number-drift-check.md` lands a drift check that requires `Rationale:` (case-insensitive) within ±5 lines of any UPPER_SNAKE_CASE numeric literal where `abs(value) > 10` in `trading_app/live/`. Design v3's `# UNGROUNDED — operational heuristic.` comment alone would FAIL Pass Three's regex. This stage doc requires every UNGROUNDED comment to ALSO contain a `Rationale:` block, satisfying both `institutional-rigor.md` § 7 (honesty) AND Pass Three's discipline rule (Carver Ch. 4). Note: Pass Three scopes `trading_app/live/` only, NOT `trading_app/account_hwm_tracker.py` directly — but the `Rationale:` discipline is being applied here anyway as institutional default.

2. **Boundary-direction explicit (CLARITY IMPROVEMENT):** `>= 30 days RAISES` because we measure file AGE — a file 30 days old has been silent for the entire window. TopStep's "more than 30 days" rule (exclusive boundary on broker activity) is a DIFFERENT semantic ("inactivity = no trades during"). Each constant carries an inline comment naming the boundary direction so a future reader does not have to derive it.

3. **Test pattern reference (CLARITY IMPROVEMENT):** Existing tests at `test_account_hwm_tracker.py:481-720` use `patch("trading_app.account_hwm_tracker.datetime")` to control wall-clock for boundary checks. New staleness-boundary tests reuse this pattern — AC explicit so implementer doesn't reinvent.

## Behavior changes (verbatim from design v3 § 5, with sanity-check improvements applied)

### B1. Optional `notify_callback` constructor parameter

```
def __init__(self, ..., *, notify_callback: Callable[[str], None] | None = None):
    self._notify_callback = notify_callback
```

Default `None` preserves all existing behavior — every current call site (3 in pre_session_check, 1 in session_orchestrator at line 698, 1 in weekly_review's read-only path, plus all existing tests) continues to construct without a callback. The orchestrator wires the callback in Stage 3.

### B2. Stale-state load — fail-closed at 30 days

When `_load_state()` reads a state file whose `last_equity_timestamp` is 30 calendar days or older (`>= 30 days`), raise `RuntimeError` with:
- prefix token `STALE_STATE_FAIL`
- absolute file path
- exact age in days
- repair recipe: archive (`.STALE_<YYYYMMDD>.json` rename) or delete

**No env-var bypass** — auditor v2 rejected this as creating an undocumented bypass on a safety gate. Resolution paths are file-system actions only (archived files visible to operator).

The raise message MUST NOT contain `@canonical-source`, `topstep_xfa_parameters.txt:349`, or any phrase claiming the 30-day figure is grounded by TopStep. The figure is borrowed by analogy; the message says so.

### B3. Stale-state load — soft warning at 24 hours to 30 days

When `_load_state()` reads a state file whose `last_equity_timestamp` is 24 hours or older AND less than 30 days, emit exactly ONE `log.warning` naming the age in days. Under 24 hours: silent (suppresses noise during normal-restart cycles).

If `notify_callback` is set, also dispatch the warning via the callback (operator-visible).

### B4. Corrupt-state recovery

The existing `try/except (json.JSONDecodeError, ValueError, TypeError)` block at `_load_state` lines 165-177 logs `log.error`, copies the corrupt file to a `_CORRUPT_<TS>.json` backup, and reinits to fresh state. Behavior unchanged.

NEW: if `notify_callback` is set, dispatch an operator-visible message naming the file path and the exception. Wrap the dispatch in a narrow try/except so a callback failure does NOT break tracker construction.

### B5. Poll-failure recovery notify

In `update_equity(current_equity)` at the success branch (currently line 325 — `self._consecutive_poll_failures = 0`), capture the prior count BEFORE resetting. If the prior count was non-zero AND `notify_callback` is set, dispatch ONE message containing the prior count.

Steady-state successful polls (prior count was already zero) MUST NOT dispatch — mutation guard in tests.

### B6. Persistence I/O failure notify

In `_save_state()`, wrap the `tmp.write_text(json.dumps(data, indent=2))` and `tmp.replace(self._state_file)` lines in a narrow `try/except OSError` (NOT broad `Exception`):
- On `OSError`: if `notify_callback` is set, dispatch ONE message with prefix token `STATE_PERSIST_FAIL`, the file path, and `repr(exc)`. THEN re-raise the original `OSError` so callers see the failure (existing semantics preserved — no behavior change for callers).
- The `OSError` path MUST NOT increment `_consecutive_poll_failures` — that counter is for broker-poll failures (a different signal class). Mutation guard in tests.

### B7. UNGROUNDED + `Rationale:` annotations on six constants (Pass Three forward-compat)

For each of the six constants below, insert a comment block within ±5 lines of the assignment containing BOTH the `UNGROUNDED` label AND a `Rationale:` block:

| Constant | Status | Comment template |
|---|---|---|
| `_MAX_SESSION_LOG = 30` (existing, line 63) | UNGROUNDED operational | `# UNGROUNDED — operational default.\n# Rationale: ~5 trading days at 6 sessions/day. Bounds session-log retention so persisted JSON does not grow unboundedly. No literature citation.` |
| `_MAX_CONSECUTIVE_POLL_FAILURES = 3` (existing, line 64) | UNGROUNDED operational | `# UNGROUNDED — operational default.\n# Rationale: ~30 min visibility window at 10-bar polling cadence. Long enough to absorb a transient broker outage; short enough to halt before DD ages out of view. No literature citation.` |
| Warning thresholds at `_build_state` (existing, line 523-526 — `0.75`, `0.50`) | UNGROUNDED operational | Single comment block above the warning constants. |
| `_STATE_STALENESS_FAIL_DAYS = 30` (NEW) | UNGROUNDED operational, figure borrowed by analogy | `# UNGROUNDED — operational heuristic.\n# Rationale: the 30-day figure is borrowed by analogy from TopStep's account-trading-inactivity window\n# (topstep_xfa_parameters.txt:349-351) but the concept is distinct: that rule fires on no broker trades;\n# this threshold fires on no bot equity polls. Operator may legitimately trade manually with no bot running,\n# in which case this raise is a false positive that the operator must explicitly resolve via state-file\n# archive or delete. Boundary direction: >= 30 days raises (file age, not match window).` |
| `_STATE_STALENESS_WARN_DAYS = 1` (NEW) | UNGROUNDED operational | `# UNGROUNDED — chosen to suppress warning during normal-restart cycles while surfacing meaningful absences.\n# Rationale: 24h floor is the explicit suppression boundary. Under 24h: silent. 24h to 30 days: log.warning + notify.\n# >= 30 days: raises (handled by _STATE_STALENESS_FAIL_DAYS).` |

### B8. `@canonical-source / @verbatim / @audit-finding` block on class docstring

The `AccountHWMTracker` class docstring at lines 67-78 currently describes the eod_trailing mechanic informally as "matches the supported EOD trailing mechanics used in the active project". Add an annotation block matching the existing pattern at `trading_app/live/session_orchestrator.py:680-684`:

```
@canonical-source: docs/research-input/topstep/topstep_xfa_parameters.txt:289 — "If you break a rule, your Express Funded Account will be liquidated immediately"
@canonical-source: docs/research-input/topstep/topstep_mll_article.md — EOD trailing mechanic verbatim
@verbatim: see source files for exact wording
@audit-finding: HWM persistence integrity audit 2026-04-25 (UNSUPPORTED-1) — class docstring did not cite primary sources for the EOD trailing claim. Closed by Stage 2 of design v3.
```

### B9. Shared `state_file_age_days` pure function

```
def state_file_age_days(path: Path) -> float | None:
    """Compute age in days from persisted last_equity_timestamp.

    Pure function. No side effects, no logging. Returns None when:
    - file does not exist, OR
    - file unreadable / parse error / missing/empty timestamp.

    Used by both _load_state's stale-state gate (B2/B3) and Stage 4's
    pre-session inactivity check. Single source of truth — never
    reimplement either consumer.
    """
```

Module-level. Reads via stdlib only (`json.loads`, `datetime.fromisoformat`). No `log` calls, no `print`, no file mutation.

## Acceptance criteria (verbatim from design v3 § 5 — 17 items, with sanity-check improvements applied)

1. Constructor accepts `notify_callback` kwarg; existing callers are unaffected (default `None` preserves prior behavior). All 49 existing tests in `test_account_hwm_tracker.py` continue to pass UNCHANGED (no caplog/notify allowance edits).
2. State file with `last_equity_timestamp` 30 calendar days or older raises `RuntimeError`. The raised message contains `STALE_STATE_FAIL`, the absolute file path, the exact age in days, and the repair recipe (archive or delete). The message does NOT claim canonical-source grounding for the 30-day figure (regression-asserted).
3. State file with `last_equity_timestamp` 24 hours or older AND less than 30 days emits exactly one `log.warning` naming the age in days and continues to load.
4. State file with `last_equity_timestamp` under 24 hours emits no warning.
5. Boundary tests: 30 days minus 1 second warns (does NOT raise); 30 days plus 1 second raises. 24 hours minus 1 second is silent; 24 hours plus 1 second warns. All four boundary tests use `patch("trading_app.account_hwm_tracker.datetime")` per existing pattern at `test_account_hwm_tracker.py:481-720`.
6. Corrupt state with `notify_callback` set produces exactly one operator dispatch in addition to the existing `log.error` and backup behavior.
7. Corrupt state with `notify_callback=None` preserves existing behavior exactly (`log.error`, backup, reinit) — backwards compatible.
8. Poll-failure recovery from a count of 1 or higher produces exactly one operator dispatch with the prior count in the message body.
9. Steady-state successful polls (prior count zero) produce no dispatch (mutation guard against spam).
10. Persistence-write failure simulated by an `OSError` produces exactly one operator dispatch and re-raises the exception. The dispatch message contains `STATE_PERSIST_FAIL`.
11. The class docstring at `account_hwm_tracker.py:67-78` contains the `@canonical-source`, `@verbatim`, and `@audit-finding` tokens (positional check via `AccountHWMTracker.__doc__`).
12. The four pre-existing UNGROUNDED constants AND the two NEW staleness constants all carry comment blocks containing BOTH the `UNGROUNDED` token AND a `Rationale:` block within ±5 lines (sanity-check improvement #1; positional check via greppable token + line-distance assertion).
13. The shared `state_file_age_days` helper is implemented as a pure function — no log output, no file-system side effects beyond reading. Test asserts via `caplog` empty record list AND no extra files on disk after the call.
14. The `notify_callback` dispatch on the corrupt and stale paths during construction is wrapped so that a callback exception does not break tracker construction. Acceptance test injects a callback that raises and asserts construction proceeds.
15. Drift check passes at full count (`pipeline/check_drift.py`).
16. All existing tracker tests pass without modification (49 in `test_account_hwm_tracker.py`). All existing orchestrator tests pass (175 in `test_session_orchestrator.py`).
17. Adversarial-audit pass returns PASS or all CONDITIONAL findings closed.

## Tests required (mutation-proof)

### Unit tests (added to `tests/test_trading_app/test_account_hwm_tracker.py`)

| Name | What it pins |
|---|---|
| `test_load_stale_30_days_plus_1s_raises_with_stale_state_fail_token` | 30+ raises; message contains `STALE_STATE_FAIL`, file path, age in days. Mutation: `>= 30` → `> 30` flips this test. |
| `test_load_stale_30_days_minus_1s_warns_does_not_raise` | Boundary direction reverse: warns, no exception. |
| `test_load_stale_29_days_logs_warning_continues` | Mid-band warn behavior. |
| `test_load_warn_24h_plus_1s_emits_log_warning` | Suppression boundary direction (24h+ warns). |
| `test_load_warn_24h_minus_1s_silent` | Under-24h silent. |
| `test_load_recent_silent` | 1-hour-old state file emits no warning (negative case). |
| `test_load_stale_message_does_not_claim_canonical_grounding` | Mutation guard: assert `@canonical-source` and `topstep_xfa_parameters.txt:349` substrings NOT present in raised message. |
| `test_load_corrupt_invokes_notify_callback_when_provided` | Corrupt branch with callback → 1 dispatch + log.error + backup. |
| `test_load_corrupt_no_callback_preserves_existing_behavior` | Backwards compat — log.error, backup, reinit, no exception. |
| `test_load_corrupt_callback_raises_does_not_break_construction` | Callback exception → construction succeeds, log.error from dispatch failure. |
| `test_poll_recovery_from_one_dispatches_notify_with_prior_count` | Recovery from count=1 fires notify with `1` in message. |
| `test_poll_recovery_from_two_dispatches_notify_with_prior_count` | Mutation guard — count is actual prior count, not constant. |
| `test_poll_recovery_from_zero_does_not_dispatch` | Steady-state silent (mutation guard against spam). |
| `test_save_state_oserror_dispatches_notify_and_reraises_with_persist_fail_token` | Persist failure: notify with `STATE_PERSIST_FAIL`, then `OSError` re-raised. |
| `test_save_state_oserror_does_not_increment_poll_failure_counter` | Persistence and broker-poll failure modes stay separate. |
| `test_eod_trailing_class_docstring_has_canonical_source_annotation_block` | `AccountHWMTracker.__doc__` contains `@canonical-source`, `@verbatim`, `@audit-finding`. |
| `test_ungrounded_constants_have_explicit_label_and_rationale_within_5_lines_above` | Each of 6 constants has UNGROUNDED + Rationale: tokens within ±5 lines (positional, not file-wide grep). Sanity-check improvement #1. |
| `test_state_file_age_days_pure_function_no_logging` | Helper produces no log output and no file-system side effects beyond reading. |
| `test_state_file_age_days_returns_none_on_missing_file` | Helper returns `None` when path does not exist. |
| `test_state_file_age_days_returns_none_on_corrupt_json` | Helper returns `None` on parse error. |

### Integration tests (NEW FILE: `tests/test_trading_app/test_account_hwm_tracker_integration.py`)

5 scenarios per design v3 § 10:

1. 19-day-old state file (mirrors audit case): exactly one log.warning, no exception, no notify with callback unset, one notify with callback set.
2. 30-day-plus-1-second: `RuntimeError` with `STALE_STATE_FAIL` token, file path, age in days.
3. 30-day-minus-1-second: warning, no exception.
4. Corrupt JSON file with notify-callback: exactly one notify dispatch in addition to `log.error` and backup file creation.
5. Synthetic `OSError` injected into the save-state path: `STATE_PERSIST_FAIL` notify dispatch and re-raise.

## Blast Radius

- `trading_app/account_hwm_tracker.py` — new constructor kwarg, two new constants, one new module-level helper, three behavior changes in `_load_state`/`update_equity`/`_save_state`, class docstring annotation. ~120 added lines.
- `tests/test_trading_app/test_account_hwm_tracker.py` — ~20 new tests, no edits to existing tests.
- `tests/test_trading_app/test_account_hwm_tracker_integration.py` — NEW FILE, 5 scenarios.

No orchestrator change. No pre_session_check change. No weekly_review change. (All those land in Stage 3.)

The 30-day raise is a NEW fail-closed path. State file `account_hwm_20092334.json` (1 day old at design date 2026-04-25, now 20 days old at 2026-04-26) is approaching the warn band but well clear of the 30-day raise. Pre-Stage-3 the raise is dormant — no orchestrator yet constructs the tracker with notify_callback. Stage 3 wiring + Stage 4 pre-session inactivity check land separately under their own audit-gates.

## Rollback

`git revert` of the single judgment commit. No persisted state file is modified by this stage; only loading semantics change. Stage 3 and Stage 4 depend on Stage 2 outputs — if Stage 2 reverts later, those stages must revert first.

## Audit-gate hand-off

After this commit lands, dispatch `evidence-auditor` per `.claude/rules/adversarial-audit-gate.md`. Auditor scope:
- This commit only.
- Verify all 17 acceptance criteria with execution evidence.
- Verify no regression in existing 49 tracker tests.
- Verify no regression in 175 orchestrator tests (callback default-None backward compat).
- Independent trace: does the new staleness raise interact correctly with existing `_advance_hwm` and halt logic? (Should not — `_load_state` runs before any of those.)
- Verify Pass Three forward-compat: do the new constants' UNGROUNDED+Rationale comments survive a future `pipeline/check_drift.py` rationale-discipline scan?

Stage 3 dispatches only after this audit returns PASS.
