# HWM Persistence Layer + Adjacent Live-Trading Silence Gaps — Design

**Status:** DESIGN v2 — awaiting approval. NO CODE will be written until user explicitly approves.
**Date:** 2026-04-25
**Author:** Claude (design skill, plan mode)
**Audit basis:** evidence-auditor independent pass on the layer (2026-04-25) — verdict CONDITIONAL. Then a second evidence-auditor pass on this design itself (2026-04-25) — verdict CONDITIONAL with required revisions; v2 incorporates those revisions.
**Related plan (parallel, do not collide):** `docs/plans/2026-04-25-ralph-crit-high-burndown-v5.md`
**Audit gate:** `.claude/rules/adversarial-audit-gate.md` — fires after every judgment commit on truth-layer paths.

## Revision history

- **v1** (initial draft) — failed design audit on three counts: (a) false citation — `topstep_xfa_parameters.txt:349-351` was claimed to ground state-file staleness, but it actually grounds account trading inactivity (different concepts); (b) invented env-var bypass with no codebase precedent; (c) wrong section number for `integrity-guardian.md` (§ 4 cited but content lives at § 2).
- **v2** — citation corrected to honest UNGROUNDED operational heuristic with the 30-day figure borrowed from the TopStep inactivity window but not derived from it; env-var bypass removed in favor of archive-or-delete as the only resolution path; section numbers corrected; Stage 4 redesigned to delegate to a shared computation in the tracker module (eliminates the two-enforcement-surface flaw); cross-stage integration scenario promoted from optional manual script to automated pytest integration test; tests strengthened against 5 specific mutations; downstream behavior-change risks for Stage 3 explicitly flagged; startup-latency hazard for synchronous Telegram inside `_load_state` acknowledged.
- **v3** (this version) — second design audit returned CONDITIONAL with three must-fix items: (a) Stage 3 baseline characterization of `check_hwm_tracker` corrupt-state behavior was factually wrong (it already returns `(False, ...)`, not `(True, ...)`) — Stage 3 reframed from "boolean fix" to "message-format unification" with the divergence corrected to its true form; (b) `weekly_review.py:37,46` is a third hidden consumer of `account_hwm_*.json` via `json.loads`, outside the Stage 3 scope_lock — added to scope_lock so the shared-reader extraction is complete; (c) AC-11 annotation location was misidentified as `update_equity:342` but the eod_trailing docstring is in the class docstring at lines 70-78 — corrected. Plus three lower-priority improvements: (d) Stage 1 halt-path test now pins call ordering; (e) `bot_dashboard.py` removed from Risk 4 (it reads a different file path entirely); (f) volatile state-file ages in § 5 blast radius now carry as-of-date caveat; (g) Stage 2 integration scenarios moved earlier (scenarios 1-5 land with Stage 2; scenarios 6-7 remain in Stage 4) so CI gets earlier coverage; (h) notify-helper-construction-order claim corrected (`_notify` is a class method available throughout `__init__`, not an init-time attribute; the real risk is which instance attributes `_notify` references that may not be set yet).

---

## 1. Purpose

Close the silence gaps in the persistent dollar-based HWM tracking layer (`trading_app/account_hwm_tracker.py`) and the orchestrator code that consumes it (`trading_app/live/session_orchestrator.py`, `trading_app/pre_session_check.py`). The independent audit found three CRITICAL silences, seven SILENT gaps, and seven UNSUPPORTED constants. All of these can mask a real-money drawdown event from the operator until the kill-switch fires at the hard limit.

Why this matters: the prop-firm Maximum Loss Limit is the single hard-fail event for a TopStep XFA account. The tracker is the only software component that turns that broker rule into an operator-visible signal. Every silence in this layer compresses the operator's reaction window. The auditor's highest-priority finding (DD warning tier 50/75 never reaches Telegram) is the canonical example: the warning tiers exist exactly to give advance notice before the halt — and they are operationally dead today on a 24-hour overnight run.

---

## 2. Institutional grounding

Every threshold and constant introduced or modified by this design cites a canonical source from `docs/research-input/topstep/` or a project rule in `.claude/rules/`. Where a constant remains operationally chosen rather than literature-derived, an explicit `UNGROUNDED` annotation is added so the next reader is not misled.

### Primary canonical sources

| Source | Use |
|---|---|
| `docs/research-input/topstep/topstep_xfa_parameters.txt:39` | XFA day-1 starting balance is $0 — used in tracker docstring grounding |
| `docs/research-input/topstep/topstep_xfa_parameters.txt:289` | Rule breach equals immediate liquidation — used in tracker docstring grounding |
| `docs/research-input/topstep/topstep_xfa_parameters.txt:349-351` | TopStep 30-day account-trading-inactivity closure rule. **NOTE:** This grounds the 30-day NUMERIC FIGURE that the design borrows for state-file staleness, but does NOT ground state-file staleness as a concept. The two are distinct: TopStep's rule is about no trades on the broker account; the design's threshold is about no equity polls into the bot. The figure is borrowed by analogy, not derived. See § 2 honest grounding gaps below |
| `docs/research-input/topstep/topstep_mll_article.md` | EOD trailing mechanic verbatim — anchors the eod_trailing docstring grounding |
| `.claude/rules/institutional-rigor.md` § 6 | No silent failures — anchors every notify-callback wiring |
| `.claude/rules/institutional-rigor.md` § 7 | Ground in local resources before training memory; label training-memory claims explicitly — anchors every UNGROUNDED label in this design |
| `.claude/rules/integrity-guardian.md` § 2 | Canonical Sources — Never Hardcode — anchors the pre-session shared-reader extraction (Stage 3) |
| `.claude/rules/integrity-guardian.md` § 3 | Fail-closed mindset (never report success after exception) — anchors the corrupt-state behavior |
| `.claude/rules/adversarial-audit-gate.md` | Per-commit independent audit — anchors the staging cadence |

### Honest grounding gaps (carried forward as labels, not pretended-to-be-grounded)

Every constant or threshold introduced or modified by this design that is not derived from a literature citation, a canonical project source, or a prop-firm rule is listed here. The implementation will carry an `UNGROUNDED` annotation comment at the constant's definition site so future readers see the gap, not just the value.

| Constant or threshold | Grounding status | Annotation pattern in code |
|---|---|---|
| `_MAX_CONSECUTIVE_POLL_FAILURES = 3` (existing) | UNGROUNDED — operational default | `# UNGROUNDED — operational default; ~30 min visibility window at 10-bar polling. No literature citation as of 2026-04-25 design audit.` |
| `_MAX_SESSION_LOG = 30` (existing) | UNGROUNDED — operational default | `# UNGROUNDED — operational default; ~5 trading days at 6 sessions/day. No literature citation.` |
| Warning thresholds 50% and 75% (existing) | UNGROUNDED — operational defaults | Single comment block above the warning constants explaining both are operational defaults |
| `_STATE_STALENESS_FAIL_DAYS = 30` (NEW, Stage 2) | UNGROUNDED operational heuristic; figure borrowed from TopStep inactivity window by analogy, not derived from it | `# UNGROUNDED — operational heuristic. The 30-day figure is borrowed by analogy from TopStep's account-trading-inactivity window (topstep_xfa_parameters.txt:349-351) but the concept is distinct: that rule fires on no broker trades; this threshold fires on no bot equity polls. Operator may legitimately trade manually with no bot running, in which case this raise is a false positive that the operator must explicitly resolve via state-file archive or delete.` |
| `_STATE_STALENESS_WARN_DAYS = 1` (NEW, Stage 2) | UNGROUNDED — chosen to suppress warning during normal-restart cycles while surfacing meaningful absences | Comment notes the absence-detection intent; resolves the v1 contradiction between "every load warns" and "under-1-day silent" by making 24 h the explicit floor |
| `_INACTIVITY_WARN_DAYS = 25` (NEW, Stage 4) | Soft-grounded — 5-day operator-action buffer ahead of `_STATE_STALENESS_FAIL_DAYS = 30` | `# UNGROUNDED — chosen as 5-day operator-action buffer ahead of the 30-day fail-closed threshold. The 5-day figure is operational, not literature-derived.` |
| 6-Telegram-messages-per-hour spam tolerance during DD warning band | UNGROUNDED — operational acceptance | Documented in § 10 failure-mode notes; not a code constant. If observed in operation as too noisy, rate-limiting is a follow-up stage |

### Annotation patterns

The codebase already establishes a `@canonical-source / @verbatim / @audit-finding` block for grounded constants (see `pre_session_check.py:241-249` and `session_orchestrator.py:680-684`). For UNGROUNDED choices the pattern is a plain `# UNGROUNDED — <reason>; <what would be needed to ground it>` comment immediately above the constant or function, no `@verbatim` line. Tests assert the presence of either pattern by greppable token.

---

## 3. Scope

### In scope

| Stage | Files |
|---|---|
| 1 | `trading_app/live/session_orchestrator.py` (one region around line 1601) and its companion test file |
| 2 | `trading_app/account_hwm_tracker.py` and its companion test file |
| 3 | `trading_app/live/session_orchestrator.py` (HWM-construction region around line 698 and EOD-close region around line 3062), `trading_app/pre_session_check.py` (the two HWM-reader functions delegate to the shared reader), `trading_app/weekly_review.py` (the third hidden consumer at lines 37/46 — added per v3 audit; converts to the shared reader), and the three companion test files (`test_session_orchestrator.py`, `test_pre_session_check.py`, `test_weekly_review.py`) |
| 4 | `trading_app/pre_session_check.py` (one new check function), `docs/ralph-loop/deferred-findings.md` (three new escalation rows), `HANDOFF.md` (mark items closed) |

### Out of scope (named, with reasons)

| Item | Reason |
|---|---|
| Ralph commits 6dafda10, 64d0952d, f8f993b7 | Belong to the v5.2 burndown plan and the iter 178 audit pass. Audit-gate rule forbids editing across the iteration boundary |
| `_check_trading_day_rollover` idempotency guard, `_fire_kill_switch` persistence path, `compute_trading_day_utc_range` usage | v5.2 plan do-not-touch list — already audit-verified correct |
| Codex parallel-session WIP files | HANDOFF instruction — leave until Codex commits |
| The R3 numeric values (1800-second stable-run threshold, 50 max reconnects) | Ungrounded but in Ralph's domain — escalated as Stage 4 deferred-findings entries, not changed by this plan |
| Replacing the tracker's existing 50% and 75% warning thresholds | Operationally functional; removing changes downstream behavior. Honest annotation chosen over silent change |
| The two-layer DD architecture itself (RiskManager + AccountHWMTracker) | Audit-verified canonical at session_orchestrator.py:655-664. Preserve, do not rearchitect |
| The `query_equity` returns-None-on-exception contract | WF-07 in deferred-findings: intentional contract; F5 design depends on it; do not change |

---

## 4. Stage 1 — DD warning tier reaches operator (auditor highest-priority)

### Purpose

The tracker's `check_halt` returns a string containing the substring `WARN` whenever the dollar drawdown crosses 50% or 75% of the account's DD limit but has not yet hit the halt threshold. The 10-bar equity-poll path in the orchestrator already dispatches `_notify` on the halt branch but logs only on the warning branch. On a 24-hour unattended session, drawdown can cross 75% with the operator unaware until the full halt fires.

### Files (scope_lock)

- `trading_app/live/session_orchestrator.py` — exactly the warning-branch line in the 10-bar HWM-poll block. No other change.
- `tests/test_trading_app/test_session_orchestrator.py` — new behavior tests for the warning dispatch and the symmetric absence-of-dispatch in the OK case.

### Behavior change

When `check_halt` returns a non-halted result whose reason string contains the warning marker, the orchestrator dispatches an operator-visible message via the existing notification helper and continues to log at warning level. When the reason string is the OK case, behavior is unchanged.

### Acceptance criteria

1. Warning at 50% triggers operator dispatch exactly once per poll, with the full reason string including dollar amounts.
2. Warning at 75% triggers operator dispatch exactly once per poll.
3. OK case dispatches nothing.
4. Halt branch is unchanged (existing dispatch and kill-switch path verified intact by mutation test).
5. Drift check passes at full count.
6. Adversarial-audit pass returns PASS or all CONDITIONAL findings closed.

### Tests required (mutation-proof)

| Name | What it pins |
|---|---|
| `test_hwm_warning_50_dispatches_notify` | Inject a 50% warning reason into a stub tracker on the 10-bar poll path; assert exact notify call with exact reason substring |
| `test_hwm_warning_75_dispatches_notify` | Same, at 75% |
| `test_hwm_ok_does_not_dispatch_notify` | Mutation guard: ensure OK case is silent on Telegram |
| `test_hwm_halt_path_unchanged_by_warning_wiring` | Regression: halt branch still notifies, fires kill switch, calls emergency flatten — and the call ORDER is `_notify` THEN `_fire_kill_switch` THEN `_emergency_flatten` (assert via call-tracker `mock.call_args_list` index ordering, not just presence). v3 strengthening per design audit |

### Blast radius

Limited to a single block in the orchestrator. Other call paths into `_notify` are unaffected. The kill-switch and emergency-flatten paths are not touched. The risk manager, the broker layer, and the tracker itself are not touched.

### Rollback

`git revert` of the single judgment commit restores prior behavior. No persisted state changed.

---

## 5. Stage 2 — Tracker integrity package

### Purpose

The tracker has no operator-visible channel — its only output is the Python logger. When state is stale, corrupt, or persistence fails, the operator sees only a log file unless someone is actively watching it. Three CRITICAL findings and four SILENT findings collapse into one architectural primitive: an optional notify_callback the tracker uses to escalate state-integrity events. Plus the four UNSUPPORTED constants get explicit honest annotations.

### Files (scope_lock)

- `trading_app/account_hwm_tracker.py`
- `tests/test_trading_app/test_account_hwm_tracker.py`

### Behavior changes

| Item | Change | Grounding |
|---|---|---|
| Tracker constructor | Accept an optional notify-callback parameter typed as a string-consuming callable, defaulting to None. When None, behavior is identical to today (silent). When provided, the tracker dispatches operator-visible messages on the integrity events listed below | Single primitive enables every wiring below |
| Stale-state load — fail-closed at 30 days | When the persisted last-equity timestamp is 30 calendar days or older, raise a runtime error during load with a repair-recipe message. The message names: (a) the absolute file path, (b) the exact age in days, (c) an explicit prefix token `STALE_STATE_FAIL` that distinguishes this raise from the existing `FAIL-CLOSED: Account HWM DD limit breached...` raise at session_orchestrator.py:715-718, (d) a one-line repair recipe stating the operator must archive (rename to `<file>.STALE_<YYYYMMDD>.json`) or delete the file. **No env-var bypass** — the only resolution paths are archive or delete. This forces the operator action to be visible in the file system | UNGROUNDED operational heuristic. See § 2 honest grounding gaps |
| Stale-state load — soft warning at 24 hours or older | Compute the persisted age. If age is 24 hours or greater AND less than 30 days, emit a single log.warning naming the age in days. Under 24 hours: silent. This resolves the v1 contradiction between "every load warns" and "under-1-day silent" by setting the 24-hour floor as the explicit suppression boundary | UNGROUNDED — chosen to suppress noise during normal-restart cycles while surfacing meaningful absences. See § 2 |
| Corrupt-state recovery | When the persisted file is unreadable, in addition to the existing log.error and backup behavior, dispatch via notify-callback if present. Behavior when callback is None is unchanged from today | Resolves CRITICAL-3 |
| Poll-failure recovery | When a successful poll resets a previously non-zero consecutive-failure counter, dispatch via notify-callback exactly once with the prior count. Steady-state successful polls do NOT dispatch | Resolves SILENT-2 |
| Persistence I/O failure | Wrap the state-write body in a narrow try-except for `OSError` only (not the broad Exception class). On failure, dispatch via notify-callback if present and re-raise to the caller. This is a different signal class from `update_equity(None)` (which signals broker unreachability) — operator sees "persistence degraded" without confusing the F5 broker-poll counter | Resolves SILENT-3 root; preserves WF-07 contract |
| Constant annotations | Add explicit UNGROUNDED labels with rationale comments above the four existing ungrounded constants. Add `@canonical-source / @verbatim / @audit-finding` annotation block on the eod_trailing docstring matching the pattern at session_orchestrator.py:680-684 | Resolves UNSUPPORTED-1, 2, 3, 4 honestly |
| Shared age computation | Implement a single module-level pure function `state_file_age_days(path: Path) -> float | None` that computes the age in days from the persisted `last_equity_timestamp`. Both `_load_state()` (for the warn / raise gate) and Stage 4's pre-session inactivity check delegate to this single computation. Eliminates the v1 risk of two enforcement surfaces drifting apart | `integrity-guardian.md` § 2 — never re-encode |

### Acceptance criteria

1. Constructor accepts notify-callback parameter; existing callers are unaffected (default None preserves prior behavior).
2. State file with last-equity timestamp 30 calendar days or older raises runtime error. The raised message contains the prefix token `STALE_STATE_FAIL`, the absolute file path, the exact age in days, and the repair recipe (archive or delete). The message does NOT claim canonical-source grounding for the 30-day figure.
3. State file with last-equity timestamp 24 hours or older AND less than 30 days emits exactly one log.warning naming the age in days and continues to load.
4. State file with last-equity timestamp under 24 hours emits no warning.
5. Boundary tests: 30 days minus 1 second warns (does NOT raise); 30 days plus 1 second raises. 24 hours minus 1 second is silent; 24 hours plus 1 second warns.
6. Corrupt state with notify-callback set produces exactly one operator dispatch in addition to the existing log.error and backup behavior.
7. Corrupt state with no callback (callback=None) preserves existing behavior exactly (log.error, backup, reinit) — backwards compatible.
8. Poll-failure recovery from a count of 1 or higher produces exactly one operator dispatch with the prior count in the message body.
9. Steady-state successful polls (prior count zero) produce no dispatch (mutation guard against spam).
10. Persistence-write failure simulated by an `OSError` produces exactly one operator dispatch and re-raises the exception to the caller. The dispatch message contains a distinguishing prefix token `STATE_PERSIST_FAIL`.
11. The `@canonical-source / @verbatim / @audit-finding` annotation block matching the existing pattern at session_orchestrator.py:680-684 is added to the `AccountHWMTracker` class docstring at lines 70-78 (where the eod_trailing mechanic is currently described informally as "matches the supported EOD trailing mechanics used in the active project"). The annotation cites `topstep_xfa_parameters.txt:289` for the rule-breach-equals-immediate-liquidation grounding and `topstep_mll_article.md` for the EOD trailing mechanic verbatim. (v3 correction: v2 incorrectly referenced `update_equity:342` as the location; verified that the eod_trailing mechanic is documented in the class docstring at 70-78, not at the branch site.)
12. The four pre-existing UNGROUNDED constants and the two NEW staleness constants (`_STATE_STALENESS_FAIL_DAYS`, `_STATE_STALENESS_WARN_DAYS`) all carry explicit `# UNGROUNDED` comments with the standardized rationale format.
13. The shared `state_file_age_days` helper is implemented as a pure function (no side effects, no logging) so that both the tracker's gate and Stage 4's pre-session check call exactly the same code.
14. Synchronous-Telegram startup-latency hazard mitigated: the notify-callback dispatch on the corrupt and stale paths during construction is wrapped so that a callback exception does not break tracker construction, and the call is observably bounded (the design accepts the existing `_notify` sync-fallback timeout; this acceptance is documented in § 10 failure modes, not changed in code).
15. Drift check passes at full count.
16. All existing tracker tests still pass without modification (no behavior change for callers that do not provide a callback). Existing tests that suppress no warnings will need to be checked for whether they construct trackers from state files older than 24 hours; if any do, they receive an explicit `caplog` allowance.
17. Adversarial-audit pass returns PASS or all CONDITIONAL findings closed.

### Tests required (mutation-proof — strengthened per design audit)

| Name | What it pins |
|---|---|
| `test_load_stale_30_days_plus_1s_raises_with_stale_state_fail_token` | Exact boundary direction (30+ raises). Asserts the `STALE_STATE_FAIL` prefix token, the file path, and the age-in-days are all present in the raised message. Mutation: any change that switches the comparison from `>=` to `>` causes test failure |
| `test_load_stale_30_days_minus_1s_warns_does_not_raise` | Boundary direction confirmation in the other direction |
| `test_load_stale_29_days_logs_warning_continues` | Mid-band warn behavior |
| `test_load_warn_24h_plus_1s_emits_log_warning` | Suppression boundary direction (24h+ warns) |
| `test_load_warn_24h_minus_1s_silent` | Suppression boundary direction (under-24h silent) |
| `test_load_recent_silent` | 1-hour-old state file emits no warning (negative case) |
| `test_load_stale_message_does_not_claim_canonical_grounding` | Mutation guard: assert the raised message does NOT contain `@canonical-source` or `topstep_xfa_parameters.txt:349` substrings — the design v2 explicitly disclaims that grounding |
| `test_load_corrupt_invokes_notify_callback_when_provided` | Corrupt branch operator-visible |
| `test_load_corrupt_no_callback_preserves_existing_behavior` | Backwards compatibility — log.error and backup still happen, no exception propagates |
| `test_poll_recovery_from_one_dispatches_notify_with_prior_count` | Recovery from count=1 fires notify exactly once with `1` in the message |
| `test_poll_recovery_from_two_dispatches_notify_with_prior_count` | Mutation guard: pin that the count is the actual prior count, not a constant |
| `test_poll_recovery_from_zero_does_not_dispatch` | Mutation guard: steady state does not spam |
| `test_save_state_oserror_dispatches_notify_and_reraises_with_persist_fail_token` | Persistence failure is loud and re-raised. Asserts `STATE_PERSIST_FAIL` prefix token. Distinguishes from broker-poll failure path (no `update_equity(None)` side effect) |
| `test_save_state_oserror_does_not_increment_poll_failure_counter` | Mutation guard: persistence and broker-poll failure modes stay separate |
| `test_eod_trailing_class_docstring_has_canonical_source_annotation_block` | Class docstring of `AccountHWMTracker` (lines 70-78) must contain `@canonical-source`, `@verbatim`, and `@audit-finding` tokens — positional check via inspecting the class' `__doc__` attribute at runtime, asserting the three tokens appear in the docstring text (not file-wide grep). v3 correction of the location from update_equity:342 to class docstring 70-78 |
| `test_ungrounded_constants_have_explicit_label_within_5_lines_above` | UNGROUNDED token must appear within 5 lines above each named constant definition (positional, not just file-wide grep) |
| `test_state_file_age_days_pure_function_no_logging` | The shared age helper produces no log output and has no file-system side effects beyond reading. Pin so future modifications cannot quietly add side effects |

### Blast radius

Tracker file plus tracker tests. The constructor change is backward-compatible (new parameter defaults to None). The 30-day raise is a new fail-closed path that did not exist before — operators of any account whose state file is 30-plus days old must explicitly acknowledge or archive. Three current state files are within this window: 20859313 (one day old) and 21944866 (one day old) are safe; 20092334 (19 days old, the original audit case) is at 11 days from the boundary.

### Rollback

`git revert` of the single judgment commit. No persisted state file is modified by this stage; only loading semantics change.

---

## 6. Stage 3 — Orchestrator and pre-session integration

### Purpose

Consume the new notify-callback primitive from Stage 2. Resolve the EOD-close silent-skip and the split-authority between the tracker object and the pre-session JSON readers.

### Files (scope_lock)

- `trading_app/live/session_orchestrator.py` — HWM-construction region around line 698 and EOD-close region around line 3062
- `trading_app/pre_session_check.py` — the two HWM-reader functions and a new shared reader extracted from them
- `tests/test_trading_app/test_session_orchestrator.py`
- `tests/test_trading_app/test_pre_session_check.py`

### Behavior changes

| Item | Change | Grounding |
|---|---|---|
| Wire notify-callback at HWM tracker construction | Pass the orchestrator's existing notify helper as the new callback parameter introduced in Stage 2. The notify helper is already constructed before the tracker (verified at line 698 vs the architecture comment block at lines 658-664) | Consumes Stage 2 primitive |
| EOD HWM-skip loud warning, suppressed when kill-switch already fired | When end-of-session equity query returns None or raises, AND the kill switch has NOT already fired, in addition to the existing log.warning, dispatch via the orchestrator's notify helper. Suppress the dispatch when `_kill_switch_fired` is True so the operator does not receive a redundant EOD-equity-unavailable notify on top of the kill-switch notify | Resolves SILENT-4. Suppression resolves Gap 3 from design audit |
| Shared canonical state-reader | Extract a single read-state-file helper into the tracker module (`account_hwm_tracker.read_state_file(path) -> dict | None`). All three callers delegate to it: `pre_session_check.check_dd_circuit_breaker`, `pre_session_check.check_hwm_tracker`, and `weekly_review` (line 37/46 reader loop). Returns None on corrupt or unreadable; calling functions map None to their respective response. **Corrected baseline (v3):** v2 incorrectly claimed `check_hwm_tracker` returns `(True, "BLOCKED ...")` on corrupt state. Verified against `pre_session_check.py:234-238`: `any_halt = True` on exception → `not any_halt = False` → returns `(False, "DD TRACKER: ... BLOCKED ...")`. Both functions ALREADY return `False` on corrupt state. The actual divergence is message format and accumulation behavior: `check_dd_circuit_breaker` early-returns on first corrupt file with `(False, "BLOCKED: HWM file unreadable (...)")`; `check_hwm_tracker` accumulates results and returns `(False, "DD TRACKER: ... | BLOCKED filename: error")` at end. **Stage 3 actual change:** unify message format prefix to `BLOCKED` and unify accumulation pattern via the shared reader. Boolean behavior is unchanged for both functions on corrupt state. `weekly_review` currently has its own corrupt-handling that calls `print` warnings on `Exception`; converting it to the shared reader changes its corrupt response from a warning print to a None-return that the loop handles | `integrity-guardian.md` § 2 (Canonical Sources — Never Hardcode) — never re-encode. § 3 (Fail-closed) — corrupt is no-pass. Resolves SILENT-7 |
| Pre-session check JSON output schema preservation | The new shared reader returns the same dict shape that `check_hwm_tracker` and `check_dd_circuit_breaker` currently parse. Stage 3 must verify by greppable evidence that no other code path consumes the existing return tuples beyond the two pre-session functions and `run_checks` | Mitigates downstream Risk 4 from design audit (external JSON consumers like bot_dashboard) |
| Signal-only mode comment | Add a short code comment at the signal-only gate around line 665 explaining that pre-session check handles the HWM authority for signal-only sessions, and the tracker object is intentionally not constructed. No behavior change | Resolves SILENT-5 documentation gap |
| Import-direction safety | The new shared reader is in `account_hwm_tracker.py` (tracker module). `pre_session_check.py` will import from the tracker module. Verified safe: `account_hwm_tracker` imports only stdlib (`json`, `pathlib`, `datetime`, `shutil`, `dataclasses`, `zoneinfo`, `logging`) at module level. No circular import risk | `integrity-guardian.md` § 4 (Impact Awareness) — verify import-graph impact before adding |

### Acceptance criteria

1. Tracker constructed by orchestrator now receives the orchestrator's notify helper as callback (verified by direct attribute inspection on the constructed tracker).
2. EOD equity-unavailable path dispatches an operator notify in addition to log.warning, ONLY when kill switch has not already fired.
3. EOD equity-unavailable path with kill switch already fired emits log.warning but does NOT dispatch (suppression test).
4. The two pre-session HWM-check functions both call the new shared reader; their JSON parsing is no longer duplicated.
5. Both pre-session functions return identical-shape tuples on the same corrupt input: boolean `False` (already true today, pinned for regression) AND message format begins with the `BLOCKED` token AND contains the file name. The message-format unification is the actual user-visible change in this stage; the boolean is unchanged from current behavior. `weekly_review`'s corrupt-handling moves from print-warning-and-continue to shared-reader None-return-and-skip.
6. The signal-only gate carries an explanatory comment naming the pre-session authority path within 3 lines above the gate condition.
7. The new `read_state_file` helper is greppable as the only `json.loads` call against `account_hwm_*.json` paths in the codebase. Three known call sites are converted in this stage: `pre_session_check.py:133` (check_dd_circuit_breaker), `pre_session_check.py:213` (check_hwm_tracker), `weekly_review.py:46` (weekly review reader loop). Test asserts zero remaining `json.loads` calls against this path pattern outside the shared reader. (NOTE: `bot_dashboard.py:1457` references `data/account_hwm.json` — a different file path with no `account_id` suffix and no `state` subdirectory — and is NOT a tracker-state consumer; verified in v3 audit.)
8. All existing orchestrator tests still pass.
9. All existing pre-session tests still pass — but if any test's expected message string changes due to the unified format, that test is updated as part of Stage 3 (with the change documented in the commit body).
10. Drift check passes at full count.
11. Adversarial-audit pass returns PASS or all CONDITIONAL findings closed.

### Tests required (mutation-proof — strengthened)

| Name | What it pins |
|---|---|
| `test_orchestrator_constructs_tracker_with_notify_callback_wired` | Direct attribute check: `tracker._notify_callback is orchestrator._notify`. Mutation guard: passing None instead must fail the test |
| `test_eod_equity_unavailable_dispatches_notify_when_kill_switch_not_fired` | Resolves SILENT-4 — happy path |
| `test_eod_equity_unavailable_no_dispatch_when_kill_switch_already_fired` | Suppression — Gap 3 resolution |
| `test_pre_session_dd_circuit_breaker_calls_shared_reader` | Delegation pinned via call-tracker mock |
| `test_pre_session_hwm_tracker_calls_shared_reader` | Delegation pinned via call-tracker mock |
| `test_weekly_review_calls_shared_reader` | Delegation pinned for the third consumer added in v3 |
| `test_pre_session_corrupt_state_returns_identical_tuple_across_both_functions` | Both return `(False, str)` (matches existing behavior — regression guard), both messages start with `BLOCKED`, both contain the file name. Pins the message-format unification — the actual change in this stage |
| `test_pre_session_run_checks_unchanged_for_clean_state` | Backward compat: a clean state file still produces a passing pre-session check; the JSON output schema is unchanged |
| `test_weekly_review_corrupt_state_skips_file_via_shared_reader` | Pin that the previous print-warning behavior is replaced by None-return-skip without altering downstream review semantics |
| `test_signal_only_gate_carries_authority_comment_above` | Greppable: comment containing "pre-session" appears within 3 lines above the gate condition |
| `test_no_other_callers_of_account_hwm_json_paths` | Codebase search assertion: no `json.loads` against `account_hwm_*.json` outside the new shared reader. Excludes test fixtures and the unrelated `bot_dashboard.py` legacy `data/account_hwm.json` path |

### Blast radius

Orchestrator HWM construction site and EOD close site (both narrow, line-level). Pre-session check refactor (one new helper, two consumer functions updated). No other code path. The two-layer DD architecture is preserved; this stage only wires the new callback through and unifies the divergent JSON readers.

### Rollback

`git revert` of the single judgment commit. Pre-session functions revert to independent JSON parsing (degraded but functional).

---

## 7. Stage 4 — Inactivity monitor and escalation docs

### Purpose

Surface the TopStep 30-day inactivity boundary as a pre-session pre-flight check (per UNSUPPORTED-7). Document SILENT-6 (S2/S3/S4 references in the v5.2 plan that have no entry in the deferred-findings ledger) and UNSUPPORTED-5/6 (R3 numeric values asserted without measurement) as deferred-findings rows so they are tracked, not lost.

### Files (scope_lock)

- `trading_app/pre_session_check.py` — one new check function
- `tests/test_trading_app/test_pre_session_check.py`
- `docs/ralph-loop/deferred-findings.md` — three new rows under "Open Findings"
- `HANDOFF.md` — mark closed audit findings

### Behavior changes

| Item | Change | Grounding |
|---|---|---|
| New `check_topstep_inactivity_window` | For every non-corrupt account-state file, compute the days since last equity update by calling the shared `account_hwm_tracker.state_file_age_days()` helper introduced in Stage 2 (single source of truth for the age computation; eliminates the v1 two-enforcement-surface flaw). Warn at 25 days. Fail at 30 days. Annotation: `@canonical-source` for the 30-day TopStep inactivity rule (where the figure is borrowed from), `# UNGROUNDED` comment for the 25-day operator-action buffer, mirroring the design's overall honesty pattern | The 30-day figure is borrowed from `topstep_xfa_parameters.txt:349-351` by analogy. The 25-day buffer is UNGROUNDED operational. Documented in § 2 |
| Three deferred-findings rows | One for SILENT-6 (S2/S3/S4 undefined in repo, escalated to Ralph next-audit). Two for UNSUPPORTED-5/6 (R3 numeric values lack measurement, escalated to Ralph for log-based rationale) | Process compliance — `adversarial-audit-gate.md` artifact requirements |
| HANDOFF update | Mark the closed audit findings (CRITICAL-1, 2, 3 plus SILENTs 1, 2, 3, 4, 5, 7 plus UNSUPPORTEDs 1, 2, 3, 4, 7) as closed with their commit hashes. SILENT-6, UNSUPPORTED-5, UNSUPPORTED-6 listed as deferred-findings rows. Reference this design doc | Bookkeeping |

### Acceptance criteria

1. New check function returns OK for state files under 25 days old.
2. Returns a warning verdict (still OK to proceed) for files between 25 and 30 days old.
3. Returns a fail-closed verdict (blocks) for files 30 days or older.
4. Function carries the canonical-source annotation block.
5. Three new deferred-findings rows present with the correct ID format and severity tags.
6. HANDOFF is updated.
7. Drift check passes at full count.
8. Adversarial-audit pass returns PASS or all CONDITIONAL findings closed.

### Tests required (mutation-proof)

| Name | What it pins |
|---|---|
| `test_inactivity_check_under_25_days_returns_ok` | Healthy path |
| `test_inactivity_check_25_to_30_days_warns_continues` | Boundary minus one |
| `test_inactivity_check_30_days_or_older_blocks` | Fail-closed boundary |
| `test_inactivity_check_carries_canonical_source_annotation` | Grounding pinned (greppable) |

### Blast radius

One new pre-session check function. One docs file (deferred-findings) gains three rows. HANDOFF gets a routine update. No tracker or orchestrator changes in this stage. Behavior change for operators: any future session against an account whose state file is 30-plus days old will be blocked at pre-session check.

### Rollback

`git revert` restores prior pre-session check behavior. Deferred-findings rows can be removed independently if Ralph closes them in his own iteration.

---

## 8. Sequencing and audit-gate cadence

```
Stage 1 (judgment commit)
  -> evidence-auditor pass (independent context)
  -> verdict closes or findings logged
Stage 2 (judgment commit)
  -> evidence-auditor pass
  -> verdict closes or findings logged
Stage 3 (judgment commit)
  -> evidence-auditor pass
  -> verdict closes or findings logged
Stage 4 (judgment + ledger)
  -> evidence-auditor pass on the new check function only
  -> verdict closes
HANDOFF closure note
```

Per `.claude/rules/adversarial-audit-gate.md`, the next stage does not commit until the prior stage's audit returns PASS or all CONDITIONAL findings are closed or explicitly deferred. This mirrors Ralph's iteration cadence in `docs/plans/2026-04-25-ralph-crit-high-burndown-v5.md`.

---

## 9. Ralph-collision analysis

Ralph's v5.2 plan remaining iterations: 178 (audit), 179 (Pass-one shutdown helper plus drift check 114), 180 (already landed — Pass-two docs), 181 (R4 live-signals rotation), 182 (audit), 183 (R5 heartbeat re-notify), 184 (audit), 185 (F7 fill-poller timeout), 186 (audit), 187 (Pass-three magic-number drift check 115), 188 (silent-gap cleanup S2 plus ledger semantics).

| Ralph iter | Touches | Collision with this plan? |
|---|---|---|
| 179 | session_orchestrator.py shutdown block, pipeline/check_drift.py | No — different region |
| 181 | session_orchestrator.py live-signals jsonl region | No — different region |
| 183 | session_orchestrator.py heartbeat region | No — different region |
| 185 | session_orchestrator.py fill-poller region | No — different region |
| 187 | pipeline/check_drift.py and live-folder magic-number sweep | No — my plan does not touch check_drift.py |
| 188 | "Silent-gap cleanup (S2, ledger semantics)" | Possible overlap — but S2 is undefined in repo; will be clarified by Stage 4 deferred-findings entry. If Ralph 188 lands first, my Stage 3 may need to rebase the pre-session refactor |

The narrow line-region edits (Stage 1 line 1601, Stage 3 line 698 and line 3062) are well-separated from Ralph's planned regions. No expected merge conflict.

---

## 10. Validation — failure modes and test strategy

### Stage 1 failure modes

- Risk: a non-WARN reason string accidentally contains the substring WARN. Mitigation: substring check is the same as the existing code path; not introducing new matching logic. Test pins the OK case explicitly.
- Risk: notification spam if the warning fires on every poll while DD lingers in the warning band. Mitigation: not in this stage. Acknowledged as a follow-up if observed in operation. The current behavior of polling every 10 bars (~10 minutes) bounds spam to roughly 6 messages per hour in the warning state, which is operationally acceptable for a hard-failure safety tier.

### Stage 2 failure modes

- Risk: a 30-day fail-closed raise blocks legitimate resumption after a long break. Mitigation: the raise message names two explicit resolution paths (archive the file with a `.STALE_<YYYYMMDD>.json` suffix, or delete it — tracker re-inits cleanly from broker on next poll). No env-var bypass (rejected in v2 audit because it had no codebase precedent and would create an undocumented bypass on a safety gate). Operator action is visible in the file system, which is auditable.
- Risk: an operator who legitimately trades manually for 30+ days while the bot is idle (broker account active, no inactivity-closure risk on TopStep's side) hits the fail-closed raise as a false positive. Mitigation: the raise message explicitly states this is an operational staleness heuristic, not a TopStep rule, so the operator understands they are resolving the bot's state freshness, not the broker's account state. Acceptance criterion 2 pins that the message does NOT claim canonical grounding.
- Risk: notify-callback signature drift. Mitigation: typed as a string-consuming callable; orchestrator already exposes a string-consuming notify helper; no signature mismatch possible.
- Risk: corrupt-state notify fires during construction and references instance attributes that have not yet been assigned. Mitigation (v3 corrected): `_notify` is a class method (defined at session_orchestrator.py:1178), so the bound method is callable from the moment `__init__` enters. The actual risk is which INSTANCE ATTRIBUTES the `_notify` method body accesses — e.g. `self._stats`, `self._notifications_broken` — which may not yet be set when `_notify` fires from inside the tracker constructor at line 698. The `_notify` method already guards against this with `getattr(self, "_stats", None)` defensive patterns (per session_orchestrator HWM/B6 prior work). Stage 2 acceptance test asserts this defensive read by injecting a partial-init scenario: construct the orchestrator only up to the tracker line, fire `_notify` via the callback, assert no AttributeError. (v3 correction: v2 incorrectly cited `lines 658-664` as evidence of construction order; those lines are the architecture comment, not an init step.)
- Risk: synchronous Telegram timeout latency during construction. The `_notify` helper falls back to a synchronous send when no event loop is running (orchestrator code, sync fallback path). If the corrupt or stale path fires inside `_load_state()` during synchronous construction, the notify call may block for up to the Telegram client timeout (currently observed at ~10 seconds). This is not a deadlock but it does extend startup latency. Mitigation: this design accepts the existing sync-fallback latency as the cost of operator visibility on a corrupt-state event. Future work can move construction-time notifies to a deferred dispatch queue if observed as a problem; not in scope for this design.
- Risk: the notify-callback dispatch itself raises (e.g. Telegram client failure). Mitigation: wrap the dispatch site in a narrow try/except that logs the dispatch failure at log.error level and continues. The tracker's load semantics must not be broken by a notify failure. Acceptance criterion 14 pins this.

### Stage 3 failure modes

- Risk: shared reader changes the failure semantics of one of the two pre-session functions in a way that breaks an existing test. Mitigation: Stage 3 acceptance criteria require all existing pre-session tests still pass; if any fails, the shared reader is wrong and must be revised before commit.
- Risk: EOD notify spam if EOD equity is unavailable on every session close (broker partial outage at 09:00 Brisbane). Mitigation: notify fires once per close call, not per retry; rate-bounded by the close cadence (one per session). Acknowledged as operationally acceptable.

### Stage 4 failure modes

- Risk: the 25-day soft warning fires too often and operators ignore it. Mitigation: warning-tier-only; does not block; operator can dismiss until 30 days. Single-cite grounding documented honestly.

### Cross-stage validation (AUTOMATED pytest integration tests, split per stage)

Per design audit Flaw 4, the cross-stage scenarios are implemented as automated pytest integration tests (NOT a manual run-script), so they run in CI and protect against regression. **v3 improvement:** the integration test is SPLIT across stages so each stage's CI gets coverage as it lands, rather than deferring all integration coverage to Stage 4.

#### Stage 2 integration tests

File: `tests/test_trading_app/test_account_hwm_tracker_integration.py` (NEW, lands in Stage 2 commit)

Scenarios:
1. Construct an `AccountHWMTracker` with a synthetic state directory containing a 19-day-old state file (mirrors the audit case). Verify exactly one `log.warning` is emitted naming the age, no exception, no notify dispatch (callback unset case) and one notify dispatch (callback set case).
2. Construct with a 30-day-plus-1-second state file. Verify `RuntimeError` raised with `STALE_STATE_FAIL` token, file path, and age in days.
3. Construct with a 30-day-minus-1-second state file. Verify warning, no exception.
4. Construct with a corrupt JSON file and a notify-callback. Verify exactly one notify dispatch in addition to `log.error` and backup file creation.
5. Inject a synthetic `OSError` into the save-state path via monkeypatch. Verify `STATE_PERSIST_FAIL` notify dispatch and re-raise.

#### Stage 3 integration test

Added to the same file in the Stage 3 commit (or a Stage 3-specific file if scope demands):

6. Construct a real `SessionOrchestrator` in test mode, verify the constructed tracker's `_notify_callback` attribute is the orchestrator's `_notify` bound method.

#### Stage 4 integration test

Added to the same file in the Stage 4 commit:

7. Call the Stage 4 `check_topstep_inactivity_window` against the same synthetic state directory at 24-day, 25-day, 29-day, and 30-day file ages. Verify OK, WARN, WARN, BLOCKED respectively. Confirm via call-tracker that the function calls the shared `state_file_age_days` helper from the tracker module (single-source-of-truth verification).

This split ensures: (a) Stage 2 ships with five integration scenarios in CI from day one; (b) Stage 3 adds the wire-up scenario as it lands; (c) Stage 4 adds the cross-stage coordination scenario as the final piece. A Stage 2 regression introduced by Stage 3 will be caught by CI on the Stage 3 commit, not deferred to Stage 4.

---

## 11. Rollback plan and downstream-behavior-change risk register

### Rollback dependencies

- Stage 1: independent. Can be reverted alone with no impact on other stages.
- Stage 2: introduces the optional notify-callback constructor parameter and the `state_file_age_days` shared helper. Stage 3 and Stage 4 both consume Stage 2 outputs. Reverting Stage 2 requires reverting Stage 3 and Stage 4 first.
- Stage 3: consumes Stage 2's notify-callback. Reverting Stage 3 alone is safe (Stage 2's parameter remains in the constructor signature with `default=None`).
- Stage 4: consumes Stage 2's `state_file_age_days` shared helper. Reverting Stage 4 alone is safe (the helper remains in the tracker module, used only by Stage 2's load gate). However Stage 4 is also LOGICALLY coupled to Stage 2 — reverting Stage 2 alone would leave Stage 4 testing for a 30-day boundary that no longer fires at the constructor level. So if Stage 2 is reverted, Stage 4's value is reduced to a pre-session-only enforcement, which is still useful but no longer paired.

### Downstream behavior-change risk register (per design audit)

| Risk | Stage | Consumer | Mitigation |
|---|---|---|---|
| `check_hwm_tracker` and `check_dd_circuit_breaker` message-format unification on corrupt state. Both already return `False` (not the v2-claimed True/False divergence); v3 audit corrected the baseline. Stage 3's actual change is unifying the message prefix to `BLOCKED` and the accumulation pattern via the shared reader. Boolean is unchanged — no gate-behavior change | 3 | `run_checks` in `pre_session_check.py:562-566`; consumers of the printed `BLOCKED ...` strings | Documented in Stage 3 behavior table. Operator-visible difference is message text only |
| `weekly_review.py` corrupt-handling moves from print-warning-and-continue to shared-reader None-return-and-skip | 3 | `weekly_review.main()` text output to operator | Documented in Stage 3 behavior table. Output text changes; downstream review semantics unchanged (corrupt files were already skipped, just with a different log message) |
| Soft warning at every load with age 24h or older creates log noise during testing | 2 | Test suite that constructs trackers from old state files | Acceptance criterion 16 requires checking existing tests; if any construct from old state files, they receive an explicit `caplog` allowance |
| Synchronous Telegram timeout latency (~10 s) on tracker construction if corrupt or stale path fires before event loop starts | 2 | Orchestrator startup path | Documented as accepted cost in Stage 2 failure modes; future deferred-dispatch queue is out of scope |
| Pre-session checklist JSON consumed by external monitoring tools may break if message format changes | 3 | Any monitoring tool that parses `STATE_DIR/session_checklist_*.json` for specific message substrings | v3 correction: `bot_dashboard.py:1457` references `data/account_hwm.json` (no account-id suffix, no `state` subdirectory) — a different file from the tracker state files; it is NOT a consumer of the changed paths. Verified during v3 audit. Remaining mitigation: Stage 3 grep at AC-7 confirms only the three named consumers (`pre_session_check` x2, `weekly_review`) call the relevant paths. If any future external monitor parses the checklist JSON for specific substrings, the message-format change in Stage 3 may break it; not flagged today as a known concrete consumer |

### Persisted state files

No persisted state file is mutated by any stage in this plan. State-file ages cited below are AS-OF DESIGN DATE 2026-04-25 (volatile data — re-query before action per `docs/STRATEGY_BLUEPRINT.md` volatile-data rule):

- `account_hwm_20859313.json` — 1 day old at design date — safe under both warn and fail thresholds.
- `account_hwm_21944866.json` — 1 day old at design date — safe.
- `account_hwm_20092334.json` — 19 days old at design date — will trigger the soft warning on every load until the operator either trades on the account (which updates the timestamp) or archives the file. Approaching the 30-day fail-closed boundary on or around 2026-05-06.

Re-verify these ages with `ls -la data/state/account_hwm_*.json` before Stage 2 commits. If any file has aged past 30 days by the time Stage 2 lands, the operator must archive or delete that file BEFORE the new tracker code is run, or the tracker will fail-closed on first construction against that account.

On the first run after Stage 2 lands, the operator will see the soft warning for 20092334 in any session log that constructs a tracker for that account. Stage 4's pre-session inactivity check covers the case where the operator runs `pre_session_check --session` without constructing a tracker.

---

## 12. Guardian prompts

Not applicable. This design touches only safety/persistence infrastructure, not entry models, filter logic, ML, or pipeline data flow. ENTRY_MODEL_GUARDIAN and PIPELINE_DATA_GUARDIAN do not apply.

---

## 13. Approval

Awaiting user approval. On approval, Stage 1 stage-doc will be written to `docs/runtime/stages/hwm-warning-tier-notify-dispatch.md` with scope_lock and acceptance criteria copied from § 4 above. Implementation begins under `quant-tdd` discipline (test-first per stage). No code is written before approval.
