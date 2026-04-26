---
slug: hwm-stage3-orchestrator-pre-session
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 3
of: 4
created: 2026-04-26
updated: 2026-04-26
parent_design: docs/plans/2026-04-25-hwm-persistence-integrity-hardening-design.md (v3)
prior_stage_audit: commits 1f29009b + e67f46f6 + c5be3453 — Stage 2 audit-gate PASS (2026-04-26)
audit_gate: `.claude/rules/adversarial-audit-gate.md` — fires after this judgment commit before Stage 4 dispatches
task: HWM Stage 3 — wire Stage 2's `notify_callback` into the orchestrator at HWM construction; resolve EOD silent-skip on equity-unavailable AND on exception, both suppressed when kill switch already fired; extract `read_state_file(path) -> dict | None` shared reader into `account_hwm_tracker.py`; convert the three external consumers (pre_session × 2, weekly_review × 1) to delegate; unify pre-session corrupt-state message format to `BLOCKED <filename>: <reason>`. 12 mutation-proof unit tests + 1 integration scenario.
---

# Stage: hwm-stage3-orchestrator-pre-session

mode: IMPLEMENTATION
date: 2026-04-26

scope_lock:
  - trading_app/account_hwm_tracker.py
  - trading_app/live/session_orchestrator.py
  - trading_app/pre_session_check.py
  - trading_app/weekly_review.py
  - tests/test_trading_app/test_account_hwm_tracker.py
  - tests/test_trading_app/test_account_hwm_tracker_integration.py
  - tests/test_trading_app/test_session_orchestrator.py
  - tests/test_trading_app/test_pre_session_check.py
  - tests/test_trading_app/test_weekly_review.py
  - docs/runtime/stages/hwm-stage3-orchestrator-pre-session.md

## Why

Stage 2 shipped the architectural primitives (`notify_callback`, `state_file_age_days`, `_safe_notify` bounded-dispatch wrapper, six UNGROUNDED+Rationale annotations, class docstring canonical-source block). The orchestrator does not yet wire the callback in. The EOD HWM session-end recording at `session_orchestrator.py:3086-3095` has TWO silent paths (equity-unavailable and exception) that are operator-invisible. Three external consumers (`pre_session_check.check_dd_circuit_breaker`, `pre_session_check.check_hwm_tracker`, `weekly_review.section_0_account_health`) each parse `account_hwm_*.json` independently — three forks of the same JSON read logic, two with different message-format conventions on corrupt input.

Stage 3 wires the callback, closes the silent paths, and consolidates the three readers behind a single canonical helper.

Grounding (operational hardening — no new statistical/academic claim):

- `institutional-rigor.md` § 4 — canonical sources, never re-encode (shared reader)
- `institutional-rigor.md` § 6 — no silent failures (EOD dispatch + read_state_file granular logging)
- `integrity-guardian.md` § 2 — canonical sources never hardcode (single reader path)
- `integrity-guardian.md` § 3 — fail-closed (corrupt → BLOCKED uniformly)
- TopStep XFA parameter file — already cited via Stage 2 class docstring; Stage 3 adds no new TopStep claim
- `docs/institutional/literature/` — NOT applicable. Stage 3 is operational hardening, no academic citation needed. Confirmed during 2026-04-26 grounding pass.

## Pre-execution audit findings — three improvements over design v3 § 6

Caught during 2026-04-26 self-audit, baked into this stage doc:

1. **Bound-method identity bug in design v3 AC test (CORRECTNESS FIX):** design v3 § 6 Tests-Required line specified `tracker._notify_callback is orchestrator._notify`. Python bound-method semantics: `obj.method is obj.method` always returns False (each attribute access constructs a fresh bound-method wrapper). The test would fail every run. Fix: use `==` for the check (bound methods compare equal when `__self__` and `__func__` match), OR check the underlying `__func__ is type(orchestrator)._notify`. This stage uses `==` per Python convention.

2. **`read_state_file` granular logging on None-return (IR §6 COMPLIANCE):** design v3 specifies `read_state_file(path) -> dict | None`. A strict pure-silent reader violates `institutional-rigor.md` § 6 ("every except must record the exception"). Resolution: `read_state_file` returns `dict | None` to callers (preserves the design's type signature), but emits `log.warning` with the file path AND the granular reason (missing / empty / json-error / value-error / oserror) on each None-return path. Callers produce their own user-facing BLOCKED message; granular reason captured in operator log file. This is documented as Stage 3 sanity-check improvement, not a design-v3 deviation that requires re-audit — the type signature is unchanged.

3. **EOD has TWO silent branches, not one (DESIGN-V3 GAP):** design v3 § 6 said "EOD HWM-skip loud warning, suppressed when kill-switch already fired". The implementation site at `session_orchestrator.py:3086-3095` has TWO silent paths:
   - `end_equity is None` → falls through with no log/notify (silent)
   - `except Exception` arm → log.warning only (operator-invisible)

   Both must dispatch via `_notify`, both suppressed when `_kill_switch_fired` is True. Tests pin both paths plus both suppression cases (4 combinations).

## Behavior changes

### B1. Wire `notify_callback` at HWM construction

`session_orchestrator.py:698-704`:

```python
self._hwm_tracker = AccountHWMTracker(
    account_id=acct_id,
    firm=prof.firm,
    dd_limit_dollars=float(tier.max_dd),
    dd_type=firm_spec.dd_type,
    freeze_at_balance=freeze,
    notify_callback=self._notify,  # NEW — Stage 3 wire-up
)
```

`self._notify` is a bound method (defined `session_orchestrator.py:1178`) with internal 10s timeout via urllib + asyncio.to_thread escape; never raises (per docstring lines 1184-1190). Tracker's `_safe_notify` wrapper at `account_hwm_tracker.py:246-259` adds an additional try/except so a callback failure cannot break tracker construction or update_equity flow (Stage 2 AC 14, already pinned).

### B2. Signal-only authority comment

Insert a 1-line comment within 3 lines above the `if not signal_only` gate at `session_orchestrator.py:665`:

```python
# Signal-only sessions: HWM tracker intentionally NOT constructed.
# Pre-session check (pre_session_check.check_hwm_tracker) is the
# DD authority for signal-only — operates on persisted state files.
self._hwm_tracker = None
if not signal_only and portfolio is not None and portfolio.strategies:
```

Resolves SILENT-5 documentation gap. No behavior change.

### B3. EOD silent-skip resolution — both branches, with kill-switch suppression

`session_orchestrator.py:3086-3095` becomes:

```python
if self._hwm_tracker is not None and self.positions is not None and self.order_router is not None:
    try:
        end_equity = self.positions.query_equity(self.order_router.account_id)
        if end_equity is not None:
            self._hwm_tracker.update_equity(end_equity)
            self._hwm_tracker.record_session_end(end_equity)
            _, reason = self._hwm_tracker.check_halt()
            log.info("HWM session close: %s", reason)
        else:
            # SILENT-4 (Stage 3): EOD equity unavailable — operator-visible
            # unless kill switch already fired (operator already notified by
            # the kill-switch path; suppression prevents duplicate alerts).
            msg = "HWM EOD: broker equity unavailable — session-end DD not recorded"
            log.warning(msg)
            if not self._kill_switch_fired:
                self._notify(msg)
    except Exception as e:
        # SILENT-4 (Stage 3): EOD recording exception — same suppression rule.
        msg = f"HWM EOD: session-end recording failed: {e}"
        log.warning(msg)
        if not self._kill_switch_fired:
            self._notify(msg)
```

Resolves SILENT-4 (both branches). Suppression resolves Gap 3 from design audit.

### B4. Shared `read_state_file` helper in tracker module

New module-level function in `trading_app/account_hwm_tracker.py`:

```python
def read_state_file(path: Path) -> dict | None:
    """Read and parse a tracker state file. Single source of truth for
    external consumers (pre_session_check, weekly_review, scripts).

    Returns:
        Parsed dict on success.
        None on missing / empty / JSON-parse-error / value-error / OSError.
        Granular reason logged via log.warning on each None-return path
        (per institutional-rigor.md § 6 — never silently swallow exceptions).

    Used by:
      - trading_app.pre_session_check.check_dd_circuit_breaker
      - trading_app.pre_session_check.check_hwm_tracker
      - trading_app.weekly_review.section_0_account_health

    NOT used by tracker-internal _load_state (which performs corrupt-rename
    + reinit beyond this helper's scope) or state_file_age_days (which has
    its own mtime-fallback semantics for the SG1 audit-gate fix).
    """
    if not path.exists():
        log.warning("read_state_file: %s does not exist", path)
        return None
    try:
        text = path.read_text()
    except OSError as exc:
        log.warning("read_state_file: %s OSError on read: %s", path, exc)
        return None
    if not text.strip():
        log.warning("read_state_file: %s is empty", path)
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("read_state_file: %s JSON parse failed: %s", path, exc)
        return None
    if not isinstance(data, dict):
        log.warning("read_state_file: %s top-level is not a dict (got %s)", path, type(data).__name__)
        return None
    return data
```

`state_file_age_days` (Stage 2) NOT touched — its mtime-fallback path is purpose-specific to the audit-gate SG1 fix. The two helpers serve different purposes and are intentionally distinct. Tracker's internal `_load_state` (Stage 2 audit-fixed) NOT touched — its corrupt-rename + reinit logic is beyond `read_state_file`'s scope.

### B5. Convert `check_dd_circuit_breaker` to delegate

`trading_app/pre_session_check.py:125-152`:

```python
def check_dd_circuit_breaker() -> tuple[bool, str]:
    """Check DD status from AccountHWMTracker state files.

    Reads the authoritative HWM tracker state via the shared
    account_hwm_tracker.read_state_file helper. Corrupt or unreadable
    state files block the check (fail-closed per integrity-guardian.md § 3).
    """
    from trading_app.account_hwm_tracker import read_state_file

    hwm_files = list(STATE_DIR.glob("account_hwm_*.json"))
    hwm_files = [f for f in hwm_files if "CORRUPT" not in f.name]
    if not hwm_files:
        return True, "No DD tracker state (first session — will init from broker)"

    for f in hwm_files:
        data = read_state_file(f)
        if data is None:
            return False, f"BLOCKED {f.name}: state file unreadable (see logs)"
        if data.get("halt_triggered"):
            return (
                False,
                f"DD HALT ACTIVE: account {data.get('account_id', '?')} — "
                f"DD ${data.get('dd_used_dollars', 0):.0f} >= "
                f"limit ${data.get('dd_limit_dollars', 0):.0f}",
            )
    return True, "DD circuit breaker: clear (HWM tracker)"
```

Boolean behavior on corrupt is unchanged (was False, still False). Message format moves to unified `BLOCKED <filename>:` prefix.

### B6. Convert `check_hwm_tracker` to delegate

`trading_app/pre_session_check.py:201-238`:

```python
def check_hwm_tracker() -> tuple[bool, str]:
    """Check account HWM DD tracker status."""
    from trading_app.account_hwm_tracker import read_state_file

    hwm_files = list(STATE_DIR.glob("account_hwm_*.json"))
    if not hwm_files:
        return True, "No HWM tracker active (first session — will init from broker)"

    results = []
    any_halt = False
    for f in hwm_files:
        if "CORRUPT" in f.name:
            continue
        data = read_state_file(f)
        if data is None:
            any_halt = True
            results.append(f"BLOCKED {f.name}: state file unreadable")
            continue
        acct = data.get("account_id", "?")
        hwm = data.get("hwm_dollars", 0)
        used = data.get("dd_used_dollars", 0)
        limit = data.get("dd_limit_dollars", 0)
        pct = data.get("dd_pct_used", 0)
        halted = data.get("halt_triggered", False)
        hwm_date = (data.get("hwm_timestamp") or "")[:10]
        remaining = limit - used

        if halted:
            any_halt = True
            results.append(f"HALT {acct}: DD ${used:.0f} >= limit ${limit:.0f}")
        elif pct >= 0.75:
            results.append(
                f"WARN {acct}: DD ${used:.0f}/{limit:.0f} ({pct:.0%}) — "
                f"${remaining:.0f} remaining. HWM ${hwm:.0f} on {hwm_date}"
            )
        else:
            results.append(
                f"OK {acct}: DD ${used:.0f}/{limit:.0f} ({pct:.0%}) — "
                f"${remaining:.0f} remaining"
            )

    msg = " | ".join(results)
    return (not any_halt), f"DD TRACKER: {msg}"
```

Boolean behavior unchanged. Message format on corrupt: `BLOCKED <filename>: state file unreadable` (was `BLOCKED <filename>: <exception-repr>`). Existing tests for corrupt-state messages updated as part of Stage 3 (commit body documents the change).

### B7. Convert `weekly_review.section_0_account_health` to delegate

`trading_app/weekly_review.py:37-77`:

```python
hwm_files = list(STATE_DIR.glob("account_hwm_*.json"))
if not hwm_files:
    print("  No HWM tracker files found. Will init on first live session.")
    return

from trading_app.account_hwm_tracker import read_state_file

for f in hwm_files:
    if "CORRUPT" in f.name:
        continue
    data = read_state_file(f)
    if data is None:
        # Per Stage 3 design v3 § 6: corrupt-handling moves from
        # print-and-continue to silent skip (granular reason in logs
        # via read_state_file's log.warning).
        continue
    acct = data.get("account_id", "?")
    firm = data.get("firm", "?")
    # ... rest of the per-file print loop unchanged
```

Behavior change documented in commit body: corrupt files no longer print "ERROR: <name>: <exc>" to stdout; they silently skip with the granular reason in the operator log file via `read_state_file`'s `log.warning`. Operationally equivalent (corrupt files were skipped either way).

## Acceptance criteria

1. Tracker constructed by orchestrator receives the orchestrator's `_notify` bound method as `notify_callback`. Verified via `tracker._notify_callback == orchestrator._notify` (NOT `is` — bound-method identity correction per pre-execution audit).
2. EOD `end_equity is None` path with kill switch NOT fired: dispatches one `_notify` AND emits one `log.warning`.
3. EOD `end_equity is None` path with kill switch ALREADY fired: emits `log.warning` but does NOT dispatch.
4. EOD exception path with kill switch NOT fired: dispatches one `_notify` containing exception text AND emits one `log.warning`.
5. EOD exception path with kill switch ALREADY fired: emits `log.warning` but does NOT dispatch.
6. The signal-only gate at `session_orchestrator.py:665` carries the authority comment within 3 lines above (greppable).
7. New `read_state_file` helper exists at module scope in `trading_app/account_hwm_tracker.py`.
8. `read_state_file` returns `dict` on a valid state file; returns `None` on missing / empty / JSON-error / value-error / OSError; emits `log.warning` with file path and granular reason on each None-return path.
9. The three external consumers — `check_dd_circuit_breaker`, `check_hwm_tracker`, `weekly_review.section_0_account_health` — call `read_state_file` and have no `json.loads` calls against `account_hwm_*.json` paths anywhere in their bodies.
10. Greppable assertion: `json.loads` paired with `account_hwm` token is zero in `trading_app/pre_session_check.py` AND in `trading_app/weekly_review.py` (test scopes to those two files; `account_hwm_tracker.py` is the canonical owner and is exempt; `bot_dashboard.py:1466` references the disjoint `data/account_hwm.json` path and is verified non-matching).
11. Both pre-session HWM-check functions return `(False, str)` on corrupt input (regression guard — was already False before Stage 3, pinned now). Both messages start with `BLOCKED` and contain the file name.
12. Existing pre-session tests pass (any test with hardcoded message-format expectations on the corrupt branch is updated as part of Stage 3 with rationale in commit body).
13. Existing orchestrator tests pass (callback wiring is additive — default `None` path preserved).
14. Existing weekly_review tests pass (corrupt-handling change is operator-visible text only — no test depends on the specific `ERROR:` format).
15. Drift check passes at full count (`pipeline/check_drift.py`).
16. Adversarial-audit pass returns PASS or all CONDITIONAL findings closed.

## Tests required (mutation-proof)

### Unit tests — tracker (`tests/test_trading_app/test_account_hwm_tracker.py`)

| Name | What it pins |
|---|---|
| `test_read_state_file_returns_dict_on_valid_state` | Happy path. |
| `test_read_state_file_returns_none_on_missing_file` | Missing → None + log.warning containing file path. |
| `test_read_state_file_returns_none_on_empty_file` | Empty → None + log.warning containing "empty". |
| `test_read_state_file_returns_none_on_corrupt_json` | JSONDecodeError → None + log.warning containing "JSON parse failed". |
| `test_read_state_file_returns_none_on_non_dict_top_level` | `[1,2,3]` → None + log.warning containing "not a dict". |
| `test_read_state_file_returns_none_on_oserror` | Permission denied / unreadable → None + log.warning containing "OSError". Mutation: silently swallowing the exception (no warning) flips this test. |

### Unit tests — pre_session_check (`tests/test_trading_app/test_pre_session_check.py`)

| Name | What it pins |
|---|---|
| `test_check_dd_circuit_breaker_calls_shared_reader` | Delegation pinned via `unittest.mock.patch` of `read_state_file`; verify call count matches file count. Mutation: re-introducing inline `json.loads` flips this test. |
| `test_check_hwm_tracker_calls_shared_reader` | Same pattern, second function. |
| `test_pre_session_corrupt_state_returns_blocked_filename_format` | Both functions on corrupt input return `(False, msg)` where `msg` starts with `BLOCKED` and contains the file name. Mutation: removing the BLOCKED prefix flips this test. |
| `test_pre_session_clean_state_unchanged_behavior` | Backward compat regression guard. |
| `test_no_json_loads_against_account_hwm_in_pre_session_or_weekly_review` | Greppable static-source assertion: read both files, count `json.loads` co-occurring with `account_hwm` token, assert zero. Mutation: any future re-introduction of inline parsing flips this test. |

### Unit tests — weekly_review (`tests/test_trading_app/test_weekly_review.py`)

| Name | What it pins |
|---|---|
| `test_weekly_review_corrupt_state_silently_skips` | Corrupt file no longer prints `ERROR:`; loop continues; granular reason captured by `read_state_file`'s log.warning. |
| `test_weekly_review_calls_shared_reader` | Delegation pinned. |

### Unit tests — orchestrator (`tests/test_trading_app/test_session_orchestrator.py`)

| Name | What it pins |
|---|---|
| `test_orchestrator_constructs_tracker_with_notify_callback_wired` | `tracker._notify_callback == orchestrator._notify` (bound-method equality per audit improvement #1). Mutation: passing `None` flips. |
| `test_eod_equity_unavailable_dispatches_when_kill_switch_not_fired` | `end_equity is None` + kill switch False → 1 `_notify` call + 1 `log.warning`. |
| `test_eod_equity_unavailable_no_dispatch_when_kill_switch_fired` | `end_equity is None` + kill switch True → 0 `_notify` calls + 1 `log.warning`. |
| `test_eod_exception_dispatches_when_kill_switch_not_fired` | `query_equity` raises + kill switch False → 1 `_notify` containing exception text + 1 `log.warning`. |
| `test_eod_exception_no_dispatch_when_kill_switch_fired` | `query_equity` raises + kill switch True → 0 `_notify` + 1 `log.warning`. |
| `test_signal_only_gate_carries_authority_comment_above` | Source-file scan: comment containing `pre_session_check` and `signal-only` within 3 lines above the `if not signal_only` gate. |

### Integration test — append to existing `tests/test_trading_app/test_account_hwm_tracker_integration.py`

| Scenario 6 (per design v3 § 12.2) | What it pins |
|---|---|
| Construct a real `SessionOrchestrator` in test mode; verify the constructed tracker's `_notify_callback` equality with the orchestrator's `_notify` bound method | End-to-end wire-up integration; mutation guard against `notify_callback=None` regression at the construction site. |

## Blast Radius

- `trading_app/account_hwm_tracker.py` — adds 1 module-level function (~25 lines). No edits to existing functions, no edits to the class. Stage 2 contracts preserved.
- `trading_app/live/session_orchestrator.py` — 3 narrow edits: line 665 comment, line 698-704 add 1 kwarg, lines 3086-3095 expand to 16 lines. ~15 net added lines. No structural change.
- `trading_app/pre_session_check.py` — 2 functions converted; ~30 lines net (mostly replacing inline parse with `read_state_file` delegation).
- `trading_app/weekly_review.py` — 1 function loop converted; ~5 lines net.
- 5 test files: 12 new unit tests + 1 integration scenario + minor edit to any pre-existing test asserting old corrupt-state message format (target: 0 such edits if grep finds none; documented if any).

No external API change. No public function signature change. No DB schema touch. The two-layer DD architecture is preserved; this stage only wires the new callback through and unifies the divergent JSON readers.

State files on disk (`data/state/account_hwm_*.json`): NOT modified by this stage. Read-only consumers shifted to a single helper. Existing files continue to load correctly.

## Risk flags

- **Risk 1 (LOW):** existing pre-session test asserts old `BLOCKED: HWM file unreadable (...)` or `BLOCKED <name>: <exception-repr>` format. Mitigation: AC 12 explicitly permits these tests to be updated; commit body documents each update with line citation.
- **Risk 2 (LOW):** orchestrator test for HWM construction asserts `notify_callback` is absent (would have been True before Stage 2). Mitigation: Stage 2 already added the parameter with `default=None`; if any test asserted absence, Stage 2's audit-gate would have caught it. Verify zero such tests during execution.
- **Risk 3 (LOW):** `bot_dashboard.py:1466` uses an unrelated path. Mitigation: AC 10 grep test scoped to `pre_session_check.py` + `weekly_review.py` only — does not touch `bot_dashboard.py` and cannot regress against it.

## Rollback

`git revert` of the single judgment commit. Pre-session and weekly_review revert to independent JSON parsing. Orchestrator reverts to no `notify_callback` (Stage 2 default `None` path). EOD silent-skip returns. Stage 4 depends on the `read_state_file` helper not existing in revert-target state — Stage 4 must revert first if Stage 3 is reverted later.

## Audit-gate hand-off

After this commit lands, dispatch `evidence-auditor` per `.claude/rules/adversarial-audit-gate.md`. Auditor scope:

- This commit only.
- Verify all 16 acceptance criteria with execution evidence.
- Verify no regression in Stage 2's 49 tracker tests (callback default-None path) and 5 integration scenarios.
- Independent trace: does the EOD dispatch interact correctly with the kill-switch flow? Confirm operator does not receive duplicate notifies on a kill-switch-then-EOD-equity-unavailable sequence.
- Verify Pass Three forward-compat: do new constants (none added in Stage 3) and new lines survive `pipeline/check_drift.py` rationale-discipline scan? (Stage 3 adds no new UPPER_SNAKE_CASE numeric literals — verify.)
- Verify the AC 10 greppable assertion is correct as written and resilient against false-positives (test fixtures, comment-only mentions of `account_hwm` in docstrings).
- Verify bound-method equality (`==` not `is`) is used in AC 1 test per pre-execution audit improvement #1.

Stage 4 dispatches only after this audit returns PASS.
