# Adversarial-Audit Gate Artifact — Gap #2 kill-switch / post_session broker-truth fix

**Date:** 2026-05-27
**Gate:** `.claude/rules/adversarial-audit-gate.md` (CRITICAL/HIGH `[judgment]` fix in `trading_app/live/`)
**Actor:** independent-context `evidence-auditor` pass (separate conversation; commit-message claims treated as claims requiring proof)
**Commits under review:**

- `b5bb4a02` — `[judgment] fix(live): verify broker-flat before skipping EOD close on kill switch` (production change to `trading_app/live/session_orchestrator.py`)
- `dc2500a8` — `test(live): pin kill-switch degraded-flat skip path (audit finding)` (test-only follow-up; closes the LOW finding below)

**Why this artifact exists:** the `dc2500a8` commit message asserted "Evidence-auditor verdict on b5bb4a02 was PASS" but no structured artifact file was written. The gate rule's "Artifact required" section mandates a searchable record with all fields. A re-run independent audit on 2026-05-27 confirmed the substance AND flagged the missing artifact as the sole CONDITIONAL condition. This file satisfies that condition.

---

## Verdict: CONDITIONAL → satisfied by this file

Production logic and tests are SOUND for the `--live` flip. The only condition was the absence of this artifact. No capital-impact defect found across five adversarial probes.

## Per-probe findings (PREMISE → TRACE → EVIDENCE → CONCLUSION)

1. **Attribute aliasing `self.positions` vs `self._positions` — CONFIRMED not-a-bug.**
   `self.positions` (broker adapter with `query_open()`) is assigned at `session_orchestrator.py:665` (or None at :634 signal-only). `self._positions` (in-memory `PositionTracker`) is constructed unconditionally at `:737`. Different objects by design. Guard at `:3042` checks `self.positions is None` before `:3045` `self.positions.query_open()` — safe. `self._positions` is never None, so the `:3047`/`:3073` `.active_positions()` fallback has no NPE path.

2. **Canonical-reuse / startup-orphan parity (`:670`) — CONFIRMED with caveat.**
   Startup orphan check (`:670`) and new helper (`:3045`) both call `self.positions.query_open(account_id)`. Exception discipline differs intentionally: startup is fail-open on generic `Exception` (`:688` log.error, non-blocking); the new helper is fail-CLOSED (`:3061` returns False + alert). The helper is STRICTER — correct for the EOD path. Docstring "same path/exception discipline" is slightly loose but the deviation is in the safer direction.

3. **No new exposure path created (C1 proof-case class) — CONFIRMED.**
   `execution_engine.on_trading_day_end()` (`:583-652`) emits only `SCRATCH` (exit) events for ENTERED trades; ARMED/CONFIRMING trades are discarded with no event. No entry events. `on_bar()` returns early once `_kill_switch_fired` is set (`:1042`), so no new ARMED trades exist post-kill-switch. Running the EOD close loop after a kill switch cannot double-submit or open a new position.

4. **CopyOrderRouter.account_id — CONFIRMED valid.**
   `copy_order_router.py:50` → `super().__init__(account_id=primary.account_id, ...)`. `self.order_router.account_id` resolves to `primary.account_id`, matching the startup orphan check usage. The `:3045` query uses a valid primary account ID under a copy router.

5. **Test reality — CONFIRMED.**
   39 targeted tests pass. `test_post_session_attempts_close_when_broker_shows_open` exercises the `:3839` elif (broker-open → close). `test_post_session_skips_close_when_query_unsupported_and_local_flat` (added by `dc2500a8`) exercises the `:3060` NotImplementedError+flat → skip branch. Mutation reasoning sound: flipping `:3060` return True→False fires the elif and calls `on_trading_day_end`, failing the test's `assert_not_called()`.

## Critical issues
NONE. No capital-impact defect.

## Silent gaps
- The EOD close loop (`on_trading_day_end` → `_handle_event`) after a kill switch could itself fail if the broker is degraded (the same condition that may have fired the kill switch). `log.error` at `:3855` is the only per-event guard — no retry, no secondary alert on the loop. ACCEPTABLE because the `MANUAL CLOSE REQUIRED` alert fires at `:3844` before the close is attempted, so the operator is already notified the position is not confirmed flat.

## Unsupported assumptions (now resolved)
- "Evidence-auditor verdict on b5bb4a02 was PASS" was asserted in the `dc2500a8` commit message with no artifact. RESOLVED by this file.

## Tests missing
None for production logic. The degraded-flat branch (`NotImplementedError` + local-flat) lacked a dedicated test before `dc2500a8`; now covered by `test_post_session_skips_close_when_query_unsupported_and_local_flat`.

## Do-not-touch (audit-verified correct)
- Two-object design: `self.positions` (broker adapter) vs `self._positions` (local tracker).
- The `:3042` guard (`signal_only or order_router is None or positions is None`).
- The `:3045` `query_open(self.order_router.account_id)` call.
- `on_trading_day_end` SCRATCH-only (exit-only) event emission.

## Highest-priority fix
Create this artifact (done). No code change required. The fix is cleared for the `--live` flip.
