---
task: A.6.5 preflight gap — add copy-trading account-resolution check to preflight
mode: IMPLEMENTATION
updated: 2026-05-16
scope_lock:
  - scripts/run_live_session.py
  - tests/test_scripts/test_run_live_session_preflight.py
blast_radius: "scripts/run_live_session.py adds one new PreflightContext field group (profile_id, requested_account_id) + one new check function _check_copy_trading_accounts + appends to PREFLIGHT_CHECKS + threads two new kwargs through _run_preflight signature and both call sites. Companion test extends. checks_total goes 6 -> 7 (dynamic via len() — no hardcoded count to update). No production live-trading path mutated; preflight-only. Reads: broker contracts.resolve_all_account_ids() (network, dry-run only). Writes: none."
acceptance:
  - python pipeline/check_drift.py exits 0
  - New test exercises (a) copies==1 SKIPPED path, (b) copies>1 happy path (None auto-discover), (c) copies>1 with requested_account_id not in broker accounts FAILS preflight
  - Existing tests in test_run_live_session_preflight.py pass unchanged (checks_total derives dynamically)
  - tests/test_trading_app/test_session_orchestrator.py unchanged-green
---

# A.6.5 — Preflight Copy-Trading Account-Resolution Gap

## The gap (referenced from fix-account-id-sentinel-mismatch.md § Deferred)

Preflight passes 6/6 even when live-start crashes on copies>1 profiles because preflight
runs in `signal_only` mode — `_run_preflight()` never exercises the
`_select_primary_and_shadow_accounts()` branch at `run_live_session.py:688-691`.

Today this is harmless because the sentinel bug it would have caught is now fixed
(commit a0b3c24b). But preflight is still silently incomplete — operator confidence
that "preflight green = live-start safe" is unjustified for the copy-trading branch.

## Fix design

Add one new check to `PREFLIGHT_CHECKS` that dry-runs the copy-trading account
resolution:

1. **`PreflightContext`** gains `profile_id: str | None = None` and
   `requested_account_id: int | None = None`.
2. **`_run_preflight()`** signature gains `profile_id=None, requested_account_id=None`
   kwargs. Backwards-compatible defaults (existing callers and test fixtures unchanged).
3. **`_check_copy_trading_accounts(ctx)`** new function:
   - SKIP if `ctx.profile_id is None` (raw-baseline path, no copy trading).
   - SKIP if `ACCOUNT_PROFILES[profile_id].copies <= 1` (single-account profile).
   - SKIP with FAIL=False if `ctx.components is None` (auth already failed — would skip
     anyway; mirror `_check_contracts` SKIPPED pattern).
   - Otherwise: instantiate `contracts_cls(auth, demo)`, call
     `resolve_all_account_ids()`, dry-run `_select_primary_and_shadow_accounts(...)`,
     discard the result. Any `RuntimeError` becomes a preflight FAIL with the error
     message.
4. **`PREFLIGHT_CHECKS`** list gains the new check at the end (after trade journal —
   ordering is "auth → portfolio → daily features → contracts → notifications →
   trade journal → copy-trading-accounts"). State coupling: depends on `ctx.components`
   set by `_check_auth`.
5. **Both `_run_preflight` call sites** at lines 568 and 573 thread
   `profile_id=args.profile, requested_account_id=args.account_id`.

### Why not refactor preflight to optionally run not-signal-only

That would change the contract of `_run_preflight` ("no broker order-router actions,
no real trades"). The new check is a DRY RUN — it reads broker account list but never
constructs a router or places an order. Safer to add a single new check than to widen
the preflight contract.

### Companion test

Extend `tests/test_scripts/test_run_live_session_preflight.py`. Three new tests:

- `test_copy_trading_check_skipped_when_no_profile` — `profile_id=None` → SKIPPED.
- `test_copy_trading_check_skipped_when_copies_le_1` — single-account profile → SKIPPED.
- `test_copy_trading_check_fails_on_unknown_requested_account_id` — copies=2, broker
  returns `[21944866]`, `requested_account_id=999999` → preflight FAILS with the same
  error live-start would have raised.
- `test_copy_trading_check_passes_with_none_account_id` — copies=2, broker returns
  multiple accounts, `requested_account_id=None` → PASSES.

### Self-review checklist

- [ ] `python pipeline/check_drift.py` PASS
- [ ] `python -m pytest tests/test_scripts/test_run_live_session_preflight.py -v` all green
- [ ] `checks_total` in preflight output increments 6 → 7 automatically
- [ ] New check never constructs an `OrderRouter` — confirms dry-run-only invariant
- [ ] `grep -n "PREFLIGHT_CHECKS" scripts/run_live_session.py` — only one definition
- [ ] No re-encoded `resolve_account_id` or `_select_primary_and_shadow_accounts` logic

### Adversarial audit

This is `[judgment]` MEDIUM (preflight surface, not live-trading path). Per
`adversarial-audit-gate.md` the audit gate is mandatory for `trading_app/live/` HIGH/CRIT
commits. This commit touches `scripts/` not `trading_app/live/`, so the formal gate
does not auto-trigger — but we still self-review per institutional-rigor § 1+2 and
optionally dispatch evidence-auditor if anything looks risky after implementation.
