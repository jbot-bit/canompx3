---
task: Harden singular resolve_account_id() to fail-closed on >1 active account (Capital review A defense-in-depth)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/projectx/contract_resolver.py
  - trading_app/live/rithmic/contracts.py
  - trading_app/live/tradovate/contracts.py
  - tests/test_trading_app/test_projectx_contract_resolver.py
  - tests/test_trading_app/test_rithmic_router.py
  - tests/test_trading_app/test_tradovate.py
---

## Blast Radius

- `trading_app/live/projectx/contract_resolver.py` — adds a `len(accounts) > 1`
  fail-closed guard to the SINGULAR `resolve_account_id()`. Plural
  `resolve_all_account_ids()` untouched.
- `trading_app/live/rithmic/contracts.py` — same guard, its own data shape
  (`auth.client.accounts`).
- `trading_app/live/tradovate/contracts.py` — same guard; counts ACTIVE accounts
  (its `resolve_all_account_ids` already filters `active=True`).
- Callers of SINGULAR `resolve_account_id`: `session_orchestrator.py:647` and
  `webhook_server.py:131` — both fallback-only (`account_id is None or == 0`).
  With explicit `--account-id`/profile binding set, the singular method is never
  called. Copy-trade (`copies=5`) uses the PLURAL method → unaffected.
- Live-arm preflight `_check_account_binding` (run_live_session.py:784) already
  fail-closes the multi-account-no-binding case; this inner guard is
  defense-in-depth for any caller that bypasses preflight.
- Tests: extends tradovate (existing `test_resolve_account_id` currently enshrines
  the silent multi-account pick — updated to assert the raise + a new
  single-active happy path); extends rithmic (new ambiguity test; existing
  singular tests use 1 account → unaffected); new projectx resolver test file.
- Reads: none new. Writes: none (no gold.db, no live state, no profile schema).
- NOT touched: prop_profiles.py, account_survival.py, the preflight gate, C11.

## Verification
- New/updated tests pass (raise on >1 active; return on exactly 1).
- `ruff check` + `py_compile` clean on the 3 resolvers.
- `python pipeline/check_drift.py` no regression.
- Adversarial-audit gate (capital path) before any live arm. No live arm without
  operator GO.
