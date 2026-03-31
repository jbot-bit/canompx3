---
stage: IMPLEMENTATION
mode: IMPLEMENTATION
task: Multi-account copy trading — CopyOrderRouter + account discovery
updated: 2026-04-01T12:00:00Z
scope_lock:
  - trading_app/live/projectx/contract_resolver.py
  - trading_app/live/copy_order_router.py
  - trading_app/live/session_orchestrator.py
  - scripts/run_live_session.py
  - trading_app/live/broker_base.py
blast_radius:
  - contract_resolver.py: additive only (new method)
  - copy_order_router.py: NEW file
  - session_orchestrator.py: order_router construction when copies > 1
  - run_live_session.py: --copies flag
  - broker_base.py: NOT changing ABC (CopyOrderRouter inherits BrokerRouter directly)
acceptance:
  - resolve_all_account_ids() returns all TopStep Express accounts
  - CopyOrderRouter.submit() fans out to N accounts
  - --copies flag discovers accounts and creates CopyOrderRouter
  - Existing single-account flow unchanged
  - Preflight passes
  - No test failures
---
