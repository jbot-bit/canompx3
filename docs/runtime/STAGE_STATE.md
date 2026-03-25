---
mode: IMPLEMENTATION
task: TopstepX live bot deployment — fix blockers and wire Apex lanes
scope_lock:
  - trading_app/portfolio.py
  - scripts/run_live_session.py
  - trading_app/live/projectx/order_router.py
  - docs/plans/topstepx-preflight-checklist.md
  - scripts/e2e_sim_test.py
acceptance:
  - pysignalr installed and SignalR data feed verified
  - 4 Apex lanes load via --profile apex_50k_manual
  - Order cancel format verified on sim account
  - End-to-end sim test passes (7 checks)
  - Preflight checklist committed
  - Drift checks pass
---
