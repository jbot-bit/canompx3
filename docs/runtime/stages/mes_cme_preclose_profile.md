---
task: "Tier 3: Build MES CME_PRECLOSE deployment profile"
mode: IMPLEMENTATION
stage_purpose: >
  Create a new topstep_50k_mes_auto profile with a single MES CME_PRECLOSE lane
  (ORB_G8 RR1.0). Pre-flight completed: SR CONTINUE on 2026 OOS, COST_LT08 is
  strict subset of ORB_G8 (deploy ONE only), year-by-year PASS.
scope_lock:
  - trading_app/prop_profiles.py
  - tests/test_trading_app/test_prop_profiles.py
acceptance:
  - "python -m pytest tests/test_trading_app/test_prop_profiles.py -x -q passes"
  - "PYTHONPATH=. python pipeline/check_drift.py shows no new violations"
  - "New profile topstep_50k_mes_auto exists with 1 lane MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8"
  - "Profile active=False (user activates when ready)"
proven:
  - "MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8 is active in validated_setups"
  - "SR pre-flight CONTINUE (max_SR=11.46, threshold=31.96, N=11 OOS)"
  - "COST_LT08 is strict subset of ORB_G8 — deploy ORB_G8 only"
  - "WF 5/5 passed, FDR significant, p=0.00123"
unproven:
  - "OOS N=11 is small — monitor closely"
  - "Declining trend in avg_R (2020-2022: ~0.25, 2024-2025: ~0.05)"
blast_radius: "prop_profiles.py ACCOUNT_PROFILES consumed by pre_session_check, session_orchestrator, prop_portfolio, check_drift#95. Inactive profile = zero runtime impact. Tests: test_prop_profiles.py. No schema/data/config cascade."
blockers: []
updated: 2026-04-12T14:00:00+10:00
agent: claude
---
