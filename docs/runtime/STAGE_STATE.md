---
task: "Master guardian audit: fix fail-open defects, stale narrative, canonical violations, test staleness"
mode: IMPLEMENTATION
stage: 3
stage_of: 4
stage_purpose: "Phase 3: Replace hardcoded values with canonical imports"
updated: 2026-03-24T22:00+10:00
terminal: main
scope_lock:
  - scripts/tools/zero_context_audit.py
  - scripts/tools/hypothesis_test.py
  - scripts/tools/ml_hybrid_experiment.py
  - scripts/tools/rolling_portfolio_assembly.py
  - trading_app/ai/cli.py
  - pipeline/asset_configs.py
  - pipeline/check_drift.py
acceptance:
  - "Hardcoded instrument lists replaced with ACTIVE_ORB_INSTRUMENTS"
  - "Hardcoded session lists replaced with SESSION_CATALOG"
  - "Duplicated FRICTION dict replaced with COST_SPECS"
  - "MBT has explicit orb_active: False"
  - "Drift check #61 recognizes frl_join variable"
  - "Behavioral audit passes (0 violations)"
blockers: []
---
