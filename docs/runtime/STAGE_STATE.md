---
task: "Master guardian audit: fix fail-open defects, stale narrative, canonical violations, test staleness"
mode: IMPLEMENTATION
stage: 4
stage_of: 4
stage_purpose: "Phase 4: Fix stale tests and env setup"
updated: 2026-03-24T22:30+10:00
terminal: main
scope_lock:
  - tests/test_trading_app/test_outcome_builder.py
  - tests/test_trading_app/test_multi_runner.py
  - tests/tools/test_pinecone_manifest.py
  - ui/copilot_helpers.py
  - ui/session_helpers.py
  - scripts/tools/pinecone_manifest.json
acceptance:
  - "Time-stop tests pass with patch.dict"
  - "Multi-runner test derives instrument count from canonical"
  - "Copilot translator includes X_MES_ATR70"
  - "No new drift introduced"
blockers: []
---
