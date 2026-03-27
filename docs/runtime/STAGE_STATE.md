---
mode: IMPLEMENTATION
task: "ML Phase 1: Fix A-F methodology rehabilitation"
stage_purpose: "Implement 6 approved ML methodology fixes, retrain, bootstrap, BH FDR at K=12"
scope_lock:
  - trading_app/ml/features.py
  - trading_app/ml/meta_label.py
  - trading_app/ml/config.py
  - scripts/tools/ml_bootstrap_test.py
  - scripts/tools/ml_v2_retrain_all.py
  - scripts/tools/ml_license_diagnostic.py
  - docs/pre-registrations/ml-v2-preregistration.md
  - tests/test_trading_app/test_ml/test_config.py
  - tests/test_trading_app/test_ml/test_meta_label.py
  - docs/runtime/STAGE_STATE.md
acceptance:
  - "Fix A: deterministic config selection on repeated calls"
  - "Fix B: _get_session_features('EUROPE_FLOW') drops prior_sessions_broken"
  - "Fix C: constant-column detection uses train_idx only"
  - "Fix F: X_e6.shape[1] == 5 after core feature selection"
  - "Fix E: negative-baseline sessions skipped with reason='negative_baseline'"
  - "Fix D: BH FDR at K=12 applied, both K=12 and K=108 reported"
  - "Pre-registration committed before retrain"
  - "pytest tests/test_trading_app/test_ml/ -x -q passes"
  - "compute_config_hash() output changes from pre-fix hash"
updated: 2026-03-27T18:00:00+10:00
terminal: main
---
