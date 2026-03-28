---
mode: IMPLEMENTATION
task: "ML V2 cleanup — remove V1 dead code, fix safety gaps, harden"
stage_purpose: "1) Delete stale model, 2) Fix predict_live safety (hash reject, backfill), 3) Remove V1 dead code (~1300 lines), 4) Harden config hash, instrument args, integration test"
blast_radius: "ML module only. No pipeline/ changes. predict_live.py fail-open behavior preserved. V1 CLI paths removed. 8 stale tests deleted, 1 new test added. ml_audit.py and ml_license_diagnostic.py updated to use V2 loaders."
scope_lock:
  - trading_app/ml/predict_live.py
  - trading_app/ml/meta_label.py
  - trading_app/ml/config.py
  - trading_app/ml/features.py
  - trading_app/ml/__init__.py
  - trading_app/ml/evaluate.py
  - trading_app/ml/evaluate_validated.py
  - trading_app/ml/importance.py
  - scripts/tools/ml_v2_retrain_all.py
  - scripts/tools/ml_bootstrap_test.py
  - scripts/tools/ml_audit.py
  - scripts/tools/ml_license_diagnostic.py
  - tests/test_trading_app/test_ml/test_features.py
  - models/ml/meta_label_MNQ_hybrid.joblib
  - pipeline/check_drift.py
  - docs/runtime/STAGE_STATE.md
acceptance:
  - "No imports of removed functions (grep clean)"
  - "ML tests pass (pytest tests/test_trading_app/test_ml/ -x)"
  - "check_drift.py passes"
  - "predict_live.py rejects hash-mismatched models (fail-closed)"
updated: 2026-03-28T16:00:00+10:00
terminal: main
---
