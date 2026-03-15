## Iteration: 65
## Target: pipeline/check_drift.py:565-582
## Finding: check_entry_models_sync() (check #13) is a circular self-check — it imports ENTRY_MODELS from trading_app.config and compares against a hardcoded copy ["E1","E2","E3"]. It will never detect real drift because it's comparing the canonical source to itself. Fix: cross-check config.ENTRY_MODELS against sql_adapter.VALID_ENTRY_MODELS (the separate set that gates AI queries).
## Blast Radius: 3 files: pipeline/check_drift.py (fix), tests/test_pipeline/test_check_drift_ws2.py (test passes unchanged), scripts/audits/phase_4_config_sync.py (read-only reference)
## Invariants:
## 1. check_entry_models_sync() must still return [] when config and adapter are in sync
## 2. The check must handle ImportError gracefully (keep except clause)
## 3. No change to check #47 (check_trading_rules_authority) — its hardcoded expected list is intentional
## Diff estimate: 8 lines
