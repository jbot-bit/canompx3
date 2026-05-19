---
task: AM3.3-AUDIT-LOG-DRIFT: drift check asserting chordia_audit_log.yaml theory_grants parity with prereg metadata.theory_grant
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift_am33_audit_log_drift.py
  - docs/ralph-loop/deferred-findings.md
---

## Blast Radius
- pipeline/check_drift.py — adds check_am33_audit_log_theory_grant_parity + CHECKS entry (blocking, file-level only, no DB)
- tests/test_pipeline/test_check_drift_am33_audit_log_drift.py — new injection-test file
- docs/ralph-loop/deferred-findings.md — marks AM3.3-AUDIT-LOG-DRIFT as CLOSED
