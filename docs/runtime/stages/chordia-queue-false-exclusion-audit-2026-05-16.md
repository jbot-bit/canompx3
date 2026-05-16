---
task: Audit May 12 Chordia queue for false exclusions
mode: IMPLEMENTATION
slug: chordia-queue-false-exclusion-audit-2026-05-16
created: 2026-05-16
scope_lock:
  - research/audit_chordia_queue_false_exclusions.py
  - docs/audit/hypotheses/2026-05-16-chordia-queue-false-exclusion-audit.yaml
  - docs/audit/results/2026-05-16-chordia-queue-false-exclusion-audit.md
  - docs/audit/results/2026-05-16-chordia-queue-false-exclusion-audit.csv
  - docs/audit/results/2026-05-16-chordia-queue-rescued-quarantine.csv
---

## Blast Radius

- research/audit_chordia_queue_false_exclusions.py — new file, zero callers, read-only DB access via duckdb.connect(..., read_only=True)
- docs/audit/hypotheses/2026-05-16-chordia-queue-false-exclusion-audit.yaml — new confirmatory pre-reg (thin per research-truth-protocol § 10; not new discovery)
- docs/audit/results/2026-05-16-chordia-queue-false-exclusion-audit.md — verdict doc; pooled_finding=true front-matter if any cross-instrument aggregation
- docs/audit/results/2026-05-16-chordia-queue-false-exclusion-audit.csv — 11-row per-gate summary (one row per audited gate)
- docs/audit/results/2026-05-16-chordia-queue-rescued-quarantine.csv — rescued rows (quarantine-only; NO deployment)
- Reads: gold.db (read-only via pipeline.paths.GOLD_DB_PATH); docs/audit/results/2026-05-12-chordia-audit-queue-candidates.csv; docs/runtime/lane_allocation.json (read-only); docs/runtime/chordia_audit_log.yaml (read-only)
- Writes: only the 5 audit artifact paths above. NO writes to validated_setups, chordia_audit_log.yaml, lane_allocation.json, gold.db, or any pipeline/trading_app file.

Null hypothesis: H0 = funnel applies its stated exclusion rules correctly (zero false exclusions). Exit conditions: rescue_count==0 -> FUNNEL_VALIDATED; 1..20 -> FUNNEL_BUGS_FOUND; >20 -> FUNNEL_SYSTEMIC_BIAS (escalate).
