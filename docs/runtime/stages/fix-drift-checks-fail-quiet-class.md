---
task: Fix drift-check fail-quiet class — empty lanes[], chordia load, raw return-on-no-DB
mode: IMPLEMENTATION
stage: 1/1
classification: judgment
severity: CRITICAL (1 capital-class fail-open + 7 HIGH consistency sites)

scope_lock:
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift.py

acceptance:
  - check_lane_allocation_chordia_gate returns a violation when lane_allocation.json
    has empty lanes[] (was: silent pass — capital-class fail-open per evidence-auditor)
  - check_lane_allocation_chordia_gate returns a violation when load_chordia_audit_log
    raises (was: silent fall-back to hardcoded freshness=90)
  - 7 missing-DB return-paths (in 6 functions) now use _skip_db_check_for_ci helper
    instead of raw `return violations` — consistent with the rest of the file's
    CI-aware fail-closed pattern
  - python pipeline/check_drift.py passes (118+ checks)
  - pytest tests/test_pipeline/test_check_drift.py passes

reasoning:
  Evidence-auditor subagent on pipeline/check_drift.py (8239 lines) returned 5 findings.
  Triaged to 3 confirmed real bugs in this PR (CRITICAL/HIGH only). Class is the same
  silent-default-fall-through that PR #189 and PR #204 fixed — the safety-net layer
  itself has the bug class, which is the worst place to have it.

  Deferred (NOT in this stage — will go to separate stages):
    - execution_engine.py:717 fail-closed conversion (HIGH; core file; triggers
      adversarial-audit gate per rule)
    - paper_trader.py orb_minutes hardcoded =5 (HIGH; non-live but stat-affecting)
    - lane_allocation.json rebalance_date staleness (HIGH; needs canonical decision)
    - conditional_overlays defaults (INSUFFICIENT evidence; needs caller trace)
    - check_data_years_disclosure is_advisory mismatch (cosmetic; LOW)

blast_radius:
  - pipeline/check_drift.py (8239 lines): 7 fix sites in 6 functions:
    - check_lane_allocation_chordia_gate (Check #134) — 2 fixes:
      empty-lanes[] silent pass (CRITICAL), chordia load except (HIGH)
    - check_doc_stats_consistency, check_no_active_e3,
      check_active_validated_filters_routable (×2 sites — function has 2 try-blocks),
      check_active_micro_only_filters_on_real_micros,
      check_active_micro_only_filters_after_micro_launch,
      check_orphaned_validated_strategies — all converting raw `return violations`
      on missing DB to `_skip_db_check_for_ci(...)` (HIGH; consistency).
  - All edits are inside drift checks; no production trade-firing logic touched.
  - No schema changes, no canonical-source touches, no live capital exposure.
  - Tests: existing test_check_drift.py covers the chordia gate; ≥1 new case to add
    for empty-lanes[] behavior.

verification:
  - pytest tests/test_pipeline/test_check_drift.py
  - DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db python pipeline/check_drift.py
