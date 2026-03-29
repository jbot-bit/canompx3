---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: FDR methodology — correctness + honesty fixes (3 phases)
pass: 2
scope_lock:
  - trading_app/strategy_validator.py
  - scripts/tools/audit_fdr_integrity.py
  - docs/specs/fdr_methodology.md
blast_radius:
  - discovery_k freeze (behavioral change to UPDATE — first write same, subsequent guarded)
  - Re-validation kills >=1 MNQ strategy (MNQ_LONDON_METALS_E2_RR3.0_CB1_ORB_G8_NOMON_S075, 6 siblings survive)
  - edge_families rebuild needed after re-validation
acceptance:
  - Phase 1: discovery_k freeze verified (existing values preserved after validator run)
  - Phase 2: audit_fdr_integrity.py runs 13/13 clean (no exceptions)
  - Phase 3: methodology spec exists at docs/specs/fdr_methodology.md
  - Phase 4: drift check passes
updated: 2026-03-30T16:00:00Z
---
