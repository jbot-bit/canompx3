---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: FDR audit script + methodology spec — fix 3 critical bugs + 3 grounding errors
pass: 1
scope_lock:
  - scripts/tools/audit_fdr_integrity.py
  - docs/specs/fdr_methodology.md
blast_radius:
  - audit script: reference BH function, check_9 pool query, rounding
  - methodology spec: page citations, PRDS claim, FST K=228 math
acceptance:
  - audit_fdr_integrity.py runs 13/13 clean
  - bh_fdr_reference matches production benjamini_hochberg exactly
  - check_9 uses full canonical pool (not just survivors)
  - All spec citations verified against local PDFs
  - Drift checks pass
updated: 2026-03-30T18:00:00Z
---
