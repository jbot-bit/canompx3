---
mode: IMPLEMENTATION
stage: IMPLEMENTATION
task: Universal filter test — is MES_ATR60 a replacement for all session filters?
pass: 1
scope_lock:
  - scripts/tmp_universal_filter_test.py
  - scripts/tmp_golden_scan.py
  - scripts/tmp_xmes_audit.py
blast_radius:
  - Read-only. No production changes.
acceptance:
  - Head-to-head on raw orb_outcomes with filter classes applied
  - IS and OOS separate
  - Portfolio-level comparison
updated: 2026-03-30T16:00:00Z
---
