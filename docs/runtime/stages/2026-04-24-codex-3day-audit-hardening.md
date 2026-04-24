---
mode: IMPLEMENTATION
slug: 2026-04-24-codex-3day-audit-hardening
task: Apply hardening fixes from 3-day Codex audit — T4.A pulse trim and RA-3 doc-hygiene placeholder extension
created: 2026-04-24
updated: 2026-04-24
scope_lock:
  - scripts/tools/project_pulse.py
  - pipeline/check_drift.py
  - trading_app/conditional_overlays.py
  - docs/audit/2026-04-24-codex-3day-audit.md
acceptance:
  - command: PYTHONPATH=. python -m pytest tests/test_tools/test_pulse_integration.py tests/test_pipeline/test_check_drift_doc_hygiene.py -x -q
    expect: all pass
  - command: PYTHONPATH=. python pipeline/check_drift.py
    expect: 108+ checks pass; RA-3 may surface NEW violations on the 3 known commit_sha:PENDING prereg files which is the intended detection
  - audit doc updated with hardening status
blast_radius: project_pulse.py text formatter only (markdown/json formatters untouched, JSON consumers still get full doctrine/backbone lists). check_drift.py extends 1 regex inside check_doc_hygiene_contracts. May surface new violations on 3 stale prereg files (commit_sha PENDING) — intended behavior, not regression. No live trading, no canonical layer, no schema, no DB writes touched.
---
