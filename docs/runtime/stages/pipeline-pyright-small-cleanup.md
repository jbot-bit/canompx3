---
slug: pipeline-pyright-small-cleanup
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-27
updated: 2026-04-27
task: Drive 6 small pipeline files to 0 pyright errors. 9 errors total — all pyright Optional / dict-invariance noise, no real bugs.
---

# Stage: pipeline small-files pyright cleanup

mode: IMPLEMENTATION
date: 2026-04-27

scope_lock:
  - pipeline/dst.py
  - pipeline/daily_backfill.py
  - pipeline/system_brief.py
  - pipeline/health_check.py
  - pipeline/export_parquet.py
  - pipeline/ingest_statistics.py
  - docs/runtime/stages/pipeline-pyright-small-cleanup.md

## Why

Six pipeline files each have 1-2 pyright errors of two patterns:

1. **`fetchone()[0]` Optional-subscript** (daily_backfill ×2, health_check ×1, export_parquet ×1, ingest_statistics ×1) — same as cluster F resolved in build_daily_features.py / check_drift.py / build_bars_5m.py.
2. **`dt.utcoffset().total_seconds()` Optional-attr** (dst.py ×2) — `dt` is constructed with explicit `tzinfo=`, so `utcoffset()` never returns None at runtime; pyright can't see that.
3. **`dict[str, str]` literal → `list[dict[str, object]]` dict-invariance** (system_brief.py ×2) — explicit `dict[str, object]` annotation on the dict literal.

## Blast Radius

- 6 files modified, ~9 single-line annotations.
- No behavioral change.
- Reversibility: single commit.
