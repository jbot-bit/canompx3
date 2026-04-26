---
slug: build-bars-5m-pyright-narrow
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-27
updated: 2026-04-27
task: Drive pipeline/build_bars_5m.py to 0 pyright errors. All 8 errors are the same `con.execute(...).fetchone()[0]` Optional-subscript pattern. Inline type-ignore × 8.
---

# Stage: build_bars_5m pyright cleanup

mode: IMPLEMENTATION
date: 2026-04-27

scope_lock:
  - pipeline/build_bars_5m.py
  - docs/runtime/stages/build-bars-5m-pyright-narrow.md

## Why

Same pattern as check_drift.py / build_daily_features.py Cluster F: pyright types `fetchone()` as Optional, but COUNT/SELECT-scalar SQL guarantees ≥1 row. Inline `# type: ignore[index]` on each `.fetchone()[0]` site (8 lines: 66, 142, 160, 193, 232, 250, 265, 280).

## Blast Radius

- **Files modified:** `pipeline/build_bars_5m.py`.
- **Diff size:** ~8 single-line additions.
- **Tests:** `tests/test_pipeline/test_build_bars_5m.py` if present, else integration smoke via `pytest tests/test_pipeline/`.
- **Production behavior change:** zero.
- **Reversibility:** single commit.
