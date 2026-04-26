---
slug: detect-break-pyright-narrow
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-27
updated: 2026-04-27
task: Silence pyright Scalar→float / Scalar→Timestamp narrowing errors in pipeline/build_daily_features.py:detect_break (lines 323-325). Type-annotation-only fix, zero runtime impact.
---

# Stage: detect_break pyright Scalar narrowing

mode: IMPLEMENTATION
date: 2026-04-27

scope_lock:
  - pipeline/build_daily_features.py
  - docs/runtime/stages/detect-break-pyright-narrow.md

## Why

Pyright reports 3 errors at `pipeline/build_daily_features.py:323-325` because pandas `df.itertuples()` types row attributes as the union `Scalar = int | float | bool | str | bytes | complex | date | datetime | timedelta | datetime64[...]` even when the underlying DataFrame columns are typed `DOUBLE` and `TIMESTAMPTZ` in DuckDB. The runtime is correct; only the static checker complains. Smallest diff is 3 inline `# type: ignore` comments.

## Blast Radius

- **Files modified:** `pipeline/build_daily_features.py` only (3-line change at 323-325).
- **Function:** `detect_break()` — invoked by the per-ORB feature builder; returns `dict`. Internal `close`/`bar_open`/`bar_ts` types not visible to callers.
- **Tests:** `tests/test_pipeline/test_build_daily_features.py` (84 tests). Zero behavior change → no test changes required.
- **Drift checks:** none affected (113 checks, all unrelated).
- **Production behavior:** zero impact. `# type: ignore` comments are erased at runtime.
- **Reversibility:** single commit; revert via `git revert`.

## Self-check

- Happy path: `float(np.float64)` → float. `pd.Timestamp.to_pydatetime()` → datetime. ✓ unchanged.
- NaN row: `float(nan)` → nan; `nan > orb_high` is False. ✓ unchanged.
- ts_utc None: schema says NOT NULL; current behavior preserved (would AttributeError if violated).
- Pyright recheck: rule-specific suppression (`arg-type` / `union-attr`) preserves all other type checks on the line.

## Why not cast()

`cast(pd.Timestamp, ...)` requires `pd` at runtime, but this file lazy-imports pandas (TYPE_CHECKING-only at module level). Using `cast("pd.Timestamp", ...)` works but adds an import for `cast` from `typing` and 3 wrapped expressions — a larger diff. The user requested smallest diff.

## Adversarial-audit-gate

Not applicable: `[mechanical]` classification, type-annotation-only, zero behavior change. The adversarial-audit gate fires for `[judgment]` commits touching truth-layer paths. This is neither.
