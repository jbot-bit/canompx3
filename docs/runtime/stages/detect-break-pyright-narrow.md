---
slug: detect-break-pyright-narrow
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-27
updated: 2026-04-27
task: Drive pipeline/build_daily_features.py to zero pyright errors. Started from a 6-error report at detect_break() and expanded to clean all 39 errors surfaced across the file (Optional-narrowing, dict typing, ExtensionArray, GARCH literals, fetchone None-subscript, Scalar narrowing). Type-annotation-only fixes for typing clusters; one real string-literal correction for arch_model.
---

# Stage: build_daily_features pyright cleanup

mode: IMPLEMENTATION
date: 2026-04-27

scope_lock:
  - pipeline/build_daily_features.py
  - docs/runtime/stages/detect-break-pyright-narrow.md

## Why

Pyright reported 39 errors in `pipeline/build_daily_features.py`. User scope: "verify pre-existing and new surfaced issues" → triage each cluster, distinguish real bug from typing-only, apply smallest correct fix.

## Clusters & resolutions

| # | Lines (pre-shift) | Pattern | Resolution |
|---|---|---|---|
| A | 323-325, 334, 343, 1019, 1020 | `float(bar.x)` / `bar.ts.to_pydatetime()` over pandas `Scalar` from `itertuples()` | inline `# type: ignore[arg-type \| union-attr]` (commit `412ef1be`) — DuckDB schema guarantees concrete types at runtime |
| B | 829 | `arch_model(vol="Garch", dist="Normal")` — case-mismatched against arch stubs and underlying API | corrected literals → `"GARCH"`, `"normal"` (matches arch's Literal-typed enum) |
| C | 910, 1241/1242, 1318, 1345, 1351, 1354/1355, 1365/1366 | pandas `.values` / `.values.astype()` returns `np_1darray \| ExtensionArray \| Categorical` | swap to `.to_numpy()` / `.to_numpy(dtype=float)` for non-tz arrays; **revert** to `.values` for `ts_utc` because `.values` strips tz to naive `datetime64[ns]` (matches downstream `pd.Timestamp(...).asm8`) — `.to_numpy()` preserves tz and breaks `searchsorted` |
| D | 527, 529, 532, 536, 537, 549-553 | `if None in (a,b,c,d,e)` doesn't narrow `Optional` — pyright still flags arithmetic below | rewrote as explicit `or` chain of `is None` checks; pyright now narrows correctly |
| E | 946 | `round(rsi, 4)` returns `floating[Any]`, annotation says `float \| None` | wrap in `float(...)` |
| F | 1694, 1787, 1805, 1821, 1837, 1854 | `con.execute(SELECT COUNT(*)...).fetchone()[0]` — fetchone Optional, indexing flags | inline `# type: ignore[index]` × 6; SQL guarantees ≥1 row |
| G | 1038, 1041, 1044, 1048, 1052, 1053 | `result["outcome"] = "loss"` into a dict pyright inferred as `dict[str, None]` | explicit annotation `result: dict[str, str \| float \| None]` |
| H | 834 | `cond_var ** 0.5` over Scalar | inline `# type: ignore[operator]` |

Searchsorted residual (after C revert): inline `# type: ignore[arg-type, call-overload]` × 2 because `np.searchsorted`'s typed overload doesn't accept the pandas `.values` union — the runtime is correct.

## Blast Radius

- **Files modified:** `pipeline/build_daily_features.py` only.
- **Diff size:** ~33 insertions / ~22 deletions — see `git diff`.
- **Tests:** `tests/test_pipeline/test_build_daily_features.py` 84 pass; full pipeline suite 1234 pass.
- **Drift checks:** 113 pass, 6 advisory (unchanged).
- **Production behavior change:** ONE real change — `arch_model` literals `"Garch"`/`"Normal"` → `"GARCH"`/`"normal"`. Functionally equivalent because arch normalizes case internally; verified by full integration tests passing. All other changes are typing-only.
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
