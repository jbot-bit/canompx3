---
slug: ingest-dbn-pyright-cleanup
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-27
updated: 2026-04-27
task: Drive 3 ingest_dbn files to 0 pyright errors. 27 errors across mixed clusters — fetchone()[0] (×13), pandas DatetimeIndex .tz/.tz_convert/.isna (×5), Scalar vs date comparisons (×6), Path/str = None default (×3 — real annotation tightening), tuple-to-bool return (×1, type-ignore), sys.stdout.reconfigure (×1, type-ignore for TextIO union narrowing).
---

# Stage: ingest_dbn pyright cleanup

mode: IMPLEMENTATION
date: 2026-04-27

scope_lock:
  - pipeline/ingest_dbn.py
  - pipeline/ingest_dbn_daily.py
  - pipeline/ingest_dbn_mgc.py
  - docs/runtime/stages/ingest-dbn-pyright-cleanup.md

## Why

Last 3 pipeline files with pyright noise. Mixed clusters require per-error judgement.

## Cluster resolutions

| Cluster | Sites | Resolution |
|---|---|---|
| `fetchone()[0]` Optional-subscript | ingest_dbn ×3, ingest_dbn_daily ×3, ingest_dbn_mgc ×6 | `# type: ignore[index]` (same as prior batches — SQL guarantees ≥1 row) |
| Scalar vs date in `groupby("trading_day")` loop | ingest_dbn ×3, ingest_dbn_daily ×3 | `# type: ignore[operator, arg-type]` — pandas groupby key types are inferred broadly; runtime is always python `date` from `compute_trading_days()` |
| pandas `Index[Any]` `.tz` / `.tz_convert` / `.isna().any()` | ingest_dbn_mgc ×5 | `# type: ignore[attr-defined, union-attr]` — `df.index` is `pd.DatetimeIndex` at runtime (validated upstream) but pyright sees `Index[Any]` |
| `Path = None` / `str = None` default annotations | ingest_dbn_mgc L85, L170, L394 | **real fix**: change to `Path \| None = None` / `str \| None = None`. PEP 484 disallows implicit Optional |
| `validate_chunk` 3-tuple return — `df.head()` returns `DataFrame \| Series` | ingest_dbn_mgc L224 | `# type: ignore[return-value]` — boolean-mask indexing always returns DataFrame at runtime |
| `sys.stdout.reconfigure` | ingest_dbn_daily L35 | `# type: ignore[attr-defined]` — runtime is `TextIOWrapper`, stub says `TextIO` |

## Blast Radius

- 3 files modified, ~27 single-line additions plus 3 annotation tightenings.
- Behavior changes:
  - `Path = None` → `Path | None = None` and `str = None` → `str | None = None` (3 sites): PEP 484 strict-Optional. Runtime IS already passing None at call sites; the annotation just becomes truthful.
- Tests: `tests/test_pipeline/` 1234 must still pass.
- Reversibility: single commit.
