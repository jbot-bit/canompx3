---
task: Pyright typing fixes on pipeline/build_bars_5m.py and pipeline/build_daily_features.py (Stage 1 of 2026-05-15 punch list)
mode: IMPLEMENTATION
scope_lock:
  - pipeline/build_bars_5m.py
  - pipeline/build_daily_features.py
---

## Blast Radius

- pipeline/build_bars_5m.py — narrow Optional[Tuple] returns from `con.execute(...).fetchone()` before subscripting. Typing-only; no runtime path change. 8 reportOptionalSubscript sites.
- pipeline/build_daily_features.py — coerce pandas Scalar/Series-union at boundary (pd.Timestamp / float / int / str) at ~36 reportArgumentType/reportAttributeAccessIssue sites. Typing-only.
- Reads: gold.db (read-only via existing code paths). Writes: none from this stage's diff. CI pyright job (.github/workflows/ci.yml:39-41) is `continue-on-error: true` — promoting to hard gate is OUT OF SCOPE.
- No callers affected. No schema change. No behavior change. Mutation-proof: if a fetchone returns None at runtime, the new guard raises ValueError with a clear message instead of crashing with TypeError on subscript (strictly better diagnostic).

## Verification

- `uv run pyright pipeline/build_bars_5m.py pipeline/build_daily_features.py` → 0 errors
- `python pipeline/check_drift.py` → all checks pass
- `pytest tests/test_pipeline/ -k "build_bars_5m or build_daily_features" -x` → green (sanity)
