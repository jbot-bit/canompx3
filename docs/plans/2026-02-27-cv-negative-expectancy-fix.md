# Plan: Fix CV Silent Failure on Negative Expectancy (C3)

## Root Cause (Phase 1 Complete)

**File:** `scripts/tools/build_edge_families.py:255`

```python
cv_expr = std_expr / mean_expr if mean_expr > 0 else None
```

When `mean_expr <= 0`, `cv_expr` is set to `None`. This has two effects:

### Effect 1: WHITELISTED gate (N=3-4) — Actually SAFE
`classify_family()` line 57 checks `cv_expr is not None and cv_expr <= 0.5`. When cv_expr is None, this **correctly rejects** — the family gets PURGED, not whitelisted. So WHITELISTED is not getting a free pass.

### Effect 2: Data loss in DB — THE REAL BUG
- `cv_expectancy` stored as NULL instead of a meaningful value
- Any downstream reporting/analysis loses the signal that this family has non-positive mean
- ROBUST families (N>=5) have **no metrics gate at all** — they skip CV entirely regardless

## Fix (Single Change)

**Line 255:** Replace `None` with `float('inf')` when mean_expr <= 0:

```python
# Before
cv_expr = std_expr / mean_expr if mean_expr > 0 else None

# After
cv_expr = std_expr / mean_expr if mean_expr > 0 else float('inf')
```

### Why `float('inf')` over `abs(mean_expr)`:
- `abs()` gives a valid-looking number that hides the sign problem — a CV of 0.3 on a negative-mean family looks "stable" when it's actually garbage
- `float('inf')` semantically means "infinitely variable" — correct for non-positive mean
- DuckDB DOUBLE columns accept infinity
- WHITELISTED gate: `inf <= 0.5` → False → still correctly rejects
- DB value is now informative (infinity = bad) instead of NULL (unknown)

## Scope
- **1 line change** in `scripts/tools/build_edge_families.py:255`
- No classify_family() changes needed (already handles correctly)
- No schema changes needed (cv_expectancy is DOUBLE)
- No test changes expected (existing tests use positive expectancy families)

## Verification
1. Run existing tests: `python -m pytest tests/ -x -q`
2. Check if any ROBUST families in DB currently have NULL cv_expectancy with negative mean (diagnostic query)
3. Optionally re-run `build_edge_families.py --all` to backfill correct values
