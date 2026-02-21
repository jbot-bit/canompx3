# Parallel Walkforward Validation

**Date:** 2026-02-21
**Status:** Approved
**Affected files:** `trading_app/strategy_validator.py` (primary), no other files changed

## Problem

`strategy_validator.py` processes 12,996+ experimental strategies serially. Walkforward (phase 6) dominates at ~100-400ms per strategy. Total runtime: hours. With 32 cores available, this is wasteful.

## Design: Approach A — Parallel Walkforward Only

### Architecture

```
Main Process                          Worker Pool (default 8)
─────────────                         ─────────────────────────
1. Open write con, load strategies
2. Serial loop: phases 1-5
   (cull ~60%, instant)
3. CLOSE write con explicitly         <- Required: no write con open during reads
4. Submit survivors to pool       --> Worker N:
                                       - Receive: (strategy_row, params, db_path)
                                       - Open READ-ONLY DuckDB con
                                       - run_walkforward()
                                       - compute_dst_split()
                                       - Return: lightweight result namedtuple
                                       - Close con
5. Collect results via as_completed <-- (tqdm progress bar)
6. Re-open write con:
   - UPDATE experimental_strategies
   - INSERT validated_setups
   - Write walkforward JSONL
7. FDR correction (unchanged)
8. Log speedup: wall-clock vs sum(individual durations)
```

### Critical Implementation Rules

1. **Serialization-safe worker inputs:** Workers receive `(strategy_id, params_dict, db_path_str)` — no DataFrames, no connection objects. ProcessPoolExecutor serializes everything crossing the process boundary. Workers open their own DuckDB connection inside the function.

2. **No concurrent write+read:** Main process MUST close its write connection before submitting work to the pool. DuckDB allows multiple read-only connections to the same file, but only if no write connection is open. Explicit `con.close()` with comment explaining why.

3. **Speedup logging:** Log wall-clock time for parallel section AND sum of individual walkforward durations. Gives real speedup ratio on first run for tuning `--workers`.

### CLI Addition

```
--workers N    # Default: min(8, cpu_count-1). Set to 1 for serial mode.
```

### What Changes

- `strategy_validator.py`:
  - Refactor main loop to split phases 1-5 (serial) from phase 6 (parallel)
  - Add `_walkforward_worker()` function (serialization-safe, opens own con)
  - Add batch write function for collected results
  - Add `--workers` CLI flag
  - Add speedup logging

### What Stays The Same

- All validation logic (phases 1-6) — identical results
- `walkforward.py`, `strategy_fitness.py`, `db_manager.py` — untouched
- All existing flags (`--dry-run`, `--no-walkforward`, `--no-regime-waivers`, etc.)
- FDR correction (post-hoc)
- `--resume` compatibility
- JSONL checkpoint format
- `--workers 1` produces identical output to current serial code

### Safety Guarantees

- Workers never write to DB (read-only connections)
- All writes in single main-process transaction after pool completes
- Idempotent — safe to re-run if interrupted
- One worker exception doesn't kill the run — strategy marked FAILED, others continue
