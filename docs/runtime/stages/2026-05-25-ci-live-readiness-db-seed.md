---
task: |
  Fix CI red (10+ consecutive red runs, HEAD 559eb0e7). The single failing
  test `test_json_cli_stdout_is_parseable_without_warning_preamble` in
  tests/test_tools/test_live_readiness_report.py spawns
  `scripts/tools/live_readiness_report.py --format json` as a real subprocess
  with check=True. On CI there is no gold.db by policy (CLAUDE.md "local disk,
  no cloud sync"; CI drift checks use _skip_db_check_for_ci). The script's
  _load_validated_strategy_ids (line 132) runs SELECT strategy_id FROM
  validated_setups, which raises duckdb.CatalogException when the table is
  absent -> uncaught -> exit 1 -> test fails with CalledProcessError. Fix: make
  the test seed a minimal temp DuckDB (validated_setups with strategy_id,status)
  and point the subprocess at it via DUCKDB_PATH (canonical override per
  pipeline/paths.py:114, honored only when the file exists). Preserves the
  stdout-cleanliness contract the test actually verifies; does NOT weaken the
  script's fail-loud behavior on a genuinely missing local DB.
mode: IMPLEMENTATION
scope_lock:
  - tests/test_tools/test_live_readiness_report.py
blast_radius: |
  Test-only change. tests/test_tools/test_live_readiness_report.py — modifies
  ONE test (test_json_cli_stdout_is_parseable_without_warning_preamble) to seed
  a temp DuckDB and pass DUCKDB_PATH in the subprocess env. No production code
  touched. No schema change. Reads: none (test creates its own temp DB via
  tmp_path). Writes: temp DB only. Zero callers of the test. The script under
  test (live_readiness_report.py) is unchanged — its fail-loud-on-missing-table
  behavior is intentional and preserved. Root cause proven by execution:
  empty duckdb -> CatalogException at line 132; seeding validated_setups(2 cols)
  -> exit 0, clean stderr, stdout starts with '{'.
---

## Root cause (execution-proven)

```
_duckdb.CatalogException: Catalog Error: Table with name validated_setups does not exist!
  at scripts/tools/live_readiness_report.py:132 in _load_validated_strategy_ids
  via build_live_readiness_report (line 481) <- main (line 779)
```

Reproduced locally by pointing DUCKDB_PATH at an empty (tableless) DuckDB file.
With `validated_setups(strategy_id, status)` seeded -> exit 0, clean stderr,
pure-JSON stdout. Confirms `_load_validated_strategy_ids` is the ONLY hard DB
dependency; all downstream lifecycle/survival/overlay readers already fail-soft
(report valid=False rather than raise).

## Why fix the test, not the script

- Every other unit test in this file monkeypatches `_load_validated_strategy_ids`
  and runs in-process; only the subprocess test hits the real DB.
- The script SHOULD fail loud when its truth source (validated_setups) is absent
  locally — that signals broken setup. Making it silently succeed would be the
  band-aid institutional-rigor.md forbids.
- CI legitimately has no gold.db (documented policy). The canonical convention
  is CI-awareness at the consumer boundary (cf. `_skip_db_check_for_ci`). Here
  the consumer is the test, so the test seeds a temp DB — keeping real coverage
  of the subprocess + JSON-cleanliness contract on CI.

## Done criteria

- The target test passes locally AND would pass on CI (no gold.db dependency).
- Other tests in the file still pass.
- `python pipeline/check_drift.py` passes.
- Full test suite run; pass/fail counts reported.
