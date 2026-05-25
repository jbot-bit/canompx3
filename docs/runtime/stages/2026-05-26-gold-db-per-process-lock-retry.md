---
task: |
  CAPITAL-CLASS — gold.db per-process lock causes IOException for ANY second
  tool (refresh_data.py, drift, MCP server) even with `read_only=True` while
  the orchestrator holds the DB. Third documented recurrence (per
  `feedback_duckdb_windows_lock_is_per_process.md`).

  ROOT CAUSE (officially documented):
  Per DuckDB official docs (https://duckdb.org/docs/current/connect/concurrency.html):
  > "DuckDB handles concurrent database access requests using file locks.
  >  Exercise extra caution when accessing a DuckDB database file in a
  >  shared directory."
  >
  > Read-write mode: one process can both read and write to the database.
  > Read-only mode: multiple processes can read from the database, but no
  > processes can write.
  >
  > Writing to DuckDB's native database format from multiple processes is
  > supported through the Quack remote protocol [BETA as of v1.5.2].

  The error is documented DuckDB behavior, not a bug. While the orchestrator
  holds read-write mode, ANY other process — including `read_only=True` —
  gets IOException on Windows because DuckDB's exclusive lock prevents
  even the shared-read open. Confirmed:
    - duckdb/duckdb-r#56 (2023-12) — same class
    - duckdb/duckdb#5481 (2022-11) — same class
    - duckdb/duckdb#15641 — Windows file-share behavior
    - this project's `feedback_duckdb_windows_lock_is_per_process.md` (2026-05-22)

  FIX: retry-with-jitter wrapper on every non-orchestrator `duckdb.connect()`.
  Officially-grounded pattern — DuckDB docs § "Conflict Handling": "A common
  workaround when a transaction conflict is encountered is to rerun the
  transaction." We extend that doctrine from transaction-conflict to
  file-lock-IOException with backoff.

  Precedent: 2026-05-24 commit added `_open_writer_with_retry` to
  `trading_app/strategy_validator.py:92` for the same class (validator-vs-peer
  write contention). That fix proved out: 6 attempts × exponential jitter
  (1s/2s/4s/8s/16s/30s cap), ~61s ceiling, non-lock IOExceptions re-raise
  unchanged.

  This stage extends the pattern to ALL non-orchestrator readers:
    - scripts/tools/refresh_data.py:52 (FAILS today)
    - trading_app/mcp_server.py (probable, same class)
    - pipeline/check_drift.py read paths
    - any scripts/tools/*.py that opens gold.db

mode: CLOSED
status: CLOSED_CONTROLLED_LOCK_REPRO_2026_05_25
priority: P1_CAPITAL_CLASS
deferred_reason: |
  Filed during Brisbane Mon 2026-05-26 live-debut session (08:00 CME reopen
  + 4 deployed MNQ lanes trading tonight). Touches canonical-source
  `pipeline/paths.py`-adjacent contract. 158-check drift parse vulnerable
  to any change in DB-open semantics. Must NOT land mid-session.
  Implement after Brisbane Tue 2026-05-26 ~08:00 (post-NYSE_OPEN close).

scope_lock:
  - pipeline/db_connect.py  # NEW — canonical retry wrapper
  - scripts/tools/refresh_data.py
  - trading_app/mcp_server.py
  - pipeline/check_drift.py  # only read paths
  - tests/test_pipeline/test_db_connect.py  # NEW
  - pipeline/paths.py  # optional — re-export _open_with_retry as canonical API

agent: claude (opus 4.7)
---

## Codex Implementation Pass

Status: CLOSED after controlled manual lock repro. The repro used a local
writer-mode DuckDB process holding `gold.db` instead of starting the live
orchestrator; this exercises the same documented DuckDB file-lock class while
avoiding live-session/broker side effects.

What landed:

- Added `pipeline/db_connect.py` with `open_read_only_with_retry()` and
  `open_writer_with_retry()`.
- Refactored `trading_app/strategy_validator.py` to delegate writer retry to
  the canonical helper and use the read-only helper on validator DB readers.
- Migrated `scripts/tools/refresh_data.py` and DB-dependent
  `pipeline/check_drift.py` read-only opens to `open_read_only_with_retry()`.
- Added `tests/test_pipeline/test_db_connect.py`.

Verification:

- `./.venv-wsl/bin/python -m pytest tests/test_pipeline/test_db_connect.py tests/test_trading_app/test_strategy_validator.py::TestOpenWriterWithRetry tests/test_tools/test_refresh_data.py -q`
  -> 16 passed.
- `./.venv-wsl/bin/python -m pytest tests/test_pipeline/test_check_drift.py tests/test_pipeline/test_check_drift_ws2.py tests/test_pipeline/test_check_drift_db.py -q`
  -> 355 passed.
- `./.venv-wsl/bin/python -m pyright pipeline/db_connect.py scripts/tools/refresh_data.py trading_app/strategy_validator.py pipeline/check_drift.py`
  -> 0 errors, 0 warnings.
- `./.venv-wsl/bin/python -m ruff check pipeline/ trading_app/ scripts/ tests/test_pipeline/test_db_connect.py --quiet`
  -> pass.
- `./.venv-wsl/bin/python scripts/tools/audit_behavioral.py` -> pass.
- `./.venv-wsl/bin/python scripts/tools/audit_integrity.py` -> pass.
- `./.venv-wsl/bin/python pipeline/check_drift.py --quiet` -> clean,
  164 passed, 20 advisory.
- `git diff --check` -> pass.
- Full `./.venv-wsl/bin/python -m pytest -q` did not complete inside the
  Codex sandbox: it timed out first on the known dashboard subprocess test
  `test_prepare_profile_for_start_propagates_mode_to_subprocess`, then on a
  Starlette `TestClient` portal-thread CSRF test when the first test was
  deselected. Both affected dashboard checks passed outside the sandbox:
  exact subprocess test -> 1 passed in 0.20s; CSRF file -> 10 passed in 0.24s.

Manual repro evidence:

- Lock holder:
  `./.venv-wsl/bin/python -c "import duckdb, time; con=duckdb.connect('gold.db'); print('LOCK_HELD', flush=True); time.sleep(8); con.close(); print('LOCK_RELEASED', flush=True)"`
  -> printed `LOCK_HELD`, then `LOCK_RELEASED`.
- A concurrent dry-run refresh:
  `./.venv-wsl/bin/python scripts/tools/refresh_data.py --instrument MNQ --dry-run`
  -> logged `[duckdb-read-only-retry] gold.db locked (attempt 1/6)` and
  `attempt 2/6`, then completed the dry-run summary with `MNQ OK`.
- The dry-run did not mutate `gold.db`; the only external-data warning was
  Databento DNS resolution for range metadata, and the command correctly
  proceeded without clamp in dry-run mode.

## Blast Radius

- WRITES: NEW `pipeline/db_connect.py` (~120 lines) — `open_read_only_with_retry(path, max_attempts=6, base_delay=1.0)` mirroring `strategy_validator._open_writer_with_retry`. Catches `duckdb.IOException` only when message contains "being used by another process" OR "Could not set lock". Other IOExceptions re-raise unchanged. Logs WARN on attempt, INFO on success-after-retry, ERROR on exhaustion.
- WRITES (modify): `refresh_data.py:52` and any other identified caller — wrap `duckdb.connect(str(GOLD_DB_PATH), read_only=True)` in the canonical retry helper. ~1 line each.
- WRITES: companion test file with 4 tests: (a) succeeds first attempt no lock, (b) succeeds after 2 lock-then-clear, (c) fails after max attempts with all-lock, (d) re-raises non-lock IOException unchanged. Use threading.Event to coordinate fake-lock release in tests.
- READS: `strategy_validator.py:92` `_open_writer_with_retry` as canonical pattern (do NOT duplicate logic — promote it to `pipeline/db_connect.py` and have validator re-import).
- LIVE-IMPACT: zero functional behavior change in steady-state. Behavior change ONLY when lock contention occurs: hard-fail → soft-defer up to ~60s. Live orchestrator's bursty writes (validator writeback, outcome_builder, daily_features rebuild) typically resolve in <30s.
- Idempotency: retries are functionally idempotent (each attempt is an open-only operation; nothing committed until caller transacts).
- Rollback: revert helper file + per-call-site reverts.

## Acceptance

1. `pipeline/db_connect.open_read_only_with_retry` exists with full 4-test coverage; all tests pass (show output).
2. `refresh_data.py:52` uses the helper. Manual repro: start orchestrator → run `python scripts/tools/refresh_data.py` → succeeds within 60s instead of hard-failing.
3. `strategy_validator.py:92` `_open_writer_with_retry` REFACTORED to import from `pipeline.db_connect` (canonical-source delegation per `institutional-rigor.md` § 4). Pre-existing 31 validator tests still pass.
4. Drift check #45+ scan: no new `duckdb.connect` literals introduced; existing direct opens audited for whether they need wrapping (some legitimately don't — the orchestrator itself MUST open in read-write).
5. `pipeline/check_drift.py` passes with no new violations.

## Why retry is the correct fix (and not Quack / DuckLake)

- Quack is **beta as of v1.5.2** per official docs. Production stability not certified.
- DuckLake requires Postgres catalog — adds infrastructure dependency for marginal benefit (this project is single-machine).
- Retry-with-wait honors DuckDB's official "rerun the transaction" guidance and is the same pattern the project has already proven once (F1 validator fix 2026-05-24). N=2 same-class doctrine threshold: precedent exists, expand the pattern.

## Doctrine references

- `feedback_duckdb_windows_lock_is_per_process.md` (n=1, 2026-05-22)
- `memory/feedback_validated_setups_partial_refresh_n1_2026_05_21.md` (related class)
- `institutional-rigor.md` § 4 (canonical-source delegation), § 6 (no silent failures)
- `feedback_n3_same_class_doctrine_threshold.md` (n=3 mechanical-enforcement doctrine — this is n=3+ for the lock class)

## Sources

- DuckDB Concurrency docs: https://duckdb.org/docs/current/connect/concurrency.html
- DuckDB issue #56 (duckdb-r): https://github.com/duckdb/duckdb-r/issues/56
- DuckDB issue #5481: https://github.com/duckdb/duckdb/issues/5481
- DuckDB issue #15641: https://github.com/duckdb/duckdb/issues/15641
- Precedent commit: 2026-05-24 F1 validator retry (trading_app/strategy_validator.py:92)
