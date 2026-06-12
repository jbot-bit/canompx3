task: Route all raw duckdb.connect() sites in the EOD backfill chain (daily_backfill + the 4 write-children it spawns) through the pipeline.db_connect retry wrappers. Stage 1 = production swap only.

mode: IMPLEMENTATION

stage: 1 of 2

## Scope Lock

- pipeline/daily_backfill.py
- pipeline/ingest_dbn.py
- pipeline/build_bars_5m.py
- pipeline/build_daily_features.py
- trading_app/outcome_builder.py

## Blast Radius

- daily_backfill.py — 3 in-process connects (L25 READ, L47 READ, L107 WRITE) gain lock-retry. Acquisition-only: configure_connection + try/finally close byte-identical. Drops dead top-level `import duckdb` (L15) once all 3 swapped (no other duckdb. ref).
- ingest_dbn.py — 1 WRITER (L224, bare-assign + atexit). Swap connect call only; con=None sentinel, configure_connection(con, writing=True), _close_con closure + atexit.register all preserved. atexit guards sys.exit paths.
- build_bars_5m.py — 1 WRITER (L347, `with`-form). Swap connect token in the `with` header. KEEP `import duckdb` (L42/56/232 use duckdb.DuckDBPyConnection type hints).
- build_daily_features.py — 1 WRITER (L2119, `with`-form). Same `with`-header swap. KEEP `import duckdb` (many DuckDBPyConnection type hints).
- outcome_builder.py — 1 WRITER (L747, `with`-form, function-local). Swap connect token; DROP function-local `import duckdb` (L731) — only used at L747.
- Reads/Writes to gold.db: unchanged in semantics — only connection-acquisition path changes (retry-and-wait on lock-class IOError). Behavior strictly improved: stale lock self-heals, live holder reported after ~39s backoff.
- NOT touched: drift check + tests (Stage 2), orchestrator EOD path, is_up_to_date logic.

## Approach

Drop-in call swaps using the already-proven patterns from the 2026-06-12 live-side stage (all 11 trading_app/live/ connects converted, evidence-auditor PASS). Wrapper returns a plain duckdb connection, so .close() / atexit / `with ... as con` all work transparently. No force-unlock (corruption risk, permanently rejected). Independent evidence-auditor pass on the diff before commit (pipeline/ truth-layer + capital-path + lock-exposure = HIGH).

Plan: C:/Users/joshd/.claude/plans/1-relaunch-the-bot-swift-pretzel.md
