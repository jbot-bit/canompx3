task: Route all 11 raw duckdb.connect() sites in trading_app/live/ through the pipeline.db_connect retry wrappers, and add a fail-closed drift check that bans new raw connects from re-entering the live path.

mode: IMPLEMENTATION

## Scope Lock

- trading_app/live/bot_dashboard.py
- trading_app/live/session_orchestrator.py
- trading_app/live/bar_persister.py
- trading_app/live/quote_persister.py
- trading_app/live/trade_journal.py
- pipeline/check_drift.py
- tests/test_pipeline/test_check_drift.py

## Blast Radius

- bot_dashboard.py — 7 read-only connects (:560 plain-assign, :1001/:1710/:1791/:1832/:3385/:3557 with-blocks) gain lock-retry. Strictly additive: no-lock path identical; transient peer lock now backs off then succeeds instead of raising. configure_connection + error handling preserved.
- session_orchestrator.py — 1 read-only with-block (:1228) gains lock-retry. Local `import duckdb` inside try replaced by top-level wrapper import.
- bar_persister.py — 1 WRITER (:99), persistent con on object. Converts open call to open_writer_with_retry; configure_connection(con, writing=True) + DELETE/INSERT semantics unchanged. Live capital hot-path (tick→bar persistence).
- quote_persister.py — 1 WRITER (:79), persistent con. Same writer conversion. Live capital hot-path (quote persistence).
- trade_journal.py — 1 WRITER (:104), self._con held for object lifetime (capital ledger). Convert the open call ONLY; object semantics identical; configure_connection + schema/migration calls unchanged.
- pipeline/check_drift.py — one new fail-closed check (check_no_raw_duckdb_connect_in_live), scoped to trading_app/live/, allowlists db_connect.py. After all 11 edits expected violations = 0. Adds ~1s (single-dir rglob).
- tests/test_pipeline/test_check_drift.py — known-violation injection test proving the guard guards.
- Reads/Writes to gold.db & journals: unchanged in semantics — only the connection-acquisition path changes.
- NOT touched: execution engine, risk manager, broker, pipeline/ connects (named follow-up).

## Approach

Drop-in call swaps using two already-proven patterns (trading_app/strategy_validator.py:623 reader with-block, :1389 writer with-block, refresh_data.py:51 reader plain-assign). No new helper, no wrapper change, no force-unlock (permanently rejected — corruption risk). Drift check mirrors check_no_hardcoded_scratch_db (#62). Independent evidence-auditor pass on the diff before final commit per adversarial-audit-gate (touches trading_app/live/ + trade-ledger writer).

## Audit verdict — PASS (2026-06-12, evidence-auditor independent context)

All 3 mandated claims MEASURED+PASS: (1) wrapper retries ONLY the 2 documented lock markers, everything else (schema/corruption/real write failure) re-raises immediately; converted sites' except clauses (duckdb.IOException subclasses duckdb.Error) still fire after retries exhaust. (2) Writer DELETE+INSERT+close semantics unchanged — wrapper returns a plain duckdb connection, no mode change, no context-manager wrapping on writer path. (3) trade_journal lifetime self._con identical — only the open call changed; the :110 except duckdb.IOException PID-extraction path still fires for a genuinely-held lock. No false-PASS hole in the drift regex (rglob recurses subdirs; db_connect.py exempt by name).

CAVEAT RESOLVED: trade_journal.__init__ (and the preflight.py:486 journal-health gate that constructs it) now backs off up to ~90s on a GENUINELY-held lock before reporting TradeJournalLockedError, vs immediate failure before. Verified no timeout wraps the preflight journal-health construction (the timeouts at preflight.py:538/:769 are unrelated subprocess git calls). Behavior is strictly IMPROVED: a STALE lock (dead holder — DuckDB releases on exit) now self-heals during the retry window instead of false-reporting LOCKED; a genuinely-held lock (live holder) is still reported correctly after backoff, and in that case the operator must stop the holder anyway, so the wait is benign.
