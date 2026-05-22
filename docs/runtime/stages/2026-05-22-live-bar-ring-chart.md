---
task: IMPLEMENTATION — Live bars ring for dashboard chart. Today the dashboard chart reads `bars_1m` from `gold.db` via `_bars_watcher` (bot_dashboard.py:2734) and `_query_bars_recent` (bot_dashboard.py:2879). `BarPersister.flush_to_db()` only writes bars at session end (bar_persister.py:48-98; called from session_orchestrator.py:3820-3823). Result — operator watching live cockpit sees stale historical bars while session shows "Bars: 9" / heartbeat 37s. DuckDB on Windows takes per-process exclusive file locks (`feedback_duckdb_windows_lock_is_per_process.md`); a second process opening `gold.db` read-only would still race the live writer at session-end flush, so we cannot solve this by writing incrementally to `gold.db`. Solution — mirror the existing `bot_state.json` atomic-file IPC pattern (`bot_state.py:93-127`) for live bars. Session writes `data/live_bars/<SYMBOL>.json` ring-buffered (≤240 bars = ~4h of 1m) atomically; dashboard reads it the same way it reads `bot_state.json`. No DuckDB writes during the session — `flush_to_db()` at session end continues unchanged so the research pipeline still gets clean batched gold.db writes. Closes carry-over (c-i) `dashboard /api/bars-recent empty bars` deferred from `feedback_duckdb_windows_lock_is_per_process.md`.
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bar_ring.py
  - trading_app/live/bar_persister.py
  - trading_app/live/bot_dashboard.py
  - trading_app/live/session_orchestrator.py
  - tests/test_trading_app/test_bar_ring.py
  - tests/test_trading_app/test_bar_persister.py
implementation_status: AUDIT_PENDING
closed_note: |
  Implementation complete (1 commit). data/live_bars/ created via mkdir at
  runtime; project .gitignore already excludes data/ so no per-dir .gitignore
  needed. 23 unit tests pass (12 new in test_bar_ring, 2 new in test_bar_persister).
  Drift: 158 PASSED, 0 violations. Adversarial-audit gate (evidence-auditor)
  REQUIRED before CLOSED — touches trading_app/live/, severity HIGH.
  Manual smoke (done-criterion 7) requires live session — operator-driven.
---

## Blast Radius

- **`trading_app/live/bar_ring.py`** — NEW (~180 lines). Public: `write_bar_ring(symbol, bars, session_id) -> WriteResult`, `read_bar_ring(symbol) -> RingSnapshot`, `clear_ring(symbol)`, `is_stale(snapshot, max_age_s=90.0) -> bool`. Private: background-thread writer with bounded queue (capacity 480, drops oldest on overflow with WARNING — never blocks trading), per-symbol PID metadata, mock-contamination guard mirroring `bot_state._sanitize_for_state`. Atomic `.tmp + os.replace` write idiom copied verbatim from `bot_state.py:93-117`. Fail-open on disk errors per `institutional-rigor.md` § 6; consecutive-failure counter on `WriteResult` so caller can surface staleness to operator.
- **`trading_app/live/bar_persister.py`** — `append()` (line 39) gains `enqueue_to_ring=True` default; pushes bar onto module-level ring writer's queue. `is_valid()` filter applied at enqueue (rejects same bars `flush_to_db()` rejects at line 80, no new corruption surface). Adds `clear_ring()` method that delegates to `bar_ring.clear_ring(self.symbol)`. `flush_to_db()` unchanged — gold.db batch-write semantics preserved. Trading path adds one `queue.put_nowait` call (~microsecond); ring writer thread does the JSON serialize + disk write off-path.
- **`trading_app/live/session_orchestrator.py`** — TWO lines added. (1) On shutdown path, next to existing `bot_state.clear_state()` call (or end-of-session block at line 3820-3823), call `self._bar_persister.clear_ring()` AFTER `flush_to_db()` succeeds. Reads: none. Writes: deletes `data/live_bars/<SYMBOL>.json`. No trading behavior change.
- **`trading_app/live/bot_dashboard.py`** — `_bars_watcher` (line 2734): replace `duckdb.connect(GOLD_DB_PATH, read_only=True)` block (lines 2762-2806) with `bar_ring.read_bar_ring(inst)`. Add `is_stale` check + cross-reference with `bot_state.heartbeat_utc` — if both stale, log INFO once-per-instrument and skip SSE pushes (prevents yesterday's ring showing as "live"). `last_seen` bootstrap semantics preserved verbatim (lines 2768-2781) — first tick records max ts, no backfill SSE. `_query_bars_recent` (line 2879): read ring first; if requested `lookback_minutes` exceeds ring's oldest bar timestamp, ALSO query gold.db for `ts_utc < ring_oldest AND ts_utc > now() - lookback`; merge + dedup on `ts_utc` (overlap expected at session-end flush boundary). Empty ring AND no active session → gold.db only (historical chart for stopped bot, unchanged from today).
- **`tests/test_trading_app/test_bar_ring.py`** — NEW. 12 tests covering: (1) round-trip identity, (2) ring cap at 240 drops oldest, (3) invalid bars rejected with counter bump, (4) concurrent reader/writer torn-read safety (1000 reads × 100 writes, zero JSONDecodeError), (5) stale detection, (6) PID-mismatch logging, (7) corrupt-file fail-closed empty return, (8) `clear_ring` deletes + subsequent read empty, (9) bounded-queue overflow drops oldest + WARNING, (10) Mock-contamination refusal mirroring `test_bot_state_strict_types.py`, (11) consecutive-failure counter CRITICAL at ≥3, (12) gold.db + ring merge dedup.
- **`tests/test_trading_app/test_bar_persister.py`** — EXTEND. Add 2 tests: (a) `append()` enqueues to ring; (b) `clear_ring()` removes ring file. Existing batch-flush tests unchanged.
- **`data/live_bars/.gitignore`** — NEW. Contents: `*\n!.gitignore\n` (ring files are runtime artifacts).
- **Reads:** `data/live_bars/<SYMBOL>.json` (new), `data/bot_state.json` (existing), `gold.db` (existing, read-only path unchanged). **Writes:** `data/live_bars/<SYMBOL>.json` from session process only. **No DuckDB schema change. No `live_journal.db` touch. No gold.db write path change. No trading logic change.**
- **Concurrent-write safety:** only one `BarPersister` exists per symbol per process (constructed at `session_orchestrator.py:727`); session singleton already enforced by `instance_lock.py`. Two readers (dashboard duplicates seen this session) → safe by atomic-replace.
- **Tests blast radius:** companion test files listed in scope_lock. No other test file edits required.

## Non-goals (deferred)

- Real-time tick streaming (sub-second) — out of scope; bars close at minute boundaries by `BarAggregator` and this stage matches that cadence.
- Backfill ring on dashboard restart mid-session beyond what `/api/bars-recent` already does on browser connect — bootstrap semantics preserved verbatim.
- Multi-process ring writers for the same symbol — forbidden by existing session singleton; out of scope.
- Promoting `_iso_utc` from `bot_state.py` to a public name — separate debt item per `bot_state.py:213-219`.
- Web-socket alternative to SSE — SSE plumbing unchanged.

## Edge cases addressed (from v2 audit)

1. **Invalid bars** → filtered at enqueue via `bar.is_valid()`; rejected count exposed on `RingSnapshot.invalid_rejected_count` for operator visibility.
2. **Two persisters same symbol** → ring payload carries writer PID; reader logs WARNING on PID flip. Session singleton remains the real defense.
3. **Stale ring from prior session** → (a) shutdown path calls `clear_ring()` after `flush_to_db()`; (b) `is_stale` + heartbeat cross-check refuses SSE pushes when both stale. Belt + suspenders for crash recovery.
4. **Dashboard restart mid-session** → bootstrap records `last_seen` only; chart fills via `/api/bars-recent` on browser connect (semantics preserved).
5. **Lookback exceeds ring depth** → `_query_bars_recent` merges ring + gold.db with `ts_utc` dedup; no chart discontinuity.
6. **Atomic-replace race on Windows** → consecutive-failure counter; ≥3 → CRITICAL log + operator-visible staleness (mirrors `_BAD_BAR_ALERT_THRESHOLD` in `bar_aggregator.py:57`).
7. **Ring write blocking trading path** → background thread + bounded queue + `put_nowait` on hot path; overflow drops oldest (not newest), trading never waits on disk.
8. **Partial-read corruption** → `read_bar_ring` catches `JSONDecodeError` + `OSError`, returns empty `RingSnapshot`, logs WARNING — never raises to caller.

## Done criteria

1. All 12 new tests in `test_bar_ring.py` pass (show output).
2. All extended `test_bar_persister.py` tests pass (show output).
3. `python pipeline/check_drift.py` passes (show count + violations=0).
4. `grep -r` confirms no dead code: no orphaned imports, no commented-out DuckDB read in `_bars_watcher`.
5. Self-review against `institutional-rigor.md` §§ 1, 4 (canonical sources — uses `pipeline.paths.GOLD_DB_PATH`), 5 (no dead code), 6 (no silent failures — every except logs), 8 (verify before claim).
6. **Adversarial-audit gate** per `.claude/rules/adversarial-audit-gate.md` — this stage touches `trading_app/live/`, severity HIGH (live cockpit operator-facing surface that could mislead during live trading if stale). Dispatch `evidence-auditor` AFTER fix commit, BEFORE marking stage CLOSED. Block next phase until audit returns PASS/CONDITIONAL with no CRITICAL findings.
7. Manual smoke: start signal session against current profile; within 90s, dashboard chart shows a fresh candle for the latest minute; kill session; ring file deleted; chart falls back to gold.db historical view.
8. Companion feedback entry in `memory/` if any new failure-class surfaces during implementation (per n=1 doctrine — `feedback_n3_same_class_doctrine_threshold.md`).

## Execution ordering

1. Write `bar_ring.py` + `test_bar_ring.py`. Verify in isolation. Commit.
2. Modify `bar_persister.py` + extend `test_bar_persister.py`. Verify. Commit.
3. Modify `session_orchestrator.py` shutdown hook. Verify session shutdown clears ring. Commit.
4. Modify `bot_dashboard.py` `_bars_watcher` + `_query_bars_recent`. Manual smoke per done-criterion 7. Commit.
5. Dispatch adversarial-audit gate (`evidence-auditor` subagent). Address any CRITICAL findings. Commit fix(es).
6. Mark stage CLOSED with closed_note summarizing test counts + audit verdict.

## Risk register

- **HIGHEST:** background-thread writer interaction with `BarPersister.flush_to_db()` at session end. Order required: drain queue → flush_to_db (gold.db) → clear_ring. Mitigation: explicit `bar_ring.drain_and_stop()` call before flush in session shutdown sequence.
- **MEDIUM:** Windows `os.replace` on file held by reader. Mitigation: consecutive-failure counter surfaces it; fail-open contract preserved.
- **LOW:** test flakiness from threading/timing in tests 4 and 9. Mitigation: use `queue.Queue.join()` to await drain; avoid `time.sleep` polls.

## Cross-references

- Origin: `feedback_duckdb_windows_lock_is_per_process.md` carry-over (c-i).
- Pattern: `bot_state.py:93-127` atomic-write idiom.
- Adversarial gate: `.claude/rules/adversarial-audit-gate.md`.
- Operator-friendly-error precedent: commit `53d25742` (PR #312) — same rigor expected for stale-ring messaging.
