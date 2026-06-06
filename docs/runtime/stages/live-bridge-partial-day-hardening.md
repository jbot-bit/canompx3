---
task: Harden the live-bar bridge × daily_features chain against partial trailing-day artifacts (3-gap fix, grounded 2026-06-06 audit)
mode: IMPLEMENTATION
scope_lock:
  - pipeline/daily_backfill.py
  - pipeline/build_daily_features.py
  - trading_app/live/bar_persister.py
  - tests/test_pipeline/test_daily_backfill.py
  - tests/test_pipeline/test_build_daily_features.py
  - tests/test_trading_app/test_bar_persister.py
---

## Blast Radius

Root cause (audited, MEASURED): live bridge writes partial trailing-day bars →
`is_up_to_date` (max-ts only) skips Databento full ingest → `get_trading_days_in_range`
builds a 1-of-3-aperture day → drift Check 77 hard-blocks every commit/refresh/preflight.
Full audit: `docs/audit/2026-06-06-live-bar-bridge-partial-day-audit.md`.

**Canonical completeness signal (REAL-DATA VERIFIED, no magic number):** a
`(trading_day, symbol)` is COMPLETE iff `MAX(ts_utc) >=` the latest ORB session's
window end for that day (`orb_utc_window(td, LAST_SESSION, 30)`). Verified:
half-days (MES 2025-11-28 = 510 bars) reach 04:14 next-morning > NYSE_CLOSE window →
correctly 3 rows; partial day (MGC 2026-06-05 = 60 bars) stops 09:59 ~21h short → 1 row.
A bar-COUNT threshold would WRONGLY exclude the 510-bar half-day — rejected.

### Site 1 — `pipeline/daily_backfill.py::is_up_to_date` (:37) + `run_backfill_for_instrument` (:104)
- Today: returns True if `MAX(ts_utc) FROM bars_1m >= as_of` (:27) — max-ts only.
- Change: require the `as_of` day to be COMPLETE (bars reach the day's last-session
  window end via canonical `pipeline.dst`), not just max-ts present.
- Callers: `run_backfill_for_instrument` (:113 short-circuit); `session_orchestrator.py:4312`
  (nightly trigger in post_session). Effect: backfill runs MORE often on live-bridge days
  (correct — that's the bug). Downstream: drives ingest_dbn→build_bars_5m→build_daily_features→outcome_builder subprocess chain.
- Canonical dep: must ADD import of `compute_trading_day_utc_range`/`orb_utc_window` (dst.py). Never re-encode the offset.
- Tests: `test_daily_backfill.py` — 2 existing tests (:47/:57) enshrine OLD max-ts logic → REWRITE for complete-day semantics + NEW partial-day-present→False case.

### Site 2 — `pipeline/build_daily_features.py::get_trading_days_in_range` (:155)
- Today: selects ANY trading_day with ANY bar.
- Change: add complete-day guard (same canonical signal as Site 1) → a day whose bars
  don't reach the last-session window end is SKIPPED, not built into a 1-row partial.
  Build stays all-3-or-nothing per (day, symbol).
- Callers: ONE — internal `build_daily_features.py:1307`. No external caller. ~40 importers
  of the module import OTHER symbols, not this function.
- Canonical dep: `compute_trading_day_utc_range` already imported (:49, used :186/:1320). No new import.
- Downstream: fewer partial days → Check 77 (`check_drift.py:4776`) IMPROVES. orb_outcomes
  gets no rows for skipped days (correct — they're incomplete). LAG features tolerate gaps already (live-bridge boundary).
- Tests: NO direct test today → ADD: partial-day excluded, complete-day (incl. half-day) included.
- **Complements, does NOT duplicate** `verify_daily_features` (:1790) which checks WRITTEN-row
  integrity (dupes/bar_count/enums), not day SELECTION.

### Site 3 — `trading_app/live/bar_persister.py::flush_to_db` except (:124)
- Today: `except (duckdb.Error, OSError): return 0` logged at ERROR; exception detail lost.
- Change: escalate to CRITICAL + record exc class/message in shutdown trace. RETURN VALUE
  UNCHANGED (still 0) — purely observability.
- Callers: `session_orchestrator.py:4269` (outer except already CRITICAL — additive);
  `recover_ring.py:133` (handles return-0 already). No behavior change.
- Tests: `test_bar_persister.py` — return==0-on-error tests stay valid; a log-LEVEL assertion may need updating.

### HIGHEST RISK (from blast-radius) — RESOLVED
Holiday/early-close half-days must NOT be excluded. A fixed-24h or bar-count guard would
permanently drop CME half-days. RESOLVED by using the canonical last-session-window-end
signal: real-data check confirms half-days reach past that window and stay included.
Regression test MUST assert a known half-day (MES 2025-11-28, 510 bars) is INCLUDED.

### No new drift; canonical sources untouched (read-only): SESSION_CATALOG, ACTIVE_ORB_INSTRUMENTS, GOLD_DB_PATH.
### DB: reads bars_1m (guards), writes daily_features (Site 2 output, existing DELETE+INSERT idempotent — unchanged).

## Verification
- TDD per fix (RED→GREEN): write failing test first, then guard.
- Half-day inclusion regression test (MES 2025-11-28) — the key correctness gate.
- `python pipeline/check_drift.py` no regression (Check 77 stays clean).
- Independent adversarial audit (capital/data path) before commit.
- One fix at a time; verify each before next. No live arm.
