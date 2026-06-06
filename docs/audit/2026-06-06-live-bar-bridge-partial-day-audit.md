# Live-Bar Bridge × daily_features — Partial-Day Integrity Audit (2026-06-06)

**Status:** ROOT CAUSE CONFIRMED. Fixes NOT yet implemented (awaiting operator GO — Tier B: `pipeline/` + `trading_app/live/`).
**Trigger:** Operator stuck — every `rebuild` / `refresh` / `preflight` / `git commit` failed. Symptom traced to drift Check 77 hard-blocking on a partial `daily_features` day.

---

## TL;DR — the loop that blocked everything

Three gaps **chain together** into a self-perpetuating block:

1. **Live bridge writes partial bars.** When the bot runs (even signal-only), `BarPersister.flush_to_db()` (`trading_app/live/bar_persister.py:76`) writes the broker-feed bars it captured — often just the first hour of a 24h (09:00→09:00) trading day — into `gold.db:bars_1m`. This is the by-design "eliminate Databento dependency" bridge.
2. **Databento full-day ingest is then SKIPPED.** `post_session()` (`session_orchestrator.py:4312`) calls `run_backfill_for_instrument()`, which calls `is_up_to_date()` (`pipeline/daily_backfill.py:37`). That check only reads `MAX(ts_utc) FROM bars_1m` (`:27`) — **max timestamp, not day completeness**. The live partial bar already advanced `MAX(ts_utc)` → the check returns `True` → the full-day Databento ingest never runs.
3. **The build then produces a PARTIAL day.** `get_trading_days_in_range()` (`pipeline/build_daily_features.py:155`) selects **any** day with **any** bar — **no minimum-bar / complete-day guard**. The 5m ORB window completes at ~09:05 (data present) → 5m row written. The 15m/30m windows extend past where bars stop → those apertures produce nothing → **1 aperture row instead of 3**.

**Result:** drift Check 77 (`check_daily_features_row_integrity`, `check_drift.py:4776`) — a **pre-commit BLOCKER** — fires on the 1-of-3 day. Every commit, every refresh that runs drift, every preflight fails on it. The operator could not get out because the artifact poisons the gate, and re-running the build just re-creates the partial day (or skips via `is_up_to_date`).

Confirmed instance: **MGC `trading_day=2026-06-05`** — only `orb_minutes=5` present. MGC `bars_1m` for that day stop at **09:59 Brisbane** (60 bars). MNQ had 0 full-day bars; MES cut at 09:59 too.

---

## Independent audit verdict (live-risk-auditor, 2026-06-06)

**"Is the live bridge fully functioning and professional-grade?" → UNVERIFIED.** Three confirmed gaps:

| # | Severity | File:line | Finding |
|---|----------|-----------|---------|
| A | HIGH | `daily_backfill.py:37,113` | `is_up_to_date` passes on max-ts alone → Databento full-day ingest silently skipped once a live partial bar exists. **Unifying root cause.** |
| B | HIGH | `build_daily_features.py:155` | `get_trading_days_in_range` has NO complete-day guard → builds partial (1-of-3 aperture) days → drift Check 77 blocker. |
| C | HIGH | `bar_persister.py:105-126` + `bar_aggregator.py` | `bars_captured=1 → n_persisted=0`: a captured bar silently dropped (is_valid filter OR fail-open `except (duckdb.Error, OSError): return 0` at `:124`, logged only at ERROR). Ring-preservation recovery exists (`recover_ring.py`) but requires manual operator action. |
| D | MED | `bar_persister.py:124` | Fail-open `except` on a CAPITAL data write logs ERROR not CRITICAL; exception detail lost. |
| E | MED | `daily_backfill.py:111` | EOD backfill targets `today-1`; today's live bars never produce a DB `daily_features` row (in-memory only) — contradicts the "eliminates Databento dependency" docstring. |
| F | LOW | `bar_persister.py:102-103` | DELETE+INSERT over `[ts_min,ts_max]` with no `source_symbol` discriminator can silently substitute broker-feed bars over Databento bars for an overlapping range; provenance lost. |

**Do-not-touch (audit-verified correct):**
- `bar_persister.py:82-83` lock-then-copy (thread-safe).
- `session_orchestrator.py:4279-4297` ring-preserve + CRITICAL log + `recover_ring.py` escape valve (sound fail-open recovery).
- `bar_aggregator.py` `is_valid()` (catches NaN/inf/negative/high<low).
- `post_session()` `finally`-block placement (`run_live_session.py:1459`) — runs after `asyncio.run()`.

**Caveat (not guaranteed):** `post_session()` is in a `finally` block — Python does NOT run `finally` on SIGKILL / `kill -9`. Hard-kill leaves captured bars unflushed (ring file is the safety net).

---

## The bridge IS wired (operator's "didn't we build this?" — YES)

`BarPersister` docstring: *"Eliminates Databento dependency for daily bar data."* Confirmed wired:
- Instantiated with `GOLD_DB_PATH` (`session_orchestrator.py:758`).
- Fed on every bar via `_on_bar` → `append` (`:1911`).
- Flushed at shutdown (`:4265-4296`) via idempotent DELETE+INSERT.
- The live path's `_build_daily_features_row` (`:1146`) builds an **in-memory** row for filter eval only — it does NOT write `daily_features` (the only writer is `build_daily_features.py:1746`).

The wiring is sound; the **completeness gating** around it is the gap.

---

## Prioritized fix plan (root-cause, NOT band-aid) — awaiting GO

1. **[A — root cause] `is_up_to_date` must check day COMPLETENESS, not max-ts.** Require the target day to have a full session's bars (or all-3-aperture coverage) before declaring "up to date" — otherwise a live partial bar can never suppress the Databento full-day ingest. This is the fix that breaks the loop.
2. **[B — the blocker] complete-day guard in `get_trading_days_in_range`.** Do not build a `(trading_day, symbol)` unless all 3 aperture windows have data (or a minimum-bar threshold). Build must be **all-3-or-nothing per (day, symbol)** — never leave a 1-row state. This is what trips Check 77.
3. **[C/D — silent loss] escalate `flush_to_db` except to CRITICAL** + record exception class/message in the shutdown trace; confirm whether the dropped MNQ bar was a genuine in-progress bar or a swallowed write error.
4. **[E — honesty] document** that today's live bars are excluded from the DB `daily_features` by design, OR extend the post-flush trigger to rebuild today after flush.
5. **[F — provenance] source-aware DELETE** so live-bridge bars never silently overwrite Databento bars for an overlapping range.

**Immediate unblock (operator-directed):** investigate → save findings (this file) → re-ingest 06-05 fully (needs `DATABENTO_API_KEY`, **NOT set in current shell** — dry-run confirmed). The re-ingest is blocked on the API key; the partial-day row can alternatively be DELETEd to unblock the gate immediately (MGC not in live MNQ book → zero trading impact).

---

## Reproduction / evidence

```python
# Partial day (live, read-only):
from pipeline.paths import GOLD_DB_PATH; import duckdb
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
con.execute("SELECT orb_minutes FROM daily_features WHERE symbol='MGC' AND trading_day='2026-06-05'").fetchall()
# -> [(5,)]   ← 1 row, expected 3 (5/15/30)
con.execute("""SELECT COUNT(*), MAX(ts_utc AT TIME ZONE 'Australia/Brisbane')
               FROM bars_1m WHERE symbol='MGC'
               AND ts_utc >= TIMESTAMP '2026-06-04 23:00:00+00'
               AND ts_utc <  TIMESTAMP '2026-06-05 23:00:00+00'""").fetchone()
# -> (60, 2026-06-05 09:59)   ← only first hour ingested
```

Shutdown trace (silent flush loss): `data/live_bars/MNQ.shutdown_trace.txt`
```
flush_attempt:bars_captured=1
flush_returned:n_persisted=0
ring_preserved:bars_captured=1,n_persisted=0
```
