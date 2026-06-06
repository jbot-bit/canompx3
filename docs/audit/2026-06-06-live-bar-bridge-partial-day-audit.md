# Live-Bar Bridge × daily_features — Partial-Day Integrity Audit (2026-06-06)

> **⚠ CORRECTION (2026-06-06, second investigation pass — see § "Completeness-signal correction" at bottom).**
> The originally-mandated completeness signal (`MAX(ts_utc) >= last-session ORB window end`)
> is **FALSIFIED against real DB data**. It wrongly excludes legitimate CME half-days AND
> cannot distinguish a live-partial day from a quiet low-volume complete day (they are
> bar-for-bar identical). Sites 1 & 2 as specified are unsound. Read the correction
> section before implementing anything. The Site 3 observability fix (CRITICAL log) is
> unaffected and still valid.

**Status:** ROOT CAUSE CONFIRMED; **completeness-signal design REOPENED** (mandated signal falsified). Fixes NOT yet implemented (awaiting operator design decision — Tier B: `pipeline/` + `trading_app/live/`).
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

---

## Completeness-signal correction (2026-06-06, second investigation pass)

The first audit proposed Sites 1 & 2 guard on a **bar-derived completeness signal**
(`MAX(ts_utc) >= last ORB session window end`). Re-investigation against the full
`gold.db` (read-only) **falsifies every bar-derived signal**. Evidence below.

### Finding 1 — last-session-window-END anchor wrongly excludes real CME half-days
The audit claimed MES 2025-11-28 (510 bars) "reaches 04:14 next-morning > NYSE_CLOSE
window → correctly 3 rows." That conflated **Brisbane-displayed** 04:14 with **UTC**.
In UTC the half-day's last bar is **18:14 UTC**, below NYSE_CLOSE's 30m window-end of
**21:30 UTC**. The guard would compute COMPLETE=False on a day that legitimately has
3 aperture rows. Same for MES 2025-12-24 and 2024-11-29 (all max at 18:14 UTC).
DST flips the latest-ending session (NYSE_CLOSE in winter @21:30 UTC, CME_REOPEN in
summer @22:30 UTC), so even a dynamic max-over-sessions anchor fails the same way.

### Finding 2 — last-session-START anchor also fails (both directions)
Lowering the anchor to the latest session's window *start* still excludes the winter
half-days (max 18:14 UTC < NYSE_CLOSE start 21:00 UTC) **and** falsely includes a
live-partial (MGC 2026-04-19 max 22:59 UTC ≥ CME_REOPEN start 22:00 UTC). No single
timestamp anchor separates the two classes.

### Finding 3 (DECISIVE) — partial days are bar-for-bar identical to quiet complete days
| sym | trading_day | bars | span_h | aperture_rows | true class |
|---|---|---:|---:|---:|---|
| MES | 2026-06-05 | 60 | 1.0 | 0 | **LIVE-PARTIAL (bug)** |
| MES | 2026-06-03 | 60 | 1.0 | **3** | COMPLETE (quiet day, valid) |
| MES | 2026-05-31 | 60 | 1.0 | 3 | COMPLETE (valid) |
| MGC | 2026-05-25 | 40 | 0.7 | 3 | COMPLETE (valid) |
| MNQ | 2026-05-27 | 1 | 0.0 | 3 | COMPLETE (valid) |

A live-partial (60 bars / 1.0h) is **indistinguishable** from a legitimate quiet
low-volume session (60 bars / 1.0h, full 3 aperture rows). Span, bar-count, and
timestamp-position ALL fail. There exist 1-bar complete days and 60-bar complete days.
**Completeness is NOT a property of the bars** — it is a property of (a) whether the
trading day is over in wall-clock time, and (b) whether the authoritative full-day
source (Databento) has ingested it. The live bridge writes the *same* `source_symbol`
contract code as Databento, so there is no provenance discriminator in `bars_1m` today.

### Finding 4 — there is a SECOND, separate gap masquerading as the same bug
Full-span days with **0 aperture rows** exist: MES/MGC 2026-05-26 (1320 bars, 23h, 0 aps),
MES 2026-05-14, MGC 2026-04-20/21/22. These have complete bar data but daily_features
was simply never built for them (backfill didn't run). This is an **UNBUILT** gap, not
a partial-write gap — a completeness guard would not fix it (and must not skip them).

### Why the loop is currently CLEARED
Peers DELETEd the partial `daily_features` rows for 2026-06-05, so drift Check 77 now
passes (verified: full `check_drift.py` exit 0). The immediate block is gone; what
remains is preventing recurrence — which the falsified design would not achieve and
could regress (excluding half-days).

### Recommended corrected design (root-cause, source-of-truth chain)
The sound fix is **provenance + wall-clock**, not bar heuristics:

1. **Site 1 root cause — `is_up_to_date` must not let an in-progress / live day count
   as "ingested."** The correct signal is temporal: a trading day is a valid backfill
   target only if it is **fully in the past** (its trading-day UTC window has ended) AND
   Databento ingest has run for it. Live bars for *today/in-progress* must never satisfy
   "up to date" for a *completed* target day. Option A (lightest): backfill target stays
   `today-1`, but `is_up_to_date` checks that the *target day specifically* has bars
   reaching its own trading-day window end (`compute_trading_day_utc_range(target)[1]`),
   not global MAX(ts_utc). A live-partial on day D does not advance day (D-1)'s coverage,
   so it can't suppress (D-1)'s ingest. This sidesteps the half-day problem because the
   target is always a *past complete* day, and half-days' bars DO reach their own
   trading-day window end (they just close the auction early; CME data runs to ~08:00
   Brisbane next morning regardless).
2. **Site 2 — `get_trading_days_in_range` / build should EXCLUDE the current in-progress
   trading day** (the only day a live-partial can poison), not guard on bar-completeness.
   Equivalent: only build days whose trading-day window has fully elapsed in wall-clock.
3. **Provenance hardening (deferred, Tier B schema)** — add a feed-source tag to `bars_1m`
   so live-bridge bars are distinguishable from Databento. This is the durable fix for
   Finding 3 and Gap F (source-aware DELETE) but requires a schema migration.
4. **Site 3 (unchanged, still valid)** — escalate `flush_to_db` except to CRITICAL +
   record exc class/message. Pure observability, no completeness dependency.
5. **UNBUILT gap (Finding 4) — separate ticket** — re-run build for the full-span/0-aps
   active-instrument days; do NOT fold into the partial-day fix.

**Verification required before implementing #1/#2:** prove the "target day reaches its own
trading-day window end" signal classifies all known half-days as COMPLETE and the live
partials as the in-progress day (excluded), using the audit table dates above.

### Finding 5 — the recommended "own-trading-day-window-end" signal ALSO fails (self-corrected)
Tested before recommending: a day is complete iff `MAX(ts_utc) >= compute_trading_day_utc_range(td)[1]`.
FAILS — marks normal complete days False (MES 2025-11-26 stops 21:59 UTC, window-end 23:00 UTC;
nothing trades the final pre-reopen hour) and half-days 4.8h short. Trading-day-boundary mapping
compounds it: a morning-only live-partial's `max_ts` lands on the *previous* UTC calendar date,
so naive UTC `>=` comparisons are meaningless without boundary-aware handling.

**Net: 5 distinct bar-timestamp signals tested (window-end, window-start, own-window-end, span,
bar-count) — ALL fail on half-days or quiet days. Completeness is NOT inferable from `bars_1m`.**

### Sound options (only two survive)
- **Option W (wall-clock, lightest, no schema change):** never build or count-as-ingested the
  **current in-progress trading day** — the only day a live-partial can poison. A day is buildable
  iff `now()` is past that trading day's window end (`compute_trading_day_utc_range(td)[1]`).
  Past days are always built from whatever bars exist (Databento full-day by then); the live
  bridge only ever writes the in-progress day, which is excluded until tomorrow's backfill runs
  Databento for it. Half-days/quiet days are unaffected (they're past days → built normally).
  Does NOT fix Finding 4 (UNBUILT past days) — separate ticket.
- **Option P (provenance, durable, Tier B schema):** add a `feed_source` column to `bars_1m`
  ('databento' | 'live_bridge'); `is_up_to_date` requires a `databento` row for the day. Fixes
  Findings 3 & F properly. Heavier: schema migration + backfill the column + update both writers.

**Recommendation:** Option W now (breaks the loop, zero schema risk, half-day-safe by construction),
Option P later as the durable provenance fix. Site 3 (CRITICAL log) lands regardless.

---

## Finding 4 root-cause investigation (2026-06-06, operator-ordered — NO remediation yet)

Operator directive: Finding 4 (full bars present, daily_features absent) is a **separate
integrity defect**. Investigate the source-of-truth chain; do not patch or backfill.

### Evidence
1. **bars_1m present, daily_features absent — confirmed** for all sampled days:
   MES 2026-05-26 (1320 bars, 0 aps), MES 2026-05-14 (1320, 0), MGC 2026-04-20/21/22 (1380, 0).
2. **Per-(instrument,date), NOT global.** On 2026-05-14 MES=0 aps but MNQ & MGC = 3 aps same date.
   On 2026-04-20 MGC=0 aps but MES = 3 aps same date. Rules out holiday/calendar exclusion.
3. **Isolated INTERIOR gaps, not trailing.** MES timeline: 05-13 built → **05-14 unbuilt** → 05-15
   built; 05-25 built → **05-26 unbuilt** → 05-27 built. A contiguous trailing gap would implicate
   the incremental `start = last_ingested.date()+1` jump; an *isolated interior* hole does not.
   **Refutes the "incremental date-range skip" hypothesis.**
4. **Build path has no per-day skip.** `build_daily_features.py:1373-1390` iterates `trading_days`
   and `rows.append(build_features_for_day(...))` unconditionally. A day absent from output means
   it was absent from `trading_days` for that run — i.e. **never covered by a build's --start/--end
   range** at the time it would have been built (e.g., an interrupted/narrow incremental run, or a
   run whose range ended before that day's data landed).
5. **Shape anomaly noted:** MES 2026-05-26 bars start at 10:00 (not 09:00) — first ORB hour missing.
   Not proven causal; flagged for the remediation pass.

### Verdict
- **Defect class:** historical coverage gap (days that fell between build invocations), distinct
  from the live-partial loop. **Not reproducible from current state** — the exact skipping build
  invocation is not recoverable from the DB alone (no build-run audit log).
- **Can it still create NEW missing days?** Yes in principle — any narrow/interrupted incremental
  build whose range doesn't include a day with late-landing bars leaves an interior hole, and
  nothing re-detects it. There is **no drift check for "bars present but daily_features absent"**
  on active instruments (Check 77 only catches the 1-of-3 partial, not 0-of-3).

### Recommendation (NOT yet implemented, per directive)
1. **Root-cause guard:** add a drift/integrity check `active-instrument days with bars_1m but 0
   daily_features rows` → fails closed. This *detects* the gap class going forward regardless of
   how a build skipped it (defends the generator, not just the symptom).
2. **One-time backfill:** re-run `build_daily_features` for the exact gap dates per instrument
   (enumerated via the check query below), then verify zero remaining.
3. **Verification query** (proves zero active-instrument bars-without-features days):
   ```sql
   WITH bars AS (SELECT symbol, CAST((ts_utc AT TIME ZONE 'Australia/Brisbane' - INTERVAL '9 hours') AS DATE) td, COUNT(*) n
                 FROM bars_1m WHERE symbol IN ('MES','MNQ','MGC') GROUP BY 1,2),
        feat AS (SELECT symbol, trading_day td FROM daily_features WHERE symbol IN ('MES','MNQ','MGC') GROUP BY 1,2)
   SELECT b.symbol, b.td, b.n FROM bars b LEFT JOIN feat f ON b.symbol=f.symbol AND b.td=f.td
   WHERE f.td IS NULL AND b.n >= 200 ORDER BY b.td;   -- expect 0 rows after backfill
   ```
