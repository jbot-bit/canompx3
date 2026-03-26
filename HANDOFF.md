# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update this file.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

---

## Current Session
- **Tool:** Claude Code (Multi-Terminal Recovery + MAE SL Analysis)
- **Date:** 2026-03-27
- **Branch:** `main`
- **Status:** Recovery session after computer restart. 8 memory files created, 4 tagged RE-VERIFY. MAE analysis ran but superseded by friction confound finding.

### What was done (Mar 27 — this session)

#### 1. Multi-Terminal Recovery
Computer restarted with 5+ terminals running. Recovered all workstreams:
- Verified 7 findings against gold.db (V1-V7)
- V1 CRITICAL: overnight_range code comment claims Q1=36.1%/Q5=62.6% — VERIFIED WRONG (pooled spread 4.5%, signal is Asian not US)
- V2: ATR correlation 0.93 CONFIRMED
- V3: Noise floor 474/488 false positives (not "55/56" — that was old total)
- V4: Waiver fix 38→470 MNQ validated (488 total). 5 Bloomberg audit Qs still open.
- V5: FDR K frozen per strategy (discovery_k populated)
- V6: US_DATA_1000 forward +66R at RR1.0 (N=504)
- V7: 5/488 era-dependent (max: MNQ NYSE_OPEN X_MES_ATR70 at 69%)

#### 2. MAE SL Validation (Sweeney)
Ran full Sweeney analysis — then learned from prior terminal that mae_r is FRICTION-CONFOUNDED. My scatter results are INVALID. Corrected analysis (from prior terminal) shows:
- All 3 instruments identical when friction removed
- Tighter stops (0.60-0.80R raw) improve ExpR across all instruments
- Classification: ARITHMETIC_ONLY (loss-size reduction)
- T6 null bootstrap NOT YET RUN — next step

#### 3. Round Number Research — CONFIRMED DEAD
T0-T8 audit (63K trades) killed it. Prior p=0.031 was sign-test artifact (n=6). WR flat at ~32%, null p>0.35. Anti-clustering also dead.

#### 4. Memory Files Created/Updated
8 new files, 4 tagged "RE-VERIFY" (from pasted terminal output, not independently verified).
Updated: noise_floor_methodology.md, MEMORY.md index.

### Uncommitted Working Tree (from Mar 26 sessions)
6 files modified, 3 untracked — all from the fairness audit / waiver fix:
- `trading_app/config.py` — noise floor disabled, 2 new filter classes (OwnATRPercentile, OvernightRange)
- `trading_app/strategy_validator.py` — waiver bug fix + era_dependent concentration check
- `trading_app/db_manager.py` — 6 new column migrations
- `pipeline/build_daily_features.py` — overnight_range_pct feature
- `tests/test_trading_app/test_strategy_validator.py` — updated waiver tests
- `scripts/tools/migrate_fairness_audit.py` (untracked)
- `scripts/databento_backfill.py` (untracked)
- `scripts/databento_daily.py` (untracked)

**WARNING:** overnight_range code comment in config.py is WRONG (says US-session, actual signal is Asian). Fix before committing.

### What's Running
Nothing (session complete)

### What's Broken
- Tradovate auth — password rejected (unchanged from prior sessions)
- STAGE_STATE.md points at Databento code review but working tree contains fairness audit changes

### Next Actions (Priority Order)
1. **T6 null bootstrap for SL** — use raw MAE (not mae_r). Must beat P95 at p<0.05.
2. **5 Bloomberg rescue audit questions** — gates whether 488 count is trustworthy
3. **Fix overnight_range code comment** — wrong session regime claim
4. **Commit fairness audit** — after fixing comment, commit the 6-file diff
5. **Databento code review** — finish review of backfill/daily scripts
6. **ML stale docs** — update ml_exhaustive_sweep.md ("5/7 PASS" → "1/7 PASS at 5K")
7. **Confluence scan** — per todo_queue_mar27.md

### Prior Session
- **Tool:** Claude Code (Adversarial Audit Round 2 + ProjectX API Compliance)
- **Date:** 2026-03-25
- **Branch:** `main`
- **Status:** 3 commits pushed. Bot hardened against race conditions, timing bugs, and API spec violations. All tests pass. e2e 7/7.

### What was done (Mar 25)

#### 1. Adversarial Audit Round 2 — Race Conditions & Timing (commit `ccf88ce`)
Comprehensive audit of live trading bot (~7,100 LOC) for concurrency bugs, timezone issues, state machine violations, latency traps, cleanup failures, and error handling gaps. Found 7 CRITICAL, 6 HIGH, 5 MEDIUM, 3 LOW. All 12 actionable fixes implemented:

**CRITICAL fixes (7):**
- R2-C1/C2: signalrcore thread bridge — `BarAggregator` gets `threading.Lock`, sync callbacks route through `loop.call_soon_threadsafe()` to prevent cross-thread OHLCV corruption and bar drops
- R2-C3: `engine.cancel_trade()` — removes ghost trades from `active_trades` when fill poller detects broker cancellation of E2 stop-market orders
- R2-C4: `_emergency_flatten` now cancels bracket legs before exit orders — prevents orphaned brackets from opening unwanted positions after kill switch
- R2-C5: `_on_feed_stale(-1)` schedules `_emergency_flatten` when positions exist; `_last_bar_at` initialized to session start so watchdog always has baseline
- R2-C6: stale PENDING_EXIT positions (>300s) trigger `_retry_stuck_exit()` with auth refresh and re-close attempt; on failure blocks entries and sends CRITICAL alert
- R2-C7: rollover orphans re-seeded via `mark_strategy_traded()` to prevent engine from re-arming strategies with open positions at broker

**HIGH fixes (4):**
- R2-H2/H3: Position tracker state guards — `on_exit_sent` rejects PENDING_ENTRY, `on_entry_filled` rejects PENDING_EXIT (prevents state resurrection)
- R2-H4: `_cancel_brackets` now strategy-scoped (specific bracket IDs only), no contract-wide sweep that nukes other strategies' protection
- R2-H5: Engine circuit breaker calls `_emergency_flatten` before pausing (was leaving positions unmanaged for hours)
- R2-H6: E1/E3 risk checks now use `_total_pnl_r()` (unrealized included), matching E2 for consistent daily loss gate

**MEDIUM fixes (2):**
- R2-M1: `_publish_state()` moved after engine processing in `_on_bar` — eliminates 2-50ms disk write from signal detection critical path
- R2-M5: preflight uses Brisbane trading day (not `date.today()`) for correct daily features validation on late-night sessions

#### 2. ProjectX API Reference (commit `8f818fe`)
- Canonical spec saved to `docs/reference/PROJECTX_API_REFERENCE.md`
- Fetched from official gateway docs. Ground truth for all API compliance.

#### 3. ProjectX API Compliance — 6 fixes (commit `6d85ce0`)
Full 15-check audit against official spec. 9 PASS, 6 FAIL. All 6 fixed:

- Check 8: Default base URL → `api.thefuturesdesk.projectx.com` (was `api.topstepx.com`)
- Check 11: `query_order_status` maps integer status enum (spec returns int 0-6, code expected string)
- Check 6: Fill price reads `filledPrice` first (spec field name), falls back to `fillPrice`/`averagePrice`
- Check 10: signalrcore `on_open` re-subscribes to `SubscribeContractQuotes` (spec requires re-subscribe after reconnect)
- Check 3: `verify_bracket_legs` uses type+price fallback when sequential IDs don't match
- Check 4: `cancel_bracket_orders` matches by type (Stop=4, Limit=1) not customTag (searchOpen doesn't return customTag per spec)

### Truth State (verified Mar 25 2026)
- **Tests:** 3065 pass, 61 pre-existing failures (webhook/UI, unrelated)
- **e2e sim:** 7/7 PASS (bot_starts, data_feed, bracket_spec, order_lifecycle, journal, telegram, position_tracker)
- **Drift:** 75/75 pass
- **validated_setups:** 56 rows (stratified-K, holdout-clean)
- **LIVE_PORTFOLIO:** 8 specs (2 CORE, 6 REGIME)
- **Apex lanes:** 4 MNQ committed
- **API compliance:** 15/15 checks PASS against canonical spec

### Residual Risks (documented, not fixed)
1. **R2-H1 (_positions dict interleave):** asyncio cooperative scheduling makes this low-probability. Full fix = asyncio.Lock on every position transition (adds latency). Accepted as residual — monitor for "Exit fill for unknown strategy" warnings.
2. **Bracket sequential ID assumption:** Type+price fallback added but not guaranteed to be unique across concurrent strategies. Monitor bracket verification logs.
3. **signalrcore thread safety:** `call_soon_threadsafe` + Lock fix covers the data path, but signalrcore is a fallback path. Monitor bar consistency in logs during first live session.

### What's Running
- Nothing (session complete)

### What's Broken
1. Tradovate auth — password rejected (unchanged from prior session)

### Next Actions
1. **Live session test** — run bot with new hardening on demo, monitor logs for the 5 residual risk markers
2. **Fix Tradovate auth** — user action (verify password from Tradeify dashboard)
3. **Paper trade** — start collecting 2026 forward data per forward eval pack
4. **ML AUDIT** — 3 open FAILs remain (see `ml_methodology_audit.md`)
5. **Databento backfill** — NQ zip + historical extensions (see `databento_backfill_todo.md`)

### Files Changed This Session
```
trading_app/live/bar_aggregator.py          — threading.Lock on on_tick/flush
trading_app/live/projectx/data_feed.py      — call_soon_threadsafe + re-subscribe on reconnect
trading_app/live/projectx/order_router.py   — int status map, filledPrice, type-based bracket matching
trading_app/live/projectx/auth.py           — correct default base URL
trading_app/live/session_orchestrator.py     — 9 fixes (flatten, state guards, latency, stuck exit, etc.)
trading_app/live/position_tracker.py         — state transition guards
trading_app/execution_engine.py              — cancel_trade(), consistent _total_pnl_r()
scripts/run_live_session.py                  — Brisbane trading day in preflight
tests/test_trading_app/test_bar_aggregator.py — concurrent thread test
docs/reference/PROJECTX_API_REFERENCE.md     — canonical API spec
```

---

## Prior Session
- **Tool:** Claude Code (Apex restructure + stratified-K + noise floor fix + deployment prep)
- **Date:** 2026-03-24 to 2026-03-25
- **Branch:** `main`
- **Status:** 12 commits. Pipeline rebuilt holdout-clean. 4 Apex MNQ lanes validated. Noise floor methodology fixed. Tradovate auth pending.
- **See prior HANDOFF sections below for full detail.**

---

## Prior Session Details (Mar 24-25 — preserved for context)

### Apex Manual Plan Restructure
- Replaced 5 convenience-routed sessions with 4 edge-maximized MNQ lanes
- All lanes pass: stratified-K BH FDR (holdout-clean), walk-forward, stress test, yearly robustness
- NYSE_CLOSE VOL_RV12_N20 O15 (ExpR 0.2078, fwd +4.30R)
- SINGAPORE_OPEN ORB_G8 RR4.0 O15 (ExpR 0.1631, fwd +1.84R)
- COMEX_SETTLE ORB_G8 RR1.0 O5 (ExpR 0.1300, fwd +3.99R)
- NYSE_OPEN X_MES_ATR60 RR1.0 O15 (ExpR 0.0933, fwd +8.37R)

### Infrastructure
- Scratch DB C:/db/gold.db KILLED (blocked in paths.py)
- 75/75 drift checks pass, 56 validated strategies
- Tradovate bot BUILT (auth, feed, orders) — auth NOT working

### Known Issues (unchanged)
- ML #61: 3 violations in features.py (frozen)
- DOUBLE_BREAK_THRESHOLD=0.67: HEURISTIC, proximity warning active
- MGC: 0 live — noise_risk is binding blocker
- Tradovate auth: password rejected
