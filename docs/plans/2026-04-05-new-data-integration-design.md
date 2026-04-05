# New Data Integration Design — Apr 5 2026

## Context

Downloaded 6 schemas from Databento ($14.17 total):
- statistics (OI, settlement, volume) — 16yr FREE
- ohlcv-1s (1-second bars) — 16yr FREE
- ohlcv-1h (hourly bars) — 16yr FREE
- tbbo (trade+quote) — 12mo L1 included
- trades (tick-by-tick) — 12mo L1 included
- mbp-1 (top of book) — 1mo L2 included

Currently NONE of this is ingested. Pipeline runs entirely on ohlcv-1m (bars_1m).

## What daily_features already computes from 1m bars (275 columns)

- 39 volume columns: per-session ORB volume, break bar volume, rel_vol (20-day rolling)
- 48 break columns: direction, timestamp, delay_min, break_bar_continues, pre_velocity
- ~30 regime columns: ATR, GARCH, compression, overnight range, prev_day stats, gap, RSI, VWAP

## What's genuinely NEW in each data type

| Data | New info? | Already have at 1m? | Mechanism for ORB? |
|------|----------|---------------------|-------------------|
| statistics: OI | YES | NO | Positioning proxy. High OI = stops clustered at known levels. |
| statistics: exchange vol | PARTIAL | per-session vol from bars | Total exchange volume, different from session-windowed |
| statistics: settlement | PARTIAL | daily_close from last bar | Settlement != close, but difference small |
| ohlcv-1s | PARTIAL | break_delay_min at 1m | 60x break timing resolution. Refinement. |
| tbbo | YES | NO | Spread at break = book thickness proxy. Fakeout risk? |
| trades | YES | NO | Order flow imbalance. Speculative. |
| mbp-1 | YES | NO | Book depth. Speculative. |

## Critical findings from verification

### MGC OI data requires separate download
- Current MGC statistics download used `GC.FUT` (full-size gold) — contains GC OI, NOT MGC OI
- MGC.FUT has its own OI data (MGCM6 = 465,860 contracts on Apr 1 2026)
- Available from 2018+, FREE
- **ACTION NEEDED: download MGC.FUT statistics separately**

### MES and MNQ have micro-specific OI
- MES.FUT: MESM6 = 1,316,239 OI (front month)
- MNQ.FUT: MNQM6 = 1,979,699 OI (front month)
- Both downloaded correctly with native symbols

### Multiple OI snapshots per day
- Typically 2 per day (~01:40 UTC and ~13:37 UTC)
- Use LAST update per day per contract (end-of-day = standard)

### Front-month selection for OI
- Sum all months = total exchange positioning (use this for "market crowding" signal)
- Front-month only = current trading month (use this for "stop clustering" signal)
- Decision: ingest BOTH (total_oi and front_month_oi) — let the research decide which matters

### Historical symbol mapping for statistics
- MGC 2010-2018: GC.FUT statistics (full-size gold OI as proxy)
- MGC 2018+: MGC.FUT statistics (micro-specific OI)
- MNQ pre-2024: NQ.FUT statistics (already downloaded)
- MES pre-2019: ES.FUT statistics (already downloaded)

## Decision: INGEST vs RAW FILE vs SKIP

| Data | Action | Rationale |
|------|--------|-----------|
| statistics: OI | INGEST | Genuinely new, daily granularity, testable, tiny DB impact |
| statistics: exchange vol | INGEST (with OI) | Near-zero marginal cost |
| statistics: settlement | INGEST (with OI) | Near-zero marginal cost |
| ohlcv-1s | RAW FILE | Only useful after break speed filter deployed |
| tbbo | RAW FILE + RESEARCH | Spread-at-break hypothesis worth testing ad-hoc |
| trades | RAW FILE | No hypothesis formulated |
| mbp-1 | RAW FILE | No hypothesis formulated |
| ohlcv-1h | RAW FILE | Redundant with 1m, 24 MB, not worth deleting or ingesting |

## Literature grounding (verified Apr 5 2026)

### Published evidence:
- **Bessembinder & Seguin (1993), J. Finance** — Higher OI = LOWER volatility in futures (8 contracts, CBOT/CME/COMEX). OI proxies depth/liquidity, dampening moves. WEAKLY CONTRADICTS using high OI as breakout quality predictor.
- **Hong & Yogo (2012), J. Finance** — OI growth predicts returns at MONTHLY frequency (30 commodities, 1965-2008, p<0.05). WRONG TIMESCALE for intraday ORB.
- **Osler (2003), J. Finance** — Stop-loss clustering at round numbers (FX only). WRONG MARKET.
- **Chan (2013), Algorithmic Trading Ch7 p155** — Stop cascading as structural cause of intraday momentum. Mechanism description only, no OI test.
- **Spread at breakout** — ZERO published evidence for futures fakeout prediction.

### Assessment:
ALL four hypotheses (OI level, OI delta, spread at break, order flow) are UNSUPPORTED at ORB timescale. Bessembinder weakly contradicts. No published study tests OI as intraday breakout quality predictor for CME futures. This work is NOVEL — requires in-sample validation from scratch with strict kill criteria.

### Risk:
Bessembinder's finding creates dual-mechanism tension: high OI could mean better stop cascading (higher WR) but also more liquidity (smaller ORBs = higher friction drag = lower ExpR). These could cancel out. Empirical test required before building infrastructure.

## VERDICT: OI PIPELINE BUILD = KILLED (Apr 5 2026)

### Phase 0 results (3 rounds of testing):

**Round 1 (unfiltered, outlier-contaminated):** p=0.31. Initial KILL.
**Round 2 (cleaned, validated strats, cross-asset):** 4/11 BH FDR survivors. Signal found.
**Round 3 (confound-controlled — ATR stratification):** Signal COLLAPSES for 3/4 sessions.

| Session | Raw spread | After ATR control | Direction consistent? | Verdict |
|---------|-----------|-------------------|----------------------|---------|
| EUROPE_FLOW | +3.1% | +4.8% avg | NO (flips in low ATR) | CONFOUNDED |
| COMEX_SETTLE | +2.9% | +0.3% avg | NO (flips in high ATR) | CONFOUNDED |
| NYSE_CLOSE | +7.3% | +9.3% avg | NO (flips in low ATR) | PARTIAL |
| CME_PRECLOSE | -5.8% | -6.5% avg | YES | INDEPENDENT but narrow |

**Root cause:** OI correlates r=0.4-0.6 with ATR, ORB size, overnight range. Existing filters (COST_LT, ORB_G, OVNRNG) already capture the same information. OI is a noisier proxy for volatility/size, not an independent signal.

**One exception:** CME_PRECLOSE MNQ shows independent negative OI effect (high OI = worse, direction-consistent after ATR control). But: 1 session, 1 instrument, 2 years of data = too narrow for infrastructure.

### Decision: NO statistics ingestion. NO daily_features columns. NO filter changes.
- Raw statistics files stay on disk as research archive (zero cost)
- CME_PRECLOSE finding documented for future monitoring
- If MES CME_PRECLOSE shows same pattern with more data, revisit

### What was saved:
- ~1 day of build work (ingestion script, schema change, rebuild, tests)
- Avoided adding a confounded feature to the pipeline
- Explore-first methodology validated: 3 rounds of testing caught what 1 round missed

## ~~Implementation plan~~ (CANCELLED)

### Prerequisites
- Download MGC.FUT statistics (2018-2026, FREE) — IN PROGRESS
- Verify NQ.FUT and ES.FUT statistics cover the pre-native periods — VERIFIED (files exist)

### Step 1: New table `statistics_daily` in init_db.py
Columns:
- trading_day DATE — keyed from `ts_ref.date()` (NOT ts_event — ts_ref is the CME business date the stat refers to)
- symbol VARCHAR — mapped to pipeline symbol (MGC, MNQ, MES)
- total_oi BIGINT (sum of ALL outright contract months, excluding calendar spreads)
- front_month_oi BIGINT (highest-OI single contract month)
- front_month_symbol VARCHAR (which contract, e.g. MESM6)
- exchange_volume BIGINT (sum of all outright contract months)
- settlement_price DOUBLE (front month settlement — LAST update per ts_ref date)
Idempotent: DELETE+INSERT per date range.
Primary key: (trading_day, symbol).

### Step 2: New script `pipeline/ingest_statistics.py`
- Reads statistics .dbn.zst files from data/raw/databento/statistics/{INST}/
- Filters to stat_type 3 (settlement), 6 (OI), 9 (volume)
- **FILTERS OUT calendar spreads** (symbols containing `-`, e.g. GCM0-GCZ1)
- Groups by `ts_ref.date()` (reference date = trading day)
- **Dedup: takes LAST snapshot per ts_ref date** (handles multiple daily updates)
- Aggregates: total OI = sum all outright months; front month = max OI contract
- Settlement: LAST update per ts_ref date for front-month contract
- Maps symbols: GC.FUT -> MGC (pre-2018), MGC.FUT -> MGC (2018+), NQ.FUT -> MNQ (pre-2024), ES.FUT -> MES (pre-2019)
- Writes to statistics_daily table

### Step 3: Enrich daily_features in build_daily_features.py
- Add columns: prev_day_oi, prev_day_front_oi, prev_day_exchange_vol, prev_day_settlement
- Computed as LAG(1) over statistics_daily, joined on trading_day/symbol
- Added to Stage 6 (overnight/pre-session stats section)
- All 4 columns nullable (NULL for dates before statistics coverage)

### Step 4: Update build chain ordering
Current: download -> ingest 1m -> build 5m -> build daily_features -> outcomes
New: download -> ingest 1m -> **ingest statistics** -> build 5m -> build daily_features -> outcomes
Modify: databento_daily.py (add statistics ingestion after download)
Modify: refresh_data.py (add statistics ingestion after 1m ingest, before 5m build)

### Step 5: Research scan
- Add `prev_day_oi / atr_20` and `prev_day_exchange_vol / atr_20` to presession feature scan
- Also test raw `prev_day_oi` and `prev_day_front_oi` (not just normalized)
- Full T1-T8 battery
- Kill criterion: 0 BH FDR survivors at honest K (existing 4 features + new ones)

### Step 6 (separate): Spread-at-break hypothesis
- Standalone research script reads tbbo files from disk
- For each ORB break event in daily_features:
  - Find the BBO at break_ts from tbbo data
  - Compute spread in ticks at break moment
- Test: does spread quintile predict win rate?
- If signal found, THEN consider ingestion
- Not blocked on statistics ingestion — independent research track

## OI look-ahead warning (3 early sessions)

OI for day T is published at ~01:40 UTC on day T+1. Three sessions start BEFORE that:
- TOKYO_OPEN (00:00 UTC) — 1h40m before OI published
- BRISBANE_1025 (00:25 UTC) — 1h15m before OI published
- SINGAPORE_OPEN (01:00 UTC) — 40m before OI published

All other 9 sessions start after 07:00 UTC — SAFE.

**Handling:** In backtesting, prev_day_oi is computed from batch data (no real-time issue). In live trading, these 3 sessions would be using OI that's 2h old at most. The research scan tests per-session — if OI only validates for SAFE sessions, no issue. If it validates for an early session, flag for review and consider LAG(2) for those sessions specifically.

## Kill criteria (defined upfront)

### Statistics OI/volume features:
- KILL if: 0 BH FDR survivors across all session/instrument combos at honest K
- KILL if: WFE < 50% for all survivors
- KILL if: fewer than 3 years of consistent direction
- On KILL: drop statistics_daily table, remove daily_features columns, revert schema

### Spread-at-break hypothesis:
- KILL if: no monotonic WR relationship across spread quintiles
- KILL if: p > 0.05 on WR difference between Q1 and Q5
- On KILL: no schema changes to revert (ad-hoc research only)

## Blast radius

Files to modify:
- pipeline/init_db.py (add statistics_daily table)
- pipeline/build_daily_features.py (add 4 columns, JOIN on statistics_daily)
- scripts/databento_daily.py (add statistics ingestion step BETWEEN download and feature build)
- scripts/tools/refresh_data.py (add statistics ingestion step AFTER 1m ingest, BEFORE 5m build)
- New: pipeline/ingest_statistics.py
- New: tests/test_pipeline/test_ingest_statistics.py (required by TEST_MAP drift check)
- New: scripts/research/spread_at_break.py (ad-hoc, no pipeline change)

Files NOT touched:
- pipeline/ingest_dbn.py (unchanged)
- pipeline/build_bars_5m.py (unchanged)
- trading_app/* (unchanged)
- No orb_outcomes schema change
- No validated_setups change

## Data integrity rules for ingest_statistics.py
1. Filter OUT calendar spreads (symbol contains `-`)
2. Key on `ts_ref.date()` (reference date), NOT `ts_event.date()` (publication date)
3. Dedup: LAST snapshot per ts_ref date (by max ts_event)
4. Front month = contract with highest OI on that ts_ref date
5. Settlement = LAST price update for front-month contract per ts_ref date
6. Volume = sum of all outright months (same dedup: last snapshot per ts_ref)
7. Symbol mapping must handle GC→MGC, ES→MES, NQ→MNQ transitions with explicit date cutoffs
8. NULL is acceptable for dates before statistics coverage (build_daily_features handles NULLs)
9. Spot-check: compare front_month_oi against CME daily reports for 3 random dates post-build

## Rebuild requirement
Full daily_features rebuild after schema change:
- 3 instruments x 3 apertures x 16 years
- Estimated time: ~30-45 minutes
- Must happen once after statistics_daily is populated

## Raw file lifecycle
- Daily downloads: ~60 MB/day across 6 schemas x 3 instruments
- Keep daily files for 90 days, then prune
- Keep backfill archives (12mo L1, 1mo L2) permanently as research cache
- Total steady-state storage: ~5.4 GB/quarter for daily files

## Subscription decision (updated after OI KILL)

### What the pipeline NEEDS daily:
- ohlcv-1m: FREE on any plan (pennies on usage-based)
- statistics: FREE on any plan (not ingested, but useful as archive)

### What's downloaded daily but NOT used:
- ohlcv-1s, tbbo, trades, bbo-1s, mbp-1: downloaded by databento_daily.py
- None of these are ingested or used by any production code
- They accumulate ~60 MB/day on disk

### Decision: STOP daily tick downloads, keep 1m + statistics only
Modify `databento_daily.py` DAILY_SCHEMAS to remove tbbo, trades, bbo-1s, mbp-1.
Keep ohlcv-1s (FREE, useful for break speed research later) and statistics (FREE).

### Subscription: cancel Standard → usage-based AFTER spread-at-break test
- If spread-at-break test finds signal → keep Standard 1 more month for fresh tbbo
- If spread-at-break test is KILLED → cancel immediately, switch to usage-based
- Savings: ~$197/mo = $2,364/yr

### Remaining research before cancellation:
1. Spread-at-break test (tbbo downloading now, ~30 min test once ready)
2. That's it. OI is dead. Trades/mbp-1 have no hypothesis.
