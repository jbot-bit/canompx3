# Data Pipeline Gap Report — Institutional ORB Backtesting Readiness

**Date:** 2026-06-10
**Scope:** MNQ (Micro Nasdaq), MES (Micro S&P), MGC (Micro Gold)
**Method:** Live read against canonical `gold.db` + source-code trace of every ingestion path.
**Verdict:** **ADEQUATE for the ORB strategy this system is actually designed for** (1-minute OHLCV breakout backtesting on front-month outright contracts). **NOT adequate** for any strategy needing sub-minute timing, quoted spread, or microstructure execution modelling. Two real data-integrity defects found in the *live* path (not the backtest path).

> Honesty note: the "institutional requirements" in the brief (millisecond timestamps, bid/ask spread, tick-level volume) are the requirements for a **microstructure / execution-sensitive** strategy. This codebase runs a **1-minute bar ORB breakout** strategy. The right question is not "do we have tick data?" (we deliberately don't) but "is the 1-minute OHLCV data clean, leak-free, and cost-modelled?" Both questions are answered below.

---

## 1. Self-Identification — Every Data Source & Field In Use

### 1.1 Ingestion paths (verified in code)

| Path | Code | Source | Schema | Writes |
|---|---|---|---|---|
| **Backfill (canonical)** | `pipeline/ingest_dbn.py` → `ingest_dbn_mgc.py` | Databento DBN files (`.dbn.zst`), **`ohlcv-1m`** schema, dataset GLBX.MDP3 | `ohlcv-1m` (hard-gated, fail-closed at `ingest_dbn.py:190,206`) | `bars_1m` |
| **Exchange statistics** | `pipeline/ingest_statistics.py` | Databento `statistics` DBN | stat_type codes 1–9 | `exchange_statistics` |
| **Pit-range backfill** | `pipeline/backfill_pit_range_atr.py` | reads `exchange_statistics` | — | `daily_features.pit_range_atr` |
| **Live feed (API)** | `trading_app/live/projectx/data_feed.py` | **ProjectX Market Hub (SignalR)** — `GatewayQuote` / `GatewayTrade` events | tick → `BarAggregator` → 1m bar | `bars_1m` (via `bar_persister.py`) |
| **Derived features** | `pipeline/build_daily_features.py` | reads `bars_1m` | — | `daily_features` (incl. ORB outcomes) |

### 1.2 Fields actually stored — `bars_1m` (the trading data) — verified live

```
ts_utc         TIMESTAMP WITH TIME ZONE
symbol         VARCHAR            (MNQ / MES / MGC — logical instrument)
source_symbol  VARCHAR            (front-month outright, e.g. MNQM6)
open  high  low  close   DOUBLE
volume                  BIGINT
```
**That is the complete field set.** No bid, no ask, no spread, no tick count, no trade-type, no microsecond. This is by design — Databento `ohlcv-1m` is a pre-aggregated 1-minute OHLCV product.

### 1.3 Live coverage (verified `2026-06-10` against `gold.db`, max trading day `2026-06-08`)

| Symbol | bars_1m rows | Date range | Contracts |
|---|---|---|---|
| MES | 2,470,418 | 2019-05-06 → 2026-06-08 | 29 |
| MNQ | 2,467,987 | 2019-05-06 → 2026-06-09 | 30 |
| MGC | 1,375,481 | 2022-06-13 → 2026-06-08 | 21 |

`exchange_statistics` (settlement / session H-L / OI / cleared volume): MES 4,723 / MNQ 4,792 / MGC 4,788 rows, 2010-06-06 → 2026-04-03, near-fully populated.

---

## 2. Data Gap Report — Current Fields vs. Brief's Institutional Requirements

| Requirement | Status | Evidence |
|---|---|---|
| **Timestamp precision (ms/µs, not minute)** | ❌ **MINUTE-ONLY by design** | `bars_1m.ts_utc` is `TIMESTAMPTZ` but every value lands on a minute boundary (`epoch % 60 == 0`, verified). Source product is `ohlcv-1m`. No sub-minute timing exists or can exist without re-buying tick/MBO data. |
| **OHLCV + Bid/Ask spread** | ⚠️ **OHLCV yes, spread NO** | OHLCV fully present & clean. Bid/Ask: the live `GatewayQuote` event *carries* `bestBid`/`bestAsk` (`data_feed.py:8,78`) but the parser **discards them** — only `lastPrice` is kept (`parse_quote`, line 76-82). Backfill DBN `ohlcv-1m` has no quote fields at all. **No spread data is persisted anywhere.** |
| **Tick / actual traded contracts (not tick count)** | ⚠️ **MIXED — clean in backfill, BROKEN in live** | Backfill `volume` = real exchange cleared contracts per minute (median ~821 MNQ, max ~27k — plausible). **Live path volume is corrupt** (see §3, Defect A): live-written `src=MNQ` rows show per-bar volume up to **1.6 billion** because the aggregator sums *cumulative* session volume. |
| **Timezone metadata + DST** | ✅ **CORRECT & rigorous** | All timestamps stored `TIMESTAMPTZ` in UTC. DST is fully event-based via `pipeline/dst.py` using stdlib `zoneinfo` (`America/New_York`, `America/Chicago`, `Europe/London`, `Australia/Brisbane`). Sessions resolve per-day; `us_dst`/`uk_dst` flags stored in `daily_features`. This is the **strongest** part of the pipeline. |
| **Delayed / aggregated / no-bid-ask source flag** | ⚠️ **AGGREGATED (intentionally); see flags below** | — |

### 2.1 Source-quality flags (the brief's explicit ask)

- ✅ **Not Yahoo/free-crypto-grade.** Backfill is **Databento GLBX.MDP3** (CME's official MDP3 feed) — institutional-grade, point-in-time, survivorship-correct (dead contracts preserved as `source_symbol`). Live is **ProjectX/Topstep broker SignalR** — real-time, not delayed.
- ⚠️ **AGGREGATED:** all backtest data is **1-minute pre-aggregated** (`ohlcv-1m`). No tick, no MBO, no MBP/L2. Correct for a 1m-bar ORB strategy; insufficient for any sub-minute or order-book strategy.
- ⚠️ **NO QUOTED SPREAD anywhere** — slippage/spread is modelled, not measured (see §4).
- ⚠️ **Live volume unreliable** (Defect A) — do not use live-captured bars for any volume-dependent feature (`rel_vol_*`, VOL filters) without the fix.

---

## 3. Data-Integrity Defects Found (live path only — backtest data is clean)

### Defect A — Live bar volume is structurally corrupt (HIGH)
- **Where:** `trading_app/live/projectx/data_feed.py:81` (`vol = quote.get("volume", 1)`) + `bar_aggregator.py:100` (`self._current.volume += volume`).
- **What:** `GatewayQuote.volume` is the contract's **cumulative session volume**, but the aggregator treats it as a per-tick delta and sums it. Result: live-written rows (`source_symbol='MNQ'`, 1,037 rows from 2026-04-24) carry per-minute volume up to **1,632,277,639** vs. a real per-minute max of ~27k from clean DBN backfill.
- **Impact:** Any volume feature computed on live-captured days is garbage (`rel_vol_*`, VOL/ATR-VOL filters). Price OHLC is unaffected. **Does not affect the historical research corpus** (DBN-sourced), only days captured live since 2026-04-24.
- **Note:** this should be reconciled against the live-feed `GatewayTrade` path, which *does* carry per-trade volume and would be the correct source.

### Defect B — Live feed discards bid/ask (MEDIUM, design gap)
- **Where:** `data_feed.py:70-82` — `bestBid`/`bestAsk` arrive on every `GatewayQuote` and are used only as a price fallback, then dropped.
- **Impact:** We *receive* live quoted spread for free and persist none of it. This is the single cheapest path to real measured spread (would let §4's cost model be validated against reality). Currently a missed-capture, not a corruption.

---

## 4. Mitigant — Why "no spread/tick" is survivable for THIS strategy

The absence of bid/ask is partially offset by an explicit cost model (`pipeline/cost_model.py`, `COST_SPECS`) applied to every backtest outcome — commission + slippage are *modelled per instrument*, not ignored. This is the institutionally-accepted substitute when quoted spread is unavailable, **provided the model is conservative and validated**. It is currently modelled but **never validated against measured spread** — Defect B is the gap that would close that loop.

---

## 5. Bottom Line

| Use case | Verdict |
|---|---|
| 1-minute ORB breakout backtest (the actual system) | ✅ **Data is adequate, leak-aware, cost-modelled, DST-correct.** |
| Volume/relative-volume features on **historical** data | ✅ Clean (DBN cleared volume). |
| Volume features on **live-captured** days (post 2026-04-24) | ❌ Blocked by Defect A. |
| Sub-minute timing, scalping, queue position | ❌ Impossible without re-purchasing tick/MBO data. |
| Realistic spread/slippage *measurement* | ❌ Modelled only; Defect B is the cheap fix. |
| Microstructure / order-book strategies | ❌ Out of scope; no L2 data. |

**Recommended next actions (in priority order):**
1. **Fix Defect A** — switch live volume to the `GatewayTrade` delta or diff cumulative `GatewayQuote.volume`. Backfill-correct the 1,037 contaminated live rows or re-pull those days from Databento. *(Capital-path / live code — requires stage-gate + adversarial audit per `institutional-rigor.md`.)*
2. **Close Defect B** — persist `bestBid`/`bestAsk` from the live feed into a new `quotes` table to validate the cost model against measured spread.
3. Leave the backtest pipeline as-is — it is sound for its design.

*All figures verified live against `C:\Users\joshd\canompx3\gold.db` on 2026-06-10. No code was modified by this audit.*
