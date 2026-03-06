# AI Trading Coach — Design Document

**Date:** 2026-03-06
**Status:** Design approved (4TP flow)
**Problem:** Trading across multiple prop firm accounts (TopstepX, Tradovate/Apex), sometimes via the system, sometimes manually. No unified performance view, no pattern detection, no coaching intelligence. The discipline coach MVP captures self-reported behavioral data but has no analytical layer.

---

## 1. Orient — Current State

### What Exists (Working)

| Component | Status | What It Does |
|-----------|--------|-------------|
| Discipline Coach MVP | DEPLOYED | Post-trade debriefs, cooling periods, pre-session priming |
| Paper Trader | WORKING | Replays from gold.db, generates JournalEntry records (in-memory) |
| Live Trading Infra | WORKING | SessionOrchestrator → ExecutionEngine → BrokerRouter |
| Broker Abstraction | WORKING | 5 ABCs: Auth, Feed, Router, Positions, Contracts |
| TopstepX Integration | WORKING | Auth + order routing + position queries |
| Tradovate Integration | PARTIAL | Auth failing (stale creds), order routing implemented |
| ML Meta-Labeling | PRODUCTION | Per-aperture P(win) classifiers, 12 models |
| Pinecone Knowledge Base | WORKING | 5-tier content sync (static, living, memory, research, generated) |

### What's Missing

1. **No trade history fetch from broker APIs** — system only logs forward, can't see past trades
2. **No round-trip trade reconstruction** — broker APIs return individual fills, not trades
3. **No AI analysis layer** — raw stats only (adherence %), no pattern detection or coaching
4. **No unified multi-account view** — 3+ TopstepX accounts, plus Tradovate
5. **No persistent "model of the trader"** — every interaction starts from scratch

### Key Discovery

**TopstepX `/api/Trade/search` returns complete fill history per account.** Verified working with 3 active accounts (205 + 32 + 39 fills). Each fill includes: timestamp, price, P&L, fees, side, size, contractId, orderId.

This means the AI coach can see ALL trades — both system-executed and manually placed — without any instrumentation in the trading app.

---

## 2. Design — The AI Trading Coach

### Approach: Evolving Trader Profile + Session Digests + Chat

**Core insight:** The broker API IS the source of truth for what happened. The AI coach's job is to interpret WHY and coach WHAT NEXT.

### Architecture

```
┌─────────────────────────────────────────────────┐
│  1. BROKER FILL FETCHER                         │
│     TopstepX /api/Trade/search ──┐              │
│     Tradovate /fill/list ────────┤              │
│     (future brokers) ───────────┘              │
│         ↓                                       │
│  2. TRADE MATCHER                               │
│     Raw fills → round-trip trades               │
│     Pairs BUY+SELL by time/contract             │
│     Calculates: entry, exit, P&L, hold time     │
│     Detects: scalps, reversals, pyramiding      │
│         ↓                                       │
│  data/broker_trades.jsonl  (canonical log)      │
│         ↓                                       │
│  3. SESSION DIGEST ENGINE (Claude API)          │
│     Reads: broker_trades, trader_profile,       │
│            daily_features, trade_debriefs        │
│     Writes: coaching_digests.jsonl              │
│     Updates: trader_profile.json                │
│         ↓                                       │
│  4. INTERACTIVE CHAT                            │
│     Profile always in context (~2-4KB)          │
│     RAG over digests via Pinecone               │
│     "Why do I keep losing on Fridays?"          │
└─────────────────────────────────────────────────┘
```

### Data Model

#### Broker Fills (`data/broker_fills.jsonl`)
Raw audit trail — one record per fill from broker API.
```json
{
  "fill_id": "topstepx-2241049355",
  "broker": "topstepx",
  "account_id": 19858923,
  "account_name": "50KTC-V2-451890-20967121",
  "instrument": "MNQ",
  "contract_id": "CON.F.US.MNQ.H26",
  "timestamp": "2026-03-06T13:43:48.140526+00:00",
  "side": "BUY",
  "size": 4,
  "price": 24740.75,
  "pnl": 556.0,
  "fees": 1.48,
  "order_id": 2587762443
}
```

#### Round-Trip Trades (`data/broker_trades.jsonl`)
Matched trade records — the coach's primary input.
```json
{
  "trade_id": "topstepx-19858923-2026-03-06-001",
  "broker": "topstepx",
  "account_name": "50KTC-V2-451890-20967121",
  "instrument": "MNQ",
  "direction": "LONG",
  "entry_time": "2026-03-06T13:19:47Z",
  "exit_time": "2026-03-06T13:20:59Z",
  "entry_price_avg": 24847.5,
  "exit_price_avg": 24844.5,
  "size": 4,
  "pnl_dollar": -24.0,
  "fees": 2.96,
  "hold_seconds": 72,
  "num_fills": 4,
  "trade_type": "scalp",
  "source": "manual",
  "strategy_id": null
}
```

#### Trader Profile (`data/trader_profile.json`)
Persistent, evolving model of the trader. Git-versioned.
```json
{
  "version": 1,
  "last_updated": "2026-03-06",
  "strengths": [],
  "growth_edges": [],
  "behavioral_patterns": [],
  "goals": [],
  "session_tendencies": {},
  "emotional_profile": {
    "tilt_indicators": [],
    "calm_indicators": []
  },
  "account_summary": {}
}
```

Profile fields:
- `strengths`: List of `{trait, confidence, evidence_count}` — things the trader does well
- `growth_edges`: Same structure — areas for improvement
- `behavioral_patterns`: `{pattern, trigger, frequency, avg_cost, first_seen}` — recurring behaviors
- `goals`: `{goal, set_date, progress, status}` — behavioral goals set by the coach
- `session_tendencies`: Per-session stats (trade count, win rate, avg hold, avg size)
- `emotional_profile`: Inferred tilt/calm indicators from trading behavior
- `account_summary`: Per-account balance, total P&L, status

#### Coaching Digests (`data/coaching_digests.jsonl`)
AI-generated session reviews, synced to Pinecone for chat RAG.
```json
{
  "date": "2026-03-06",
  "session_label": "manual_US_session",
  "accounts": ["50KTC-V2-451890-20967121"],
  "summary": "11 MNQ trades in 30 minutes. 3 winners, 8 losers...",
  "trade_grades": [
    {"trade_id": "...", "grade": "A", "reason": "Clean breakout, held to target"},
    {"trade_id": "...", "grade": "D", "reason": "Revenge entry 15 seconds after stop-out"}
  ],
  "patterns_observed": ["rapid_reversal_cluster", "size_unchanged_after_loss"],
  "coaching_note": "The first 2 trades were solid. Then you got stopped and entered 6 times in 90 seconds...",
  "metrics": {"trades": 11, "win_rate": 0.27, "gross_pnl": -732, "fees": 57.72, "net_pnl": -789.72},
  "profile_version_before": 11,
  "profile_version_after": 12
}
```

#### Coach State (`data/coach_state.json`)
Tracks incremental fetch state.
```json
{
  "last_fetch": "2026-03-06T14:00:00Z",
  "accounts": {
    "topstepx-19858923": {"last_fill_id": 2241049355, "last_fill_ts": "2026-03-06T13:51:17Z"},
    "topstepx-17189309": {"last_fill_id": 1951529956, "last_fill_ts": "2026-01-20T04:46:10Z"}
  }
}
```

### Trade Matcher Algorithm

1. Sort fills by account + contract + timestamp
2. Track running position per (account, instrument)
3. On each fill:
   - Same direction as position → add to position (pyramid/scale-in)
   - Opposite direction → partial/full close
   - Position goes to zero → emit round-trip trade record
   - Position flips → emit close + open new trade
4. Calculate: entry_price_avg (VWAP of entry fills), exit_price_avg (VWAP of exit fills)
5. Classify: `scalp` (<5min), `swing` (5-60min), `position` (>60min), `reversal` (flip within 2min)

### Source Detection (System vs Manual)

When our execution engine is running, it writes signal records to `live_signals.jsonl`. The trade matcher checks:
- Is there a signal entry within ±60s of the broker fill entry time?
- Does the instrument match?
- If yes → `source = "system"`, attach `strategy_id`
- If no → `source = "manual"`

When paper trading or no live session → all trades are `source = "paper"` or `source = "manual"`.

### Session Digest Engine

**Trigger:** `python scripts/tools/coaching_digest.py [--date 2026-03-06]`

**Claude API prompt structure:**
1. System prompt: "You are a professional trading coach at a prop firm..."
2. `TRADING_RULES.md` excerpt (grounding)
3. Full `trader_profile.json` (current model)
4. Today's `broker_trades` (matched round-trips)
5. Today's `daily_features` (market context)
6. Recent `trade_debriefs` (if any filed)
7. Instructions: "Generate a session digest AND a profile update patch"

**Output parsing:** Claude returns structured JSON (trade grades, coaching note, profile patch). Parsed and written to respective files.

**Profile update safety:**
- Profile is git-versioned — `version` field incremented on each update
- AI generates a `profile_patch` (additive), not a full replacement
- Code merges patch into existing profile with validation
- Bad updates visible via `git diff data/trader_profile.json`

### Interactive Chat

**Implementation:** `python scripts/tools/trading_coach.py --chat`

**System prompt includes:**
1. Full `trader_profile.json`
2. Last 5 coaching digests
3. `TRADING_RULES.md` excerpt
4. Instructions: "You are this trader's personal coach. You know their patterns, strengths, and growth edges. Answer questions using evidence from their profile and trading history."

**For deeper queries:** RAG search over all coaching_digests via Pinecone.

### Integration with Existing Systems

| System | Integration |
|--------|------------|
| Discipline Coach MVP | Pre-session priming includes AI coaching note from profile |
| Pinecone Knowledge Base | Coaching digests added as new tier in sync_pinecone.py |
| Pipeline Status | `coaching_digest.py` can be added to rebuild chain |
| Health Check | Advisory: "last coaching digest was 7 days ago" |

---

## 3. Detail — Implementation Phases

### Phase 1: Broker Fill Fetcher
- `scripts/tools/fetch_broker_fills.py` — CLI to fetch fills from all configured brokers
- Reuses existing `ProjectXAuth` from `trading_app/live/projectx/auth.py`
- Normalizes to unified fill schema → `data/broker_fills.jsonl`
- Incremental fetch via `data/coach_state.json`
- Tests: mock API responses, incremental fetch, multi-account

### Phase 2: Trade Matcher
- `scripts/tools/trade_matcher.py` — fills → round-trip trades
- Position tracking per (account, instrument)
- VWAP entry/exit pricing
- Trade type classification (scalp/swing/position/reversal)
- Source detection (system vs manual) via signal correlation
- Output: `data/broker_trades.jsonl`
- Tests: known fill sequences → expected trades, edge cases (pyramids, reversals, partial fills)

### Phase 3: Session Digest Engine
- `scripts/tools/coaching_digest.py` — generates AI analysis
- Creates `data/trader_profile.json` (initial empty profile)
- Claude API call with structured prompt
- Parses response → digest + profile patch
- Output: `data/coaching_digests.jsonl` + updated profile
- Tests: mock Claude API, profile merge logic, digest schema validation

### Phase 4: Interactive Chat
- `scripts/tools/trading_coach.py --chat` — conversational interface
- Loads profile + recent digests into system prompt
- Pinecone RAG for historical digest queries
- Conversation loop with history
- Tests: prompt construction, profile loading

### Phase 5: Discipline Coach Integration
- Modify `ui/discipline_data.py` — load trader profile for priming
- Modify `ui/discipline.py` — show AI coaching note in pre-session priming
- Auto-populate debrief form with broker fill data
- Tests: priming with/without profile, debrief pre-population

### Phase 6: Pinecone Sync
- Modify `scripts/tools/sync_pinecone.py` — add `coaching` tier
- Sync coaching_digests.jsonl and trader_profile.json
- Tests: coaching content appears in Pinecone search

---

## 4. Validate — Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Claude API cost runaway | One call per session (~$0.10-0.30). Monthly cap ~$10 |
| Profile drift from bad AI update | Git-versioned, version counter, patch-based (not full replace) |
| Broker API rate limits | TopstepX: no documented limits. Fetch once per session, not continuous |
| Tradovate auth failure | TopstepX works standalone. Tradovate is additive |
| Trade matcher errors (wrong pairing) | Unit tests with known sequences. Raw fills preserved as audit trail |
| Claude API unavailable | Digest generation queued. Profile unchanged. Chat unavailable but trading unaffected |
| Privacy (trade data to Claude API) | Anthropic doesn't store API call data. All data in-transit only |
| Manual trades have no strategy context | Analysed standalone — patterns still detectable from timing, sizing, P&L |

### What This Does NOT Build
- No real-time alerts during trading (discipline coach handles cooling)
- No auto-execution or position management
- No Streamlit chat UI (CLI only for V1)
- No local LLM (Claude API only)
- No fine-tuning or model training

### Tests Required
1. `test_broker_fills` — mock API responses, incremental fetch, multi-account handling
2. `test_trade_matcher` — known fill sequences → expected trades, pyramids, reversals, partial fills
3. `test_coaching_digest` — mock Claude API, profile merge, digest schema validation
4. `test_trader_profile` — patch merge, version increment, rollback
5. `test_trading_coach_chat` — prompt construction, profile loading

### Rollback Plan
Delete new files, remove Pinecone coaching tier. Zero impact on existing pipeline or trading.

---

## 5. Scope Summary

| Deliverable | New/Modified | Lines (est) |
|-------------|-------------|-------------|
| `scripts/tools/fetch_broker_fills.py` | NEW | ~200 |
| `scripts/tools/trade_matcher.py` | NEW | ~250 |
| `scripts/tools/coaching_digest.py` | NEW | ~300 |
| `scripts/tools/trading_coach.py` | NEW | ~200 |
| `data/trader_profile.json` | NEW | ~50 |
| `ui/discipline.py` | MODIFY | +30 |
| `ui/discipline_data.py` | MODIFY | +20 |
| `scripts/tools/sync_pinecone.py` | MODIFY | +20 |
| `tests/test_coaching/*.py` | NEW | ~400 |
| Total | | ~1,470 |

---

## 6. Dependency Chain

```
1. fetch_broker_fills.py (broker auth + fill fetch)
   └── 2. trade_matcher.py (fills → round-trip trades)
       └── 3. coaching_digest.py (Claude API + profile)
           ├── 4. trading_coach.py (interactive chat)
           ├── 5. discipline.py integration
           └── 6. sync_pinecone.py (coaching tier)
```

Phases 4-6 can be parallelized after Phase 3.
