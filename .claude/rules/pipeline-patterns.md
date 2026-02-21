# Pipeline Patterns

## Core Principles
- **Fail-closed:** Any validation failure aborts immediately
- **Idempotent:** All operations safe to re-run (INSERT OR REPLACE / DELETE+INSERT)
- **One-way dependency:** pipeline/ -> trading_app/ (never reversed)

## DST Canonical Model
- All DB timestamps are UTC. Local timezone: Australia/Brisbane (UTC+10, no DST).
- Sessions 0900/1800/0030/2300 shift relative to market events with DST.
- ANY analysis touching these sessions MUST split by DST regime (US for 0900/0030/2300; UK for 1800).
- DST-aware session times live in `pipeline/dst.py` SESSION_CATALOG.
- The `SESSION_WINDOWS` dict in `build_daily_features.py` is Brisbane-time approximations for stats only — NOT actual market open times.

## Database Write Pattern
Uses idempotent DELETE-then-INSERT: delete existing rows for the date range, then insert new ones.
Prevents duplicates without requiring upsert logic.

## Time & Calendar
- Trading day: 09:00 local -> next 09:00 local
- Bars before 09:00 assigned to PREVIOUS trading day
- February (EST): US data 11:30pm Brisbane, NYSE open 12:30am Brisbane
