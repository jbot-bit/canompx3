---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Dashboard V2 — Bloomey Fixes Applied

**Date:** 2026-03-08
**Trigger:** Bloomey review of Dashboard V2 design — Grade B+, 7 action items
**Status:** IMPLEMENTED

## Fixes Applied

### 1. State Persistence (`data/session_state.json`)
- **File:** New `ui_v2/state_persistence.py`
- Cooling state and commitment checklist now persist across server restarts
- Atomic write via temp file + rename (corruption-safe)
- `server.py` loads from disk on startup, saves on every mutation

### 2. SQL Injection Fix (data_layer.py)
- **All 7 query functions** converted from f-string SQL to parameterized queries (`?` placeholders)
- `query_df()` now accepts `params: list | None` parameter
- Even though inputs come from validated backend sources, defense-in-depth is correct

### 3. DuckDB Query Retry
- `query_df()` now retries once on `IOException` (stale connection from concurrent writer)
- Drops cached connection and reconnects, then retries the query
- Previous behavior only retried the initial connection, not subsequent queries

### 4. Session Lock (`/api/session/start`)
- `asyncio.Lock` prevents concurrent session starts
- Ready for Phase 2 wiring to SessionOrchestrator

### 5. First-Run Detection
- `/api/briefings` now returns `first_run: true` when LIVE_PORTFOLIO is empty
- `/api/health` reports `portfolio_status` and `portfolio_strategies` count
- Frontend can show onboarding message instead of empty state

### 6. `/api/health` Endpoint
- DB connectivity check (SELECT 1)
- SSE client count
- Server uptime
- Current state
- Portfolio status (ok/empty/unavailable/error)
- Cooling active status

### 7. Package `__init__.py`
- Already existed — confirmed present

## Test Coverage
- 8 new tests for state persistence (roundtrip, corruption, coexistence)
- All 108 ui_v2 tests pass
- 68 drift checks pass

## Files Changed
| File | Change |
|------|--------|
| `ui_v2/data_layer.py` | Parameterized SQL, query retry |
| `ui_v2/server.py` | Health endpoint, session lock, persistence wiring, first-run detection |
| `ui_v2/state_persistence.py` | NEW — JSON state persistence |
| `tests/test_ui_v2/test_state_persistence.py` | NEW — 8 tests |
