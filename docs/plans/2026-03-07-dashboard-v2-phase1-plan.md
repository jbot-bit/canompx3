# Dashboard V2 Phase 1 — Backend Core

> **Parent:** `docs/plans/2026-03-07-dashboard-v2-design.md`
> **Scope:** `ui_v2/` backend only (state_machine, data_layer, discipline_api, server)
> **No modifications to existing files** (except pyproject.toml for sse-starlette dep)

## Task 0: Add sse-starlette dependency
- `uv add sse-starlette`
- Verify: `python -c "import sse_starlette; print('OK')"`

## Task 1: Create ui_v2/state_machine.py
Port `ui/session_helpers.py` and extend with 3 new states (ORB_FORMING, IN_SESSION, DEBRIEF).
- AppState dataclass with `name` enum of 8 states
- SessionBriefing dataclass (same as current)
- `get_app_state(now)` — returns current state
- `get_upcoming_sessions(now)` — returns sorted list
- Session stack for multi-session overlap
- Cooling state: server-side dict, not Streamlit session_state
- Use `ZoneInfo("US/Eastern")` for ET time (not month-based heuristic)

## Task 2: Create ui_v2/data_layer.py
Port `ui/db_reader.py` and add new queries.
- Connection caching with retry (3 attempts, exponential backoff)
- Existing: `get_prior_day_atr()`, `get_previous_trading_day()`, `get_today_completed_sessions()`
- New: `get_session_history(session_name, limit=10)` — last N occurrences with outcomes
- New: `get_rolling_pnl(days=20)` — daily R totals for sparkline + week/month aggregates
- New: `get_overnight_recap(trading_day)` — overnight session outcomes
- New: `get_fitness_regimes()` — fitness status for live portfolio strategies

## Task 3: Create ui_v2/discipline_api.py
Port `ui/discipline_data.py` — pure functions, no Streamlit.
- All functions identical, just remove Streamlit session_state dependency
- Cooling state: pass explicit `cooling_state: dict` parameter
- Keep same JSONL paths and format

## Task 4: Create ui_v2/server.py
FastAPI application with all REST endpoints.
- Mount static files at `/static/`
- Serve `index.html` at `/`
- 16 REST endpoints per design doc
- CORS middleware for dev
- Shared app state for cooling (server-side dict)
- Port 8766 (avoid conflict with webhook_server on 8765)

## Task 5: Create tests
- `tests/test_ui_v2/__init__.py`
- `tests/test_ui_v2/test_state_machine.py` — all 8 states, transitions
- `tests/test_ui_v2/test_data_layer.py` — connection retry, query functions
- `tests/test_ui_v2/test_discipline_api.py` — JSONL I/O with tmp_path
- `tests/test_ui_v2/test_server.py` — FastAPI TestClient endpoint tests

## Task 6: Verification
- `ruff check ui_v2/ tests/test_ui_v2/`
- `ruff format ui_v2/ tests/test_ui_v2/`
- `python -m pytest tests/test_ui_v2/ -x -q`
- `python pipeline/check_drift.py`
- `python -m pytest tests/ -x -q` (full suite, no regressions)

## Task 7: Commit
- Commit all new files
- Regenerate REPO_MAP.md
