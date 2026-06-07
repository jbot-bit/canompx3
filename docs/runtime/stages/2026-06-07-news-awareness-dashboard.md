---
task: IMPLEMENTATION — News-awareness integration (display + alert only) into the live dashboard. Surface high-impact USD economic-news awareness on the ORB dashboard (a news panel + a heads-up operator alert) so the operator sees volatility risk before a session, WITHOUT touching any entry/filter/sizing/capital path. Salvages the sound core module from the untracked handoff dir (`docs/handoff/news_awareness/`) with re-verified anchors. Module delegates session timing to canonical `pipeline.dst` (orb_utc_window/compute_trading_day_from_timestamp/SESSION_CATALOG) — no re-encoded logic. Alert reuses canonical `alert_engine.record_operator_alert` — no parallel channel. The handoff's `badgeSessions()` is DROPPED (targets `[data-session-label]` DOM nodes that do not exist) and replaced with a self-contained in-panel next-session line. `dashboard_rework_snippets.html` (heartbeat pill + survival strip) is EXCLUDED — separate capital-adjacent feature, planned later. Live arm/fire path (`/api/action/start`, `run_live_session`) is OUT OF SCOPE and untouched.
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/news_calendar.py
  - tests/test_live/test_news_calendar.py
  - tests/test_live/__init__.py
  - trading_app/live/bot_dashboard.py
  - trading_app/live/bot_dashboard.html
  - trading_app/live/alert_engine.py
  - docs/runtime/stages/2026-06-07-news-awareness-dashboard.md
implementation_status: IN_PROGRESS
blast_radius:
  - trading_app/live/news_calendar.py — NEW module; imported only by bot_dashboard.py (isolation invariant). Delegates to pipeline.dst (orb_utc_window, compute_trading_day_from_timestamp, SESSION_CATALOG) — read-only canonical consumers, no mutation.
  - trading_app/live/bot_dashboard.py — +1 GET route (/api/news); no change to existing routes, no new write path to gold.db.
  - trading_app/live/bot_dashboard.html — +news panel (markup/CSS/JS); new poll registration in _COCKPIT_POLLERS + bootstrap. No change to live-arm button or /api/action/start wiring.
  - trading_app/live/alert_engine.py — +1 _ALERT_RULES tuple entry; classify_operator_alert gains one category. No change to record/read/summarize signatures or persistence path.
  - data/runtime/news_calendar_cache.json + news_fired.json — NEW runtime files written by the endpoint (atomic tmp+os.replace, fail-open). Not committed (data/ gitignored).
  - tests: tests/test_live/test_news_calendar.py (NEW, 65 assertions). No downstream test consumers — module is leaf.
preflight:
  - "faireconomy feed verified live (WebFetch): JSON array, fields title/country/date/impact/forecast/previous (+actual post-release) — matches module parser."
  - "FastAPI house style verified: all 26 dashboard routes are async def; blocking I/O offloaded via `await asyncio.to_thread(...)` in 9+ places. /api/news follows this idiom."
  - "Anchors re-verified at edit time: /api/alerts :2591-2596, _ALERT_RULES :43, #alerts-shell :3272-3285, poll bootstrap :6201, _COCKPIT_POLLERS array, fetchAlerts :4719."
  - "tests/test_live/ did not exist — created with __init__.py (matches sibling test-subdir convention)."
phases:
  - "Phase 1 — module copy + real pytest (≥60 assertions, network-isolated). DONE: 65 passed."
  - "Phase 2 — @app.get('/api/news') wrapper over news_payload(fetch_calendar(...)), async def + to_thread, fail-open."
  - "Phase 3 — news panel markup/CSS/JS after #alerts-shell; badgeSessions dropped → in-panel next-session line; renderNews in poll block + bootstrap."
  - "Phase 4 — _ALERT_RULES += ('news_event','warning',('NEWS HEADS-UP',)); due_alerts → record_operator_alert + save_fired fire-once."
  - "Phase 5 — check_drift exit 0, pytest, ruff clean, isolation grep, delete handoff dir, checkpoint commits (no push)."
isolation_invariant: "grep -rn 'news_calendar' trading_app/ scripts/ must show import ONLY from bot_dashboard.py — zero references from execution/engine/risk/sizing."
---

# News-Awareness Dashboard Integration

Operator-approved plan, executed in worktree `canompx3-news-awareness` off
origin/main. Display + alert only. No push (operator drives push).

See the plan in the session transcript for the full phase breakdown and the
verified ground-truth anchors. This stage file authorizes edits to the
scope-locked files above and is the IMPLEMENTATION-mode gate for the work.
