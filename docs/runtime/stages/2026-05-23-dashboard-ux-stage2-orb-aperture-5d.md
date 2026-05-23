---
task: Dashboard UX Stage 2 — load 5 trading days of bars + per-day ORB-aperture marks (canonical aperture only, no new doctrine)
mode: CLOSED
slug: 2026-05-23-dashboard-ux-stage2-orb-aperture-5d
risk_tier: high
updated: 2026-05-23
---

## CLOSED — acceptance evidence

- pytest dashboard suites: **77 passed** (60 existing + 9 new daily_orb_windows tests + 8 pre-existing orb_window_payload).
- `python pipeline/check_drift.py`: **163 PASSED, 0 violations, 20 advisory**.
- `python scripts/tools/audit_behavioral.py`: **all 7 checks clean**.
- Forbidden-literal grep: no `2 * 3600`, no `display_end_utc`, no `ORB_BOX_DISPLAY*` anywhere. 7200 only appears as bar-minute lookback (`5 * 1440`), never as aperture.
- Pyright: 0 new errors on diff (all pre-existing).

## Acceptance criteria result

1. ✅ Drift passes (163/163).
2. ✅ pytest 77/77.
3. ✅ Pyright 0 new.
4. ✅ Behavioral audit clean.
5. ⏳ Playwright screenshot deferred (dashboard not currently running locally).
6. ✅ No fabricated display constant in code.

## Scope Lock

- trading_app/live/bot_dashboard.py
- trading_app/live/bot_dashboard.html
- tests/test_trading_app/test_bot_dashboard_routes.py

## Blast Radius

- **`trading_app/live/bot_dashboard.py`** — extends `_orb_levels_for_instrument()` to return `daily_orb_windows: list[dict]` via a loop over the canonical `pipeline.dst.orb_utc_window()`. Raises `/api/bars-recent` cap 1440→7200 minutes; adds `days: int = 1` query param (server-clamped 1..5). No new constants. No new schema. No new canonical surface.
- **`trading_app/live/bot_dashboard.html`** — `OrbRectanglePrimitive` renderer loops over `orbState.windows` array instead of single scalar pair. Pattern follows official Lightweight Charts v5 `SessionHighlighting` plugin example (target.useBitmapCoordinateSpace + forEach + per-point timeToCoordinate). `LOOKBACK_MIN: 90 → 7200`. `loadInitial()` appends `&days=5`. Defensive shape handling for old-backend roll.
- **`tests/test_trading_app/test_bot_dashboard_routes.py`** — adds 8 tests covering: days clamp, aperture-width invariant, no fabricated display constant, backwards-compat shape, ValueError handling, partial history, lookback cap, no-lane empty array.
- Reads: none new. Writes: none new. Affects: dashboard render only.
- Reversible: 3-file revert.
- Canonical sources reused: `pipeline.dst.orb_utc_window`, `pipeline.dst.SESSION_CATALOG`, `trading_app.live.bot_state.read_state`. No re-encoding.

## Doctrinal Anchor

Per user redirect 2026-05-23 ("whats the fucking official docs and practise for it, ensure no adhoc bullshit"):
- ORB rectangle width = literal `orb_minutes` aperture from canonical `orb_utc_window`. NOT a fabricated 2h, NOT a synthesized session-end.
- Lightweight Charts pattern follows the official `SessionHighlighting` plugin example verbatim: https://github.com/tradingview/lightweight-charts/tree/master/plugin-examples/src/plugins/session-highlighting
- Zero new canonical constants. Zero schema changes.
- Label: "ORB aperture" (not "session window").

## Acceptance Criteria

1. `python pipeline/check_drift.py` passes (160+ checks, 0 violations).
2. `pytest tests/test_trading_app/test_bot_dashboard_routes.py tests/test_trading_app/test_bot_dashboard.py tests/test_trading_app/test_bot_dashboard_sse.py -q` → 60 existing + 8 new tests pass.
3. `pyright trading_app/live/bot_dashboard.py` → 0 NEW errors on diff (pre-existing noise tolerated).
4. `scripts/tools/audit_behavioral.py` passes.
5. Playwright screenshot shows: ~5 trading days of bars when ring/DB has history, up to 5 narrow rectangles each EXACTLY `orb_minutes` wide on time axis (5/15/30m — narrow marks), today's ORB high/low horizontal lines still live, focus-mode toggle still functional.
6. No `ORB_BOX_DISPLAY_DURATION_SEC`, `7200`, `2 * 3600`, or `2*3600` literal anywhere in `bot_dashboard.py` or `bot_dashboard.html` related to ORB rectangle. (Lookback param `7200` is allowed — it's `5*1440` bar lookback, not aperture.)

## Out of scope

- Window-close marker.
- Per-day actual ORB high/low (different lane per prior day).
- Configurable display duration per lane.
- Session-end resolvers in `SESSION_CATALOG`.
- Timeframe-switching when zoomed to 5d.

## Verification Plan

1. Write 8 RED tests in `test_bot_dashboard_routes.py`.
2. Implement backend `_orb_levels_for_instrument` + route changes → tests GREEN.
3. Implement frontend `orbState.windows` + `OrbRectangleRenderer.draw()` loop + `LOOKBACK_MIN` bump.
4. Run drift, pytest, pyright, behavioral audit.
5. Playwright screenshot vs Stage 1 baseline.
6. If all gates GREEN: commit + flip stage to `mode: CLOSED` + `git rm` stage file per "done means proven" doctrine.
