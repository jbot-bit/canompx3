---
task: Dashboard UX Stage 1 — typography bump + demote legacy sections behind "Show all" toggle (CSS/HTML-only, no JS/API)
mode: CLOSED
slug: 2026-05-23-dashboard-ux-stage1-css-only
risk_tier: high
closed_note: |
  Shipped in `a810690b` (`improve bot dashboard focus layout`) with follow-up
  blocked-state UI cleanup in `19861135` (`fix(dashboard): tighten start bot
  blocked-state UI`). Current 2026-05-24 verification caught and fixed a test
  harness hang caused by Starlette `TestClient`'s portal thread under pytest;
  dashboard route tests now use `httpx.ASGITransport` and still exercise the
  ASGI app. Evidence: `./.venv-wsl/bin/python -m pytest
  tests/test_trading_app/test_bot_dashboard_sse.py
  tests/test_trading_app/test_bot_dashboard_routes.py -q` => 29 passed;
  `./.venv-wsl/bin/python -m pytest tests/test_trading_app/test_bar_ring.py
  tests/test_trading_app/test_bar_persister.py
  tests/test_trading_app/test_bot_dashboard_sse.py -q` => 47 passed.
---

## Scope Lock

- trading_app/live/bot_dashboard.html

## Blast Radius

- Single file changed: `trading_app/live/bot_dashboard.html` (~5,984 lines).
- Edits target two narrow CSS regions and one new section-visibility-toggle button in the topbar:
  - **`:root` font-size variables (lines ~801-807)** — bump `--fs-body 14→16`, `--fs-small 12→13`, `--fs-micro 11→12`, `--fs-label 10.5→12`, `--fs-title 15→17`, `--fs-hero 26→30`, `--fs-hero-sm 18→22`. All ~30 component CSS rules that already `font-size: var(--fs-X)` inherit the bump for free. No layout reflow risk because the dashboard already uses a `body.cockpit` mode that gates most font references.
  - **New `body.focus-mode` CSS rules + topbar toggle button** — adds a single class on `<body>` that hides the demoted sections via `display: none`. Demoted = `#broker-accounts-section`, `#specs-section`, `#profiles-section`, `#signal-strip`, `#session-timeline` (already cockpit-hidden), `#alerts-shell` (when no warn/crit), plus preflight panel inside operator-shell. Critical alerts/errors and runtime alerts when red stay visible (gated by an `:not(.has-alerts)` selector — already wired by alert-shell render code).
- **JS untouched.** No `getElementById`, no fetch, no SSE, no API. Confirmed via grep — no JS reads the section element directly by ID for behavior; the `toggleSection()` clicks operate on the same nodes and remain functional.
- **Python untouched.** `bot_dashboard.py` not in scope_lock.
- **Reads:** none. **Writes:** none. **Affects:** browser rendering of `http://localhost:8080/` only.
- Reversible: revert the single file. Zero backend state, zero migrations.

## Acceptance Criteria

1. `python pipeline/check_drift.py` passes (expected — no production logic changed).
2. Existing dashboard pytests still pass (`pytest trading_app/live/ -k dashboard`).
3. Playwright screenshot BEFORE + AFTER comparison shows:
   - Larger fonts on metric values, lane titles, section titles.
   - Demoted sections (account specs, profiles, signal-strip, preflight breakdown, connection list) hidden by default.
   - Topbar shows a new "Show all" / "Focus" toggle button.
   - Hero row (chart + connection + operator-shell) visually dominant.
4. No regression in collapsible-section behavior on the sections that REMAIN visible (chart, trade blotter, live lanes when positions exist, runtime alerts when present).

## Out of scope (deferred to Stage 2 / Stage 3)

- ORB box ending at start+2h (Stage 2 — JS primitive change)
- Chart loading 5d historical bars (Stage 2 — JS `loadInitial()` + py `lookback_minutes` cap raise)
- Markup reorder, file split, JS refactor (deferred indefinitely — not the UX problem)

## Verification Plan

1. Baseline screenshot via Playwright (dashboard at current state, mocked or empty).
2. Apply edits.
3. After-screenshot via Playwright.
4. Run `python pipeline/check_drift.py`.
5. Run targeted pytest.
6. Side-by-side screenshot comparison delivered to user.
