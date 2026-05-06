---
task: feat-live-app-ux-stage-A
mode: IMPLEMENTATION
slug: feat-live-app-ux-stage-A
phase: 1
total_phases: 3
---

# feat/live-app-ux — Stage A (dashboard-first operator surface)

Implements the Stage A slice of the 2026-05-06 "Live Trading Rollout — App-First, No-Gaps Edition" plan. Pure UX + preflight hardening — no production trading-logic changes.

## Goal

Operator double-clicks a desktop launcher → browser opens to `http://localhost:8080` → dashboard shows mode `SIGNAL`, lanes listed with paused-lane badges, "Trade Book" link reveals the 2 real broker fills + 580 paper trades cleanly separated. Preflight refuses to launch if the live trade journal is unhealthy.

## Scope Lock

- scripts/run_live_session.py
- trading_app/live/bot_dashboard.py
- trading_app/live/bot_dashboard.html
- tests/test_trading_app/test_bot_dashboard.py
- tests/test_trading_app/test_bot_dashboard_routes.py
- Start-Live-App.bat
- docs/runtime/stages/feat-live-app-ux-stage-A.md
- docs/runtime/stages/feat-live-app-ux-smoke.md
- docs/audit/results/2026-05-06-feat-live-app-ux.md

Per-file purpose:
- `scripts/run_live_session.py` — preflight only (add TradeJournal health block)
- `trading_app/live/bot_dashboard.py` — two new read-only endpoints
- `trading_app/live/bot_dashboard.html` — header link + lane-card badge
- `tests/test_trading_app/test_bot_dashboard.py` — coverage for new endpoints
- `Start-Live-App.bat` — new file, repo root
- `docs/runtime/stages/feat-live-app-ux-stage-A.md` — this stage file

## Blast Radius

- `scripts/run_live_session.py` — preflight bumps `checks_total` 5 → 6. Failure is hard-fail per existing fail-closed pattern. Affects only the manual `--preflight` and live-launch path; no impact on backtest, discovery, or pipeline.
- `trading_app/live/bot_dashboard.py` — adds `GET /api/trade-book` (reads `live_journal.db.live_trades` + `gold.db.paper_trades` read-only) and `GET /api/lane-status` (reads `lane_ctl.get_paused_strategy_ids` + `get_lane_override`). Zero new write paths. Zero changes to existing endpoints. Existing endpoint patterns mirrored exactly.
- `trading_app/live/bot_dashboard.html` — additive markup: one anchor in header, one badge per lane card. No CSS/JS framework changes.
- `Start-Live-App.bat` — new file, zero callers; just a venv-activate + `python scripts/run_live_session.py --signal-only` wrapper.
- Reads: `gold.db` (read-only via existing `pipeline.paths.GOLD_DB_PATH`), `live_journal.db` (read-only via existing `live_journal_path()`), lane override JSON (via canonical `lane_ctl` accessor — never re-encoded).
- Writes: none beyond stdout / stderr.
- Test deltas: new pytest cases in `tests/test_trading_app/test_bot_dashboard.py` for `/api/trade-book` and `/api/lane-status` shape + paused-lane filter behavior.

## Implementation order

1. Preflight `is_healthy` check in `_run_preflight` (smallest, isolated).
2. `/api/trade-book` endpoint — reuse the existing `_query_trades` connection idiom and the `paper_trades` schema verified live (15 columns).
3. `/api/lane-status` endpoint — call `get_paused_strategy_ids` + `get_lane_override` per id; expose `{strategy_id, reason, expires_on}` rows.
4. HTML — anchor in header (after Preflight button), `<span class="lane-paused">` badge wired to `/api/lane-status` payload.
5. `Start-Live-App.bat` — minimal wrapper.
6. Tests for the two new endpoints.

## Done When

- `python pipeline/check_drift.py` passes (118+ checks).
- `pytest tests/test_trading_app/test_bot_dashboard.py -q` passes including new cases.
- `ruff check` + `ruff format --check` clean on all touched files.
- Manual smoke (operator-side, optional): `Start-Live-App.bat` launches the dashboard at `localhost:8080`; `/api/trade-book` returns `{live_trades: [2 rows], paper_trades: [580 rows]}` shape; `/api/lane-status` lists `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` as paused.
- Self-review (institutional-rigor §1) noted in commit body.

## Out of Scope

- Stage B (one-day signal-only run) — operational, not implementation.
- Stage C (flip to LIVE) — operational, gated behind Stage B success.
- "Start Live" button auto-confirm wiring — Stage C concern.
- Express account shadow mode — open question, not Stage A.
- Any change to `trading_app/strategy_*`, `pipeline/`, or schema.
