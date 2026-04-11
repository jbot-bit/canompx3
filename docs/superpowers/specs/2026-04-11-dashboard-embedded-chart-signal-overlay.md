# Dashboard Embedded Chart + Signal Overlay — PARKED SPEC

**Date:** 2026-04-11
**Status:** PARKED — documented, not scheduled, not implemented
**Author:** Captured from design discussion; brainstormed then parked per user direction
**Activation gate:** See § 14 Activation Criteria
**Depends on:** Signal Monitor (`2026-04-06-signal-monitor-design.md`, done), Dashboard Connections (`2026-04-06-dashboard-connections-page-design.md`, done), `bot_state.json` lane_cards schema (existing), `trading_app/live/bot_state.py:build_state_snapshot` canonical source

---

## 0. TL;DR

Embed a TradingView-style interactive chart directly in the existing `bot_dashboard.html`, overlaid with the ORB box, filter-pass/fail state, entry/stop/target lines, and entry markers — all driven from the existing `bot_state.json` → `lane_cards` pipeline. Optional one-click order execution reuses the existing `broker_connections` + Rithmic/ProjectX adapter path. **No new research authority, no new truth claims, no new data pipeline.** The chart is a *visual projection* of state that already exists.

This spec does **not** authorize implementation. It exists so the idea is not lost and can be picked up cold in a future session. Activation criteria are in § 14.

---

## 1. Purpose & Motivation

### 1.1 What the user asked for

Verbatim from the 2026-04-11 discussion:

> "What about as a visual like the trading view embedded into my app so I can use it like the actual trading view app — and my project makes indicators exactly for my strategies — then they pop up and if it's set to auto trade or w/e it does — or I can manually trade from that screen?"

Translated to requirements:

1. A **TradingView-quality chart** embedded inside the project's own dashboard (not an external app)
2. **Custom indicators rendered from validated strategies** — ORB box, filter state, entry/stop/target lines, directional arrows
3. **Signal visibility** — when a validated-strategy breakout is live, it's visually obvious on the chart
4. **Optional trade execution from the chart** — either the bot auto-fires, or a human clicks a button on the same screen to route the order

### 1.2 Why this is the mature version of the original "TradingView MCP/CDP" idea

The initial framing — "set up a TradingView MCP/CDP bridge so Claude can puppet the TradingView Desktop app via remote debugging" — was **rejected in the same discussion** for these reasons:

- Puppets someone else's UI; fragile to TV's Electron version
- TradingView's data layer is not research-grade for CME futures (Databento is authoritative)
- Doesn't integrate with the Rithmic/TopStep XFA execution path
- Violates `.claude/rules/integrity-guardian.md` Rule 7 ("Never trust metadata — screenshots and DOM scraping are not verification")
- No official TradingView MCP server exists; a custom one is a side project that competes with active research priorities

**Embedding a chart *library* inside the dashboard the project already owns** solves the actual user need without any of those problems. That's the path this spec documents.

### 1.3 Where this fits in the existing roadmap

The memory note [`dashboard_app_vision`](../../.claude/projects/C--Users-joshd-canompx3/memory/dashboard_app_vision.md) defines three phases:

| Phase | Name | Status |
|---|---|---|
| Phase 1 | Live Equity (ProjectX auth → real balance on dashboard) | Partially built; see `dashboard_todo.md` |
| Phase 2 | Account Switcher + Settings | Not started |
| Phase 3 | Full App Architecture (multi-broker, historical equity, WebSocket realtime) | Not started |

**This spec formalizes the chart-embed as a Phase 3 sub-scope**, not a new phase. It explicitly requires Phases 1 & 2 to ship first, because the chart's backend (market data relay, WebSocket plumbing, profile dispatch) reuses the Phase 1/2 infrastructure.

---

## 2. Non-Goals (explicit — what this is NOT)

Anti-scope creep. Everything below is **out of scope** for this spec. A future expansion that touches any of these requires a new spec.

- ❌ **Research authority.** The chart is DEPLOYMENT_ANALYTICS per `docs/specs/research_modes_and_lineage.md` § 2.3. It cannot generate new edge claims, cannot promote new strategies, cannot write to `validated_setups` or `experimental_strategies`. If the chart "reveals" a new pattern, that pattern MUST be re-tested in DISCOVERY mode per § 2.5 escape-hatch rule.
- ❌ **New data pipeline.** The chart consumes `bars_1m` from `gold.db` (historical) and the existing ProjectX/Rithmic live bar stream (realtime). Zero new bar sources, zero new parsers.
- ❌ **New signal computation.** All signal state comes from `trading_app/live/bot_state.py:build_state_snapshot()` → `lane_cards`. The chart does not re-compute ORB boundaries, filter pass/fail, entry prices, or any other lane state. Re-encoding any of this would violate `.claude/rules/institutional-rigor.md` Rule 4 ("Delegate to canonical sources — never re-encode").
- ❌ **New order routing.** Execution reuses `trading_app/live/broker_connections.py` → `rithmic_adapter.py` / ProjectX order path. The chart's "click to trade" button is a *UI wrapper* over the existing path.
- ❌ **Replacement of the signal strip.** The table-view signal strip from `2026-04-06-signal-monitor-design.md` stays as the primary "at a glance" view. The chart is a *drill-down* — click a lane row → chart zooms to that instrument/session.
- ❌ **TradingView account integration.** No TV Pine Script, no TV alerts, no TV broker links, no TV watchlist sync. The only thing the project takes from TradingView is the open-source *chart rendering library* (see § 5.2).
- ❌ **Multi-timeframe analysis tools beyond 1m/5m/15m/30m.** The project's canonical bar resolution is 1m (with 5/15/30m ORB apertures). The chart MUST NOT introduce new timeframes or interpolate sub-minute data. That would contradict the 1s-break-speed NO-GO (memory: `1s_break_speed_killed.md`).
- ❌ **Pine Script or custom indicator scripting language.** Indicators are rendered from canonical Python state (`lane_cards`). No user-scriptable indicator surface.
- ❌ **Replacement of the existing `bot_dashboard.html` page.** The chart is a *new component* on the existing page (or a new tab), not a rewrite.

---

## 3. Document Authority & Mode

Per `docs/specs/research_modes_and_lineage.md`:

- **Mode:** DEPLOYMENT_ANALYTICS (§ 2.3) — "Within already-validated strategies, what does the live portfolio look like?"
- **Allowed inputs:** `validated_setups`, `live_config`, `prop_profiles.ACCOUNT_PROFILES`, `bars_1m`, `daily_features`, `orb_outcomes`, `lane_cards` from `bot_state.json`.
- **Forbidden outputs:** no writes to `experimental_strategies`, `validated_setups`, `edge_families`, `family_rr_locks`. Chart-surfaced patterns must be re-tested in DISCOVERY mode before becoming claims.
- **Authority note:** This spec is feature-level. On conflicts, code > CLAUDE.md > `docs/specs/research_modes_and_lineage.md` (for mode discipline) > this spec.

---

## 4. Alternatives Evaluated

### 4.1 TradingView Desktop + CDP remote-debugging bridge — **REJECTED**

Original framing of the user's question. Rejected for:

- Fragile (Electron version-dependent)
- Not research-grade data
- No Rithmic/TopStep broker integration
- Violates Rule 7 (screenshot ≠ verification)
- No official MCP
- Competes with research priorities for zero net benefit

### 4.2 TradingView Widget (iframe) — **REJECTED**

TradingView's free embeddable widget:

- Shows TV's data, not the project's own data → worthless for overlaying project-specific signals
- Cannot draw custom shapes, indicators, or entry markers driven by external state
- License restricts commercial use
- Eliminated on first look

### 4.3 TradingView Charting Library (paid-tier feature, free for approved non-commercial) — **CONSIDERED, DEFERRED**

TradingView's "Advanced Charts" / Charting Library:

- Free for non-commercial use, requires application + approval from TradingView
- You supply the data; you get the full "TradingView feel" (drawing tools, indicator panels, symbol search)
- Larger footprint (~2 MB), more setup
- Pros: genuine TV look-and-feel, drawing tools out of the box
- Cons: license application process, delayed approval, heavier
- **Decision:** defer to Phase 3.2 if Lightweight Charts (§ 4.4) hits a UX wall

### 4.4 TradingView Lightweight Charts (MIT-licensed, open source) — **CHOSEN**

Repo: `github.com/tradingview/lightweight-charts`

- **License:** Apache 2.0 (verify at activation — see § 13 Open Questions)
- **Size:** ~40 KB gzipped
- **Install:** single `<script>` tag in `bot_dashboard.html` or `npm` if a build step is added later
- **API:** direct data feed, custom markers, horizontal price lines, series coloring
- **No approval, no sign-up, no commercial restrictions**
- **Cons:** no built-in drawing tools, no symbol search, no indicator panels — but we don't need those (indicators come from `lane_cards`, not user drawing)

**Why this is the right default:** it matches the project's existing no-framework, single-HTML-page dashboard style. Adding a `<script>` to `bot_dashboard.html` preserves the "local tool, no build step" design decision implicit in the current dashboard.

### 4.5 Build a custom chart from scratch (Canvas / D3 / Plotly) — **REJECTED**

- Reinventing a well-solved problem
- D3 is powerful but heavy-hand; Plotly is slow for tick-level updates
- Maintenance burden for zero upside over Lightweight Charts

### 4.6 Switch to NinjaTrader / Sierra Chart / Quantower — **REJECTED (scope)**

These platforms already do "chart + custom indicators + Rithmic execution" with zero code. But:

- The project's edge is the custom pipeline (literature-grounded discovery, Bailey MinBTL, era-stability audits). Re-platforming to NinjaScript / ACSIL / Quantower C# means rewriting `trading_app/` entirely
- Rejected not because these platforms are bad (they're good), but because the user has invested heavily in the custom pipeline and the dashboard is already working
- Keep as a fallback option if the in-house chart turns out to be a maintenance sink

---

## 5. Architecture (sketch, not code)

### 5.1 High level

```
┌─────────────────────────────────────────────────────────────────┐
│ bot_dashboard.html  (existing, FastAPI-served, Tailwind theme) │
│                                                                 │
│   ┌──────────────────┐   ┌──────────────────────────────────┐ │
│   │ Signal Strip     │   │ Chart Panel  (NEW)               │ │
│   │ (existing)       │   │                                  │ │
│   │                  │   │  [Lightweight Charts instance]   │ │
│   │  MGC TOKYO_OPEN  │──▶│                                  │ │
│   │  MNQ NYSE_OPEN   │   │  • 1m candles from gold.db       │ │
│   │  MES CME_REOPEN  │   │  • Live bar from ProjectX WS     │ │
│   │                  │   │  • ORB box (high/low lines)      │ │
│   └──────────────────┘   │  • Entry/stop/target price lines │ │
│                          │  • Signal marker (arrow)         │ │
│                          │  • Filter-state badge            │ │
│                          │                                  │ │
│                          │  [Trade panel] (optional Phase 3)│ │
│                          └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                                        │
         │ polls /api/state every 5s              │ subscribes /ws/bars
         ▼                                        ▼
┌─────────────────┐                  ┌─────────────────────────┐
│ bot_state.json  │                  │ ProjectX SignalR feed   │
│ (lane_cards)    │                  │ (already connected)     │
└────────┬────────┘                  └─────────────┬───────────┘
         │                                         │
         ▼                                         ▼
┌────────────────────────────────────────────────────────────────┐
│ session_orchestrator.py  (existing — canonical state source)   │
│  • ORB detection (uses pipeline.dst.orb_utc_window)            │
│  • Filter evaluation (uses trading_app.config.StrategyFilter)  │
│  • Signal generation (existing logic, zero changes)            │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Frontend layer

- **Library:** TradingView Lightweight Charts via CDN `<script>` in `bot_dashboard.html`
- **Framework:** none — matches existing dashboard's vanilla-JS + Tailwind style
- **Chart element:** new `<div id="chart-panel">` rendered next to or below the signal strip
- **Data binding:** existing `/api/state` poll already returns lane_cards; extend it with a new `/api/bars?instrument=X&session=Y` endpoint for historical candles and `/ws/bars?instrument=X` for live updates
- **Signal overlay:** when `lane_card.status` changes (WAITING → ORB_FORMING → ARMED → SIGNAL → IN_TRADE), the chart redraws:
  - ORB_FORMING: shaded rectangle from session start to current time
  - ARMED: horizontal price lines at ORB high + ORB low
  - SIGNAL: directional arrow marker at entry price, stop line (red), target line (green), filter-badge annotation
  - IN_TRADE: position marker + live P&L readout
  - DONE: grey-out state, show win/loss result

### 5.3 Backend layer

Two new endpoints on `bot_dashboard.py` (FastAPI):

#### `GET /api/bars?instrument=MNQ&session=NYSE_OPEN&date=YYYY-MM-DD&minutes=1`

- Reads `bars_1m` from `pipeline.paths.GOLD_DB_PATH` (canonical DB path)
- Uses `pipeline.dst.orb_utc_window(trading_day, orb_label, orb_minutes)` to compute ORB window bounds for overlay (canonical — see `docs/postmortems/2026-04-07-e2-canonical-window-fix.md`)
- Returns `{bars: [{time, open, high, low, close, volume}, ...], orb_window: {start, end}}`
- **Read-only.** Fails closed on any DB error (returns HTTP 503, no partial data).

#### `WS /ws/bars?instrument=MNQ`

- WebSocket channel that relays new bars from the existing ProjectX SignalR subscription
- Reuses `trading_app/live/projectx/market_data.py` subscription (do not create a new ProjectX client — that would violate connection-uniqueness per `dashboard_known_limitations.md` item "One connection per broker type")
- Fan-out: one ProjectX subscription, multiple WS clients on the dashboard side

### 5.4 Signal state binding

**No new signal code.** All state derives from `bot_state.json` → `lane_cards`, which is populated by `session_orchestrator.py`. The chart polls `/api/state` on the existing 5s cadence, reads the lane_card for the chart's active instrument, and renders overlays from those fields.

Required `lane_card` fields (some already exist, some are prerequisites — see § 6):

| Field | Source | Status |
|---|---|---|
| `lane_id` | existing | ✅ |
| `instrument` | existing | ✅ |
| `session` (orb_label) | existing | ✅ |
| `status` (WAITING/…/DONE) | existing per signal-monitor spec | ✅ |
| `orb_high`, `orb_low` | PREREQUISITE (dashboard_todo #1) | ❌ BLOCKING |
| `entry_price` | PREREQUISITE (dashboard_todo #1) | ❌ BLOCKING |
| `stop_price` | PREREQUISITE (dashboard_todo #1) | ❌ BLOCKING |
| `target_price` | PREREQUISITE (derived from stop + RR) | ❌ BLOCKING |
| `direction` (LONG/SHORT) | existing per signal-monitor spec | ✅ |
| `filters_state` (list of filter checks with pass/fail) | existing per signal-monitor spec | ✅ |
| `entry_model` (E1/E2/E3) | existing | ✅ |
| `rr_target` | existing | ✅ |
| `signal_time_utc` | PREREQUISITE (for entry marker x-coord) | ❌ BLOCKING |

### 5.5 Order execution (optional, Phase 3.3)

- "Click to trade" button next to the chart
- Reuses `trading_app/live/broker_connections.py` → currently-active profile's broker adapter
- For TopStep XFA: routes via Rithmic adapter (`trading_app/rithmic_adapter.py`)
- For ProjectX accounts: routes via existing ProjectX order path
- **HARD GATE:** cannot be enabled until F-1 orchestrator wiring (see `topstep_canonical_audit_apr8.md` Known Follow-ups) is complete. The Scaling Plan Day-1 2-lot cap check must be wired before any live-trade button exists on any UI. This is non-negotiable and enforced by the activation criteria in § 14.

---

## 6. Prerequisites (must be closed before any work starts)

### 6.1 BLOCKING prerequisites (from `dashboard_todo.md`)

1. **`bot_state.py:build_state_snapshot()` lane_card schema extension** — add `orb_high`, `orb_low`, `entry_price`, `stop_price`, `target_price`, `signal_time_utc`. Source: `session_orchestrator.py` already knows all these; just needs to propagate them. **This is blocking for BOTH the signal strip (`dashboard_todo.md` #1) and this chart spec.**
2. **Auto-start signal-only session on dashboard startup** (`dashboard_todo.md` #5) — without a running signal session, the chart has no live state to render.

### 6.2 Recommended prerequisites

3. **Dashboard Phase 1 ship** — live equity card + ProjectX account list working end-to-end. The chart spec's backend reuses the same FastAPI structure; shipping Phase 1 first de-risks the plumbing.
4. **F-1 orchestrator wiring** (`topstep_canonical_audit_apr8.md` Known Follow-ups) — required ONLY for § 5.5 trade execution. Chart read-only view can ship without F-1.

---

## 7. Phase Plan

Sub-phases under dashboard vision Phase 3. Each sub-phase is independently useful and independently shippable.

### Phase 3.1 — Historical Chart (read-only, no live data)

- Single `bar_data` endpoint serves 1m candles from `bars_1m`
- Chart renders in `bot_dashboard.html` for a user-selected (instrument, date, session)
- ORB box overlay from `pipeline.dst.orb_utc_window`
- **No live updates, no signals, no trade buttons**
- **Value:** quick visual QA of historical bars + ORB bounds; useful even standalone
- **Estimated scope:** 1 focused session (~4–6 hours)

### Phase 3.2 — Live Chart with Signal Overlay (read-only, live feed)

- Depends on: § 6.1 prerequisites #1 and #2
- `/ws/bars` WebSocket fan-out from ProjectX feed
- Lane-card-driven overlay: ORB lines, entry/stop/target, filter badges, signal arrow
- Status transitions drive chart redraws in real time
- **No trade buttons** — purely a visual projection of state
- **Value:** full "see what the bot sees" experience; operational confidence pre-live
- **Estimated scope:** 2–4 focused sessions

### Phase 3.3 — Click-to-Trade (optional, gated)

- Depends on: § 6.1 prerequisite #4 (F-1 orchestrator wiring)
- Adds order entry panel adjacent to chart
- Routes via existing broker connections; no new order path
- Requires human-in-the-loop confirmation dialog for every order (not a "trade panel" that fires on click)
- **Value:** manual trading inside the dashboard
- **Estimated scope:** 1–2 focused sessions after F-1 ships

### Phase 3.4 — (deferred) Upgrade to Charting Library for drawing tools

- Only if Phase 3.2 hits a UX ceiling with Lightweight Charts
- Requires TV Charting Library license application
- **Estimated scope:** 1 session once approval lands

---

## 8. Acceptance Criteria

### Phase 3.1

- [ ] `GET /api/bars` returns 1m candles from `bars_1m` via `pipeline.paths.GOLD_DB_PATH`
- [ ] Chart renders in `bot_dashboard.html` using Lightweight Charts
- [ ] ORB window overlay uses `pipeline.dst.orb_utc_window` (canonical — no re-encoding)
- [ ] No writes to any table (DEPLOYMENT_ANALYTICS mode)
- [ ] All existing dashboard pages unchanged / unbroken
- [ ] Lighthouse UI test: chart loads in <2s for a single day of 1m bars
- [ ] `python pipeline/check_drift.py` still passes
- [ ] Unit tests for `/api/bars` endpoint covering: happy path, invalid instrument, DB missing, ORB window edges

### Phase 3.2

- All Phase 3.1 criteria PLUS:
- [ ] Prereq #1 and #2 closed on main
- [ ] `/ws/bars` fans out ProjectX bars to ≥2 concurrent clients without double-subscribing
- [ ] Lane card schema extension ships first (separate PR)
- [ ] Chart overlay updates in <500ms from lane_card status change
- [ ] Signal marker, stop line, target line, filter badge render exactly per signal-monitor spec states
- [ ] Lane-card status → overlay mapping has unit tests (pure function, no DOM)
- [ ] Behavioral audit (`scripts/tools/audit_behavioral.py` if applicable)

### Phase 3.3

- All Phase 3.2 criteria PLUS:
- [ ] F-1 orchestrator wiring shipped and verified on main
- [ ] RiskManager `topstep_xfa_account_size` set from `prof.account_size` at session start
- [ ] Day-1 2-lot cap check hit-tests every proposed order from the chart button
- [ ] Confirmation dialog is mandatory (no "remember my choice" bypass)
- [ ] Integration test: simulated click → existing order path → mock broker adapter

---

## 9. Canonical Source Dependencies (do NOT re-encode)

Per `.claude/rules/institutional-rigor.md` Rule 4. Every cell in this table is a "call, do not re-implement" contract.

| Concern | Canonical source |
|---|---|
| DB path | `pipeline.paths.GOLD_DB_PATH` |
| ORB window (UTC) | `pipeline.dst.orb_utc_window(trading_day, orb_label, orb_minutes)` |
| Session catalog | `pipeline.dst.SESSION_CATALOG` |
| Session DOW guard | `pipeline.dst` Brisbane DOW handling |
| Active instruments | `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS` |
| Cost specs | `pipeline.cost_model.COST_SPECS` |
| Entry models / filters | `trading_app.config` |
| Filter introspection | `filter.describe()` method on filter instances |
| Live lane state | `trading_app.live.bot_state.build_state_snapshot` |
| Session lifecycle | `trading_app.live.session_orchestrator` |
| Broker connections | `trading_app.live.broker_connections` |
| Prop profile active | `trading_app.prop_profiles.ACCOUNT_PROFILES` |
| Rithmic order path | `trading_app.rithmic_adapter` |
| ProjectX order path | `trading_app.live.projectx/*` |
| TopStep XFA risk | `trading_app.risk_manager` + `topstep_scaling_plan.py` |
| Holdout policy | `trading_app.holdout_policy` (read-only for display) |

---

## 10. Tradeoffs

| Axis | Chart embed (this spec) | Fallback (NinjaTrader/Sierra) | Do nothing |
|---|---|---|---|
| Dev time | 1–2 weeks across 3 sub-phases | ~0 (already built) | 0 |
| Custom indicator fit | Exact — driven by our state | Requires NinjaScript/ACSIL port | N/A |
| Execution path | Reuses existing Rithmic/ProjectX | Platform-native | N/A |
| Research authority | DEPLOYMENT_ANALYTICS only | N/A | N/A |
| Maintenance | In-house, future us owns it | Vendor-owned, locked in | Zero |
| Licensing risk | Apache 2.0 (verify) | Platform subscriptions | Zero |
| Fits existing dashboard | Perfectly — no new app shell | Requires separate window | N/A |
| Competes with research priorities | Yes — ~1–2 weeks of capacity | No | No (opportunity cost zero) |

**The honest summary:** this is a "nice operational UI" investment, not a research investment. It has real value when the bot is live and trading real money and the user wants eyes on what the bot is doing — but that condition is not met today (bot is in signal/demo mode, F-1 hard gate not passed).

---

## 11. Risks & Failure Modes

| Risk | Impact | Mitigation |
|---|---|---|
| **Scope creep into a full trading platform** | Wave 4 / Phase 3d research stalls | Hard cap: never touch research layers from the chart. Non-goals § 2 enforced in PR review. |
| **Silent signal miss from desync between chart and bot** | User sees "no signal" when bot has one | Chart is read-only consumer of `bot_state.json`. If bot_state is stale (>30s heartbeat), chart shows a banner and refuses to display signals. |
| **ProjectX feed multiplexing bug** | Double-subscription breaks both chart and signal strip | Reuse existing `market_data.py` subscription via fan-out, do NOT create a new ProjectX client. Integration test with 3+ concurrent WS clients. |
| **Lightweight Charts API breaking change** | Chart breaks after npm update | Pin version; no automatic updates. Add pin to a new `requirements.lockfile` note or `package.json` if one is introduced. |
| **Chart click fires a live order when user intended a click-to-zoom** | Real-money mistake | Confirmation dialog for ALL orders. No hotkeys for order entry. Separate visual region for "trade panel" vs "chart" vs "signal strip". |
| **Signal-to-chart render lag during high volatility** | User sees stale overlays at critical moments | 500ms SLA in acceptance criteria. If not met in Phase 3.2, fall back to "chart is for review only, use the signal strip table for realtime" and document the limitation. |
| **"Look, the chart shows a pattern" → "let's trade it" without re-validation** | Data snooping, silent escalation (OPERATIONS → DISCOVERY without discipline) | Explicit in § 2 non-goals + § 3 mode declaration. Any pattern noticed on the chart MUST re-enter DISCOVERY mode before becoming a claim. PR review enforces. |
| **F-1 orchestrator wiring never ships** | Phase 3.3 click-to-trade can never activate | Spec declares Phase 3.3 hard-gated on F-1. Phases 3.1 and 3.2 ship value without it. |
| **Single HTML page becomes unmaintainable** | Phase 3 forces a React/Vue migration | Monitor line count; if `bot_dashboard.html` exceeds ~2000 lines, revisit framework choice in a new spec. |
| **TradingView Lightweight Charts license changes** | Legal risk | Verify Apache 2.0 at activation (§ 13 Q1). If license changed, fall back to custom Canvas or Plotly. |

---

## 12. Seven Sins Check

Run against `.claude/rules/quant-agent-identity.md` as a sanity gate.

| Sin | Risk in this spec | Mitigation |
|---|---|---|
| Look-ahead bias | Could the chart show future bars to the user on a historical replay? | Historical mode renders all data at once (no walk-forward animation). Live mode only shows bars that have closed. `double_break` explicitly NOT rendered (it's look-ahead). |
| Data snooping | Could visual pattern-matching drive new strategy claims? | § 2 non-goals: any pattern noticed must re-enter DISCOVERY. § 3 mode declaration. PR review. |
| Overfitting | N/A — no new research | N/A |
| Survivorship bias | Would the chart hide dead instruments / purged entry models? | Chart shows whatever is in the active profile. Dead instruments (MCL, SIL, M6E, MBT, M2K) are not in any active profile, so they won't appear. E0 is purged and not in lane_cards. |
| Storytelling bias | Would a pretty chart persuade the user into trades the statistics don't support? | Mitigation: the "click to trade" button requires confirmation dialog + per-order risk check. The chart does not imply "this will work". |
| Outlier distortion | Could a single candle dominate a chart's y-axis and mislead? | Lightweight Charts auto-scales price axis. Document the auto-scale behavior in UI tooltip. |
| Transaction cost illusion | Would the chart's profit-line ignore costs? | Stop/target lines are drawn at nominal prices. P&L readout in IN_TRADE state MUST use `pipeline.cost_model.COST_SPECS` (not nominal). Acceptance criterion. |

---

## 13. Open Questions

1. **Lightweight Charts license** — verified Apache 2.0 at time of spec writing (2026-04-11) per the repo README, but verify against the current `LICENSE` file at activation. If TV has moved it to a restrictive license, fall back to § 4.5 custom Canvas.
2. **Chart location on page** — new tab, modal overlay, or inline next to signal strip? Recommend inline (clicking a signal-strip row expands the chart for that lane). Decide at activation.
3. **Single chart or multi-chart grid?** Phase 3.2 could support multiple charts (one per active lane). Probably out of scope for first version; revisit after 3.1 ships.
4. **How to handle the ProjectX feed being down** — grey out chart? Show last-known bar + "STALE" banner? Decide at activation based on how the signal strip handles it.
5. **Historical bar cache TTL** — `/api/bars` hits DuckDB on every request. Cache per `(instrument, date)` key for ~60s? Only matters if multi-client use cases emerge.
6. **Mobile viewport** — does the dashboard need to work on a phone? Currently desktop-only. Phase 3.2 may inherit that constraint.
7. **Chart theme integration** — Lightweight Charts has a `ColorType.Solid` theme API. Match existing Tailwind dark palette exactly or use defaults? Cosmetic, defer.

---

## 14. Activation Criteria

This spec is PARKED. Activation (transition to scheduled work) requires **ALL** of the following:

1. **Wave 4 lit-grounded discovery fully frozen** (no pending adversarial audits, no T6 follow-ups open)
2. **Phase 3d validated_setups audit closed** (the 124 grandfathered strategies have been audited against Phase 3a era discipline per Phase 3c handover doc)
3. **Prereq #1 closed** — `bot_state.py` lane card schema extended with `orb_high`, `orb_low`, `entry_price`, `stop_price`, `target_price`, `signal_time_utc`
4. **Prereq #2 closed** — auto-start signal-only session on dashboard startup (`dashboard_todo.md` #5)
5. **Dashboard Phase 1 shipped** — live equity card shows real ProjectX balance
6. **User explicit go** — user says "implement the chart spec" or equivalent. Stage gate writes `stages/dashboard-embedded-chart.md` with scope_lock before any code.

If these criteria are met and the user wants to proceed, the implementing session should:

- Re-verify all canonical-source references in § 9 (files may have moved / been renamed since 2026-04-11)
- Run `git log --since="2026-04-11" -- trading_app/live/` to catch dashboard changes that may invalidate assumptions
- Re-check § 13 Question 1 (Lightweight Charts license)
- Start with Phase 3.1 only (historical chart, no live data, no signals, no trades)

---

## 15. Cross-Links

- **Memory:** `dashboard_app_vision.md` (evolution plan), `dashboard_known_limitations.md` (current gaps), `dashboard_todo.md` (blocking/important items)
- **Specs:** `docs/superpowers/specs/2026-04-06-signal-monitor-design.md` (signal strip, parent), `docs/superpowers/specs/2026-04-06-dashboard-connections-page-design.md` (connections tab, sibling), `docs/specs/research_modes_and_lineage.md` (mode discipline authority)
- **Rules:** `.claude/rules/institutional-rigor.md` (Rule 4: delegate to canonical sources), `.claude/rules/integrity-guardian.md` (Rule 7: never trust metadata), `.claude/rules/research-truth-protocol.md`
- **Audits/Plans:** `topstep_canonical_audit_apr8.md` (F-1 hard gate), `docs/plans/2026-04-07-holdout-policy-decision.md` (Mode A holdout), `docs/postmortems/2026-04-07-e2-canonical-window-fix.md` (ORB window canonical contract)
- **ROADMAP:** this spec is referenced from the "Dashboard Phase 3 — Embedded Chart + Signal Overlay" entry

---

## 16. Status Tracker

| Date | Status | Note |
|---|---|---|
| 2026-04-11 | PARKED | Spec written from design discussion. No implementation. Activation criteria in § 14. |

---

**End of spec.**
