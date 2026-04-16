# Dashboard Polish + Usability Upgrade — Design Spec

**Date:** 2026-04-15
**Status:** Approved (iterative — checkpoints per tier)
**Scope:** `trading_app/live/bot_dashboard.html` (primary), `trading_app/live/bot_dashboard.py` (Tier 2 only, additive API fields)

## Purpose

Dashboard is the trader's live-trading control surface. Current build is functional but feels dated next to modern SaaS tools (Linear, Vercel, Stripe, X). More importantly: walking through actual trader workflows surfaced usability friction the polish pass alone would not fix.

## Architecture

**Layer boundary:** UI only. No trading logic, no data-pipeline, no canonical source touched.
**Hot reload:** `DASHBOARD_HTML.read_text()` on every request → browser refresh picks up changes immediately.
**Preserve:** every existing id, data-attribute, `onclick` handler, JS function, state class, responsive media query.

## Three tiers (execution order)

### Tier 3 — Polish foundation (FIRST, safest, biggest visual lift)
- Inter variable font (Google Fonts, preconnect, font-display swap)
- Base 14px / 1.55 line-height, tighter tracking on body, wider tracking on labels
- Hero metric values 18→26px, small metrics 15→18px, section titles 13→15px
- Color discipline: cyan as sole primary accent; green/red reserved for pnl and kill; drop blue, orange; yellow restricted to stale/warning (not action buttons)
- Softer softs: alpha 0.10→0.08
- Surface depth: two-layer shadow on elevated cards, inset top-highlight for subtle gradient
- Bg `#0a0e13`→`#0b1117` (one notch softer)
- Borders softened to `rgba(255,255,255,0.06)`
- Glass topbar: `backdrop-filter: blur(14px) saturate(1.4); background: rgba(11,17,23,0.72);`
- Motion: global `transition: 180ms cubic-bezier(0.4, 0, 0.2, 1)` on cards/buttons/chips
- Hover: `translateY(-1px)` + shadow bloom on cards; `scale(0.98)` on button active
- Radii: 8→10, 12→14
- Spacing: snap to 8px grid (8/12/16/20/24)
- Focus ring: uniform `outline: 2px solid cyan; outline-offset: 2px`

Implementation: additive override block appended to existing `<style>` before `</style>`. No original CSS deleted. Risk LOW. Rollback: delete override block.

### Tier 1 — Highest trader-value, CSS+markup+small JS (AFTER Tier 3 ships)
- Primary Start/Stop CTA in topbar (context-aware from operator state)
- Alert hierarchy: sticky CRITICAL panel + acknowledge action + compressed INFO list
- Connection health dot in topbar (broker auth freshness)

Implementation: markup additions in topbar + alerts section, wire to existing data (`lastOperatorState`, `lastAlertsData`, broker connection endpoint). Backend recon required: confirm acknowledge endpoint exists or add stub.

### Tier 2 — Highest cognitive-load reduction, touches backend (LAST, most risk)
- "Tonight's Book" panel — deployed lanes only, countdown + status
- R-column in trade blotter
- Unrealized P&L on open positions
- Kill button arming (2-click with countdown, shift-click bypass)

Implementation: may require 1-2 additive fields on `/api/equity` or `/api/trades` endpoints. Backend recon required.

## What is explicitly NOT in scope

- Audio cues (preference UI out-of-scope)
- Broker deep-linking (per-firm URL template config)
- After-session summary widget (overlaps with future Payout/Consistency surface)
- Mode-per-profile aggregate rework
- Rewrite of the base CSS block (additive-only approach)
- Tailwind / framework adoption (too much markup churn)

## Blast radius

| Tier | Files | LOC added | Risk |
|---|---|---|---|
| 3 | bot_dashboard.html | ~350 CSS | LOW |
| 1 | bot_dashboard.html | ~100 HTML + ~60 JS + ~80 CSS | LOW-MED |
| 2 | bot_dashboard.html + bot_dashboard.py | ~60 HTML + ~100 JS + ~60 Python | MED |

## Rollback

Per-tier: single-file revert of the additions. Nuclear: `git checkout` the HTML file.

## Acceptance

Each tier:
1. HTTP 200 at localhost:8080
2. No console errors on reload
3. All existing data ids present in DOM
4. Tab switch still works (Dashboard / Connections)
5. Responsive breakpoints (1100px, 820px) still collapse correctly
6. Visual approval from user (screenshot comparison optional)

## Failure modes + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Google Fonts blocked | Low | Low | System-ui stack fallback |
| Override breaks state class | Low | Med | Do not touch `.pulse`, `.on`, `.active`, `.hidden`, `.compat-hidden`, `.stale-indicator`, `.collapsed` |
| Markup addition breaks JS binding | Med (Tier 1/2 only) | High | Additions are new nodes with new ids; existing nodes untouched |
| Backend endpoint missing (Tier 1/2) | Med | Med | Recon before implementing; stub if not present |
| Browser refresh during edit | Low | Low | Hot-reload is intentional; user refreshes after commit |

## Guardian prompts

None required. UI layer only, no trading logic, no data, no canonical source touched.
