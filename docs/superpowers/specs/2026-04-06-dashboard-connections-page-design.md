# Dashboard Connections Page — Design Spec

**Date:** 2026-04-06
**Status:** Approved
**Scope:** Frontend only (`bot_dashboard.html`). Backend already committed.

## Purpose

The dashboard is a bot control + account management tool. Manual trading happens on TopStepX/NinjaTrader directly. The dashboard shows: what accounts you have, what state they're in, and what the bot is doing. This design adds broker connection management via a dedicated Connections tab.

## Architecture: Two-Tab System

The dashboard splits into two views using a tab bar below the topbar:

- **Dashboard tab** (existing): Metrics, sessions, positions, trades, profiles. No changes.
- **Connections tab** (new): Broker CRUD, account cards, add-broker form.

Both tabs share: topbar (brand, account selector dropdown, mode badge, action buttons). Tab state persisted in localStorage. No URL routing — pure JS show/hide.

## Topbar (shared, both tabs)

```
ORB │ [▼ Account Selector] │ ● Mode Badge │ [Preflight] [Refresh] [Kill All]
```

Account selector dropdown visible on both tabs. Always shows tradeable accounts from all connected brokers.

## Tab Bar

```
[■ Dashboard]  [○ Connections]
```

- Active tab: cyan underline + white text
- Inactive tab: muted text, clickable
- localStorage key: `orb_active_tab` (values: "dashboard" | "connections")

## Connections Page Layout

### Connection Cards

One card per broker connection. Shows:
- Status dot: green (connected), grey (disabled), red (error), yellow (stale >90s)
- Display name + broker type label
- Account summary: "3 tradeable · 42 archived"
- Connected since timestamp
- Last fetched timestamp (stale if >90s yellow, >300s red)
- Action buttons: Disable/Enable, Reconnect, Remove

Error state: red border, full error message shown below header (never truncated).

### Account Cards (inside connection card)

Grid of mini cards per tradeable account:
- Account name (shortened: remove `-V2-{user_id}-` prefix, show last 6 digits)
- Balance in large mono font, green if positive
- "Realized" label (excludes unrealized P&L — honest about limitation)
- HWM value
- DD used: `HWM - balance` of `max_dd`
- Status chip: TRADEABLE (green) / RESTRICTED (yellow) / BLOWN (grey)
- Star icon on whichever account is selected in the topbar dropdown
- Click to select (updates topbar dropdown + Dashboard metrics)

"Show N archived" toggle per connection. Archived accounts at 30% opacity.

### Add Connection Form

Triggered by "+ Add Connection" button. Inline form (not modal). Shows:
- Broker type selector as clickable cards (ProjectX / Tradovate / Rithmic)
- Credential fields auto-populate based on broker type (from `BROKER_TYPES` registry)
- Password fields masked with show/hide toggle
- Base URL pre-filled with default
- Display name field
- "Test Connection" button → hits `POST /api/broker/test` → inline result (green/red)
- "Save & Connect" button → only enabled after successful test
- Cancel button returns to connection list

### Empty State

No connections at all: centered "Connect your first broker" CTA with broker type cards.

### Safety Gates

- Remove requires typing connection display name to confirm
- Disable during running bot session shows warning
- Cannot remove connection with active bot session

## Data Flow

```
Connections page load:
  fetchBrokerList() → GET /api/broker/list → renderConnectionCards()
  fetchEquity() → GET /api/equity → renderAccountCards() inside each connection

Add broker:
  Form → [Test] → POST /api/broker/test → show result
         [Save] → POST /api/broker/add → invalidate cache → re-fetch both lists

Toggle/Remove:
  POST /api/broker/toggle or /remove → invalidate cache → re-fetch
```

Polling: broker/list every 30s on Connections tab. Equity polling continues on both tabs (shared cache).

## Gap Fixes (from audit)

| Gap | Fix |
|---|---|
| Error visibility | Full error message on connection card, never truncated |
| Balance type honesty | "Realized" label on every balance display |
| Stale indicator | Yellow border >90s, red >300s, timestamp shown |
| Network down | Connection card goes red, "Last fetched: Xm ago" |
| Concurrent add | Backend uses threading.Lock on connection manager |
| Duplicate broker | Allowed (different accounts, same platform) |
| Loading skeleton | "Fetching accounts..." while equity loads |
| Tab switch form state | Form preserved (not destroyed) on tab switch |
| Empty state | "Connect your first broker" CTA |

## Files Changed

| File | Action | Notes |
|---|---|---|
| `bot_dashboard.html` | MODIFY | Tab bar + Connections page + add form + JS |
| `bot_dashboard.py` | NO CHANGE | Already committed (5 endpoints + equity refactor) |
| `broker_connections.py` | NO CHANGE | Already committed (connection manager) |

## Test Plan

| Test | Expected |
|---|---|
| Tab switching | Content shows/hides, state persists across refresh |
| Connection cards render | Status dots, account counts, timestamps correct |
| Account cards render | Balances, HWM, DD, status from equity API |
| Add with valid creds | Test green → Save → connection appears → equity updates |
| Add with bad creds | Test red → Save disabled → error shown |
| Remove connection | Type name → confirm → gone → equity updates |
| Toggle disable | Grey status → no equity fetch → enable reconnects |
| Error state | Red border + full message when auth fails |
| Stale indicator | Yellow/red border after 90s/300s without refresh |
| Account select sync | Click account card → topbar updates → Dashboard metrics update |
| Empty state | No connections → CTA shown |
| Keyboard | Tab navigates, Enter submits, Escape cancels |
