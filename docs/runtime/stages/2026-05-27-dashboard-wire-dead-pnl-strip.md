task: Wire dead bottom P&L strip cells (pnl-day/pnl-open/pnl-trades) to existing live data; fix stuck "Loading accounts..." dropdown; make idle state read as intentional. Frontend-only.
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.html

## Blast Radius
- trading_app/live/bot_dashboard.html — frontend only. Wires existing `state.daily_pnl_r` / `state.bars_received` (already consumed by hero card + metrics in updateStatus) into the three orphaned P&L-strip cells that currently show static "—" and are never written by any JS setter (dead placeholders / institutional-rigor §5 violation).
- Adds a catch-branch reset to fetchEquity so the acct-select dropdown can never stay frozen on the "Loading accounts..." placeholder when /api/equity throws.
- Reads: /api/status (already polled), /api/equity (already polled). Writes: none. No backend change.
- No callers outside this file; the strip IDs (pnl-day/pnl-open/pnl-trades) have zero existing JS references (confirmed via grep). bot_dashboard.py untouched.
- Verification: bot_dashboard pytests + live Playwright screenshot (STOPPED idle state).
