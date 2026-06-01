# Stale-session GO LIVE latch fix

task: Fix dashboard hiding GO LIVE/Alerts/Demo controls because a dead-dirty
  signal session leaves bot_state.mode=SIGNAL, latching client runningProfile
  forever. Add authoritative server is_running + client mirror + conservative
  self-heal of dead state.

mode: IMPLEMENTATION

## Scope Lock
- trading_app/live/bot_dashboard.py
- trading_app/live/bot_dashboard.html
- tests/test_trading_app/test_bot_dashboard.py

## Blast Radius
- bot_dashboard.py /api/status (line ~1522): add authoritative `is_running`
  field derived from _session_snapshot() (reuses canonical 120s rule). Also add
  conservative self-heal: when session detected dead (tracked_alive=False) AND
  heartbeat age > 300s ("definitely dead" — SAME threshold as the existing
  lifespan startup cleaner at line ~108), call clear_state() so the stale
  bot_state is actively removed, not just hidden. Reads data/bot_state.json;
  writes: clear_state() unlinks it only in the definitely-dead case.
- bot_dashboard.html updateStatus (line ~4372): clear runningProfile when
  is_running===false (server-authoritative) OR mode==="STOPPED" (back-compat).
  Reuses existing state.heartbeat_age_s / STALE_AFTER_S=120 already in client.
- CAPITAL SAFETY: self-heal gated on 300s (definitely-dead) NOT 120s, AND on
  tracked_alive=False, so a briefly-unresponsive LIVE session (GC pause, slow
  tick at 121-299s) is NEVER cleared — only its launch buttons stay hidden.
  Button-visibility (is_running) uses 120s; state-destruction uses 300s. Two
  thresholds, two risk profiles. No live-launch LOGIC touched (LIVE_PILOT_PROFILE
  lock, strict-zero-warn, preflight all untouched) — only control VISIBILITY.
- tests: regression for (1) /api/status is_running=false on stale, (2) self-heal
  clears state >300s dead but NOT at 200s, (3) is_running=true when alive.
