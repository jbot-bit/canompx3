---
task: |
  STAGE 4 of 5 — Live go-live plan (`~/.claude/plans/get-live-trading-working-safe-secure.md`).
  Prove the `--demo` order-path end-to-end against tonight's NYSE_OPEN window
  on `topstep_50k_mnq_auto` (TopstepX practice endpoint, zero real capital).
  Stages 0-3 SHIPPED: Stage 0 cp1252 crash fix pushed; Stage 1 $450/account
  daily-loss breaker (89f8e97b re-audit PASS); Stage 2 SR-alarm strict
  doctrine LOCKED (5b2e00d9); Stage 3 bar-ring iter2 (a7d0e4c5+6d5c248b,
  CONDITIONAL audit cleared). Last `--demo` attempt 2026-05-27 was rolled
  back (commit b8937962) after a stale-tab "Failed to fetch" dashboard event;
  the order path was NEVER proven across a full trade cycle. Tonight's
  smoke fills that gap. After this stage closes green, Stage 5 (`--live`
  flip) becomes operator-attended-only.
mode: IMPLEMENTATION
scope_lock:
  - START_BOT.bat
  - docs/runtime/stages/2026-05-28-live-golive-stage4-demo-smoke.md
  - data/live_bars/MNQ.json  # pre-smoke cleanup of 2026-05-27 stale residue
  - HANDOFF.md  # baton update post-smoke
implementation_status: IMPLEMENTATION

# ──────────────────────────────────────────────────────────────────────────
# WHY THIS STAGE NOW
# ──────────────────────────────────────────────────────────────────────────
# - 4 deployed MNQ lanes on topstep_50k_mnq_auto fire across the US block:
#     * MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12  (23:30 Brisbane)
#     * MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15  (00:00 Fri Bris)
#     * MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100  (03:25 Fri Brisbane)
#   (4th: MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT08 already passed at 10:00 today)
# - Three lane opportunities in ~5h gives 3x the chance to fire a real
#   bracket order on the TopstepX practice endpoint inside one operator-
#   attended session.
# - $0 capital at risk: --demo routes to TopstepX practice; AccountHWMTracker
#   still active but balance is paper. $2K MLL + $450 daily-loss breaker run
#   identically to --live — flushes plumbing without exposure.

# ──────────────────────────────────────────────────────────────────────────
# DONE CRITERIA (all required)
# ──────────────────────────────────────────────────────────────────────────
# 1. At least ONE order submitted → filled → bracket attached → exit reached
#    (TP/SL/scratch — any) at TopstepX practice endpoint. Logged in
#    `logs/live/live_<ts>.log` with order-id, fill-id, exit-id reconciliation.
# 2. Ring lifecycle observed: ring file ABSENT at session start →
#    `data/live_bars/MNQ.json` created with fresh ts within 90s of feed
#    connect → ring updates as bars stream → ring deleted post-graceful-Ctrl+C
#    (`MNQ.shutdown_trace.txt` shows `ring_cleared:flushed` line).
# 3. Graceful Ctrl+C exercises `post_session` hook end-to-end:
#    `post_session:entry` → `drain_ok` → `flush_attempt:bars_captured=N` →
#    `flush_returned:n_persisted=N` → `ring_cleared:flushed`. NOT a hard
#    taskkill — that bypasses the hook and re-creates the 2026-05-27 evidence
#    gap (memory: project_live_golive_demo_session_inflight_2026_05_27).
# 4. `live_readiness_report.py --strict-zero-warn` re-run post-smoke; capture
#    delta vs pre-smoke baseline. Telemetry day count should increment by 1.
# 5. No CRITICAL alerts in `logs/live/live_<ts>.log` (search for
#    `ALERT|MANUAL CLOSE|broker_unreachable|HWM breach|emergency_flatten`).
# 6. Dashboard `/api/bars-recent` returns non-empty bars during session
#    (closes c-i carry-over from `feedback_duckdb_windows_lock_is_per_process.md`).
# 7. `logs/live/*.log` FileHandler gap (c-ii) observed: confirm log file
#    EXISTS during the session, not just at end.
# 8. ZERO real capital affected. Verify `broker.signal_only:false,
#    is_practice:true` in `/api/status` before allowing any trade.

# ──────────────────────────────────────────────────────────────────────────
# EXECUTION ORDERING (pre-market → market → post-market)
# ──────────────────────────────────────────────────────────────────────────

## Pre-market (between now and 22:30 Brisbane — ~9.5h)
# 1. Wait for peer Claude DB lock to release. Poll: `python -c "import duckdb;
#    duckdb.connect(r'C:\Users\joshd\canompx3\gold.db', read_only=True).close()"`
#    Once clears, run baseline `live_readiness_report.py --strict-zero-warn`,
#    capture output to `docs/audit/results/2026-05-28-stage4-prebaseline.txt`.
# 2. Run `python scripts/tools/refresh_control_state.py --profile
#    topstep_50k_mnq_auto` to clear pulse's "BLOCKED: no Criterion 11 survival
#    report" finding. This is the same gold.db that the smoke uses — must
#    succeed cleanly before any --demo run.
# 3. Delete stale residue: `rm data/live_bars/MNQ.json` + the matching
#    `MNQ.shutdown_trace.txt` (2026-05-27 unrecoverable fragment per
#    `recover_ring.py --dry-run`). Verifies ring lifecycle starts from ABSENT.
# 4. Edit START_BOT.bat:41: `set BOT_MODE_FLAGS=--signal-only` →
#    `set BOT_MODE_FLAGS=--demo`. ONE-LINE diff. Commit as
#    `chore(live): flip START_BOT.bat to --demo for Stage 4 NYSE_OPEN smoke`.
# 5. Operator pre-flight checklist (in chat, before bot launch):
#    - START_BOT.bat shows --demo
#    - `data/live_bars/` is empty
#    - peer DB lock clear (gold.db opens read-only)
#    - dashboard tab not open in stale browser session (close + relaunch)

## Market (22:30 Brisbane → 04:00 Fri Brisbane)
# 6. Operator launches START_BOT.bat at 22:30 Brisbane (60min pre-NYSE_OPEN
#    warm-up gives the bar-feed time to populate the ring + lets preflight
#    settle). NOT earlier — see lessons in `feedback_app_style_no_ops_menus.md`
#    (no operator-overhead ceremony pre-market).
# 7. Operator confirms in dashboard: `/api/status` shows `broker.signal_only:
#    false, broker.is_practice: true, feed_status: healthy, kill_switch:
#    armed, daily_loss_dollars_remaining: 450`.
# 8. Through NYSE_OPEN (23:30): observe bar-ring count climb, chart updates
#    live. If lane fires: order-id appears in `bot_state.json`, dashboard
#    lane card flips to IN_TRADE, bracket SL/TP visible in TopstepX practice
#    portal.
# 9. Through US_DATA_1000 (00:00 Fri) + COMEX_SETTLE (03:25 Fri): same
#    observation pattern. ANY single complete trade across the three windows
#    satisfies done-criterion 1.
# 10. Once at least one trade cycle complete (or after COMEX_SETTLE +1h if
#     none fired — instrument fitness can shut all lanes for the night):
#     Operator presses Ctrl+C in the orchestrator terminal (NOT taskkill,
#     NOT close-window — those bypass post_session). Observe shutdown_trace
#     gets written with all 5 expected lines.

## Post-market (04:00 Fri)
# 11. Re-run `live_readiness_report.py --strict-zero-warn`, capture to
#     `docs/audit/results/2026-05-28-stage4-postsmoke.txt`. Diff baseline vs
#     postsmoke — telemetry day should be +1.
# 12. Inspect `logs/live/live_<ts>.log` for criterion 5 (no CRITICAL).
#     Inspect `data/live_bars/` — should be empty (ring cleared).
# 13. Commit smoke result + close stage:
#     - HANDOFF.md updated with smoke outcome summary
#     - This stage file: mode → CLOSED, closed_note with trade count + audit
#     - Adversarial-audit gate dispatched per
#       `.claude/rules/adversarial-audit-gate.md` (HIGH severity — live path)
# 14. If green: revert START_BOT.bat to --signal-only (safe-default) UNTIL
#     Stage 5 explicitly authorizes the --live flip. Stage 5 is operator-
#     attended-only.
# 15. If RED (no trade fired across all 3 windows AND lanes were healthy):
#     This is NOT a Stage 4 failure — it's market regime. Reschedule one more
#     attempt next active session window. Two consecutive failures-to-fire on
#     healthy lanes triggers investigation (lane-fitness silent block? entry
#     model bug?).

# ──────────────────────────────────────────────────────────────────────────
# RISK REGISTER
# ──────────────────────────────────────────────────────────────────────────
# - HIGHEST: operator hard-kills the bot instead of graceful Ctrl+C, bypassing
#   post_session (same evidence gap as 2026-05-27). Mitigation: explicit
#   pre-launch reminder + criterion 3 makes graceful shutdown done-criterion.
# - HIGH: TopstepX practice endpoint disconnect mid-trade. Mitigation: per
#   `project_gap1_autobracket_disconnect_tif_confirmed_2026_05_27.md`,
#   SL/TP legs are server-side native at TopstepX, survive bot disconnect.
#   `_emergency_flatten` triggers on >300s feed death.
# - MEDIUM: peer Claude session holds gold.db lock during operator launch
#   window (22:30). Mitigation: pre-flight checklist item 5 (verify lock free)
#   blocks launch.
# - MEDIUM: dashboard stale-tab false-fetch causes operator to think bot is
#   down. Mitigation: pre-flight checklist item 5 forces dashboard re-launch.
#   STALE banner (commit 2503d4e9) catches this state.
# - LOW: TOKYO_OPEN window already passed today (10:00 — 3h ago) means we
#   only get 3 lane opportunities not 4. Mitigation: 3 windows is sufficient
#   for done-criterion 1; we don't need all 4 to fire.

# ──────────────────────────────────────────────────────────────────────────
# WHAT THIS STAGE WILL NOT DO
# ──────────────────────────────────────────────────────────────────────────
# - NOT flip START_BOT.bat to --live. That is Stage 5, explicitly operator-
#   gated, requires this stage GREEN first.
# - NOT touch lane_allocation.json. Lanes are the canonical 4. No new edges.
# - NOT mutate any production code. Only START_BOT.bat (one line) +
#   data/live_bars/ cleanup + docs.
# - NOT widen scope to multi-instrument or shadow accounts. topstep_50k_mnq_
#   auto current-4-lanes only.
# - NOT skip the graceful Ctrl+C. Hard-kill bypasses the validation this
#   entire stage exists to perform.

# ──────────────────────────────────────────────────────────────────────────
# CROSS-REFERENCES
# ──────────────────────────────────────────────────────────────────────────
# - Plan: ~/.claude/plans/get-live-trading-working-safe-secure.md (Stage 4)
# - Stage 3 close-out: docs/runtime/stages/2026-05-22-live-bar-ring-chart.md
# - 2026-05-27 prior attempt: memory/project_live_golive_demo_session_inflight_2026_05_27.md
# - Disconnect-TIF doctrine: memory/project_gap1_autobracket_disconnect_tif_confirmed_2026_05_27.md
# - Kill-switch race fix: memory/project_gap2_killswitch_post_session_broker_truth_2026_05_27.md
# - Shared-state hygiene (peer DB lock): .claude/rules/multi-terminal-shared-file-hygiene.md
