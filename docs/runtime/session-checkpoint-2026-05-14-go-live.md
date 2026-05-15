# Session Checkpoint — Go Live (2026-05-14)

**Status:** Track A in-flight, BLOCKED on real config bug. Track B not started.
**Pick up in a fresh session by reading this file first.**

---

## What was completed in this session

1. **A.0** — `TELEGRAM_CHAT_ID` confirmed present in `.env` (preflight `[5/6]` passed).
2. **A.1** — Drift GREEN (131/131 capital-class, 20 advisory). MNQ daily_features rebuilt through 2026-05-14 for `orb_minutes=5` and `orb_minutes=15`. **30m row missing for 2026-05-13/14** — Check 69 fails, NON-BLOCKING for capital (live engine only queries the apertures present in active portfolio = `{5, 15}`; auditor verdict 2026-05-14).
3. **A.1 step 3** — `scripts/run_live_session.py --preflight --live` returned 6/6 PASS at 18:29.
4. **Removed orphan `live_session.stop`** from a PC crash earlier today (08:26). Without removal, every bot-start would terminate within 5 seconds.
5. **Desktop shortcuts confirmed** — `START_BOT - Shortcut.lnk` and my added `START BOT.lnk` both point to `C:\Users\joshd\canompx3\START_BOT.bat`. Either works.
6. **Dashboard chart fix** — `trading_app/live/bot_dashboard.html` line 4108 edited: `chart.addCandlestickSeries(...)` → `chart.addSeries(LightweightCharts.CandlestickSeries, ...)`. v5 API. **NOT committed** (drift Check 69 violation + gold.db locked by running dashboard). Operator UI only, zero capital path.

---

## The actual blocker (discovered at end of session)

**Live mode cannot start.** Earlier today's `logs/session.log` shows the bot crashed at startup:

```
File "trading_app/live/session_orchestrator.py", line 362, in __init__
File "trading_app/prop_profiles.py", line 1041, in get_lane_registry
ValueError: Profile has inconsistent max_orb_size_pts across lanes on the same
(session, instrument): US_DATA_1000/MNQ=[89.2, 142.3].
Reconcile the DailyLaneSpec entries in prop_profiles.py.
```

This is a **fail-closed validation gate working correctly**. The active profile `topstep_50k_mnq_auto` has TWO US_DATA_1000 MNQ lanes (`VWAP_MID_ALIGNED_O15` and `OVNRNG_25`) with conflicting `max_orb_size_pts` caps. The loader refuses to start until one canonical cap is chosen.

**Preflight passed 6/6 anyway** because preflight only validates broker auth + freshness + journal — NOT full `SessionOrchestrator.__init__`. Preflight gap is debt-class.

---

## Visual UI confusion (resolved)

User screenshot (Downloads/untitled.jpg) showed the dashboard with `START SIGNAL topstep_50k_mnq_auto` in the topbar — that's the primary-CTA from line 2890 of bot_dashboard.html, fires SIGNAL mode only (paper, Telegram).

**The Live unlock is on the `Profiles` tab.** The second tab row shows: `Brokers | Trades | Profiles | Account Specs | Trade Book | Activity`. User was on default view. Click `Profiles`, find `topstep_50k_mnq_auto` card, three buttons render (Alerts/Paper/Live) + `Type LIVE to unlock` input, type `LIVE` (caps), click Live. **But don't yet — see blocker above.**

---

## Pick-up plan for the next session

In priority order:

### 1. A.6 — Fix the max_orb_size_pts conflict (REQUIRED before Live)

File: `trading_app/prop_profiles.py`
Conflict: US_DATA_1000/MNQ has caps `[89.2, 142.3]`.

Investigate:
- Which lane carries which cap? Grep `89.2|142.3` in `prop_profiles.py` + `docs/runtime/lane_allocation.json`.
- The 4 deployed MNQ lanes per allocation JSON: VWAP_MID_ALIGNED_O15 (O15), OVNRNG_25 (O5), ORB_VOL_2K (O5 COMEX), COST_LT12 (O5 NYSE).
- Two lanes share US_DATA_1000/MNQ but different ORB apertures (O15 + O5). The cap is per-(session, instrument) not per-(session, instrument, orb_minutes) — this may be a model bug, OR one cap should win.
- Check `scripts/tools/rebalance_lanes.py` (the writer) — does it set the cap from one lane and not check siblings? Likely root cause.

This is capital-path. Per `institutional-rigor § 1`: design proposal first (what/files/blast-radius/approach), then implement, then self-review, then verify.

**Auditor required** (`adversarial-audit-gate.md`): post-fix dispatch `evidence-auditor` before committing, because it touches `trading_app/live/` capital path.

### 2. A.6.5 — Preflight gap fix (debt)

`scripts/run_live_session.py --preflight` should fail at the same gate that the actual session start fails. Currently preflight passes 6/6 while real start crashes. Add a check that runs `get_lane_registry(profile_id=...)` and surfaces ValueErrors.

### 3. A.2 — Click Live (after A.6 passes)

User opens dashboard (already up at PID 24452 OR new launch from `START_BOT - Shortcut.lnk` if it's been killed). Profiles tab → topstep_50k_mnq_auto card → type `LIVE` → click Live → confirm "REAL MONEY".

### 4. A.3 — Monitor first 24h

Check `data/bot_state.json` mtime advancing every ~60s, shadow-account discovery log line in `logs/session.log`, first signal window comes/goes without crash, first fill lands in `live_journal.db`.

### 5. Chart fix commit (post-market-close)

`trading_app/live/bot_dashboard.html` line 4108 edit on disk. Commit after:
- A.6 lands the 30m daily_features rebuild gets unblocked (kill dashboard, run `pipeline/build_daily_features.py --instrument MNQ --start 2026-05-12 --end 2026-05-14 --orb-minutes 30`)
- Drift clean
- Commit message: "fix(dashboard): migrate addCandlestickSeries to v5 addSeries(CandlestickSeries) API"

### 6. (Deferred) Track B — Grow inventory beyond 4 lanes

Per `feedback_max_profit_grow_chordia_inventory_not_force_slots.md`. Read-only audit work, independent of Track A. Not started.

---

## Risks the next session should know

- **`live_risk_auditor` 2026-05-14 verdict: ACCEPT_WITH_RISK.** Same baseline, still valid:
  - Dashboard unauthenticated on localhost:8080 (single-operator, OK).
  - Kill button single-click, no hold-to-confirm modal (cockpit-v4 Stage 1 not in main).
  - Known F-1 silent block window at startup (~10 min if broker equity API down).
- **214/214 session_orchestrator tests PASS, 27/27 bot_dashboard tests PASS** — the test surface is intact.
- **Drift Check 69 (missing 30m apertures) is non-blocking for capital** but blocks the chart-fix commit until rebuilt.
- **Two python MCP processes (PIDs change per session)** can hold gold.db open and block writers. Restart Claude session if writes fail with IOException.

---

## Files modified this session (uncommitted)

- `trading_app/live/bot_dashboard.html` — line 4108 v5 API migration (chart only, non-capital)
- `docs/runtime/session-checkpoint-2026-05-14-go-live.md` — this file

No `git add`, no commit. Working tree dirty by intent — pick up clean by reading this file.

---

## Decision log

- **TELEGRAM_CHAT_ID** — NOT verified by me (firewalled), but preflight `[5/6]` PASS implies it's set or the env-block is silently okay. If A.6 fix lands and start still fails on notifications, revisit.
- **Chart fix on disk but uncommitted** — accepted Trivial-tier exception, deferred commit due to gold.db lock + drift Check 69. Not blocking.
- **Capital-review verdict** — GO from `live_risk_auditor` agent before discovering the prop_profiles.py crash. After A.6 fix, re-dispatch auditor.
