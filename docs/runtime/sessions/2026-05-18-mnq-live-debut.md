# 2026-05-18 — MNQ live debut (Monday)

**Profile:** `topstep_50k_mnq_auto`
**Option path:** A (two TopstepX accounts confirmed live)
**Branch / HEAD at runbook start:** `main` @ `b2948a82`
**Drift state at runbook start:** 133/133 passed
**Author:** Terminal 1 (writer)

---

## Account inventory (verified 2026-05-18 09:34 BNE via `contracts.resolve_all_account_ids()`)

- **Primary:** id `21944866` / `EXPRESS-V2-451890-53179846` (Express)
- **Shadow:** id `23055112` / `50KTC-V2-451890-29512053` (Standard 50K, no broker-side daily-loss limit)
- **Profile.copies:** 2 (matches discovered count)
- **`_select_primary_and_shadow_accounts` dry-run:** primary=`21944866`, shadows=`[23055112]` — PASS

## Account asymmetry — recorded risk

The Express primary carries TopstepX's standard daily-loss enforcement (~$1,000 on 50K Express). The Standard 50K shadow has **no broker-side daily-loss limit**. Local 1 R hard kill (Kill #1 below) is the ONLY stop on the Standard shadow tonight. Honor the 1 R cutoff before either broker engages — ~$200 well below the $1,000 Express daily-loss floor.

**Carry-over for next session (not blocking tonight):** verify `copy_order_router.py` halts shadow accounts when primary is broker-side-closed. Stage candidate post-debut.

---

## Pre-commit hard kill conditions (any ONE → Ctrl+C + manual flatten BOTH accounts)

These are the receipts. If I'm tempted to relax mid-session, the receipt overrides the temptation.

1. **1 R loss on day** — across all 3 lanes combined, summed in R units. Bounds the asymmetric-shadow exposure.
2. **Any `RuntimeError` / `AssertionError` / `KeyError` / `ConnectionError`** in stdout or `logs/live/live_<ts>.log`.
3. **Broker connection drop > 2 min** — no `tick` log line or quote update.
4. **Position state mismatch** — any open position size / direction differs from portfolio expectation; orphaned bracket leg.
5. **`risk_manager.can_enter → (False, <non-obvious reason>)`** — anything other than `daily_trade_count` or `concurrent_positions` indicates a gate misfiring.
6. **`kill_switch` not armed within 60s of session start** — check `session_orchestrator._fire_kill_switch` reachability in the startup log.

## Soft watches (log + reassess, do not auto-kill)

- F-1 silent block notification fires unexpectedly.
- HWM tracker `hwm_dollars` still `0.0` after broker activity (carry-over from 2026-05-16 debut).
- Dashboard chart panel empty despite live ticks (carry-over c-i).
- `live_<ts>.log` not written to disk (carry-over c-ii).

---

## Deployed lanes tonight (3 — verified `lane_allocation.json` rebalance 2026-05-18)

1. `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` — ExpR 0.2159, Annual R 30.9, HOT
2. `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` — ExpR 0.2416, Annual R 27.1, HOT
3. `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` — ExpR 0.0951, Annual R 21.9, HOT

Note: preflight portfolio-load reported `ExpR=0.087 N=1508` for NYSE_OPEN (recent regenerate), vs `lane_allocation.json` snapshot `0.0951`. Same lane — different snapshot windows. Tracking under lane_allocation.json verdict.

---

## Step status (filled live)

- [x] Step 0 — Sanity verify: PASS (b2948a82, 133/133 drift)
- [x] Step 1 — Option A confirmed (2 TopstepX accounts live)
- [x] Step 2 — Preflight signal-only 8/8 PASS + direct account-resolution dry-run PASS
- [x] Step 3 — Dashboard launched via `START_BOT.bat`; "Start Live" UI gate passed; bot subprocess ran real preflight
- [x] Step 4 — Pre-commit hard kills written (this file)
- [BLOCKED] Step 5 — `--live` arm: bot's own real preflight returned 7/8 with Step 7 FAIL (telemetry maturity 11/30). Bot did NOT start. Two additional governance signals surfaced — see § Governance gates below.
- [ ] Step 6 — Post-launch handoff (no live trades fired tonight)

---

## Governance gates surfaced at Step 5 — DIAGNOSIS

### Gate 1 (BLOCKING) — Telemetry maturity (Criterion 8 / RULE 3.2-3.3)

- `_check_telemetry_maturity` returned `FAILED: UNVERIFIED_INSUFFICIENT_TELEMETRY (11/30 distinct MNQ trading_days; run --signal-only until 30)` in live mode.
- Same gate auto-passes in `--signal-only` mode (line 369-378 of `run_live_session.py`); fails-closed in capital-touching modes per docstring.
- Track A path: run `--signal-only` tonight + subsequent sessions until n≥30. ETA ≈ 4 calendar weeks at MNQ cadence.

### Gate 2 (BLOCKING for L3 NYSE_OPEN_COST_LT12) — Persisted SR-monitor pause

- `lifecycle_state.read_lifecycle_state` shows L3 `block_source: pause`, `pause_source: sr_monitor`, `since: 2026-04-23`, `expires: 2026-05-23` (5 days from now).
- `sr_review_registry.SR_ALARM_REVIEWS` has a 2026-05-17 `watch` review entry for L3 with detailed regime-audit + OOS-by-direction analysis (R5 ExpR +0.128 N=257, R6 forward holdout +0.082 N=80, OOS long +0.094, OOS short +0.072 — no directional sign-flip).
- The pause record was written 2026-04-22 with `apply_pauses=True` and is in the `paused_strategy_ids` set. `lifecycle_state.py:233` short-circuits on this table BEFORE reaching the sr_review_outcome=watch branch. The recent watch review cannot override the persisted pause.
- This matches memory `feedback_sr_monitor_peak_vs_current_misread.md`: "--apply-pauses shadows the sr_review_registry watch outcomes via lifecycle_state.py:232 short-circuit."
- Path forward: (a) wait until 2026-05-23 expiry, OR (b) clear the pause record via a deliberate stage with audit trail (do NOT silent-mutate).

### Gate 3 (CLEAR — false alarm from terminal header) — SR alarms are stale peak values

- Terminal output showed all 3 lanes in `ALARM` with `sr_stat` of 35.39, 41.60, 33.27 (above threshold 31.96).
- **`sr_stat` is the TRIGGER (peak) value, NOT current SR.** This is the memory-recorded UX misread (`feedback_sr_monitor_peak_vs_current_misread.md`).
- Direct `sr_state.json` read of `current_sr_stat`:
  - L1 COMEX_SETTLE OVNRNG_100: current=**11.54** (peak 35.39 at trade 24, 44 trades since), recent_10_mean_r=0.187 vs validated 0.215
  - L2 US_DATA_1000 VWAP_MID_ALIGNED_O15: current=**0.53** (peak 41.60 at trade 6, 34 trades since), recent_10_mean_r=0.470 vs validated 0.210 (running HOT)
  - L3 NYSE_OPEN COST_LT12: current=**3.26** (peak 33.27 at trade 4, 19 trades since), recent_10_mean_r=0.178 vs validated 0.180
- All 3 current SRs are well below threshold. No real-time mean-shift signal. The 3-alarm terminal display is historical noise, not a portfolio-wide decay event.
- L1 and L2 watch reviews (2026-04-12, 2026-04-14) DO unblock those lanes correctly — they have no persisted pause record. Only L3 has the persisted-pause shadowing problem.

---

## Verdict

**No live trades fired tonight. No capital risk incurred. No code modifications made.**
Two real institutional gates blocked the launch; one was a stale UX display, two are legitimate fail-closed behaviors. The bot is doing exactly what it's supposed to do.

Pre-commit hard kills (Step 4 above) were never tested under live conditions tonight — carry them forward unchanged for the next attempt.

---

## Carry-over evidence capture (during Step 5)

At T+3 min after first tick (Terminal 2):

```bash
date -u +%FT%TZ
ls -la C:/Users/joshd/canompx3/logs/live/ 2>/dev/null | tail -5
curl -s http://localhost:8088/api/bars-recent?instrument=MNQ | head -c 500
echo
ls -la C:/Users/joshd/canompx3/data/state/account_hwm_*.json 2>/dev/null
```

Paste full results below (c-i bars-recent body, c-ii live.log presence, HWM-populated status).

---

## Post-session findings (filled at Step 6)

_pending_
