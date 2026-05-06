# Live Trading Rollout — Phased Plan

**Owner:** Josh
**Created:** 2026-05-05
**Status:** AWAITING_PHASE_0_APPROVAL
**Branch context:** `feat/openrouter-research-followups` (== `origin/main` at `58e92894`)
**Account context:** TopstepX user `joshdlees@gmail.com`, two accounts visible:
- `21390438` `50KTC-V2-451890-67605663` (50K Trading Combine V2 — eval, $50K simulated capital, real P&L tracked by TopStep)
- `21944866` `EXPRESS-V2-451890-53179846` (Express — free practice, $0 funded, no scaling)

---

## Why this plan exists

User has been pigeonholed in research/audit cycles for weeks. The actual blocker to live trading
turned out to be *not* statistical (we have validated lanes), *not* environmental (creds are in
`data/broker_connections.json` and the auth handshake works — see
`scripts/tools/broker_handshake_check.py`), but **operational**: nothing has run end-to-end against
the live broker pipe with live bars and the live allocator simultaneously.

This plan eliminates that gap with a **gated, literature-anchored, capital-aware sequence**. No
phase advances until its exit gate is met in writing. No phase invents new research; we are
**operationalizing** an already-validated portfolio (`topstep_50k_mnq_auto`).

## Anti-pigeonhole rules baked in

1. **Truth re-query at every phase boundary** — never trust the memory index across a phase gate.
   Authority hierarchy per `.claude/rules/integrity-guardian.md` § 1.
2. **Each phase has a literature anchor** in `docs/institutional/literature/`. No anchor → no phase.
3. **Each phase has an explicit halt condition.** "Keep going" is not the default.
4. **F-1 behavior is empirically verified, not assumed.** The TC-detection logic in
   `trading_app/live/session_orchestrator.py:43` (`_is_trading_combine_account`) auto-disables F-1
   for the TC account name pattern, but that disable must be confirmed in a real log line, not
   inferred from reading the code (per `.claude/rules/integrity-guardian.md` § 7 — never trust
   metadata, always verify).
5. **No phase claims "done" until tests pass + drift passes + dead code swept + self-review
   passed** (per `.claude/rules/institutional-rigor.md` § 8).

---

## Account model — what's actually true

The TopstepX/ProjectX API exposes one user → many accounts via a single API key. Both accounts
above are visible to the running handshake. **Important consequences:**

- The 50K TC is a **real live-broker account** with real fills, real bid/ask, real slippage.
  P&L is simulated against TopStep's books, but the order pipe and market microstructure are real.
- The Express is a free practice account on the same API. Useful as a shadow/divergence-detector
  or for low-stakes debugging, not as a primary.
- F-1 (TopStep XFA scaling ladder) does **not** apply to either account by design — it activates
  only on Funded (`XFA-...` / `EFA-...` named) accounts, which we don't have yet. F-1 in the
  current state is dormant for our flow; this is correct, not a bug.

Authority: `trading_app/live/session_orchestrator.py:43-90` (`_is_trading_combine_account` and
`_resolve_topstep_xfa_account_size`).

## Validated portfolio context (re-query before each phase, do not cite from memory)

Profile: `topstep_50k_mnq_auto` (single instrument MNQ).
3 strategies (preflight 2026-05-05 21:52 confirms):

| strategy_id | session | RR | aperture | N | WR | ExpR |
|---|---|---|---|---|---|---|
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100` | COMEX_SETTLE | 1.5 | O5 | 513 | 52% | +0.215 |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15` | US_DATA_1000 | 1.5 | O15 | 701 | 50% | +0.210 |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | NYSE_OPEN | 1.0 | O5 | 1508 | 56% | +0.087 |

Stop multiplier: 0.75x (prop-tight).
DD budget: $300 / $2000 (15%).

Allocator (rebal 2026-05-03, `docs/runtime/lane_allocation.json`):
- `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` is currently **lifecycle-blocked** (SR alarm
  stat=33.27 ≥ thr=31.96 on `paper_trades_first_50` baseline). 2 of 3 are unblocked.

---

## Phase 0 — Truth foundation (no broker writes, no capital risk)

**Goal:** every input the live runner consumes today is fresh, canonical, and verified.

**Wall-clock budget:** ~30 min.

**Exit gate (all four must pass, written into this file before Phase 1):**

- [ ] **0.1** `daily_features` refreshed to today's Brisbane trading day. Currently 4 days stale
      (latest 2026-05-01 per preflight 2026-05-05 21:52). Run
      `python pipeline/build_daily_features.py` and confirm row count + latest date.
- [ ] **0.2** `python pipeline/check_drift.py` exits 0 with all checks passing. Show output.
- [ ] **0.3** `docs/runtime/lane_allocation.json` rebal date re-read. If newer than 2026-05-03,
      re-run preflight to confirm the 3-lane portfolio still stands. If older, no action.
- [ ] **0.4** `python scripts/tools/broker_handshake_check.py` returns rc=0 with both accounts
      and MNQ front-month resolved. (Idempotent, safe to re-run.)

**Literature anchor:** López de Prado 2020, *ML for Asset Managers*, Ch1 — backtest-overfitting and
data quality. Stale features in live = silent backtest/live divergence. Cite
`docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md`.

**Halt conditions:**

- Drift check fails on anything other than advisories → stop, fix root cause, re-gate.
- daily_features rebuild produces fewer rows or older latest-date than current → stop, investigate
  pipeline before any session.
- Allocator JSON has rotated to a new lane set → re-run preflight, update this file's portfolio
  table, do not silently proceed on stale lane assumptions
  (`feedback_closeout_verify_against_canonical.md`).

---

## Phase 1 — Signal-only shake-down (zero capital risk, real broker pipe)

**Goal:** prove bars flow, ORB windows compute correctly, signals fire on real market data,
the orchestrator hot path raises zero unhandled exceptions across one full Brisbane trading day.

**Wall-clock budget:** 1 trading day (≈24h calendar) for first run; signal-only is safe to run
indefinitely.

**Command:**

```bash
python scripts/run_live_session.py --profile topstep_50k_mnq_auto --signal-only
```

This:
- Auto-launches the dashboard (`trading_app/live/bot_dashboard.py`).
- Spawns the bar feed against TopstepX SignalR market hub.
- Constructs `SessionOrchestrator` with `signal_only=True, demo=True` — auth is required (for the
  feed), orders are NOT placed.
- Logs each ORB window open/close, each break, each entry signal.

**Exit gate (all five must pass, written into this file before Phase 2):**

- [ ] **1.1** ≥1 full Brisbane trading day of clean run (no orchestrator-level exceptions).
- [ ] **1.2** ≥1 expected signal logged from at least one of the three lanes. Specifically: a
      `⚡ SIGNAL` line in the log with the strategy_id, direction, and entry_price.
- [ ] **1.3** F-1 TC-detection log line confirmed: a line containing
      `Trading Combine marker in account name` after F-1 evaluation. If F-1 stays armed against the
      TC (no TC detection log), HALT — this is a real bug not anticipated by reading the code.
- [ ] **1.4** Bar-aggregator did not drop bars: zero `on_stale` callbacks fired in the log over the
      session window. Source of truth: `trading_app/live/broker_base.py:46`.
- [ ] **1.5** `bot_state.json` and the dashboard reflect realistic last-bar timestamps within
      ≤2 min of wall-clock end of session.

**Literature anchor:** Chan 2013, *Algorithmic Trading*, Ch1 p.4 — look-ahead bias and live-vs-
backtest divergence. The most common live failure mode in this literature class is data-quality /
timing drift, not strategy decay.
`docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md`.

**Halt conditions:**

- Any unhandled exception in `trading_app.live.*` → stop, capture stack trace, fix before retry.
- ≥1 `on_stale` event without recovery → stop, investigate feed reconnect logic before any orders.
- `bot_state.json` shows last-bar timestamp diverging from wall-clock by >5 min → stop, the feed is
  broken.
- Signals fire at obviously-wrong prices (>±5% from contemporaneous bar close) → stop, the bar
  pipeline or break-detector is bugged.

---

## Phase 2 — TC live orders (real fills against the 50K Combine, real microstructure exposure)

**Goal:** prove the broker round-trip — entry submit, bracket placement (stop + target), fill
confirm, exit submit, position reconciliation. Capital exposure: $50K simulated TC balance with
real P&L tracking by TopStep.

**Wall-clock budget:** ≥3 round-trip trades (probably 2-5 trading days at 0-2 trades/day).

**Decision required from user before this phase:**

1. **Which lane traded first?** Recommendation: `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`
   (N=513, ExpR=+0.215, COMEX_SETTLE 06:00-07:00 Brisbane). Rationale: thin-book session = strict
   slippage stress test; smallest validated cohort = most conservative DSR posture (Bailey-LdP DSR
   penalises larger trial counts but rewards prior-strong evidence — N=513 is well above
   Chordia/Harvey-Liu single-strategy floors).
2. **`--demo` vs `--live`?** TopstepX does not expose a separate demo endpoint distinct from the
   TC; for our infra, `--demo` and `--live` differ only in the local CONFIRM prompt. Effective
   capital exposure on the TC is identical. Recommendation: skip `--demo`, go straight to
   `--live --account-id 21390438` with `--auto-confirm` OFF the first time.
3. **Single contract per signal?** Yes — never scale up before reconciliation has been verified.
   `prop_profiles.ACCOUNT_PROFILES["topstep_50k_mnq_auto"]` controls this; trust the profile.

**Command (after user signs off):**

```bash
python scripts/run_live_session.py \
    --profile topstep_50k_mnq_auto \
    --instrument MNQ \
    --live \
    --account-id 21390438
```

(No `--auto-confirm` first time — the operator must type CONFIRM at the prompt.)

**Exit gate (all six must pass, written into this file before Phase 3):**

- [ ] **2.1** ≥3 distinct round-trip trades on the TC (entry→exit, broker-confirmed fills).
- [ ] **2.2** Zero orphan brackets after each trade closes
      (`trading_app/live/session_orchestrator.py` should clear stop+target on exit).
- [ ] **2.3** Zero broker-vs-bot position drift across the 3+ trades. Reconcile via:
      - `trading_app/live/projectx/positions.py:query_open()` post-trade
      - bot's internal position state in `bot_state.json`
      - TopstepX UI / web positions page
- [ ] **2.4** `scripts/trade_book.py` includes the new trades and matches broker history exactly.
- [ ] **2.5** No `is_degraded()` events on the router (single-account = always False, but verify
      in the log we never spuriously flag).
- [ ] **2.6** Slippage and cost realised are within 1.5x the modelled `COST_SPECS["MNQ"]`
      friction. If realised slippage > 1.5x model, halt and audit cost model before further trades
      (per `.claude/rules/institutional-rigor.md` § 4 — canonical sources are the single SoT).

**Literature anchors:**
- Bailey & López de Prado 2014, *Deflated Sharpe Ratio* —
  `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md`. Why the
  smallest-N highest-ExpR lane is the most conservative first live: DSR penalises trial inflation,
  and our portfolio of 3 is small enough that the survivor's posterior is meaningfully positive.
- Carver 2015 Ch12, *Speed and size* —
  `docs/institutional/literature/carver_2015_ch12_speed_and_size.md`. Single-contract first.
  Position-sizing decisions belong AFTER round-trip mechanics are proven.
- Fitschen 2013, *Path of Least Resistance* —
  `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md`. Validates ORB as a
  premise for intraday momentum on commodities + index futures (which is the entire rationale for
  this portfolio existing). Cite once at Phase 2 start to anchor "we are not testing whether ORB
  works; we are testing whether our broker integration works against an ORB strategy that already
  works."

**Halt conditions:**

- Orphan bracket detected after any trade → stop, investigate bracket-management code, do not run
  another lane until resolved.
- Position drift between broker and bot for any duration > 1 bar → stop. This is a classified-
  CRITICAL state (`adversarial-audit-gate.md` qualifies any unrecovered position drift as
  CRIT/HIGH).
- Realised slippage on entry > 1.5x model on >1/3 of trades → stop, the cost model is wrong for
  this session/lane and the validated ExpR overstates live edge.
- TopStep flags any rule violation (DD limit, max-position, etc.) → stop, audit risk_manager.

---

## Phase 3 — Pass the Combine, become Funded

**Goal:** convert the operational pipe into a live Funded Account. F-1 then arms automatically.

**Wall-clock budget:** entirely dependent on TopStep's pass criteria + market regime. Could be days
or weeks. The bot does not optimise for "passing the Combine" — it trades validated edge. Whether
the regime cooperates is a separable question.

**Exit gate:**

- [ ] **3.1** TopStep flips the account from `50KTC-V2-...` to `XFA-...` or `EFA-...`.
- [ ] **3.2** `broker_handshake_check.py` re-run shows the renamed account; `_is_trading_combine_
      account()` no longer matches; F-1 arms.
- [ ] **3.3** `live_readiness_report.py` Criterion 11 + 12 stay green under Funded.
- [ ] **3.4** F-1 scaling ladder is correctly resolved (`_resolve_topstep_xfa_account_size`
      returns the right tier for the account_size). Verify in log.

**Literature anchor:** N/A — this is operations, not research. The bot's behaviour does not change
across the TC→Funded boundary except for F-1 activation.

**Halt conditions:**

- Account renamed but F-1 fails to arm → bug in the detector, halt and fix.
- Funded account_size doesn't map to a known F-1 tier → `_resolve_topstep_xfa_account_size` raises
  fail-closed, which is the correct behaviour. Halt until tier is added explicitly.

---

## Phase 4 — Multi-account scaling (out of scope for this plan)

Listed only so we don't pretend it doesn't exist:

- 4a. Add Express as shadow account → cross-account divergence detection via `CopyOrderRouter`.
- 4b. Add second prop firm (Bulenox / Tradeify / MFFU per repo memo
      `docs/plans/active/2026-05/2026-05-03-prop-firm-automation-compatibility-memo.md`).
- 4c. Self-funded Rithmic via AMP/EdgeClear (NQ minis).

None of these begin until Phase 3 is solid.

---

## Phase log (append as phases progress)

### Phase 0 — executed 2026-05-06 (DEVIATION_RECORDED + remediation queued)

**Outcome:** Phase 0 verification ran with extended scope per [Phase 0 execution plan](#).
Original 4-gate plan (0.1-0.4) extended with gate 0.5 (`live_readiness_report.py`) and 4 added
halt-conditions to close anti-pigeonhole gaps. Gate 0.5 surfaced two institutional-doctrine halts
the original 4-gate plan would have missed silently. Disposition Option C (DEVIATION_RECORDED +
remediation queued) authorised by user 2026-05-06.

**Wall clock:** ~25 min. **Git HEAD start = end = `f04a10b8`** (no commits during Phase 0).

#### Gate 0.1 — `daily_features` rebuild [DEFERRED — surfaced as plan gap]

Plan called for narrow-window rebuild of MNQ orb_minutes=5 across 2026-04-25..2026-05-06.
Failed with FK constraint: `orb_outcomes(symbol, trading_day, orb_minutes) → daily_features`.
DuckDB enforces FK strictly (no cascade). 1764 child rows in window blocked the parent DELETE.

Canonical orchestrator `scripts/tools/rebuild_with_daily_features.py` is **full-instrument
all-aperture full-history** scope (~hours), not narrow-window. Manual `DELETE FROM orb_outcomes`
on the narrow window was correctly blocked by the permission guardrail (per
`RESEARCH_RULES.md § Discovery Layer Discipline`).

**Pre-rebuild state captured (acceptable for Phase 1 signal-only):**

- `daily_features` MNQ orb_minutes=5: n=2045, earliest=2019-05-06, latest=2026-05-05
- `bars_1m` MNQ: n=2,454,672, earliest=2019-05-06 10:00 BNE, latest=2026-05-06 09:59 BNE
- Gap: 1 day (daily_features lags bars_1m by one trading-day boundary; expected — fresh bars
  for 2026-05-06 not yet rolled into daily_features)

**Plan gap surfaced (G1):** the canonical rebuild contract has no narrow-window parent-child
option. Next refresh is multi-hour full-instrument or requires a sanctioned narrow-window path
that does not yet exist. Phase 1 signal-only is not blocked because all 3 deployed lanes are
status WAITING and the 1-day stale `daily_features` does not affect signal generation for
sessions that fire on data already in the table.

#### Gate 0.2 — `pipeline/check_drift.py` [FAIL — 10 violations, all non-deployed]

Exit code 1. **138 checks total: 118 PASS, 0 SKIP (DB available), 19 ADVISORY, 10 VIOLATIONS.**

All 10 violations are from **Check 50** ("Active native trade-window provenance matches canonical
recomputation") — `validated_setups` snapshot drift for 8 strategies whose
`(min_trading_day, max_trading_day, n_trades)` is stale by ~5-13 days vs canonical recomputation.

| Strategy | Stored max | Canonical max | N delta | Deployed? |
|---|---|---|---|---|
| MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15 | 2026-04-23 | 2026-05-04 | +1 | NO |
| MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT15_S075 | 2026-04-23 | 2026-05-04 | +1 | NO |
| MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G4_S075 | 2026-04-30 | 2026-05-04 | +1 | NO |
| MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6_S075 | 2026-04-23 | 2026-05-04 | +1 | NO |
| MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G4 | 2026-04-28 | 2026-05-05 | +3 | NO |
| MNQ_COMEX_SETTLE_E2_RR1.0_CB1_PD_CLEAR_LONG | 2026-04-30 | 2026-05-05 | +1 | NO |
| MNQ_EUROPE_FLOW_E2_RR1.0_CB1_CROSS_SGP_MOMENTUM | 2026-04-30 | 2026-05-05 | +1 | NO |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_CROSS_SGP_MOMENTUM | 2026-04-30 | 2026-05-05 | +1 | NO |

**Cross-reference vs `lane_allocation.json`:** 0 of 8 flagged strategies are in the deployed lane
set (`MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100`, `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`,
`MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`). All 3 deployed lanes are clean of this drift.

**Root-cause fix:** validator re-run to refresh `validated_setups` snapshots for affected rows.
Out of Phase 0 scope (multi-hour, multi-instrument). Per `RESEARCH_RULES.md`, `validated_setups`
is a DERIVED layer banned for truth-finding; live runner reads `lane_allocation.json` not
`validated_setups`, so live correctness is unaffected.

#### Gate 0.3 — Allocator re-read [PASS]

| Field | Value |
|---|---|
| `rebalance_date` | 2026-05-03 (3 days old, expected) |
| `profile_id` | topstep_50k_mnq_auto |
| Lane 1 | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 (status=DEPLOY, regime=None) |
| Lane 2 | MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15 (status=DEPLOY, regime=None) |
| Lane 3 | MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 (status=DEPLOY, regime=None) |
| `sha256` | `0302385e759f361e1c77e52738ef3ece40c2d9556da3b872661b43f4b9a24bc9` |
| File size | 14400 bytes |

Note: `regime=None` in JSON; memory rollup said HOT. Regime field lives in SR state (gate 0.5),
not in allocator JSON. Not a halt — surfaced as evidence note.

#### Gate 0.5 — `live_readiness_report.py` [HALT — both institutional gates invalid]

This gate was added by the Phase 0 execution plan to close the pigeonhole the original 4-gate
plan left open. It surfaced two halts the original would have missed:

- **C11 `gate_ok=False`** — "BLOCKED: Criterion 11 state db identity mismatch. Re-run account
  survival." (`age_days=1`). Per `pre_registered_criteria.md` Criterion 11, this is a
  non-waivable institutional gate (90-day account survival ≥ 70%).
- **C12 `valid=False`** — profile fingerprint mismatch. SR state envelope was computed against
  a different lane set than the current allocator. Result: alarm count returned (`alarms=0`)
  is **not trustworthy** because the envelope itself is invalid.
- **NYSE_OPEN `blocked=True`** — SR alarm `stat=33.27 >= thr=31.96`, baseline=`paper_trades_first_50`,
  stream=`paper_trades`. Lifecycle correctly blocking this lane.
- Other 2 lanes show `blocked=False` with reason "Session HOT" but `sr=None` — meaning their
  SR state isn't being read at all (fingerprint mismatch downstream symptom).

**Allocator visible:** available=True, deployed=3, validated_active=57, validated_only=54,
deployed_not_validated=0, paused=52, stale=0.

#### Gate 0.4 — Broker handshake [PASS]

```
Auth health: True (token 740 chars)
Accounts visible: 2
  - id=21390438 name=50KTC-V2-451890-67605663 (Topstep XFA Combine)
  - id=21944866 name=EXPRESS-V2-451890-53179846 (Topstep Express)
MNQ front-month contract: CON.F.US.MNQ.M26 (MNQM6)
Exit code: 0
```

#### Step 8 — Criterion 12 disposition: DEVIATION_RECORDED + remediation queued (Option C)

**Disposition:** `DEVIATION_RECORDED` with explicit remediation queued in `HANDOFF.md`.

**Rationale:**

1. **C11/C12 are pre-registered institutional gates** (`pre_registered_criteria.md` § 11-12).
   Both are currently False. Phase 1 (`--signal-only`) does not place broker orders, so the
   gates' load-bearing role is for **Phase 2 (live orders)** dispatch, not Phase 1 shake-down.
2. **C12 alarm count is itself untrustworthy** because the envelope is invalid (fingerprint
   mismatch). The original locked-in disposition assumed alarm-count was known and informational;
   it is neither.
3. **Phase 1 truly carries no capital risk** under signal-only. The deviation is bounded
   (no orders dispatched) and the remediation actions are concrete and queued.

**Hard remediation actions (NON-WAIVABLE before Phase 2 dispatch):**

- [x] **R1 — Re-run account survival Monte Carlo** to refresh C11 state DB.
      **CLOSED 2026-05-06 in same session** via
      `python scripts/tools/refresh_control_state.py --profile topstep_50k_mnq_auto`.
      Result: `C11.valid=True, gate_ok=True`, operational survival 90.9%, paths=10000,
      age=0d.
- [x] **R2 — Refresh SR state fingerprint** so envelope matches current 3-lane allocator.
      **CLOSED 2026-05-06 in same session** via the same `refresh_control_state.py` run.
      Result: `C12.valid=True, counts={'ALARM': 3}`. Post-refresh truth: all 3 deployed
      lanes alarm under SR (L1 COMEX_SETTLE OVNRNG_100 SR=35.39, L2 US_DATA_1000
      VWAP_MID_ALIGNED SR=41.60, L3 NYSE_OPEN COST_LT12 SR=33.27, threshold 31.96).
      Alarm action is "report only" because `apply_pauses=false` in sr_state. The
      DEVIATION_RECORDED scope (alarm informational under signal-only) STILL HOLDS for
      this fact-pattern — but Phase 2 dispatch must now answer: are 3-of-3 lanes truly
      decaying, or is this paper-stream noise during shake-down?
- [ ] **R3 — Validator re-run** to clear the 8 `validated_setups` snapshot drift entries
      (Check 50). Lower priority than R1/R2 because none of the 8 are deployed; queue before
      any candidate-promotion decision touches those rows. Currently blocking the Phase 0
      evidence-packet commit (pre-commit refuses on drift); evidence packet sits in working
      tree pending R3.
- [ ] **R4 — `daily_features` next refresh** runs full-instrument scope (G1 limitation);
      MES + MGC are also stale (2026-05-01) per memory and will be caught up at the same time.

**What this disposition does NOT authorise:**

- Phase 2 dispatch with C11 or C12 still invalid.
- Treating the C12 invalid envelope as if `alarms=0` were a real state.
- Bypassing R1-R3 with another DEVIATION_RECORDED if Phase 2 timing pressure surfaces.

#### Step 9 — Evidence packet (this section)

- [x] Gate 0.1 — DEFERRED (FK constraint, plan gap G1 documented)
- [x] Gate 0.2 — FAIL (10 violations, all non-deployed; documented above)
- [x] Gate 0.3 — PASS (allocator SHA256 captured)
- [x] Gate 0.4 — PASS (broker handshake clean)
- [x] Gate 0.5 — HALT (C11 + C12 invalid; remediation R1-R3 queued)
- [x] Step 8 disposition — DEVIATION_RECORDED + remediation queued (Option C, user-authorised
      2026-05-06)
- [x] Step 9 evidence packet — this section
- Git HEAD start: `f04a10b8` (clean tree, 0 ahead/0 behind)
- Git HEAD end: `f04a10b8` (no commits during Phase 0; doc-only edit pending)
- Wall clock: ~25 min
- Phase 0 verification verdict: **PROCEED to Phase 1 signal-only with Option C disposition;
  HARD-BLOCK Phase 2 until R1 + R2 close.**

---

### Phase 1 — gated by Phase 0

### Phase 2 — gated by Phase 1 + user sign-off on lane choice

### Phase 3 — gated by Phase 2 + Combine outcome

---

## Decisions still required from user (must be answered before Phase 2 starts)

- [ ] First-trade lane: COMEX_SETTLE OVNRNG_100 (recommended) or other?
- [ ] First-trade `--auto-confirm` setting: OFF (manual CONFIRM, recommended) or ON?
- [ ] Express account use: dormant for now (recommended) or shadow from Phase 1?
- [ ] Should plan be updated as a working document (recommended) or frozen at Phase 0 approval?

---

## Resume protocol after `/clear`

When context is cleared, the next session should:

1. `git rev-parse --abbrev-ref HEAD && git status --short` — branch-state first check.
2. Read this file end-to-end.
3. Read `HANDOFF.md` for any cross-tool updates.
4. Identify the current Phase by the most recent unchecked exit-gate box.
5. Resume from the next unchecked item. Do not re-derive the plan.

This file IS the plan. Memory index is supplementary, not authoritative
(`feedback_closeout_verify_against_canonical.md`).
