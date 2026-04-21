# Phase 6e Monitoring & Alerting — Design Doc

**Date:** 2026-04-21
**Worktree:** `deploy/live-trading-buildout-v1`
**Authority:** `docs/plans/2026-02-08-phase6-live-trading-design.md` § Phase 6e (lines 410-446).
**Purpose:** Design-only. Enumerate the gap between what already exists under `trading_app/live/` and what Phase 6e specifies, so a follow-on build session can execute with a clean blast-radius map.
**Scope this run:** NO code. Design gate per CLAUDE.md § Design Proposal Gate. Build is explicitly out-of-scope.
**Why now:** Phase 6e is the single remaining engineering prerequisite for real-capital flip per `docs/audit/hypotheses/2026-04-21-deploy-live-participation-shape-shadow-v1.yaml` `kill_criteria.real_capital_flip_block.blocked_until_all_true[0]`.

---

## 1. Current-state inventory (what exists)

Surveyed `trading_app/live/` on branch `deploy/live-trading-buildout-v1` at commit `d88ecde7`:

| File | Primary primitive | Phase 6e relevance |
|---|---|---|
| `sr_monitor.py` | `ShiryaevRobertsMonitor` class + `calibrate_sr_threshold()` + `_estimate_arl()` Monte-Carlo ARL-based threshold calibration | Drift detection at trade-level (Pepelyshev-Polunchenko 2015 Eq. 11 + Eq. 17-18). Directly covers ExpR drift alert if wired. |
| `cusum_monitor.py` | `CUSUMMonitor` class with `update(actual_r)`, `clear()`, `drift_severity()` | Drift detection, complementary to SR. Covers WR drift alert if wired. |
| `alert_engine.py` | `OperatorAlert` dataclass + `classify_operator_alert()` + `record_operator_alert()` + `read_operator_alerts()` + `summarize_operator_alerts()` | Alert persistence + retrieval infrastructure. Ready to consume events from detectors. |
| `performance_monitor.py` | `PerformanceMonitor` class + `TradeRecord` + `_compute_std_r()` + `record_trade()` + `daily_summary()` + `reset_daily()` + `get_cusum()` | Per-strategy / per-day PnL aggregation. Has per-strategy CUSUM hook. Directly covers drawdown + circuit-break alerts. |
| `bot_dashboard.py` | FastAPI app + `_collect_data_status()` + `_collect_broker_status()` + `_collect_alert_summary()` + `_profile_session_ambiguity()` + operator-state derivation | Dashboard shell is built. Lane cards + alert summary + broker status already rendered. |
| `trade_journal.py` | `TradeJournal` + `generate_trade_id()` | Trade logging primitive; consumed by PerformanceMonitor. |

**Net read:** the 6e spec's "primitives" already exist. What's missing is (a) **wiring**, (b) **specific thresholds**, (c) **2 of 7 alert types not yet implemented** (ORB size regime + strategy stale), (d) **dashboard panels** named in the 2026-02-08 spec, (e) **pre-reg-style numeric contracts** so thresholds are not re-tunable post-hoc.

---

## 2. Gap analysis (7-alert map)

Ordering matches `docs/plans/2026-02-08-phase6-live-trading-design.md:420-428`.

| # | Alert (2026-02-08 spec) | Trigger | Severity | Existing primitive | Gap |
|---|---|---|---|---|---|
| 1 | Drawdown | Daily PnL < -3R | WARNING | `PerformanceMonitor.daily_summary()` | Threshold wiring + alert_engine call on daily-summary tick |
| 2 | Circuit Break | Daily PnL < -5R | CRITICAL | `PerformanceMonitor.daily_summary()` | Threshold wiring + CRITICAL-severity alert + auto-halt hook (to existing `session_orchestrator`) |
| 3 | Win Rate Drift | Rolling 50-trade WR < backtest WR − 10pp | WARNING | `PerformanceMonitor.get_cusum()` OR a separate rolling-WR window | Rolling-WR tracker; comparator to per-strategy backtest WR (source = `active_validated_setups` or per-lane baseline); threshold computation + alert |
| 4 | ExpR Drift | Rolling 50-trade ExpR < 50% of backtest | CRITICAL | `ShiryaevRobertsMonitor` + `PerformanceMonitor` | SR calibration per-strategy to backtest ExpR; threshold mapping SR statistic → 50%-of-backtest-ExpR; alert wiring |
| 5 | ORB Size Regime | 30-day median ORB size > 2× backtest median | INFO | None — no current primitive reads `orb_size_pts` rolling windows | **Build required.** New function: query `orb_outcomes.orb_size_pts` for last-30-days per (instrument × session), compare against per-lane baseline median from IS distribution, emit INFO alert. |
| 6 | Missing Data | Expected bar count < 80% of normal | WARNING | `bot_dashboard._collect_data_status()` has the infra; alert not wired | Expected-bar-count baseline per (instrument × session); compute ratio per trading day; alert if < 0.80 |
| 7 | Strategy Stale | No trade in 30+ calendar days | INFO | `bot_dashboard._profile_session_ambiguity()` surfaces recency but no alert | Per-strategy last-trade timestamp check; scheduled daily; alert if > 30d |

**Net:** 5 of 7 alerts have existing primitives and need wiring only. 2 of 7 (ORB Size Regime + Strategy Stale) need new detection primitives. Zero alerts require rewriting the existing primitives — the gap is glue code + thresholds + tests.

---

## 3. Dashboard panel map (2026-02-08 spec § Dashboard)

Spec line 432-437 lists 5 panels. Mapped to `bot_dashboard.py`:

| Panel (2026-02-08) | Current state |
|---|---|
| Active portfolio strategies | PARTIAL — lane-cards exist (`_legacy_lanes_to_lane_cards`) but not a dedicated "active portfolio" panel |
| Daily PnL curve (cumulative R) | MISSING — no cumulative-R plot in dashboard |
| Per-strategy performance vs backtest | MISSING — lane cards show live stats but no vs-backtest overlay |
| Risk utilization (% of limits used) | MISSING — `RiskLimits` tracked by `RiskManager` but not rendered in dashboard |
| ORB size regime tracker | MISSING — ties to Alert #5 (ORB Size Regime) |

**Net:** 1 of 5 panels partial, 4 of 5 missing. Build on existing FastAPI + lane-card pattern. No new dashboard framework needed.

---

## 4. Pre-reg-style numeric contracts (locked before build)

Per CLAUDE.md § Design Proposal Gate + pre_registered_criteria.md § no-post-hoc-relaxation rule, every alert threshold must be committed in writing BEFORE the first code change. Proposed locked values:

| Alert | Threshold parameter | Value | Source / rationale |
|---|---|---|---|
| 1 Drawdown | Daily PnL threshold | −3R | 2026-02-08 spec (direct copy) |
| 2 Circuit Break | Daily PnL threshold | −5R | 2026-02-08 spec |
| 3 WR Drift | Window size | 50 trades | 2026-02-08 spec |
| 3 WR Drift | WR delta threshold | 10 pp below backtest | 2026-02-08 spec |
| 4 ExpR Drift | Window size | 50 trades | 2026-02-08 spec |
| 4 ExpR Drift | ExpR ratio threshold | 0.50 (i.e., < 50% of backtest) | 2026-02-08 spec |
| 4 ExpR Drift | SR alarm ARL₀ target | 1000 trades | matches shadow pre-reg G9 kill criteria; Pepelyshev-Polunchenko 2015 Eq. 11 |
| 5 ORB Size Regime | Rolling window | 30 calendar days | 2026-02-08 spec |
| 5 ORB Size Regime | Median ratio threshold | 2.0× | 2026-02-08 spec |
| 6 Missing Data | Ratio threshold | 0.80 of expected | 2026-02-08 spec |
| 6 Missing Data | Baseline source | Per-session historical mean bar count (gold.db) | Needs per-session look-up at build time |
| 7 Strategy Stale | Inactivity threshold | 30 calendar days | 2026-02-08 spec |

**Locked.** No post-hoc tuning. Any future adjustment requires a new pre-reg-style amendment block in this doc, cited by date.

---

## 5. Build plan (blast radius, for a future session — NOT this run)

**Files NEW (build session creates):**
- `trading_app/live/monitor_runner.py` — orchestrator: subscribes to `PerformanceMonitor.record_trade()`, runs all 7 detectors on cadence, dispatches to `alert_engine.record_operator_alert()`. ~150 LOC.
- `trading_app/live/detectors/` — package with 7 detector modules (one per alert). Each detector takes PerformanceMonitor state + thresholds dict + returns `list[OperatorAlert]`. ~50-100 LOC each.
- `trading_app/live/monitor_thresholds.py` — frozen dataclass containing the numeric contracts from § 4. `@revalidated-for`-annotated per CLAUDE.md § Research Provenance Rule (source = 2026-02-08 spec + this design doc, not research).
- `tests/test_trading_app/live/test_monitor_runner.py` — wiring tests (~8).
- `tests/test_trading_app/live/test_detectors_*.py` — per-detector tests (~10 per detector, ~70 total).

**Files MODIFIED (build session touches):**
- `trading_app/live/session_orchestrator.py` — hook `monitor_runner` into bar-event flow post-trade-record. Minimal delta: ~10 lines.
- `trading_app/live/bot_dashboard.py` — add 4 new panels (daily PnL curve, vs-backtest overlay, risk utilization, ORB size tracker). ~200 LOC additive.
- `trading_app/live/performance_monitor.py` — add rolling-WR + rolling-ExpR accessors if not already present. ~30 LOC additive.

**Files READ-ONLY:**
- `trading_app/live/{sr_monitor,cusum_monitor,alert_engine,trade_journal}.py` — primitives consumed, not modified.

**Files UNTOUCHED (must stay):**
- `pipeline/*` — monitoring reads `orb_outcomes` for baseline stats (read-only).
- `trading_app/config.py`, `trading_app/prop_profiles.py`, canonical registries — do not modify.
- `gold.db` schema — monitor writes ONLY to `operator_alerts` table via existing `alert_engine` (already schema'd).

**Companion tests estimate:** ~88 unit tests + integration smoke test (inject synthetic trade stream, confirm 7 alerts fire at correct thresholds). Aligns with 2026-02-08 spec line 439-442 "~10 tests" (outdated; spec underestimates).

**Blast-radius summary:** all new code lives in `trading_app/live/` subtree, reads canonical registries read-only, writes only to `operator_alerts` via existing infrastructure. Zero canonical-config touches. Minimal `session_orchestrator.py` hook. Dashboard build-out is additive (no panel removals). Companion test burden moderate (~88 new tests).

---

## 6. Acceptance criteria (for the future build session)

Per CLAUDE.md § "Done" definition:
1. All 7 alerts fire at locked thresholds on synthetic test data.
2. Dashboard renders 5 new/updated panels with real paper_trades data.
3. `check_drift.py` passes.
4. All new + existing tests pass. No CRLF churn committed.
5. Dead-code sweep (`grep -r` new module names) shows expected call sites.
6. Self-review pass: simulated happy path, edge case (no trades), and one failure mode (one detector fails — others still fire) all walked through in writing.
7. Pre-reg-style numerics in `monitor_thresholds.py` match § 4 exactly. No post-hoc tuning.

---

## 7. Not-this-run / out-of-scope

- Real-time data-feed selection (Databento, IB, Tradovate) — deferred per 2026-02-08 spec § Open Questions Q3.
- Account-size / risk-% parameterization — deferred per spec Q1.
- Prop firm mode (max-drawdown vs equity sizing) — deferred per spec Q5; largely answered by the active TopStep XFA profile in `prop_profiles.py` but not bound to monitor thresholds.
- Shadow deployment wiring for PR #48 shape signal (separate Workstream C sub-phase, currently MNQ-only per `docs/audit/hypotheses/2026-04-21-deploy-live-participation-shape-shadow-v1.yaml` post-F7 amendment).
- Phase 6a-6d (spec § Sub-Phase Architecture): already marked IMPLEMENTED per the header of the 2026-02-08 spec.

---

## 8. Unlocks after build

- Real-capital-flip block item #1 (`Phase 6e monitoring & alerting delivered`) CLEARS.
- Remaining real-capital prerequisites per the shadow pre-reg: (a) per-instrument-K=1 MNQ DSR pre-reg (separate), (b) credentials + profile flip (user-owned), (c) MFFU / Bulenox written confirmations (user-owned).
- After 6e lands, a second agent session can wire shadow monitoring for the PR #48 MNQ shadow population using the new `monitor_runner` infrastructure — naturally extends 6e rather than duplicating it.

---

## 9. Decisions required before build

1. **Confirm threshold lock** per § 4. If any thresholds need change from 2026-02-08 spec values, decide BEFORE the build session commits `monitor_thresholds.py`.
2. **Confirm dashboard panel priority** per § 3. If risk-utilization panel needs prop-firm-specific sizing, resolve user Open Question Q5 first.
3. **Confirm test coverage target** per § 5. ~88 tests is current estimate; actual burden may vary.
4. **Confirm build sequence** — detectors first (parallel), then monitor_runner, then dashboard, then session_orchestrator hook. Each sub-step gets its own commit.

---

## 10. Provenance

- Authority: `docs/plans/2026-02-08-phase6-live-trading-design.md` § Phase 6e (commit origin/main @ `f567cfe6`).
- Inventory source: `grep -n "^def \|^class \|^async def "` on `trading_app/live/*` at branch HEAD commit `d88ecde7` (2026-04-21 post-A commit).
- Threshold source: 2026-02-08 spec verbatim except SR ARL₀ = 1000 which cross-references shadow pre-reg G9 (both cite Pepelyshev-Polunchenko 2015 Eq. 11).
- No live data queried (design doc only).
- No training-memory fallbacks.

**Status:** DESIGN LOCKED. Ready for a future build session. NOT buildable in this worktree session.
