# ROADMAP.md

Features planned but NOT YET BUILT. Move items to CLAUDE.md as they are implemented.

---

## Project Status Notes

### SIL (Silver) Onboarding Initiative — PAUSED / ARCHIVED (2026-02-20)

Per owner decision, SIL onboarding + Volume Session Analysis Tool planning is paused and archived.
Do not schedule active delivery work against SIL until explicitly re-opened.

---

## Phase 1: Daily Features Pipeline — DONE

- `pipeline/build_daily_features.py` — BUILT (6 staged modules, 42 tests)
- `daily_features` table — BUILT (58 columns, schema in init_db.py)
- Configurable ORB duration (5/15/30 min via --orb-minutes)
- Wired into run_pipeline.py orchestrator

## Phase 2: Cost Model — DONE

- `pipeline/cost_model.py` — BUILT (CostSpec, R-multiples, stress test, 20 tests)
- MAE/MFE wired into build_daily_features.py outcome calculation

## Phase 3: Trading App — DONE

- `trading_app/config.py` — 12 ORB size filters + NO_FILTER, ENTRY_MODELS (E1/E2/E3)
- `trading_app/execution_spec.py` — ExecutionSpec dataclass with entry_model field (17 tests)
- `trading_app/entry_rules.py` — detect_confirm + detect_break_touch + resolve_entry E1/E2/E3 (43 tests)
- `trading_app/db_manager.py` — 4 trading_app tables with entry_model column (8 tests)
- `trading_app/outcome_builder.py` — pre-compute outcomes for 6 RR x 5 CB x 3 EM grid (14 tests)
- `trading_app/setup_detector.py` — filter daily_features by conditions (7 tests)
- `trading_app/strategy_discovery.py` — bulk-load grid search 6,480 combos (11 tests)
- `trading_app/strategy_validator.py` — 6-phase validation + risk floor (17 tests)
- Integration test: end-to-end outcome->discovery->validation (5 tests)

## Phase 4: Database/Config Sync — DONE

- `tests/test_app_sync.py` — 31 sync tests (ORB_LABELS, ALL_FILTERS, ENTRY_MODELS, grid params, schema columns)
- Drift check 12: config filter_type sync enforcement
- Drift check 13: ENTRY_MODELS sync enforcement
- Drift check 14: entry price sanity (no entry_price = ORB level without E3 guard)
- Zero tolerance for mismatches, fail-closed

## Phase 5: Expanded Scan — DONE

- Grid: 6 ORBs x 6 RRs x 5 CBs x 13 filters x 3 EMs = 6,480 combos (post-5b expansion)
- **Current state (Feb 27 2026):** 15m/30m rebuilt + 5 new volume-validated sessions added (MGC NYSE_OPEN, MNQ US_DATA_830+COMEX_SETTLE, MES COMEX_SETTLE, M2K COMEX_SETTLE). Validated active: 627 → 206 edge families (34 ROBUST, 38 WHITELISTED). Full rebuild chain completed for all 4 instruments.
- Edge requires G4+ ORB size filter (NO_FILTER and L-filters ALL negative ExpR)
- MARKET_PLAYBOOK.md: comprehensive empirical findings

## Phase 5b: Entry Model Fix + Risk Floor + Win PnL Fix — DONE

- 3 entry models: E1 (next bar open, market), E2 (stop-market at ORB + slippage), E3 (limit retrace). E0 purged Feb 2026 (3 optimistic biases).
- entry_rules.py: ConfirmResult + detect_confirm() + detect_break_touch() + resolve_entry() + _resolve_e1/_resolve_e2/_resolve_e3
- Risk floor: tick-based (10 ticks * 0.10 = 1.0pt) + outcome-based (median/avg_risk_points)
- Win PnL bug fix: changed pnl_points_to_r -> to_r_multiple (friction was missing from wins)
- test_trader_logic.py: 24 trader/math sanity checks
- Grid expanded: 6 ORBs x 6 RRs x 5 CBs x 12 filters x 3 EMs = 6,480 combos
- DB rebuild COMPLETE: ~6.1M outcomes (7 instruments, 5/15/30m ORB), 627 validated active strategies (Feb 27 2026)
- Validator: exclude_years + min_years_positive_pct params added

## Phase 6: Live Trading Preparation — DONE (6a-6d), 6e TODO

### 6a. Strategy Portfolio Construction — DONE
- `trading_app/portfolio.py` — diversified selection, position sizing, capital estimation (32 tests)
- PortfolioStrategy/Portfolio dataclasses with JSON serialization
- compute_position_size() standard + prop firm modes
- diversify_strategies() with max per ORB + max per entry model
- correlation_matrix() from daily R-series with shared calendar + overlap guard
- estimate_daily_capital() for risk budgeting

### 6b. Execution Engine — DONE
- `trading_app/execution_engine.py` — bar-by-bar state machine (19 tests + 20 integration tests)
- State: ARMED -> CONFIRMING -> ENTERED -> EXITED
- ORB detection, break detection, confirm bar counting
- E1/E2/E3 entry model resolution with RiskManager fully wired
- Target/stop/session-end exit handling
- armed_at_bar guard: E1/E3 never fill on the confirm bar itself

### 6c. Risk Management — DONE
- `trading_app/risk_manager.py` — RiskLimits + RiskManager (22 tests)
- Circuit breaker (daily loss limit)
- Max concurrent positions, max per ORB, max daily trades
- Drawdown warning threshold
- Fully integrated with ExecutionEngine (on_trade_entry/on_trade_exit)

### 6d. Paper Trading / Simulation — DONE
- `trading_app/paper_trader.py` — historical replay with journal (9 tests)
- replay_historical(): feeds bars_1m through ExecutionEngine + RiskManager
- Engine handles risk internally (emits REJECT events)
- EOD scratch PnL uses mark-to-market from engine
- CLI for replay runs

### 6e. Monitoring & Alerting — TODO
- Strategy performance tracking (live vs backtest drift)
- Alert on: drawdown exceeding historical, win rate divergence, ORB size regime shift
- Dashboard for live strategy status

---

## Phase 7: Audit & Analysis — DONE (2026-02-08)

### 7a. Codebase Audit — DONE
- Full codebase audit completed. See `AUDIT_FINDINGS.md` for details.
- 5 critical bugs fixed (C1-C5), 6 important fixes (I1-I4, I6-I7)
- 97 new tests across 10 coverage gaps (T1-T10)
- 3 new drift checks (17-19)
- **1811 tests pass, 35 drift checks pass** (as of Feb 27 2026)
- R1 (fill-bar granularity) logged as HIGH PRIORITY R&D task

### 7b. Independent Bars Coverage Audit — DONE
- `pipeline/audit_bars_coverage.py` (~300 LOC)
- Samples ~60 trading days across 4 tiers: boundary, roll, anomaly, random
- Test run (90 days): 86 PASS, 4 WARN (roll-day expected), 0 FAIL
- No modifications to existing pipeline code

### 7c. Strategy Analysis — DONE
- `STRATEGY_ANALYSIS_ASIA_OPEN.md` with live recommendations
- CME_REOPEN: E1 CB2 RR2.5 G6+ = +0.40 ExpR (TOP)
- LONDON_METALS: E3 CB4-5 RR2.0 G6+ = +0.43 ExpR, best Sharpe
- 3-leg core portfolio: +0.55 combined ExpR, 6.5R max drawdown

### 7d. Ship-Ready Hardening — DONE (2026-02-09)
- Ruff linting: cleaned all warnings across codebase
- Strategy viewer CLI (`trading_app/view_strategies.py`) + dashboard panel
- Pre-commit hook wired via `.githooks/`
- GitHub Actions CI on push/PR
- README and backup scripts

---

## Multi-Aperture ORB (5m/15m/30m) — DONE (Phase 8, validated Feb 27 2026)

Wider ORB apertures (15m/30m) validated and integrated into production portfolio via standard pipeline.

### Results
- 15m/30m outcomes rebuilt from scratch with E1/E2/E3 using standard pipeline (`--orb-minutes` parameter), NOT the nested/ pipeline (separate tables, 5m entry bars)
- **627 validated strategies** (all 5m) → 206 edge families (34 ROBUST, 38 WHITELISTED)
- 15m/30m outcomes and experimental strategies exist but zero survived validation
- Portfolio assembly needs re-run with current 627 5m-only strategies
- `trading_app/nested/` modules remain for reference but standard pipeline handles all apertures
- Isolation enforced: drift checks 15-17 block cross-contamination

---

## Rolling Window Evaluation — DONE (2026-02-11)

- `scripts/rolling_eval.py` — 38-window rolling evaluation (12m + 18m training)
- `trading_app/rolling_portfolio.py` — stability scoring + family aggregation
- `tests/test_trading_app/test_rolling_portfolio.py` — unit tests
- Results in `docs/strategy/rolling_eval_results.json`, `rolling_families_12m.json`, `rolling_families_18m.json`
- New daily_features columns: `daily_open/high/low/close`, `gap_open_points`, 6x `orb_*_double_break`
- Portfolio integration: `--include-rolling` CLI flag in portfolio.py
- **Key finding**: Only 2 STABLE families (TOKYO_OPEN_E2_G2, TOKYO_OPEN_E1_G2); CME_REOPEN G3+ are TRANSITIONING; LONDON_METALS/US_DATA_830/SINGAPORE_OPEN/NYSE_OPEN AUTO-DEGRADED by double-break
- 1811 tests pass, 35 drift checks pass (as of Feb 27 2026)

---

## Phase 8: Remaining Work — TODO

### 8a. Fill-Bar Granularity (R1) — DONE (2026-02-09)
- `_check_fill_bar_exit()` in outcome_builder.py checks fill bar OHLC for E1/E3
- E1: entry at bar open, full bar OHLC is post-fill — check stop/target
- E3: intra-bar limit fill, check bar OHLC (stop already guarded by entry_rules)
- E2: skipped (entry at bar close, no post-fill action on that bar)
- Ambiguous fill bar (both stop+target hit): conservative loss
- 9 new tests in TestFillBarExits, 670 tests pass, 19/19 drift checks
- **NOTE:** orb_outcomes rebuilt Feb 2026 with fill-bar logic applied. Current counts: ~6.1M rows across 7 instruments (5/15/30m ORB apertures).
- See `AUDIT_FINDINGS.md` for original finding

### 8b. Multi-Instrument Discovery Grid Update — DONE (Feb 2026)

H2 band filters and H5 direction filter implemented in `config.py`. Discovery grid uses `get_filters_for_grid()` for session-aware filter dispatch. DirectionFilter class (lines 141-150), `_MES_1000_BAND_FILTERS` (lines 227-236), `get_filters_for_grid()` (lines 269-281).

Based on hypothesis test results (Feb 2026, `scripts/tools/hypothesis_test.py`).
Two confirmed findings require config.py + discovery changes:

**Change 1: MES TOKYO_OPEN band filters (H2 confirmed)**
- Add `G4_L12` and `G5_L12` band filters to MES TOKYO_OPEN discovery grid
- `OrbSizeFilter(min_size=4.0, max_size=12.0)` — class already supports `max_size`
- Do NOT apply cap to MES CME_REOPEN (12pt+ is the best zone there)
- Requires: instrument+session-aware filter dispatch in `config.py`

**Change 2: LONG-ONLY direction filter for TOKYO_OPEN session (H5 confirmed)**
- New `DirectionFilter` class in `config.py` (filter_type="DIR_LONG" / "DIR_SHORT")
- `matches_row()` checks `orb_{label}_break_dir` column in daily_features
- Add to grid for TOKYO_OPEN session only, all instruments
- MGC TOKYO_OPEN shorts are negative (-0.09 avgR). MNQ shorts near-zero (+0.03).
- Requires: new filter class + grid builder update in strategy_discovery.py

**Implementation sequence:**
1. Add `DirectionFilter` class to `config.py`
2. Add band filter entries (G4_L12, G5_L12) to filter registry
3. Create `get_filters_for_instrument(instrument, session)` dispatch function
4. Update `strategy_discovery.py` to call dispatch instead of `ALL_FILTERS`
5. Update `_FILTER_SPECIFICITY` to rank new filters
6. Re-run discovery for MES only (band filters)
7. Re-run discovery for MGC, MNQ, MES TOKYO_OPEN only (direction filter)
8. Re-run validator on new strategies
9. Update `live_config.py` if new validated edges emerge

**Files touched:**
- `trading_app/config.py` — new filter classes + dispatch
- `trading_app/strategy_discovery.py` — grid builder + specificity
- `trading_app/strategy_validator.py` — no changes expected
- `tests/test_app_sync.py` — add sync tests for new filters

### 8c. Research Priority Stack (Feb 2026 Birds-Eye Review) — TODO

Six high-leverage research items identified by cross-referencing all findings. Ordered by expected impact.

**P1. Cross-Instrument Correlation + Multi-Instrument Portfolio — DONE (NO-GO for TOKYO_OPEN LONG stacking)**
- Script: `research/research_cross_instrument_portfolio.py` — COMPLETED Feb 2026
- Result: MNQ/MES daily R correlation at TOKYO_OPEN = +0.83 (effectively same trade)
- MGC/equity correlation at TOKYO_OPEN = +0.40 to +0.44 (moderate, not the diversification freebie hoped for)
- Adding MNQ+MES to MGC at TOKYO_OPEN LONG worsens portfolio Sharpe — DO NOT STACK
- Action: pick ONE equity micro (MNQ or MES) per session, don't run both
- Next: find a truly uncorrelated asset class (bonds, FX, ags) for real portfolio diversification

**DST Edge Audit — DONE (fully remediated Feb 2026)**
- Script: `research/research_dst_edge_audit.py` — COMPLETED Feb 2026
- Initial finding: fixed slightly outperformed dynamic on summer-only matched days (+0.13R MGC, +0.14R MES)
- Winter edges stronger across all instruments (MES +0.35R, MGC +0.18R, MNQ +0.09R)
- Full remediation DONE: validator split, volume analysis, revalidation, doc updates. See DST REMEDIATION section above.
- Winter seasonality signal feeds into P2 (calendar effects)

**DST CONTAMINATION REMEDIATION — DONE (Feb 2026)**
- The old fixed sessions (0900/1800/0030/2300) blended two market contexts. Now replaced by event-based sessions (CME_REOPEN, LONDON_METALS, NYSE_OPEN, US_DATA_830).
- US_DATA_830 (formerly 2300) special case: was NEVER aligned with US data release under old fixed naming. Now resolves correctly.
- Step 1: ✅ 24-hour ORB time scan with winter/summer split (`research/research_orb_time_scan.py`) — DONE
  - TOKYO_OPEN is the anchor (STBL all instruments). MGC 19:00 STBL. MES CME_REOPEN winter = NEGATIVE.
- Step 1b: ✅ Strategy revalidation with DST split (`research/research_dst_strategy_revalidation.py`) — DONE
  - 1272 strategies: 275 STABLE, 155 WINTER-DOM, 130 SUMMER-DOM, 10 SUMMER-ONLY, 48 UNSTABLE, 654 LOW-N.
  - NO validated strategies broken. Red flags: MES CME_REOPEN E1 experimental only.
  - CSV: `research/output/dst_strategy_revalidation.csv`
- Step 2: ✅ Winter/summer split baked into `strategy_validator.py` — DONE
  - DST columns (`dst_winter_n`, `dst_winter_avg_r`, `dst_summer_n`, `dst_summer_avg_r`, `dst_verdict`) on `experimental_strategies` and `validated_setups`.
  - Auto-migration in `db_manager.py:init_trading_app_schema()`.
  - `classify_dst_verdict()` in `pipeline/dst.py`.
- Step 3: ✅ Volume analysis confirms event-driven edges (`research/research_volume_dst_analysis.py`) — DONE
  - CME_REOPEN: 63% volume drop summer. US_DATA_830: +76-90% summer. LONDON_METALS: 31% winter premium (MGC).
  - Findings: `research/output/volume_dst_findings.md`, CSV: `research/output/volume_dst_analysis.csv`
- Step 4: ✅ TRADING_RULES.md session playbooks updated with DST split numbers — DONE
- Step 5: ✅ New session candidates evaluated — ALL REJECTED (insufficient G4+ frequency or too much overlap)
- Step 6: ✅ Migrate DST columns to production gold.db — DONE (auto-migration in init_trading_app_schema())
- All sessions are now event-based — DST contamination is fully resolved

**P2. Calendar Effect Scan (Day-of-Week, FOMC, NFP, Opex) — DONE (Feb 2026)**
- Script: `research/research_day_of_week.py` — COMPLETED
- Output: `research/output/day_of_week_breakdown.csv`, `day_of_week_macro_overlay.csv`, `day_of_week_skip_filter.csv`
- **NFP_SKIP:** First-Friday NFP days are toxic for breakout strategies. Actionable skip filter.
- **OPEX_SKIP:** Monthly options expiration days degrade edge. Actionable skip filter.
- **FRIDAY_SKIP (CME_REOPEN only):** Friday CME_REOPEN underperforms other weekdays. Session-specific skip.
- **DOW at TOKYO_OPEN:** Day-of-week has no significant effect — noise. Do NOT filter.
- Calendar filters implemented in `pipeline/calendar_filters.py` and integrated as portfolio-level overlays in `trading_app/config.py` (`CALENDAR_OVERLAYS`).

**P3. Signal Stacking (Size + Direction + Concordance + Volume) — DONE (Feb 2026)**
- Script: `research/research_signal_stacking.py` — COMPLETED
- Output: `research/output/signal_stacking_*.{csv,md}`
- Reference: E3/CB1/RR2.0 fixed (avoids Bailey Rule). In-sample only.
- **MES NYSE_OPEN summer G4/G5 L4 (size+dir+conc+vol): SURVIVED BH FDR** — avgR=+0.42/+0.41, N=136/124, p_bh=0.023/0.028 [PRELIMINARY]
  - Mechanism: in US DST summer, NYSE_OPEN bar is POST equity open.
    Vol spike + MES/MNQ concordance = institutional follow-through from already-open US market.
  - Caveat: E3 reference only. Not yet tested at E1 (the production entry for NYSE_OPEN).
- **MGC TOKYO_OPEN G5 L4** — avgR=+0.545, N=37 (REGIME), p_bh=0.16 — interesting but small N.
- **MGC CME_REOPEN winter G4/G5 L3-L5** — avgR=+0.35-+0.37 (REGIME), not BH-significant.
- **184/216 cells positive** but only 2 survived BH — individual filters already capture most of the edge.
- **TODO (implement):**
  1. Test MES NYSE_OPEN summer L4 at E1 entry (production entry for NYSE_OPEN) — if still positive, wire as position-size overlay
  2. Add `conc_tier` + `rel_vol` columns to live portfolio overlay (not a new strategy — 2x size on qualifying days)
  3. Re-test MGC TOKYO_OPEN G5 L4 at a 2-year window once N grows to PRELIMINARY threshold

**P4. E3 x Direction Interaction**
- H5 direction test used E1 only. E3 (retrace entry) could behave differently.
- If TOKYO_OPEN LONG bias is structural flow → E3 LONG = buying the dip in a session that wants to go up.
- Could be highest-quality single trade in entire system.
- Script: extend `hypothesis_test.py` or new `research/research_e3_direction.py`

**P5. Time-to-Target (Winner Speed Profiling)**
- We know MFE/MAE but not temporal signature of winners.
- If 80% of winners hit target in 2 hours, 7-hour hold is dead exposure.
- Fast winners under specific filters = more turns = more total R per year.
- Script: `research/research_winner_speed.py` (to build)

**P6. Regime Hedging via Multi-Instrument Allocation — REVISED (P1 findings change approach)**
- ORB sizes 15x larger 2025 vs 2021. Current system is gold-vol-dependent.
- When gold cools, G5+ days drop to 5/year. System becomes unfundable on MGC alone.
- P1 showed MNQ/MES correlation = +0.83 at TOKYO_OPEN (NOT the free hedge expected).
- MGC/equity correlation = +0.40-0.44 (moderate). MNQ OR MES adds some diversification but not enough.
- Revised plan: need truly uncorrelated asset (bonds micro? FX micro? ags?) for real regime hedge.
- Depends on: user selecting candidate asset + data acquisition.

### 8d. Monitoring & Alerting (Phase 6e) — TODO
- Strategy performance tracking (live vs backtest drift detection)
- Alert on: drawdown exceeding historical, win rate divergence, ORB size regime shift
- Dashboard for live strategy status

### 8e. orb_outcomes Backfill 2016-2020 — DONE (Feb 2026)
- MGC: 2016-02-01 to 2026-02-04, E1/E2/E3 (E0 purged Feb 2026)
- MES: 2019-02-12 to 2026-02-11, E1/E2/E3 (E0 purged Feb 2026)
- 10-year validation window now available for MGC
- TODO (separate task): re-run rolling_eval.py with 10-year range for stability scores

---

## Rules to Enforce (ACTIVE)

### Strategy Family Isolation
- Cross-family inference FORBIDDEN
- Each family is a memory container

### daily_features RR=1.0 Warning
- Outcomes in daily_features are RR=1.0 ONLY
- NEVER use for RR > 1.0 backtesting
- Use strategy_discovery.py for higher RR

### Database/Config Sync Protocol
- NEVER update validated_setups without updating config.py
- Run test_app_sync.py after every change
- Zero tolerance for mismatches
- Drift check 12 catches filter_type mismatches automatically

### ORB Size Regime Awareness
- ORB sizes are 3-4x larger in 2025 vs 2021
- L-filters (max_size) will match FEWER days going forward
- G-filters (min_size) will match MORE days going forward
- Re-validate strategies periodically as volatility regime evolves
