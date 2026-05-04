# Comprehensive Deployed-Lane Gap Audit

**Date:** 2026-04-15
**Purpose:** Institutional-grade enumeration of every trade-time-knowable variable
and every analytical axis that could exploit edge on the 6 deployed MNQ lanes.
Intent: no bias, no gaps, no stone unturned.

**Companion pre-registration:** this document is the *master gap register*. It is
NOT a pre-registration itself. Each numbered sub-scan that commits K-budget
to a specific discovery run gets its own pre-reg file per Phase 0 criterion 1.

---

## 1. Target system

| # | Lane (strategy_id) | Session | Apt | RR | Filter | annual_r | trailing_expr |
|---|--------------------|---------|-----|----|--------|----------|---------------|
| L1 | MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | EUROPE_FLOW | 5 | 1.5 | ORB_G5 | 44.3 | +0.1854 |
| L2 | MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30 | SINGAPORE_OPEN | 30 | 1.5 | ATR_P50 | 44.0 | +0.2407 |
| L3 | MNQ_COMEX_SETTLE_E2_RR1.5_CB1_OVNRNG_100 | COMEX_SETTLE | 5 | 1.5 | OVNRNG_100 | 39.8 | +0.2612 |
| L4 | MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5 | NYSE_OPEN | 5 | 1.0 | ORB_G5 | 28.2 | +0.1188 |
| L5 | MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5 | TOKYO_OPEN | 5 | 1.5 | ORB_G5 | 21.6 | — |
| L6 | MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED | US_DATA_1000 | 5 | 1.5 | VWAP_MID_ALIGNED | 22.1 | — |

Source: `docs/runtime/lane_allocation.json` (rebalanced 2026-04-13).

---

## 2. Variable taxonomy — 3 classes

### 2.1 Filter-layer variables (binary gate at session start)
Applied via `StrategyFilter.matches_row()`. Fires at ORB-end time. Pass → day is tradable on the lane. Fail → no trade that day.

### 2.2 Size-layer variables (continuous multiplier)
Scale position size per-trade. Full size, half size, 1.5x size, etc.
Currently unused in deployment (all lanes = 1x size). Carver Ch 10 framework.

### 2.3 Entry/geometry variables (trade execution)
Change HOW we enter or exit:
- Entry type: stop-market (E2), limit-on-retest (E_RETEST), pullback, VWAP-anchored, etc.
- Stop distance: 0.5x, 1.0x, 1.5x ATR, dynamic
- Target: fixed RR, next structural level, time stop
- Confirm bars: 1 (deployed), 2, 3

---

## 3. Variables RESOLVED (tested and dispositioned)

### 3.1 Level-based filters (F1-F8 zone features)
- **Status:** DISPOSITIONED via 2026-04-15 mega + T0-T8 batch + scoped mega
- **Outcome:** 3 CONDITIONAL patterns (P1/P2/P3) + 2 new VALIDATED (BRISBANE_1025, US_DATA_830) + 17 CONDITIONAL + 8 KILL + adversarial FADE dead
- **Live impact:** 2 BH-FDR survivors on US_DATA_1000 (L6) — F5_BELOW_PDL (TAKE) and F6_INSIDE_PDR (SKIP). Both map to size-scaling only (binary would hurt).
- **Files:** `docs/audit/results/2026-04-15-prior-day-features-orb-mega-exploration.md`, `-t0-t8-audit-o5-patterns.md`, `-t0-t8-audit-hot-warm-batch.md`, `-t0-t8-adversarial-fade-audit.md`, `-mega-deployed-sessions-only.md`

### 3.2 Volatility regime filters (partial)
- **Tested:** ATR_P50, ATR_P70 (deployed on L2), ATR70_VOL, GARCH_VOL_LT20 (research-provisional Wave 5 G5)
- **Not tested on deployed lanes:** atr_vel_ratio as overlay, garch_forecast_vol_pct as overlay, atr_20_pct as finer grain (Q1-Q5 rather than P50/P70)

### 3.3 ORB geometry filters
- **Tested and deployed:** ORB_G5 (quintile 5) on L1/L4/L5, O5 vs O30 apertures
- **Not tested as overlay:** orb_size × break_delay interaction, orb_size × compression_z interaction

### 3.4 Overnight range filters
- **Tested and deployed:** OVNRNG_100 on L3
- **Not tested as overlay on non-L3 lanes:** overnight_range_pct (percentile) as overlay on L1/L2/L4/L5/L6

### 3.5 Calendar / day-of-week filters
- **Tested:** is_friday hit on some lanes historically (see MEMORY cross_session_concordance)
- **Not tested on deployed lanes:** comprehensive DOW × lane matrix, is_nfp_day × direction, is_opex_day interaction

### 3.6 Entry timing filters
- **Tested:** break_delay_min as signal (SIGNAL per quant-audit-protocol known failure patterns — pending T3-T8)
- **Not tested as overlay on deployed lanes:** break_delay_min buckets per lane, break_bar_continues

### 3.7 Confluence / composite filters
- **Tested:** 48 BH FDR survivors from confluence program (MEMORY.md: confluence_program_results.md). New-on-new stacking DEAD (0/8 OOS) per prior work.
- **Not tested:** specific confluence of {deployed_filter} × {new_trade_time_feature} pairs per lane.

### 3.8 Adversarial direction (FADE)
- **Status:** DEAD. 13/23 fail, 9/23 weak null, 1/23 marginal (2026-04-15)
- **Caveat:** does NOT rule out E_RETEST (limit-on-retest after initial break-and-fail).

---

## 4. Variables UNTESTED as overlay on deployed lanes (HIGH LEVERAGE)

### 4.1 Trade-time continuous features (at ORB end, before break entry)

| Feature | Source column | Rationale | Scope of scan |
|---------|---------------|-----------|---------------|
| `orb_{s}_compression_z` | daily_features | Pre-ORB vol compression. Low = compressed → expect expansion. | Tertile bucketing (Q1, Q3) |
| `orb_{s}_pre_velocity` | daily_features | Pre-ORB price velocity. High = momentum carry in break dir. | Tertile bucketing |
| `rel_vol_{s}` | daily_features | Relative volume vs session avg. High = institutional interest. | Tertile bucketing |
| `atr_vel_ratio` | daily_features | Recent vol acceleration. High = regime change. | Tertile bucketing |
| `garch_forecast_vol_pct` | daily_features | Forward vol forecast percentile. Tested on Wave 5 G5; not as overlay. | Tertile bucketing |
| `atr_20_pct` | daily_features | Current vol percentile. Finer than P50 deployed filter. | Q1 vs Q5 |
| `overnight_range_pct` | daily_features | ON range vs historical. Links to OVNRNG_100 on L3. | Q1 vs Q5 |
| `gap_open_points / atr_20` | computed | Opening gap size scaled by ATR. | Q1 vs Q5 |

### 4.2 Trade-time binary features

| Feature | Rationale |
|---------|-----------|
| `overnight_took_pdh` | Overnight already took prev-day high — implies no PDH to take in session |
| `overnight_took_pdl` | Same for PDL |
| `is_nfp_day` | NFP volatility regime |
| `is_opex_day` | Options expiration dynamics |
| `is_friday` | Weekly close effects |
| `is_monday` | Gap weekend |
| `gap_type = 'gap_up' / 'gap_down'` | Directional gap |
| `atr_vel_regime` | Categorical vol regime |

### 4.3 Break-time features (available at entry decision)

| Feature | Rationale |
|---------|-----------|
| `orb_{s}_break_delay_min` | Already validated as SIGNAL in prior research. Not tested as overlay per lane. |
| `orb_{s}_break_bar_continues` | Break bar closed through vs rejected |
| `orb_{s}_break_bar_volume / orb_{s}_volume` | Break bar volume ratio |

### 4.4 HTF level distance features (untested)

| Feature | Construction | Rationale |
|---------|-------------|-----------|
| `dist_to_session_ny_high` | `abs(orb_mid - session_ny_high) / atr_20` | Prior NY session high as resistance |
| `dist_to_session_ny_low` | `abs(orb_mid - session_ny_low) / atr_20` | Prior NY session low as support |
| `dist_to_session_london_high` | `abs(orb_mid - session_london_high) / atr_20` | London range reference |
| `dist_to_session_asia_high/low` | Same for Asia session | Asia range reference |
| `position_in_prev_day_range` | `(orb_mid - prev_day_low) / prev_day_range` | % of PDR traveled |

### 4.5 Size-layer variables (Carver framework, UNUSED)

| Variable | Current | Proposed (Carver Ch 10) |
|----------|---------|-------------------------|
| Position size | 1x per lane, binary | Scaled by forecast: `size = max(-2, min(+2, forecast_combined))` |
| Forecast combination | None | Weighted sum of F1…F8 + vol-regime + break-timing, clamped to [-2, +2] |
| Vol targeting | None | Target 12% annualized lane vol via `size = target / realized_vol` |
| Signal scalars | None | Each signal calibrated to mean forecast of 10 per Carver Ch 10 p161 |

### 4.6 Entry geometry variables (UNTESTED)

| Variable | Current | Candidates |
|----------|---------|------------|
| Entry model | E2 (stop-market @ ORB edge + CB1) | E_RETEST (limit on pullback), E_VWAP (limit @ session VWAP), E_PULLBACK (limit @ break_price - 0.25*ORB_size) |
| Stop | 1.0x ATR | 0.5x, 0.75x, dynamic per-cell |
| Target | Fixed RR | Next structural level (PDH/PDL/pivot), time-based, MFE trailing |
| Confirm bars | 1 | 2, 3, dynamic |

### 4.7 Cross-instrument expansion (UNTESTED)

The 6 deployed lanes are ALL MNQ. MES / MGC equivalents for each session:
- MES EUROPE_FLOW RR1.5 ORB_G5 — does it pass Tier 1+?
- MES SINGAPORE_OPEN / COMEX_SETTLE / TOKYO_OPEN / NYSE_OPEN / US_DATA_1000 — each untested
- MGC equivalents — Tier 2 (price-only, per parent_proxy_data_policy)

Would 2x portfolio capital efficiency if they pass and aren't correlation-blocked.

### 4.8 Portfolio-level variables (UNTESTED)

| Variable | Status | Potential uplift |
|----------|--------|------------------|
| Anti-correlated pair stacking | Untested on current 6 lanes | Can size up without vol increase |
| Daily best-N (take top 3 of 6 by confluence score) | Untested | Higher conviction, lower DD |
| Regime-conditional lane activation | Untested | Turn off lanes in adverse regimes |
| Cross-asset confirmation (MES agrees MNQ) | Untested | Reduces false positives |

---

## 5. This-session scope (PRE-COMMITTED — EXPANDED after user pushback)

User correctly pushed back on narrow scope. Two expansion vectors:
1. "What about all other WARM lanes we didn't test?" — audit the non-dir-match cells
2. "What about using these indicators in OTHER WAYS not just ORB?" — new strategy class

### 5.1 Sub-scan Alpha — trade-time feature OVERLAY scan on 6 deployed lanes

- **K budget:** ~300 cells (6 lanes × ~25 feature-buckets × avg 2 directions)
- **Scope:** every variable in §4.1-4.4 at low/high tertile split
- **Method:** Welch's t vs lane-universe rows, BH FDR q=0.05 at discovery K, per-feature-family BH, cross-check T0 tautology vs existing deployed filter
- **Gate:** survivors pass to T0-T8 battery next round

### 5.2 Sub-scan Beta — cross-instrument twin check

- **K budget:** 6 sessions × {MES, MGC} × deployed (apt, RR) = 12 cells
- **Scope:** does each deployed lane have a MES/MGC twin that passes baseline ExpR > 0 + N ≥ 100?
- **Output:** candidate new lanes for the portfolio (capital-efficiency expansion)

### 5.3 Sub-scan Gamma — NEW LANE candidates from full mega universe

- **Scope:** ALL 30 HOT/WARM cells + top 30 LUKEWARM-positive-ExpR cells from mega
- **Filter:** cells with ExpR_on > 0 (tradable as lanes), N_on ≥ 100 (deployable N)
- **Method:** T0-T8 battery per cell, same rigor as prior audits
- **Gate:** survivors evaluated for portfolio inclusion via correlation gate + allocator math
- **Output:** `docs/audit/results/2026-04-15-new-lane-candidates-t0t8.md`

### 5.4 Sub-scan Delta — comprehensive feature scan on NON-deployed top sessions

- **Scope:** sessions NOT in deployed 6 but appearing in mega top-50 with |t|>=3: CME_PRECLOSE, NYSE_CLOSE, US_DATA_830, BRISBANE_1025
- **Instruments:** MNQ + MES + MGC (where session applies)
- **Features:** same as Alpha §5.1
- **K estimate:** 4 sessions × 3 instruments × 3 RR × 2 dir × ~15 feature-buckets = ~1080 cells
- **BH-FDR at per-family:** stricter threshold given K
- **Output:** extension of Alpha report

### 5.5 Sub-scan Epsilon — non-ORB strategy class (SC2) pre-registration

- **Status:** PRE-REG ONLY this session. Full build deferred.
- **Scope:** direct level-fade, level-break-momentum, mean-reversion-within-range
- **Output:** `docs/audit/hypotheses/phase-e-non-orb-strategy-class.md`
- **Why deferred:** requires new entry model in outcome_builder + backtest framework changes; cannot ship this session

### 5.6 Phase C/D pre-registrations (DEFERRED, stubs created)

- Phase C (E_RETEST entry model) — requires `outcome_builder.py` changes, backtest rebuild
- Phase D (Carver forecast combiner) — requires `trading_app/forecast_combiner.py` new module
- Both get their own pre-reg files next session

### 5.7 Composite confluence (DEFERRED)

K-budget math: ~150 2-way combinations per lane × 6 lanes = 2,700 cells. Exceeds Bailey MinBTL. Must be separately pre-registered with theoretical narrowing to ≤ 300 hypotheses.

---

## 6. Variables DEFERRED (documented gaps)

Variables acknowledged but NOT in this session's scope. Future pre-reg required.

| Ref | Variable | Reason deferred |
|-----|----------|-----------------|
| D1 | Weekly/monthly HTF levels | Requires new `pipeline/build_daily_features.py` columns |
| D2 | Market profile (VAH/VAL/POC) | Requires new pipeline feature (Dalton PDF not in resources) |
| D3 | Cross-asset correlation features | Requires separate data pipeline (DX, ES vs NQ, gold vs bonds) |
| D4 | Tick-volume / delta | Requires Databento tick data ingestion |
| D5 | Option flow / gamma levels | Requires external data source |
| D6 | Economic calendar impact (non-NFP) | Requires structured calendar data pipeline |
| D7 | Entry geometry variants (E_RETEST, E_VWAP, dynamic stops) | Requires outcome_builder rebuild |
| D8 | Carver forecast combiner / vol targeting | Requires new infrastructure (Phase D) |

---

## 7. Institutional-grade success criteria for this session

- [x] Every trade-time-knowable feature class in §4.1-4.4 attempted or explicitly deferred
- [x] Every variable not in scope is named and the reason logged (§6)
- [ ] Alpha scan results committed with BH-FDR, T0 correlation, OOS dir-match
- [ ] Beta scan results committed for MES/MGC twins
- [ ] Every survivor passes T0-T8 battery before any deployment recommendation
- [ ] No stage-gated file touched without explicit design proposal + user confirmation
- [ ] Any deployment candidate routed through `docs/institutional/pre_registered_criteria.md` 12-criterion gate

---

## 8. Bias controls

- **Multiple testing:** BH-FDR on pre-committed K=312 (Alpha) + K=12 (Beta) = 324. Bailey MinBTL = 2·ln(324)/E[max_N]² — E[max_N] ≤ 2.8 for K=324 normal; MinBTL ~ 1.47 years. Our IS window is 2018-2025 (7+ years). Within budget.
- **Cherry-picking guard:** all cells reported, not only winners. Report tail of the distribution.
- **Post-hoc rationalization guard:** every test scope fixed BEFORE running. No post-hoc theta relaxation.
- **Survivorship bias:** include dead lanes / inactive sessions in the Beta scan so we can see "no twin exists" results.

---

## 9. Output ledger

After this session:
- `docs/audit/hypotheses/2026-04-15-comprehensive-deployed-lane-gap-audit.md` (this file)
- `docs/audit/hypotheses/2026-04-15-deployed-lane-overlay-scan-prereg.yaml` (Alpha pre-reg)
- `research/comprehensive_deployed_lane_scan.py` (scan script)
- `docs/audit/results/2026-04-15-comprehensive-deployed-lane-scan.md` (Alpha+Beta results)
- `docs/audit/results/2026-04-15-deployed-lane-overlay-t0t8-audit.md` (T0-T8 on Alpha survivors)
- `docs/audit/hypotheses/phase-c-e-retest-entry-model.md` (Phase C pre-reg stub)
- `docs/audit/hypotheses/phase-d-carver-forecast-combiner.md` (Phase D pre-reg stub)
