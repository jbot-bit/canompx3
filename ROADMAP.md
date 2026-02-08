# ROADMAP.md

Features planned but NOT YET BUILT. Move items to CLAUDE.md as they are implemented.

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
- `trading_app/entry_rules.py` — detect_confirm + resolve_entry E1/E2/E3 (30 tests)
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

- Grid: 6 ORBs x 6 RRs x 5 CBs x 12 filters = 2,160 combos
- orb_outcomes: 229,770 rows | experimental: 2,160 | validated: 252
- CB5 dominates (122/252), win rate monotonic CB1=39% -> CB5=53%
- Year-over-year analysis: all top strategies stable or improving
- MARKET_PLAYBOOK.md: comprehensive empirical findings

## Phase 5b: Entry Model Fix + Risk Floor + Win PnL Fix — DONE

- 3 entry models: E1 (next bar open), E2 (confirm close), E3 (limit-at-ORB retrace)
- entry_rules.py: ConfirmResult + detect_confirm() + resolve_entry() + _resolve_e1/e2/e3
- Risk floor: tick-based (10 ticks * 0.10 = 1.0pt) + outcome-based (median/avg_risk_points)
- Win PnL bug fix: changed pnl_points_to_r -> to_r_multiple (friction was missing from wins)
- test_trader_logic.py: 24 trader/math sanity checks
- Grid expanded: 6 ORBs x 6 RRs x 5 CBs x 12 filters x 3 EMs = 6,480 combos
- DB rebuild COMPLETE: 689,310 outcomes, 6,480 strategies, 312 validated
- Validator: exclude_years + min_years_positive_pct params added

## Phase 6: Live Trading Preparation — IN PROGRESS

### 6a. Strategy Portfolio Construction — DONE
- `trading_app/portfolio.py` — diversified selection, position sizing, capital estimation (22 tests)
- PortfolioStrategy/Portfolio dataclasses with JSON serialization
- compute_position_size() standard + prop firm modes
- diversify_strategies() with max per ORB + max per entry model
- correlation_matrix() from daily R-series
- estimate_daily_capital() for risk budgeting

### 6b. Execution Engine — DONE
- `trading_app/execution_engine.py` — bar-by-bar state machine (19 tests)
- State: ARMED -> CONFIRMING -> ENTERED -> EXITED
- ORB detection, break detection, confirm bar counting
- E1/E2/E3 entry model resolution
- Target/stop/session-end exit handling
- Ambiguous bar = conservative loss
- armed_at_bar guard: E1/E3 never fill on the confirm bar itself

### 6c. Risk Management — DONE
- `trading_app/risk_manager.py` — RiskLimits + RiskManager (22 tests)
- Circuit breaker (daily loss limit)
- Max concurrent positions
- Max per ORB positions
- Max daily trades
- Drawdown warning threshold

### 6d. Paper Trading / Simulation — DONE
- `trading_app/paper_trader.py` — historical replay with journal (9 tests)
- replay_historical(): feeds bars_1m through ExecutionEngine + RiskManager
- JournalEntry/DaySummary/ReplayResult dataclasses
- Risk rejection tracking
- CLI for replay runs

### 6e. Monitoring & Alerting — TODO
- Strategy performance tracking (live vs backtest drift)
- Alert on: drawdown exceeding historical, win rate divergence, ORB size regime shift
- Dashboard for live strategy status

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
