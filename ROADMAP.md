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

- `trading_app/config.py` — 8 filters in ALL_FILTERS registry (34 tests)
- `trading_app/execution_spec.py` — ExecutionSpec dataclass (17 tests)
- `trading_app/entry_rules.py` — confirm bars detection (15 tests)
- `trading_app/db_manager.py` — 4 trading_app tables (8 tests)
- `trading_app/outcome_builder.py` — pre-compute outcomes for RR×CB grid (14 tests)
- `trading_app/setup_detector.py` — filter daily_features by conditions (7 tests)
- `trading_app/strategy_discovery.py` — grid search 864 combos (11 tests)
- `trading_app/strategy_validator.py` — 6-phase validation framework (17 tests)
- Integration test: end-to-end outcome→discovery→validation (5 tests)

## Phase 4: Database/Config Sync — DONE

- `tests/test_app_sync.py` — 28 sync tests (ORB_LABELS, ALL_FILTERS, grid params, schema columns)
- Drift check 12: config filter_type sync enforcement
- Zero tolerance for mismatches, fail-closed

## Phase 5: Live Trading Requirements

- Run trading_app pipeline on real gold.db data
- Boundary + state test suites (80%+ coverage for live code)
- Entry/exit logic, position sizing, risk management testing

## Phase 6: Skills System (Optional)

- `skills/code-guardian/` — auto-protection for production files
- `skills/strategy-validator/` — 6-phase validation skill
- `skills/focus-mode/` — ADHD task management

---

## Rules to Enforce (ACTIVE)

### Strategy Family Isolation
- Cross-family inference FORBIDDEN
- Each family is a memory container
- Families: ORB_L4, ORB_BOTH_LOST, ORB_RSI, ORB_NIGHT

### daily_features RR=1.0 Warning
- Outcomes in daily_features are RR=1.0 ONLY
- NEVER use for RR > 1.0 backtesting
- Use strategy_discovery.py for higher RR

### Database/Config Sync Protocol
- NEVER update validated_setups without updating config.py
- Run test_app_sync.py after every change
- Zero tolerance for mismatches
- Drift check 12 catches filter_type mismatches automatically
