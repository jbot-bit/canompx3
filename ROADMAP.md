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

## Phase 3: Trading App

- `trading_app/config.py` — filter definitions (MGC_ORB_SIZE_FILTERS)
- `trading_app/setup_detector.py` — setup detection logic
- `trading_app/data_loader.py` — data loader with filter checking
- `trading_app/strategy_discovery.py` — backtesting engine
- `trading_app/execution_spec.py` — ExecutionSpec system
- `trading_app/entry_rules.py` — entry rule implementations

## Phase 4: Strategy Validation

- `validated_setups` table — production strategies
- `validated_setups_archive` — historical versions for audit
- `experimental_strategies` table — parallel strategy source
- 6-phase autonomous validation framework
- Strategy family isolation (ORB_L4, ORB_BOTH_LOST, ORB_RSI, ORB_NIGHT)

## Phase 5: Database/Config Sync

- `test_app_sync.py` — mandatory sync validation between DB and config.py
- Zero tolerance for mismatches
- Fail-closed enforcement

## Phase 6: Live Trading Requirements

- Boundary + state test suites (80%+ coverage for live code)
- Test templates: boundary_test_template.py, state_test_template.py
- Entry/exit logic, position sizing, risk management testing

## Phase 7: Skills System (Optional)

- `skills/code-guardian/` — auto-protection for production files
- `skills/quick-nav/` — navigation helper
- `skills/project-organizer/` — file organization
- `skills/focus-mode/` — ADHD task management
- `skills/strategy-validator/` — 6-phase validation
- `skills/database-design/` — schema guidance
- `skills/code-review-pipeline/` — multi-agent review
- `skills/brainstorming/` — feature design
- `skills/reflect.md` — session learning

---

## Rules to Enforce When Built

These rules apply ONLY when the corresponding features exist:

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
