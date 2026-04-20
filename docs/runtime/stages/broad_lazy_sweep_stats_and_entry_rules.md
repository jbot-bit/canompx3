---
task: Phase 2 (live_config chain) — lazy-load pipeline/stats.py (988ms culprit) + trading_app/entry_rules.py (outcome_builder cascade)
mode: IMPLEMENTATION
scope_lock:
  - pipeline/stats.py
  - trading_app/entry_rules.py
  - docs/plans/2026-04-20-broad-lazy-import-sweep.md
  - docs/runtime/stages/broad_lazy_sweep_stats_and_entry_rules.md
blast_radius: pipeline/stats.py — 65 LOC, 2 functions (per_trade_sharpe, jobson_korkie_p), module-top imports pandas+numpy+scipy costing ~988ms cold. Only 2 real importers (trading_app/live_config.py line 31, scripts/tools/select_family_rr.py line 42) — both use named-symbol import (`from pipeline.stats import jobson_korkie_p`), zero module-attribute access. trading_app/entry_rules.py — module-top imports pandas+numpy, binding-constraint for outcome_builder/strategy_discovery/strategy_validator cascade; 16 pd sites + 4 np sites. Importers verified zero module-attribute access. NO companion tests file for pipeline/stats.py (covered indirectly by live_config tests and strategy_fitness tests). Companion tests for entry_rules — tests/test_trading_app/test_entry_rules.py (if exists), test_outcome_builder.py (cascades through it). No schema changes, no DB writes.
acceptance:
  - pipeline/stats.py cold import <0.1s (was 0.988s)
  - trading_app/live_config cold import <0.5s (was 1.363s)
  - trading_app/entry_rules cold import <0.1s (was 0.435s)
  - outcome_builder / strategy_discovery cascade no longer pulls pandas transitively
  - jobson_korkie_p + per_trade_sharpe unchanged behaviorally (same numeric output for known inputs)
  - tests/test_trading_app/test_outcome_builder.py + test_strategy_discovery.py + test_strategy_validator.py + test_build_daily_features.py all pass
  - test_entry_rules if exists passes
  - python -m pipeline.check_drift (isolated) = 0 violations
  - two commits (stats, entry_rules)
agent: claude
---

# Stage — Phase 2 (live_config chain) + Phase 3b (entry_rules cascade)

Two small surgical refactors unlocking the final transitive bindings.
Pattern: PEP 563 + TYPE_CHECKING + function-body lazy imports (same as outcome_builder / build_daily_features).

## pipeline/stats.py
- `from __future__ import annotations` already present
- Add `from typing import TYPE_CHECKING`
- Move `import numpy as np` + `from scipy import stats` + `import pandas as pd` into TYPE_CHECKING
- `per_trade_sharpe` uses only Series methods (no runtime pd/np needed)
- `jobson_korkie_p` needs `import numpy as np` + `from scipy import stats` at body top

## trading_app/entry_rules.py
- Add `from __future__ import annotations` + TYPE_CHECKING block
- Find each function with runtime pd/np use, add lazy imports
- Unlock downstream: outcome_builder, strategy_discovery, strategy_validator chain
