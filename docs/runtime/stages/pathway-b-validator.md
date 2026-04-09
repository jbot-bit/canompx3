---
task: "Pathway B validator support (Amendment 3.0)"
mode: IMPLEMENTATION
stage: 1/1
scope_lock:
  - trading_app/strategy_validator.py
  - trading_app/hypothesis_loader.py
  - tests/test_trading_app/test_hypothesis_loader.py
  - tests/test_trading_app/test_strategy_validator.py
blast_radius: "FDR hard gate in strategy_validator.py — the single most important data-snooping guard. hypothesis_loader.py already changed (testing_mode field). No pipeline/ changes."
acceptance:
  - "hypothesis_loader exposes testing_mode at top level (4 tests pass)"
  - "Pathway B gate: raw p < 0.05 + positive sharpe_ann passes strategies"
  - "Pathway B direction gate: negative sharpe_ann rejected"  
  - "Pathway A (BH FDR) unchanged when testing_mode=family"
  - "Full test suite passes with zero regressions"
  - "Drift checks pass (pre-existing check 59 only failure)"
updated: 2026-04-09T12:30:00+10:00
---
