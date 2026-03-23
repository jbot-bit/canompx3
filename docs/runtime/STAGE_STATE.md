---
task: "Threshold-grounding hardening — annotations + DOUBLE_BREAK proximity warning"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Add missing @research-source / @sensitivity-tested annotations to all rolling/regime thresholds. Add DOUBLE_BREAK proximity advisory. No threshold value changes."
updated: 2026-03-23T19:00+10:00
terminal: main
scope_lock:
  - scripts/infra/rolling_eval.py
  - trading_app/strategy_validator.py
  - trading_app/rolling_portfolio.py
  - trading_app/strategy_fitness.py
  - HANDOFF.md
acceptance:
  - "All 9 thresholds have @research-source and @sensitivity-tested annotations"
  - "DOUBLE_BREAK_THRESHOLD has proximity warning (DOUBLE_BREAK_PROXIMITY_WARN=0.05)"
  - "No threshold values changed"
  - "No live portfolio impact"
proven:
  - "DOUBLE_BREAK cliff: CME_PRECLOSE 63.9%, threshold 67%, margin 3.1pp"
  - "All other thresholds robust to +-20%"
  - "rolling_eval.py already patched (annotation + proximity warn)"
unproven: []
blockers: []
---
