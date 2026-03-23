---
task: "Fix self-introduced bug (self._instrument) + second-pass live trading safety findings"
mode: IMPLEMENTATION
stage: 1
stage_of: 1
stage_purpose: "Fix CRITICAL self._instrument AttributeError + HIGH JK fallback NULL guard + HIGH bracket/exit safety"
updated: 2026-03-25T01:30+10:00
terminal: main
scope_lock:
  - trading_app/execution_engine.py
  - trading_app/live_config.py
  - trading_app/live/session_orchestrator.py
acceptance:
  - "execution_engine: self._instrument → self.portfolio.instrument (CRITICAL regression fix)"
  - "live_config: JK fallback NULL guards on sharpe/sample_size"
  - "session_orchestrator: bracket spec None → log.warning"
  - "No new drift"
blockers: []
---
