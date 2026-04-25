task: Ralph Loop iter 181 — R4 live_signals.jsonl daily rotation + disk-full notify
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/signal_log_rotator.py
  - trading_app/live/session_orchestrator.py
  - scripts/tools/trade_matcher.py
  - pipeline/check_drift.py
  - tests/test_trading_app/test_signal_log_rotator.py
blast_radius:
  - trading_app/live/signal_log_rotator.py (new file — SignalLogRotator helper with daily rotation, cleanup, OSError notify)
  - trading_app/live/session_orchestrator.py (_write_signal_record delegates to rotator; SIGNALS_FILE constant repurposed to dir; disk-full notify added)
  - scripts/tools/trade_matcher.py (_load_signals reads daily-suffix files instead of monolithic live_signals.jsonl)
  - pipeline/check_drift.py (new drift check: _write_signal_record must delegate to rotator not raw open())
  - tests/test_trading_app/test_signal_log_rotator.py (4 new tests covering rollover, OSError notify rate-limit, cleanup, trade_matcher compat)
updated: 2026-04-25T00:00:00+10:00
agent: ralph
