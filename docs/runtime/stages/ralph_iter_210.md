---
task: Ralph Loop iter 210 — Fix all Pyright type errors in check_drift.py, bot_dashboard.py, derived_state.py
mode: CLOSED
scope_lock:
  - pipeline/check_drift.py
  - trading_app/live/bot_dashboard.py
blast_radius:
  - pipeline/check_drift.py (fetchone None-guards + CRG list assertions — no behavior change)
  - trading_app/live/bot_dashboard.py (_bg_processes dict type widened to Any; casts added at dict.get call sites — no behavior change)
updated: 2026-05-25T00:20:00+10:00
agent: codex
---

## Closeout

Closed by Codex on 2026-05-25.

Implementation was limited to `trading_app/live/bot_dashboard.py`; live Pyright
showed `pipeline/check_drift.py` and `trading_app/derived_state.py` already
clean under the scoped gate. The dashboard fix is typing-only: explicit
coercion helpers for JSON-shaped `dict[str, object]` payloads, accurate
background-process registry typing for process/log-handle values, a guarded
DuckDB `fetchone()` count read, and string coercion at UI-label/sort boundaries.

Verification:

- `uv run pyright pipeline/check_drift.py trading_app/live/bot_dashboard.py trading_app/derived_state.py`
  -> `0 errors, 0 warnings, 0 informations`
- `ruff check trading_app/live/bot_dashboard.py` -> pass
- `pytest -q tests/test_trading_app/test_bot_dashboard.py` -> 31 passed
  (run outside the Codex sandbox; inside the sandbox, the existing
  `asyncio.to_thread` subprocess test hung, while the same focused test passed
  outside the sandbox in 0.17s)
- `python pipeline/check_drift.py --quiet` -> clean, 163 passed, 20 advisory
