---
task: HoldToKill canonical-field fix — kill modal currently always skipped (fail-OPEN)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/live/bot_dashboard.html
  - trading_app/live/bot_dashboard.py
  - tests/test_trading_app/test_bot_dashboard_holdtokill.py
---

## Blast Radius
- `trading_app/live/bot_dashboard.html` — patch HoldToKill `_countOpenPositions` (line 4544-4556) + hero state callsite (line 4291). Both reference non-existent fields `position_qty` / `open_position`.
- `trading_app/live/bot_dashboard.py` — add `_count_open_positions_from_state(state: dict) -> int | None` helper. Pure read-only on the state dict shape. Zero new routes.
- `tests/test_trading_app/test_bot_dashboard_holdtokill.py` — NEW pytest file exercising the helper against representative `bot_state.json` fixture shapes.
- Reads: in-process state dict only (no new I/O). Writes: none. Routes: none added.
- Existing 36 cockpit tests are unaffected (helper is purely additive; HTML change is a within-function field rename).

## Why
Audit CRITICAL #2 (Gap #4 in plan adversarial table) discovered that `position_qty` and `open_position` fields do NOT exist anywhere in `bot_state.build_state_snapshot` (verified via grep + reading bot_state.py:151-237). The canonical lane status field is `lane["status"]` ∈ `{"WAITING","ARMED","IN_TRADE","FLAT"}` (bot_state.py:166,190,193,200). `_countOpenPositions` therefore returns 0 unconditionally, so the kill-switch confirmation modal is ALWAYS skipped — kill fires without operator confirmation regardless of whether positions are open. Capital-class fail-OPEN on the kill path.

## Fix shape
1. Add Python helper `_count_open_positions_from_state(state: dict) -> int | None` in `bot_dashboard.py` near `_orb_levels_for_instrument`. Returns:
   - `None` when state is missing or `present=False` (modal SHOULD show — fail-closed UX path)
   - `0` when state present but no lane has `status == "IN_TRADE"`
   - `n` (positive integer) when n lanes have `status == "IN_TRADE"`
2. Mirror the truth in HTML `_countOpenPositions()` (line 4544-4556): iterate `state.lanes`, count where `l.status === "IN_TRADE"`. Match the Python tri-state: missing state → return `null`, signaling caller to OPEN the modal (safe default).
3. Patch `_fire()` (line 4537-4542): `openCount === null || openCount > 0` opens the modal; only `openCount === 0` proceeds to direct `_sendKill()`.
4. Patch hero-state line 4291: change `l.open_position || l.position_qty` to `l.status === "IN_TRADE"`.
5. New pytest (`test_bot_dashboard_holdtokill.py`) exercises the Python helper with 4 fixture shapes: empty/missing state, all-WAITING, one-IN_TRADE, two-IN_TRADE.

## Canonical sources
- Lane status taxonomy: `trading_app/live/bot_state.py:166,190,193,200` — `WAITING`/`ARMED`/`IN_TRADE`/`FLAT`.
- State dict shape: `trading_app/live/bot_state.py:220-237` (`build_state_snapshot` return).

## Acceptance
- `python -m pytest tests/test_trading_app/test_bot_dashboard_holdtokill.py -v` → all 4 fixture cases pass
- Existing 36 cockpit tests still pass (`tests/test_trading_app/test_bot_dashboard_*.py`)
- `python pipeline/check_drift.py` → all guardrails pass
- Adversarial-audit dispatched on the commit (evidence-auditor — kill-switch is exposure-creating per `.claude/rules/adversarial-audit-gate.md`)

## Stage close
Toggle `[ ]` → `[x]` for **Stage 1** in the cockpit-v4 plan ledger. /clear-safe to stop here; Stage 2 (CSRF middleware) is independent.
