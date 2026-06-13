---
task: Self-classifying C11/C12 control-state staleness message (+ doctrine belt)
mode: IMPLEMENTATION
scope_lock:
  - trading_app/derived_state.py
  - trading_app/account_survival.py
  - trading_app/lifecycle_state.py
  - scripts/tools/project_pulse.py
  - .claude/rules/c11-c12-staleness-is-expected.md
  - tests/test_trading_app/test_lifecycle_state.py
  - tests/test_trading_app/test_account_survival.py
---

## Blast Radius

- `trading_app/derived_state.py` — NEW pure helper `classify_state_reason`. This file
  is a member of `_criterion11_code_paths`, so editing it changes the C11 code-fingerprint
  by design and invalidates the current cached C11 PASS. EXPECTED fail-closed staleness,
  not a regression — operator regens once post-merge via `refresh_control_state`.
- `trading_app/account_survival.py` — 2 BLOCKED-message strings consume the classifier
  (`:1659-1660`, `:1664`). Message TEXT only; no logic change.
- `trading_app/lifecycle_state.py` — re-export `classify_state_reason` if project_pulse
  needs it surfaced (project_pulse imports lifecycle_state, not derived_state).
- `scripts/tools/project_pulse.py` — survival `summary` (`:1417`) + C12 branch
  (`:1469-1486`) reframed. TEXT only; `category`/`severity`/`action` UNCHANGED.
- Reason literals LEFT UNCHANGED — classification is additive, so reason-pinning tests stay green.
- No schema change, no DB write, no live-execution-path logic change.
- New doctrine `.claude/rules/c11-c12-staleness-is-expected.md` — loads by existing manual convention.

## Approach

Single canonical `classify_state_reason(reason) -> (klass, guidance)` in `derived_state.py`
(lowest in import chain, no cycle). Owns BOTH the envelope vocabulary AND the read-layer
reasons. Exact-match on fixed reasons + prefix-match on interpolated families (substring-trap
guard). EXPECTED_STALE → "regen, don't debug"; DEFECT → "regen first; investigate if it
persists". Advisory metadata only — NOTHING gates regen on it.
