---
task: PR #307 code-review action items — add intentional-full-window doctrine comments at the two bare-ctor StrategyTradeWindowResolver callsites + extend backfill module docstring with the strict-IS-provenance-shelf sentence.
mode: CLOSED
closed_commit: f6dc3d2f
closed_date: 2026-05-22
closed_note: |
  Comment/docstring-only PR #307 review actions landed in `f6dc3d2f`.
  `git show --stat f6dc3d2f` confirms the three scoped files plus this
  stage file were touched, with no executable behavior change.
scope_lock:
  - trading_app/strategy_validator.py
  - pipeline/check_drift.py
  - scripts/migrations/backfill_validated_trade_windows.py
---

## Blast Radius

- trading_app/strategy_validator.py — comment-only addition at line 1738 (promotion-path bare-ctor callsite). Zero logic change. No callers affected.
- pipeline/check_drift.py — comment-only addition at line 2767 (micro-launch check bare-ctor callsite). Zero logic change. No callers affected.
- scripts/migrations/backfill_validated_trade_windows.py — extend module docstring with one sentence pointing at chordia.py:158-170 doctrine. No logic change.
- Reads: none beyond the three files above. Writes: none to gold.db or any state file.
- Tests: existing tests added in 7ca99486 still cover the contract; no test changes needed because no behavior changes.

## Acceptance

1. Three files edited with comment-only additions.
2. `python pipeline/check_drift.py` still passes.
3. No diff in any executable line — git show should report only added comment lines + docstring lines.
