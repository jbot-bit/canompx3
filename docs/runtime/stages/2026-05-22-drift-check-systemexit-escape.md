---
task: TRIVIAL — drift check exits 2 at Check 19 because `scripts.tools.context_views._ensure_repo_python` raises `SystemExit(2)` when imported from system python (re-exec path), and `check_all_imports_resolve` uses `except Exception` which does NOT catch `BaseException` subclasses like `SystemExit`. Imports of `scripts.tools.context_views` exist in `trading_app/ai/openrouter_runtime.py:16` and `trading_app/ai/research_packet.py:14`, so any `pipeline/check_drift.py` run from a non-`.venv` python crashes mid-Check-19 with no violations report. Fix: widen the inner `except Exception` to `except BaseException` so the import failure becomes a recorded violation rather than a fatal exit. Surfaces as drift exit 2 only on out-of-venv runners — operationally a P0 since drift is the canonical pre-commit guardrail.
mode: CLOSED
closed_date: 2026-05-22
closed_note: |
  Drift now exits 0 (158 PASSED, 0 violations, 20 advisory). CLI invocation
  `python scripts/tools/context_views.py --view trading --format markdown` still
  renders the trading view normally. 25 drift-related tests pass.
  Two-layer fix:
    (1) pipeline/check_drift.py:1296 — widened `except Exception` to
        `except BaseException` so import-time SystemExit becomes a recorded
        violation instead of crashing the drift run.
    (2) scripts/tools/{context_views,context_resolver}.py — `_ensure_repo_python`
        now early-returns when `__name__ != "__main__"`, mirroring the existing
        pytest guard. Bootstrap re-exec only runs when invoked as a script.
  Test file in original scope_lock not added (no new test created — verified by
  drift count delta and CLI smoke). The bug was a class of "import-time
  side-effect raises BaseException" and the fix is to (a) make the side effect
  __main__-only and (b) make the drift check robust to anything an arbitrary
  import might raise.
original_mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
  - scripts/tools/context_views.py
  - scripts/tools/context_resolver.py

## Blast Radius

- `pipeline/check_drift.py:1296` — change `except Exception as e` to `except BaseException as e` inside `check_all_imports_resolve`. Recorded violation format unchanged (`{module}: {err_type}: {msg[:100]}`). Only behavioral change: SystemExit / KeyboardInterrupt / GeneratorExit from a module's import side-effects now appear as a violation instead of crashing the drift run.
- No other call sites. Function is invoked once from the check registry at line 13185.
- Reads: none. Writes: none. No DB touch. No schema change.

## Done criteria

1. `python pipeline/check_drift.py` runs to completion (no exit 2 from this code path).
2. The Check 19 line shows either a violation row for `trading_app.ai.openrouter_runtime` (recording the SystemExit) OR continues past it cleanly.
3. Total check count matches recent main (157 PASSED per HANDOFF 2026-05-22).
---
