---
task: pyright-cluster-mechanical
mode: IMPLEMENTATION
scope_lock:
  - pipeline/check_drift.py
  - trading_app/live/bot_dashboard.py
  - tests/test_trading_app/test_session_orchestrator.py
---

# Follow-up — Pyright Cluster Mechanical Fixes

**Spun off from:** `docs/runtime/stages/deferred-bucket-audit.md` Pass 4.

**Why a separate stage:** The Pass 4 audit revealed 55 distinct Pyright
errors across three independent root causes. Pass 4 closed the highest-
signal one (F7 cancel-path capital-class invariant in
`session_orchestrator.py:2942/2959`). The remaining 53 are mechanical type
annotations with **zero behavior change**, and bundling them into a single
commit violated the per-pass discipline.

## Blast Radius

- **Annotations only.** No runtime behavior changes.
- `pipeline/check_drift.py` 18 errors — all `Optional[tuple]` unpacks from
  DuckDB `fetchone()` on `SELECT COUNT(*)` / `re.match()[group]` patterns
  where the developer knows the result is not None. Fix: inline
  `# type: ignore[index]` with reason or migrate to a typed `_scalar()`
  helper.
- `trading_app/live/bot_dashboard.py` 35 errors — three classes:
  1. `state.get(key, 0)` returning `object | None` then passed to int()/
     float(). Fix: narrow with `isinstance` or explicit cast.
  2. `_bg_processes` dict union of `Popen | file_handle` — `.close()` only
     on file_handle path. Fix: structural narrow or split the dict.
  3. `subprocess.run(...).stdout.splitlines()` typed as `list[bytes]` when
     consumer treats as str. Fix: pass `text=True` to subprocess.run or
     decode at the boundary.
- `tests/test_trading_app/test_session_orchestrator.py` 5 errors — FakeRouter/
  FakePositions test doubles don't subclass `BrokerRouter` / `BrokerPositions`.
  Fix: subclass the base classes (cleanest) or add `# type: ignore[assignment]`.

## Approach

One commit per root-cause class. Per-pass protocol same as parent stage
(PREMISE -> TRACE -> EVIDENCE -> VERDICT). Drift + pytest after each.

## Why this is NOT capital-class

None of the 53 errors flag a runtime path that can execute with the
unexpected None. The F7 capital-class path was the one that did — closed
in parent Pass 4a. The rest are type-system pedantry on operator-visible
dashboard read paths and on test-only fakes.

## Status

PENDING. Not blocking any other work. Schedule when (a) operator surfaces
a real bug rooted in one of these classes, or (b) a code-review batch
covers these files and we want green Pyright as the baseline.
