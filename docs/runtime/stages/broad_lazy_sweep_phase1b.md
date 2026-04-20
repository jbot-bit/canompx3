---
task: Phase 1b — lazy-load pandas/numpy/duckdb in trading_app/outcome_builder.py to unlock Phase-1 deferrals
mode: IMPLEMENTATION
scope_lock:
  - trading_app/outcome_builder.py
  - docs/plans/2026-04-20-broad-lazy-import-sweep.md
  - docs/runtime/stages/broad_lazy_sweep_phase1b.md
blast_radius: trading_app/outcome_builder.py — CLI entry + importable module. 27 importers verified (trading_app.strategy_discovery, trading_app.regime.discovery, trading_app.nested.{discovery,audit_outcomes}, research/*, tests/*, scripts/tools/*, pipeline.check_drift). ALL importers use named symbols only (CONFIRM_BARS_OPTIONS, RR_TARGETS, compute_single_outcome, build_outcomes, _annotate_time_stop, _compute_outcomes_all_rr, main). Grep confirmed zero access to outcome_builder.pd / .np / .duckdb as module attributes, no `import *`. Companion tests — tests/test_trading_app/test_outcome_builder.py, test_early_exits.py, test_outcome_builder_utc.py, test_nested/test_builder.py, test_integration.py, test_integration_l1_l2.py. No schema changes, no DB writes from this refactor.
acceptance:
  - tests/test_trading_app/test_outcome_builder.py passes
  - Full companion test suite (5 files above) passes
  - python -m trading_app.outcome_builder --help still works
  - python -m pipeline.check_drift (isolated, other-terminal WT mods stashed) = 0 violations
  - warm import 5-run median: before/after measured on current OS-cache state and reported honestly
  - A/B delta reported; strategy_discovery cold import re-measured to confirm Phase-1 deferral now visible
  - one commit with honest numbers
agent: claude
---

# Stage — broad lazy-sweep Phase 1b

Plan: `docs/plans/2026-04-20-broad-lazy-import-sweep.md` § Phase 1b (implied — added after Phase 1 closure discovery that `trading_app.outcome_builder` transitively pulls pandas through its module-top import chain).

Pattern: PEP 8 delayed imports + PEP 563 `from __future__ import annotations` + PEP 484 `TYPE_CHECKING` guard. Same template as PR #24 `trading_app/ai/claude_client.py`.

Use-site audit (verified via grep):
- pd: 13 sites (annotations at 58/131/198/412; runtime at 74/158/266/271/569/767/795/796/954)
- np: 12 sites (all runtime, lines 369-388, 601-631)
- duckdb: 1 site (line 675, inside `build_outcomes`)
- No @dataclass, pydantic, or get_type_hints patterns that would clash with PEP 563.

Functions needing lazy imports:
- `_check_fill_bar_exit` (57): pd
- `_annotate_time_stop` (128): pd
- `_compute_outcomes_all_rr` (197): pd + np
- `compute_single_outcome` (411): pd + np
- `build_outcomes` (656): pd + duckdb
