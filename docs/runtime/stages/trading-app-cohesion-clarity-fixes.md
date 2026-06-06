---
task: "trading_app cohesion audit — clarity fixes (sr_monitor rename + orphan relocation)"
mode: IMPLEMENTATION
---

## Scope Lock

- trading_app/sr_monitor.py
- trading_app/live/sr_score_kernel.py
- tests/test_trading_app/test_sr_monitor.py
- research/archive/spa_test.py
- research/archive/asia_session_analyzer.py
- trading_app/spa_test.py
- trading_app/analysis/asia_session_analyzer.py
- trading_app/lifecycle_state.py
- trading_app/analysis/__init__.py

## Scope expansion note (2026-06-07)

trading_app/analysis/__init__.py REMOVED (git rm): after moving the sole module
(asia_session_analyzer.py) to research/archive/, the package held only an empty
__init__.py with zero importers — dead package per institutional-rigor §5. No
live code references `trading_app.analysis` (only auto-gen REPO_MAP.md, which
refreshes).


lifecycle_state.py ADDED to scope mid-implementation. REASON: the blast-radius
agent's import-grep missed TWO runtime *path* references to live/sr_monitor.py
(string/Path constructions, not import statements) that feed the Criterion-12
lifecycle code-fingerprint: sr_monitor.py:74 (`_sr_code_paths()`) and
lifecycle_state.py:139 (`build_code_fingerprint([...])`). Leaving these pointing
at the old (now-nonexistent) path would weaken a capital-path integrity check —
the fingerprint would hash a missing file. Both updated in lockstep with the rename.

## Blast Radius

- trading_app/live/sr_monitor.py → renamed (git mv) to live/sr_score_kernel.py. Math kernel (Pepelyshev-Polunchenko SR score recursion); confusing 1:1 name collision with root sr_monitor.py (the orchestrator). Blast: 2 IMPORT sites (trading_app/sr_monitor.py:56, tests/test_trading_app/test_sr_monitor.py:9) PLUS 2 runtime PATH-fingerprint refs (sr_monitor.py:74, lifecycle_state.py:139) — the latter feed Criterion-12 lifecycle integrity and MUST update in lockstep or the code-fingerprint hashes a missing file. No drift-check string literal references the live filename. Docs/audit-result references to the old path are historical record (NOT updated — they describe the file as it was on those dates).
- trading_app/spa_test.py → git mv to research/archive/spa_test.py. Zero importers anywhere (verified). Absolute imports only (pipeline.paths) — survive relocation. No drift-check string ref.
- trading_app/analysis/asia_session_analyzer.py → git mv to research/archive/. Zero importers. Absolute imports only (pipeline.cost_model, pipeline.paths). check_drift.py:1967 has a BASENAME-only allowlist entry (check_db_config_usage EXEMPT set) — basename unchanged by move, so exemption preserved, no drift update needed.
- LIVE_PORTFOLIO removal (audit Finding #2) is EXCLUDED from scope: blast-radius proved it is NOT dead code — 2 active drift checks + 3 scripts + 4 tests import and iterate it at runtime. Already self-documented with a DEPRECATED banner. No edit.
- Reads: gold.db (none — pure code move/rename). Writes: none. No schema, no capital path, no entry-model logic touched.

## Verification plan

- ruff check on the 3 touched py files
- pytest tests/test_trading_app/test_sr_monitor.py (proves rename import resolves)
- python -c import smoke for trading_app.sr_monitor (proves root orchestrator resolves the renamed kernel)
- python pipeline/check_drift.py (proves no drift regression from the moves)
