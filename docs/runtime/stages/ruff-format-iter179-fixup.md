---
mode: TRIVIAL
task: Unblock CI for 5 open PRs (#101-#105) and stabilize local pre-commit by fixing 4 Ralph-loop leftovers — ruff format (iter-179), git env-var test isolation (iter-179), DuckDB schema-migration robustness (iter-179), and the CORRELATION_REJECT_RHO → RHO_REJECT_THRESHOLD rename in research/audit_allocator_rho_excluded.py (iter-171, broke test collection on origin/main).
created: 2026-04-25
updated: 2026-04-25
---

# Stage: iter-179 CI + pre-commit fixup

## Scope

Three related fixes, all consequences of PR #107 not being CI- or local-clean.

### 1. ruff format on 6 files

CI flagged "Would reformat" on these 6 files at origin/main commit 0d54d52e. Pure formatter pass. No semantic changes.

- scripts/tools/context_views.py
- tests/test_tools/test_claude_superpower_brief.py
- tests/test_tools/test_context_views.py
- tests/test_trading_app/test_phase_4_discovery_gates.py
- trading_app/phase_4_discovery_gates.py
- trading_app/strategy_discovery.py

### 2. git env-var test isolation in 2 classes

`TestCheckGitCleanlinessIntegration` (test_phase_4_discovery_gates.py) and `TestPhase4DiscoveryEnforcement` (test_strategy_discovery.py) both call `subprocess.run(["git", ...], cwd=tmp_path)` to drive a temp-repo workflow. When the test runner inherits `GIT_DIR` / `GIT_WORK_TREE` from a parent (pre-commit hook, CI runner with a previous `git` invocation), git ignores `cwd=` and commits land in the parent repo. Reproduced locally during this session — the literal "pre-register test" + "edit" commits landed twice on the active branch during pre-commit retries.

Fix: per-class autouse fixture `_isolate_git_env(monkeypatch)` that drops `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`, `GIT_OBJECT_DIRECTORY`, `GIT_COMMON_DIR`, `GIT_NAMESPACE`, `GIT_PREFIX`. monkeypatch auto-restores on teardown.

### 4. CORRELATION_REJECT_RHO rename leftover (iter-171)

`research/audit_allocator_rho_excluded.py` imports `CORRELATION_REJECT_RHO` from `trading_app.lane_allocator`. PR-merging Ralph iter-171 (`9809f1b8`) renamed that symbol to `RHO_REJECT_THRESHOLD` (canonicalised to `lane_correlation.py`) but missed the call site in `research/`. Test collection in `tests/test_research/test_allocator_rho_audit.py` consequently fails on every PR rebased on main.

Fix: rename both occurrences (`research/audit_allocator_rho_excluded.py:46, 330`) to `RHO_REJECT_THRESHOLD`. The lane_allocator re-exports it from lane_correlation, so the import path stays the same. 19 tests in `test_allocator_rho_audit.py` collect and pass after the rename.

### 3. DuckDB schema-migration robustness in `init_trading_app_schema`

25 ALTER-TABLE-ADD-COLUMN sites in `trading_app/db_manager.py` used the pattern `try: ALTER ... ADD COLUMN ... except CatalogException: pass`. On DuckDB ≥ 1.5 this leaves the implicit transaction in aborted state, so any subsequent ALTER on the same connection raises `TransactionException`. CI never tripped this because `uv.lock` pins DuckDB 1.4.4; locally the .venv installed 1.5.2 and `pytest tests/test_trading_app/test_strategy_discovery.py::TestZeroSampleNotWritten::test_zero_sample_strategies_not_written` failed inside pre-commit but not standalone (same env, redirect-only difference triggered it via fixture ordering).

Fix: replace every `try/except CatalogException: pass` with `ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...`. DuckDB ≥ 0.8 supports the syntax. Single-line, no try/except, no transaction-state concern, version-agnostic.

## Why TRIVIAL

- (1) is whitespace-only.
- (2) is a test-only fix.
- (3) is a strict robustness improvement — the new code is functionally identical on DuckDB < 1.5 and correctly behaved on ≥ 1.5. No tables, columns, types, or business logic change.

## Acceptance

- `ruff format --check` clean on all 7 staged production/test files.
- `ruff check` clean.
- `pytest TestCheckGitCleanlinessIntegration TestPhase4DiscoveryEnforcement` with `GIT_DIR=.git` set: HEAD unchanged, 9 tests pass.
- `pytest tests/test_trading_app/test_strategy_discovery.py tests/test_trading_app/test_phase_4_discovery_gates.py`: 78/78 pass on DuckDB 1.5.2.
- Pre-commit passes without race.
- CI green.
