---
task: Hardcore test-quality verification — mutation-test the capital-critical core (~7 modules) to grade the TESTS, not the code. Tooling = cosmic-ray (NOT mutmut — mutmut 3.x hard-refuses native Windows on every subcommand, issue #397, which would pin mutation testing to a non-portable WSL/.venv-wsl/Python-3.12 island; cosmic-ray runs natively on Windows+Linux+CI on this repo's Python 3.11.9 and mutates in-place so there is no `mutants/` tree colliding with check_drift.py's rglob). Scope is TEST + CONFIG + DOCS only — NO production-logic change. If a surviving mutant reveals a real production bug, that is a separate STOP-and-surface design-gate item, not an auto-fix. DoD rests only on mechanical checkers (cosmic-ray score, pytest on 3.11.9, Hypothesis, check_drift, evidence-auditor as hypothesis-generator only).
mode: IMPLEMENTATION
original_mode: IMPLEMENTATION
scope_lock:
  - pyproject.toml
  - .gitignore
  - scripts/tools/run_mutation.sh
  - tests/test_pipeline/test_cost_model.py
  - tests/test_pipeline/test_dst.py
  - tests/test_trading_app/test_account_survival.py
  - tests/test_trading_app/test_strategy_fitness.py
  - tests/test_trading_app/test_execution_engine.py
  - tests/test_trading_app/test_lifecycle_state.py
  - tests/test_pipeline/test_build_daily_features.py
  - docs/audit/2026-06-07-mutation-testing-capital-core.md
  - docs/runtime/stages/2026-06-07-mutation-testing-capital-core.md

## Blast Radius

- `pyproject.toml` — add `cosmic-ray>=8.4,<9` + `hypothesis>=6,<7` to both dev declarations (`[project.optional-dependencies].dev` and `[dependency-groups].test`, kept in sync). Dev-tooling only; no runtime/production import path touched.
- `.gitignore` — add `.mutation/` and `.hypothesis/` (cosmic-ray session sqlite + Hypothesis example DB); ephemeral, must never reach a commit or a peer worktree (multi-terminal-shared-file-hygiene).
- `scripts/tools/run_mutation.sh` — NEW. Portable per-module runner (config gen → baseline → init → exec → cr-rate/cr-report → cleanup trap). Repo-relative paths only; no machine/WSL assumptions.
- `tests/...` — EXTEND existing companion test files with killing tests for surviving mutants. No production code under `pipeline/`/`trading_app/` is edited.
- `docs/audit/2026-06-07-mutation-testing-capital-core.md` — NEW. Per-module before/after scores + survivor triage table (the durable record).

## Verification

- pytest tests/ green on Python 3.11.9 (canonical runner), output shown.
- `python pipeline/check_drift.py` exit 0 (run AFTER `.mutation/` removed).
- cosmic-ray survival rate per module meets the bar (capital-arithmetic ≥90% killed on covered lines; state-machine/large ≥80% with residuals justified).
- New Hypothesis property tests pass and demonstrably kill residual mutants.
- evidence-auditor dispatched as hypothesis-generator only — never a sign-off.
