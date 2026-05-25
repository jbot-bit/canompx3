---
task: Close n=4 CI pytest-timeout watchdog race class — unload plugin with `-p no:timeout` + add drift check #184 to forbid `--timeout=0` and require `-p no:timeout` in every `uv run pytest` invocation in `.github/workflows/*.yml`.
mode: IMPLEMENTATION
scope_lock:
  - .github/workflows/ci.yml
  - pipeline/check_drift.py
---

## Scope Lock

- .github/workflows/ci.yml
- pipeline/check_drift.py

Notes on scope:
- `.github/workflows/ci.yml` — replace every `--timeout=0` with `-p no:timeout` across all 9 `uv run pytest` invocations, update the explanatory comment.
- `pipeline/check_drift.py` — add `check_ci_pytest_no_timeout_plugin(workflows_dir)` function + new CHECKS list entry (Check 184).

## Blast Radius

- `.github/workflows/ci.yml` — touched 9 lines + 1 comment block. CI behavior changes: pytest-timeout plugin no longer loaded in CI, removing the `_thread.interrupt_main()` race surface that has failed the last 5+ pushes on the GH-hosted Windows runner. No test code changes. Per-step `timeout-minutes: 10` still provides genuine hang protection.
- `pipeline/check_drift.py` — adds a pattern-matching function (~55 lines) that reads `.github/workflows/*.yml` and asserts every `uv run pytest` line carries `-p no:timeout` and none carries `--timeout=0`. Adds one CHECKS entry. No imports, no canonical-source changes, no schema changes, no behavioral mutation of any existing check.
- Reads: `.github/workflows/*.yml` (read-only).
- Writes: none (drift check returns violations as strings).
- Downstream: pre-commit hook + GitHub Actions both run drift via `pipeline/check_drift.py`. New check joins existing 163-check set as Check #184.

## Class history

- n=1 (2026-05-23): `test_session_start_mutex` hung — patched with per-module `@pytest.mark.timeout(0)`. Commit `0b56ff82`.
- n=2 (2026-05-24): `test_integration_l1_l2` hung — patched by routing the file to a different CI step. Commit `7f5253c6`.
- n=3 (2026-05-24): All test steps hit the race — patched with `--timeout=0` CLI flag in every step. Commit `6ba30ec0`.
- n=4 (2026-05-25 THIS): `test_llm_hypothesis_proposer` hung despite `--timeout=0`. Root cause: `--timeout=0` disables the per-test timeout value but does NOT unload the plugin; the watchdog thread is still installed and still races capture-manager on Windows.

Per `memory/feedback_n3_same_class_doctrine_threshold.md`: at n=3+ the class warrants registry + mechanical enforcement. This stage promotes the per-instance patches to a mechanical drift check (n=3+ tier).

## Acceptance criteria

1. CI run on `main` after push is GREEN (the very run that lands this stage proves the fix).
2. `python pipeline/check_drift.py` PASSES with one new check (164 total, was 163).
3. Manual mutation test: temporarily revert one `-p no:timeout` to `--timeout=0` in `ci.yml` → drift check fires with a clear violation pointing to file:line.

## Done criteria

- tests pass (drift check output shown with violations=0, total=164)
- dead code swept (`grep -r "--timeout=0" .github/workflows/` returns nothing)
- self-review passed (described in commit body)
