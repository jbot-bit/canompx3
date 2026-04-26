mode: TRIVIAL
task: Burn down 4 debt items flagged in PR #125 + #126 follow-up — production-code fixes that replace the test-only workarounds with proper structural fixes.

## Scope lock

- `scripts/tools/task_route_packet.py`
- `scripts/tools/project_pulse.py`
- `scripts/tools/worktree_manager.py`
- `.githooks/pre-commit`
- `tests/test_pipeline/test_work_queue.py`
- `tests/test_tools/test_project_pulse.py`

## Why TRIVIAL

- All four are surgical structural fixes (move a function call site, add a parameter with default, add `env=` kwarg to subprocess.run, add `env -u` to a bash invocation).
- No business logic touched — no pipeline, no trading_app, no canonical sources, no schema, no DB, no validation thresholds.
- All four already have test coverage; the production fix makes those tests pass without monkeypatch/fixture workarounds.
- Failure surface is bounded: import side-effects that already misbehaved in PR #125's hook context, plus test-env contamination already documented.

## Acceptance

- 77/77 tests in test_work_queue.py + test_project_pulse.py pass without the `_scrub_git_env` fixture or `_utc_now` monkeypatch.
- Hook smoke test still passes: `echo '{"session_type":"startup"}' | python .claude/hooks/session-start.py` prints the `Origin:` drift line.
- Pre-commit hook still runs all 8 checks green.
