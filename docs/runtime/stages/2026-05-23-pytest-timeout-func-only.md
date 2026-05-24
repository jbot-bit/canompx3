---
task: Fix pytest-timeout watchdog crashing pytest 9.0.2 on Windows during inter-module teardown — restrict to test-function runtime only
mode: CLOSED
scope_lock:
  - pyproject.toml
closed_note: |
  Shipped in `85ae67fa` (`fix(ci): timeout_func_only=true — stop
  pytest-timeout crashing in teardown`). Current verification on 2026-05-24:
  `./.venv-wsl/bin/python -m pytest tests/test_tools/test_codex_local_env.py
  tests/test_tools/test_codex_doctor.py tests/test_tools/test_codex_launcher_scripts.py
  tests/test_tools/test_windows_agent_launch_light.py
  tests/test_tools/test_windows_agent_launch.py -q` => 74 passed. Pytest
  header confirms `timeout func_only: True`.
---

## Scope Lock

- pyproject.toml

## Blast Radius

- `pyproject.toml` — add ONE line `timeout_func_only = true` to `[tool.pytest.ini_options]` (currently sets `timeout=120` + `timeout_method="thread"`). This restricts pytest-timeout's threaded watchdog to test-function execution only, NOT setup/teardown/inter-module transitions where `_pytest/capture.py:802` asserts `_global_capturing is not None` and crashes.
- Affects: every pytest invocation (local + CI). Pre-change: watchdog crashes on Windows GH runner at module-boundary teardown (caught after test_session_start_mutex's 6 tests passed in 9.2s, fires KeyboardInterrupt in pytest-timeout's timer thread). Post-change: watchdog only counts time inside `test_*` functions; teardown/inter-module work is excluded from the timer.
- Reads: none. Writes: none. Affects: CI test step on Windows. Tests still time-bounded — just by the test body, not the test+teardown+next-module-setup composite window.
- No production code or canonical source touched. Pure pytest configuration.

## Why this works

- The crash is in `pytest_timeout.timeout_timer` (line 518 of pytest-timeout 2.4.0) calling `capman.read_global_capture()`, which asserts `self._global_capturing is not None` (`_pytest/capture.py:802`).
- pytest 9.0.2 nulls `_global_capturing` between test modules during the capture-manager teardown. If the timer fires in that gap, the assertion blows up.
- `timeout_func_only = true` (documented at https://pypi.org/project/pytest-timeout/#timeout-func-only) starts the timer only at the `pytest_runtest_call` phase and cancels it before teardown. The timer can no longer fire during the capture-manager-null window.

## Verification

1. Commit + push directly to main (trivial config change, no PR ceremony per workflow-preferences.md trivial tier — ≤1 line, no production/schema/config-truth change).
2. Watch GH Actions CI on the resulting commit. Success = test job passes ALL the way through (the test_session_start_mutex tests already pass functionally — we just need the watchdog to stop crashing in teardown).
3. If still red, the failure mode is genuinely something else (e.g., a real hang in another module) and we revisit.
