---
task: Skip _ensure_repo_python() reinvocation when --help is in argv (fixes CI hang on test_system_context_script_help_runs_via_direct_path) + post-audit fixes to merged content (RULE 3.2 mis-citation, Check #134 → #136, Carver Ch 9-10 unverified absence claim)
mode: IMPLEMENTATION
scope_lock:
  - scripts/tools/system_context.py
  - docs/runtime/chordia_audit_log.yaml
  - docs/runtime/action-queue.yaml
  - docs/audit/results/2026-05-02-prior-day-context-blocker-memo.md
---

## Blast Radius

- `scripts/tools/system_context.py` — adds 2 lines at top of `_ensure_repo_python()` to early-return when `-h` or `--help` is in `sys.argv[1:]`.
- Reads: none beyond stdlib.
- Writes: none.
- Affects: only argparse `--help` invocations of the script. The bootstrap re-invocation path is preserved for all other invocations (orientation, session_start_*, log-decision).

## Why

PR #221 CI run `25321117722` failed twice (7m 36s and 9m 26s) at the same test:
`tests/test_pipeline/test_system_context.py::TestCliBootstrap::test_system_context_script_help_runs_via_direct_path`.

The test runs `python scripts/tools/system_context.py --help` as a subprocess. Locally this returns in <1s with exit 0. On the GH Actions Windows runner, pytest emits the test name then a `KeyboardInterrupt` arrives at the same microsecond.

Root cause: `_ensure_repo_python()` (line 28-47 of `scripts/tools/system_context.py`) compares `Path(sys.prefix).resolve()` to `_preferred_repo_prefix(.venv/Scripts/python.exe)`. On the GH runner uv installs into a cache-link layout where `sys.prefix` resolves to `D:\a\_temp\setup-uv-cache\environments-v2\<hash>` while `.venv/Scripts/python.exe`'s parent.parent resolves to `D:\a\canompx3\canompx3\.venv`. The two prefixes diverge → bootstrap forks `subprocess.call([.venv/Scripts/python.exe, __file__, '--help'])`. Triple-nested stdio (pytest → subprocess.run → subprocess.call) on Windows can deadlock when the inner pipe buffer fills with argparse `--help` text.

argparse `--help` is pure-Python, requires zero project imports beyond stdlib, and the bootstrap re-invocation adds nothing for help output. Skipping the fork when `--help` is requested is correct on its merits and removes the deadlock vector.

## Files

- `scripts/tools/system_context.py` — 2-line guard at top of `_ensure_repo_python()`.

## Verification

- Locally: `python scripts/tools/system_context.py --help` already works (no fork on local). Verify still works post-edit (no behavior change locally).
- Test: `tests/test_pipeline/test_system_context.py::TestCliBootstrap::test_system_context_script_help_runs_via_direct_path` runs subprocess + asserts returncode 0. Test was already passing here in pre-commit (1537 fast-tests pass); CI is the failing surface.
- CI: push fix and re-run PR #221 CI; expect green at full test suite duration (~17 min).
