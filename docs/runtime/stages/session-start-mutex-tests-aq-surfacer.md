---
mode: TRIVIAL
task: Add companion tests for `_session_lock_lines` mutex (PR #138 review HIGH finding) + `_action_queue_ready_lines` surfacer (review improvement #3)
scope_lock:
  - tests/test_hooks/__init__.py
  - tests/test_hooks/test_session_start_mutex.py
  - .claude/hooks/session-start.py
acceptance:
  - 4 new pytest cases for `_session_lock_lines`: clean creates lock, held blocks, corrupted JSON still blocks, OSError-on-write warns without blocking. All pass.
  - 2 new pytest cases for `_action_queue_ready_lines`: missing file returns [], real fixture with status=ready returns ID surfaced.
  - `_action_queue_ready_lines` delegates to `pipeline.work_queue.load_queue()` per institutional-rigor §4 (no inline yaml.safe_load).
  - Drift check passes (`python pipeline/check_drift.py`).
  - Pre-commit hook passes.
updated: 2026-04-26
---

# Session-start mutex tests + action-queue surfacer

## Why

PR #138's body claimed "4 scenarios tested: clean / held / corrupted / 10x TOCTOU race" but the merged commit added zero test files (verified via `git show --stat e306284b`). The mutex code at `.claude/hooks/session-start.py:249-339` is now load-bearing: every Claude session start runs it, and it's the only line of defense against the `cannot lock ref 'HEAD'` race documented in `memory/feedback_shared_worktree_concurrent_commits.md`. Untested = regression bait.

Separately, PR #140 only existed because a stale `status: ready` action-queue item sat undiscovered for 2 days. Surfacing ready items at session start prevents the next #140-style miss.

## Approach

**Stage 1: Test the mutex.**

New file `tests/test_hooks/test_session_start_mutex.py`. 4 cases per acceptance. Pattern:
- `tmp_path` + `subprocess.run(["git", "init"])` for an isolated `.git` directory.
- Load the hook module via `importlib.util.spec_from_file_location` (filename has hyphen, can't import normally).
- `monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)` — safe per blast-radius report (the constant is only consumed in `subprocess.run(cwd=PROJECT_ROOT)` calls, evaluated at call time).
- For OSError case: `monkeypatch.setattr(hook.os, "open", side_effect)` selectively raises only when `flags & os.O_EXCL` is set, leaving non-mutex `os.open` calls intact.

**Stage 2: Action-queue surfacer.**

In `.claude/hooks/session-start.py`:
- New function `_action_queue_ready_lines()` that calls `pipeline.work_queue.load_queue(action_queue_path)` and emits `f"  Action queue READY: {ids}"` only when the result has items with `status == "ready"`.
- Wrapped in `try/except BaseException` matching the existing pattern of other lenient hook helpers — if `pipeline.work_queue` import fails or YAML is malformed, return `[]` (graceful degrade — must never block session start).
- Call site: `main()` line 456 area, after `_parallel_session_lines()`.

Tests in same file: missing file → `[]`; valid fixture with `status: ready` item → list contains ID.

## Blast radius

Per blast-radius report:
- **Callers of session-start.py:** zero (invoked only by Claude runtime).
- **Importers:** zero.
- **CI:** doesn't call this hook directly.
- **Drift checks:** none reference session-start or the mutex.
- **TEST_MAP in post-edit-pipeline.py:** doesn't apply (hooks are not pipeline/trading_app code).
- **Parallel readers of action-queue.yaml** (system_context, project_pulse, compact_handoff, checkpoint_guard): none emit a session-start "ready" line, no duplication.

## Canonical-source compliance

- Action-queue parser: `pipeline.work_queue.load_queue()` — single source of truth for the YAML schema (`StrictModel`, `extra="forbid"`, `QueueStatus` literal). Per institutional-rigor §4: never re-encode logic that already exists in a canonical source.

## Out of scope

- Stop-hook auto-release of `.claude.pid` (review improvement #2). Skipped: design intent at session-start.py:256-261 explicitly rejects auto-cleanup; my proposed PID-guard wouldn't work because session-start and Stop hooks are separate processes (no shared PID). Manual cleanup remains the documented safe default.
- Aggressive zombie-branch sweep / `clean_gone` skill — separate session.
