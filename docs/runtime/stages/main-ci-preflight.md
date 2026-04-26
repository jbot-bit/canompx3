---
slug: main-ci-preflight
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-26
updated: 2026-04-26
task: Add main-CI-red pre-flight check to session-start.py — surfaces last completed run conclusion on origin/main at session start, preventing PR-#108-class cascade where work is sunk into a PR before discovering main is red. Single helper following existing _*_lines() idiom; repo-wide cache; silent on offline/unauth.
---

# Stage: Main-CI-red Pre-flight Hook

mode: IMPLEMENTATION
date: 2026-04-26

scope_lock:
  - .claude/hooks/session-start.py
  - .claude/hooks/tests/__init__.py
  - .claude/hooks/tests/test_main_ci_preflight.py
  - .claude/rules/pooled-finding-rule.md
  - docs/plans/2026-04-26-token-efficiency-design.md
  - docs/runtime/stages/main-ci-preflight.md

## Why

Memory file `feedback_ralph_loop_leftover_cascade.md` documents PR #108 incident: ~3 hours burned diagnosing four cascading inherited failures because main was red and Ralph-loop iterations had landed broken state that every downstream PR inherited. First diagnostic should have been `gh run list --branch main --limit 3`. Automating this at session-start eliminates the discovery lag.

Anthropic 2026 hooks documentation publishes a CI/Build Status Check SessionStart template — sanctioned pattern, not improvisation.

## Blast Radius

- **Files modified:** `.claude/hooks/session-start.py` (one new helper + one wire-up line in `main()`).
- **Files created:** `.claude/hooks/tests/__init__.py` (empty marker), `.claude/hooks/tests/test_main_ci_preflight.py` (6 mock-based scenarios).
- **Files modified (Track A bundled):** `.claude/rules/pooled-finding-rule.md` (4-line `paths:` frontmatter add).
- **Production behavior:** zero. Hook is read-only; on any failure path it returns empty list silently.
- **Drift count:** unchanged. Hook scripts are not scanned by `pipeline/check_drift.py`.
- **Pre-commit hook impact:** none. The session-start hook runs at Claude Code session-start, not at git-commit-time.
- **Reversibility:** single commit; revert via `git revert`.

## Implementation Order

1. Add `paths:` frontmatter block to `.claude/rules/pooled-finding-rule.md` (Track A).
2. Add `_main_ci_status_lines()` helper to `.claude/hooks/session-start.py`. Helper signature returns `list[str]`. Uses `_git()` wrapper for `git rev-parse --git-common-dir` to locate cache dir. Uses `subprocess.run` directly for `gh run list ... --json ...` with 5-second timeout. Cache write via `tempfile.NamedTemporaryFile(dir=common_dir, delete=False)` + `os.replace(tmp, cache_path)` for atomicity. JSON cache schema: `{"timestamp": <unix epoch>, "conclusion": "success"|"failure"|..., "run_id": <int>, "workflow": <str>}`.
3. Wire helper into `main()` — append after `_parallel_session_lines()` call, only on `session_type == "startup"` path (skip on resume/clear/compact to avoid redundant API calls).
4. Create `.claude/hooks/tests/__init__.py` (empty file marking tests as a Python package).
5. Create `.claude/hooks/tests/test_main_ci_preflight.py` with 6 pytest functions:
   - `test_cache_hit_returns_cached_without_gh_call`
   - `test_cache_miss_red_emits_warning_and_writes_cache`
   - `test_cache_miss_green_emits_confirmation`
   - `test_gh_not_installed_returns_empty_silently`
   - `test_no_completed_runs_returns_empty_silently`
   - `test_cache_stale_triggers_refresh`
6. Run drift check.
7. Run pytest.
8. Smoke-test hook with synthetic startup JSON.

## Acceptance Criteria

- `python pipeline/check_drift.py` passes (count unchanged).
- `pytest .claude/hooks/tests/test_main_ci_preflight.py -v` — all 6 tests pass.
- `echo '{"session_type":"startup"}' | python .claude/hooks/session-start.py 2>&1` — runs to exit 0, produces output that includes either a `Main CI:` line OR is silent on the CI portion (offline/unauth path).
- Cache file `<git-common-dir>/.claude.main-ci-status` is written atomically (verified by inspecting after a real call).
- No regression of existing helpers — origin-drift, env-drift, parallel-session, session-lock all still emit their existing lines.

## Adversarial-audit-gate

This does NOT touch `pipeline/check_drift.py` — drift checks are not modified. Production code (pipeline/, trading_app/) untouched. Per `.claude/rules/adversarial-audit-gate.md` the hook scope is config/tooling, evidence-auditor pre-merge dispatch is OPTIONAL not mandatory. Will run self-review after implementation per CLAUDE.md "2-Pass Implementation Method" before declaring done.
