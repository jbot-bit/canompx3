---
task: PR #310 CI fix — apply ruff format to 10 files flagged by Pipeline CI
mode: TRIVIAL
slug: 2026-05-22-pr310-ruff-format-fix
parent_pr: 310
---

## Task

Pipeline CI `Ruff format check` failed on PR #310 (Stage 1a+1b multi-profile lane allocation) with 10 files flagged as `Would reformat`. Apply `ruff format` to those 10 files, verify clean, push to branch `session/joshd-multi-profile-lane-allocation`.

## Scope Lock

- pipeline/canonical_inline_copies.py
- pipeline/check_drift.py
- scripts/research/fast_lane_trial_ledger.py
- scripts/tools/allocation_intel.py
- tests/test_pipeline/test_check_drift_lane_allocation_grep_gate.py
- tests/test_trading_app/test_prop_profiles.py
- trading_app/lane_allocator.py
- trading_app/live/session_orchestrator.py
- trading_app/opportunity_awareness.py
- trading_app/prop_profiles.py

## Blast Radius

- Pure formatting (whitespace / line wrapping). Zero logic change. `ruff format` is deterministic per pyproject.toml config.
- Reads: none. Writes: 10 files listed above.
- Live-trading path (`session_orchestrator.py`) touched but formatting-only — no broker behavior change.
- Tests: existing test files reformatted; behavior unchanged.

## Verification

1. `ruff format pipeline/ trading_app/ scripts/ tests/` — applies the formatting.
2. `ruff format --check pipeline/ trading_app/ scripts/ tests/` — confirms clean.
3. `python pipeline/check_drift.py` — drift still passes (formatting is invisible to drift checks).
4. `git diff --stat` — confirms only the 10 expected files changed.
5. Push to `session/joshd-multi-profile-lane-allocation` and watch PR #310 CI re-run.

## Acceptance

- [ ] All 10 files pass `ruff format --check`
- [ ] Drift unchanged
- [ ] Commit + push + PR #310 CI green (or down to known mutex hang only)
