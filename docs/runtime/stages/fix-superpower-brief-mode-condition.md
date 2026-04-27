---
slug: fix-superpower-brief-mode-condition
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-27
updated: 2026-04-27
task: Fix inverted `mode != "session-start"` condition in `scripts/tools/claude_superpower_brief.py` (introduced by PR #149 commit `45fca4bb`). Test `test_build_brief_renders_high_signal_sections` calls `mode="session-start"` and asserts `"Memory topics: Trading | Tooling"` is in the brief — but the code only renders Memory topics when mode is NOT session-start. Pure 2-character fix (`!=` → `==`) in two call sites. Aligns code with test spec; unblocks main CI.
---

# Stage 1: fix superpower brief mode condition

scope_lock:
  - scripts/tools/claude_superpower_brief.py
  - HANDOFF.md
  - docs/runtime/stages/fix-superpower-brief-mode-condition.md

## Blast Radius

- `scripts/tools/claude_superpower_brief.py` — 2 sites, removed the mode gate entirely. The 4 tests in `tests/test_tools/test_claude_superpower_brief.py` exercise three modes (`session-start`, `post-compact`, `interactive`) and ALL expect memory rendering (topics or notes) in their assertions — so memory topics + recent notes should render unconditionally. Original `mode != "session-start"` was wrong for session-start (per failing test 1); my first attempt `mode == "session-start"` was wrong for the other two modes. Correct fix: no gate.
- Test `tests/test_tools/test_claude_superpower_brief.py::test_build_brief_renders_high_signal_sections` — currently FAILING on main, will PASS after this fix. No change to the test file.
- Downstream consumer: `.claude/hooks/session-start.py` calls `build_brief(root, mode="session-start")` for the session-start brief. Memory topics will now appear in the session-start surface — desired behavior.

## Why

PR #149 (`45fca4bb`, "chore(claude): add token hygiene workflow") added Memory topics + recent notes rendering to `claude_superpower_brief.py` but inverted the mode gate. The companion test was added with the correct expectation but the code shipped with the wrong condition, breaking the test on main and cascading failures into all PRs branched from main since.

Confirmed via blame: lines 139, 237 both authored by `45fca4bb`. Test in `tests/test_tools/test_claude_superpower_brief.py` (also added by PR #149) expects `mode="session-start"` to produce Memory topics output.

## Acceptance

- `python -m pytest tests/test_tools/test_claude_superpower_brief.py -q` passes (3 tests).
- `python pipeline/check_drift.py` clean.
- Pre-commit hooks pass.
