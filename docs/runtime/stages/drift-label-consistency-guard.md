---
task: drift-label-consistency-guard
mode: IMPLEMENTATION
agent: claude-opus-4-7
updated: 2026-04-20T00:00:00Z
worktree: .worktrees/drift-label-consistency-guard
branch: fix/drift-label-consistency-guard
scope_lock:
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift.py
  - docs/runtime/stages/drift-label-consistency-guard.md
blast_radius:
  - pipeline/check_drift.py — add module-level _assert_slow_labels_valid() guard after SLOW_CHECK_LABELS definition
  - tests/test_pipeline/test_check_drift.py — 4 new tests in TestSlowCheckLabelsConsistency class
  - callers of check_drift module — guard runs at import; clean repo already satisfies so behavior unchanged on landing
  - pre-commit hook, CI, post-edit hook — all subprocess check_drift.py; gain fail-closed drift detection for slow-label set
---

# Stage — drift-label-consistency-guard

Classification: non-trivial (edits pipeline/check_drift.py which is on the NEVER_TRIVIAL list).
Worktree: `.worktrees/drift-label-consistency-guard`
Branch: `fix/drift-label-consistency-guard` based on `origin/main` @ e4fb8432.

## Task

Add a startup consistency guard asserting that every label in
`SLOW_CHECK_LABELS` also appears in `CHECKS`. Without this guard, renaming
a check label in `CHECKS` silently removes it from the `--fast` skip set;
fast mode would then run a slow check every post-edit hook invocation,
exceed the 30s timeout, and fail silently.

Source: 2026-04-20 code review Section D MEDIUM finding on commit
f22052ad. Institutional-rigor rule #4 ("Delegate to canonical sources —
never re-encode") — SLOW_CHECK_LABELS is a parallel registry of label
strings duplicating CHECKS[0]; a load-time subset assertion makes drift
impossible.

## Approach

Module-level function `_assert_slow_labels_valid()` called at import
time, immediately after `SLOW_CHECK_LABELS` definition. Computes
`SLOW_CHECK_LABELS - {label for label,*_ in CHECKS}`; raises
RuntimeError with diagnostic if non-empty. Does NOT assert the reverse
direction (SLOW_CHECK_LABELS is intentionally a proper subset).

## Self-check

- Happy path: all 19 labels match → no raise. ✓ (verified on clean origin/main — 0 stale labels).
- Label renamed: stale entry → RuntimeError naming the orphan. ✓
- Label typo: same as rename. ✓
- CHECKS emptied (edge): all labels stale → RuntimeError. ✓

## Acceptance

1. `python pipeline/check_drift.py` exits 0.
2. `python pipeline/check_drift.py --fast` exits 0.
3. `pytest tests/test_pipeline/test_check_drift.py -v` passes, including 4
   new TestSlowCheckLabelsConsistency tests.
4. `scripts/tools/audit_behavioral.py` passes.
5. Self-review (Bloomey) grades B+ or better.

## Out of scope

- Automating SLOW_CHECK_LABELS regeneration from profile output.
- Changing --fast threshold (0.3s) or hook timeout.
- Any MGC research work (HANDOFF explicitly closed that thread).
