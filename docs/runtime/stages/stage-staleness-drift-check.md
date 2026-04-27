---
slug: stage-staleness-drift-check
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-28
updated: 2026-04-28
task: Add advisory drift check `check_stage_file_landed_drift` to pipeline/check_drift.py — surfaces stage files whose `updated:` field is >7 days old AND have ≥3 commits referencing the slug since. Catches the "stage file claims Stage 2 in progress while commits show Stage 8 done" failure mode discovered during the 2026-04-28 cleanup pass. ADVISORY only; never blocks. Operator-gated stages are exempt via body-text scan.
---

# Stage 1: stage-staleness drift check (hardening, single-stage)

## Scope Lock

- pipeline/check_drift.py
- docs/runtime/stages/stage-staleness-drift-check.md

## Blast Radius

- `pipeline/check_drift.py` — additive: one new function
  `check_stage_file_landed_drift()` placed after `check_magic_number_rationale`
  and one new entry in `CHECKS` registered as advisory
  (`is_advisory=True`). Pure read-only — calls `git log` via
  `subprocess.run` (already imported), reads stage files, never writes.
  Function is bounded (10s subprocess timeout per stage file; ~5 stage
  files ≈ 50s worst case but in practice <2s). Will appear as
  Check #121 in the drift run output.
- No tests touched yet — the check is heuristic and advisory; behavior
  of existing 120 checks unchanged.

## Why this stage exists

User feedback 2026-04-28: "harden and future-proof as we go mate. i
don't like wasting tokens for nothing. ensure we are institutional
grade and ground."

Trigger: this session's `/next` invocation read 4 active stage files
under `docs/runtime/stages/`, but git log showed 3 of 4 had landed work
that was never closed (`8bef5eb1` PR #158, `a2b8a970` PR #152,
`68ee35f8`/`ecfeb33c`/`2a6a2293` for scratch-eod-mtm Stages 0–8). The
fourth (`pr48-mgc-shadow-only-overlay-contract.md`) was legitimately
operator-gated. Without an advisory check, every future session would
waste cycles re-discovering this state by hand.

## Acceptance

- `python pipeline/check_drift.py` exits 0 (the new check is advisory).
- The advisory line for `pr48-mgc-shadow-only-overlay-contract.md`
  does NOT fire (operator-gated body text exempts it).
- The advisory line for any stage file matching the pattern
  (>7d old + ≥3 slug-mentioning commits) DOES fire — manually
  verifiable by setting up a fixture or grep'ing real run output.
- Pre-commit gauntlet PASS (8/8).

## Out of scope

- Hard fail on staleness — explicitly NOT chosen. Heuristics produce
  false positives; operators must decide whether a stale-looking stage
  is actually stale.
- Cross-worktree stage scanning — explicitly NOT chosen. Each worktree
  has its own copy; the mode-rule already enforces global discipline.
- Auto-deletion of detected-stale stage files — explicitly NOT chosen.
  Closure must be human-approved (e.g. PR #154's pattern).

## Verification

Manual verification on the live repo state:
- Pre-edit: 1 active stage file remains (`pr48-mgc-shadow-only-overlay-contract.md`)
- Post-edit: drift check passes; advisory line for pr48 NOT emitted
  (operator-gated exemption); no other stage files match the pattern
  yet, so output silent — which is the correct steady state.
