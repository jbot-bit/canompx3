## Scope

Stage hygiene cleanup + new advisory drift check. **No production trading logic, holdout policy, or canonical-truth tables touched.**

## Commits

- `55c458d7` — close 2 stage files for already-merged PRs (#158 NQ-mini Stage 1, #152 registry hygiene)
- `2a474ea7` — Check 121 `check_stage_file_landed_drift` (advisory, never blocks); surfaces stages where `updated:` >7d AND ≥3 commits reference the slug
- `828ba9b4` — close stage-staleness-drift-check (its own 1-stage work landed)

## Files changed

- `pipeline/check_drift.py` — additive (+103 lines, 1 new advisory check)
- `docs/runtime/stages/nq-mini-execution-stage1-account-profile.md` — deleted (PR #158 landed)
- `docs/runtime/stages/recover-registry-hygiene.md` — deleted (PR #152 landed)
- `HANDOFF.md` — auto-update stub (post-commit hook)

## Evidence

```
$ python pipeline/check_drift.py | tail -3
NO DRIFT DETECTED: 114 checks passed [OK], 0 skipped (DB unavailable), 7 advisory

$ python -m pytest tests/test_pipeline/ -q -k "check_drift or drift" | tail -2
===================== 265 passed, 970 deselected in 4.51s =====================

Pre-commit gauntlet: 8/8 PASS on every commit.
```

Direct invocation of `check_stage_file_landed_drift()` in steady state returns empty (no false positives). Operator-gated exemption verified: `pr48-mgc-shadow-only-overlay-contract.md` body's "operator observation" text correctly excludes it.

## Disconfirming Checks

- The new advisory check is OPT-IN visibility only — never blocks pre-commit or CI.
- 7-day age threshold + 3-commit slug-match threshold deliberately conservative to avoid false positives during normal staged work.
- Operator-gated stages (e.g., `pr48-mgc-shadow-only-overlay-contract`) explicitly skipped via body-text scan.
- Closed stage files leave nothing: stage tracking is filesystem-only, no orphan references.

## Grounding

- Pattern matches PR #154 ("chore(stages): close landed stages from PRs #147 and #153")
- User feedback driving the drift check: "harden and future-proof as we go — don't waste tokens for nothing"
- `.claude/rules/stage-gate-protocol.md` § Stale detection (>4h ageing rule extended via this advisory check to commit-vs-claim drift)

## Not done by this PR

- Does NOT close `pr48-mgc-shadow-only-overlay-contract.md` (operator-gated; awaiting MGC shadow observation)
- Does NOT close `scratch-eod-mtm-canonical-fix.md` (lives on `research/mnq-unfiltered-high-rr-family`; will be closed by that branch's PR)
- Does NOT enable auto-merge — merge after CI green via `gh pr merge --merge --delete-branch`
