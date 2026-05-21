---
task: Apply ruff auto-fixable lint errors (97 of 106) across pipeline/, trading_app/live/, scripts/, tests/ to unblock CI ruff lint gate that is blocking PR #307 and all subsequent PRs against main. Auto-fix only (no --unsafe-fixes); 9 non-auto-fixable lint findings (B007, B023, B904, E741, F841, SIM115, SIM118, SIM300×4, SIM401×2) deferred to a separate stage requiring per-site judgment.
mode: IMPLEMENTATION
updated: 2026-05-21
scope_lock:
  - pipeline/check_drift.py
  - scripts/research/cherry_pick_grounder.py
  - scripts/research/cherry_pick_journal_enricher.py
  - scripts/research/cherry_pick_ranker.py
  - scripts/research/fast_lane_graveyard_digest.py
  - scripts/research/fast_lane_promote_queue.py
  - scripts/research/fast_lane_to_heavyweight_bridge.py
  - scripts/research/ingest_idea.py
  - scripts/research/triage_validated_setups.py
  - scripts/tools/fast_lane_status.py
  - scripts/tools/fast_lane_walk.py
  - scripts/tools/go_portal.py
  - trading_app/live/bot_dashboard.py
  - trading_app/live/session_orchestrator.py
  - tests/**
acceptance:
  - `uv run ruff check pipeline/ trading_app/ scripts/ tests/` returns 0 errors OR only the 9 known non-auto-fixable findings (B007, B023, B904, E741, F841, SIM115, SIM118, SIM300, SIM401 sites)
  - `uv run ruff format --check pipeline/ trading_app/ scripts/ tests/` clean
  - python pipeline/check_drift.py passes (modulo pre-existing MGC carry-over from PR #307)
  - Targeted pytest on every file touched in trading_app/ passes
  - Diff shows only mechanical changes (import reorder, datetime.UTC alias, f-string fix, deprecated-import update, timeout-error alias)
  - PR opened against origin/main; #304 (stale 4-file ruff-recovery) closed in favor of this PR

## Blast Radius

- pipeline/check_drift.py — I001 + UP017 sites only. Read by every drift gate. Zero behavior change.
- trading_app/live/bot_dashboard.py — 1 I001 site (deeply-nested import block lines 2993-2994). Capital-class file; SSE dashboard. Zero runtime behavior change (Python imports resolve identically regardless of order).
- trading_app/live/session_orchestrator.py — NOT auto-fixed here (SIM401 sites at 1507-1508 require human review; defer to follow-up stage).
- scripts/research/* — backtest + research scripts. Touched only for I001 / UP017 / UP035 / UP041 / F541. No production-path code.
- scripts/tools/* — operator tooling (fast_lane_status, fast_lane_walk, go_portal). I001 + UP017.
- tests/** — test files. I001 + UP017 + UP035 + UP041 + F541. Pre-test mechanical fixes; no test-assertion change.

Reads: zero. Writes: zero (no data/file mutation). Schema changes: zero. Entry models: untouched. Pure code-style.

## Order of operations

1. Run `uv run ruff check pipeline/ trading_app/ scripts/ tests/ --fix --no-unsafe-fixes` — auto-fix the 97.
2. Confirm 9 remaining errors are the documented non-auto-fixable set (B007, B023, B904, E741, F841, SIM115, SIM118, SIM300×4, SIM401×2).
3. Run `uv run ruff format pipeline/ trading_app/ scripts/ tests/` to clean up any line-length fallout.
4. Run targeted pytest on every modified file under trading_app/ (capital-class verification).
5. Run `python pipeline/check_drift.py` — confirm only pre-existing MGC carry-over violations.
6. Commit single mechanical change with full provenance.
7. Push branch; open PR against origin/main; reference this stage file in PR body.
8. Note in PR body: PR #304 should be closed in favor of this one.

## Hard Constraints

- NO --unsafe-fixes flag (would introduce semantic-divergent edits).
- NO manual code changes outside ruff output.
- NO new tests (mechanical fix; existing tests are the gate).
- NO touching the 9 non-auto-fixable sites (deferred stage).
- Capital-class file edits limited to auto-fix output verbatim — no opportunistic refactor.

## Companion

This stage is downstream of `2026-05-21-canonical-blocks-out-of-stage-files.md` (closed at commit 849b40f0). It unblocks PR #307 CI; it does not change any acceptance criterion of that PR.
