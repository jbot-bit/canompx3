# Stage 2A.3 — Fast-lane walk orchestrator (Stage 4 of connective-tissue plan; Stage 3 deferred)

task: Build scripts/tools/fast_lane_walk.py + scripts/infra/fast-lane-walk.sh that compose the existing fast-lane writers into a single read-only operator entry point. Emits an awareness Markdown report to stdout (counts per stage, top-3 stalled, ERROR roll-up, next-operator-action footer). Stage 3 (age-staleness view) deferred per CTX 74% + no documented consumer; orchestrator works against the Stage 2A.2 status roll-up alone.

mode: IMPLEMENTATION

scope_lock:
  - scripts/tools/fast_lane_walk.py
  - scripts/infra/fast-lane-walk.sh
  - tests/test_tools/test_fast_lane_walk.py

## Blast Radius

- scripts/tools/fast_lane_walk.py — new file, zero callers. Composes existing main(argv) entry points from fast_lane_promote_queue, cherry_pick_ranker, cherry_pick_journal_enricher, and fast_lane_status. Writes only the optional report markdown; never touches capital-class state.
- scripts/infra/fast-lane-walk.sh — new front-door wrapper. Same venv discovery pattern as scripts/infra/prereg-loop.sh.
- tests/test_tools/test_fast_lane_walk.py — new file. Smoke: end-to-end walk runs; report markdown is non-empty; idempotent across two runs; non-zero exit on injected ERROR.
- Reads: docs/runtime/promote_queue.yaml, cherry_pick_journal.yaml, cherry_pick_ranking_*.csv, fast_lane_status.yaml + every artifact those scripts read (all read-only via composed mains).
- Writes: only `docs/runtime/fast_lane_walk_<date>.md` (optional, --write-report) plus the upstream caches the composed scripts already write.
- Capital-class? No. Composed scripts already have their own capital-class boundaries enforced by their existing drift checks (Check #157, #160, #161, #168). Orchestrator inherits.

## Stage 3 deferral (decision rationale)

Stage 3 (`fast_lane_age_staleness.yaml`) is DEFERRED for these reasons:
1. **No documented consumer.** Stage 2A.2 roll-up already carries `age_days` per strategy_id. Per-stage transition ages (queued/ranked/bridged/grounded/enriched) have no operator workflow consuming them today.
2. **mtime brittleness.** Plan § 4.1 acknowledges mtime drift on fresh clone or cross-worktree edit. Git-log fallback is a Stage 3 concern that adds complexity for an unconsumed signal.
3. **n=1 meta-tooling doctrine.** `feedback_meta_tooling_n1_2026_05_01.md` — don't build infrastructure ahead of demonstrated need. Stage 3 can ship later if operator workflow demand surfaces.

Decision recorded; state-graph `fast_lane_age_staleness` node retains `proposed: true` flag.

## Acceptance criteria

1. `python scripts/tools/fast_lane_walk.py --dry-run` runs end-to-end against the live repo without exception; emits a Markdown report to stdout.
2. Report contains all five Connector 4 surfaces: counts-per-stage table, top-3 stalled by age_days, ERROR roll-up (zero entries currently), next-operator-action footer naming exactly one strategy_id + one stage, schema_version banner.
3. `pytest tests/test_tools/test_fast_lane_walk.py -v` — all tests PASS. Coverage: smoke run end-to-end, idempotent two-run report parity, non-zero exit on injected ERROR upstream, footer-emits-correct-priority (highest-rank QUEUED > oldest ungrounded draft > oldest verdict-pending journal entry).
4. `bash scripts/infra/fast-lane-walk.sh --dry-run` resolves venv and executes successfully (smoke only; not asserted by pytest — manual operator gate).
5. `python pipeline/check_drift.py` still 168 PASSED + 1 carry-over (no new drift check this stage; orchestrator inherits upstream checks).
6. Idempotency: two `--dry-run` invocations back-to-back produce identical Markdown report (modulo `generated_at` date stamp).
7. Non-zero exit semantics: returns 2 if any composed `main()` returns non-zero (matches `fast_lane_promote_queue.py` convention).
8. Stage 3 deferral recorded in stage file + decision-ledger entry (NOT in CLAUDE.md or HANDOFF — this is a within-plan deferral, not a permanent design change).
