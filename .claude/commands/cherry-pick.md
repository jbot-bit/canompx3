---
description: Rank fast-lane PROMOTE survivors and optionally generate a heavyweight Chordia draft for the top candidate.
---

Run the cherry-pick research loop:

1. Score every QUEUED entry in `docs/runtime/promote_queue.yaml` via `scripts/research/cherry_pick_ranker.py`.
2. Print the top-N ranking to stdout (default N=10).
3. If `--write-draft` is passed, also generate a heavyweight Chordia prereg DRAFT under `docs/audit/hypotheses/drafts/` for the top-1 candidate via `scripts/research/fast_lane_to_heavyweight_bridge.py`.
4. If `--ground` is passed (after `--write-draft`), attach a literature-grounded `theory_citation` to the bridge draft via `scripts/research/cherry_pick_grounder.py`. On PASS writes `<slug>.grounded.yaml` (theory_grant=true, t≥3.00 hurdle). On REFUSED writes `<slug>.grounded.rejected.txt` and the operator either authors the missing lit extract or accepts the no-theory strict 3.79 hurdle.

The ranker is read-only by default. The bridge always writes to the quarantine `drafts/` directory — never to active `hypotheses/`. The grounder writes alongside the bridge draft, never mutates the original. The operator manually moves drafts to active after literature/power/era review.

This command does NOT mutate `chordia_audit_log.yaml`, `validated_setups`, `lane_allocation.json`, or any file under `trading_app/live/`. It does not invoke the heavyweight Chordia runner — operator does that explicitly after reviewing the draft.

Arguments: $ARGUMENTS

Steps:

1. Run `python scripts/research/cherry_pick_ranker.py --write` to score and write `docs/runtime/cherry_pick_ranking_<date>.csv`.
2. Read the CSV and identify the top-1 candidate by rank.
3. If $ARGUMENTS contains `--write-draft`, run `python scripts/research/fast_lane_to_heavyweight_bridge.py <top_candidate_result_md>` to emit the heavyweight draft under `docs/audit/hypotheses/drafts/`.
4. If $ARGUMENTS contains `--ground`, run `python scripts/research/cherry_pick_grounder.py <draft_path>` to attach a literature-grounded `theory_citation`. On PASS the grounder writes `<slug>.grounded.yaml` (theory_grant=true earns Criterion 4 t≥3.00 hurdle); on REFUSED writes `<slug>.grounded.rejected.txt`.
5. Print the operator next-step checklist (citation review if grounded / literature decision if not, OOS power tier, era stability, K-budget re-verify, move from drafts/ to hypotheses/).
6. Remind the operator to append a row to `docs/runtime/cherry_pick_journal.md` including the new `grounded_verdict` column (GROUNDED | NO_LOCAL_LIT | LLM_REFUSED | CONTENT_MISMATCH | INVALID_OUTPUT | —).

Always pass `repo_root="C:/Users/joshd/canompx3"` when invoking from a worktree per `feedback_crg_worktree_repo_root_resolution.md`.
