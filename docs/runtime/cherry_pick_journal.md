# Cherry-Pick Research Loop — Journal

Append-only research-process log. One row per cherry-pick iteration: the
ranker scores fast-lane PROMOTE survivors, the bridge generates a heavyweight
Chordia draft, the operator decides whether to advance, and the heavyweight
verdict (if a replay is run) is recorded with a one-sentence lesson.

This file is **research process documentation, not doctrine**. Binding
governance entries live in `docs/runtime/decision-ledger.md`.

## Schema

| Column | Meaning |
|---|---|
| iter | Sequential iteration number, ascending |
| date | ISO date (Brisbane) of the cherry-pick decision |
| strategy_id | Exact strategy_id from validated_setups / promote_queue |
| rank_score | Total score from `scripts/research/cherry_pick_ranker.py` |
| pooled_t | Pooled IS t-stat from the fast-lane source MD |
| oos_power_tier | CAN_REFUTE / DIRECTIONAL_ONLY / STATISTICALLY_USELESS / NA_NO_OOS / NA_N_BELOW_FLOOR |
| bridge_draft_path | Path under `docs/audit/hypotheses/drafts/` if bridge ran, else `—` |
| grounded_verdict | GROUNDED / NO_LOCAL_LIT / LLM_REFUSED / CONTENT_MISMATCH / INVALID_OUTPUT / — (— when grounder not invoked) |
| heavyweight_verdict | PASS_CHORDIA / FAIL_STRICT / PARK / DEFERRED_NOT_RUN / — |
| lesson | One sentence: what fast-lane signature predicted this heavyweight outcome |

## Authority

- Plan: `C:/Users/joshd/.claude/plans/or-linknin-them-togehr-delegated-gizmo.md`
- Ranker: `scripts/research/cherry_pick_ranker.py` (Stage A, commit `81da1099`)
- Bridge: `scripts/research/fast_lane_to_heavyweight_bridge.py` (Stage B, commit `b3bb9bdf`)
- Grounder: `scripts/research/cherry_pick_grounder.py` (Stage C grounding extension, 2026-05-19)
- Doctrine: `.claude/rules/backtesting-methodology.md` § RULE 3.3 (OOS power floor)
- Field-presence trap defense: `memory/feedback_chordia_theory_citation_field_presence_trap.md`

## Entries

| iter | date | strategy_id | rank_score | pooled_t | oos_power_tier | bridge_draft_path | grounded_verdict | heavyweight_verdict | lesson |
|-----:|------|-------------|-----------:|---------:|----------------|-------------------|------------------|---------------------|--------|
| 1 | 2026-05-19 | MNQ_US_DATA_1000_E1_RR1.0_CB2_PD_CLEAR_LONG_O30 | 0.250 | 3.064 | NA_N_BELOW_FLOOR | docs/audit/hypotheses/drafts/2026-05-19-mnq-us-data-1000-e1-rr1-0-cb2-pd-clear-long-o30-chordia-heavyweight-v1.draft.yaml | — | DEFERRED_NOT_RUN | OOS underpowered (N=14, IS+0.17 / OOS-0.02 sign-flip, dir_match=N); deflation_headroom=0 since pooled_t=3.06 < strict 3.79; draft authored for record but NOT promoted to active hypotheses/ — loop correctly identifies "not ready yet". Wait for OOS N to accrue past 30 with same-sign ExpR before re-cherry-picking. |
