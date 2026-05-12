# Handover — feat/proposer-autonomy

**Branch:** `feat/proposer-autonomy` (pushed to origin)
**Worktree:** `C:/Users/joshd/canompx3/.worktrees/proposer-autonomy`
**Commit:** `641bc54c`
**PR URL:** https://github.com/jbot-bit/canompx3/pull/new/feat/proposer-autonomy
**Date:** 2026-05-13

---

## What this branch does

Stops the LLM hypothesis proposer from drafting pre-regs that fail at the runner. Yesterday (2026-05-12) all 3 LLM-drafted candidates (`ATR_P30_O15`, `ORB_VOL_16K_O15`, `ATR_VEL_GE105`) were REJECTED on Criterion 8 (OOS/IS < 0.40) — wasted prereg-loop runs because the proposer didn't pre-screen.

This branch adds 4 pre-LLM screens that catch all 3 yesterday-rejections **before any LLM call**, in ~2-3 seconds total.

---

## Files changed (10)

| Path | Status | Purpose |
|---|---|---|
| `scripts/research/lhp/adjacency.py` | M | +`screen_candidate_mode_a` — delegates to `_evaluate_criterion_8_oos`. |
| `scripts/research/lhp/graveyard.py` | NEW | NO-GO / KILL / PARK detection. Primary: canonical `_search_research_catalog`. Fallback: file-grep. |
| `scripts/research/lhp/neighbor_scan.py` | NEW | Threshold (±20%) + aperture + (opt) session sibling Mode A + graveyard. Reports `family_health`. |
| `scripts/research/lhp/literature_index.py` | M | +`verify_citation_content` (token-overlap), +`find_citation_entries`. |
| `scripts/research/lhp/static_checks.py` | M | +5 new gates: scratch_policy, oos_power_floor, sensitivity_test, prior_art, citation_content. |
| `scripts/research/llm_hypothesis_proposer.py` | M | +`--candidate-strategy-id`, `--require-screen-pass`, `--auto-run`. Screen results injected into LLM context; prior_art injected into draft yaml. |
| `docs/prompts/hypothesis-proposer-system.md` | M | +Sections 8–11: Mode A/B trap, NO-GO awareness, new schema fields, refusal triggers. |
| `tests/fixtures/lhp/good_yaml_{1,2}.yaml` | M | Updated to satisfy new schema gates. |
| `docs/runtime/stages/proposer-autonomy.md` | NEW | Stage file. |

**Pipeline / trading_app touched: zero.** Read-only delegation only.

---

## How to use the new pipeline

### Pre-screen a candidate before drafting a pre-reg

```bash
python scripts/research/llm_hypothesis_proposer.py \
  --slug my-candidate \
  --candidate-strategy-id MNQ_<SESSION>_E2_RR1.0_CB1_<FILTER>_O15 \
  --dry-run --fixture tests/fixtures/lhp/good_yaml_1.yaml
```

- Exit 0 = clean draft written.
- Exit 1 = `REFUSED: candidate fails Mode A pre-screen` OR `graveyard_blocks_candidate`.
- Exit 5 = strategy_id not in `validated_setups` or `experimental_strategies`.

### One-shot draft + run

```bash
python scripts/research/llm_hypothesis_proposer.py \
  --slug my-candidate \
  --candidate-strategy-id <id> \
  --auto-run
```

`--auto-run` promotes the .draft.yaml → .yaml and invokes `scripts/infra/prereg-loop.sh --execute`. Requires `--candidate-strategy-id` (refuses without one).

### Override the screen (NOT recommended)

```bash
... --no-require-screen-pass
```

Surfaces the screen results in the LLM context but doesn't block. Use only when explicitly re-litigating a NO-GO with reopen_criteria cited.

---

## Verification

- `python -m pytest tests/test_llm_hypothesis_proposer.py -q` → 40/40 pass.
- `ruff check scripts/research/lhp/ scripts/research/llm_hypothesis_proposer.py` → clean.
- Regression: all 3 of yesterday's `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O15`, `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_O15`, `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105` → exit 1 at pre-screen, zero LLM calls.

---

## Known limitations / future-proof items

1. **Literature quote check is lexical only.** Catches fabrications and topic-mismatches; can't catch same-vocabulary category errors (e.g. Carver-sizing cited for entry-filter — words overlap). Future fix: require `passage_quote` field in pre-reg, verify exact match in literature file.
2. **Exit code 1 collision.** Pre-screen refusal and `LLMRefusalToGround` both return 1. Future split: 6 = candidate failed screen, 1 = LLM refused to ground.
3. **`_inject_prior_art` heuristic.** Only checks first 50 lines for an existing `prior_art:` block. Future fix: parse + merge instead of heuristic.
4. **`--auto-run` is blunt.** Promotes and invokes runner sequentially with no rollback. If runner fails, draft is already promoted. Future fix: 2-phase commit with rollback on runner failure.
5. **Session-axis neighbour scan is opt-in.** Costs ~12x more SQL — left off by default. Operator can pass `include_session=True` if needed.

---

## Pre-existing issue surfaced (NOT in scope of this branch)

**Drift check #64 fails on `origin/main`:**
> MGC: 1 trading day(s) with != 3 rows in daily_features

This is a pre-existing MGC `daily_features` row gap, present before this branch was cut. `git diff origin/main -- pipeline/` returns empty from `feat/proposer-autonomy`. Pre-commit was bypassed with `--no-verify` for this commit only, with user authorization, since the drift is provably not introduced here.

**Follow-up needed:** investigate why one MGC trading_day has ≠ 3 aperture rows in `daily_features`. Likely partial backfill or a skipped aperture. Quick first step:

```sql
SELECT trading_day, COUNT(*) AS rows
FROM daily_features
WHERE symbol = 'MGC'
GROUP BY trading_day
HAVING COUNT(*) != 3
ORDER BY trading_day DESC;
```

---

## Decision points / what to consider for the PR review

- **Schema gate strictness.** `scratch_policy`, `oos_power_floor`, `sensitivity_test`, `prior_art` are all FATAL fails. This will break any existing draft pre-regs in `docs/audit/hypotheses/drafts/`. Acceptable trade-off: drafts not yet locked are by definition mutable; locked yamls won't be re-validated until they're amended.
- **`--require-screen-pass` defaults True.** This is a fail-closed default. Override path documented.
- **`--no-verify` on commit.** Justified by pre-existing drift, but a future PR cleaning up MGC `daily_features` would let drift go green on the next branch.

---

## Open PR

Draft the PR via:
```
https://github.com/jbot-bit/canompx3/pull/new/feat/proposer-autonomy
```

Title suggestion: `feat(lhp): autonomous self-detecting hypothesis proposer`

PR description can lift directly from the commit message + this handover.
