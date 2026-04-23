# PR48 conditional-edge recovery audit

**Date:** 2026-04-23
**Scope:** recover the unfinished `PR48` conditional-edge / multi-confluence line, verify whether it still reproduces on current published `main`, and determine whether the current system can carry it through validation/allocation without redundant rediscovery.

## Grounding

- Canonical data layers used for reruns: `daily_features`, `orb_outcomes`
- Published repo base used for recovery: `origin/main` at `2c49fc13`
- Recovered branch lineage: `origin/wt-codex-conditional-edge-framework`
- Replayed stack in clean worktree: `3b7a20fd` .. `85ebbe78`

## What was recovered

The unfinished path is not the `.4R` MES rel-vol family. That side path remains separately documented and currently closes `KILL`.

The real unfinished line is the conditional-role framework and its bounded PR48 sequence:

- `docs/institutional/conditional-edge-framework.md`
- `research/pr48_conditional_role_implementation_v1.py`
- `research/pr48_role_followthrough_v1.py`
- `research/pr48_promotion_shortlist_v1.py`
- the three paired hypothesis/result docs from `2026-04-22`

This is the correct continuation of the repo's open `MES/MGC -> MNQ` conditioner / allocator / confluence question. It is not a restart of the older broad confluence program.

## What was re-verified

The recovered commit stack replayed cleanly onto current published `origin/main` in an isolated worktree.

The three PR48 scripts were re-run against the current canonical DB path and reproduced the saved result docs with no structural drift:

- `research/pr48_conditional_role_implementation_v1.py`
- `research/pr48_role_followthrough_v1.py`
- `research/pr48_promotion_shortlist_v1.py`

Current reproduced shortlist truth:

| candidate | IS mean_daily_delta_$ | BH survives | OOS sign |
|---|---:|:---:|:---:|
| `MES:q45_exec` | `+17.98` | `Y` | `+` |
| `MGC:cont_exec` | `+15.29` | `Y` | `+` |
| `DUO:mes_q45_plus_mgc_cont_exec` | `+25.92` | `Y` | `+` |
| `MNQ:shadow_addon` | `+44.71` | `Y` | `+` |

Latest canonical trading day visible to this replay remains `2026-04-16`.

## Current system fit

The current repo can validate and allocate **validated standalone lanes**. It does **not** yet have a native promotion path for PR48-style conditional-role outputs.

Measured reasons:

1. `trading_app/strategy_validator.py` promotes validated strategies into `validated_setups`, which is a standalone-lane shelf.
2. `trading_app/lane_allocator.py` ranks and selects from `deployable_validated_relation(...)`, not from conditional-role result docs or shadow candidate maps.
3. `docs/runtime/lane_allocation.json` currently routes six `MNQ` lanes only. None of the recovered PR48 shortlist arms are present there.
4. The current `validated_setups` schema includes modern provenance fields such as `validation_pathway`, `slippage_validation_status`, `validation_run_id`, and `promotion_git_sha`, but there is no native role field for `filter` / `allocator` / `confluence` outputs from this branch.

## What this means

- The PR48 branch is **not lost**.
- The PR48 branch is **still alive** as a conditional-role / allocator shortlist.
- The PR48 branch is **not already integrated** into the current validation/allocation pipeline.
- Starting a fresh generic multi-confluence discovery pass would be redundant and lower-signal than resuming this recovered line.

## Verdict

`CONTINUE`

Continue as a **bounded validation/allocation translation problem**, not as a new confluence discovery program.

## Biggest issue

The open gap is not discovery. The open gap is **representation and translation**:

- how a recovered conditional-role result should be carried forward honestly
- how to compare it to current live routing without pretending it is a standard promoted shelf row

## Next best step

Use `docs/runtime/stages/pr48-conditional-role-validation-translation.md`.

That stage is the exact bridge from recovered PR48 research to current repo validation/allocation surfaces. It keeps the branch honest:

- no broad confluence rediscovery
- no direct auto-promotion from research docs
- no claim that conditional-role outputs are already live-ready

## Reproduction

- `python3 -m py_compile research/pr48_conditional_role_implementation_v1.py research/pr48_role_followthrough_v1.py research/pr48_promotion_shortlist_v1.py`
- `./.venv-wsl/bin/python research/pr48_conditional_role_implementation_v1.py`
- `./.venv-wsl/bin/python research/pr48_role_followthrough_v1.py`
- `./.venv-wsl/bin/python research/pr48_promotion_shortlist_v1.py`
- `./.venv-wsl/bin/python scripts/tools/check_claim_hygiene.py docs/audit/results/2026-04-22-pr48-promotion-shortlist-v1.md`

## Caveats

- The reproduced docs still reflect canonical data through `2026-04-16`; this is not a fresh post-`2026-04-16` evaluation.
- OOS remains thin and should still be treated as monitoring, not re-specification input.
- This audit does not authorize live routing or validated-shelf promotion by itself.
