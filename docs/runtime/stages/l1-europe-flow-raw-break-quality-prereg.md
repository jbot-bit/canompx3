---
slug: l1-europe-flow-raw-break-quality-prereg
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 1
of: 1
created: 2026-04-21
updated: 2026-04-21
task: Prereg (docs-only) for L1 EUROPE_FLOW pre-break context overlay — action queue item #2. Corrects backtesting-methodology § RULE 6.1 E2 look-ahead claim first, then writes the Stage-1 binary prereg limited to pre-ORB-end features only.
---

# Stage: L1 EUROPE_FLOW pre-break context prereg

## Question

Action queue item #2 (`docs/plans/2026-04-21-post-stale-lock-action-queue.md`):

> L1 `EUROPE_FLOW` raw break-quality overlay — status PREREG NEXT.
> Why: ATR-normalized replacement failed (verified 2026-04-21,
> `docs/audit/results/2026-04-21-l1-europe-flow-filter-diagnostic.md`), but raw
> break-quality / pre-break context was explicitly not killed.
> Exact action: prereg a small K family using raw break geometry only;
> admissible features limited to break-bar / pre-break context; no
> ATR-normalized ratio variants.

## Real-data finding that scoped this stage

Initial plan was to admit `break_delay_min` / `break_bar_continues` / `break_bar_volume`
per `.claude/rules/backtesting-methodology.md` § RULE 6.1's claim these are
"known at break-bar close, before E2 CB1 entry". Verified against real
`gold.db` on 2026-04-21:

| Year | N | entry_ts == break_ts | entry_ts < break_ts (LEAK) | entry_ts > break_ts |
|------|---|----------------------|-----------------------------|----------------------|
| 2019 | 171 | 107 | 64 (37.4%) | 0 |
| 2020 | 256 | 143 | 113 (44.1%) | 0 |
| 2021 | 259 | 158 | 101 (39.0%) | 0 |
| 2022 | 258 | 162 | 96 (37.2%) | 0 |
| 2023 | 258 | 153 | 105 (40.7%) | 0 |
| 2024 | 259 | 154 | 105 (40.5%) | 0 |
| 2025 | 257 | 132 | 125 (48.6%) | 0 |
| TOTAL | 1718 | 1009 | 709 (41.3%) | 0 |

**Root cause:** E2 fires on RANGE-cross (bar `high > orb_high` / `low < orb_low`,
including wick-only fakeouts; see `trading_app/entry_rules.py:157-216 detect_break_touch()`).
`break_ts` in `daily_features` is defined on CLOSE-cross (bar closes outside ORB;
see `pipeline/build_daily_features.py:285-340 detect_break()`). For wick-touch
bars that close back inside, E2 fires then — but the close-break bar arrives
later, carrying `break_delay_min`, `break_bar_continues`, `break_bar_volume`
that post-date entry. Using them as predictors = look-ahead.

**mechanism_priors.md § 4** bans these columns — correct.
**backtesting-methodology.md § RULE 6.1** lists them as safe-before-E2-CB1 — WRONG. Fix.

## Scope Lock

- `.claude/rules/backtesting-methodology.md` — correct § RULE 6.1: remove `orb_{s}_break_delay_min`, `orb_{s}_break_bar_continues`, `orb_{s}_break_bar_volume` from the E2 CB1 safe list; add explicit E2 caveat; point to `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md` for the real-data reasoning.
- `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md` — new postmortem capturing the real-data finding (per-year breakdown, root-cause code citations, affected features, scope of false-survivor risk in existing scans).
- `docs/audit/hypotheses/2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml` — new Stage-1 binary prereg, admissible feature set = pre-ORB-end only (`orb_EUROPE_FLOW_pre_velocity`, `orb_EUROPE_FLOW_vwap`, `orb_EUROPE_FLOW_size`, `orb_EUROPE_FLOW_high`, `orb_EUROPE_FLOW_low`); banned set = break_* columns + `compression_z` (0% populated for EUROPE_FLOW) + ATR-normalized ratio variants (action-queue constraint).
- `docs/runtime/stages/l1-europe-flow-raw-break-quality-prereg.md` — this stage file, closed at end.

Zero production-code edits. Zero data-build edits. Zero new filter registrations.

## Blast Radius

- **Rule doc correction** (`backtesting-methodology.md`): high-importance doctrine. Readers: research/discovery agents, prereg authors, hook injection. No code currently consumes `break_*` as a predictor in production (verified by repo-grep). Existing research artifacts that used these columns under the old "safe" reading must be re-audited — surfaced as follow-up items in the postmortem, not fixed in this PR.
- **Postmortem**: append-only; no risk.
- **Prereg YAML**: additive; no downstream reader until a scan script is written in the next stage.
- **Drift guard**: `python pipeline/check_drift.py` must still pass post-edit.

## Approach

1. Repo-grep: confirm no production code currently reads `break_delay_min` / `break_bar_continues` / `break_bar_volume` / `break_dir` / `break_ts` as predictors under the old "safe" reading. Research and one-shot audit scripts may — surface for follow-up.
2. Edit `.claude/rules/backtesting-methodology.md` § RULE 6.1:
   - remove `orb_{s}_break_delay_min`, `orb_{s}_break_bar_continues`, `orb_{s}_break_bar_volume` from § 6.1 safe list
   - add an explicit E2 caveat explaining the range-cross vs close-cross distinction
   - add cross-reference to the new postmortem
3. Write `docs/postmortems/2026-04-21-e2-break-bar-lookahead.md` containing: the per-year leak table, root-cause code citations, affected features, the § RULE 6.1 correction, and follow-up action items.
4. Write the Stage-1 binary prereg YAML conforming to `docs/institutional/pre_registered_criteria.md`:
   - 2-3 numbered hypotheses each with theory citation (Fitschen Ch 3 for trend-continuation; order-flow grounding via canonical extracts)
   - admissible features enumerated with feature-validity-gate evidence (100% populated, pre-ORB-end)
   - K_budget ≤ 50 cells, pre-committed
   - kill criteria and expected fire rate (RULE 8.1 compliant: 5-95%)
   - OOS / dir_match discipline (RULE 3)
   - explicitly NO ATR-normalized ratio variants (action queue constraint)
   - explicitly NO break_* / mae_r / mfe_r / outcome / double_break columns (E2 look-ahead)
   - explicitly NO pooled ML reframing (action queue constraint)
5. Commit in two logical commits on this branch:
   - commit A: § RULE 6.1 correction + new postmortem
   - commit B: new L1 prereg YAML
6. Run `python pipeline/check_drift.py` after each commit to confirm no regression.
7. Push and open PR against `main`. PR body cites this stage file, the action queue doc, and the real-data postmortem.

## Acceptance criteria

1. Real-data leak evidence printed to chat ✅ (done: 709/1718 = 41.3%, per-year 37-49%).
2. Repo-grep: zero production-code reads of `break_delay_min` / `break_bar_continues` / `break_bar_volume` as predictors; any research/audit hits surfaced in the postmortem as follow-up.
3. § RULE 6.1 correction is minimal and includes E2 caveat + postmortem cross-reference.
4. Postmortem contains per-year table, root-cause code citations, affected-features list, follow-up items.
5. Prereg YAML exists at `docs/audit/hypotheses/2026-04-21-l1-europe-flow-pre-break-context-prereg.yaml` and contains: numbered hypotheses, theory citations, K_budget, kill criteria, expected fire rates, OOS discipline, explicit feature-validity gate, explicit ban list.
6. `python pipeline/check_drift.py` passes (modulo pre-existing `anthropic` import gap).
7. PR opened with scope diff = 4 doc files (+ this stage file). `git log --oneline origin/main..HEAD` shows exactly the intended commits. `git diff --stat origin/main HEAD` limited to `.claude/rules/` + `docs/**`.
8. No production code touched.

## Non-goals

- Not running any scan (next stage after prereg freeze + PR merge).
- Not re-auditing prior research artifacts that may have used the banned break_* features under the old "safe" reading — surfaced as follow-up in postmortem, not fixed here.
- Not touching `trading_app/`, `pipeline/`, `scripts/`, or any production code.
- Not interfering with the parallel Codex terminal's WIP on `project_pulse.py` / `trading_app/ai/*` (stashed at `stash@{0}`; user directive).
- Not committing the GARCH R3 work still stashed (separate PR — PR 3 in the cleanup roadmap).
- Not touching the 2026-01-01 sacred holdout.
