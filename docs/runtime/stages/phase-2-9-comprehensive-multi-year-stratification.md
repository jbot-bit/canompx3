---
slug: phase-2-9-comprehensive-multi-year-stratification
mode: IMPLEMENTATION
task: Phase 2.9 — comprehensive 7-year × 38-lane stratification with per-cell subset-t + multi-framing BH + session×year heat map. Answers the questions Phase 2.8 v1 didn't (session-directional 2024 asymmetry, per-cell significance, symmetric DRAG/BOOST).
updated: 2026-04-19
origin: docs/audit/results/2026-04-19-phase-2-8-reframe-addendum.md § 7 honest next test
---

## Purpose

Replace the Phase 2.8 v1 bare-threshold classification with an institutional-grade multi-year stratification:
- 7 years (2019-2025, all within Mode A IS).
- 38 active `validated_setups` lanes (includes Phase 2.5 Tier-4 retirement candidates — they are already in the active set).
- Per-cell subset-t + BH FDR at multiple K framings.
- Session × year heat map as first-class output.
- Symmetric DRAG / BOOST labels with bootstrap significance.
- Fragility check column for Phase 2.5 Tier-1 / Phase 2.7 GOLD candidates.

This does NOT retract Phase 2.8 v1 retirement verdicts (the 2 SGP PURE_DRAG lanes are independently Phase 2.4/2.7-confirmed). It replaces v1's REGIME_REFUTATION framing with a properly tested one.

## Scope lock

Files to CREATE (all research-only — no production code):
- `docs/audit/hypotheses/2026-04-19-phase-2-9-comprehensive-multi-year-stratification.yaml` (pre-reg, MUST land before script run)
- `research/phase_2_9_comprehensive_multi_year_stratification.py` (new script; does not modify Phase 2.8)
- `tests/test_research/test_phase_2_9_comprehensive_multi_year_stratification.py` (new tests)
- `docs/audit/results/2026-04-19-phase-2-9-comprehensive-multi-year.md` (result doc, AFTER run)
- `research/output/phase_2_9_*.csv` (multiple: main, heat_map, fragility)

Files READ-ONLY (canonical delegations):
- `research/mode_a_revalidation_active_setups.py` (compute_mode_a, load_active_setups, direction_from_execution_spec)
- `research/filter_utils.py` (filter_signal)
- `trading_app/holdout_policy.py` (HOLDOUT_SACRED_FROM)
- `pipeline/paths.py` (GOLD_DB_PATH)
- `pipeline/dst.py` (SESSION_CATALOG)

Not in scope (NEVER edit):
- Phase 2.8 existing script or result doc (v1 stands with its verdict narrowed via the addendum already committed).
- Phase 2.5 script, result doc, CSV.
- Any pipeline/, trading_app/, scripts/ production file.

## Blast radius

- Zero production blast. Pure research + docs.
- Canonical delegations import-only. No parallel encoding.
- No writes to `experimental_strategies` or `validated_setups`.

## Approach

1. Pre-reg YAML (Pathway A, testing_mode=family, MinBTL budget pre-committed).
2. Script writes 3 CSVs:
   - `phase_2_9_main.csv` — 266 lane×year cells with subset_t / p / year_expr / delta / pattern / BH survivor flags at K_global / K_session / K_year.
   - `phase_2_9_session_year_heat.csv` — session × year mean ExpR grid, weighted by n_on per lane.
   - `phase_2_9_gold_fragility.csv` — Phase 2.5 Tier-1 + Phase 2.7 GOLD candidates with ex-each-year subset_t for fragility disclosure.
3. Tests cover: expected row count (266 = 38 × 7), subset_t computation, BH function, symmetric pattern labeller (DRAG ↔ BOOST), session×year aggregation, canonical delegation equivalence.
4. Result doc includes: heat-map rendered as markdown table, BH survivor list at each K, fragility table, narrowed verdicts for Phase 2.5 Tier-1 / Phase 2.7 GOLD, next-step proposal.

## K budget + MinBTL

- K_global = 266 cells (38 lanes × 7 years).
- K_session (per session, across years) = 7 (variable per session; reported per session).
- K_year (per year, across lanes) = 38.
- MinBTL = 2·ln(266) / E[max_N]². Computed in script header before run; aborts if min N fails.

## Acceptance criteria

1. Pre-reg YAML committed BEFORE script runs (Phase 0 mandatory).
2. Script uses `compute_mode_a` and `filter_signal` canonical delegations — no inline filter logic.
3. CSV row counts match: main=266, heat_map=7 sessions × 7 years, fragility=9 Tier-1 + 5 GOLD=14 (or narrower if deduped).
4. All tests pass.
5. Drift check runs clean except the expected Check 37 (worktree has no local gold.db).
6. Result doc cites canonical extracts from `docs/institutional/literature/` for the statistical methods used (BH-FDR: Harvey-Liu; subset_t convention: Chordia 2018; MinBTL: Bailey 2013).
7. Final verdict explicitly states which alternative interpretations I1-I5 from the addendum are supported / refuted / undecided by the data.

## Done = all 4 required (per CLAUDE.md)

- Tests pass (show output)
- Dead code swept (`grep -r` in research/)
- `python pipeline/check_drift.py` passes (or only Check 37)
- Self-review with line-citation evidence

## Iteration checkpoints (user-facing)

1. After pre-reg YAML written — pause, show pre-reg to user for approval before running script.
2. After script + test files written, before script execution — show diff summary, run drift check, run tests.
3. After script run, before committing result doc — show BH survivor counts, heat-map shape, one-line verdict. Await confirmation.

No ad-hoc expansion of scope. Any discovery-finding that suggests another test = pre-registered separately as Phase 2.10.
