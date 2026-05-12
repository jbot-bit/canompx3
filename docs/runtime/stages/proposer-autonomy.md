---
task: Autonomous self-detecting LLM hypothesis proposer
mode: IMPLEMENTATION
stage: 1/2
scope_lock:
  - scripts/research/llm_hypothesis_proposer.py
  - scripts/research/lhp/adjacency.py
  - scripts/research/lhp/literature_index.py
  - scripts/research/lhp/static_checks.py
  - scripts/research/lhp/graveyard.py
  - docs/prompts/hypothesis-proposer-system.md
  - tests/test_research/lhp/test_graveyard.py
  - tests/test_research/lhp/test_mode_a_screen.py
  - tests/test_research/lhp/test_static_checks.py
---

## Task

Upgrade the existing LLM hypothesis proposer to be autonomous and self-detecting. Currently the proposer happily drafts pre-regs against Mode B `validated_setups.oos_exp_r` baselines that crash at the runner under Mode A strict recomputation (3/3 LLM-drafted pre-regs rejected on 2026-05-12).

## Blast Radius

- `scripts/research/llm_hypothesis_proposer.py` — adds 4 pre-LLM screening stages before `propose_with_mock_support`. Existing CLI flags unchanged; new `--auto-run` flag optional.
- `scripts/research/lhp/adjacency.py` — adds `screen_candidate_mode_a()` delegating to `trading_app.strategy_validator._evaluate_criterion_8_oos`. Read-only against gold.db.
- `scripts/research/lhp/literature_index.py` — extends with `verify_citation_content(slug, terms)` for quote-presence (not just filename existence).
- `scripts/research/lhp/static_checks.py` — tightens `run_all` to require scratch_policy / OOS power-floor / sensitivity_test / prior_art.
- `scripts/research/lhp/graveyard.py` — NEW. Greps `docs/audit/results/` + `memory/nogo_*.md` + `memory/feedback_*.md` for filter-family/session matches.
- `docs/prompts/hypothesis-proposer-system.md` — adds Mode A vs Mode B warning, NO-GO doctrine, schema requirements.
- Reads: gold.db (read-only via canonical validator function), filesystem (audit results, memory files, literature extracts). Writes: `.draft.yaml` / `.rejected.txt` under `docs/audit/hypotheses/drafts/` only — NO gold.db writes, NO production code mutation.
- Production code unchanged. `pipeline/` and `trading_app/` consumed via existing imports only. No schema, no canonical-source modification, no `validated_setups` write.
- Tests: 3 new test modules under `tests/test_research/lhp/`. Existing proposer integration test (`tests/test_research/test_llm_hypothesis_proposer.py` if present) updated for new screen stages.

## Acceptance Criteria

1. Re-run proposer against yesterday's 3 rejected candidates (`MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ATR_P30_O15`, `MNQ_CME_PRECLOSE_E2_RR1.0_CB1_ORB_VOL_16K_O15`, `MNQ_TOKYO_OPEN_E2_RR1.0_CB1_ATR_VEL_GE105`) — all 3 must be dropped at stage 1 (Mode A screen) with logged Mode A OOS/IS ratios < 0.40 (or n_oos < 30 for the TOKYO_OPEN case).
2. Graveyard check fires on the L2 ATR_P50 NO-GO when proposer sees an ATR_P30 candidate at CME_PRECLOSE — auto-rejects unless reopen criteria cited.
3. Static checks reject any draft missing scratch_policy / OOS power-floor / sensitivity_test / prior_art.
4. Literature quote check fails when economic_basis cites Carver for entry filtering (sizing-only source) — proposer must regenerate or fall back to Harris 2002 microstructure.
5. Existing tests stay green. New tests cover the 4 added functions.
6. `python pipeline/check_drift.py` passes.
