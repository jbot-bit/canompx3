---
slug: scratch-eod-mtm-canonical-fix
classification: IMPLEMENTATION
mode: IMPLEMENTATION
stage: 2
of: 8
created: 2026-04-27
updated: 2026-04-27
task: Canonical scratch-EOD-MTM fix for outcome_builder.py + downstream re-verification + MFE-distribution endogenous-RR research. 8-stage plan. Stage 0-1 complete (literature, failure log, CORRECTION result). Currently Stage 2 — drift-check guard.
---

# Multi-stage canonical scratch-handling fix

## Plan summary

Class bug discovered 2026-04-27: `trading_app/outcome_builder.py:586-594, :612-616, :357-381` produces `outcome="scratch"` rows with NULL `pnl_r`. Every downstream consumer using `WHERE pnl_r IS NOT NULL` silently drops these rows (≥18 production paths). Bias scales with target distance — MNQ NYSE_OPEN 15m: 9.9% scratch at RR=1.0 → 44.6% at RR=4.0. Empirical ExpR inflation 10–45% (verified 2026-04-27 IS only). Sign of every conclusion preserved (0/144 sign flips).

Stage progression:
- Stage 0: Literature extraction (3 new files in `docs/institutional/literature/`) — DONE
- Stage 1: Failure log + memory + CORRECTION result file + hypothesis YAML correction_notice — DONE
- Stage 2: Drift-check guard `check_research_scratch_policy_annotation` — IN PROGRESS
- Stage 3: Pre-registered Criterion 13 + mechanism_priors cross-ref + prereg-writer prompt update
- Stage 4: DESIGN ONLY spec at `docs/specs/outcome_builder_scratch_eod_mtm.md` (USER REVIEW GATE)
- Stage 5: `outcome_builder.py` 4-site fix + 4 unit tests + companion drift check `check_orb_outcomes_scratch_pnl`
- Stage 5b: orb_outcomes rebuild for MNQ/MES/MGC × {5,15,30}m (DESTRUCTIVE-SHARED — USER REVIEW GATE)
- Stage 6: Downstream re-verification (USER REVIEW GATE if any DEPLOYED lane flips to DECAY)
- Stage 7: paper_trades parity audit
- Stage 8: MFE-distribution endogenous-RR research

## scope_lock

scope_lock:
  - pipeline/check_drift.py
  - tests/test_pipeline/test_check_drift_scratch_policy.py
  - trading_app/outcome_builder.py
  - tests/test_trading_app/test_outcome_builder.py
  - docs/institutional/literature/carver_2015_ch12_speed_and_size.md
  - docs/institutional/literature/chan_2009_ch1_intraday_session_handling.md
  - docs/institutional/literature/bailey_lopezdeprado_2014_dsr_sample_selection.md
  - docs/institutional/pre_registered_criteria.md
  - docs/institutional/mechanism_priors.md
  - docs/prompts/prereg-writer-prompt.md
  - docs/specs/outcome_builder_scratch_eod_mtm.md
  - docs/specs/paper_trades_scratch_policy.md
  - .claude/rules/backtesting-methodology-failure-log.md
  - memory/feedback_scratch_pnl_null_class_bug.md
  - memory/MEMORY.md
  - docs/audit/results/2026-04-27-mnq-unfiltered-high-rr-family-v1-CORRECTION.md
  - docs/audit/results/2026-04-27-canonical-scratch-fix-downstream-impact.md
  - docs/audit/results/2026-04-27-paper-trades-scratch-parity.md
  - docs/audit/hypotheses/2026-04-27-mnq-unfiltered-high-rr-family-v1.yaml
  - docs/audit/hypotheses/2026-04-27-mnq-mfe-distribution-endogenous-rr-v1.yaml
  - docs/audit/results/2026-04-27-mnq-mfe-distribution-endogenous-rr-v1.md
  - research/mnq_mfe_distribution_endogenous_rr_v1.py
  - trading_app/paper_trader.py
  - tests/test_trading_app/test_paper_trader.py
  - docs/runtime/stages/scratch-eod-mtm-canonical-fix.md

## Blast Radius

- `pipeline/check_drift.py` — additive: 2 new check functions (`check_research_scratch_policy_annotation` Stage 2, `check_orb_outcomes_scratch_pnl` Stage 5). Both registered in CHECKS list. Drift-check count auto-increments via `len(CHECKS)`.
- `trading_app/outcome_builder.py` — behavior change at 4 sites: scratches now populate `pnl_r`/`exit_ts`/`exit_price` from last bar of `post_entry`. Pathological `post_entry.empty` retains current NULL behavior.
- `orb_outcomes` (DB) — Stage 5b rebuild touches all rows for MNQ/MES/MGC × {5,15,30}m. ~2.1M rows MNQ alone. Multi-hour. Idempotent DELETE+INSERT per pipeline-patterns rule.
- Downstream consumers (Stage 6) — `/trade-book` MCP, `get_strategy_fitness`, `trading_app/portfolio.py`, `trading_app/sprt_monitor.py`, `trading_app/sr_monitor.py`, `trading_app/pbo.py`, `trading_app/rolling_correlation.py`, `trading_app/rolling_portfolio.py`, `validated_setups.expectancy_r`. Re-verified, possibly re-derived, but no production code edits unless behavior is broken by data fix.
- `trading_app/paper_trader.py` — Stage 7 conditional fix if `paper_trades` table has same NULL-on-scratch pattern.

## Stage 5b safety

The orb_outcomes rebuild is the highest-blast-radius action in this plan. Before kicking off:
- User explicit OK required (USER REVIEW GATE per plan).
- Backup of current `gold.db` advised before rebuild.
- MNQ first (smallest production blast radius for live trading).
- Sequential per-instrument-aperture to avoid DuckDB write-lock contention.

## What's been completed

- 3 literature extracts written. Chan 2009 §1.4 honestly marked UNSUPPORTED (does not exist in PDF).
- Failure log entry appended (2026-04-27 entry citing all line numbers).
- Memory feedback file + MEMORY.md index entry.
- CORRECTION result file with corrected ExpR for all 5 CANDIDATE_READYs + 4 RESEARCH_SURVIVORs + 10 worst-case KILL cells.
- Hypothesis YAML carries correction_notice block.
- 0 of 144 cells flipped sign under scratch-as-0R policy — directional conclusions all preserved.
