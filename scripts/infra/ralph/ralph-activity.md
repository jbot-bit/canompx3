# Ralph Activity Log

CG audit started 2026-03-05. Fresh run — all tasks pending.

## 2026-03-05 — Task 1: CG Pass 1 — Pipeline M2.5 Audit + Triage

**What:** Ran M2.5 audit (bugs mode) on 6 pipeline files in 2 batches:
- Batch 1: ingest_dbn.py, build_bars_5m.py, build_daily_features.py
- Batch 2: dst.py, cost_model.py, asset_configs.py

**Triage results:** 11 findings total
- 0 TRUE findings requiring action
- 2 TRUE but no-action (GARCH warm-up perf, validate_catalog dead branch)
- 8 FALSE POSITIVES (ATR velocity off-by-one was wrong, MBT/dead instrument claims irrelevant, is_winter_for_session is NOT dead code)
- 1 WORTH EXPLORING (brisbane_1025_brisbane naming)
- M2.5 false positive rate: 73%

**Key verification:** ATR velocity check at build_daily_features.py:1067-1071 — `range(max(0, i-5), i)` gives exactly 5 elements when i>=5. M2.5 claimed off-by-one but was wrong.

**Files modified:** scripts/infra/ralph/ralph-audit-report.md (CG-1 section added)
**Output files:** scripts/infra/ralph/m25_pipeline.md, scripts/infra/ralph/m25_pipeline2.md

### Re-run (same session): Fresh M2.5 produced 16 findings (vs 11 prior). 13/16 FALSE POSITIVE (81%). 0 new TRUE findings.
Notable: M2.5 hallucinated stress_test_costs() missing slippage multiplier — actual code at line 357 multiplies it.
Re-run confirmation table appended to CG-1 section in ralph-audit-report.md.

## 2026-03-05 — Task 2: CG Pass 1 — Trading App M2.5 Audit + Triage

**What:** Ran M2.5 audit (bias mode) on 4 trading_app files in 2 batches:
- Batch 1: outcome_builder.py, strategy_discovery.py
- Batch 2: strategy_validator.py, config.py

**Triage results:** 7 findings total
- 0 TRUE findings requiring action
- 1 PARTIALLY TRUE but no-action (ATR query unbounded — functionally harmless)
- 6 FALSE POSITIVES
- M2.5 false positive rate: 86%

**Key verifications:**
- FDR not in experimental_strategies: BY DESIGN — validated_setups is the FDR target. Intentional separation.
- FST hurdle uses full grid: CONSERVATIVE direction (safe). Intentional.
- stress_test_costs slippage: ALREADY MULTIPLIED at line 357 (M2.5 hallucinated again in batch 1 re-run).

**Files modified:** scripts/infra/ralph/ralph-audit-report.md (CG-2 section added)
**Output files:** scripts/infra/ralph/m25_trading1.md, scripts/infra/ralph/m25_trading2.md

## 2026-03-05 — Task 3: CG Pass 1 — ML Module M2.5 Audit + Triage

**What:** Ran M2.5 audit on 5 ML module files in 2 batches:
- Batch 1: meta_label.py, cpcv.py (bias mode)
- Batch 2: features.py, evaluate.py, predict_live.py (bugs mode)

**Triage results:** 14 findings total
- 0 TRUE findings requiring action
- 1 TRUE but no-action (is_bool_feat redundant init — cosmetic)
- 1 PARTIALLY TRUE but no-action (backfill trigger — 3 features vs full GLOBAL_FEATURES list)
- 9 FALSE POSITIVES
- 3 WORTH EXPLORING (threshold sweep docs, feature drift expansion, column drop transparency)
- M2.5 false positive rate: 64%

**Key verifications:**
- duckdb import in features.py: IS used at lines 41, 468, 564, 714. M2.5 didn't read full file.
- GLOBAL_FEATURES import in predict_live.py: LAZY import (not duplicate). M2.5 hallucinated "module level import at line 23".
- stress_test_costs at cost_model.py:357: slippage IS multiplied (`spec.slippage * multiplier`). M2.5 pipeline2 re-run was wrong.
- Threshold sweep: real concern but 4-gate quality system is the documented mitigation.

**Files modified:** scripts/infra/ralph/ralph-audit-report.md (CG-3 section added)
**Output files:** scripts/infra/ralph/m25_ml1.md, scripts/infra/ralph/m25_ml2.md
