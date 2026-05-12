# HANDOFF.md — Cross-Tool Session Baton

**Rule:** If you made decisions, changed files, or left work half-done — update the baton.

**CRITICAL:** Do NOT implement code changes based on stale assumptions. Always `git log --oneline -10` and re-read modified files before writing code.

**Compact baton only:** Durable decisions live in `docs/runtime/decision-ledger.md`, design history lives in `docs/plans/`, and archived session detail lives in `docs/handoffs/archived/`.

## Last Session
- **Tool:** Claude Code
- **Date:** 2026-05-12
- **Commit:** 526f3d87 — chore(resources): add ml4am code companion (third-party LdP 2020 implementations)
- **Files changed:** 20 files
  - `HANDOFF.md`
  - `resources/ml4am_code_companion/Chapter_2/02Denoising and Detoning.ipynb`
  - `resources/ml4am_code_companion/Chapter_2/AssetManagerMachineLearning.py`
  - `resources/ml4am_code_companion/Chapter_2/Chapter 2.ipynb`
  - `resources/ml4am_code_companion/Chapter_2/ConstantResidual.py`
  - `resources/ml4am_code_companion/Chapter_2/Detoning.py`
  - `resources/ml4am_code_companion/Chapter_2/FittingMarcenkoPastur.py`
  - `resources/ml4am_code_companion/Chapter_2/MarcenkoPastur.py`
  - `resources/ml4am_code_companion/Chapter_2/RandomMatrixWithSignal.py`
  - `resources/ml4am_code_companion/Chapter_2/TargetedShrinkage.py`
  - `resources/ml4am_code_companion/Chapter_3/03DistanceMetrics.ipynb`
  - `resources/ml4am_code_companion/Chapter_3/AssetManagerMachineLearning.py`
  - `resources/ml4am_code_companion/Chapter_3/ConstantResidual.py`
  - `resources/ml4am_code_companion/Chapter_3/Detoning.py`
  - `resources/ml4am_code_companion/Chapter_3/FittingMarcenkoPastur.py`
  - ... and 5 more

## Next Steps — Active
1. Track D MNQ COMEX_SETTLE Gate 0 runner design — Design the Databento top-of-book table and bounded runner needed to execute the DESIGN_ONLY prereg.

## Durable References
- `docs/runtime/action-queue.yaml`
- `docs/runtime/decision-ledger.md`
- `docs/runtime/debt-ledger.md`
- `docs/plans/2026-04-22-handoff-baton-compaction.md`
