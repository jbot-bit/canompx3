## Iteration: 130
## Target: trading_app/ml/meta_label.py:386
## Finding: CPCV failure logged at DEBUG (invisible at INFO level), silently bypasses Gate 2 (CPCV AUC check)
## Classification: [mechanical]
## Blast Radius: 1 file (meta_label.py only — logging change, no callers affected)
## Invariants: 1) cpcv_auc remains None on failure (Gate 2 skips with `if cpcv_auc is not None`), 2) training continues after CPCV failure, 3) no functional behavior change
## Diff estimate: 1 line
