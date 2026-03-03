# ML Meta-Label Live Integration Spec

## Date: 2026-03-03
## Status: IN PROGRESS

## Goal
Wire ML meta-label P(win) predictions into the paper trader and execution engine so trades are automatically skipped when the model says P(win) < threshold.

## Architecture

```
LiveMLPredictor (trading_app/ml/predict_live.py)
  ├─ Loads RF model + threshold from models/ml/meta_label_{inst}.joblib
  ├─ Builds feature vector from daily_features + strategy params
  ├─ Predicts P(win) via model.predict_proba()
  └─ Returns (p_win, take) where take = p_win >= threshold

Paper Trader (Phase 1)
  └─ --use-ml flag → instantiate LiveMLPredictor
      └─ After ENTRY event → check ML → skip if take=False

Execution Engine (Phase 2 — future)
  └─ In _arm_strategies() → check ML before creating ActiveTrade
```

## Phase 1: Paper Trader Integration

### New File: `trading_app/ml/predict_live.py`

**Class:** `LiveMLPredictor`

**Interface:**
```python
predictor = LiveMLPredictor(db_path, instruments=["MGC", "MNQ", "MES", "M2K"])

# For a given trading day + strategy combo:
p_win, take = predictor.predict(
    instrument="MNQ",
    trading_day=date(2026, 3, 3),
    orb_label="US_DATA_830",
    orb_minutes=5,
    entry_model="E2",
    rr_target=1.5,
    confirm_bars=1,
)
# p_win = 0.63, take = True (>= 0.59 threshold)
```

**Internals:**
1. On init: load all model .joblib files, cache in dict
2. On first predict for a (instrument, trading_day): load daily_features row from DB, cache
3. Build feature vector: global features + session features + trade config + categoricals
4. Align to model's stored `feature_names` order
5. Fill missing features (0.0 for one-hot, -999 for numeric)
6. predict_proba → p_win
7. Cache result per (instrument, trading_day, orb_label, orb_minutes, entry_model, rr_target, confirm_bars)

**Fail modes:**
- Missing model file → WARNING log, return (0.5, True) — fail-open, trade anyway
- Config hash mismatch → WARNING log, still predict (model may be slightly stale)
- Model >90 days old → WARNING log, still predict
- Feature build error → WARNING log, return (0.5, True) — fail-open

### Modified File: `trading_app/paper_trader.py`

**Changes:**
- Add `--use-ml` CLI flag (default False)
- When True: create LiveMLPredictor before replay loop
- In replay loop: after engine returns ENTRY events, check ML
- If take=False: log ML_SKIP in journal (p_win, threshold, strategy_id)
- Summary output: trades taken vs skipped, avgR with vs without ML

### New Drift Checks (in check_drift.py)

| # | Check | Severity |
|---|-------|----------|
| 51 | Model files exist for all active instruments | FAIL |
| 52 | Config hash matches current config | WARNING |
| 53 | Model freshness < 90 days | WARNING |

## Design Principles (per de Prado AIFML)

1. **Meta-labeling = secondary model.** Primary model (ORB rules) says direction. ML says confidence. ML never overrides direction.
2. **Fail-open.** Missing model = trade anyway. The edge exists without ML. ML only improves it.
3. **One model per instrument.** Session/entry_model/rr_target are INPUT features, not separate models. This prevents overfitting to small subsets.
4. **CPCV-validated threshold.** The threshold was optimized on OOS holdout with purge+embargo. Not perfect (Phase 2 nested CV) but sound.
5. **Feature alignment via stored feature_names.** Model bundle stores exact column order. Prediction aligns to that order. Unknown categories get fill values.
6. **No retraining in production.** Retrain is a manual step after data changes. Drift checks catch staleness.

## NOT in scope
- Real-time engine integration (Phase 2)
- Kelly bet sizing from P(win) (future research)
- Auto-retrain scheduler (manual on demand)
- Per-combo models (one per instrument is correct granularity)
- Trade sheet ML column (add after paper trader proves value)

## Drift Protection
- `config_hash` in model bundle → any feature config change invalidates
- `trained_at` timestamp → staleness detection
- `feature_names` stored → alignment check at prediction time
- Rebuild chain updated: retrain ML after any discovery re-run
