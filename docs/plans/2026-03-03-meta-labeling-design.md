# Meta-Labeling Design — v2.0

**Date:** 2026-03-03
**Status:** Implementation
**Source:** de Prado AIFML (NotebookLM Backtesting Rules), ML best practices research

## Problem

817 validated ORB strategies fire signals. Currently we take ALL of them.
Meta-labeling adds a secondary ML classifier: P(win | market_state, config) → skip/take.

## Architecture Decision: Per-Instrument Models (NOT Per-Combo)

**v1.0 (rejected):** 207 separate models, 1,000-2,500 samples each → fragile, overfitting risk
**v2.0 (chosen):** 4 per-instrument models, 100K-850K samples each → robust, model LEARNS session/config effects

Each instrument gets ONE Random Forest. Session, entry model, RR target, filter type, and aperture
become categorical INPUT FEATURES. The model discovers interaction effects (e.g., "TOKYO_OPEN + low ATR +
small gap → high P(win)") that our grid search never tested.

Sample sizes per instrument (O5 aperture alone):
- MGC: 852K | MNQ: 577K | MES: 729K | M2K: 578K

## Module Structure

```
trading_app/ml/
  __init__.py         # Version, public API
  config.py           # Feature lists, hyperparams, look-ahead blacklist
  features.py         # DB → feature matrix (shared by train + live)
  cpcv.py             # Combinatorial Purged Cross-Validation
  importance.py       # MDI/MDA feature importance (Phase 1)
  meta_label.py       # Classifier train/predict/evaluate (Phase 2-3)
  evaluate.py         # Before/after quantification (Phase 3)
```

## Feature Engineering

### Global Market State (normalized by ATR where applicable)
- `atr_20`, `atr_vel_ratio`, `rsi_14_at_CME_REOPEN`
- `gap_open_points / atr_20` (stationary gap measure)
- `garch_atr_ratio`, `garch_forecast_vol`
- `prev_day_range / atr_20`, `overnight_range / atr_20`
- `prev_day_direction` (categorical: UP/DOWN)
- `gap_type` (categorical)

### Session-Specific (dynamically extracted per orb_label)
- `orb_size / atr_20` (ORB range normalized)
- `compression_z` (if available for session)
- `rel_vol` (relative volume at break)
- `break_delay_min` (time from ORB formation to break)
- `break_bar_continues` (momentum signal)
- `break_dir` (LONG/SHORT — known at entry time)

### Trade Configuration (from orb_outcomes)
- `orb_label` (one-hot: 11 sessions)
- `entry_model` (one-hot: E1/E2/E3)
- `rr_target` (numeric: 1.0-4.0)
- `confirm_bars` (numeric: 1-5)
- `orb_minutes` (5/15/30)

### Calendar
- `day_of_week` (one-hot: Mon-Fri)
- `is_nfp_day`, `is_opex_day`

### Look-Ahead BLACKLIST (NEVER use)
- `double_break` — full-session look-ahead
- `day_type` — full-session look-ahead
- `outcome`, `mae_r`, `mfe_r`, `pnl_r` — these ARE the target
- `took_pdh_before_1000`, `took_pdl_before_1000` — time-dependent, may be look-ahead

## Validation: CPCV

Per de Prado: "Makes overfitting practically impossible."

Config: N=10 groups, k=2 test groups = C(10,2) = 45 unique train/test splits.
Purge: 1 trading day. Embargo: 1 trading day.
Groups defined by trading_day (time-ordered), NOT random.

## Threshold Optimization

Optimize P(win) threshold on PROFIT METRIC (total R improvement), not accuracy.
For each threshold t ∈ [0.40, 0.70]:
  - Keep trades where P(win) >= t
  - Compute total_R, avgR, Sharpe on kept trades
  - Select t* that maximizes Sharpe improvement vs baseline

Expected: skip 30-50% of signals, +10-30% avgR improvement on kept signals.

## RF Hyperparameters

```python
n_estimators=500, max_depth=6, min_samples_leaf=50,
max_features='sqrt', class_weight='balanced', random_state=42
```

Conservative depth (6) prevents overfitting. `min_samples_leaf=50` ensures stable leaves.
`class_weight='balanced'` handles 20-33% win rate imbalance (no SMOTE needed).

## Downstream Integration (Future-Proofing)

### Phase 1 → live_config.py
Add `ml_threshold` field to ExecutionSpec. paper_trader checks P(win) >= threshold.

### Phase 2 → Bet Sizing (Kelly)
P(win) directly feeds Kelly fraction: f* = (p*b - q) / b where p=P(win), b=RR, q=1-p.
Same model, new output consumer. No retraining needed.

### Phase 3 → Cross-Session Context
Earlier session outcomes (e.g., CME_REOPEN result) become features for later sessions
(e.g., NYSE_OPEN). FeatureBuilder supports this via session time ordering.

### Phase 4 → Adaptive Retraining
strategy_fitness monitors ML model AUC monthly. When AUC drops below training AUC - 1σ,
trigger retrain. Hooks into existing FIT/WATCH/DECAY framework.

### Phase 5 → New Features
Add columns to daily_features → add to config.py FEATURE_LIST → retrain.
No structural code changes. Feature pipeline is declarative.

## What This Does NOT Do

- Does NOT replace the ORB rule-based system (that's the primary model)
- Does NOT predict direction (the ORB break already determines direction)
- Does NOT require new data sources (all features already in daily_features)
- Does NOT change the validation pipeline (BH FDR, walk-forward still run)
- Does NOT position-size (Phase 2 handles that, using same P(win) output)
