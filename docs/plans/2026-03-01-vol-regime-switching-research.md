# Research Plan: Volatility Regime-Dependent Parameter Switching

**Date:** 2026-03-01
**Script:** `research/research_vol_regime_switching.py`
**Output:** `research/output/vol_regime_switching_summary.md` + `research/output/vol_regime_switching_results.csv`

## Hypothesis

**H0:** Optimal ORB breakout parameters (RR target, filter stringency) are independent of the current volatility regime.
**H1:** Different vol regimes have systematically different optimal parameters, and regime-adaptive switching improves risk-adjusted returns.

## Mechanism

- High-vol: larger ORBs, friction is smaller fraction of risk, higher RR targets reachable
- Low-vol: smaller ORBs, friction eats more, lower RR targets more realistic, stricter G-filters needed
- Extends existing ATR contraction AVOID signal from binary to continuous regime adaptation

## Regime Definition

**Primary variable:** `atr_20` from `daily_features` (20-day ATR SMA).

**Classification:** Expanding-window percentile rank with `shift(1)` (no look-ahead). Minimum 60-day warmup.

```python
df["atr_pct_rank"] = df.groupby("symbol")["atr_20"].transform(
    lambda s: s.expanding(min_periods=60).rank(pct=True).shift(1)
)
df["vol_regime"] = pd.cut(df["atr_pct_rank"], bins=[0, 0.333, 0.667, 1.001],
                          labels=["LOW", "MID", "HIGH"], right=False)
```

**Lookahead assertion:** Script must verify `vol_regime[i]` depends only on `atr_20[0:i-1]`.

## Parameter Grid Per Regime

| Parameter | Values |
|-----------|--------|
| RR Target | 1.0, 1.5, 2.0, 2.5, 3.0, 4.0 |
| G-Filter | G4, G5, G6, G8, NO_FILTER |
| Confirm Bars | CB1, CB2 |
| Entry Models | E1, E2 |

## Test Count

~1,500-2,500 effective tests after N>=30 filtering (from ~5,760 raw combos).

**BH FDR at q=0.05** (stricter than standard q=0.10 due to high comparison count).

## Statistical Tests

1. **Per-regime one-sample t-test** vs zero for each (instrument, session, entry_model, regime, RR, filter, CB)
2. **Regime comparison:** Kruskal-Wallis (3-group) then pairwise Welch t-tests
3. **Optimal parameter shift:** Does best RR/filter differ across regimes?
4. **Adaptive vs static:** Paired t-test on daily pnl_r -- regime-adaptive vs best single parameter set

## Key Risks

- Small N per tercile (N/3 per regime)
- MGC regime shift (gold price tripled -> early years systematically low-vol)
- MNQ only ~2 years of data -> REGIME class at best
- Finding may just proxy for ORB size (already captured by G4+)

## Data Sources

```sql
SELECT d.trading_day, d.symbol, d.atr_20, d.orb_{SESSION}_size,
       o.rr_target, o.confirm_bars, o.entry_model, o.pnl_r
FROM daily_features d
JOIN orb_outcomes o ON o.trading_day = d.trading_day
  AND o.symbol = d.symbol AND o.orb_minutes = d.orb_minutes
WHERE d.orb_minutes = 5 AND o.outcome IN ('win','loss') AND o.pnl_r IS NOT NULL
```

## Script Pattern

Follow `research/research_avoid_crosscheck.py` for multi-instrument vol analysis with BH FDR.

## Canonical Imports

```python
from pipeline.asset_configs import ACTIVE_ORB_INSTRUMENTS, get_enabled_sessions
from pipeline.paths import GOLD_DB_PATH
```
