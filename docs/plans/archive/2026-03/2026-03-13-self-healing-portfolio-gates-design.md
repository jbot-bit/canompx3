---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Self-Healing Portfolio Gates

**Date:** 2026-03-13
**Status:** Approved (rev 2 — incorporates code-path review feedback)
**Triggered by:** Adversarial audit of MGC CME_REOPEN G4_FAST10 (SUSPICIOUS) and MNQ CME_PRECLOSE VOL_RV12_N20 (LEGIT BUT DECAYING)

## Problem

`LiveStrategySpec` is binary — strategies are fully on (weight=1.0) or excluded. No mechanism for:
- Partial weights (decay response)
- Seasonal restrictions (month-based gating)
- Auto-recovery (promote back when rolling eval shows the edge returned)

Demoted strategies risk going dormant and forgotten. The system must self-heal.

## Design

### Scope

**v1 applies to CORE specs only.** Gate precedence within the core loop:
1. `exclude_instruments` (existing)
2. Seasonal skip (`active_months`)
3. Variant selection (rolling → baseline fallback, existing)
4. Dollar gate (existing)
5. Weight override + recovery (`weight_override`, `recovery_expr_threshold`)

REGIME tier already has fitness gating. HOT tier is dormant. Neither is touched.

### New Fields on `LiveStrategySpec`

```python
@dataclass(frozen=True)
class LiveStrategySpec:
    # ... existing fields ...

    active_months: frozenset[int] | None = None
    weight_override: float | None = None
    recovery_expr_threshold: float | None = None
```

**Validation** (`__post_init__`):
- `active_months`: all values in 1..12, non-empty if set
- `weight_override`: in [0.0, 1.0] if set
- `recovery_expr_threshold`: > 0.0 if set; requires `weight_override` to also be set
  (recovery without demotion is meaningless; seasonal-only specs should NOT set this)

### Rolling Metric Fix (critical)

`load_rolling_validated_strategies()` returns the latest-window variant, but recovery
needs the **family rolling average** `fam.avg_expectancy_r` — not single-window ExpR.

**Change in `rolling_portfolio.py`**: After building each result dict (line ~493), inject:
```python
result["rolling_avg_expectancy_r"] = fam.avg_expectancy_r
result["rolling_weighted_stability"] = fam.weighted_stability
```

Recovery then compares `match["rolling_avg_expectancy_r"] >= spec.recovery_expr_threshold`.

### Date Handling

`build_live_portfolio` gains `as_of_date: date | None = None`. If None, defaults to
`date.today()`. Month resolution: `as_of_date.month`. No bare `datetime.now()`.

### Build-Time Logic (`build_live_portfolio`)

In the core spec loop:

1. **Seasonal gate** (hard skip): `as_of_date.month not in spec.active_months` → `continue`.
   No variant loaded, no DB query. Recovery cannot override seasonality.

2. **Weight resolution** (after match + dollar gate):
   ```
   weight = 1.0
   if spec.weight_override is not None:
       weight = spec.weight_override  → log "DEMOTED"
       if (spec.recovery_expr_threshold is not None
           and source == "rolling"
           and match["rolling_avg_expectancy_r"] >= spec.recovery_expr_threshold):
           weight = 1.0  → log "RECOVERED"
   ```

   Recovery only fires when `source == "rolling"`. Baseline fallback cannot trigger
   recovery — if the strategy isn't STABLE in rolling eval, it shouldn't auto-promote.

### Observability

CLI output (`main()`) adds a third section for fractional-weight strategies:

```
Active strategies: N
  ...
Demoted strategies (0 < weight < 1): M
  strategy_id   weight  note
  ...
Gated OFF (weight=0): K
  ...
```

### Applied Configurations

**MGC CME_REOPEN E2 ORB_G4_FAST10** (adversarial verdict: SUSPICIOUS)
- `active_months=frozenset({11, 12, 1, 2})` — Nov-Feb only
- No `weight_override` or `recovery_expr_threshold` — seasonal gate is the full restriction.
  Recovery does not apply because seasonal is a hard skip before variant lookup.

**MNQ CME_PRECLOSE E2 VOL_RV12_N20** (adversarial verdict: LEGIT BUT DECAYING)
- `weight_override=0.5` — half weight due to monotonic decay
- `recovery_expr_threshold=0.25` — auto-promote if rolling avg ExpR recovers > 0.25

### Threshold Rationale

| Strategy | Gate | Derivation |
|----------|------|------------|
| MGC FAST10 | Seasonal Nov-Feb | 68% of wins in Q4+Q1, June-July nearly dead |
| MNQ VOL recovery | 0.25R rolling avg | Midpoint of 2024 (0.40R) and 2025 (0.30R) decay |

### Institutional Grounding

Design decisions are grounded in the following institutional references from `resources/`:

| Design Element | Source | Rationale |
|----------------|--------|-----------|
| `weight_override` as fractional allocation | Carver *Systematic Trading* Ch.7 (forecast scaling: -20 to +20, 10=avg buy, 5=weak buy) | Positions should be continuously scaled by conviction, not binary on/off. `weight_override=0.5` ≈ forecast of 5 in Carver's framework. |
| `weight_override=0.5` default demotion | Chan *Algorithmic Trading* Ch.8 (half-Kelly), Carver Ch.4 Table 12 (Column B weight adjustments) | Half-Kelly is the canonical risk-reduction factor under parameter uncertainty. Carver's Table 12 shows 0.50 SR disadvantage → multiplier of 0.32-0.65. |
| Recovery must pass same gates | Man AHL *Overfitting and Its Impact* (rejection rate monitoring, staged lifecycle) | Man AHL: recovery requires passing the same gates again, not a lower bar. Demotion is not permanent — strategies can recover after structural changes revert. |
| Never auto-kill demoted strategies | Man AHL (non-stationarity discussion): decay can be (a) overfitting, (b) structural change, (c) crowding | Only (a) should be permanently removed (already handled by BH FDR + WF). Categories (b) and (c) can revert — preserve optionality. |
| CUSUM for future regime detection | Pepelyshev & Polunchenko *Real-Time Financial Surveillance via CUSUM* | CUSUM detects structural breaks faster than rolling averages. Future enhancement: dual CUSUM (negative accumulator for demotion, positive for recovery). |
| Rolling avg as fitness metric | Carver Ch.9, Ch.12 (rolling volatility estimation) | Rolling windows are standard in institutional practice. Applied to expectancy rather than volatility, but same principle of time-weighted evidence. |
| `active_months` seasonal gating | **No academic support** — purely empirical finding | Zero institutional references support calendar-month gating. Held to higher evidence bar: BH FDR survived + 68% win concentration in Q4+Q1. Treat as data-driven restriction, not methodology. |

### Files Touched

1. `trading_app/rolling_portfolio.py` — thread `rolling_avg_expectancy_r` + `rolling_weighted_stability` into returned dicts (~2 lines)
2. `trading_app/live_config.py` — dataclass fields + `__post_init__` validation + `as_of_date` param + build logic + CLI output + 2 spec edits
3. `tests/test_trading_app/test_live_config.py` — validation, seasonal skip, weight override, recovery

### Not In Scope

- Applying gates to other strategies (audit showed no other red flags)
- Recovery overriding seasonal gate (by design: seasonal is permanent)
- REGIME/HOT tier changes
- New DB tables or schema changes
- Pinecone snapshot updates (defer to next sync)

## Adversarial Audit Context

### Portfolio-Wide Findings (33 strategies audited)
- 107/132 unfiltered session combos are NEGATIVE — filters work everywhere
- MGC is most filter-dependent instrument (biggest lifts 0.50-0.80R)
- MNQ is most robust (570 FDR-significant strategies)
- M2K CME_PRECLOSE and NYSE_CLOSE are dead sessions (no filter saves them)
- No other strategies showed seasonal clustering or monotonic decay red flags

### Target Strategy Detail

**MGC CME_REOPEN G4_FAST10 (SUSPICIOUS)**
- Unfiltered session: -0.348R (NEGATIVE)
- G4 only: +0.209R | G4+FAST10: +0.365R — double filter dependency
- 68% wins in Nov-Feb, June-July nearly dead
- Top 3 trades = 34% of PnL
- FDR p=0.001, WF 84.3% retention (stats are clean)
- Cost exceeds risk at G4 minimum (4pts = $40 vs $41.74 RT)

**MNQ CME_PRECLOSE VOL_RV12_N20 (LEGIT BUT DECAYING)**
- Unfiltered session: +0.286R (ALREADY POSITIVE)
- Filter adds only +0.054R (enhancer, not crutch)
- Monotonic decay: 0.54R (2022) -> 0.30R (2025) at -0.06R/year
- FDR p < 1e-6, #1 in cell, 5 WF windows 79.5% retention
- Cost-immune (4.6% drag, negligible at 1.5x slippage)
- Top 3 trades = 6% of PnL (no outlier dependency)
