# Adaptive Lane Allocator — Design Spec

**Date:** 2026-04-02
**Status:** APPROVED (pending implementation)
**Author:** Claude + Josh
**Grounding:** Carver Ch.11-12, LdP Ch.12, Chan Ch.7, Pardo Ch.9

---

## Purpose

Monthly rebalancer that weights validated strategy lanes by trailing forward performance. Rotates capital to sessions that are HOT now, pauses sessions that have gone cold. Integrates with prop_profiles.py for deployment.

**What breaks without it:** NYSE_CLOSE lost 184R in 2025 with no alert. TOKYO_OPEN went cold with no pause. Static lanes don't adapt to regime shifts. Capital sits in dead sessions while hot sessions are undeployed.

---

## Architecture: Two-Layer System

### Layer 1: Validation (Static)
- BH FDR + walk-forward on full history (2016-2025)
- Proves the edge EXISTS with statistical rigor
- Updated quarterly or after pipeline rebuild
- Output: set of validated strategy_ids (currently 210)

### Layer 2: Allocation (Dynamic)
- Monthly rebalancer using trailing performance
- Weights validated lanes by recent forward data
- Updated 1st of each month
- Output: ranked lanes, weights, pause/resume signals
- Written to: `docs/runtime/lane_allocation.json`
- Human reviews and applies to prop_profiles.py

---

## Trailing Windows

| Purpose | Window | Justification |
|---------|--------|---------------|
| **Deploy ranking** | 12 months trailing | Carver Ch.11: forecast weighting by recent accuracy. Balances responsiveness with stability. |
| **Session regime gate** | 6 months trailing | Chan Ch.7: regime half-life = 1-3 months. 6 months captures full regime cycles. Used for REGIME-class strategy gating. |
| **Pause trigger** | 2 months consecutive negative | Chan Ch.7 + Pardo Ch.9: act within 1-3 months. Magnitude override: 3mo ExpR < -0.10 = immediate. |
| **Resume trigger** | 3 months consecutive positive | Carver Ch.12: asymmetric switching — harder to turn ON than OFF. Recovery needs more proof than death. |

---

## Regime Detection: Two-Level

### CORE strategies (N >= 100)
- Use OWN trailing ExpR for regime detection
- 12-month trailing window has 20+ trades per month = robust signal
- Self-sufficient regime gate

### REGIME strategies (N = 30-99)
- Use SESSION-LEVEL trailing ExpR (unfiltered E2 RR1.0)
- Strategy's own trailing window has too few trades (5-15/year = noise)
- Session unfiltered has 200-500 trades/year = robust signal
- If session is HOT → deploy the REGIME strategy
- If session is COLD → pause the REGIME strategy regardless of own performance
- Grounding: Carver Ch.11 "forecast blending" — use higher-level signal for lower-level bet

---

## Ranking Metric

```
annual_r_estimate = trailing_expr * trailing_n / (trailing_months / 12)
```

- Proportional to both per-trade edge AND trade frequency
- Self-correcting: decaying lanes drop automatically
- One winner per instrument x session (highest annual_r)
- Track top 3 per session in report (for monitoring, not deployment)

---

## Kill/Resume Rules

### PAUSE (fast kill)
- Trailing 2-month ExpR < 0 for 2 consecutive months → PAUSE
- OR: trailing 3-month ExpR < -0.10 → IMMEDIATE PAUSE (magnitude override)
- REGIME strategies: session-level 2-month ExpR < 0 → PAUSE

### RESUME (slow promote)
- Must be POSITIVE for 3 consecutive months after pause
- AND: trailing 12-month annual_r > 0
- REGIME strategies: session must be positive for 3 months at session level

### Hysteresis (anti-churn, Carver Ch.12)
- Only replace a deployed lane if the new candidate scores >20% higher in annual_r
- Prevents monthly churn for marginal improvements
- "Switching costs" — even a better strategy costs something in the transition

---

## Constraints (from AccountProfile)

The allocator respects per-profile constraints:
- `max_slots`: maximum lanes per account (cognitive cap for manual, slot cap for auto)
- `allowed_sessions`: which sessions this account can trade (time-based for manual)
- `allowed_instruments`: which instruments this account can trade (firm restrictions)
- `stop_multiplier`: prop firm stop sizing (0.75x for prop, 1.0x for self-funded)

Manual vs auto uses the SAME ranking — only `allowed_sessions` differs (manual = Brisbane daytime, auto = overnight).

---

## Data Model

```python
@dataclass
class LaneScore:
    strategy_id: str
    instrument: str
    orb_label: str
    rr_target: float
    filter_type: str
    stop_multiplier: float
    trailing_expr: float          # Mean pnl_r in 12mo trailing window
    trailing_n: int               # Trade count in 12mo trailing window
    trailing_months: int          # Window size (12)
    annual_r_estimate: float      # trailing_expr * trailing_n / (trailing_months / 12)
    trailing_wr: float            # Win rate in trailing window
    session_regime_expr: float    # Session-level 6mo trailing ExpR (unfiltered)
    months_negative: int          # Consecutive months with negative ExpR
    months_positive_since_pause: int  # Consecutive positive months since last pause
    status: str                   # DEPLOY / REDUCE / PAUSE / RESUME
    status_reason: str            # Why this status
```

---

## Output

### Machine-readable: `docs/runtime/lane_allocation.json`
```json
{
  "rebalance_date": "2026-04-01",
  "trailing_window_months": 12,
  "profiles": {
    "apex_100k_manual": {
      "lanes": [
        {"strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G6", "annual_r": 18.0, "status": "DEPLOY"},
        ...
      ],
      "paused": [...],
      "changes_from_prior": [...]
    }
  }
}
```

### Human-readable: stdout report
- All candidate scores (ranked)
- Selected lanes per profile
- Paused lanes with reasons
- Changes from prior month
- Session regime status (HOT/COLD/STABLE)

---

## Files

| File | Action | Lines |
|------|--------|-------|
| `trading_app/lane_allocator.py` | NEW | ~200 |
| `scripts/tools/rebalance_lanes.py` | NEW | ~80 |
| `tests/test_trading_app/test_lane_allocator.py` | NEW | ~150 |

**Zero modifications to existing files.** The allocator is read-only against gold.db. Outputs recommendations only. Human applies to prop_profiles.py after review.

---

## Zero Look-Ahead Guarantee

All queries use `trading_day < rebalance_date`. Enforced by:
1. SQL WHERE clause (hard boundary)
2. Unit test: mock future trades, verify they're excluded
3. Audit trail logs the exact date boundary used

---

## Tests

1. `test_zero_lookahead` — Future trades excluded from trailing window
2. `test_pause_after_2_negative_months` — Strategy paused correctly
3. `test_resume_after_3_positive_months` — Strategy resumes correctly
4. `test_magnitude_override` — 3mo ExpR < -0.10 triggers immediate pause
5. `test_regime_gate_for_regime_class` — REGIME strategy inherits session regime
6. `test_respects_max_slots` — Profile constraints honored
7. `test_respects_allowed_sessions` — Session filtering works
8. `test_hysteresis_20pct` — Lane not replaced for <20% improvement
9. `test_annual_r_ranking` — High-N medium-ExpR beats low-N high-ExpR
10. `test_report_completeness` — Audit trail has all required fields

---

## Backtest Validation

After implementation, run the allocator monthly from 2022-01 to 2025-12:
- Compute portfolio equity curve (adaptive allocation)
- Compare to static allocation (fixed lanes, no rebalancing)
- Compare to oracle (perfect hindsight)
- Report: annual R, max DD, Sharpe, monthly turnover
- This is the PROOF that adaptive allocation adds value

---

## Integration with Existing Systems

- **strategy_fitness.py**: Consumed for FIT/WATCH/DECAY classification per strategy
- **prop_profiles.py**: Output target. Human reads allocation report → updates daily_lanes
- **execution_engine.py**: No changes. Reads daily_lanes from prop_profiles as before.
- **pipeline/check_drift.py**: Add check #85 verifying allocator imports from canonical sources only
