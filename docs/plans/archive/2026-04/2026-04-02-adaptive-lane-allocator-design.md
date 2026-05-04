---
status: archived
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
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

### Self-sufficient strategies (trailing_n >= 20 in window)
- Use OWN trailing ExpR for regime detection
- Enough trades in the trailing window for a robust signal
- Applies regardless of CORE/REGIME total-N classification

### Thin-data strategies (trailing_n < 20 in window)
- Use SESSION-LEVEL trailing ExpR for regime gate
- Session regime = unfiltered E2 RR1.0 CB1 O5 for the instrument × session
- Session unfiltered has 200-500 trades/year = robust signal even in short windows
- If session is HOT (trailing 6mo ExpR > 0) → deploy the strategy
- If session is COLD (trailing 6mo ExpR < 0) → pause regardless of own sparse performance
- Grounding: Carver Ch.11 "forecast blending" — use higher-level signal for lower-level bet
- NOTE: A CORE strategy (N=200 total) with a strict filter may have trailing_n < 20.
  Total N does not determine regime gate — trailing N does.

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
- Compute ExpR for each of the last 2 calendar months INDIVIDUALLY
- If BOTH individual months are negative → PAUSE
- OR: trailing 3-month average ExpR < -0.10 → IMMEDIATE PAUSE (magnitude override)
- Thin-data strategies (trailing_n < 20): session-level 2-month check instead

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

The allocator respects per-profile constraints. These are FIRM RULES, not
trading preferences. The allocator selects the best lanes within these walls.

- `max_slots`: maximum lanes per account (firm contract limits / cognitive cap)
- `allowed_instruments`: FIRM-LEVEL restriction only. Set per firm's actual rules:
    - Apex: `frozenset({"MNQ", "MES"})` — metals BANNED
    - TopStep: `frozenset({"MNQ", "MGC"})` — allows gold
    - Tradeify: `frozenset({"MNQ", "MES", "MGC"})` — allows all
    - Self-funded IBKR: `None` (no restrictions)
  Do NOT hardcode instrument preferences here. The allocator picks instruments.
- `allowed_sessions`: `None` for auto profiles (allocator selects sessions).
  Only set for manual practice profiles (Brisbane daytime restriction).
  Do NOT hardcode session lists for auto accounts — that defeats the allocator.
- `stop_multiplier`: prop firm stop sizing (0.75x for prop, 1.0x for self-funded)
- `max_dd`: firm's drawdown limit. Allocator sizes lanes to fit within budget.

### Migration Required (prop_profiles.py)
Current active profiles have hardcoded `allowed_sessions` and `daily_lanes`.
After allocator is built:
1. Auto profiles: set `allowed_sessions = None`, `daily_lanes = ()` (empty)
2. Allocator fills daily_lanes monthly based on what's validated and hot
3. Manual profile: keep `allowed_sessions` = Brisbane daytime set
4. Fix Apex `allowed_instruments`: currently None (wrong) → should be `{"MNQ", "MES"}`

AUTO is the primary deployment via bot. The allocator selects which
sessions to trade — only validated, currently HOT, and DD-budget-compliant
lanes make the cut. Auto profiles set `allowed_sessions = None` (allocator
decides) with constraints: max_slots, max_dd, allowed_instruments.

MANUAL is a separate practice account for learning. Same allocator output
but filtered to Brisbane daytime sessions only (allowed_sessions set).

The allocator IS the session selector. Profiles provide constraints, not
session lists. A session that goes cold gets dropped. A session that gets
hot gets added. No human decision needed for which sessions — only for
approving the allocator's monthly recommendation.

**SM-aware ranking:** Trailing ExpR is computed at the ACCOUNT's stop_multiplier.
The same strategy may rank differently for SM=0.75 (prop) vs SM=1.0 (self-funded).

**Provisional strategies:** Strategies with < 6 months of trailing data get status=PROVISIONAL.
They can be deployed but rank BELOW strategies with 6+ months data when annual_r is comparable.
Prevents chasing short hot streaks from recently validated strategies.

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
        {"strategy_id": "MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G6", "annual_r": 18.0, "status": "DEPLOY",
         "trailing_expr": 0.1009, "trailing_n": 178, "trailing_wr": 0.481,
         "months_negative": 0, "session_regime": "HOT"},
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

## DD Budget Algorithm

Lane selection respects the firm's DD limit. Greedy selection (not knapsack):

```
candidates = sorted(all_lane_scores, by=annual_r, descending)
selected = []
dd_used = 0.0

for lane in candidates:
    if len(selected) >= max_slots:
        break
    if profile.allowed_instruments and lane.instrument not in profile.allowed_instruments:
        continue
    if profile.allowed_sessions and lane.orb_label not in profile.allowed_sessions:
        continue
    
    # Worst-case DD contribution = max_orb_pts × SM × point_value
    lane_dd = lane.max_orb_pts * profile.stop_multiplier * cost_spec.point_value
    if dd_used + lane_dd > profile.max_dd:
        continue  # skip, try next (smaller ORB might fit)
    
    selected.append(lane)
    dd_used += lane_dd
```

This is simple, deterministic, and respects all constraints.

---

## Stateless Design

The allocator computes ALL state from orb_outcomes data. No dependency on prior
lane_allocation.json for months_negative or pause history.

```
# Compute months_negative from per-month ExpR series:
for each of the last 6 calendar months:
    compute month_expr = AVG(adjusted_pnl_r) for that month
months_negative = count consecutive negative months from most recent backward

# Compute months_positive_since_pause:
# Walk backward from most recent month. Find the last negative streak of 2+.
# Count positive months after that streak ended.
# If no negative streak exists → months_positive_since_pause = N/A (never paused)
```

This makes every run reproducible from the same data. No hidden state.

**First run:** No prior allocation file → skip hysteresis check. Select top N
lanes purely by annual_r. Subsequent runs read the prior file for hysteresis only.

---

## Files

| File | Action | Lines |
|------|--------|-------|
| `trading_app/lane_allocator.py` | NEW | ~250 |
| `scripts/tools/rebalance_lanes.py` | NEW | ~80 |
| `tests/test_trading_app/test_lane_allocator.py` | NEW | ~200 |
| `trading_app/prop_profiles.py` | MODIFY | Migration: allowed_sessions=None for auto, fix Apex instruments |
| `trading_app/paper_trade_logger.py` | MODIFY | Derive LANES from prop_profiles (eliminate hardcoded constant) |
| `trading_app/pre_session_check.py` | MODIFY | Add allocation staleness check |

The allocator is read-only against gold.db. Outputs recommendations to
lane_allocation.json. Human applies to prop_profiles.py after review.

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
11. `test_sm_adjustment` — SM=0.75 trailing ExpR differs from SM=1.0 for same strategy
12. `test_filter_applied_in_trailing` — Filtered trailing ExpR differs from unfiltered
13. `test_provisional_status` — Strategy with <6 months data gets PROVISIONAL
14. `test_staleness_warning` — Allocation >35 days old triggers warning
15. `test_staleness_block` — Allocation >60 days old blocks trading
16. `test_individual_month_negative` — Both individual months negative → PAUSE (not average)

---

## Backtest Validation

After implementation, run the allocator monthly from 2022-01 to 2025-12:
- Compute portfolio equity curve (adaptive allocation)
- Compare to static allocation (fixed lanes, no rebalancing)
- Compare to oracle (perfect hindsight)
- Report: annual R, max DD, Sharpe, monthly turnover
- This is the PROOF that adaptive allocation adds value

---

## Critical Implementation Details (Gap Analysis)

### SM=0.75 Adjustment in Trailing Window
orb_outcomes stores SM=1.0 base outcomes. For SM=0.75 strategies, the allocator
MUST apply tight-stop adjustment per trade using mae_r:
```
if stop_multiplier != 1.0 and trade.mae_r >= stop_multiplier:
    adjusted_pnl_r = -stop_multiplier
else:
    adjusted_pnl_r = trade.pnl_r
```
Without this, SM=0.75 trailing ExpR is computed at SM=1.0 — WRONG.

### Filter Application in Trailing Window
The allocator MUST join daily_features and apply ALL_FILTERS[filter_type].matches_row()
to determine eligible days. Unfiltered trailing ExpR may be negative while filtered is
positive — the filter IS the edge. Same logic as strategy_discovery filter application.

### Minimum Trades Threshold
Require trailing_n >= 20 to compute a LaneScore. Below 20 → status=STALE,
fall back to session-level regime gate. Prevents ranking on noise.

### Parameter Source: Literature, NOT Backtest
Window sizes (12mo deploy, 6mo regime), kill thresholds (2mo negative),
resume thresholds (3mo positive), hysteresis (20%) are ALL from literature:
- 12mo: Carver Ch.11 default forecast window
- 2mo kill: Chan Ch.7 regime half-life 1-3 months
- 3mo resume: Carver Ch.12 asymmetric switching
- 20% hysteresis: Carver Ch.12 switching cost threshold

Do NOT adjust these parameters based on the allocator backtest.
The backtest is VALIDATION, not optimization. Tuning parameters on
the backtest is overfitting the allocator itself.

### Staleness Enforcement
lane_allocation.json includes rebalance_date. Pre-session check reads it:
- > 35 days → WARNING (log, continue trading)
- > 60 days → BLOCK (refuse to trade until rebalance runs)

### Eliminate Hardcoded Lanes (paper_trade_logger.py)
paper_trade_logger.py currently has its OWN hardcoded LANES constant that
must be manually synced with prop_profiles. This is a drift vector.

FIX: paper_trade_logger derives its lanes FROM prop_profiles at runtime.
Uses `matches_row()` (already exists on all filters) instead of filter_sql.
No new `to_sql()` method needed — matches_row is the canonical filter interface.

```python
from trading_app.prop_profiles import ACCOUNT_PROFILES
from trading_app.config import ALL_FILTERS
from trading_app.strategy_discovery import parse_strategy_id

def _build_lanes_from_profile(profile_id: str) -> tuple[LaneDef, ...]:
    profile = ACCOUNT_PROFILES[profile_id]
    lanes = []
    for spec in profile.daily_lanes:
        params = parse_strategy_id(spec.strategy_id)
        lanes.append(LaneDef(
            strategy_id=spec.strategy_id,
            instrument=spec.instrument,
            orb_label=spec.orb_label,
            filter_type=params["filter_type"],
            # No filter_sql — use matches_row() at runtime instead
        ))
    return tuple(lanes)
```
One source of truth. Zero hardcoded lanes. Zero drift possible.

---

## Integration with Existing Systems

- **strategy_fitness.py**: Consumed for FIT/WATCH/DECAY classification per strategy
- **prop_profiles.py**: Output target. Human reads allocation report → updates daily_lanes
- **execution_engine.py**: No changes. Reads daily_lanes from prop_profiles as before.
- **pre_session_check.py**: Add allocation staleness check (>35d warn, >60d block)
- **paper_trade_logger.py**: Add runtime lane sync verification on import
- **pipeline/check_drift.py**: Add check #85 verifying allocator imports from canonical sources only
