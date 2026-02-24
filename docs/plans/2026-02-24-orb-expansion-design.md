# ORB Expansion Design — Extracting More From Existing Edge

**Date:** 2026-02-24
**Goal:** Four orthogonal approaches to increase profit from the validated ORB breakout system without finding new archetypes.

**Principle:** The ORB breakout edge is real (735 FDR-validated strategies). Rather than searching for new archetypes, extract more value from what's proven.

---

## Self-Audit Notes

- **15m/30m ORBs:** Code already parameterized. daily_features exists. Only orb_outcomes missing.
- **E4 re-entry:** Needs new entry model code. Must prove non-correlation with E0/E1.
- **Multi-session confirmation:** Needs new filter type. Risk: reduces N without proportional improvement.
- **Regime sizing:** Position sizing layer, not entry/exit. Most conservative expansion.
- **compute_overnight_stats "bug" at line 505:** NOT a bug. `_orb_utc_window(day, "1000", 5)[0]` only uses session start time, which is orb_minutes-independent.

---

## 1. Session Expansion: 15-Minute and 30-Minute ORBs

### Mechanism

Wider ORBs capture more of the session's initial price discovery. A 15m ORB includes the first 3 bars (vs 1 bar for 5m), giving a more established range. 30m captures the full opening rotation for slower instruments.

**Hypothesis:** Wider ORBs will have:
- Higher win rates (range is more "real" — less noise)
- Lower R-multiples (breakout has less distance to travel proportionally)
- Different session winners (sessions with false 5m breaks may have true 15m/30m breaks)
- Potentially different optimal entry models

### Existing Infrastructure

| Component | 15m Ready? | 30m Ready? |
|-----------|-----------|-----------|
| daily_features | MGC/MES/MNQ | MGC only |
| orb_outcomes | NOT BUILT | NOT BUILT |
| strategy_discovery | Parameterized | Parameterized |
| strategy_validator | Parameterized | Parameterized |
| entry_rules | No hardcoded assumptions | No hardcoded assumptions |

### Build Plan

**Phase 1: 15m ORBs (3 instruments)**
```bash
# Build outcomes
python trading_app/outcome_builder.py --instrument MGC --orb-minutes 15 --force --start 2016-02-01 --end 2026-02-04
python trading_app/outcome_builder.py --instrument MES --orb-minutes 15 --force --start 2019-02-12 --end 2026-02-11
python trading_app/outcome_builder.py --instrument MNQ --orb-minutes 15 --force --start 2024-02-05 --end 2026-02-03

# Discover strategies
python trading_app/strategy_discovery.py --instrument MGC --orb-minutes 15
python trading_app/strategy_discovery.py --instrument MES --orb-minutes 15
python trading_app/strategy_discovery.py --instrument MNQ --orb-minutes 15

# Validate
python trading_app/strategy_validator.py --instrument MGC --orb-minutes 15 --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
python trading_app/strategy_validator.py --instrument MES --orb-minutes 15 --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
python trading_app/strategy_validator.py --instrument MNQ --orb-minutes 15 --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward

# Edge families
python scripts/tools/build_edge_families.py --instrument MGC --orb-minutes 15
python scripts/tools/build_edge_families.py --instrument MES --orb-minutes 15
python scripts/tools/build_edge_families.py --instrument MNQ --orb-minutes 15
```

**Phase 2: 30m ORBs (MGC only)**
```bash
python trading_app/outcome_builder.py --instrument MGC --orb-minutes 30 --force --start 2016-02-01 --end 2026-02-04
python trading_app/strategy_discovery.py --instrument MGC --orb-minutes 30
python trading_app/strategy_validator.py --instrument MGC --orb-minutes 30 --min-sample 50 --no-regime-waivers --min-years-positive-pct 0.75 --no-walkforward
python scripts/tools/build_edge_families.py --instrument MGC --orb-minutes 30
```

### Expected Outcomes

- Discover strategies on sessions/instruments where 5m ORB has high false-break rate
- Wider ORBs may unlock 0030/2300 sessions that are currently weak at 5m
- 15m strategies may complement 5m strategies (different trade days = portfolio diversification)
- If 15m/30m show NO validated strategies, that's valuable signal too (confirms 5m is optimal)

### Success Criteria

- At least 1 FDR-validated strategy per instrument at 15m (otherwise STOP — wider ORBs don't work)
- Correlation analysis: 15m trade days vs 5m trade days (want <0.5 overlap for portfolio value)
- Combined Sharpe improvement when adding 15m/30m to existing 5m portfolio

---

## 2. ORB Re-Entry Model (E4)

### Mechanism

After a valid ORB break, price sometimes pulls back to the ORB edge before continuing in the break direction. This pullback shakes out weak hands and offers a lower-risk entry with tighter stop.

**Entry logic:**
1. ORB break occurs (establishes direction)
2. Wait for price to retrace to within X% of ORB edge (or touch the edge)
3. Enter in original break direction on the retrace
4. Stop: opposite ORB boundary (same as E1)
5. Target: same RR framework

### Key Design Questions

- **Retrace threshold:** Touch ORB edge exactly? Within 0.5 ATR? Within 25% of ORB range?
- **Time limit:** How long to wait for the retrace? 30 min? Session end?
- **Confirmation:** Require a confirming bar after the retrace? (Avoids catching a reversal)
- **Relationship to E0/E1:** E4 fires AFTER E0/E1 would have already entered. It's a different trade, not a replacement.

### Risk

- Overlap with E1 entries — if the pullback happens fast (within confirm bars), E1 and E4 could produce similar entries
- Reduces N significantly — not all breaks produce clean pullbacks
- Could be capturing failed-breakout reversals instead of continuation

### Implementation Needs

- New entry model function in `trading_app/entry_rules.py`
- New entry model enum value in config
- Outcome builder support for E4
- Research script to quantify: what % of breaks produce retraces, and at what depth?

### Status: NEEDS RESEARCH FIRST

Before building E4, run a research script to answer:
1. What % of validated 5m E1 trades had a retrace to within 25%/50%/75% of ORB edge?
2. Among those retraces, what was the hit rate vs the full E1 set?
3. Time distribution of retraces (how long to wait?)

---

## 3. Multi-Session Direction Confirmation

### Mechanism

If the 0900 session breaks long AND the 1000 session also breaks long, the directional conviction is stronger. Use prior session break direction as a filter for later sessions.

**Filter logic:**
- For 1000 session: check if 0900 broke in the same direction
- For 1800 session: check if 1000 broke in the same direction
- For 0030 session: check if prior US session broke in the same direction

### Key Design Questions

- **Which session pairs?** Not all pairs are logical (0900→1000 yes, but 0030→0900 crosses overnight)
- **Same vs opposite:** Confirmation (same direction) or contrarian (opposite direction)?
- **Lookback depth:** Only immediate prior session, or any session that day?
- **Interaction with existing filters:** How does this combine with G4+, FAST, CONT?

### Risk

- Reduces N substantially (maybe 40-60% of days have concordant sessions)
- Prior session direction already partially captured by `prev_day_direction` and gap analysis
- Cross-session concordance research (Feb 21) showed mixed results — some pairs work, some don't

### Implementation Needs

- New filter type in `trading_app/config.py` (CompositeFilter with session-pair logic)
- daily_features columns for prior-session break direction (may already exist in session stats)
- Research: which session pairs show statistically significant concordance benefit?

### Status: NEEDS RESEARCH FIRST

Previous cross-session concordance research showed limited signal. Re-examine with current FDR-validated strategy set.

---

## 4. Regime-Conditional Sizing

### Mechanism

Instead of binary in/out decisions (filters), adjust position size based on regime quality:
- High-ATR trending days: increase size
- Low-ATR compressed days: reduce or skip (already have ATR contraction AVOID)
- Strategy fitness FIT regime: full size; WATCH: half size; DECAY: quarter size

### Key Design Questions

- **Sizing model:** Kelly fraction? Fixed fractional? Volatility-targeting?
- **Regime inputs:** ATR percentile, rolling Sharpe, strategy fitness status
- **Rebalance frequency:** Daily? Per-trade? Monthly?
- **Portfolio interaction:** How does per-strategy sizing interact with the portfolio engine?

### Risk

- Over-optimization of sizing parameters creates a new curve-fitting dimension
- Regime detection lag — by the time you detect "high ATR", it may be ending
- Adds complexity without guaranteed improvement over simple fixed sizing

### Implementation Needs

- Sizing function in portfolio engine (or paper_trader)
- Fitness-based sizing weights (from get_strategy_fitness output)
- Research: does ATR percentile at entry time predict outcome quality?

### Status: LOWEST PRIORITY

This is a portfolio-level optimization. Do session expansion and E4 research first. Only add sizing complexity if the base strategies justify it.

---

## Execution Priority

| # | Idea | Status | Dependencies | Expected Timeline |
|---|------|--------|-------------|-------------------|
| 1 | 15m/30m ORBs | READY TO BUILD | None | Run now — outcomes + discovery + validation |
| 2 | E4 Re-Entry | NEEDS RESEARCH | Research script first | After session expansion complete |
| 3 | Multi-Session Confirmation | NEEDS RESEARCH | Research script first | After session expansion complete |
| 4 | Regime Sizing | LOWEST PRIORITY | Base strategies must exist | After E4 and multi-session research |

**Immediate action:** Build 15m outcomes for MGC, MES, MNQ. Then 30m for MGC. Run discovery and validation. Report results before starting E4/multi-session research.
