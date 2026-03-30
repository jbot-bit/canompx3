# Lane Routing Guide — Manual vs Auto Allocation

How to pick which strategies go where. Run `scripts/tools/score_lanes.py` first.

## Quick Reference

```bash
# Score all eligible MNQ strategies for TopStep
python scripts/tools/score_lanes.py --firm topstep

# Score only currently deployed lanes
python scripts/tools/score_lanes.py --current --firm apex

# Score MGC strategies
python scripts/tools/score_lanes.py --instrument MGC --firm topstep

# Filter to one session
python scripts/tools/score_lanes.py --session COMEX_SETTLE
```

## Composite Score (7 factors)

```
score = ExpR * sharpe_adj * ayp * n_confidence * fitness * rr_adj * prop_sm
```

| Factor | Formula | Rationale |
|--------|---------|-----------|
| ExpR | Raw expectancy_r | Edge strength |
| sharpe_adj | min(sharpe_ann / 1.5, 2.0) | Risk-adjusted return, 1.5 baseline |
| ayp | 1.2 if all_years_positive | Year-over-year consistency bonus |
| n_confidence | min(1.0, N / 300) | Penalize small samples |
| fitness | ROBUST=1.0, WHITELISTED=0.9, other=0.7 | Family robustness |
| rr_adj | RR1.0=1.0, RR1.5-2.0=0.95, RR2.5+=0.85 | Simpler exit = higher weight |
| prop_sm | Firm profit split (last tier) | Long-term payout rate |

### Switching threshold: 20%

Only swap a deployed lane if new candidate scores > current lane * 1.20.
Prevents churn from marginal improvements. Source: Carver Ch 12.

## Routing Decision Tree

### Step 1: Firm constraints (hard gates)

| Firm | Auto OK? | API | Instrument bans |
|------|----------|-----|-----------------|
| Apex | NO | N/A | Metals suspended |
| TopStep | YES | ProjectX | None |
| Tradeify | YES | Tradovate (broken Mar 2026) | None |
| Self-funded | YES | IBKR (not built) | None |

### Step 2: Auto eligibility (all must pass)

- Family: ROBUST or WHITELISTED
- FDR adjusted p-value < 0.05
- 2025 forward R > 0
- Execution risk < 5% of firm DD limit

### Step 3: Session routing (Brisbane time)

| Time Block | Brisbane Hours | Slot | Why |
|------------|---------------|------|-----|
| Sleeping | 22:00 - 06:00 | auto-preferred | Can't manually cover |
| Early morning | 06:00 - 10:00 | either | Watchable if awake |
| Daytime | 10:00 - 22:00 | manual-only | Awake, not US market hours |

**Session times (DST-dependent, summer/winter):**

| Session | Summer | Winter | Slot |
|---------|--------|--------|------|
| CME_REOPEN | 08:00 | 09:00 | either |
| TOKYO_OPEN | 10:00 | 10:00 | manual |
| SINGAPORE_OPEN | 11:00 | 11:00 | manual |
| LONDON_METALS | 17:00 | 18:00 | manual |
| EUROPE_FLOW | 18:00 | 17:00 | manual |
| US_DATA_830 | 22:30 | 23:30 | auto |
| NYSE_OPEN | 23:30 | 00:30 | auto |
| US_DATA_1000 | 00:00 | 01:00 | auto |
| COMEX_SETTLE | 03:30 | 04:30 | auto |
| CME_PRECLOSE | 05:45 | 06:45 | auto/either |
| NYSE_CLOSE | 06:00 | 07:00 | either |
| BRISBANE_1025 | 10:25 | 10:25 | manual |

### Step 4: Cross-firm filter diversity

Same session on 2+ firms → MUST use different filters.

| Session | Apex (manual) | Tradeify (auto) | TopStep (auto) |
|---------|--------------|-----------------|----------------|
| CME_PRECLOSE | VOL_RV20_N20 | ATR70_VOL | TBD (different family) |
| NYSE_CLOSE | VOL_RV20_N20 | VOL_RV20_N20 | — |
| COMEX_SETTLE | ATR70_VOL | ATR70_VOL | ATR70_VOL (same OK — different firm, same edge) |
| US_DATA_1000 | X_MES_ATR60 | X_MES_ATR60 | — |
| TOKYO_OPEN | VOL_RV30_N20 | VOL_RV30_N20 | — |

Note: Same filter across firms is OK when the edge source is the same trade.
Different filters on the same firm add regime diversification.

### Step 5: Marginal value ranking

For the FIRST auto lane, pick the session that adds the most portfolio value:
1. **Sessions you CAN'T manually cover** (sleeping hours) > sessions you already trade
2. **Higher frequency** (more trades/yr) > fewer trades (faster loop proof)
3. **Highest score among auto-eligible** after the above filters

This is why COMEX_SETTLE (03:30 AM, 50 trades/yr) was chosen for TopStep auto
over CME_PRECLOSE (05:45 AM, 30 trades/yr) despite CME_PRECLOSE scoring 2x higher —
the marginal portfolio value of a bot-only session exceeds the raw score advantage
of duplicating a session you already manually trade.

## Kill Criteria (per lane)

| Condition | Action |
|-----------|--------|
| 3 consecutive losing months | Pause lane, investigate |
| Forward ExpR < 0.10 over 50+ trades | Remove lane |
| CUSUM 4-sigma alarm sustained 2+ weeks | Remove lane |
| DD within 20% of firm limit | Downgrade to signal-only |
| Prop firm rule violation | FULL STOP all lanes |
| Dashboard/broker P&L mismatch | FULL STOP, investigate |

## Paper-to-Live Gateway

| Stage | Duration | Criteria to advance |
|-------|----------|---------------------|
| Signal-only | 2-3 sessions | Dashboard renders correctly, signals match manual calc |
| Demo (paper orders) | 5-10 sessions | Orders execute, fills match, no orphans |
| Live (1 contract) | 30 trades | >95% execution fidelity, slippage within E2 model |
| Scale (5 accounts) | 50+ trades | Consistent positive forward ExpR, no CUSUM alarms |

## Current Allocation (2026-03-31)

| Profile | Firm | Instrument | Sessions | Status | Mode |
|---------|------|------------|----------|--------|------|
| apex_100k_manual | Apex | MNQ | CME_PRE, NYSE_CL, COMEX, US_DATA, TOKYO | Active | Manual |
| topstep_50k | TopStep | MGC | TOKYO_OPEN | Active | Shadow/Conditional |
| topstep_50k_mnq_auto | TopStep | MNQ | COMEX_SETTLE | Active | Auto (new) |
| tradeify_50k | Tradeify | MNQ | All 5 sessions | Inactive | API broken |
