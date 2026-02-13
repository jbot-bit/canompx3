# TRADING PLAN

Generated: 2026-02-13
Data: 2016-02-01 to 2026-02-04 | Cost model: MGC ($10/pt, $8.40 RT)
Rolling window: 12m | Classification threshold: Sharpe >= 0.1

## Session Logic (LOCKED)

| Session | Logic | Status | Rationale |
|---------|-------|--------|-----------|
| 0900 | Fixed Target | STABLE | 6/6 recent windows pass |
| 1000 | Target Unlock | TRANSITIONING | 3/6 recent windows pass |
| 1100 | OFF | DEAD | 74% double-break, IB/ORB tautology |

## Position Sizing Rules

- **0900 (Fixed Target)**: Normal (1.0x risk)
  - Classification: STABLE
  - Last 3 window Sharpes: 0.168, 0.159, 0.176
  - Full period: N=148 WR=45.3% ExpR=+0.229 Sharpe=0.169
- **1000 (Target Unlock)**: Quarter (0.25x risk)
  - Classification: TRANSITIONING
  - Last 3 window Sharpes: -0.011, 0.057, 0.068
  - Full period: N=138 WR=35.5% ExpR=+0.115 Sharpe=0.070

## Pyramiding

**OFF** -- destroyed value at both sessions. Intraday mean-reversion snap-back
kills the second unit. Do not revisit.

## 1000 Target Unlock Rules

1. Entry: E1 CB2 G4+ (standard ORB break)
2. Limbo phase: Fixed target (2.0R) + stop active
3. IB breaks ALIGNED: cancel target, hold 7h with stop only
4. IB breaks OPPOSED: exit at market immediately
5. No IB break within 7h: fixed target stays active
6. IB definition: market open (0900 Brisbane) + 120min

## Rolling Re-evaluation

- Run monthly: `python scripts/rolling_portfolio_assembly.py --db-path C:/db/gold.db`
- STABLE -> TRANSITIONING: reduce size by 50%
- TRANSITIONING -> DEGRADED: turn OFF
- DEGRADED -> STABLE: requires 3 consecutive passing windows before re-entry
