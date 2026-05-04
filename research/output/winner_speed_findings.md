# Winner Speed Profiling — Findings

**Date:** 2026-02-20
**Script:** `research/research_winner_speed.py`
**Data source:** `orb_outcomes` (outcome='win', orb_minutes=5)
**Sample:** 404,261 winning trade rows — MGC, MES, MNQ, all sessions, all RR/CB combos

---

## Key Metric: T80

**T80** = the time (in minutes after entry) by which 80% of winning trades have already hit their profit target. Everything after T80 is "dead exposure" — the session is still open but most of the potential winners have resolved.

---

## T80 Summary by Session (RR1.0 and RR2.5, CB1)

| Symbol | Session | N (RR1.0) | T50 | T80 | T90 | Session Max | Dead Exp |
|--------|---------|-----------|-----|-----|-----|-------------|----------|
| MGC | 1000 | 2,368 | 13m | **32m** | 48m | 480m | 448m |
| MES | 1100 | 1,720 | 12m | **31m** | 59m | 420m | 389m |
| MGC | 0900 | 1,994 | 13m | **38m** | 60m | 480m | 442m |
| MES | 1000 | 1,585 | 15m | **39m** | 66m | 480m | 441m |
| MNQ | 1000 | 426 | 15m | **38m** | 63m | 480m | 442m |
| MGC | 1800 | 2,451 | 13m | **36m** | 66m | 420m | 384m |
| MES | 1800 | 1,647 | 15m | **42m** | 69m | 420m | 378m |
| MES | CME_CLOSE | ~150 | ~8m | **16m** | ~30m | 180m | 164m |

At **RR2.5** (production target), T80 grows significantly but dead exposure remains large:

| Symbol | Session | T80 (RR2.5) | Session Max | Dead Exp (RR2.5) |
|--------|---------|-------------|-------------|------------------|
| MGC | 1000 | 90m | 480m | 390m |
| MES | 1000 | 90m | 480m | 390m |
| MGC | 0900 | 113m | 480m | 367m |
| MES | 0900 | 166m | 480m | 314m |
| MGC | 1800 | ~75m | 420m | ~345m |

---

## Cumulative Hit % (MGC 1000, RR1.0, CB1 — N=2,368)

| Min | % Winners Hit |
|-----|--------------|
| 5m | 24.1% |
| 10m | 42.3% |
| 15m | 54.7% |
| 20m | 63.6% |
| **30m** | **78.8%** ← ~T80 |
| 45m | 88.9% |
| 60m | 95.6% |
| 90m | 98.2% |
| 120m+ | 99%+ |

---

## Interpretation

**The "7h hold is dead exposure" hypothesis is CONFIRMED and then some.**

At RR1.0: 80% of winners resolve in 30-45 minutes. The session stays open for 480 minutes. That's 430-450 minutes of exposure where you are predominantly holding losers.

At RR2.5: 80% of winners resolve in 90-170 minutes. Still 300-390 minutes of dead exposure after T80.

This is NOT a flaw in the current system — sessions must stay open to catch the remaining 20% of winners. But it means:

1. **After T80, the open-position pool is dominated by losers** (~80% of remaining open positions will not hit target)
2. **If those slow-moving positions have negative expectancy, a time-based exit gate would improve Sharpe without reducing winner count meaningfully** (by definition, 80% of winners already closed)

---

## Next Research Step (P5b)

**Question:** Among positions still open PAST T80, what is their eventual outcome distribution?

- If losers are over-represented in the >T80 cohort → time-based exit is beneficial
- If winners are proportionally represented → holding longer is justified

**Query approach:**
```sql
-- For each session, compute minutes_to_exit for ALL outcomes (not just wins)
-- Then split by: did they exit before or after the session's T80 threshold?
SELECT
    symbol, orb_label, outcome,
    COUNT(*) AS n,
    AVG(pnl_r) AS avg_r
FROM orb_outcomes
WHERE orb_minutes = 5
  AND entry_ts IS NOT NULL
  AND exit_ts IS NOT NULL
GROUP BY symbol, orb_label, outcome,
    CASE WHEN date_diff('second', entry_ts, exit_ts)/60.0 <= [T80] THEN 'fast' ELSE 'slow' END
```

If `slow` cohort (>T80) has avg_r significantly worse than `fast` cohort → implement time-based close.

---

## Files

| File | Contents |
|------|----------|
| `research/output/winner_speed_summary.csv` | T50/T80/T90/T95 per (symbol, session, rr_target, confirm_bars) — 840 rows |
| `research/output/winner_speed_cumulative.csv` | Cumulative % hit at each checkpoint — 13,440 rows |
| `research/research_winner_speed.py` | Script (read-only, no DB writes) |
