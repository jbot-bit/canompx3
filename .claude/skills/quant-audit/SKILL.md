---
name: quant-audit
description: Institutional hypothesis validator — falsify claims before accepting them. Fail-closed at every step. Runs T0-T8 test battery.
---
Institutional hypothesis validator — falsify research claims before accepting them: $ARGUMENTS

Use when: "audit this claim", "is this real", "stress test", "validate finding", "quant audit", "falsify", "test this hypothesis", "prop desk review", "audit:", "tautology check:", "wr monotonicity:", "sensitivity:"

## Process

Follow `.claude/rules/quant-audit-protocol.md` EXACTLY. 6 steps, 9 tests:

1. **PRE-FLIGHT** — DB freshness, row counts, schema. HALT if stale.
2. **CLAIM DECOMPOSITION** — Number, classify, state decision impact.
3. **FAILURE MODE ANALYSIS** — Score 4 risks per claim.
4. **TEST BATTERY** (in order, no skipping):
   - T0 Tautology (corr with existing filters — KILL if |r|>0.70)
   - T1 WR Monotonicity (WR flat = ARITHMETIC_ONLY, not signal)
   - T2 IS baseline
   - T3 OOS / Walk-forward (WFE > 0.50)
   - T4 Sensitivity ±20%
   - T5 Family (all sessions/instruments)
   - T6 Null floor (1000 bootstrap perms)
   - T7 Per-year (positive ≥7/10)
   - T8 Cross-instrument directional consistency
5. **DECISION RULES** — defined BEFORE results. VALIDATED / KILL / DOWNGRADE / SUSPEND.
6. **OUTPUT CONTRACT** — structured report only. No prose.

**Output the plan first. Do NOT run tests until user says "go" or "run it."**

## Critical Distinctions

- **ARITHMETIC_ONLY** = cost screen that improves payoff, NOT a win-rate predictor. Frame as "minimum viable trade size gate." Do NOT call "mechanism."
- **SIGNAL** = predicts win probability (WR changes across bins). May be genuine edge.
- **TAUTOLOGY** = mathematically equivalent to existing filter (e.g., cost/risk% = 1/orb_size)

## Hard Rules

- Canonical layers only (`bars_1m`, `daily_features`, `orb_outcomes`)
- NO conclusions before test results
- Lookahead BANNED: `double_break` (code line 393)
- Trade-time-knowable: `risk_dollars`, `break_delay_min`, `rel_vol`, `atr_20`
- All p-values adjusted for honest K
- Known failure patterns documented in protocol — check before re-testing dead claims
