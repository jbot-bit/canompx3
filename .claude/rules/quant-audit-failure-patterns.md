# Quant Audit — Known Failure Patterns

**Companion to `.claude/rules/quant-audit-protocol.md`.** No frontmatter → never auto-injects.

Split out 2026-04-20 to stop the failure-pattern block from re-injecting on every edit to `trading_app/strategy_*`, `trading_app/outcome_*`, `research/**`, `scripts/tools/backtest*`.

---

## KNOWN FAILURE PATTERNS

Confirmed findings from this project. Append new entries as discovered.

```
[2026-03-24] cost/risk% friction gate:
  VERDICT: ARITHMETIC_ONLY
  FINDING: WR flat at ~58-60% across all friction bins (quintile spread <3%).
           ExpR improvement is payoff arithmetic — bigger ORBs pay more per win
           because costs eat less of the gross win. Not a win-rate predictor.
           |corr(cost_risk_pct, 1/orb_size_pts)| = 1.0 by construction (inverse).
  CORRECT FRAMING: Minimum viable trade size gate / cost screen (same family as G-filters)
  DO NOT CALL: "breakout quality predictor" or "structural mechanism"
  IMPLEMENT AS: ORB risk$ threshold filter. Friction < 10% = risk$ > $27.40 for MNQ.

[2026-03-24] break_delay_min timeout:
  VERDICT: SIGNAL (pending full T3-T8 verification)
  FINDING: WR spread 54-64% across delay quintiles (6%+ spread).
           Early breaks (<=5m) have higher WR AND higher ExpR.
           WR improvement persists when controlling for friction.
  MECHANISM: Order flow concentration — immediate breaks = stacked stops triggering
  STATUS: Passes T1 (WR monotonic). T3-T8 pending DB unlock for verification.

[2026-03-24] double_break:
  VERDICT: BANNED — LOOKAHEAD
  FINDING: Massive signal (+0.352 vs -0.198) but computed AFTER trade entry.
           Code line 393: "LOOK-AHEAD relative to intraday entry — do NOT use as live filter."
  STATUS: Cannot be used. Correctly flagged in pipeline code.
```


