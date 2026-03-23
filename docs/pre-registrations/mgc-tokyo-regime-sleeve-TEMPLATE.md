# PRE-REGISTRATION: MGC TOKYO_OPEN REGIME SLEEVE (TEMPLATE)

**Status:** TEMPLATE — NOT ACTIVE. Activate only after shadow period + noise floor recalibration.
**Created:** 2026-03-23
**Source audit:** docs/audits/2026-03-23-mgc-regime-truth-audit.md

---

## Frozen Scope

| Field | Value |
|-------|-------|
| Instrument | MGC |
| Session | TOKYO_OPEN |
| Filter | ORB_G4 (min ORB size >= 4.0) |
| Entry model | E2 (stop-market, 1 tick slippage) |
| Aperture | 5m ORB |
| RR target | 1.5 |
| Stop | S075 (0.75x tight stop) |
| Direction | Bidirectional |
| Source | Baseline (validated_setups, not rolling) |

## Phase Gates

| Phase | Duration | Capital | Entry criteria |
|-------|----------|---------|---------------|
| SHADOW | 60+ days, 15+ trades | 0 contracts | Audit approval |
| MICRO-LIVE | 12+ months | 1 contract max | Shadow positive + recalibrated noise floor |
| CORE-ELIGIBLE | TBD | Standard sizing | 30+ forward trades above recalibrated floor + no kill-switch |

## Baseline (frozen at registration)

| Metric | Value |
|--------|-------|
| Backtest N | 177 |
| Backtest WR | 48.0% |
| Backtest ExpR | +0.235 |
| OOS ExpR | 0.186 |
| WFE | 0.833 |
| FDR adj_p | 0.008 |
| Noise floor | 0.21 (KNOWN CALIBRATION CANDIDATE DEFECT: sigma 2.79x trimmed overshoot, practical impact uncertain) |

## Kill-Switches (automatic, non-discretionary)

1. Cumulative R < -5.0
2. ATR percentile < 40
3. Win rate < 35% after 20+ trades
4. < 10 trades in 6 months
5. Median observed slippage > 3 ticks
6. **Permanent disable:** 2 different triggers within 90 days

## Rules

- Review cadence: monthly (1st of month)
- Min trades before ANY inference: 15
- Discretionary overrides: FORBIDDEN
- Reporting language: "MGC regime sleeve (probation). PRELIMINARY class. Conditional. Not standalone. Not CORE."
- Cannot be retroactively reclassified as CORE
- Does not affect LIVE_PORTFOLIO strategy counts or MNQ/MES reporting
- Thresholds cannot move after activation date

## Activation Checklist (all required)

- [ ] Noise floor recalibrated via block bootstrap (all instruments)
- [ ] Shadow period completed (60+ days, 15+ trades)
- [ ] Shadow ExpR documented
- [ ] Pre-registration frozen date set
- [ ] HANDOFF.md updated
- [ ] This file status changed from TEMPLATE to ACTIVE
