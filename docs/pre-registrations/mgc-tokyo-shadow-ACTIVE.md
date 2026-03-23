# PRE-REGISTRATION: MGC TOKYO_OPEN SHADOW TRACKER

**Status:** ACTIVE — Shadow tracking only. NOT LIVE. NOT CORE. Zero capital.
**Activated:** 2026-03-24
**Forward-only start:** First qualifying day on or after 2026-03-21 (data through 2026-03-20 is backtest, not shadow)
**Source audit:** docs/audits/2026-03-23-mgc-regime-truth-audit.md
**Strategy ID:** MGC_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G4_S075

---

## Frozen Scope (IMMUTABLE after activation)

| Field | Value |
|-------|-------|
| Instrument | MGC |
| Session | TOKYO_OPEN |
| Filter | ORB_G4 (min ORB size >= 4.0 points) |
| Entry model | E2 (stop-market, 1 tick slippage) |
| Aperture | 5m ORB |
| RR target | 1.5 |
| Confirm bars | 1 |
| Stop | S075 (0.75x tight stop) |
| Direction | Bidirectional |
| Source | Baseline (validated_setups, not rolling) |

## Baseline (frozen at registration, verified from raw 2026-03-24)

| Metric | Value | Source |
|--------|-------|--------|
| Backtest N | 177 | validated_setups.sample_size |
| Backtest WR | 48.0% | validated_setups.win_rate = 0.4802 |
| Backtest ExpR | +0.2347 | validated_setups.expectancy_r |
| OOS ExpR | 0.1858 | validated_setups.oos_exp_r |
| WFE | 0.8328 | validated_setups.wfe |
| FDR adj_p | 0.00794 | validated_setups.fdr_adjusted_p (K=105,640) |
| Sharpe | 0.2281 | validated_setups.sharpe_ratio |
| noise_risk | True | OOS ExpR 0.1858 <= floor 0.21 |
| Noise floor (MGC E2) | 0.21 | NOISE_FLOOR_BY_INSTRUMENT, P95 of 100-seed Gaussian null |

## Structural Context (disclosed, not excuses)

| Factor | Value | Implication |
|--------|-------|------------|
| Double-break rate | 83.5% (2,182/2,613 days) | TOKYO_OPEN is structurally choppy for MGC |
| Regime dependency | 84% of trades in 2025-2026 | Edge requires gold > $2,400 for G4+ qualifying days |
| Qualifying frequency | 0-6% (2016-2023), 38% (2025), 95% (2026) | Trade count is regime-conditional |
| RR lock tension | family_rr_locks picks RR1.0 (MAX_SHARPE); this template uses RR1.5 (edge family head, higher ExpR) | Deliberate divergence: shadow-track the higher-ExpR variant |
| Noise floor calibration | Gaussian null sigma=1.2, real trimmed std=0.43 (2.82x). But trade-weighted match ~1.03x in 2025 regime. Sigma-invariant for R-multiples (proven). | Candidate defect — does NOT justify lowering the floor without block bootstrap |
| Classification | PRELIMINARY (N=177, RESEARCH_RULES.md 100-199 tier) | NOT CORE by methodology authority |

## Phase Gates

| Phase | Duration | Capital | Entry criteria |
|-------|----------|---------|---------------|
| **SHADOW (current)** | 60+ days, 15+ trades | **0 contracts** | This document activated |
| MICRO-LIVE | 12+ months | 1 contract max | Shadow positive + noise floor recalibrated via bootstrap |
| CORE-ELIGIBLE | TBD | Standard sizing | 30+ forward trades above recalibrated floor + no kill-switch |

## Kill-Switches (automatic, non-discretionary)

1. Cumulative R < -5.0
2. ATR percentile < 40 (sustained — regime died)
3. Win rate < 35% after 20+ trades
4. < 10 trades in 6 months (low frequency — regime shifted)
5. Median observed slippage > 3 ticks (cost model invalid)
6. **Permanent disable:** 2 different triggers within 90 days

## Rules (IMMUTABLE)

- Review cadence: monthly (1st of month)
- Min trades before ANY inference: 15
- Discretionary overrides: FORBIDDEN
- Cannot be retroactively reclassified as CORE
- Does not affect LIVE_PORTFOLIO strategy counts or MNQ/MES reporting
- Thresholds cannot move after activation date
- Forward-only: no retroactive inclusion of pre-2026-03-21 data
- Shadow results are NOT evidence for live promotion without noise floor recalibration

## Reporting Language (mandatory for all references)

> **MGC TOKYO_OPEN shadow tracker (OBSERVATIONAL).** Not live. Not CORE. Not approved for capital.
> Forward evidence collection only. PRELIMINARY class (N=177). noise_risk=True (OOS ExpR below floor).
> No inference permitted until 15+ forward trades collected. Any positive results are provisional
> and do not constitute evidence for promotion without noise floor recalibration and independent review.

## Shadow Log

All forward signals recorded in: `docs/pre-registrations/shadow-logs/mgc-tokyo-shadow.csv`

## Promotion Checklist (ALL required before any capital)

- [ ] Noise floor recalibrated via block bootstrap (ALL instruments, not MGC-only)
- [ ] Shadow period completed (60+ days, 15+ forward trades)
- [ ] Shadow ExpR > recalibrated noise floor
- [ ] No kill-switch triggered during shadow period
- [ ] Independent review documented
- [ ] HANDOFF.md updated with promotion decision
