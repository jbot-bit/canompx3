# Pre-Registration: MNQ E2 RR1.0 Verified Sessions

**Date:** 2026-03-20
**Status:** PRE-REGISTERED (frozen before 2026 holdout test)
**Author:** Terminal audit, independently verified

## Strategies

### Verified (BH FDR at N=55 honest test count)

1. **MNQ NYSE_OPEN E2 CB1 RR1.0 O5**
   - All-sample: ExpR=+0.117R, N=1294, p=0.000012
   - Pre-2025: +0.108R (p=0.0004)
   - 2025 holdout: +0.144R (p=0.019)
   - Yearly: 6/6 positive (2021-2026)
   - BH FDR rank 1/55: p=0.000001 vs threshold 0.000909 → PASS

2. **MNQ COMEX_SETTLE E2 CB1 RR1.0 O5**
   - All-sample: ExpR=+0.121R, N=1261, p=0.000002
   - Pre-2025: +0.108R (p=0.0002)
   - 2025 holdout: +0.195R (p=0.001)
   - Yearly: 5/6 positive (2021-2025 positive, 2026 partial -0.009 on N=42)
   - BH FDR rank 2/55: p=0.00066 vs threshold 0.001818 → PASS

### Pre-Registered for 2026 Holdout (not yet tested)

3. **MNQ CME_PRECLOSE E2 CB1 RR1.0 O5**
   - All-sample: ExpR=+0.213R (pre-2025), +0.135R (2025 OOS)
   - Failed BH FDR at N=55 (p=0.007 vs threshold 0.003)
   - Pre-registered for SINGLE-TEST on 2026 holdout (p < 0.05)
   - Kill criterion: p > 0.05 on 2026 trades → dead

## 2026 Holdout Rules

- 2026 data (2026-01-01 onwards) is SACRED — no analysis, no "quick checks," no sanity testing
- These 3 strategies are the ONLY ones tested on 2026 data
- Test when N >= 100 trades per session (estimated: April 2026)
- BH FDR at N=3 (alpha=0.05, thresholds: 0.017, 0.033, 0.05)
- Results are FINAL — no re-testing, no parameter changes

## Parameters (FROZEN)

- Instrument: MNQ
- Entry: E2 (stop-market at ORB boundary)
- Confirm bars: 1
- RR target: 1.0
- ORB aperture: 5 minutes
- Filter: NONE (trade the baseline)
- Risk cap: skip ORBs > median+IQR per session (to fit prop DD limits)
- Cost model: $2.74 RT ($1.24 commission + $0.50 spread + $1.00 slippage)

## Honest Accounting

- 2025 holdout was contaminated: 55+ tests were run on it before these strategies were "pre-specified"
- The 2 verified strategies survive N=55 BH FDR — the honest test count
- CME_PRECLOSE failed at N=55 but is pre-registered for clean single-test on 2026
- This document is the verifiable audit trail (git-committed before any 2026 analysis)

## Statistical Bar for 2026

| Test count | Method | Threshold per test |
|---|---|---|
| N=3 (these strategies only) | BH FDR q=0.05 | 0.017 / 0.033 / 0.05 |
| N=1 (CME_PRECLOSE alone) | Raw | p < 0.05 |

## Kill Criteria for Paper Trading

- After 100 trades per session: actual ExpR < +0.03R → STOP that session
- After 100 trades: actual slippage > 3 ticks average → STOP (cost model wrong)
- After 200 trades total: if combined portfolio ExpR < +0.05R → STOP everything
- Sequential monitoring: O'Brien-Fleming boundaries checked monthly
