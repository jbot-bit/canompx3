# Design: MGC/MNQ Correlation & Market Context Research

**Date:** 2026-03-12
**Status:** Research complete
**Blast radius:** 1 new file (research script) + 1 modified (generate_trade_sheet.py, deferred)
**Guardian prompts:** Not required (no schema/entry model/pipeline changes)

## Motivation

User question: "Does long MGC + short MNQ give natural hedge from inverse safe-haven/risk-on correlation?"
Prior research: Lead-lag MGC->MNQ tested Feb 2026 -- NO SIGNAL (p=0.189-0.936).
What was NOT tested: daily returns correlation, portfolio P&L variance reduction, session-level
direction concordance under current session names.

## Three Research Blocks

### Block 1: Daily Returns Correlation
- Compute daily close-to-close % returns from daily_features.daily_close
- Test: Pearson r significantly different from 0?
- Rolling 60-day window: is correlation stable or regime-specific?
- Year-by-year breakdown: 2021-2026
- ATR regime split (high vs low volatility)

### Block 2: Session Direction Concordance
- Shared sessions: TOKYO_OPEN, SINGAPORE_OPEN, LONDON_METALS, NYSE_OPEN,
  US_DATA_830, US_DATA_1000, CME_PRECLOSE
- For each session: P(MNQ direction = OPPOSITE of MGC direction | both break)
- Binomial test vs base rate (50%), BH FDR at q=0.10

### Block 3: Portfolio P&L Correlation
- orb_outcomes pnl_r for E2, orb_minutes=5
- Pearson correlation of same-day pnl_r across instruments
- Co-loss and co-win rates, combined portfolio Sharpe

## RESULTS (2026-03-12)

### Block 1: r = +0.10 (positive, NOT inverse). p = 0.0001.
Year-by-year: 2021 r=+0.197**, 2022 r=+0.134*, 2023 r=+0.021 ns, 2024 r=+0.224**, 2025 r=+0.016 ns.
Rolling 60-day: swings -0.56 to +0.63 (mean +0.108, std 0.265). Highly unstable.
31% of windows negative, 69% positive.
VERDICT: The "inverse correlation / natural hedge" narrative is NOT supported.
         They move together in most regimes, near-zero in others.

### Block 2: 0 / 7 sessions survived BH FDR.
All sessions show 44-49% opposite breaks -- completely random, no directional signal.
VERDICT: No session-level directional hedge. MGC and MNQ break independently.

### Block 3: Average pnl_r correlation = +0.036. Near-zero.
Co-loss rate varies by session (34-41%). One marginal BH survivor: SINGAPORE_OPEN r=+0.068 (negligible).
VERDICT: Instruments trade independently. No shared drawdown risk. Genuine diversification.

## Portfolio Construction Implication

Long MGC + short MNQ: NOT a "natural hedge" by correlation.
The two positions are effectively independent edges. This is BETTER than a correlated hedge
because it provides genuine portfolio diversification without concentration risk.

## Trade Sheet Panel Decision

NO panel addition at this stage. Findings:
- Block 1: Correlation is positive (wrong direction for hedge narrative) -- not actionable as context
- Block 2: No session-level direction signal -- nothing to show
- Block 3: Near-zero P&L correlation = independence = a fact, not a signal

IF a panel is added later, only tested BH-surviving findings go in (none here).
"General market knowledge" (safe haven vs risk-on) is labelled as such, not as tested finding.

## Files
- research/research_mgc_mnq_correlation.py -- research script
- research/output/mgc_mnq_correlation_findings.md -- findings (to write post-run)
