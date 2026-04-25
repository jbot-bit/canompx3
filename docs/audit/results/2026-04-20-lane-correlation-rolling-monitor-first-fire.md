# Lane-Correlation Rolling Monitor — First-Fire Report (2026-04-20)

**Branch:** `research/lane-correlation-rolling-monitor`
**Script:** `scripts/reports/monitor_lane_correlation_rolling.py`
**JSON artefact:** `docs/runtime/lane_correlation_monitor.json`
**Follow-up to:** `docs/audit/results/2026-04-20-6lane-correlation-concentration-audit.md` (PR #33)

## Why this exists

The backtest-based 6-lane correlation audit (PR #33) concluded the current book is near-independent (mean pairwise correlation +0.006 over 2019-2026). That conclusion was based on 7 years of historical data. Carver Ch 11 (p170) explicitly warns that correlations can jump in crisis regimes — the book's diversification may evaporate exactly when it matters. A forward-looking tripwire was flagged as follow-up.

This monitor implements that tripwire. Run daily after the nightly backfill. Any pair whose 30-day rolling Pearson correlation exceeds **0.30 for ≥ 10 consecutive trading days** raises an alarm.

## First-fire result — 2026-04-20

**MONITOR STATUS: ALARM** — 1 alarm on first run against live `paper_trades` data.

```
MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5  <->  MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5
  run:    2026-03-18 → 2026-04-06  (14 consecutive trading days)
  peak:   +0.4281
  current (as-of 2026-04-16): +0.3703
```

Book-wide numbers as-of 2026-04-16 (30-day rolling):
- Mean pairwise correlation across 15 pairs: **+0.066** (still low)
- Max pair correlation: **+0.370** (single pair tripping the tripwire)
- 14/15 pairs are well-behaved; 1 pair is the whole signal

## How this reconciles with PR #33 audit and the other terminal's ORB_G5 overlap audit

**PR #33 (backtest 2019-2026, full history):** EUROPE_FLOW × COMEX_SETTLE correlation = **+0.009** (near zero across 7 years). Book N_eff = 5.82/6.

**Other terminal's ORB_G5 overlap audit (`orb_g5_cross_session_overlap.md`):** EUROPE_FLOW × COMEX_SETTLE IS correlation = **+0.333** on the in-sample window with no-filter flip pressure test — flagged PARTIAL_OVERLAP, no DROP.

**This monitor (2026-01 → 2026-04, 30-day rolling on live/shadow):** **+0.37** current, 14-day run peak 0.43. **Live reality is in line with the static audit's 0.333 — and somewhat higher in rolling window.**

The three numbers are consistent once you account for timing:
- 2019-2026 mean (0.009) is dominated by years where the pair was genuinely uncorrelated; recent correlation is pulling the tail of the distribution up but not moving the mean much.
- The static-audit IS number (0.333) and this monitor's rolling number (0.37-0.43) both sample the recent regime.

## What this tells us about the book

The portfolio-concentration hypothesis was structurally-REFUTED (PR #33) but live data has now identified one specific pair that is actively correlated in the current regime. The book-wide independence claim still holds (14/15 pairs well-behaved), but the deep assumption — *same-filter pairs in neighbouring sessions don't share regime exposure* — is showing stress at the ORB_G5 × European-timing axis.

Possible mechanisms (hypotheses, not claims):
1. **Filter convergence**: ORB_G5 selects for large-range days. In 2026 so far, large-range days may be running in clusters that touch both European sessions simultaneously.
2. **News regime concentration**: Both sessions are exposed to European macro news (ECB, UK data, EU opening flow). If 2026 has been news-heavy in this window, both lanes fire together.
3. **Sample noise**: 30-day rolling on ~1-2 trades/lane/day is noisy. A handful of coincidental same-direction winners/losers can push rolling correlation for a couple of weeks.

## Actions

**Immediate:**
- **NO capital action.** One alarm on a book with 14/15 clean pairs is not a regime change — it's a watch item. Book N_eff remains robust.
- The other terminal's ORB_G5 overlap audit already concluded "No DROP" on the IS pair analysis. This monitor provides independent cross-validation of the 0.33-0.43 correlation magnitude.

**Operational:**
- Wire monitor into nightly cron after backfill completes. Not yet automated.
- Re-run daily; track whether the alarm clears (correlation retreats below 0.30 for 10+ days) or expands (peak pushes past 0.50, more pairs trip).
- If a SECOND pair trips within the next 30 days → escalate to formal portfolio review (sizing down, lane retirement candidacy).

**Research follow-up (lower priority):**
- Decompose COMEX_SETTLE × EUROPE_FLOW correlation by regime. Is the 2026 regime (high vol / high range-day clustering) different from 2019-2025? If yes, the pair may reliably correlate in that regime — implication for sizing.
- If filter-convergence is the cause, consider whether the ORB_G5 filter parameter can be session-differentiated to restore decorrelation without removing the lane.

## Reproducibility

```bash
DUCKDB_PATH=C:/Users/joshd/canompx3/gold.db \
PYTHONPATH=. PYTHONIOENCODING=utf-8 \
python scripts/reports/monitor_lane_correlation_rolling.py --verbose
```

Read-only against `gold.db`. Writes only to `docs/runtime/lane_correlation_monitor.json`.

Tests: `tests/test_scripts/test_monitor_lane_correlation_rolling.py` (9 tests passing).
