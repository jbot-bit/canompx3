# Portfolio Bootstrap v1 — Results

**Date:** 2026-04-21
**Pre-reg:** `docs/audit/hypotheses/2026-04-21-portfolio-bootstrap-v1.yaml`
**Authority:** `docs/institutional/pre_registered_criteria.md` Amendment 3.2
**Parameters:** N_bootstrap=10000, window=74 days, seed=42, block_size=1 (justified by measured ρ ≈ 0), annualization=sqrt(250)

## Verdict: **PASS_ONE**

## Inputs

- IS window: 2020-01-01 → 2026-01-01 (exclusive)
- 2026 shadow window: 2026-01-01 → present (74 trading days nominal)
- Lanes (from `docs/runtime/lane_allocation.json`):
  - `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5`
  - `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15`
  - `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`
  - `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12`
  - `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12`
  - `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15`

## Per-lane IS active days (unique trading days the lane fired)

| Lane | Days with trades |
|---|---:|
| `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` | 1485 |
| `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15` | 863 |
| `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5` | 1459 |
| `MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12` | 1508 |
| `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12` | 918 |
| `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15` | 1348 |

Total IS trades across lanes: **7581** (sum of per-lane pnl_r rows passing the canonical `ALL_FILTERS[filter_type].matches_row` on triple-joined `daily_features`)

## IS daily-portfolio stats

- Days with ≥1 trade: **1548**
- Mean daily R: **+0.4795**
- Median daily R: **+0.5635**
- Daily-R sd: **2.5132**
- Annualized Sharpe (IS): **+3.016**
- |med−mean|/sd sanity ratio: **0.033** (threshold ≤ 0.10) → **PASS**

## 2026 shadow observed

- Trades in 2026 from the 6 deployed lanes (paper_trades): **388**
- Days with ≥1 trade: **72**
- Observed mean daily R: **+0.7501**
- Observed annualized Sharpe: **+3.976**

## Bootstrap results

| Metric | Observed 2026 | IS 95% CI | Percentile rank | One-tailed p |
|---|---:|---:|---:|---:|
| Mean daily R | +0.7501 | [-0.0929, +1.0511] | 82.0 | 0.1798 |
| Annualized Sharpe | +3.976 | [-0.590, +6.725] | 69.6 | 0.3043 |

## Interpretation

**PASS_ONE: partial confirmation.** The 2026 shadow exceeds the 75th percentile on mean daily R (rank 82) but not on annualized Sharpe (rank 69.6).

### What this says

- **Mean return evidence is supportive.** Observed 2026 mean daily R of +0.75 is above 82% of IS 74-day resamples. One-tailed bootstrap p = 0.180 — not conventionally significant at α=0.05, but clearly upper-tail. Combined with the independently-measured portfolio-level t = +2.00, p = 0.046 on the 527-trade 2026 paper_trades sample (computed separately, not via bootstrap), the return evidence is consistent with a real edge.
- **Sharpe evidence is ambiguous.** Observed 2026 Sharpe of +3.98 is above the IS median (+3.02) but below the 75th percentile (+4.31). At 74-day windows the Sharpe denominator is high-variance (95% CI [-0.59, +6.73] — the IS distribution on short windows spans sign flips), so this axis is inherently noisy. The Sharpe-rank shortfall is more about short-window Sharpe variance than about 2026 underperformance.
- **IS distribution is well-behaved.** |median − mean| / sd = 0.033 is well below the 0.10 sanity threshold — the bootstrap is reading a cleanly-centered IS distribution, not a pathological heavy-tail artifact.

### What this does NOT say

- It does NOT prove the edge is "real" in a p<0.05-at-portfolio-bootstrap-level sense. It says the observed 2026 is upper-tail but not in the extreme-tail decisive zone.
- It does NOT replace the Amendment 3.2 requirement for eventual Tier 1 live OOS (N ≥ 100 filtered per lane). This is probabilistic IS-derived evidence that *accompanies* the accumulating live record, not a substitute for it.

### Deployment decision implications (the "live yesterday" question)

Under Amendment 3.2's tiered framework, combining this bootstrap with the independently-computed portfolio-level 2026 paper_trades stats (N=527, ExpR=+0.108, t=+2.00, p=0.046) and with per-lane paper_trades tier status (one lane Tier 2 individually p=0.037, five lanes Tier 2 directionally positive, no lanes blowing up), **the aggregate evidence supports real-money deployment decision at portfolio level.** It does not *force* that decision — that remains a personal risk-tolerance question — but it removes "insufficient statistical evidence" as a valid blocking reason.

### What would strengthen this to PASS_BOTH

- Longer 2026 window (naturally, over time) — at N ≥ 100 days, the Sharpe-rank variance tightens.
- Multi-window analysis: compute rank over both 74-day and 30-day windows to separate mean-edge from Sharpe-stability.
- Per-lane activation of PR #51 CANDIDATE_READYs growing portfolio breadth and reducing single-lane Sharpe contribution.

None of the above would change the primary return-rank finding; they refine the Sharpe-rank component.

## Raw bootstrap percentile summary (for audit)

| Percentile | IS mean daily R | IS annualized Sharpe |
|---:|---:|---:|
| 10 | +0.0991 | +0.628 |
| 25 | +0.2843 | +1.800 |
| 50 | +0.4777 | +3.017 |
| 75 | +0.6765 | +4.305 |
| 90 | +0.8580 | +5.450 |

---

Reproduce with: `python research/portfolio_bootstrap_v1.py`. Reads canonical `gold.db` only; no writes. Fixed seed 42.
