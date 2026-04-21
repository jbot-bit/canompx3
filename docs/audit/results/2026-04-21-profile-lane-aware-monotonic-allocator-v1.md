# Profile Lane-Aware Monotonic Allocator Replay

**Date:** 2026-04-21
**Pre-registration:** `docs/audit/hypotheses/2026-04-21-profile-lane-aware-monotonic-allocator-v1.yaml`
**Profile:** `topstep_50k_mnq_auto`
**Evaluation window:** `2025-01-01` to `2025-12-31` (OOS-CV only)
**Sacred holdout:** `2026-01-01+` untouched

## Result

- Verdict: `PARK`
- Baseline total PnL: `+6,467.15`
- Candidate total PnL: `+5,690.46`
- Delta total PnL: `-776.69`
- Baseline Sharpe: `+1.5099`
- Candidate Sharpe: `+0.8709`
- Sharpe delta: `-0.6390`
- Sharpe delta 95% bootstrap CI: `[-1.5003, +0.3407]`
- Baseline max drawdown: `-3,701.41`
- Candidate max drawdown: `-5,491.75`
- Baseline Calmar: `+1.5809`
- Candidate Calmar: `+0.9532`
- Trades with size change: `36.75%`
- Skip rate: `17.81%`
- Double-contract rate: `18.94%`
- Average absolute contract change per day: `1.6815`

## Lane Summaries

| Strategy | Train N | OOS N | Train ExpR | Avg weight | Avg contracts | Changed % | Fallback |
|---|---:|---:|---:|---:|---:|---:|---|
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | 0 | 255 | +0.048781 | 1.0000 | 1.0000 | 0.00% | insufficient_complete_train_rows:0 |
| MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 | 766 | 144 | +0.102652 | 1.2038 | 1.3403 | 40.97% |  |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | 1305 | 246 | +0.062040 | 1.0976 | 1.1504 | 43.50% |  |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | 1411 | 253 | +0.072700 | 0.8691 | 0.7352 | 47.83% |  |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | 732 | 217 | +0.139765 | 0.8850 | 0.7696 | 46.08% |  |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 | 1265 | 221 | +0.127076 | 1.1087 | 1.2081 | 47.06% |  |

## Notes

- Calendar flags were excluded from this replay because `session_guard.py` does not explicitly whitelist them.
- Break-bar, rel-vol, and post-break fields were excluded by contract.
- Static baseline remains the hurdle. This replay is an overlay test, not a rescue search.

### Fallback lanes

- `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` -> `insufficient_complete_train_rows:0`
