---
pooled_finding: true
per_cell_breakdown_path: docs/audit/results/2026-05-31-adaptive-stops-h4-raw-per-lane.csv
flip_rate_pct: 0.0
---

# H4 — Time-Based No-Progress Exit (paired-ΔR vs price stop)

## Pre-registration

LOCKED: `docs/audit/hypotheses/2026-05-31-adaptive-stops-h4-time-based-exit-stack-v1.yaml` (Pathway B, K=1). Pre-committed parameter (+0.5R MFE within 120s), no sweep.

## Method

- Canonical layers: `bars_1m` (intra-trade MFE timing) + `orb_outcomes` (booked baseline, via `_load_strategy_outcomes`).
- Arm A: incumbent price stop = booked `pnl_r`. Arm B: time-based no-progress exit (path-R at entry+120s if MFE < +0.5R within window; else existing exit stands).
- Window anchored on E2 fill `entry_ts` (NOT break-bar — §6.3). Cost applied identically to both arms (cost-neutral).
- Read-only; no DB write; MGC excluded by pre-reg scope.

## Per-lane results

| Lane | N_IS | N_OOS | A(price) IS | B(time) IS | ΔR IS | ΔR OOS | OOS power | year-consist | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8 | 289 | 12 | +0.1775 | +0.1082 | -0.0693 | +0.2440 | STATISTICALLY_USELESS | SIGN_INCONSISTENT | KILL |
| MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5 | 1555 | 81 | +0.0915 | -0.0263 | -0.1178 | -0.0294 | STATISTICALLY_USELESS | SIGN_INCONSISTENT | KILL |
| MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5 | 1582 | 87 | +0.0637 | -0.0503 | -0.1140 | -0.2393 | STATISTICALLY_USELESS | CONSISTENT | KILL |
| MNQ_NYSE_OPEN_E2_RR1.0_CB1_COST_LT12 | 1669 | 87 | +0.0820 | +0.0248 | -0.0573 | -0.0510 | STATISTICALLY_USELESS | SIGN_INCONSISTENT | KILL |
| MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O15 | 913 | 65 | +0.1130 | -0.0433 | -0.1563 | -0.0877 | STATISTICALLY_USELESS | CONSISTENT | KILL |
| MNQ_TOKYO_OPEN_E2_RR1.5_CB1_COST_LT12 | 950 | 85 | +0.1222 | -0.0194 | -0.1415 | -0.1228 | STATISTICALLY_USELESS | CONSISTENT | KILL |
| MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5_O15 | 1491 | 74 | +0.1055 | +0.0049 | -0.1006 | -0.1861 | STATISTICALLY_USELESS | SIGN_INCONSISTENT | KILL |

## Pooled summary

- Lanes with verdict (N_IS>=50): 7
- Pooled IS ΔR (N-weighted): -0.1072
- Flip-rate: 0.0%
- PROCEED lanes: 0  |  KILL lanes: 7

## Classification use

PROCEED = the time-exit is measurably *less value-destroying* than the price stop on that lane (Howard "less bad, not good"). It routes to heavyweight Chordia review + independent positive-EV validation, NEVER to capital. KILL = single-layer no-progress exit dead for these lanes (does NOT kill the full 7-layer Howard stack — untested here).

