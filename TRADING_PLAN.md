# TRADING PLAN

Generated: 2026-03-01
Rolling window: 12m | Classification: STABLE >= 0.1 Sharpe in 60%+ of last 6 windows

## Portfolio Overview

| Instrument | Session | Head Strategy | Status | Size | N | ExpR | Sharpe(ann) |
|-----------|---------|---------------|--------|------|---|------|-------------|
| M2K | LONDON_METALS | `M2K_LONDON_METALS_E1_RR2.0_CB3_ORB_G6_FAST5_O15` [REGIME] | DEGRADED | OFF (0x) | 31 | +0.115 | 0.37 |
| M2K | NYSE_OPEN | `M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O30` | STABLE | Full (1.0x) | 734 | +0.086 | 1.13 |
| M2K | US_DATA_1000 | `M2K_US_DATA_1000_E2_RR1.0_CB1_VOL_RV12_N20_O30` | TRANSITIONING | Half (0.5x) | 472 | +0.132 | 1.47 |
| MES | CME_PRECLOSE | `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6` | TRANSITIONING | Half (0.5x) | 477 | +0.208 | 1.99 |
| MES | COMEX_SETTLE | `MES_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6` | STABLE | Full (1.0x) | 420 | +0.123 | 1.09 |
| MES | NYSE_CLOSE | `MES_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G4` | TRANSITIONING | Half (0.5x) | 386 | +0.119 | 1.04 |
| MES | NYSE_OPEN | `MES_NYSE_OPEN_E2_RR1.0_CB1_VOL_RV12_N20` | STABLE | Full (1.0x) | 939 | +0.111 | 1.42 |
| MGC | CME_REOPEN | `MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G5` | STABLE | Full (1.0x) | 112 | +0.252 | 1.01 |
| MGC | TOKYO_OPEN | `MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5_CONT` [REGIME] | STABLE | Full (1.0x) | 96 | +0.275 | 0.77 |
| MNQ | CME_PRECLOSE | `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G5` | STABLE | Full (1.0x) | 955 | +0.188 | 2.23 |
| MNQ | CME_REOPEN | `MNQ_CME_REOPEN_E2_RR1.0_CB1_VOL_RV12_N20` | STABLE | Full (1.0x) | 515 | +0.098 | 1.10 |
| MNQ | COMEX_SETTLE | `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_VOL_RV12_N20` | TRANSITIONING | Half (0.5x) | 705 | +0.180 | 1.36 |
| MNQ | LONDON_METALS | `MNQ_LONDON_METALS_E2_RR1.0_CB1_VOL_RV12_N20_O15` | TRANSITIONING | Half (0.5x) | 845 | +0.084 | 1.23 |
| MNQ | NYSE_CLOSE | `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20` | TRANSITIONING | Half (0.5x) | 330 | +0.203 | 1.85 |
| MNQ | NYSE_OPEN | `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4` | STABLE | Full (1.0x) | 1271 | +0.112 | 1.85 |
| MNQ | SINGAPORE_OPEN | `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G8_O15` | TRANSITIONING | Half (0.5x) | 744 | +0.108 | 1.17 |
| MNQ | TOKYO_OPEN | `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5_CONT` | TRANSITIONING | Half (0.5x) | 1227 | +0.098 | 1.37 |
| MNQ | US_DATA_1000 | `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5` | TRANSITIONING | Half (0.5x) | 1250 | +0.135 | 1.78 |

**Active: 17/18 session slots**

## M2K ($5.0/pt, $3.24 RT)

- **LONDON_METALS**: DEGRADED -> OFF (0x)
  - Head: `M2K_LONDON_METALS_E1_RR2.0_CB3_ORB_G6_FAST5_O15`
  - Full period: N=31 WR=22.6% ExpR=-0.3556 Sharpe=-0.2980
  - Last 3 window Sharpes: 0.000, 0.000, 0.000

- **NYSE_OPEN**: STABLE -> Full (1.0x)
  - Head: `M2K_NYSE_OPEN_E2_RR1.0_CB1_ORB_G5_O30`
  - Full period: N=734 WR=56.4% ExpR=+0.0804 Sharpe=0.0847
  - Last 3 window Sharpes: 0.169, 0.201, 0.245

- **US_DATA_1000**: TRANSITIONING -> Half (0.5x)
  - Head: `M2K_US_DATA_1000_E2_RR1.0_CB1_VOL_RV12_N20_O30`
  - Full period: N=472 WR=55.9% ExpR=+0.0598 Sharpe=0.0636
  - Last 3 window Sharpes: 0.130, 0.075, 0.077

## MES ($5.0/pt, $3.74 RT)

- **CME_PRECLOSE**: TRANSITIONING -> Half (0.5x)
  - Head: `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G6`
  - Full period: N=477 WR=65.4% ExpR=+0.2084 Sharpe=0.2370
  - Last 3 window Sharpes: 0.067, 0.062, 0.088

- **COMEX_SETTLE**: STABLE -> Full (1.0x)
  - Head: `MES_COMEX_SETTLE_E2_RR1.0_CB1_ORB_G6`
  - Full period: N=420 WR=61.0% ExpR=+0.1235 Sharpe=0.1373
  - Last 3 window Sharpes: 0.231, 0.265, 0.244

- **NYSE_CLOSE**: TRANSITIONING -> Half (0.5x)
  - Head: `MES_NYSE_CLOSE_E2_RR1.0_CB1_ORB_G4`
  - Full period: N=386 WR=62.7% ExpR=+0.1186 Sharpe=0.1373
  - Last 3 window Sharpes: -0.010, -0.036, -0.018

- **NYSE_OPEN**: STABLE -> Full (1.0x)
  - Head: `MES_NYSE_OPEN_E2_RR1.0_CB1_VOL_RV12_N20`
  - Full period: N=939 WR=60.5% ExpR=+0.1105 Sharpe=0.1229
  - Last 3 window Sharpes: 0.170, 0.180, 0.177

## MGC ($10.0/pt, $5.74 RT)

- **CME_REOPEN**: STABLE -> Full (1.0x)
  - Head: `MGC_CME_REOPEN_E2_RR1.0_CB1_ORB_G5`
  - Full period: N=112 WR=67.0% ExpR=+0.2522 Sharpe=0.2864
  - Last 3 window Sharpes: 0.387, 0.291, 0.284

- **TOKYO_OPEN**: STABLE -> Full (1.0x)
  - Head: `MGC_TOKYO_OPEN_E2_RR2.0_CB1_ORB_G5_CONT`
  - Full period: N=96 WR=45.8% ExpR=+0.2752 Sharpe=0.1984
  - Last 3 window Sharpes: 0.114, 0.162, 0.144

## MNQ ($2.0/pt, $2.74 RT)

- **CME_PRECLOSE**: STABLE -> Full (1.0x)
  - Head: `MNQ_CME_PRECLOSE_E2_RR1.5_CB1_ORB_G5`
  - Full period: N=955 WR=51.2% ExpR=+0.1877 Sharpe=0.1617
  - Last 3 window Sharpes: 0.148, 0.152, 0.132

- **CME_REOPEN**: STABLE -> Full (1.0x)
  - Head: `MNQ_CME_REOPEN_E2_RR1.0_CB1_VOL_RV12_N20`
  - Full period: N=515 WR=59.6% ExpR=+0.0984 Sharpe=0.1085
  - Last 3 window Sharpes: 0.160, 0.192, 0.142

- **COMEX_SETTLE**: TRANSITIONING -> Half (0.5x)
  - Head: `MNQ_COMEX_SETTLE_E2_RR2.5_CB1_VOL_RV12_N20`
  - Full period: N=705 WR=36.2% ExpR=+0.1796 Sharpe=0.1145
  - Last 3 window Sharpes: 0.088, 0.076, 0.065

- **LONDON_METALS**: TRANSITIONING -> Half (0.5x)
  - Head: `MNQ_LONDON_METALS_E2_RR1.0_CB1_VOL_RV12_N20_O15`
  - Full period: N=845 WR=56.4% ExpR=+0.0724 Sharpe=0.0768
  - Last 3 window Sharpes: 0.057, 0.025, -0.005

- **NYSE_CLOSE**: TRANSITIONING -> Half (0.5x)
  - Head: `MNQ_NYSE_CLOSE_E2_RR1.0_CB1_VOL_RV12_N20`
  - Full period: N=330 WR=64.5% ExpR=+0.2028 Sharpe=0.2271
  - Last 3 window Sharpes: 0.027, 0.058, 0.131

- **NYSE_OPEN**: STABLE -> Full (1.0x)
  - Head: `MNQ_NYSE_OPEN_E2_RR1.0_CB1_ORB_G4`
  - Full period: N=1271 WR=57.3% ExpR=+0.1115 Sharpe=0.1161
  - Last 3 window Sharpes: 0.168, 0.149, 0.133

- **SINGAPORE_OPEN**: TRANSITIONING -> Half (0.5x)
  - Head: `MNQ_SINGAPORE_OPEN_E2_RR2.0_CB1_ORB_G8_O15`
  - Full period: N=744 WR=40.7% ExpR=+0.1407 Sharpe=0.1022
  - Last 3 window Sharpes: 0.059, 0.055, 0.081

- **TOKYO_OPEN**: TRANSITIONING -> Half (0.5x)
  - Head: `MNQ_TOKYO_OPEN_E2_RR1.5_CB1_ORB_G5_CONT`
  - Full period: N=1227 WR=49.1% ExpR=+0.0983 Sharpe=0.0877
  - Last 3 window Sharpes: 0.053, 0.052, 0.046

- **US_DATA_1000**: TRANSITIONING -> Half (0.5x)
  - Head: `MNQ_US_DATA_1000_E2_RR1.5_CB1_ORB_G5`
  - Full period: N=1250 WR=47.3% ExpR=+0.1347 Sharpe=0.1123
  - Last 3 window Sharpes: -0.010, 0.043, 0.051

## Position Sizing Rules

- **STABLE**: Full size (1.0x risk per trade)
- **TRANSITIONING**: Half size (0.5x risk per trade)
- **DEGRADED**: OFF — do not trade until 3 consecutive passing windows

## Rolling Re-evaluation

- Run monthly: `python scripts/tools/rolling_portfolio_assembly.py --all`
- STABLE -> TRANSITIONING: reduce size by 50%
- TRANSITIONING -> DEGRADED: turn OFF
- DEGRADED -> STABLE: requires 3 consecutive passing windows

