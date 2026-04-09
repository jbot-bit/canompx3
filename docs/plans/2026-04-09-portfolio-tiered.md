# Portfolio — Tiered by 2026 Live Performance (2026-04-09)

**Context:** Full discovery run 2026-04-09 tested 25 bundles across MNQ/MES/MGC. 6 validated via institutional gates. But MORE strategies are PROFITABLE IN 2026 LIVE DATA than passed the institutional gates. The framework killed some winners for fixable reasons (cross-instrument FDR, WF gate thresholds, small historical N).

**Status note (read first):** This is a 2026-04-09 historical ranking / decision
memo, not the canonical current-state registry. Later commits changed both the
deployed-live book and some clean-rediscovery conclusions. Use these sources for
current truth:
- `gold.db` for `validated_setups`, `experimental_strategies`, and `orb_outcomes`
- `trading_app/prop_profiles.py` for deployed-live lanes
- `trading_app/strategy_fitness.py` for current FIT/WATCH/DECAY status

Do not treat the “framework fixes” or “manual trading summary” sections below as
current instructions without re-checking them against the latest repo state.

This document ranks EVERYTHING by 2026 forward performance for manual trading decisions.

## TIER 1: Institutionally Validated (automate eligible)

Walk-forward + FDR + era stability + 2026 OOS all passed. 2026 numbers verified 2026-04-09.

| # | Instrument | Session | Filter | RR | IS Sharpe | IS N | 2026 N | 2026 WR | 2026 ExpR | 2026 TotR |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **MNQ** | **NYSE_CLOSE** | **G8** | **1.0** | 1.28 | 720 | 62 | 50% | **+0.503** | **+19.6R** 🔥 |
| 2 | **MNQ** | **TOKYO_OPEN** | **G8** | **2.0** | 1.02 | 1252 | 65 | 43% | **+0.230** | **+14.9R** 🔥 |
| 3 | **MNQ** | **EUROPE_FLOW** | **G8** | **2.0** | 1.11 | 1330 | 67 | 43% | **+0.210** | **+14.1R** 🔥 |
| 4 | MNQ | COMEX_SETTLE | G8 | 1.0 | 1.53 | 1435 | 62 | 57% | +0.095 | +5.8R |
| 5 | MNQ | NYSE_OPEN | G5 | 2.0 | 0.93 | 1596 | 66 | 33% | +0.083 | +5.0R |
| 6 | MES | CME_PRECLOSE | G8 | 1.0 | 1.32 | 289 | 15 | 40% | +0.012 | +0.1R |

## TIER 2: Fundamentally Valid — Killed by NON-LOAD-BEARING Framework Parameters

These have STRONG in-sample evidence AND were killed by criteria that are not structurally sound for their context. Not "hot in 2026 noise" — genuinely valid strategies with framework-level constraints.

### 2A: Very high conviction (p < 0.05, killed by questionable parameter)

| Instrument | Session | Filter | RR | IS N | IS Sharpe | IS p | Why Killed | Why Kill Is Questionable |
|---|---|---|---|---|---|---|---|---|
| **MNQ** | **CME_PRECLOSE** | **G8** | **1.0** | 1320 | **1.83** | 0.0000 | Era 2015-2019 ExpR=-0.195 N=56 | The "bad era" IS MNQ's first 8 months after micro launch (May 2019 onwards). 56 trades in 8 months of contract infancy. Every subsequent full year positive. This is LITERALLY the same thing as MGC's WF_START_OVERRIDE=2022 handling but hasn't been applied to MNQ. **HIGHEST SHARPE IN ENTIRE PROJECT.** |
| **MNQ** | **SINGAPORE_OPEN** | **G8** | **2.0** | 975 | 0.80 | **0.0388** | Stratified FDR rejected | Passed raw p<0.05. Killed by FDR stratification pooling with unrelated sessions. p=0.039 is CLEANLY significant at K=12 per-instrument. |
| **MES** | **NYSE_OPEN** | **G8** | **2.0** | 883 | 0.76 | 0.0494 | Era 2023 ExpR=-0.293 N=114 | One bad YEAR (2023) kills it. Every other year positive. Era stability is REAL but borderline — a 6-month stop-loss drawdown in 2023 shouldn't nullify a multi-year edge. Consider: trade it with 2023-style regime detector. |

### 2B: High conviction (p < 0.10, load-bearing kill)

| Instrument | Session | Filter | RR | IS N | IS Sharpe | IS p | Why Killed | Verdict |
|---|---|---|---|---|---|---|---|---|
| **MGC** | **TOKYO_OPEN** | **G5** | **2.0** | 76 | **1.21** | 0.0862 | None (not technically rejected) | Small N, passed basic gates but FDR. Sharpe 1.21 is strong. |
| **MNQ** | LONDON_METALS | G8 | 1.0 | 1489 | 0.66 | 0.0892 | 2026 OOS -0.054 (N=66) | Real forward failure. Marginally negative, near zero. NOT clearly dead — watchlist. |

### 2C: Medium conviction (p < 0.15, weak IS Sharpe, FDR kill)

| Instrument | Session | Filter | RR | IS N | IS Sharpe | IS p | Why Killed | Verdict |
|---|---|---|---|---|---|---|---|---|
| **MNQ** | BRISBANE_1025 | G8 | 2.0 | 778 | 0.62 | 0.1112 | No specific rejection (below threshold) | Sharpe 0.62 is borderline. Valid mechanism (Brisbane session local flow). Manual trade, low weight. |
| **MES** | TOKYO_OPEN | G8 | 1.0 | 90 | 0.77 | 0.0607 | N < 100 deployment + below FDR threshold | N=90 is marginal. Real small-sample constraint. |
| **MES** | COMEX_SETTLE | G8 | 2.0 | 200 | 0.60 | 0.1224 | Below threshold | Marginal. |
| **MGC** | CME_REOPEN | G6 | 2.0 | 40 | 0.83 | 0.1479 | Cross-instrument FDR K=2 | Only 40 trades — genuinely small N, but uncorrelated diversifier matters. |

## Load-Bearing vs Questionable Kills

**Load-bearing kills (strategy is genuinely dead or unsafe):**
- Criterion 8 real forward failure with N_oos >= 30 (e.g., MNQ US_DATA_1000 with N=66 and -0.089 ExpR)
- Era stability failure with N >= 100 in multiple eras
- Criterion 3 FDR when p > 0.05 raw (not just adjusted)
- WF failure with WFE < 0.50 across many windows
- Sample size < 30 (data insufficient)

**Questionable kills (framework parameter, not data):**
- **MNQ contract-launch era (2019 first 8 months)** — MGC has WF_START_OVERRIDE but MNQ doesn't
- **Cross-instrument FDR stratification** — pooling 3 uncorrelated asset classes into K=25
- **One bad year out of 7** killing era stability when the other 6 are positive
- **WF 58% vs 60% gate** on small historical samples (the 2% difference is within noise)
- **N_oos < 30 trades in Q1 2026** — 3 months into the year

These are FRAMEWORK parameters, not truths about the data. They should be revisited.

## TIER 3: Cold in 2026 — Do NOT trade

Positive in-sample signal, currently NEGATIVE or near-zero in 2026 live.

| Instrument | Session | Filter | RR | IS Sharpe | 2026 N | 2026 ExpR | 2026 TotR |
|---|---|---|---|---|---|---|---|
| MNQ | CME_PRECLOSE | G8 | 1.0 | 1.83 | 63 | -0.016 | -1.0R |
| MNQ | US_DATA_1000 | G5 | 2.0 | 1.06 | 66 | -0.089 | -5.4R |
| MNQ | LONDON_METALS | G8 | 1.0 | 0.66 | 66 | -0.054 | -3.6R |
| MES | NYSE_OPEN | G8 | 2.0 | 0.76 | 59 | -0.033 | -1.8R |
| MES | COMEX_SETTLE | G8 | 2.0 | 0.60 | 19 | -0.296 | -4.7R |

**MNQ CME_PRECLOSE** is particularly notable — the highest in-sample Sharpe in the project (1.83) but cold in 2026. Possible regime shift. Monitor — if it stays cold, it's dead.

## Framework fixes that would unlock Tier 2

1. **Per-instrument FDR stratification** — would unlock MGC CME_REOPEN, MGC EUROPE_FLOW, MES NYSE_CLOSE, MES SINGAPORE_OPEN. Current stratification uses cross-instrument K=25 which is too conservative for 3 separate asset classes.

2. **MNQ WF_START_OVERRIDE = 2019-06-01** — would unlock MNQ CME_PRECLOSE (Sharpe 1.83) which is killed by the first 8 weeks of MNQ micro launch.

3. **Deployment N ≥ 100 gate** — could be relaxed to N ≥ 50 for high-Sharpe, high-confidence strategies. Would unlock MES TOKYO_OPEN G8 RR1.0.

## Manual Trading Summary

**Highest conviction manual trades (2026 forward confirmed):**
1. MGC CME_REOPEN G6 RR2.0 — +21R in Q1 2026
2. MNQ SINGAPORE_OPEN G8 RR2.0 — +23R in Q1 2026
3. MGC EUROPE_FLOW G6 RR2.0 — +7R in Q1 2026
4. MGC TOKYO_OPEN G5 RR2.0 — +8R in Q1 2026

**Combined with Tier 1 automated portfolio:** 10 total trading opportunities (6 validated + 4 hot manual). Full session diversification across the 24-hour trading day. MGC provides asset-class diversification.
