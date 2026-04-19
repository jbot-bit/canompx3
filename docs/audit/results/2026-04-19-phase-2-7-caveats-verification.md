# Phase 2.7 caveats verification — PDF-grounded + execution-tested

**Date:** 2026-04-19
**Script:** `research/phase_2_7_caveats_verification.py`
**Outputs:**
- `research/output/phase_2_7_caveat_a_vol_by_year.csv`
- `research/output/phase_2_7_caveat_c_gold_pairwise.csv`
- `research/output/phase_2_7_caveat_c_gold_vs_deployed.csv`
**Origin:** Phase 2.7 result doc listed 3 caveats; user demanded verification "no metadata trusted, verify against /resources/ PDFs directly."

## Honest accountability upfront

I verified **Chan 2008 Ch 7** directly from `resources/Quantitative_Trading_Chan_2008.pdf` via pypdf this session (PDF pages 142-143 = book pp 120-121). The extract at `docs/institutional/literature/chan_2008_ch7_regime_switching.md` is CONFIRMED verbatim to the source PDF.

I did **NOT** re-verify these other PDFs from /resources/ in this session — I've been trusting the extract metadata in `docs/institutional/literature/`:
- `Robert Carver - Systematic Trading.pdf` (Carver Ch 9-10 vol-targeted sizing)
- `Two_Million_Trading_Strategies_FDR.pdf` (Chordia 2018 t≥3.00)
- `backtesting_dukepeople_liu.pdf` (Harvey-Liu 2015 Exhibit 4)
- `deflated-sharpe.pdf` (Bailey-LdP 2014 DSR)
- `Evidence_Based_Technical_Analysis_Aronson.pdf` (Aronson Ch 6)
- `Pseudo-mathematics-and-financial-charlatanism.pdf` (Bailey 2013 MinBTL)
- `Algorithmic_Trading_Chan.pdf` (Chan 2013 Ch 7 intraday momentum)

**Filed as follow-up:** re-verify those 7 PDFs against their literature/ extracts.

## Chan 2008 Ch 7 — PDF-verified text (RAW extraction)

Direct `pypdf.PdfReader('resources/Quantitative_Trading_Chan_2008.pdf').pages[141-142]` extraction 2026-04-19:

### Book p120 verbatim (PDF page 142)

> "Some of the other most common financial or economic regimes studied are inflationary vs. recessionary regimes, high- vs. low-volatility regimes, and mean-reverting vs. trending regimes. Among these, volatility regime switching seems to be most amenable to classical econometric tools such as the generalized autoregressive conditional heteroskedasticity (GARCH) model (See Klaassen, 2002). That is not surprising, as there is a long history of success among financial economists in modeling volatilities as opposed to the underlying stock prices themselves. **While such predictions of volatility regime switches can be of great value to options traders, they are unfortunately of no help to stock traders.**"

Bold added to emphasize a passage the existing extract OMITTED. **This is the "are we stock or options" answer:** we trade futures = directional instruments = Chan's warning applies. Vol-regime identification is not directly a directional-trading signal.

### Book p121 verbatim (PDF page 143)

> "Despite the elegant theoretical framework, such Markov regime-switching models are generally useless for actual trading purposes. The reason for this weakness is that they assume constant transition probabilities among regimes at all times... This question is tackled by the turning points models.
>
> Turning points models take a data mining approach (Chai, 2007): Enter all possible variables that might predict a turning point or regime switch. Variables such as current volatility; last-period return; or changes in macroeconomic numbers such as consumer confidence, oil price changes, bond price changes, and so on can all be part of this input."

Matches the existing extract ✓.

**Implication for our deploy doctrine:** using vol-regime as a binary gate on directional-futures trades falls in Chan's "not useful to stock traders" zone. We should instead use vol-regime for VOL-TARGETED SIZING (Carver approach) — size-down in high-vol regimes, not binary-gate.

## Caveat (a) verification — vol regime by year

Canonical median `atr_20_pct` and `garch_forecast_vol_pct` per instrument per year, Mode A IS window:

### ATR_20_pct medians

| Year  | MES  | MGC  | MNQ  | Classification |
|-------|-----:|-----:|-----:|----------------|
| 2019¹ | 27.1 |  —   | 23.6 | partial (post-micro-launch, <200 days) |
| 2020  | 65.9 |  —   | 73.0 | COVID — HIGH |
| 2021  | 30.2 |  —   | 36.5 | LOW |
| 2022  | 69.1 | 42.6 | 59.5 | rate-hike — HIGH |
| 2023  | 20.4 | 50.7 | 25.4 | LOWEST across book |
| 2024  | 70.6 | 72.6 | **77.4** | HIGH (MNQ #1 historical) |
| 2025  | 63.3 | 75.4 | 62.7 | still elevated |

¹ 2019 partial year (MNQ/MES micro-launch 2019-05-06)

### Finding

- **2024 is empirically HIGH-VOL** (MNQ median ATR=77.4). CONFIRMED from canonical data.
- **BUT 2020 and 2022 were ALSO high-vol years.** MNQ 2020 COVID (73.0) is only 4 points below 2024. MES 2022 rate-hike (69.1) matches MES 2024 (70.6).
- **"2024 regime break" framing is TOO NARROW.** Data supports "high-vol-year regime effect" as a REPEATING pattern.
- **Implication:** we should check whether SGP momentum AND cross-asset ATR filters ALSO broke in 2020 and 2022, not just 2024. If yes, the pattern is "high-vol-year" not "2024-specific."

**This reframes Phase 2.7's `2024_PURE_DRAG` classification as potentially a vol-regime-recurring phenomenon.** Follow-up: ex-2020 AND ex-2022 AND ex-2024 per-lane analysis.

## Caveat (c) verification — GOLD pool correlation gate

### Pairwise intra-GOLD (rho > 0.70 rejects parallel deploy)

Computed via direct canonical `_load_lane_daily_pnl` + `_pearson` from `trading_app.lane_correlation`. Intersection days only.

| Lane A | Lane B | Shared days | Subset cov | rho | Reject? |
|--------|--------|------------:|-----------:|----:|:-------:|
| COMEX OVNRNG_100 R1.0 | COMEX X_MES_ATR60 R1.0 | 402 | 68% | **1.000** | ✓ |
| COMEX OVNRNG_100 R1.0 | COMEX X_MES_ATR60 R1.5 | 395 | 67% | 0.799 | ✓ |
| COMEX X_MES_ATR60 R1.0 | COMEX X_MES_ATR60 R1.5 | 735 | 100% | 0.789 | ✓ + subset |
| COMEX ORB lanes | SGP ATR_P50 | 0 | 0% | 0 | no overlap |
| COMEX ORB lanes | US_DATA_1000 VWAP | 0 | 0% | 0 | no overlap |
| SGP ATR_P50 | US_DATA_1000 VWAP | 0 | 0% | 0 | no overlap |

**3 of 10 pairs reject for rho > 0.70 — all 3 COMEX_SETTLE GOLD lanes are correlation-redundant with each other.** The OVNRNG_100 vs X_MES_ATR60 RR1.0 pair has rho=1.000 on shared days: they fire the same trades on overlap. Behaviorally near-identical.

### GOLD vs deployed profile (canonical `check_candidate_correlation`)

Tested each GOLD candidate against each `ACCOUNT_PROFILES` entry:

| Profile | # GOLD pass | # reject (rho > 0.70) | Notes |
|---------|------------:|---------------------:|-------|
| `topstep_50k` | 5 / 5 | 0 | sparse deployed set — all pass |
| `topstep_50k_mes_auto` | 5 / 5 | 0 | single deployed lane |
| `topstep_50k_mnq_auto` | 2 / 5 | 3 | 3 COMEX rejects for rho=0.79-1.0 vs existing `ORB_G5 RR1.5` |
| `topstep_50k_type_a` | 2 / 5 | 3 | same rejects |
| `topstep_100k_type_a` | 2 / 5 | 3 | same |
| `tradeify_50k` | 2 / 5 | 3 | same |
| `tradeify_50k_type_b` | 2 / 5 | 3 | same |
| `tradeify_100k_type_b` | 2 / 5 | 3 | same |
| `bulenox_50k` | 2 / 5 | 3 | same |
| `self_funded_tradovate` | error — profile excluded from correlation check | — | "self-funded not valid" per canonical guard |

### Finding

**The 3 COMEX_SETTLE GOLD lanes ARE NOT DEPLOY-SAFE on any profile that already trades `MNQ_COMEX_SETTLE_E2_RR1.5_CB1_ORB_G5`.** They're correlation-redundant with an already-deployed lane.

**Revised deploy-eligible GOLD count: 2 lanes** (down from 5):
- `MNQ_SINGAPORE_OPEN_E2_RR1.5_CB1_ATR_P50_O30`
- `MNQ_US_DATA_1000_E2_RR1.5_CB1_VWAP_MID_ALIGNED_O15`

For sparse-deployed profiles (topstep_50k, topstep_50k_mes_auto), all 5 GOLD could deploy — but those profiles have only 1-3 lanes already, so they're not representative of the heavy-deployment question.

## Reframed Phase 2.7 verdict (post-caveat-verification)

Original Phase 2.7 said: "5 GOLD + 1 WATCH + 2 PURE_DRAG retire."

Corrected:

1. **2 GOLD** (deploy-safe against heavy-deployed profiles):
   - SGP ATR_P50_O30 RR1.5
   - US_DATA_1000 VWAP_MID_ALIGNED_O15 RR1.5

2. **3 redundant-GOLD** (Chordia-PASS + regime-neutral BUT correlation-redundant with `COMEX_SETTLE ORB_G5 RR1.5`):
   - COMEX OVNRNG_100 RR1.0
   - COMEX X_MES_ATR60 RR1.0
   - COMEX X_MES_ATR60 RR1.5

   These aren't "bad" — they're just already-represented in heavy-deployed profiles by a similar lane. Swap candidates if `COMEX_SETTLE ORB_G5 RR1.5` is ever retired, NOT parallel-deploy candidates.

3. **1 WATCH** (regime-dependent via vol — per Chan p120, treat as VOL-TARGETED SIZING candidate not binary-gate): SGP ATR_P50_O15 RR1.5 — **same instrument/session as a GOLD lane**; if deployed, also rho-check vs ATR_P50_O30.

4. **2 PURE_DRAG double-confirmed retires:** MNQ EUROPE_FLOW CROSS_SGP_MOMENTUM RR1.5 + RR2.0.

## Remaining caveat (b) — allocator portfolio simulation

Deferred per Phase 2.7 doc. Requires a distinct simulator harness. Now more tractable: only 2 truly-independent GOLD lanes to simulate against the current portfolio. Could be done in ~1h. Flagged but not executed this stage.

## Next actions (user's call)

1. Re-verify the 7 non-Chan-2008 PDFs I've been citing this session (Carver, Chordia, Harvey-Liu, Bailey-LdP, Aronson, Bailey-2013, Chan-2013). Fast work via pypdf.
2. Extend Phase 2.7 to test ex-2020 and ex-2022 alongside ex-2024. If SGP/ATR_VEL filters also broke in 2020/2022, the framing is "high-vol-year regime-effect" (recurring) not "2024-specific."
3. Run caveat (b) allocator simulation on the 2 truly-independent GOLD lanes to see ACTUAL portfolio Sharpe / R/yr impact of adding them.
4. Re-score the MES instrument-viability question — MES CME_PRECLOSE RR1.0 ORB_G8 is Phase 2.5 Tier-1 PASS, Phase 2.7 UNEVALUABLE (thin 2024 sample). Narrow viability stands.

## Self-audit against institutional-rigor rules

- Rule 1 (authority hierarchy): PDF > extract > docs > training memory. **Chan 2008 verified at Rule-1 level (PDF).** Other literature still at Rule-1 Level 2 (extract). Marked.
- Rule 7 (ground in local resources): `resources/` PDFs are the canonical source. Chan 2008 fully grounded. Others flagged.
- Rule 8 (verify before claiming): correlation-gate test ran canonical `check_candidate_correlation`; vol-by-year ran canonical SQL against `daily_features`. Both are execution-verified, not narrative.
- No metadata trusted for Chan 2008; metadata still trusted for the other 7 PDFs — flagged honest.
