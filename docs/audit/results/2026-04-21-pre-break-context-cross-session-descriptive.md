# Cross-session pre-break context descriptive diagnostic

**Date:** 2026-04-21
**Branch:** `research/pre-break-context-cross-session-descriptive`
**Script:** `research/audit_pre_break_context_cross_session.py`
**Classification:** DESCRIPTIVE DIAGNOSTIC (read-only, no promotion)
  per `backtesting-methodology.md § RULE 10` — no pre-reg required because
  no write to `experimental_strategies` or `validated_setups`.

---

## Literature grounding

Per `institutional-rigor.md § 7` (ground in local resources before training memory), citations below are from verbatim extracts in `docs/institutional/literature/`, not from memory.

- **Fitschen 2013 Ch 3** (`fitschen_2013_path_of_least_resistance.md`, pp. 32-42): commodities trend on BOTH daily and intraday bars; stock indices trend on intraday bars (counter-trend on daily). Verbatim p.41: *"In the case of commodities, both daily bars and hourly bars have a tendency to trend."* This is the CORE ORB premise. **Prediction under Fitschen:** pre-ORB momentum direction should continue through the ORB break → aligned group should have HIGHER WR and HIGHER ExpR than opposed, universally across MNQ/MES/MGC.
- **Chan 2013 Ch 7** (`chan_2013_ch7_intraday_momentum.md`, pp. 155-168): intraday momentum exists; the **mechanism is stop-triggered cascade**. Verbatim p.155: *"There is an additional cause of momentum that is mainly applicable to the short time frame: the triggering of stops. Such triggers often lead to the so-called breakout strategies."* Chan's gap-momentum result on FSTX (equity-index future, Sharpe 1.4 over 8 years) is a direct analog for the project's MNQ/MES ORB premise. **Under Chan's mechanism,** pre-ORB directional momentum (our `pre_velocity`) should reflect the existing order-flow imbalance; a break in the SAME direction catches more stops and sustains the move → aligned wins more often.
- **Chordia et al 2018** (`chordia_et_al_2018_two_million_strategies.md`): t ≥ 3.79 strict threshold for no-prior-theory findings; t ≥ 3.00 acceptable with literature-grounded theory. K-framing is per-family not global; a broad signal passing a narrow K_family gate is legitimate (here: K_instrument = 30, K_global = 89).
- **Bailey-López de Prado 2014** (`bailey_lopez_de_prado_2014_deflated_sharpe.md`): multiple-testing haircut. Not directly used here (descriptive, not Sharpe promotion), but the framing — a result that is robust across narrow K families is evidence of genuine effect, not a tuning artifact — is load-bearing for the per-instrument panel below.
- **Harvey-Liu 2015 BHY** (`harvey_liu_2015_backtesting.md`): BH-FDR correction at the hypothesis-family level. Applied here as K_global = adequately-powered cells and K_instrument = per-asset adequately-powered cells.

**What the literature predicts vs what this descriptive finds:** under Fitschen, aligned pre-velocity should continue → positive Δ_WR universally. A flat-to-null cross-section result contradicts the *strength* of Fitschen's intraday trend prediction at the specific timescale of the 5-bar pre-ORB slope (which is finer than Fitschen's hourly bars). A null at this scale does NOT refute Fitschen — it narrows the observed continuation signal to other timescales or feature framings.

---

## Question

Does pre-ORB-end velocity direction alignment with E2 fill direction predict
outcome on the **UNFILTERED** universe, uniformly across the 12 × 3 × 3
deployed cross-section? Answer informs whether action queue item #2 should
be re-scoped to a universal pre-reg or remains lane-specific.

---

## Scope

- Sessions: 12 (all with canonical `pre_velocity` column)
- Instruments: 3 (ACTIVE_ORB_INSTRUMENTS)
- Apertures: 3 (O5, O15, O30)
- Cells attempted: **108**
- Cells with sufficient data (≥ 60 trades): **90**
- Cells underpowered per RULE 3.2 (n_aligned or n_opposed < 30): **1**
- Cells with adequate power: **89**

- Entry model: E2 CB=1 RR=1.5
- IS: trading_day < 2026-01-01 (Mode A sacred holdout)
- Source: canonical `orb_outcomes ⨝ daily_features` on (trading_day, symbol, orb_minutes)
- Feature: `orb_{session}_pre_velocity` (canonical per pipeline/build_daily_features.py)
- Direction: inferred from `entry_price > stop_price → long` (fill metadata, pre-entry-knowable)
- No pre-filter applied (unfiltered universe)

---

## Headline

- Adequately-powered cells: **89 / 108**
- Cells with p_WR < 0.05 (raw, uncorrected): **9**
- Cells surviving BH-FDR at q=0.05, K_global = 89: **0**

### Per-instrument BH-FDR (narrower K_family)

Per `backtesting-methodology.md § RULE 4`, K_family is the natural hypothesis unit. A signal that is asset-class-specific (e.g. metals vs equities) may survive under K_instrument but fail under K_global. Reported side-by-side, not as a rescue.

| Instrument | N_powered | p<0.05 (raw) | BH-FDR survivors (q=0.05) |
|---|---|---|---|
| MNQ | 33 | 2 | 0 |
| MES | 31 | 3 | 0 |
| MGC | 25 | 4 | 0 |

**No survivors at any per-instrument K_family either.** Rejects asset-class-specific universality as well as the full cross-section.

---

**No cells survive BH-FDR at K = (all adequately-powered cells), q=0.05.**
This rejects a broad universal pre_velocity alignment edge at this scope.
Individual cells may still be worth narrower pre-reg investigation, but a
cross-session Pathway-B rewrite of action queue item #2 is NOT supported.

---

## Full per-cell table (adequately-powered, sorted by |Welch t| desc)

| Instrument | Session | O | N_total | N_aligned | N_opposed | WR_a | WR_o | Δ_WR | p_WR | ExpR_a | ExpR_o | Δ_ExpR | t | p | BH |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MGC | COMEX_SETTLE | 15 | 657 | 313 | 320 | 32.9% | 43.4% | -10.53pp | 0.006 | -0.337R | -0.112R | -0.225R | -2.86 | 0.004 |  |
| MGC | COMEX_SETTLE | 5 | 847 | 400 | 412 | 38.2% | 47.1% | -8.84pp | 0.011 | -0.293R | -0.119R | -0.175R | -2.67 | 0.008 |  |
| MGC | COMEX_SETTLE | 30 | 480 | 234 | 226 | 32.1% | 43.4% | -11.31pp | 0.012 | -0.324R | -0.078R | -0.245R | -2.56 | 0.011 |  |
| MES | CME_REOPEN | 15 | 79 | 37 | 37 | 37.8% | 13.5% | +24.32pp | 0.017 | -0.219R | -0.716R | +0.498R | +2.41 | 0.019 |  |
| MNQ | CME_PRECLOSE | 5 | 1297 | 635 | 642 | 51.0% | 44.4% | +6.63pp | 0.018 | +0.166R | +0.013R | +0.153R | +2.39 | 0.017 |  |
| MES | US_DATA_1000 | 30 | 1271 | 635 | 594 | 45.7% | 39.2% | +6.44pp | 0.022 | +0.068R | -0.082R | +0.150R | +2.28 | 0.023 |  |
| MES | LONDON_METALS | 30 | 1667 | 756 | 748 | 45.2% | 39.7% | +5.53pp | 0.030 | +0.024R | -0.104R | +0.128R | +2.22 | 0.026 |  |
| MGC | LONDON_METALS | 15 | 912 | 413 | 446 | 40.0% | 47.5% | -7.58pp | 0.025 | -0.147R | -0.002R | -0.145R | -2.01 | 0.045 |  |
| MNQ | COMEX_SETTLE | 30 | 1222 | 604 | 599 | 43.2% | 37.6% | +5.65pp | 0.046 | +0.027R | -0.106R | +0.133R | +1.97 | 0.049 |  |
| MNQ | NYSE_OPEN | 15 | 1393 | 667 | 704 | 46.8% | 41.6% | +5.16pp | 0.055 | +0.140R | +0.012R | +0.129R | +1.97 | 0.049 |  |
| MNQ | US_DATA_1000 | 30 | 1181 | 564 | 608 | 45.6% | 40.6% | +4.94pp | 0.088 | +0.107R | -0.013R | +0.120R | +1.71 | 0.088 |  |
| MGC | TOKYO_OPEN | 15 | 913 | 434 | 396 | 47.2% | 42.4% | +4.81pp | 0.164 | -0.060R | -0.170R | +0.110R | +1.59 | 0.112 |  |
| MNQ | LONDON_METALS | 30 | 1677 | 809 | 826 | 45.6% | 41.9% | +3.72pp | 0.129 | +0.085R | -0.007R | +0.092R | +1.58 | 0.115 |  |
| MNQ | COMEX_SETTLE | 15 | 1498 | 727 | 749 | 45.3% | 41.3% | +4.00pp | 0.121 | +0.063R | -0.033R | +0.095R | +1.57 | 0.116 |  |
| MNQ | US_DATA_1000 | 15 | 1495 | 742 | 740 | 47.7% | 43.8% | +3.93pp | 0.129 | +0.150R | +0.053R | +0.097R | +1.56 | 0.120 |  |
| MES | US_DATA_830 | 15 | 1626 | 707 | 752 | 43.4% | 39.8% | +3.66pp | 0.156 | -0.046R | -0.131R | +0.085R | +1.50 | 0.134 |  |
| MNQ | TOKYO_OPEN | 5 | 1720 | 789 | 877 | 50.4% | 47.1% | +3.35pp | 0.172 | +0.109R | +0.032R | +0.077R | +1.42 | 0.157 |  |
| MES | US_DATA_1000 | 15 | 1553 | 757 | 740 | 46.8% | 43.4% | +3.39pp | 0.188 | +0.079R | -0.005R | +0.084R | +1.41 | 0.159 |  |
| MGC | SINGAPORE_OPEN | 15 | 913 | 421 | 429 | 39.7% | 44.1% | -4.39pp | 0.195 | -0.170R | -0.068R | -0.101R | -1.41 | 0.160 |  |
| MES | NYSE_CLOSE | 30 | 134 | 76 | 56 | 38.2% | 26.8% | +11.37pp | 0.171 | -0.174R | -0.421R | +0.246R | +1.39 | 0.168 |  |
| MES | NYSE_OPEN | 15 | 1524 | 732 | 718 | 47.1% | 43.6% | +3.54pp | 0.176 | +0.098R | +0.017R | +0.082R | +1.34 | 0.181 |  |
| MES | NYSE_CLOSE | 15 | 227 | 132 | 91 | 39.4% | 30.8% | +8.62pp | 0.187 | -0.144R | -0.326R | +0.182R | +1.28 | 0.200 |  |
| MES | NYSE_OPEN | 5 | 1682 | 816 | 779 | 43.3% | 46.2% | -2.95pp | 0.236 | -0.021R | +0.052R | -0.072R | -1.28 | 0.202 |  |
| MES | TOKYO_OPEN | 15 | 1717 | 753 | 768 | 41.3% | 44.3% | -2.97pp | 0.242 | -0.123R | -0.057R | -0.066R | -1.22 | 0.224 |  |
| MES | SINGAPORE_OPEN | 5 | 1721 | 737 | 790 | 44.2% | 40.8% | +3.47pp | 0.170 | -0.165R | -0.224R | +0.059R | +1.21 | 0.225 |  |
| MES | US_DATA_1000 | 5 | 1674 | 799 | 813 | 45.6% | 42.7% | +2.88pp | 0.245 | +0.010R | -0.053R | +0.062R | +1.13 | 0.259 |  |
| MGC | SINGAPORE_OPEN | 30 | 895 | 405 | 428 | 41.5% | 45.1% | -3.61pp | 0.293 | -0.095R | -0.010R | -0.084R | -1.12 | 0.265 |  |
| MGC | US_DATA_1000 | 30 | 440 | 194 | 234 | 37.6% | 42.7% | -5.11pp | 0.284 | -0.145R | -0.024R | -0.121R | -1.12 | 0.265 |  |
| MNQ | US_DATA_830 | 15 | 1639 | 812 | 784 | 41.1% | 44.0% | -2.87pp | 0.246 | -0.036R | +0.028R | -0.064R | -1.10 | 0.272 |  |
| MGC | US_DATA_830 | 30 | 669 | 321 | 331 | 42.7% | 47.1% | -4.45pp | 0.253 | -0.026R | +0.070R | -0.096R | -1.08 | 0.282 |  |
| MES | US_DATA_830 | 30 | 1590 | 698 | 726 | 40.3% | 37.6% | +2.65pp | 0.304 | -0.094R | -0.153R | +0.059R | +1.00 | 0.316 |  |
| MGC | US_DATA_1000 | 5 | 823 | 399 | 400 | 46.4% | 42.5% | +3.87pp | 0.272 | -0.006R | -0.082R | +0.075R | +0.99 | 0.322 |  |
| MGC | NYSE_OPEN | 15 | 738 | 374 | 336 | 44.4% | 41.1% | +3.31pp | 0.373 | +0.005R | -0.078R | +0.082R | +0.98 | 0.329 |  |
| MNQ | SINGAPORE_OPEN | 5 | 1722 | 797 | 876 | 47.9% | 45.1% | +2.84pp | 0.245 | +0.014R | -0.034R | +0.049R | +0.93 | 0.351 |  |
| MES | COMEX_SETTLE | 15 | 1518 | 721 | 711 | 41.9% | 39.5% | +2.36pp | 0.363 | -0.080R | -0.133R | +0.053R | +0.93 | 0.354 |  |
| MNQ | NYSE_OPEN | 30 | 1037 | 516 | 501 | 43.4% | 40.7% | +2.69pp | 0.385 | +0.062R | -0.005R | +0.067R | +0.88 | 0.376 |  |
| MES | US_DATA_830 | 5 | 1648 | 698 | 784 | 43.6% | 41.5% | +2.10pp | 0.414 | -0.096R | -0.142R | +0.046R | +0.85 | 0.397 |  |
| MGC | TOKYO_OPEN | 5 | 916 | 426 | 407 | 43.7% | 41.3% | +2.38pp | 0.487 | -0.205R | -0.259R | +0.053R | +0.84 | 0.402 |  |
| MES | CME_PRECLOSE | 5 | 1224 | 561 | 610 | 46.3% | 44.1% | +2.25pp | 0.440 | -0.014R | -0.066R | +0.051R | +0.82 | 0.410 |  |
| MES | SINGAPORE_OPEN | 30 | 1704 | 777 | 733 | 41.8% | 44.1% | -2.24pp | 0.380 | -0.100R | -0.054R | -0.045R | -0.82 | 0.411 |  |
| MGC | US_DATA_1000 | 15 | 652 | 312 | 321 | 45.8% | 42.4% | +3.47pp | 0.380 | +0.025R | -0.045R | +0.070R | +0.79 | 0.430 |  |
| MNQ | CME_PRECLOSE | 15 | 201 | 103 | 94 | 48.5% | 42.6% | +5.99pp | 0.399 | +0.152R | +0.019R | +0.133R | +0.78 | 0.436 |  |
| MGC | NYSE_OPEN | 30 | 520 | 260 | 240 | 44.2% | 40.8% | +3.40pp | 0.443 | +0.024R | -0.053R | +0.077R | +0.75 | 0.452 |  |
| MES | TOKYO_OPEN | 30 | 1698 | 752 | 753 | 40.0% | 42.0% | -1.94pp | 0.444 | -0.122R | -0.080R | -0.041R | -0.74 | 0.459 |  |
| MNQ | TOKYO_OPEN | 15 | 1719 | 805 | 860 | 47.1% | 45.5% | +1.62pp | 0.509 | +0.078R | +0.038R | +0.040R | +0.71 | 0.476 |  |
| MNQ | CME_REOPEN | 5 | 223 | 109 | 103 | 36.7% | 40.8% | -4.08pp | 0.542 | -0.183R | -0.081R | -0.102R | -0.68 | 0.499 |  |
| MGC | TOKYO_OPEN | 30 | 908 | 431 | 394 | 45.7% | 43.9% | +1.80pp | 0.604 | -0.048R | -0.096R | +0.048R | +0.67 | 0.506 |  |
| MES | EUROPE_FLOW | 5 | 1718 | 759 | 772 | 39.8% | 42.1% | -2.31pp | 0.358 | -0.201R | -0.170R | -0.032R | -0.63 | 0.532 |  |
| MNQ | US_DATA_1000 | 5 | 1674 | 822 | 839 | 46.7% | 45.3% | +1.42pp | 0.561 | +0.107R | +0.071R | +0.036R | +0.62 | 0.534 |  |
| MNQ | COMEX_SETTLE | 5 | 1636 | 783 | 828 | 47.9% | 46.1% | +1.76pp | 0.480 | +0.085R | +0.051R | +0.034R | +0.59 | 0.553 |  |
| MES | NYSE_OPEN | 30 | 1220 | 594 | 568 | 43.1% | 41.4% | +1.72pp | 0.552 | +0.018R | -0.022R | +0.040R | +0.59 | 0.558 |  |
| MNQ | LONDON_METALS | 15 | 1709 | 812 | 855 | 45.1% | 43.7% | +1.33pp | 0.585 | +0.055R | +0.022R | +0.033R | +0.58 | 0.564 |  |
| MNQ | TOKYO_OPEN | 30 | 1703 | 809 | 841 | 45.0% | 43.6% | +1.36pp | 0.580 | +0.049R | +0.017R | +0.031R | +0.55 | 0.584 |  |
| MGC | EUROPE_FLOW | 5 | 917 | 411 | 445 | 45.7% | 48.3% | -2.57pp | 0.451 | -0.135R | -0.100R | -0.035R | -0.54 | 0.590 |  |
| MNQ | BRISBANE_1025 | 5 | 1721 | 808 | 872 | 47.5% | 46.7% | +0.85pp | 0.727 | -0.015R | -0.042R | +0.027R | +0.53 | 0.597 |  |
| MNQ | NYSE_CLOSE | 30 | 131 | 69 | 61 | 37.7% | 42.6% | -4.94pp | 0.566 | -0.127R | -0.023R | -0.104R | -0.52 | 0.603 |  |
| MNQ | US_DATA_830 | 30 | 1611 | 791 | 777 | 40.5% | 41.8% | -1.37pp | 0.581 | -0.039R | -0.009R | -0.030R | -0.51 | 0.608 |  |
| MGC | LONDON_METALS | 30 | 897 | 395 | 450 | 44.8% | 43.8% | +1.03pp | 0.763 | -0.007R | -0.045R | +0.039R | +0.51 | 0.610 |  |
| MES | COMEX_SETTLE | 30 | 1290 | 608 | 612 | 41.1% | 39.7% | +1.41pp | 0.615 | -0.074R | -0.105R | +0.031R | +0.49 | 0.627 |  |
| MNQ | EUROPE_FLOW | 30 | 1699 | 816 | 827 | 43.3% | 44.4% | -1.12pp | 0.648 | +0.014R | +0.040R | -0.026R | -0.46 | 0.647 |  |
| MES | SINGAPORE_OPEN | 15 | 1719 | 783 | 742 | 43.4% | 44.3% | -0.92pp | 0.718 | -0.110R | -0.086R | -0.024R | -0.45 | 0.650 |  |
| MNQ | SINGAPORE_OPEN | 30 | 1707 | 822 | 837 | 47.1% | 46.1% | +0.96pp | 0.694 | +0.090R | +0.065R | +0.026R | +0.45 | 0.651 |  |
| MNQ | LONDON_METALS | 5 | 1717 | 823 | 853 | 45.3% | 46.5% | -1.22pp | 0.616 | +0.024R | +0.048R | -0.024R | -0.44 | 0.663 |  |
| MES | NYSE_CLOSE | 5 | 559 | 280 | 265 | 38.6% | 36.6% | +1.97pp | 0.636 | -0.188R | -0.224R | +0.037R | +0.41 | 0.679 |  |
| MNQ | BRISBANE_1025 | 15 | 1721 | 840 | 840 | 44.6% | 44.0% | +0.60pp | 0.806 | -0.005R | -0.026R | +0.021R | +0.39 | 0.698 |  |
| MGC | CME_REOPEN | 5 | 123 | 56 | 56 | 35.7% | 39.3% | -3.57pp | 0.696 | -0.312R | -0.241R | -0.070R | -0.39 | 0.700 |  |
| MNQ | EUROPE_FLOW | 15 | 1718 | 814 | 848 | 45.6% | 44.8% | +0.77pp | 0.754 | +0.051R | +0.031R | +0.020R | +0.35 | 0.728 |  |
| MNQ | NYSE_CLOSE | 5 | 612 | 314 | 294 | 43.0% | 44.2% | -1.22pp | 0.761 | -0.027R | +0.004R | -0.031R | -0.34 | 0.733 |  |
| MGC | LONDON_METALS | 5 | 917 | 427 | 437 | 43.3% | 44.4% | -1.07pp | 0.752 | -0.165R | -0.142R | -0.022R | -0.34 | 0.735 |  |
| MGC | SINGAPORE_OPEN | 5 | 918 | 397 | 458 | 45.1% | 43.9% | +1.20pp | 0.724 | -0.113R | -0.136R | +0.023R | +0.34 | 0.737 |  |
| MNQ | EUROPE_FLOW | 5 | 1718 | 819 | 843 | 47.6% | 46.9% | +0.76pp | 0.756 | +0.052R | +0.035R | +0.018R | +0.32 | 0.747 |  |
| MGC | EUROPE_FLOW | 30 | 905 | 405 | 440 | 44.9% | 44.1% | +0.85pp | 0.804 | -0.024R | -0.045R | +0.022R | +0.29 | 0.771 |  |
| MNQ | BRISBANE_1025 | 30 | 1715 | 826 | 849 | 43.6% | 43.2% | +0.36pp | 0.883 | +0.006R | -0.010R | +0.016R | +0.29 | 0.774 |  |
| MGC | US_DATA_830 | 15 | 766 | 369 | 376 | 44.7% | 43.6% | +1.10pp | 0.763 | -0.003R | -0.024R | +0.021R | +0.25 | 0.800 |  |
| MNQ | US_DATA_830 | 5 | 1661 | 800 | 818 | 43.8% | 44.3% | -0.50pp | 0.838 | -0.014R | -0.001R | -0.013R | -0.23 | 0.820 |  |
| MGC | EUROPE_FLOW | 15 | 913 | 408 | 444 | 44.1% | 43.5% | +0.65pp | 0.849 | -0.084R | -0.097R | +0.014R | +0.19 | 0.848 |  |
| MES | LONDON_METALS | 15 | 1701 | 758 | 777 | 42.3% | 41.8% | +0.52pp | 0.836 | -0.070R | -0.080R | +0.010R | +0.19 | 0.851 |  |
| MES | CME_PRECLOSE | 15 | 123 | 56 | 63 | 46.4% | 44.4% | +1.98pp | 0.828 | +0.058R | +0.019R | +0.039R | +0.18 | 0.854 |  |
| MGC | NYSE_OPEN | 5 | 902 | 447 | 420 | 43.6% | 44.0% | -0.42pp | 0.900 | -0.066R | -0.054R | -0.013R | -0.17 | 0.863 |  |
| MES | TOKYO_OPEN | 5 | 1719 | 752 | 771 | 44.5% | 44.6% | -0.07pp | 0.978 | -0.116R | -0.109R | -0.007R | -0.13 | 0.893 |  |
| MES | LONDON_METALS | 5 | 1717 | 756 | 789 | 43.8% | 43.7% | +0.06pp | 0.982 | -0.091R | -0.097R | +0.006R | +0.11 | 0.913 |  |
| MGC | US_DATA_830 | 5 | 834 | 379 | 429 | 43.5% | 42.9% | +0.65pp | 0.853 | -0.074R | -0.082R | +0.008R | +0.10 | 0.918 |  |
| MES | CME_REOPEN | 5 | 220 | 105 | 97 | 31.4% | 32.0% | -0.53pp | 0.936 | -0.356R | -0.342R | -0.014R | -0.10 | 0.919 |  |
| MES | EUROPE_FLOW | 15 | 1717 | 737 | 793 | 41.8% | 41.9% | -0.08pp | 0.976 | -0.105R | -0.110R | +0.005R | +0.09 | 0.930 |  |
| MNQ | NYSE_OPEN | 5 | 1650 | 814 | 809 | 45.5% | 45.7% | -0.28pp | 0.910 | +0.094R | +0.100R | -0.005R | -0.09 | 0.931 |  |
| MES | COMEX_SETTLE | 5 | 1642 | 743 | 811 | 42.7% | 42.5% | +0.12pp | 0.960 | -0.118R | -0.114R | -0.004R | -0.07 | 0.946 |  |
| MES | EUROPE_FLOW | 30 | 1700 | 729 | 788 | 42.4% | 42.4% | +0.00pp | 1.000 | -0.069R | -0.066R | -0.002R | -0.04 | 0.967 |  |
| MNQ | SINGAPORE_OPEN | 15 | 1719 | 829 | 842 | 46.9% | 46.8% | +0.13pp | 0.957 | +0.054R | +0.053R | +0.001R | +0.02 | 0.988 |  |
| MNQ | NYSE_CLOSE | 15 | 231 | 120 | 110 | 41.7% | 41.8% | -0.15pp | 0.981 | -0.029R | -0.030R | +0.001R | +0.01 | 0.994 |  |

### Underpowered cells (omitted from BH-FDR)

| Instrument | Session | O | N_total | N_aligned | N_opposed |
|---|---|---|---|---|---|
| MNQ | CME_REOPEN | 15 | 78 | 29 | 44 |

---

## Interpretation

- **No universal edge in pre_velocity direction alignment** at this scope.
- Consistent with the null hypothesis that pre-ORB-end momentum direction
  does not predict outcome on unfiltered E2 CB1 RR1.5 across the deployed cross-section.
- Does NOT preclude lane-specific edges or sub-conditioned frames
  (e.g. only when paired with volatility regime, or only on specific
  session-instrument combinations). Those would require separate pre-registered tests.

---

## Limitations

- Binary alignment (aligned / opposed) discards magnitude information in `pre_velocity`.
  A continuous-scaling quintile analysis would be a stronger next test if this
  pass shows any hint of signal.
- Fill direction inferred from `entry_price > stop_price`; relies on canonical
  backtest outcome-builder invariants being intact. Verified by `check_drift.py`.
- RR target fixed at 1.5 to match the dominant deployed lane shape. Other RRs
  (1.0, 2.0) would change the win-probability baseline and the ExpR arithmetic.
- The UNFILTERED universe includes trades where deployed cost-gate filters would
  have skipped the lane. If directional signal is masked by small-ORB cost drag,
  the descriptive null could underestimate true directional content. A follow-up
  with a cost-agnostic size-normalized baseline could disambiguate.

---

## Reproduction

```bash
python research/audit_pre_break_context_cross_session.py
```

Writes this document to
`docs/audit/results/2026-04-21-pre-break-context-cross-session-descriptive.md`.
