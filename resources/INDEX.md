# Resources Index — local corpus navigation manifest

> **Generated** by `scripts/tools/build_resources_index.py`. Do not edit by hand.
> Rebuild: `python scripts/tools/build_resources_index.py`.

**Grounding rule (CLAUDE.md § Local Academic / Project-Source Grounding):**
before citing training memory on any topic below, ground in the local source.
Where a **curated extract** exists in `docs/institutional/literature/`, that
extract is the **canonical citation source** (page-cited) — fetch it via the
`research-catalog` MCP (`get_literature_excerpt`), do NOT paraphrase the raw PDF
from memory. If no local source supports a claim, say **UNSUPPORTED**.

**22 resources indexed.**

| Resource | Topic | Curated extract (canonical cite) | Key terms (first page) |
|---|---|---|---|
| `Algorithmic_Trading_Chan.pdf` | Backtest method / look-ahead / TOC (Chan 2013) | `docs/institutional/literature/chan_2013_ch1_backtesting_lookahead.md` | ALGORITHMIC TRADING |
| `backtesting_dukepeople_liu.pdf` | Backtesting / Sharpe haircut (Harvey-Liu) | `docs/institutional/literature/harvey_liu_2015_backtesting.md` | 12 BACKTESTING FALL 2015 Backtesting CAMPBELL R. HARVEY AND YAN LIU CAMPBELL R. HARVEY is a professor at Duke University in Durham, NC, and  |
| `benjamini-and-Hochberg-1995-fdr.pdf` | Multiple-testing / FDR (BH-1995 primary) | `docs/institutional/literature/benjamini_hochberg_1995_fdr.md` | Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing Author(s): Yoav Benjamini and Yosef Hochberg Sou |
| `Building_Reliable_Trading_Systems.pdf` | Trade-system reliability (Fitschen) — ORB premise | `docs/institutional/literature/fitschen_2013_path_of_least_resistance.md` | BUILDING RELIABLE TRADING SYSTEMS |
| `deflated-sharpe.pdf` | Multiple-testing — Deflated Sharpe Ratio | `docs/institutional/literature/bailey_lopez_de_prado_2014_deflated_sharpe.md` | Electronic copy available at: http://ssrn.com/abstract=2460551 THE DEFLATED SHARPE RATIO: CORRECTING FOR SELECTION BIAS, BACKTEST OVERFITTIN |
| `Evidence_Based_Technical_Analysis_Aronson.pdf` | Data-snooping / EBTA (Aronson) | `docs/institutional/literature/aronson_2007_ebta_data_snooping.md` | ' \Vill'y Trading |
| `false-strategy-lopez.pdf` | Multiple-testing — False Strategy Theorem | `docs/institutional/literature/lopez_de_prado_bailey_2018_false_strategy.md` | Mathematical Assoc. of America American Mathematical Monthly 125:7 August 22, 2018 5:05 p.m. false-strategy.tex page 1 The False Strategy Th |
| `harris_ocr/` | Market microstructure — stop cascades, adverse selection (Harris) | `docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md` | (subdir corpus — see its README / page files) |
| `Harris_Trading_Exchanges_Market_Microstructure.epub` | Market microstructure — stop cascades, adverse selection (Harris) | `docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md` |  |
| `Harris_Trading_Exchanges_Market_Microstructure.ocr.pdf` | Market microstructure — stop cascades, adverse selection (Harris) | `docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md` | Association Survey and Syn Ox F- RD UNLV EVR ST Y PRE Ss Ss |
| `Harris_Trading_Exchanges_Market_Microstructure.pdf` | Market microstructure — stop cascades, adverse selection (Harris) | `docs/institutional/literature/harris_2002_trading_exchanges_microstructure.md` | Professor Larry Harris Trading and Exchanges Draft Copy ©2002 Oxford University Press Draft: March 1, 2002 i TRADING AND EXCHANGES: Market M |
| `Lopez_de_Prado_ML_for_Asset_Managers.pdf` | ML for Asset Managers — theory-first, CPCV | `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md` | C:/ITOOLS/WMS/CUP-NEW/21557229/WORKINGFOLDER/LOPEZ-ELE/9781108792899PRE.3D i [1–4] 14.3.2020 9:04AM Elements in Quantitative Finance edited  |
| `man_overfitting_2015.pdf` | Backtest overfitting (Man/2015) | — | For investment professionals only. Not for public distribution. M Man OVERFITTING AND ITS IMPACT ON THE INVESTOR MAN AHL ACADEMIC ADVISORY B |
| `ml4am_code_companion/` | ML4AM code companion (Lopez de Prado notebooks) | `docs/institutional/literature/lopez_de_prado_2020_ml_for_asset_managers.md` | (subdir corpus — see its README / page files) |
| `projectx_api_spec_2026_05_16.md` | Broker — ProjectX/TopstepX API spec (live execution) | — | (markdown — read directly) |
| `prop-firm-official-rules.md` | Prop-firm official rules (account death / MLL) | — | (markdown — read directly) |
| `Pseudo-mathematics-and-financial-charlatanism.pdf` | Multiple-testing — MinBTL bound (caps brute-force ≤300) | `docs/institutional/literature/bailey_et_al_2013_pseudo_mathematics.md` | Electronic copy available at: http://ssrn.com/abstract=2308659 PSEUDO-MATHEMATICS AND FINANCIAL CHARLATANISM: THE EFFECTS OF BACKTEST OVERFI |
| `Quantitative_Trading_Chan_2008.pdf` | Intraday sessions / regime (Chan 2008/09) | `docs/institutional/literature/chan_2009_ch1_intraday_session_handling.md` | Wiley Trading E R N E ST P. C H A N How to Build Your Own Algorithmic Trading Business Quantitative Trading |
| `real_time_strategy_monitoring_cusum.pdf` | Live monitoring — Shiryaev-Roberts CUSUM | `docs/institutional/literature/pepelyshev_polunchenko_2015_cusum_sr.md` | Statistics and Its Interface Volume 0 (2015) 1–14 Real-time ﬁnancial surveillance via quickest change-point detection methods Andrey Pepelys |
| `rober prado - optimization of trading strategies.pdf` | Trading-strategy optimization (Pardo) | — | The Evaluation and Optimization of Trading Strategies Second Edition R O B E R T P A R D O John Wiley & Sons, Inc. ~ WILEY |
| `Robert Carver - Systematic Trading.pdf` | Vol targeting / portfolio construction / sizing (Carver) | `docs/institutional/literature/carver_2015_volatility_targeting_position_sizing.md` | ﻿ Systematic Trading Robert Carver worked in the City of London for over a decade. He initially traded exotic derivative products for Barcla |
| `Two_Million_Trading_Strategies_FDR.pdf` | Multiple-testing — t≥3.79 empirical bound | `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` | Anomalies and Multiple Hypothesis Testing: Evidence from Two Million Trading Strategies Tarun Chordia Amit Goyal Alessio Saretto∗ May 2018 A |

## How to ground cheaply

1. Find the row for your topic above.
2. If it has a **curated extract**, cite from that file (page numbers included).
3. If not (`—`), extract the TOC + 3 mid-document pages from the PDF before
   characterizing or dismissing it (terminology differs across sources — see
   institutional-rigor.md § 7 extract-before-dismiss rule).
4. Never cite a resource you have not opened this session as if you read it.

