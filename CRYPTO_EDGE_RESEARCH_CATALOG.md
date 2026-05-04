# Crypto Trading Edge Discovery Catalog
**Target: Australian retail traders, $10–50k capital, Python 3.13 stack, non-developer operators**

---

## A) DATA SOURCES

### Exchange APIs (Australia-accessible 2026)
- **Binance API**: [binance.com/binance-api](https://www.binance.com/en/binance-api) — CCXT wrapper or native SDK; Australia not restricted as of 2026
- **Bybit API**: [bybit-exchange.github.io/docs](https://bybit-exchange.github.io/docs/) — V5 unified API (spot, perps, options); full Australia access
- **OKX API**: [okx.com/docs-v5/en](https://www.okx.com/docs-v5/en/) — Requires passphrase at setup; Australia accessible
- **Deribit API**: https://deribit.com/api — Bitcoin/Ethereum options & perps; low fees for quant; unrestricted Australia access
- **Hyperliquid API**: https://hyperliquid.gitbook.io/hyperliquid-docs — DEX perpetuals; non-custodial; free data
- **Kraken API**: https://docs.kraken.com/rest/ — AU-native legal presence; lower feature set than Binance
- **Independent Reserve API**: https://www.independentreserve.com/api — Australia-domiciled exchange; limited instrument depth

### Historical Data & Market Feeds
- **Databento**: [databento.com/pricing](https://databento.com/pricing) — $125/month free credits for CME + crypto; pay-as-you-go overage; tick-level OHLCVT
- **CCXT** (unified exchange abstraction): [docs.ccxt.com](https://docs.ccxt.com/) — Free, open-source; 100+ exchange support; built-in trade/ohlcv normalisation
- **CoinGlass API**: [docs.coinglass.com](https://docs.coinglass.com/) — Free tier: funding rates, OI, liquidations with delay; $12/month for real-time L2/L3 + heatmaps
- **Laevitas**: [laevitas.ch](https://www.laevitas.ch/) — Basis, funding, options Greeks across Binance/Deribit/OKX/Bybit; REST + WebSocket + MCP; crypto-pay friendly
- **CoinGecko API**: https://www.coingecko.com/en/api — Free tier: 10–50 calls/min; spot prices, market cap, 24h change; best for background monitoring
- **CoinMarketCap API**: https://coinmarketcap.com/api/ — Paid ($199+/month); richer metadata than CoinGecko; spot + derivatives tickers

### On-Chain & Market Intelligence
- **Glassnode**: [studio.glassnode.com/pricing](https://studio.glassnode.com/pricing) — Professional plan $700/month for API; on-chain, spot, derivatives data; 120 req/min; alternatives: free tier (daily resolution, delayed)
- **CryptoQuant**: https://www.cryptoquant.com/ — ~$800/month for professional API (300 req/min); short-term flows, derivatives positioning; macro on-chain complement to Glassnode
- **Checkonchain**: [charts.checkonchain.com](https://charts.checkonchain.com/) — 200+ Bitcoin on-chain charts (MVRV, SOPR, supply, mining); educational + free web UI; Orange membership for extra content
- **Nansen**: https://www.nansen.ai/ — Entity tracking (whale wallets, VC addresses, CEX flows); $699+/month; heavy for retail but unmatched transparency
- **Arkham Intelligence**: https://www.arkham.io/ — Wallet clustering + entity labels; free tier limited; $200+/month for API
- **Dune Analytics**: [dune.com/pricing](https://dune.com/pricing) — Free tier: unlimited teammates, free SQL executions, API access; Pro $390/month; 100+ blockchains indexed, 3+ petabytes; custom queries

### Niche / Specialized Feeds
- **TokenUnlocks.com**: https://token.unlocks.com/ — Token unlock schedules & release calendars; free
- **Farside Investors**: https://www.farsideinvestors.com/ — BTC/ETH ETF flows (inflows > on-chain proxy); free web view
- **Artemis Research**: https://www.artemisresearch.org/ — On-chain signals + trade recommendations; niche research community
- **Messari**: https://messari.io/api — Free tier: asset profiles; Pro ($3k+/year): quarterly report data + intelligence API
- **The Graph**: [thegraph.com](https://thegraph.com/) — Decentralised indexing protocol; free GraphQL subgraphs for DeFi data; foundation-backed

---

## B) MONITORING / DASHBOARD TOOLS

### Real-Time Charting & Alerts
- **TradingView**: [tradingview.com](https://www.tradingview.com/) — Pine Script alerts to webhook → Telegram/Discord/Slack; requires pro ($15/month) for webhooks; largest retail user base
  - Tutorial: [Alerts FAQ](https://www.tradingview.com/pine-script-docs/faq/alerts/)
  - Discord Library: [DiscordWebhooksLibrary](https://www.tradingview.com/script/VfDJ2Yo0-DiscordWebhooksLibrary/)
- **CoinGlass Dashboards**: [coinglass.com](https://www.coinglass.com/) — Web-based liquidation heatmaps, funding curves, multi-pair OI; free tier suitable for manual monitoring
- **Kraken Pro "Terminal"**: [kraken.com](https://www.kraken.com/) — Native multi-chart layout; replaces deprecated Cryptowatch; Australia-accessible
- **Alphaday**: https://www.alphaday.com/ — Customizable on-chain + news + portfolio dashboard; free tier available

### Self-Hosted Monitoring
- **Grafana + Prometheus**: [grafana.com](https://grafana.com/) — Time-series dashboards for custom scrapers; Docker-friendly; free + enterprise tiers
- **Telegram/Discord Bots**: [fabston/TradingView-Webhook-Bot](https://github.com/fabston/TradingView-Webhook-Bot) — Flask-based relay (TradingView → Telegram/Discord/Slack/Email); open-source
- **Amberdata**: https://www.amberdata.io/ — Institutional-grade crypto infrastructure monitoring; $2k+/month; overkill for retail

### Deprecated (Do Not Use)
- **Cryptowatch**: Shut down by Kraken 2023; [thecoininvestor.com/cryptowatch](https://thecoininvestor.com/cryptowatch/) for history; replaced by Kraken Pro Terminal or TradingView

---

## C) EXECUTION / BROKERAGE (Australia-Accessible)

### Centralised Exchanges (CEX)
- **Binance**: [binance.com](https://www.binance.com/) — Lowest fees (maker 0.02%, taker 0.05%), deepest liquidity, most pairs, Australia unrestricted API; standard for retail systematic trading
- **Bybit**: [bybit.com](https://www.bybit.com/) — Perp + spot, copy trading, sub-accounts; competitive fees; direct Australian registration; good API stability
- **OKX**: [okx.com](https://www.okx.com/) — Largest derivatives exchange, full options suite, lowest taker (0.02%); mature infrastructure; requires passphrase security
- **Deribit**: [deribit.com](https://www.deribit.com/) — Bitcoin/Ethereum options + perps; institutional options implied vol surfaces; smaller notional volume; professional-grade
- **Hyperliquid**: [hyperliquid.com](https://hyperliquid.com/) — Non-custodial DEX perpetuals; cex-grade matching engine + orderbook; no signup/KYC needed; emerging liquidity

### AU-Native Brokers
- **Independent Reserve**: [independentreserve.com](https://www.independentreserve.com/) — Australia-regulated, AFS licensed; spot-only, lower leverage, smaller pair universe
- **Kraken**: [kraken.com](https://www.kraken.com/) — US-regulated but Australia-accessible; mature API; conservative leverage limits

---

## D) BACKTESTING / RESEARCH FRAMEWORKS (Python, Open-Source, Free)

### Vectorised (Fast, Large-Scale)
- **VectorBT**: [github.com/polakowo/vectorbt](https://github.com/polakowo/vectorbt) — Numba-accelerated pandas/numpy backtesting; 1000s of strategies in seconds; signal-based; portfolio optimisation; free tier on [vectorbt.dev](https://vectorbt.dev/); Pro version has advanced features
- **Backtrader**: [github.com/mementum/backtrader](https://github.com/mementum/backtrader) — Classic event-driven; slower than VectorBT but more flexible for complex logic; large community; free
- **pyfolio**: [github.com/quantopian/pyfolio](https://github.com/quantopian/pyfolio) — Returns analysis post-backtest (Sharpe, drawdown, rolling window analysis); pairs with Zipline; free

### Event-Driven (Realistic Simulation)
- **Zipline-Reloaded**: [github.com/stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded) — Quantopian's engine, revived 2021; Pipeline API for factor definition; numpy 2.0 compatible as of v3.05; free, equity-focused
- **Nautilus Trader**: [nautilustrader.io](https://nautilustrader.io/) — Rust core, Python API; multi-venue, multi-asset; crypto + traditional; nanosecond resolution backtesting; excellent for RL; free on [GitHub](https://github.com/nautechsystems/nautilus_trader)
- **pysystemtrade**: [github.com/pst-group/pysystemtrade](https://github.com/pst-group/pysystemtrade) — Robert Carver's own system (recommended in CLAUDE.md); production-grade; 2026 move to pst-group org; free, MIT license

### Crypto-Native Frameworks
- **freqtrade**: [github.com/freqtrade/freqtrade](https://github.com/freqtrade/freqtrade) — Free crypto bot; backtesting, hyperopt (Optuna), live trading; 100% CCXT support; large strategy ecosystem; 28k+ GitHub stars; active development
- **Jesse**: [jesse.trade](https://jesse.trade/) — High-level crypto framework; multi-timeframe, multi-symbol; built-in ML pipeline; Monte Carlo; free on [GitHub](https://github.com/jesse-ai/jesse); beginner-friendly docs
- **Hummingbot**: [hummingbot.org](https://hummingbot.org/) — Market-making focus; backtesting + live trading; 14k+ stars; free; strategies on [GitHub](https://github.com/hummingbot/hummingbot); suited for execution research
- **OctoBot**: [github.com/OctoBot-Online/OctoBot](https://github.com/OctoBot-Online/OctoBot) — Web UI backtester; strategy marketplace; free tier + cloud option

### Cloud/Institutional
- **QuantConnect (Lean)**: [quantconnect.com](https://www.quantconnect.com/) — Cloud backtesting (C# or Python); live paper trading; free tier (1M RAM, 20 min/month compute); $99+/month for pro; institutional integration
- **Superalgos**: [github.com/Superalgos/Superalgos](https://github.com/Superalgos/Superalgos) — No-code/low-code visual strategy builder; backtesting + paper trading; free on GitHub

---

## E) NOTABLE OPEN-SOURCE CRYPTO TRADING REPOS (GitHub stars, last commit, quality)

| Repo | Stars | Last Commit | Quality Note |
|------|-------|-------------|--------------|
| [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade) | 28k+ | Active (2026) | Crypto-specific; strong community; battle-tested for $10–100k retail |
| [jesse-ai/jesse](https://github.com/jesse-ai/jesse) | 5k+ | Active | Best-in-class docs for non-quants; educational |
| [hummingbot/hummingbot](https://github.com/hummingbot/hummingbot) | 14k+ | Active | Market-making specialist; good for execution research |
| [robcarver17/pysystemtrade](https://github.com/robcarver17/pysystemtrade) → [pst-group/pysystemtrade](https://github.com/pst-group/pysystemtrade) | 2.5k | Active (moved 2026-01) | Production-grade; Carver's own; gold standard for systematic trading |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | 2k+ | Active | Rust-native, Python API; niche but powerful for RL + multi-venue |
| [quantopian/zipline](https://github.com/quantopian/zipline) (deprecated) | 11k | 2020 | Replaced by [stefan-jansen/zipline-reloaded](https://github.com/stefan-jansen/zipline-reloaded); use Reloaded instead |
| [polakowo/vectorbt](https://github.com/polakowo/vectorbt) | 4k+ | Active | Fastest vectorised backtest; best for parameter sweeps; strong for statistical testing |
| [mementum/backtrader](https://github.com/mementum/backtrader) | 13k+ | Maintenance mode | Classic but slower; large ecosystem; good for educational ramps |
| [quantopian/pyfolio](https://github.com/quantopian/pyfolio) | 5.5k | Maintenance | Post-backtest analysis (Sharpe, drawdown, calmar ratio); pairs with Zipline |
| [empyrical/empyrical](https://github.com/empyrical/empyrical) | 1.5k | Active | Risk metrics computation; complement to backtester output |

**Research & Arbitrage Repos**:
- [Laevitas derivatives data scraper examples](https://app.laevitas.ch/dashApi) — basis arb, funding rate tracking  
- [CoinGlass GitHub reference](https://github.com/jrxmnq629/CoinGlass) — liquidation map + OI analysis

---

## F) AI/LLM TOOLS FOR TRADING RESEARCH

### AI-Native Code Development
- **Claude Code** (current session): [claude.ai](https://claude.ai/) — Multimodal analysis, literature research, code generation; $20/month Pro or API; strongest for research synthesis and prototype building
- **Cursor IDE**: [cursor.com](https://cursor.com/) — AI-first code editor; Composer 2 agents; supports Claude Opus 4.6, GPT-5.2, Gemini 3 Pro, Grok Code; best for iterative system building; free tier + $20/month Pro
- **Aider** (open-source): [github.com/Aider-AI/aider](https://github.com/Aider-AI/aider) — Terminal-based AI pair programmer; git-native; CLI-friendly for non-IDE users; free

### NLP & Sentiment Analysis
- **FinBERT**: [github.com/ProsusAI/finBERT](https://github.com/ProsusAI/finBERT) — Pre-trained BERT for financial sentiment; 97% test accuracy on Financial PhraseBank; free on [Hugging Face](https://huggingface.co/ProsusAI/finbert)
- **CryptoBERT**: [model.aibase.com](https://model.aibase.com/models/details/1915694188903161857) — Fine-tuned FinBERT for crypto text; free on Hugging Face; superior to FinBERT on crypto news/social media
- **VADER** (open-source): https://github.com/cjhutto/vaderSentiment — Lightweight rule-based sentiment; good baseline, no training needed; free

### Agent Orchestration (Caveat: Hype-Heavy)
- **LangChain**: [langchain.com](https://www.langchain.com/) — Chains, agents, memory; integrates Claude/GPT/open models; free; heavy on abstraction; use with caution (research-grade only, not prod-stable)
- **AutoGen (Microsoft)**: [microsoft.github.io/autogen](https://microsoft.github.io/autogen/) — Multi-agent conversations; free; overkill for most retail tasks

### News & Data Ingestion
- **Perplexity API**: https://docs.perplexity.ai/ — Real-time web search + LLM synthesis; good for on-demand market news; $0.02–0.20 per query
- **Exa**: [exa.ai](https://exa.ai/) — Neural search API; built for AI agents; $0.03–0.10 per search; integrates with LangChain

---

## Key Constraints (From Project Metadata)

| Metric | Value |
|--------|-------|
| Capital | $10k test, $30–50k if proven |
| Jurisdiction | Australia (unrestricted exchange access) |
| Infrastructure | Windows dev + VPS prod |
| Operator | Non-developer; one-command, clear outputs |
| Monthly infra cost | $73–148 (VPS + Claude API + tax) |
| Round-trip taker fee | 0.30% (spot + perp) |
| Round-trip maker fee | ~0.08% |
| Funding arb breakeven | 48h (taker), 13h (maker) at 0.05%/8h |

---

## Quick-Start Recommendations

**For Backtesting Strategy Research**:  
1. VectorBT (fast parameter sweeps) or pysystemtrade (production mindset)
2. Databento + CCXT for data pipeline
3. TradingView for live charting + alerts

**For On-Chain Signal Mining**:  
1. Glassnode (professional) or Dune (free SQL) for macro flows
2. Checkonchain web UI for BTC cycles
3. Laevitas for derivatives positioning

**For Live Execution**:  
1. Binance (primary), OKX (secondary), Deribit (options/research)
2. freqtrade or Hummingbot for automation
3. Cursor for rapid iteration on new strategies

**For Edge Discovery (non-technical operator)**:  
1. Claude Code + TradingView Pine Script
2. Pre-registered research in Dune or Jupyter
3. Daily monitoring: CoinGlass + Checkonchain web UIs + Telegram alerts

---

**Total estimated research tooling cost (Monthly, AUD)**:  
- Free tier only: $0  
- Modest ($10–50k capital): CoinGlass Pro ($12) + TradingView Pro ($15) + Claude API (~$30–60) = ~$60–90  
- Serious (>$50k): Add Glassnode ($700/month) or CryptoQuant ($800/month) → ~$760–890

**Last Updated**: 2026-04-15  
**Jurisdiction**: Australia (full exchange API access verified)  
**Stack**: Python 3.13 + pandas + DuckDB + CCXT + Databento
