---
paths:
  - "resources/**"
---
# Academic Methodology — Local Resources

NotebookLM MCP has been retired due to recurring auth/connectivity failures. Academic methodology is now served from local PDFs in `resources/`.

## Routing (Two Systems, Not Three)

| System | Scope |
|--------|-------|
| **Pinecone Assistant** | Project knowledge — research findings, architecture, memory, NO-GOs, design decisions |
| **gold-db MCP** | Live data — strategy counts, performance, fitness, trade history |
| **Local PDFs** | Academic methodology — BH FDR math, walk-forward theory, overfitting literature |

## Academic Methodology → Local PDFs

When the user asks about methodology, read the relevant PDF from `resources/`:

| Topic | PDF |
|-------|-----|
| BH FDR procedure | `resources/benjamini-and-Hochberg-1995-fdr.pdf` |
| Deflated Sharpe ratio | `resources/deflated-sharpe.pdf` |
| Multiple testing / false strategies | `resources/Two_Million_Trading_Strategies_FDR.pdf`, `resources/false-strategy-lopez.pdf` |
| Walk-forward / overfitting | `resources/man_overfitting_2015.pdf`, `resources/Building_Reliable_Trading_Systems.pdf` |
| Pseudo-mathematics | `resources/Pseudo-mathematics-and-financial-charlatanism.pdf` |
| Lopez de Prado ML | `resources/Lopez_de_Prado_ML_for_Asset_Managers.pdf` |
| CUSUM monitoring | `resources/real_time_strategy_monitoring_cusum.pdf` |
| Systematic Trading (Carver) | `resources/Robert Carver - Systematic Trading.pdf` |
| Quantitative Trading (Chan) | `resources/Quantitative_Trading_Chan_2008.pdf`, `resources/Algorithmic_Trading_Chan.pdf` |
| Evidence-Based TA (Aronson) | `resources/Evidence_Based_Technical_Analysis_Aronson.pdf` |
| Harvey/Liu/Zhu backtesting | `resources/backtesting_dukepeople_liu.pdf` |
| Strategy optimization | `resources/rober prado - optimization of trading strategies.pdf` |

## Routing Keywords

- "BH FDR", "walk-forward theory", "overfitting", "deflated Sharpe", "multiple testing", "backtesting methodology", "pseudo-mathematics" → **Local PDFs** (see table above)
- "our rules", "entry model definition", "session definition" → `TRADING_RULES.md` (already loaded via CLAUDE.md)
- "what did we find about X", "research on", "why did we" → **Pinecone Assistant**
- "how many strategies", "is X still FIT", "current performance" → **gold-db MCP**

## NEVER Do This
- Query Pinecone for live strategy counts or performance (use gold-db MCP)
- Query gold-db MCP for "why did we make this decision" (use Pinecone)
- Try to call NotebookLM MCP tools — they are retired
