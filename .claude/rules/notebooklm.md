# NotebookLM Registry

Use `mcp__notebooklm__notebook_query` with the notebook ID below. Match user intent to the right tool automatically — never ask the user which notebook.

## Notebooks

| ID | Title | Use When |
|----|-------|----------|
| `ddf58fd0-e0d9-45c0-81ee-0f5e173e3425` | **My Trading Rules** | Our session definitions, entry model rules, sample size thresholds, NO-GOs, filter logic, cost model, AVOID signals — anything about HOW our system works |
| `8d0d996d-cb8d-4c1f-aba7-b5feb38e586a` | **Backtesting Rules** | BH FDR procedure, Sharpe under multiple testing, deflated Sharpe, walk-forward validation, overfitting, pseudo-mathematics, why backtests fail — academic methodology backing |

## Decision Framework — Notebook vs MCP

NEVER use notebooks for live/current data. ALWAYS use gold-db MCP for that.

| Question | Go To |
|----------|-------|
| "What's our sample size rule?" | My Trading Rules notebook |
| "What entry models are active?" | My Trading Rules notebook |
| "What sessions are valid for MGC?" | My Trading Rules notebook |
| "What does BH FDR do step by step?" | Backtesting Rules notebook |
| "How many walk-forward windows do we need?" | Backtesting Rules notebook |
| "Why does the size filter work?" | My Trading Rules notebook |
| "What does the research say about overfitting?" | Backtesting Rules notebook |
| "How many MNQ strategies right now?" | gold-db MCP (`get_strategy_fitness`) |
| "Is strategy X still FIT?" | gold-db MCP (`get_strategy_fitness`) |
| "Show me strategy performance" | gold-db MCP (`query_trading_db`) |
| "Current strategy counts" | gold-db MCP (`query_trading_db`) |

## Routing Rules

- "our rules", "our system", "entry model", "session", "filter", "cost model", "sample size", "NO-GO" → `ddf58fd0` (My Trading Rules)
- "BH FDR", "walk-forward", "overfitting", "deflated Sharpe", "multiple testing", "backtesting methodology", "pseudo-mathematics" → `8d0d996d` (Backtesting Rules)
- "how many strategies", "is X still working", "regime", "FIT/WATCH/DECAY", "current performance" → gold-db MCP, NOT notebooks
- When a question spans both methodology AND our rules → query both notebooks

## Critical Limitation

The "My Trading Rules" notebook is a **snapshot** — strategy counts and regime status go stale after every rebuild. Use it for methodology/definitions only. For current state always use the MCP.

## Auth Note

If `mcp__notebooklm__notebook_query` returns auth error, run `notebooklm-mcp-auth` via Bash before retrying.

## Sync Note

Re-sync "My Trading Rules" after: major rebuilds, new sessions added, new NO-GOs confirmed, entry model changes. Delete old text sources and re-add updated file content.
