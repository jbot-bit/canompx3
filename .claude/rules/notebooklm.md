# NotebookLM Registry

NotebookLM is for **academic methodology only**. For project-specific knowledge (research findings, architecture, config, memory), use the Pinecone Assistant — see `pinecone-assistant.md`.

Use `mcp__notebooklm__notebook_query` with the notebook ID below. Match user intent to the right tool automatically — never ask the user which notebook.

## Notebooks

| ID | Title | Use When |
|----|-------|----------|
| `ddf58fd0-e0d9-45c0-81ee-0f5e173e3425` | **My Trading Rules** | Session definitions, entry model rules, sample size thresholds, filter logic, cost model — HOW our system works (definitions only, not current state) |
| `8d0d996d-cb8d-4c1f-aba7-b5feb38e586a` | **Backtesting Rules** | BH FDR procedure, Sharpe under multiple testing, deflated Sharpe, walk-forward validation, overfitting, pseudo-mathematics — academic methodology |

## Three-System Routing

See `pinecone-assistant.md` for the full decision framework. Summary:

| System | Scope |
|--------|-------|
| **Pinecone Assistant** | Project knowledge — research findings, architecture, memory, NO-GOs, design decisions |
| **gold-db MCP** | Live data — strategy counts, performance, fitness, trade history |
| **NotebookLM** | Academic methodology — BH FDR math, walk-forward theory, overfitting literature |

## NotebookLM Routing Keywords

- "BH FDR", "walk-forward theory", "overfitting", "deflated Sharpe", "multiple testing", "backtesting methodology", "pseudo-mathematics" → `8d0d996d` (Backtesting Rules)
- "our rules", "entry model definition", "session definition", "filter definition", "cost model definition" → `ddf58fd0` (My Trading Rules)
- "what did we find about X", "research on", "why did we", "remind me" → **Pinecone Assistant** (NOT NotebookLM)
- "how many strategies", "is X still FIT", "current performance" → **gold-db MCP** (NOT NotebookLM)

## Critical Limitation

The "My Trading Rules" notebook is a **snapshot** — strategy counts and regime status go stale after every rebuild. Use it for methodology/definitions only. For current state always use gold-db MCP. For research findings use Pinecone Assistant.

## Auth Note

If `mcp__notebooklm__notebook_query` returns auth error, run `notebooklm-mcp-auth` via Bash before retrying.
