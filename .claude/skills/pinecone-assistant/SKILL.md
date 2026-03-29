---
name: pinecone-assistant
description: >
  Pinecone knowledge base routing for project history and research findings.
  Use when asked: "what did we find about X", "why did we do Y", "research on",
  "history of", "remind me", "what's the story", "NO-GOs", design decisions.
  Routes between Pinecone (project knowledge) and gold-db MCP (live data).
---
# Pinecone Assistant (orb-research)

The `orb-research` Pinecone Assistant is the project knowledge base — 51 files across authority docs, config, research findings, and memory.

## Two-System Routing

| System | What It Knows | When |
|--------|--------------|------|
| **Pinecone** | Research findings, architecture, memory, config, plans, NO-GOs, design decisions | "What did we find?", "Why did we do Y?", "History of Z" |
| **gold-db MCP** | Live data — strategy counts, performance, fitness, trade history | "How many?", "Is X FIT?", "Show performance" |
| **Local PDFs** | Academic methodology (BH FDR, walk-forward, deflated Sharpe) | See `notebooklm.md` for topic-PDF mapping |

## Keyword Routing

- "what did we find", "research on", "history of", "why did we", "remind me" → **Pinecone**
- "how many", "current", "right now", "still FIT", "performance" → **gold-db MCP**
- "BH FDR", "walk-forward theory", "deflated Sharpe" → **Local PDFs** (`resources/`)

## Sync

After rebuilds, authority doc changes, or new research: `python scripts/tools/sync_pinecone.py`
Dry run: `--dry-run`. Force: `--force`.

## Rules

- NEVER query Pinecone for live strategy counts (use gold-db MCP)
- NEVER query gold-db for "why did we make this decision" (use Pinecone)
- NotebookLM MCP is retired — use local PDFs in `resources/`
