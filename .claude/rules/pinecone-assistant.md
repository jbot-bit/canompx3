# Pinecone Assistant (orb-research)

The `orb-research` Pinecone Assistant is the project's knowledge base — 51 files across 5 tiers covering authority docs, config, research findings, and memory. Use `/pinecone:assistant-chat` or the Pinecone Assistant MCP tools to query it.

## Two-System Routing

Route every knowledge question to the RIGHT system. Zero overlap. Academic methodology is served from local PDFs (see `notebooklm.md`).

| System | What It Knows | When to Use |
|--------|--------------|-------------|
| **Pinecone Assistant** | Project knowledge — research findings, architecture, memory, config, plans, NO-GOs, design decisions | "What did we find about X?", "Why did we do Y?", "What's the history of Z?" |
| **gold-db MCP** | Live structured data — strategy counts, performance, fitness regimes, trade history, schema | "How many strategies?", "Is X still FIT?", "Show performance for MNQ" |

Academic methodology (BH FDR, walk-forward, deflated Sharpe, overfitting) → local PDFs in `resources/`. See `notebooklm.md` for the topic→PDF mapping.

## Decision Framework

| User Intent | Route To |
|-------------|----------|
| "What research have we done on session X?" | Pinecone |
| "Why is E0 dead?" | Pinecone |
| "What's the compressed spring signal?" | Pinecone |
| "How does our pipeline work?" | Pinecone |
| "What NO-GOs have we confirmed?" | Pinecone |
| "What did we find about break quality?" | Pinecone |
| "Remind me about the M2K findings" | Pinecone |
| "What's in live_config?" | Pinecone (generated snapshot) |
| "How many MNQ strategies right now?" | gold-db MCP |
| "Is strategy X still FIT?" | gold-db MCP |
| "Show trade history for strategy Y" | gold-db MCP |
| "Current strategy counts by instrument" | gold-db MCP |
| "What does BH FDR do step by step?" | Local PDF (`resources/benjamini-and-Hochberg-1995-fdr.pdf`) |
| "Walk-forward validation theory" | Local PDF (`resources/man_overfitting_2015.pdf`) |
| "Deflated Sharpe ratio math" | Local PDF (`resources/deflated-sharpe.pdf`) |

## Keyword Routing

- "what did we find", "research on", "history of", "why did we", "remind me", "what's the story" → **Pinecone**
- "how many", "current", "right now", "still FIT", "performance", "trade history" → **gold-db MCP**
- "BH FDR", "walk-forward theory", "deflated Sharpe", "academic", "methodology paper" → **Local PDFs** (see `notebooklm.md`)
- Ambiguous (spans project + methodology) → query Pinecone first, local PDFs if needed

## Content Tiers in Pinecone

| Tier | Files | What | Update Frequency |
|------|-------|------|-----------------|
| **static** | 14 | Authority docs (TRADING_RULES, RESEARCH_RULES, guardians, audits) | On change |
| **living** | 4 | Config files as markdown (config.py, dst.py, cost_model.py, live_config.py) | Every sync |
| **memory** | ~20 | Claude memory files (research findings, instrument analysis, audit notes) | Every sync |
| **research_output** | 9 | Bundled research results (82 files → 9 topic bundles) | Every sync |
| **generated** | 4 | Snapshots from gold-db (portfolio state, fitness, live config, research index) | Every sync |

## Sync Commands

```bash
# Sync only (delta upload — fast)
python scripts/tools/sync_pinecone.py

# Dry run (see what would upload)
python scripts/tools/sync_pinecone.py --dry-run

# Force re-upload everything
python scripts/tools/sync_pinecone.py --force

# Full rebuild chain + sync
bash scripts/tools/run_rebuild_with_sync.sh MGC
```

## When to Sync

- After any rebuild chain (`outcome_builder` → `strategy_validator` → `build_edge_families`)
- After updating authority docs (TRADING_RULES.md, RESEARCH_RULES.md, etc.)
- After new research findings are written to `research/output/`
- After updating memory files
- After modifying config.py, dst.py, cost_model.py, or live_config.py

## NEVER Do This

- Query Pinecone for live strategy counts or performance (use gold-db MCP)
- Query gold-db MCP for "why did we make this decision" (use Pinecone)
- Try to call NotebookLM MCP tools — they are retired; use local PDFs
- Skip sync after a rebuild — generated snapshots go stale
