---
status: active
created: 2026-04-28
purpose: ADHD-friendly single source of truth for repo tooling + git survival
related: docs/specs/crg-integration.md (CRG details), docs/external/code-review-graph/ (verbatim official CRG docs)
---

# Repo + Tooling Cheat Sheet

One page. Read this before you spiral.

---

## 1. GitNexus — PARKED (workarounds exist but not worth the tax)

- npm CLI, two bugs on Windows: (a) segfault on `analyze`, (b) files >32KB silently dropped from graph.
- **Workarounds that would work:**
  - `patch-package` to persist both fixes across `npm install` (~1 hr setup, ongoing maintenance).
  - Or run inside WSL (may sidestep segfault; buffer bug still needs patch).
  - Or fork + fix upstream (~half day, you own maintenance).
- **Why we don't bother:** CRG already works, no patches needed, MIT-licensed, honest about its weaknesses. GitNexus's value-add over CRG was small even when working.
- Re-open trigger: upstream fixes both bugs, OR CRG starts failing on something GitNexus would solve.
- Receipt: `docs/eval/2026-04-28-gitnexus-evaluation.md` (on `tooling/gitnexus-eval` worktree).

---

## 2. code-review-graph (CRG) — KEPT and wired

- Python tool + MCP server, v2.3.2.
- Code knowledge graph: callers, callees, imports, tests-for-symbol, blast radius, semantic search.
- **Not a truth layer.** Truth = canonical code + `gold.db`. CRG = navigation only.
- Graph DB: `.code-review-graph/graph.db` (gitignored, ~175 MB).
- MCP wired in `.mcp.json` (server: `code-review-graph` via `uvx`).
- Spec: `docs/specs/crg-integration.md`.
- Verbatim official docs: `docs/external/code-review-graph/` (11 files).

### Daily slash commands (use in this order)

1. `/crg-context <task>` — minimal context (~80 tokens), reads risk + suggests next tool
2. `/crg-search <free-text>` — semantic/FTS node search. Best tool in 2026-04-28 benchmark.
3. `/crg-blast <file>` — impact radius. Only if step 1 flagged risk medium/high.
4. `/crg-tests <file::symbol>` — affected tests. Fully qualify the symbol.

### Maintenance

```bash
code-review-graph update    # incremental, <2s — after pipeline/ or trading_app/ edits
code-review-graph build     # full rebuild, ~30s — after CRG version bumps
code-review-graph status    # nodes/edges/last-updated sanity check
```

### When NOT to use CRG

- Trading data, fitness, strategy stats → `gold-db` MCP
- Doctrine / thresholds → `docs/institutional/`
- Trivial 1-file edits → just Read + Grep
- "What is true right now?" → canonical code, never the graph

---

## 3. Daily flow (from CLAUDE.md routing)

**Before non-trivial code edit:**
1. `python scripts/tools/context_resolver.py --task "<intent>" --format markdown`
2. `/crg-context <intent>` (read suggested next tool)
3. `/crg-search <symbols>` if needed to locate canonical files
4. Read the canonical files directly

**Before PR / review:**
1. `/crg-context "review current diff"`
2. If risk medium/high: `/crg-blast <each-changed-file>`
3. `/crg-tests <each-touched-public-symbol>` for coverage gaps
4. Run drift + tests (existing pre-commit gauntlet)

**For "what's true right now":**
- Strategies / fitness → `gold-db` MCP
- Sessions / costs / instruments → `pipeline.dst`, `pipeline.cost_model`, `pipeline.asset_configs`
- Doctrine → `docs/institutional/`

---

## 4. Git survival rules

1. `git status` **before** anything.
2. One job per worktree. Don't switch branches mid-task.
3. Commit or stash **before** switching anything.
4. Runtime junk (`live_*.jsonl`, `.stop` files, `.completion-notify-last`) → gitignore, never commit.
5. If confused, stop and `git status` again. The repo is not lying to you.

### Worktrees (current)
| Path | Branch | Job |
|---|---|---|
| `C:/Users/joshd/canompx3` | `research/2026-04-28-phase-d-...` | Phase-D Pathway-B research |
| `.worktrees/canonaudit` | `canonaudit` | Canonical-source audits |
| `.worktrees/cockpit-ledger-20260428` | `design/cockpit-ledger-...` | Cockpit/ledger design |
| `.worktrees/gitnexus-eval` | `tooling/gitnexus-eval` | CRG/GitNexus eval (DONE — keep until merged) |
| `.worktrees/crg-wiring` | `tooling/crg-wiring` | This work — CRG MCP wiring |

---

## 5. Five commands that actually matter

```bash
git status --short                                      # where am I, what's dirty
git worktree list                                       # what branches are open
git log --oneline -5                                    # last 5 commits on this branch
python scripts/tools/context_resolver.py --task "X"     # what to read for task X
code-review-graph update                                # refresh graph after pipeline edits
```

That's it. Anything beyond this — ask first.
