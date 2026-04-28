---
status: active
created: 2026-04-28
purpose: ADHD-friendly single source of truth for repo tooling + git survival
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

## 2. code-review-graph (CRG) — KEPT, used for navigation only

- Python tool + MCP server. v2.3.2.
- Builds a code knowledge graph: callers, callees, imports, tests-for-symbol, blast radius.
- **Not a source of truth.** It's a faster grep with edges. Truth = canonical code + `gold.db`.
- DB lives at `.code-review-graph/` (gitignored).
- Wired in `.mcp.json` on `tooling/gitnexus-eval` branch — not yet on main.

### When to use CRG
- Before non-trivial code edit: ask "what calls this?" / "what tests cover this?"
- Before PR review: blast-radius / impact-radius on the diff.
- Big audits where grep would flood your context.

### When NOT to use CRG
- For trading data, strategy stats, fitness → use `gold-db` MCP.
- For doctrine / thresholds → read `docs/institutional/` directly.
- For trivial 1-file edits → just Read + Grep.

### Rebuild after pipeline/trading_app edits
```bash
code-review-graph build
```

---

## 3. Daily flow

**Before code edit:**
1. `python scripts/tools/context_resolver.py --task "<intent>" --format markdown`
2. (optional) CRG `get_minimal_context_tool` for the symbol you're touching
3. Read canonical files

**Before PR / review:**
1. CRG `get_impact_radius_tool` on changed files
2. Run drift + tests
3. Self-review against `.claude/rules/institutional-rigor.md`

**For "what's true right now":**
- Strategies / fitness: `gold-db` MCP
- Sessions / costs / instruments: `pipeline.dst`, `pipeline.cost_model`, `pipeline.asset_configs`
- Doctrine: `docs/institutional/`
- **Never CRG. Never docs. Never memory.**

---

## 4. Git survival rules

1. `git status` **before** anything.
2. One job per worktree. Don't switch branches mid-task.
3. Commit or stash **before** switching anything.
4. Runtime junk (`live_*.jsonl`, `.stop` files, `.completion-notify-last`) → gitignore, never commit.
5. If confused, stop and `git status` again. The repo is not lying to you.

### Worktrees you have right now
| Path | Branch | Job |
|---|---|---|
| `C:/Users/joshd/canompx3` | `research/2026-04-28-phase-d-...` | Phase-D Pathway-B research |
| `.worktrees/canonaudit` | `canonaudit` | Canonical-source audits |
| `.worktrees/cockpit-ledger-20260428` | `design/cockpit-ledger-...` | Cockpit/ledger design |
| `.worktrees/gitnexus-eval` | `tooling/gitnexus-eval` | CRG/GitNexus eval (DONE — merge or close) |

---

## 5. Five commands that actually matter

```bash
git status --short                                      # where am I, what's dirty
git worktree list                                       # what branches are open
git log --oneline -5                                    # last 5 commits on this branch
python scripts/tools/context_resolver.py --task "X"     # what to read for task X
code-review-graph build                                 # refresh CRG after pipeline edits
```

That's it. Anything beyond this — ask first.
