---
status: active
created: 2026-04-28
purpose: Permanent integration spec for code-review-graph (CRG) — what we adopt, what we opted out of, why, and how to use it correctly.
canonical_source: docs/external/code-review-graph/  (verbatim official docs, fetched 2026-04-28)
---

# code-review-graph integration spec

## Authority chain

- **Truth:** canonical code (`pipeline/`, `trading_app/`) + `gold.db` canonical layers.
- **Navigation aid:** code-review-graph (CRG). NOT a truth layer.
- **Doctrine:** CLAUDE.md + `docs/institutional/`. CRG never overrides doctrine.
- **Conflict rule:** if CRG output disagrees with `check_drift.py` or canonical code, canonical wins. CRG is structural-only.

## What we adopted

| Component | Where | Source |
|---|---|---|
| MCP server | `.mcp.json` entry `code-review-graph` (uvx-based, official template) | `code-review-graph install --platform claude-code --no-skills --no-hooks --no-instructions -y` |
| Ignore patterns | `.code-review-graphignore` | Hand-curated (research artifacts, generated docs, binary data) |
| Slash commands | `.claude/commands/crg-{context,search,blast,tests}.md` | Custom — wrap MCP tools with token-disciplined fallbacks |
| Verbatim official docs | `docs/external/code-review-graph/` (11 files, 110 KB) | Fetched 2026-04-28 from `github.com/tirth8205/code-review-graph` |

## What we opted out of (and why)

Per official `code-review-graph install` flags + project-fit reasoning:

| Opted-out | Flag | Reason |
|---|---|---|
| Auto-installed Claude Code skills (`build-graph`, `review-delta`, `review-pr`) | `--no-skills` | Generic; clash with our domain-aware `/orient`, `/next`, `/code-review`, etc. |
| Auto-installed PreToolUse/PostToolUse hooks | `--no-hooks` | Multiplies our 8-step pre-commit gauntlet; unmeasured perf cost |
| Auto-injected CLAUDE.md instructions | `--no-instructions` | CLAUDE.md is finely tuned per audit history; no third-party edits |

These are **first-class supported flags** in `code-review-graph install`, not workarounds.

## Tool ladder (mandatory order)

Per official `LLM-OPTIMIZED-REFERENCE.md` §usage:

1. `/crg-context <task>` — minimal-context entry, ~80 tokens. Returns risk + suggested next tools.
2. `/crg-search <query>` — semantic/FTS lookup. **Standout tool** in 2026-04-28 benchmark.
3. `/crg-blast <file>` — impact radius. ONLY if context flagged risk medium/high.
4. `/crg-tests <symbol>` — TESTED_BY edges. Always pass fully-qualified `file::symbol`.

When invoking MCP tools directly, **always pass `detail_level="minimal"`** — official docs claim 90% token savings. Verified: raw `get_impact_radius` Python output for one file = 560 KB JSON.

## Verified limitations (2026-04-28 benchmark, 6 real repo tasks)

| Limitation | Evidence | Workaround |
|---|---|---|
| `tests_for` returns 0 on file paths | `pipeline/cost_model.py` → 0 hits | Use `file_summary` first to enumerate symbols, then `tests_for` per symbol |
| `importers_of` returns duplicates (one row per import statement) | 154 hits for cost_model included 3-7 dups per file | Dedupe by file_path in wrapper |
| Blast radius is polluted by hooks/tests at depth=2 | strategy_discovery top-10 files all `.claude/hooks/*` | Filter to `pipeline/`, `trading_app/`, `research/` for code review use |
| No DataFrame column-access edges | `df['rel_vol']` reads invisible | Keep `pipeline/check_drift.py` AST scans authoritative for column-level data flow (E2 look-ahead pattern) |
| Ambiguous-name resolution requires user qualifier | `orb_utc_window` matched 3 nodes | Wrapper surfaces candidates list cleanly |
| Schema migrations wipe graph data | v1→v9 migration zeroed nodes between sessions | After CRG version bumps: `code-review-graph build` |

## Daily workflow

**Before code edit (non-trivial):**
1. `python scripts/tools/context_resolver.py --task "<intent>"` (existing front door)
2. `/crg-context <intent>` — read suggested next tools
3. If suggested: `/crg-search <symbols>` to locate canonical files
4. Read canonical files directly (Read tool, not CRG)

**Before PR/review:**
1. `/crg-context "review current diff"`
2. If risk medium/high: `/crg-blast <each-changed-file>`
3. `/crg-tests <each-touched-public-symbol>` for coverage gaps
4. Run drift + tests (existing pre-commit gauntlet)

**For "what's true right now":**
- Strategy/fitness/trades → `gold-db` MCP, never CRG
- Sessions/costs/instruments → `pipeline.dst`, `pipeline.cost_model`, `pipeline.asset_configs`
- Doctrine → `docs/institutional/`
- **CRG only answers "where is the code?", never "what is true?"**

## Maintenance

| When | What |
|---|---|
| After heavy `pipeline/` or `trading_app/` edits | `code-review-graph update` (incremental, <2s) |
| After CRG version bump (pip upgrade) | `code-review-graph build` (full rebuild, ~30s) |
| After branch switch with material code differences | `code-review-graph update` |
| Before review of unfamiliar PR | `code-review-graph update` to ensure graph reflects PR head |
| Periodic | `code-review-graph status` to verify Nodes>0 and Last-updated is recent |

## Future work (parked, not scope of this PR)

- **Drift check #N:** `check_crg_freshness` advisory in `pipeline/check_drift.py` — emits warning if graph older than HEAD. Non-blocking.
- **Embeddings:** `pip install code-review-graph[embeddings]` if hybrid FTS hits become insufficient. Don't enable without measuring.
- **Daemon mode:** `crg-daemon` for multi-repo watch — only if we add second repo (e.g. `chatgpt_bundle`).
- **Wiki generation:** `code-review-graph wiki` may complement `REPO_MAP.md`. Defer.
- **Auto-update PostToolUse hook:** officially supported; we declined to avoid hook chain bloat. Reconsider if `update` cost is measured <100ms.

## Re-evaluate-CRG triggers

Open this spec for revision if any of the following holds for >2 weeks:
- Graph repeatedly stale (`update` skipped, queries return wrong files)
- Wrapper output exceeds 1 KB on common queries (token discipline regressed)
- New CRG release breaks the install flags above
- Better tool ships (e.g. GitNexus fixes both Windows bugs upstream — see eval doc)
