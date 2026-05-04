# Code Knowledge Graph MCP — Evaluation 2026-04-28

**Verdict:** Adopt **`code-review-graph`** (not GitNexus). Wire as MCP-only; do **not** install their auto-skills/auto-hooks; keep our domain-aware tooling primary.

## TL;DR

The user identified GitNexus as something they'd been trying to build — an MCP-native code knowledge graph for Claude Code. I evaluated it, hit a known unfixed Windows segfault (issue #1136, filed same day), researched alternatives, and found a strictly better fit for our stack: **`code-review-graph`** (PyPI: `code-review-graph`, MIT, Python-native, SQLite-backed, FastMCP).

It built a real graph for canompx3 — **1040 files, 13,798 nodes, 153,898 edges** (109k CALLS, 20k TESTED_BY, 12k CONTAINS, 7.5k IMPORTS_FROM, 121 INHERITS) — in one pass with zero errors. The MCP server boots cleanly. It complements (does not replace) our existing tooling.

## What was tested

| Step | Result |
| --- | --- |
| Worktree off `origin/main` → `tooling/gitnexus-eval` | OK |
| Read GitNexus source (architecture, Python parser depth, plugin) | Real product. CLI + MCP, tree-sitter native, LadybugDB, PolyForm Noncommercial, 16 MCP tools, dedicated Python extractors. |
| `npm install -g gitnexus@1.6.3` (Node 24.13, npm 11.6, Win 11) | Installed with EPERM cleanup (non-blocking). |
| `gitnexus analyze` on canompx3 | "scope extraction failed: Invalid argument" on every Python file >32KB (= every important file: `check_drift.py`, `config.py`, `build_daily_features.py`, etc.). Index never persisted. |
| `gitnexus analyze` on a 4-function trivial Python repo | **Segfault.** Exit 139. |
| Researched issue #1136 | Confirmed: `loadVectorExtension('LOAD EXTENSION VECTOR')` crashes on Windows. Filed 2026-04-28 (today). No fix yet. Workaround: hand-patch `node_modules/gitnexus/dist/core/lbug/lbug-adapter.js` to early-return on `process.platform === 'win32'`. Survives only until next `npm install`. |
| Researched secondary bug | Files >~32KB fail scope extraction due to missing `bufferSize`. Even if segfault patched, all our canonical files would silently fall out of the graph. **Disqualifying.** |
| Tried older versions (1.5.0, 1.4.0) | 1.5.0 broken on `gitnexus-shared` workspace dep. 1.4.0 fails to build `tree-sitter-kotlin` native binding on Node 24 (issue #1094). All paths blocked. |
| `npm uninstall -g gitnexus` | OK. |
| Pivoted to alternative: `code-review-graph` | Found via web search. MIT-licensed, Python-native, SQLite-backed, FastMCP. |
| `pip install code-review-graph` | OK. Clean install. |
| `code-review-graph build` on `/tmp/gitnexus-smoketest` | OK. 1 file → 4 nodes, 6 edges. |
| `code-review-graph build` on canompx3 | **OK.** 1040 files → 13,798 nodes, 153,898 edges. Zero errors. ~30s. |
| `code-review-graph serve` (MCP) | Boots (FastMCP). |
| Bake-off: FTS search for `rel_vol`, `break_bar`, `break_delay` | Found resolved-symbol hits in actual offending research scripts (`mnq_live_context_overlays_v1.py:191`, `phase_d_d0_backtest.py:175`, `pr48_mes_mgc_sizer_rule_backtest_v1.py:164`, etc.) AND test functions documenting E2 boundaries (`test_rule13_pressure_test_e2_break_bar_features_blocked`). |

## Honest gaps (`code-review-graph` is not magic)

- **No DataFrame column-access edges.** `kind=REFERENCES` is only 48 edges. Pandas `df['rel_vol']` reads are not symbol-resolvable from static analysis. The E2 look-ahead pattern *at the column level* will not be visible in this graph. **Our existing AST-based drift checks in `check_drift.py` remain the primary defense for column-level data-flow bugs.** The graph adds a layer ABOVE that — symbol/call/test coverage — that we currently grep for.
- **Their own published metrics (from their README):** MRR ≈ 0.35 on search; flow recall ≈ 33%, most reliable on Python; "over-predicts in some cases" on impact analysis. **They publish weaknesses honestly** — better cultural fit for our "evidence over assertion" rule than GitNexus's marketing-first README.
- **Single-file change overhead can exceed naive grep.** For trivial edits, prefer Read+Grep. Use the graph for blast-radius / impact-analysis / "what tests cover X" / "what calls Y from outside this module".

## Overlap map vs our stack

| code-review-graph capability | Our equivalent | Decision |
| --- | --- | --- |
| 28 MCP tools (callers, callees, references, tests, impact, flows, communities, FTS, etc.) | `blast-radius.md` agent (grep-based) | **Both stay.** code-review-graph answers structural questions with resolved edges; blast-radius adds canonical-source / DB / domain logic checks the graph can't see. Blast-radius prompt should be updated to "call graph MCP first, then layer domain checks." |
| `serve` MCP stdio | None similar | Pure addition. |
| `detect-changes --base` (git-diff impact) | None — `verify-complete` runs tests but not graph-diff | Pure addition. Routes well into pre-commit gauntlet. |
| `wiki` (markdown wiki from communities) | `REPO_MAP.md` (manual generator) | **No conflict.** Their wiki is symbol-clusters; ours is file inventory. Different views. |
| `watch` (auto-update on file changes) | Not adopting yet — disk thrash risk during heavy research scans. | **Park.** |
| Their auto-installed Claude skills | `quant-debug`, `code-review`, `discover`, etc. | **Don't install theirs (`--no-skills`).** Generic vs domain-aware. |
| Their auto-installed PreToolUse hooks | `data-first-guard.py`, `bias-grounding-guard.py`, etc. | **Don't install theirs (`--no-hooks`).** Don't multiply hook-on-hook latency. |
| Auto-injection into CLAUDE.md | Our CLAUDE.md is finely tuned | **Don't inject (`--no-instructions`).** Document the tools manually if needed. |
| `context_resolver.py` (task → doctrine routing) | — | **Keep ours; no overlap.** Graph knows code, not your trading rules / institutional doctrine. |
| `check_drift.py` (62 drift rules) | — | **Keep ours; no overlap.** Domain-specific (E2 look-ahead AST scan, Mode A holdout, scratch DB ban, daily_features triple-join). |
| `gold-db` MCP server | — | **Keep ours; no overlap.** Trading data, not code structure. |

**Net assessment:** None of our domain tooling is redundant. code-review-graph adds the resolved-symbol call/import/test-coverage graph layer we don't currently have. Right integration is **MCP-only**, no auto-skills, no auto-hooks, no auto-CLAUDE.md.

## What landed on this branch (`tooling/gitnexus-eval`)

- `docs/eval/2026-04-28-gitnexus-evaluation.md` (this file)
- `.mcp.json` updated — added `code-review-graph` MCP server alongside existing `gold-db`
- `.gitignore` updated — `.code-review-graph/` (graph DB, regenerable, not for VCS)

Nothing in `pipeline/`, `trading_app/`, `scripts/`, `.claude/skills/`, `.claude/agents/`, or `.claude/hooks/` is touched.

## How to use it (after merge)

```bash
# One-time: build the graph
code-review-graph build

# Incremental update (fast; <1s on a few-file change)
code-review-graph update

# Status / stats
code-review-graph status

# Detect impact of current uncommitted changes
code-review-graph detect-changes --base HEAD~1

# MCP server (auto-started by Claude Code via .mcp.json)
code-review-graph serve
```

Restart Claude Code session after merging — the new MCP entry takes effect at session start. The MCP server then exposes ~28 tools to the agent (callers, callees, references, tests, impact, flows, etc.).

## Suggested follow-up (deferred — own task)

1. **Update `.claude/agents/blast-radius.md`** — Step 0: query code-review-graph MCP for resolved callers/importers/tests. Step 1+: layer canonical-source / DB / domain checks (existing logic). Replaces grep-based discovery with resolved-edge ground truth.
2. **Pre-commit hook (optional):** add `code-review-graph update` to `.githooks/pre-commit` so the graph stays incrementally fresh. Skip if perf cost noticeable.
3. **Re-evaluate GitNexus** when issue #1136 is fixed AND the >32KB scope-extraction bug is fixed AND a new release is out. Re-eval criteria documented above. Until then, don't bother.

## Branch safety

`tooling/gitnexus-eval` is off `origin/main`. Touches only:
- `docs/eval/2026-04-28-gitnexus-evaluation.md` (new)
- `.mcp.json` (1 entry added; `gold-db` preserved verbatim)
- `.gitignore` (1 line added)

Merge or delete safely. No production code, no schema, no canonical-source changes. Trivial-tier.

## License note

`code-review-graph` is **MIT**. No restriction.
GitNexus would have been PolyForm Noncommercial 1.0.0 — fine for personal research, blocked for commercial/SaaS. Moot now that we're not adopting it.
