---
status: active
owner: claude (calibration session 2026-04-28)
last_reviewed: 2026-04-28
superseded_by: ""
---

# code-review-graph calibration

**Date:** 2026-04-28
**Branch:** `crg-calibration` (off `origin/main` at `2fceb17b`)
**Tool version:** code-review-graph v2.3.2 (installed at `.venv/Scripts/code-review-graph.exe`)
**Package source:** `C:/Users/joshd/AppData/Local/uv/cache/archive-v0/zpO53BNgAIWm_3SjfGuA8/Lib/site-packages/code_review_graph/`
**Authoritative docs:** https://github.com/tirth8205/code-review-graph (v2.3.2)

## Authority

This is a **`decision` class** doc per `docs/governance/document_authority.md` § Document classes.

`code-review-graph` is **navigation/blast-radius support, not truth**. Per `CLAUDE.md` § Project Truth Protocol, discovery uses ONLY canonical layers (`bars_1m`, `daily_features`, `orb_outcomes`). Graph claims are hints — verify with `Read`/`Grep` before acting on them.

## What changed

| Surface | Before | After | Provenance |
|---|---|---|---|
| Graph files indexed | 1040 | 1034 | `code-review-graph status` 2026-04-28 19:22 |
| Graph nodes | 13775 | 13727 | same |
| Graph edges | 150025 | 149241 | same |
| Built on branch | `triage-e2-lookahead-9-candidates` | `crg-calibration` | same |
| `.code-review-graphignore` | absent | created (23 patterns, 46 lines incl. comments) | `git diff origin/main` |
| Registered repos | 0 | 1 (`canompx3`) | `code-review-graph repos` |

## Ignore file

Created at repo root: `.code-review-graphignore`. Augments `DEFAULT_IGNORE_PATTERNS` defined in `incremental.py:34-71`.

**Behavior facts (cited):**

- Filename `.code-review-graphignore` is the convention — `incremental.py:199`.
- Patterns *augment* defaults, not replace — `incremental.py:198` (`patterns = list(DEFAULT_IGNORE_PATTERNS)`).
- Matching is `fnmatch` + segment-prefix for `<dir>/**` — `incremental.py:218`. **Not full gitignore syntax.** Negation (`!pattern`) is treated as a literal `!` prefix (no negation semantics). Anchored paths (`/foo`) actively fail to match (the leading `/` makes the pattern non-matching against unanchored relative paths). Verified empirically: `fnmatch('!keep.txt', '!*.txt')=True` (literal match), `fnmatch('foo/bar', '/foo/bar')=False`.
- File enumeration prefers `git ls-files` — `incremental.py:384`. Untracked files (gitignored or otherwise) are already excluded; the ignore file is for tracked-but-noisy paths.
- USAGE.md says "same syntax as .gitignore". Source says fnmatch. **Source wins** — keep patterns fnmatch-safe.

**Patterns added** (defaults already cover node_modules/.venv/__pycache__/dist/build/coverage/*.db/*.lock/*.pyc):

- Test/coverage caches: `.pytest_cache/**`, `htmlcov/**`, `.ruff_cache/**`, `.mypy_cache/**`
- Research output (regenerated, not source): `reports/**`, `logs/**`, `output/**`
- Binary data formats: `*.dbn`, `*.dbn.zst`, `*.parquet`, `*.csv`, `*.feather`, `*.arrow`, `*.pkl`, `*.pickle`, `*.npy`, `*.npz`, `*.h5`, `*.hdf5`
- Generated/snapshot artifacts: `REPO_MAP.md`, `docs/context/**`
- Tooling local state: `.claude/hooks/.completion-notify-last`, `.gitnexus/**`

## Approved workflow

### Variant A — Task scoping (find what to read)

```
context_resolver → get_minimal_context → read canonical files directly →
  if changing code: get_impact_radius → tests → drift → self-review
```

### Variant B — PR/diff review (what changed, what's at risk)

```
detect_changes_tool (detail_level=minimal) →
  if risk≥medium escalate to detail_level=standard →
  read canonical files for any flagged area →
  tests → drift → self-review
```

### Token budget defaults

- `get_minimal_context_tool` → always (cheapest entry point)
- `detect_changes_tool` / `get_review_context_tool` → `detail_level=minimal`, `include_source=False` first
- `get_impact_radius_tool` → `max_depth=2` default; raise to 3 only on high-risk diffs
- `semantic_search_nodes_tool` → skip until embeddings exist (currently 0 embeddings); substring search isn't worth the call
- `get_architecture_overview` / `get_suggested_questions` / `traverse_graph` → never in parallel, never as first call

## Trigger-rebuild canonical files

If any of these are edited, the graph must be rebuilt before the next review uses it (otherwise downstream impact analysis is wrong):

- `pipeline/dst.py` (SESSION_CATALOG)
- `pipeline/cost_model.py` (COST_SPECS)
- `pipeline/asset_configs.py` (ACTIVE_ORB_INSTRUMENTS)
- `trading_app/eligibility/builder.py` (`parse_strategy_id`)
- `pipeline/check_drift.py` (drift checks)
- `trading_app/prop_profiles.py` (ACCOUNT_PROFILES)

Rebuild command: `code-review-graph build --repo C:/Users/joshd/canompx3` (full, ~16s for 1034 files).

For routine edits use: `code-review-graph update --base origin/main` (incremental).

## Failure modes

1. **Graph staleness across branches** — graph is keyed per-repo, not per-branch. CLI `status` surfaces a warning (`incremental.py` post-build summary) but does not block. **Mitigation:** rebuild after switching long-lived branches.
2. **Multi-writer corruption** — sqlite store at `<repo>/.code-review-graph/graph.db` is shared across worktrees and concurrent CLI/MCP processes. **Mitigation:** one writer at a time. Check `tasklist | grep code-review` before any rebuild.
3. **Windows ProcessPool quirk** — earlier versions deadlocked on Windows during full builds. v2.3.2 worked here in 16s, but if a future build hangs use `code-review-graph build --skip-postprocess` (raw parse only) or `--skip-flows`.
4. **Embedding-model versioning** — embeddings (currently absent) require `pip install code-review-graph[embeddings]`. If enabled later, pin model via `CRG_EMBEDDING_MODEL` env var; changing the model re-embeds all nodes (`docs.py:26`).
5. **fnmatch vs gitignore syntax** — see ignore-file section. Don't use `!` or `/` anchoring.
6. **Graph nodes are a frozen snapshot** (Volatile Data Rule, `CLAUDE.md`). Always verify graph claims with `Read`/`Grep` before acting on them — especially if the graph was built > 1 day ago or before recent edits to canonical files.
7. **MCP disconnect** — MCP server can disconnect mid-session. CLI binary is the fallback (`.venv/Scripts/code-review-graph.exe`). All MCP tools have CLI equivalents for `status`, `build`, `update`, `detect-changes`.

## Doctrine compliance

- **Branch discipline** (`.claude/rules/branch-discipline.md`): branched from `origin/main`, not local main. Verified `origin/main == local main == 2fceb17b` before branching.
- **Parallel session awareness** (`feedback_parallel_session_awareness.md`): announced 5 untracked files belonging to other sessions; stashed Phase D YAML with labelled message (`stash@{0}: phase-d-yaml-WIP-from-other-session-preserved-for-crg-calibration`) — pop on `prereg-phase-d-d0-v2-garch` to restore.
- **Stage-gate guard** (`feedback_stage_gate_global_mode_rule.md`): no production-code edits. Pass.
- **Pre-commit** (`feedback_precommit_overkill_for_docs.md`): drift always runs; pytest skipped for docs-only changes (`.githooks/pre-commit` line 154). Never `--no-verify`.
- **2-Pass Implementation Method** (`CLAUDE.md`): discovery done (read all 13 affected files in the package + 5 in repo); implementation done with verification after each step (status pre/post, build output, smoke test).
- **Project Truth Protocol** (`CLAUDE.md`): graph explicitly documented as non-truth navigation surface; canonical layers untouched.

## Commands run (audit trail)

```bash
# Discovery
python scripts/tools/context_resolver.py --task "..." --format markdown
git status --short && git branch --show-current
git rev-list --left-right --count origin/main...HEAD
tasklist | grep -i -E "code-review|crg|uvx"
where code-review-graph
uvx --from code-review-graph python -c "import code_review_graph; print(...)"
code-review-graph --help
code-review-graph status

# Branch
git fetch origin
git stash push -m "phase-d-yaml-WIP-..." -- docs/audit/hypotheses/...yaml
git checkout -b crg-calibration origin/main

# Calibration
# (write .code-review-graphignore with 29 patterns)
code-review-graph build --repo C:/Users/joshd/canompx3
code-review-graph status
code-review-graph detect-changes --base HEAD --brief
code-review-graph register C:/Users/joshd/canompx3 --alias canompx3
code-review-graph repos
```

## What is NOT done (deferred)

- **Embeddings** — semantic search falls back to substring without them. Recipe: `pip install "code-review-graph[embeddings]"` then `code-review-graph postprocess --embeddings` (verify flag with `--help`).
- **`watch` daemon** — auto-update on file changes. Adds a process to manage. Manual `update` after canonical-file edits is sufficient for now.
- **Cross-repo registration of other repos** — only `canompx3` registered. Add others if/when needed via `code-review-graph register <path> --alias <name>`.
- **Wiki/visualization generation** — `code-review-graph wiki` and `visualize` are nice-to-have, not load-bearing.

## Review schedule

- **Re-validate** if tool upgraded past v2.3.2 (default patterns, ignore semantics, or CLI flags may change).
- **Rebuild** after any edit to a trigger-rebuild canonical file (see list above).
- **Re-read** this doc before adopting any new MCP tool from the server.
