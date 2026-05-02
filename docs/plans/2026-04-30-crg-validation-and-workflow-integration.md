---
status: design-v2
created: 2026-04-30
authority: nav-surface (non-truth) — supplements docs/specs/crg-integration.md
supersedes: v1 of this file (v1 had pigeon-holed scope)
---

# CRG Validation and Workflow Integration — Design v2

Goal unchanged from v1: turn CRG (code-review-graph v2.3.2) from "installed but unused" into a tool actively reducing time-to-context for strategy work and PR reviews — while closing concrete worktree-resolution bugs.

## What changed vs v1

v1 was scoped through a single lens (extend the PR #179 env-var fix). After re-reading the official v2.3.2 changelog, USAGE.md, COMMANDS.md, LLM-OPTIMIZED-REFERENCE.md, TROUBLESHOOTING.md and the upstream `skills.py` `generate_hooks_config()`, three of v1's premises were wrong or under-scoped:

- v1 missed that **MCP server inherits the worktree's cwd** as its repo root — so MCP queries fragment from worktrees too, not just the CLI and the post-edit hook. Confirmed: `.mcp.json` has no `env` block on the `code-review-graph` entry.
- v1 missed that **PR #253 (cwoolum)** in v2.3.2 ships "automatic graph build for new worktrees in Claude Code" upstream. The local `skills.py` `generate_hooks_config()` is the canonical reference; we opted out via `--no-hooks` and re-implemented PostToolUse ourselves with worse parameters (no `--skip-flows`, 5s timeout instead of 30s, no env var).
- v1 missed that **v2.3.2 added 6 new MCP tools** that are more useful for strategy/review work than the existing 4 slash commands cover (`get_hub_nodes_tool`, `get_bridge_nodes_tool`, `get_knowledge_gaps_tool`, `get_surprising_connections_tool`, `get_suggested_questions_tool`, `traverse_graph_tool`). v1's "we have all 4 ladder rungs" claim is technically true for the *minimal* ladder but incomplete for the *useful* ladder.

v2 design is bigger than v1 in problems-addressed but tighter in approach: align with upstream rather than hand-roll parallel infrastructure.

## Turn 1 — ORIENT (full grounding)

### Repo-truth state of CRG (verified 2026-04-30)

- Tool version: `code-review-graph 2.3.2` (canonical `.venv` and worktree `.venv`).
- Canonical graph at `C:\Users\joshd\canompx3\.code-review-graph\graph.db`: **healthy** — 1052 files, 13912 nodes, 151617 edges, 178 MB, last built 2026-04-30 01:06 on `feature/crg-phase1-hooks-2026-04-29` at `f659711a`. Five commits stale vs `origin/main` (which is now `5a4b4450` after PR #179 merge).
- Worktree-local graph at `canompx3-crg-phase2/.code-review-graph/graph.db`: **fragment** — 4 MB, 16 files, 401 nodes, built on `fix/ci-test-app-sync-filter-keys` at `fcf885b1`. Stale and structurally tiny.
- Embeddings: 12,693 nodes embedded (`all-MiniLM-L6-v2`), per calibration. Anything added in the 5 commits since `f659711a` is **not** in the embedded space, so `semantic_search_nodes_tool` returns lower-rank or missing hits for the most recent code.
- MCP entry: `.mcp.json` `code-review-graph` server uses `uvx code-review-graph serve` with no `env` block. Inherits whatever env Claude Code launches it with. Without explicit `CRG_REPO_ROOT`, MCP queries from a worktree see the worktree fragment.
- Hooks state: `.claude/settings.json` has `PostToolUse: 3 entries` and `SessionStart: 1 entry` — NONE of them are CRG-related (we opted out per spec). The `.claude/hooks/post-edit-pipeline.py` is *part of* a Claude Code PostToolUse hook chain and includes a CRG-update step, but it has the same worktree bug as the just-fixed drift checks: invokes `code-review-graph update --base HEAD~1` with `cwd=worktree_root` and no env var. Diverges from official template (`update --skip-flows`, 30s timeout).
- Drift checks D1–D5: live in `pipeline/check_drift.py` positions 128–132. Helper `pipeline/check_drift_crg_helpers.py` is post-PR-179 and routes via `find_project_root` correctly. D1–D5 are advisory-only.
- Pre-commit step 3b: correctly exports `CRG_REPO_ROOT` for worktree commits via the canonical-sibling detection. Exists in `.githooks/pre-commit` lines 209–234.

### Upstream-canonical references (the official patterns we should align with)

- `LLM-OPTIMIZED-REFERENCE.md` `<usage>`: "ALWAYS start with `get_minimal_context_tool(task=...)`" + "Use `detail_level='minimal'` on all subsequent calls". Our `/crg-context` and `/crg-blast` already conform.
- `LLM-OPTIMIZED-REFERENCE.md` `<review-delta>`: official PR/diff workflow. Step 1 is `get_minimal_context_tool` (covered), step 2 is `detect_changes_tool(detail_level='minimal')`. **We do not have a slash command for `detect_changes_tool`.** That is the canonical "review my recent changes" entry point per upstream.
- `LLM-OPTIMIZED-REFERENCE.md` `<watch>`: "Run `code-review-graph watch` (auto-updates graph on file save) OR use PostToolUse hooks." Two valid alternatives — we use the latter, with bugs.
- `skills.py` `generate_hooks_config()`: the upstream-canonical hook config. PostToolUse on `Edit|Write|Bash` invokes `code-review-graph update --skip-flows` with timeout 30s; SessionStart invokes `code-review-graph status` with timeout 10s.
- `TROUBLESHOOTING.md` §3 "project-vs-user scoping": graph database is **project-scoped** (one per repo), MCP server is **project-scoped** (Claude Code launches per-project with `cwd=<project>`), package and registry are user-scoped. **A worktree IS a project** to Claude Code (separate cwd) — so without explicit env override, you get a separate fragment per worktree.
- `TROUBLESHOOTING.md` §4 "Using a venv?": MCP/hook commands are hardcoded at install time. Re-install if venv changes. We use `uvx` which side-steps this.
- v2.3.0 changelog: `CRG_DATA_DIR` (graph storage path) and `CRG_REPO_ROOT` (project-root override) are the two official env knobs. Both are honored by CLI and MCP.

### Doctrine ground truth (audited)

- `docs/specs/crg-integration.md` "non-truth nav surface": **CRG never overrides doctrine; canonical code wins.** v2 does NOT add any CRG-derived drift check that gates commits. New advisories continue the D1–D5 pattern: emit warnings, never block.
- `CLAUDE.md` Volatile Data Rule: graph is a frozen snapshot — slash command output should remind users to verify with Read/Grep. v1's commands already do; v2 adds nothing here.
- `CLAUDE.md` 2-Pass Implementation Method: read affected files first (done above), articulate purpose (done in v1, refined here), implement → verify → self-review.
- One-way dependency `pipeline/` → `trading_app/`: D1 already enforces. v2 doesn't change this.
- `STRATEGY_BLUEPRINT.md` SS5 NO-GO registry: no conflict; CRG is a navigation layer, not a research artefact.

### Memory ground truth (audited)

Three CRG memory files re-read in full:
- `code_review_graph_calibration.md` — workflow defined; calibration metrics; failure modes documented (fnmatch not gitignore, branch staleness, multi-writer corruption, Windows ProcessPool hang).
- `feedback_crg_v2_1_0_bugs.md` — 5 bugs against v2.1.0:
  - Bug 1, 2: `analysis_tools._func` wrappers tuple-unpack and `_validate_repo_root` Path/str. Routed around in helpers.
  - Bug 3: `tests_for` returns 0 for many real test patterns. AST scan in D3.
  - Bug 4: `find_large_functions(file_path_pattern=...)` substring not regex. Client-side filter.
  - Bug 5: qualified-name format `<rel/path>::<symbol>` not dotted. Documented in slash commands.
  Bugs 1, 2 are upstream library bugs in `analysis_tools`; we already route around. Bugs 3, 4, 5 are graph-data shape issues. v2.3.2 changelog mentions `apply_refactor_tool` dry-run, edge-confidence scoring, hub/bridge/knowledge-gap/surprise/suggested-questions tools, but does not mention fixes to the 5 bugs we observed. Assume bugs persist; existing workarounds remain valid.
- `feedback_crg_worktree_repo_root_resolution.md` — PR #179 fix for drift checks. v2 extends this pattern to two more sites (post-edit hook, MCP server config).

### Purpose statement (refined)

Why this matters: the canonical graph is healthy and embedded but the *paths that consume it* are partially broken. Two confirmed live bugs (post-edit hook, MCP env) keep CRG returning fragments to whoever queries it from a worktree session — which is *every* active session right now. Plus, the upstream tool ladder added high-value tools in v2.3.2 (`get_suggested_questions_tool`, `get_knowledge_gaps_tool`, `detect_changes_tool`) that our slash-command surface doesn't expose. Until both are fixed, "make CRG earn its keep" is unreachable: every query returns the wrong answer or returns nothing.

## Turn 2 — DESIGN v2 (multi-take, official-pattern-aligned)

### Failure modes the design must prevent

- F1 (live): Worktree edits silently fragment the worktree-local graph via `_crg_update`. Confirmed.
- F2 (live): Direct `code-review-graph` CLI invocation from a worktree (`code-review-graph status`, etc.) hits the fragment. Confirmed at 08:51 today.
- F3 (live): MCP server launched by Claude Code in a worktree session inherits worktree cwd; queries fragment. Confirmed via `.mcp.json` inspection.
- F4 (latent): Slash-command Python fallback paths pass `repo_root='.'` from a worktree. Verify; fix if true.
- F5 (latent): `detect_changes_tool` is the official "review my recent changes" entrypoint and we have no slash command for it. Discovery gap, not a bug — but solving it is high-value.
- F6 (latent): v2.3.2 added 5 advanced graph-analysis MCP tools and we don't expose any. Discovery gap.
- F7 (latent): Stale embeddings (5 commits behind origin/main) silently degrade `semantic_search` rank. After PR #179 merged, this gap reopens.
- F8 (latent): Future CRG version bump silently breaks our v2.1.0-bug workarounds. Drift check would catch.

### Three approaches considered (re-scoped vs v1)

1. **Align with upstream** *(recommended)*. Replace our `_crg_update` body with a call that mirrors `generate_hooks_config()`'s PostToolUse contract (`update --skip-flows`, 30s timeout) AND adds `CRG_REPO_ROOT` from sibling-detection. Add `env` block to `.mcp.json` `code-review-graph` entry pointing at canonical root (only when worktree differs from canonical). Add three new slash commands (`/crg-changes`, `/crg-questions`, `/crg-gaps`) for the v2.3.2 tools that map best onto our work. Add embedding-staleness advisory drift check. No CLI shim — the env var fix on the post-edit hook plus an alias in personal init covers F2.

2. **Maximalist (Approach 1 from v1, expanded)**. Add an orchestrator skill that pre-runs the entire ladder for any non-trivial task. Out of scope: high blast radius, evolves with upstream changes, premature.

3. **Defer (Approach 3 from v1)**. Same trade-off as before; rejected for the same reason — leaves live bugs.

### Bias check on the recommended approach

- Am I pigeon-holed on env-var fixes? No: v2 explicitly addresses MCP env, slash-command surface gap, and embedding staleness — three different layers.
- Am I treating CRG as truth? No: every new advisory is non-blocking, every new slash command keeps the "verify with Read/Grep" caveat from the existing four.
- Am I making CRG mandatory? No: existing escape hatches (CRG unavailable → ADVISORY emit + return) remain. New checks are advisory.
- Am I duplicating upstream? No: upstream PR #253's "automatic graph build for new worktrees" updates an EMPTY graph at worktree creation; it does not redirect a worktree to a canonical sibling. Our worktree pattern is "all worktrees share one canonical graph", which is *outside* PR #253's scope.
- Am I over-doc'd? Spec already exists. v2 only adds a "Worktree usage" subsection plus updates the `.claude/commands/` headers.
- Am I conflating layers? Drift-check advisory consumes `importlib.metadata` (not CRG output) → no doctrine conflict. New slash commands wrap official MCP tools → behaviour unchanged from upstream.

### One-way dependency check

All proposed edits live in:
- `.claude/hooks/post-edit-pipeline.py` — Claude Code hook layer.
- `.claude/commands/crg-*.md` (existing 4 + 3 new) — slash-command layer.
- `.mcp.json` — MCP config (project-scoped infra).
- `pipeline/check_drift.py` + `pipeline/check_drift_crg_helpers.py` — pipeline-only.
- `tests/test_pipeline/test_check_drift_crg.py` — tests.
- `docs/specs/crg-integration.md` — spec doc.

No production data flow, no schema, no entry-model, no canonical-source change. Stays inside the design-skill autonomous-mode envelope (blast radius = 8 modified + 3 created = 11 files; design-skill threshold is "blast > 5"... see Acceptance / safety check below).

### Safety threshold honesty

The design-skill's autonomous-mode threshold says: "If design reveals schema change, entry model change, or blast radius > 5 files → STOP and ask." v2 modifies 5 files and creates 4 (3 new slash commands + 1 stage doc). Strictly that is over the threshold. Therefore **v2 will NOT auto-execute even though autonomous mode is requested**. I will present v2, wait for explicit go, then split the work into two stages so each stage stays under 5 files.

## Turn 3 — DETAIL v2 (split into two stages)

### Stage A — Bug fixes (5 files, blast radius ≤5)

Purpose: kill the three live worktree-resolution bugs (F1, F2, F3). After Stage A, every CRG path resolves to the canonical graph from any worktree.

1. **EDIT** `.claude/hooks/post-edit-pipeline.py` `_crg_update` — set `CRG_REPO_ROOT` in subprocess `env`; switch to `update --skip-flows`; bump timeout to 30s; add comment pointing to `feedback_crg_worktree_repo_root_resolution.md`. Keep file-prefix gate. Keep fail-silent.
2. **EDIT** `.mcp.json` — add `env: { CRG_REPO_ROOT: "C:/Users/joshd/canompx3", CRG_DATA_DIR: "C:/Users/joshd/canompx3/.code-review-graph" }` to the `code-review-graph` MCP entry. Pinning to the canonical absolute path is acceptable here — `.mcp.json` is per-project and this is the user's canonical project root, recorded in `feedback_crg_worktree_repo_root_resolution.md`. Note in commit message: "if the canonical project root ever moves, update both `.mcp.json` and `.githooks/pre-commit`".
3. **EDIT** `.claude/commands/crg-context.md`, `crg-search.md`, `crg-blast.md`, `crg-tests.md` (4 files but treat as one logical edit) — change Python fallback to call `find_project_root(Path('.'))` instead of `repo_root='.'`. Wrap in try/except so `find_project_root` errors fall back to `Path('.')` with a one-line warning. **NOTE on file-count:** this edit applies the same 4-line patch to four `.md` files; conceptually one change, mechanically four edits. We count it as 4 modifications when judging blast radius.

That brings Stage A to: post-edit hook (1) + .mcp.json (1) + 4 slash commands (4) = 6 file mods. **Slightly over the 5-file threshold.** Acceptable because (a) the four slash-command edits are identical 4-line patches, (b) all 6 edits target one bug class, (c) tests will land in Stage B. If the user wants stricter splitting, Stage A becomes:

- Stage A1 (3 files): post-edit hook, .mcp.json, the most-used slash command (`/crg-search`).
- Stage A2 (3 files): the other three slash commands.

### Stage B — Surface gap + version pin (5 files)

Purpose: expose v2.3.2's high-value tools as slash commands; protect against future CRG version bumps; close embedding staleness gap.

1. **NEW** `.claude/commands/crg-changes.md` — wraps `mcp__code-review-graph__detect_changes_tool` per `<review-delta>` workflow. Pattern: `/crg-changes [--standard]`, default `detail_level=minimal`. Python fallback uses `code_review_graph.changes.detect_changes` (path verified at `C:/Users/joshd/canompx3/.venv/Lib/site-packages/code_review_graph/changes.py`).
2. **NEW** `.claude/commands/crg-questions.md` — wraps `mcp__code-review-graph__get_suggested_questions_tool`. Returns prioritised review questions from graph analysis. Python fallback path: import from `code_review_graph.analysis` per existing helper pattern.
3. **NEW** `.claude/commands/crg-gaps.md` — wraps `mcp__code-review-graph__get_knowledge_gaps_tool`. Returns isolated nodes / thin communities / untested hotspots. High-signal for "what's structurally weak" review questions.
4. **EDIT** `pipeline/check_drift.py` — append `check_crg_version_pin_advisory` (advisory) AND `check_crg_embeddings_freshness_advisory` (advisory). Both consume `importlib.metadata.version("code-review-graph")` and the graph's `last_updated` field; neither consumes CRG query output. Total drift count: 117 → 119.
5. **EDIT** `pipeline/check_drift_crg_helpers.py` — add `installed_crg_version()` and `graph_last_updated_commit()` helpers. Both fail-open: return `None` on any error.
6. **EDIT** `tests/test_pipeline/test_check_drift_crg.py` — 4 new test cases: matched-version, mismatched-version, version-unavailable, embeddings-fresh; one regression test asserting the post-edit hook subprocess receives `CRG_REPO_ROOT` in its env.

Stage B file count: 3 new + 3 edits = 6 file changes. Same overage as Stage A. Same justification — single concern (workflow surface gap + advisory pins). Splittable as 4+2 if the user prefers tighter staging.

### Stage C (optional, parked) — Skill / orchestrator

Out of scope for this design. Re-evaluate after Stage A+B land and we measure how often the new commands get invoked.

### Order of operations across stages

1. Stage A first; verify with the behavioural tests below before Stage B.
2. Stage B after Stage A pre-commit passes from this worktree.
3. Optional Stage C only if Stage A+B usage data shows the existing slash commands are not enough for the user.

### Test strategy (across both stages)

**Stage A behavioural tests (manual + scripted):**
- Edit-then-mtime test: from `canompx3-crg-phase2/`, save a `pipeline/` file. Within 30s, canonical `graph.db` mtime advances; worktree fragment mtime does NOT. (Manual; automatable but PostToolUse hooks need real Claude session to fire.)
- MCP fragment test: from a fresh Claude session in `canompx3-crg-phase2/`, call `mcp__code-review-graph__list_graph_stats_tool` (no params). Result reports 1052 files, not 16.
- Slash-command fallback test: with MCP intentionally disabled (rename `.mcp.json` temporarily), invoke `/crg-context "audit cost model"`. Output should resolve to canonical 1052-file graph, not the worktree fragment. (Manual; verifies F4.)

**Stage B unit tests:**
- 4 new test cases as listed in Stage B step 6.
- Existing `TestCrgRepoRootResolution` continues to pass.
- Regression test: assert `_crg_update` invokes subprocess with `env` containing `CRG_REPO_ROOT`. Use `unittest.mock.patch` on `subprocess.run`.

**Pre-commit gauntlet (both stages):**
- Drift checks pass: 117 (Stage A unchanged) → 119 (Stage B).
- pytest fast subset green.
- No advisory rows that point to live failure (a NEW advisory row from version-pin check is allowed and expected; matched version means no warn line).

### Migration / rebuild

None for the canonical graph. Stage A takes effect immediately on next save. Stage B's embedding-staleness advisory is informational; if it fires, action is `code-review-graph build` to refresh embeddings — done out-of-band by the user.

### Drift check impact

Stage A: no new checks.
Stage B: +2 checks, both advisory (positions 134, 135).

## Turn 4 — VALIDATE v2

### Failure modes (and mitigations)

- F-A: `.mcp.json` env-block hardcodes the user's absolute canonical path. **Mitigation:** documented in spec "Worktree usage" section; pre-commit hook already uses sibling detection, so the absolute path in `.mcp.json` is the single brittle spot. Acceptable trade-off; alternatives (relative path, env-var-only resolution) either don't work for MCP launch context (Claude Code may resolve relative paths against worktree cwd) or require Claude Code to set env vars per-project (not currently supported in `.mcp.json` schema as well as a hardcoded value).
- F-B: `find_project_root` raises in slash-command fallback if invoked outside a git repo. **Mitigation:** wrap try/except, fall back to `Path('.')`, surface one-line warning.
- F-C: New advisory checks fire on every commit forever once CRG version drifts. **Mitigation:** advisory-only; bumping `KNOWN_TESTED_CRG_VERSION` is a one-line PR after re-validating the v2.1.0 bug repros.
- F-D: 3 new slash commands clash with existing CRG ladder discipline ("always start with `/crg-context`"). **Mitigation:** each new command's frontmatter description repeats the ladder rule; `/crg-questions` and `/crg-gaps` are positioned as "after `/crg-context` flagged risk medium/high"; `/crg-changes` is a peer to `/crg-context` for the diff-review workflow.
- F-E: Embedding staleness advisory false-positives if user has not enabled embeddings. **Mitigation:** check fails open if no embedding metadata is found in the graph DB.
- F-F: MCP server caches stale env on hot-reload. **Mitigation:** out of scope — owned by upstream FastMCP. Calibration note already documents MCP restart as recovery action.
- F-G: Stage A's bumped timeout (5s → 30s) makes post-edit feel slow. **Mitigation:** PostToolUse hooks run async in Claude Code; user-visible latency is when the next *blocking* step runs, not the hook itself. Upstream chose 30s; we should align.

### What proves correctness (behavioural, not "it runs")

Stage A:
- Save a file in `pipeline/` from `canompx3-crg-phase2/`. Within 30 seconds, `C:\Users\joshd\canompx3\.code-review-graph\graph.db` mtime advances. The worktree-local fragment does not.
- From a fresh Claude session in `canompx3-crg-phase2/`, `mcp__code-review-graph__list_graph_stats_tool({})` returns "1052 files" not "16 files".
- `python pipeline/check_drift.py` exits 0 with all existing 117 checks passing.

Stage B:
- `python pipeline/check_drift.py` exits 0 with 119 checks total.
- `pytest tests/test_pipeline/test_check_drift_crg.py -v` exits 0 with 4 new tests passing.
- `/crg-changes` invoked from a session with a non-empty diff returns a populated structured response.
- `/crg-questions` invoked from any session returns a list of suggested review questions.
- Advisory check fires once when `KNOWN_TESTED_CRG_VERSION` is intentionally bumped in a test branch — proves it's wired.

### Rollback

Each file change in each stage is independently revertable. No data migration. Canonical graph self-heals at next pre-commit (which already exports `CRG_REPO_ROOT` correctly).

### Guardian prompts

None — no production trading logic, no entry-model change, no statistical methodology.

## Acceptance criteria

Stage A:
1. `.claude/hooks/post-edit-pipeline.py` `_crg_update` invokes subprocess with `env` containing `CRG_REPO_ROOT` set to the canonical sibling.
2. `.mcp.json` `code-review-graph` entry has `env: { CRG_REPO_ROOT: ..., CRG_DATA_DIR: ... }`.
3. Four slash-command Python fallbacks call `find_project_root` not `repo_root='.'`.
4. Behavioural tests above pass manually.

Stage B:
1. `python pipeline/check_drift.py` reports 119 checks; all pass.
2. `pytest tests/test_pipeline/test_check_drift_crg.py -v` reports 4 new tests passing.
3. Three new slash commands exist: `/crg-changes`, `/crg-questions`, `/crg-gaps`. Each has MCP-preferred + Python-fallback paths consistent with the existing 4 commands.
4. `docs/specs/crg-integration.md` has a "Worktree usage" subsection and an updated tool-ladder table reflecting 7 commands.

Cross-stage:
- No diff in canonical sources (`pipeline/cost_model.py`, `pipeline/dst.py`, `pipeline/asset_configs.py`, `trading_app/eligibility/builder.py`, `trading_app/prop_profiles.py`).
- `git log` shows two commits per stage, named `fix(crg): worktree resolution Stage A` and `feat(crg): v2.3.2 surface + version pin Stage B`.

## Out of scope (deferred)

- Stage C orchestrator skill. Reconsider after measuring.
- Re-validating each of the 5 v2.1.0 bug repros against v2.3.2. Cheap follow-up after Stage B's version-pin check fires once.
- Embedding rebuild post-PR-179. The advisory will fire and prompt user action.
- MCP-side caching behaviour. Upstream concern.
- Top-level `CRG_REPO_ROOT` set in user's shell init. Personal-config concern, not project.
