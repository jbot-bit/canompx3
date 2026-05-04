---
status: active
owner: canompx3-team
last_reviewed: 2026-04-28
superseded_by: ""
---
# Token-Efficiency & Rework-Prevention Sweep — Design

## Purpose

Two distinct token-cost classes were audited:

1. **Static per-session boilerplate** — files that load into context at session start. Measured ~28K characters / ~7K tokens. Already 90% well-tuned (19 of 22 `.claude/rules/*.md` files correctly path-scoped per Anthropic 2026 docs).
2. **Rework cycles when git state breaks** — single-incident token costs in the 50K+ range. Documented historical incidents: CRLF-merge-wall (PR #130), shared-worktree-collision (PR #138), ralph-loop-main-red-cascade (PR #108 burned ~3 hours). The first two are infrastructure-fixed; the third has no automated detection.

Anthropic 2026 documentation reviewed:
- `code.claude.com/docs/en/memory` — confirms CLAUDE.md target <200 lines, MEMORY.md auto-load 200 lines or 25KB whichever first, `@path` imports do NOT reduce context.
- `code.claude.com/docs/en/best-practices` — "If your CLAUDE.md is too long, Claude ignores half of it because important rules get lost in the noise."
- `code.claude.com/docs/en/hooks` — SessionStart hooks should complete <10s, output cap 10KB. Doc literally publishes a CI/Build Status Check template (curl + warn-don't-block). Sanctioned pattern.

## Tracks

### Track A — `pooled-finding-rule.md` path-scope leak (DONE in this design pass)

**Single-file change.** Add `paths:` frontmatter to `.claude/rules/pooled-finding-rule.md`. Saves ~3K chars per non-audit session.

Paths gating: `docs/audit/results/**/*.md`, `docs/audit/hypotheses/**/*.md`, `docs/institutional/**/*.md`, `research/**/*.py`.

### Track C — Main-CI-red pre-flight (PRIMARY DELIVERABLE)

**Single-file production change** plus tests.

Adds `_main_ci_status_lines()` helper to `.claude/hooks/session-start.py` matching the existing `_origin_drift_lines()` / `_env_drift_lines()` / `_parallel_session_lines()` idiom. Behavior:

- On session start, after existing helpers, query `gh run list --branch main --limit 1 --status completed --json conclusion,databaseId,name,headSha`.
- Cache result at `<git-common-dir>/.claude.main-ci-status` (repo-wide, NOT per-worktree) for 5 minutes via timestamp-stamped JSON.
- On red: emit one stderr line naming the failing run id and workflow with a `gh run view <id>` hint.
- On green: emit one line `Main CI: green ✓` to match existing "Origin: in sync" tone.
- On any failure (offline, gh missing, gh unauth, no GitHub remote, parse error, timeout): silent skip, matching the offline-tolerance pattern of every other helper in the module.

**Why repo-wide cache (per `--git-common-dir`, not `--git-dir`):** CI status of `main` is repo state, not worktree state. Across 4 active worktrees the user keeps, per-worktree cache would 4× the gh API hits.

**Why atomic write (`tempfile.NamedTemporaryFile` + `os.replace`):** corrupted cache file is parseable-handled (catch + cache-miss fallback), but atomic write is institutional-grade default — write-then-rename pattern guarantees no partial-state leak even under interrupt.

**Why 5-minute TTL:** typical session-burst is 1-10 prompts in a few minutes. Re-fetching CI on every `/clear` or `/resume` is wasteful. 5 min is short enough that a freshly-merged green is reflected before the next intentional new session.

### Track B — MEMORY.md behavioral-rules compression (DEFERRED, separate user decision)

Not bundled into this PR. Behavioral-rules section currently runs ~8K chars across ~28 entries; many violate the file's own ≤200-char-per-line contract. Proposed compression preserves rule headlines + linked feedback-file references, moves prose into the linked files. Awaiting user judgment on which entries archive vs stay.

## Design rejections (with reasoning)

- **Plugin packaging.** Anthropic docs comparison table: standalone for "personal/project-specific," plugins for "sharing with teammates." This hook references the project's specific GitHub auth and uses already-installed `gh`. Plugin form is premature.
- **`async: true` hook config.** SessionStart async runs in background and reports later. Synchronous worst-case here is <1 second, async would land the warning at unpredictable conversation points.
- **MCP-tool wrapping.** MCP introduces server dependency and JSON-RPC overhead for one read query. Anthropic's own example uses plain bash + curl — simplest mechanism wins.
- **Branch from current `chore/close-stages-141-142`.** Off-scope. Branch hygiene = atomic commit per concern.

## Failure modes & mitigations

| Mode | Mitigation |
|---|---|
| `gh` not installed | silent skip via FileNotFoundError catch |
| `gh` not authenticated | gh exits non-zero, returncode-check skips |
| Repo has no GitHub remote | empty JSON returned, treated as silent skip |
| GitHub API throttled | 5-min cache reduces hit rate to ~12/hr/worktree; ceiling is 5,000/hr |
| Stale cache after hot fix lands | ≤5-minute window of stale-RED display; user can ignore |
| Cache file corruption | JSON parse exception → cache miss; atomic write prevents in the first place |
| In-progress run on main | `--status completed` filter excludes — last completed verdict reported |

## Verification plan

- New pytest file `.claude/hooks/tests/test_main_ci_preflight.py` exercises:
  1. Cache hit fresh — gh NOT called, cached line returned.
  2. Cache miss + `gh` returns `{"conclusion":"failure",...}` — RED line emitted, cache written.
  3. Cache miss + `gh` returns `{"conclusion":"success",...}` — green line emitted.
  4. Cache miss + `gh` errors (FileNotFoundError) — empty list returned silently.
  5. Cache miss + `gh` returns empty array (no completed runs) — empty list silently.
  6. Cache stale (>5min old) — gh re-called, cache refreshed.
- `python pipeline/check_drift.py` after changes — confirm 0 new violations.
- Hook smoke test: `echo '{"session_type":"startup"}' | python .claude/hooks/session-start.py 2>&1` — confirm output shape.

## Rollback

`git revert` of the single commit. Hook returns to current behavior. Cache file is per-repo and self-rebuilds; no migration needed.

## Parallel-session safety

- Branch cut from `origin/main`; not on any other worktree's branch.
- Edits scoped to `.claude/hooks/session-start.py`, `.claude/hooks/tests/test_main_ci_preflight.py`, `.claude/rules/pooled-finding-rule.md`, plus this design doc and the stage file.
- Stash `close-stages-141-142-WIP` preserves the prior chore/close-stages branch state for restoration after merge.
