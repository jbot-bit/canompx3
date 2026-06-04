# Stage — Path-Scoped Drift Gate in pre-commit

**mode:** IMPLEMENTATION
**branch:** drift-supersonic (off origin/main `9429c540`)
**owner:** session/joshd-wt-06Thu04-20261905 (drift terminal)
**opened:** 2026-06-05

## Goal

Make docs-only commits fast by **path-scoping the heavy drift step** in
`.githooks/pre-commit` — without weakening capital/risk safety. Code commits run
full drift unchanged; full proof preserved at a new **pre-push** gate + CI.

Measured baseline (this box, cold): **188.8s total pre-commit; drift step ~172s**
(`.claude/scratch/drift-baseline-full.txt`).

## scope_lock

ONLY these files may be edited under this stage:

- `.githooks/pre-commit` — hoist `STAGED_ALL` above the drift step; add docs-safe
  classifier (skip heavy drift iff every staged path is docs-safe AND none
  denylisted AND the staged set is non-empty).
- `.githooks/pre-push` — NEW. Full `check_drift.py` (full blocking set, no
  `--skip` flags) → abort push on nonzero. The net that makes commit-time
  skipping sound.
- `tests/test_tools/test_git_hooks_env.py` — extend with classifier assertions
  (existing hook-text-assertion harness; same style).
- `docs/plans/active/2026-06/2026-06-04-drift-precommit-speed-audit.md` — append
  per-check classifier design note (TODO, not built). **Only after PR #359 lands
  this file on main** (currently the doc is in PR #359, not on origin/main).
- `docs/runtime/stages/drift-path-scope-gate.md` — this file.

## Hard rules (operator-locked, option 2 / low blast radius)

- Do NOT rewrite `pipeline/check_drift.py`.
- Do NOT migrate to pre-commit.com.
- Do NOT touch trading logic, C11, profiles, allocator, stops, live config, or
  DB/schema.
- **Fail closed** for unknown / code-risk / empty staged paths.

## Classifier contract

- **Staged set** is computed with `--diff-filter=ACMD` (Added/Copied/Modified
  **+ Deleted**) — the existing `STAGED_ALL` at L540 used `ACM` and silently
  excluded deletions; a delete-only `.py` commit would yield an empty set.
- **Empty staged set → full drift** (fail-closed). Never docs-skip on emptiness.
- **Denylist (→ full drift):** any path matching `pipeline/`, `trading_app/`,
  `scripts/`, `tests/`, `research/`, `.githooks/`, `.claude/`, `*.py`,
  `pyproject.toml`, `uv.lock`, `.mcp.json`, `docs/audit/`, `*.yaml`/`*.yml`/`*.json`,
  `.python-version`, or anything unrecognized.
- **Docs-safe allowlist (→ eligible to skip):** `*.md` OUTSIDE `docs/audit/`,
  `HANDOFF.md`, `docs/plans/**`, `docs/runtime/**` (markdown notes), `docs/**/*.md`.
- **Skip iff:** staged set non-empty AND every path docs-safe AND no path
  denylisted. Any doubt → full drift.
- Log lines: `DRIFT: skipped path-safe docs-only commit (<n> docs files)` OR
  `DRIFT: running full check due to <paths>`.

## blast_radius

- `.githooks/pre-commit` runs on EVERY commit in every worktree sharing this
  hooksPath. A classifier bug that wrongly skips = a code/config change commits
  without drift coverage. Mitigations: (a) fail-closed default, (b) pre-push net,
  (c) CI `check_drift.py` (ci.yml:68) backstop, (d) adversarial audit gate before
  main integration.
- `.githooks/pre-push` is NEW — adds latency to `git push` only (not commit).
  Fail-closed; must not wedge a legitimate push on a transient error → mirrors
  pre-commit's venv-resolution + `set +e`/capture/`set -e` exit semantics.

## Verification gates (all required before main integration)

1. Full `pipeline/check_drift.py` green (manual run).
2. Three smoke commits on branch: docs-only → skip; python → full; unknown → full.
3. Hook tests pass.
4. `evidence-auditor` PASS: prove the skip never lets a code/config/capital change
   commit without drift.

Integration: FF to main ONLY if all four pass (operator GO pre-granted for that
condition). Else hold on branch with findings.

## Coordination

- This terminal owns drift (operator-confirmed 2026-06-05). The OTHER terminal
  (`canompx3-wt-06Thu04-20261811`) owns C11 deploy / firm-tier economics —
  disjoint file set (drift = `.githooks/`; C11 = `prop_profiles.py`/HANDOFF).
- Two Codex worktrees (`precommit-drift-speed`, `1248`) have dirty
  `.githooks/pre-commit` but their changes (timing instrumentation, venv-resolution
  refactor) are ALREADY in origin/main — stale/superseded, NOT the path-scope work.
  No conflict.
