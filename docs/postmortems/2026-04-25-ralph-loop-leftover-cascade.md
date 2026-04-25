# Postmortem: Ralph-loop leftover cascade

**Date:** 2026-04-25
**Severity:** HIGH (process / dev-ergonomics)
**Resolution PR:** #108

## What happened

A single-PR fix to format six files turned into ~3 hours and tens of thousands of tokens of git/concurrency yak-shaving. Root cause: four independent leftovers from prior Ralph-loop iterations all landed on `origin/main` despite blocking CI, and any PR rebased on top inherits them.

The four:

1. **CI format failure** (iter-179): `Pipeline CI` had been red on every recent main commit because six files were merged unformatted. Five open PRs (#101-#105) all stuck on the same red `checks` job.
2. **GIT_* env leak in tests** (iter-179): `TestCheckGitCleanlinessIntegration` and `TestPhase4DiscoveryEnforcement` use `subprocess.run(["git", ...], cwd=tmp_path)`. Inside a pre-commit hook (or any parent that exports `GIT_DIR` / `GIT_WORK_TREE`), git ignores `cwd=` and commits land in the parent worktree. Repro: literal "pre-register test" / "edit" commits landed twice on the active branch during pre-commit retries before the cause was identified.
3. **DuckDB migration brittleness** (iter-179): 25 `try/except duckdb.CatalogException: pass` blocks around `ALTER TABLE ADD COLUMN`. On DuckDB ≥ 1.5 a CatalogException aborts the implicit txn, breaking subsequent ALTERs. CI is fine because `uv.lock` pins 1.4.4; local .venv had 1.5.2 and surfaced the bug only inside pre-commit.
4. **CORRELATION_REJECT_RHO rename** (iter-171): commit `9809f1b8` renamed the constant but missed `research/audit_allocator_rho_excluded.py`. Every PR's `Tests with coverage` job fails at collection.

Each layer was invisible until the previous was peeled off. The Bloomberg-quality directive ruled out skipping (`--no-verify`) or partial fixes; investigation costs compounded.

## Why it kept biting

Three structural gaps:

- **Ralph-loop PRs land with red CI.** Commits #107 (iter-179) and #105 (iter-178?) merged onto main with `Pipeline CI: failure`. Any PR rebased on top sees the failure as its own. There is no PR-merge gate that requires green CI for the "Pipeline CI" workflow.
- **Pre-commit runs the full staged-file test set against the local .venv, not the locked toolchain.** `uv.lock` pins DuckDB 1.4.4 but the .venv that pre-commit activates can drift to 1.5.x. Tests that pass in CI can fail locally with no obvious signal.
- **Test isolation under hostile env is not part of the suite's contract.** `subprocess.run(..., cwd=tmp_path)` was assumed to honour `cwd=` regardless of parent-process git env. No fixture-level guard, no smoke test that runs the suite with `GIT_DIR` artificially set.

## Permanent fixes (deferred — tracked here, not in this PR)

1. **Branch-protection rule on `main`** requiring `Pipeline CI / checks` to be green before merge. Apply via GitHub branch settings or `.github/branch-protection.yml` if a programmatic approach is preferred. Stops red commits from cascading onto every downstream PR.
2. **Pre-commit hook should pin its Python toolchain to `uv.lock`** rather than activating whatever `.venv` happens to be present. Either run `uv run pytest ...` or fail-fast if `.venv` has versions diverging from `uv.lock`. Catches local-only failures before they sink half a session.
3. **Top-level `tests/conftest.py` autouse fixture** that drops `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE`, `GIT_OBJECT_DIRECTORY`, `GIT_COMMON_DIR`, `GIT_NAMESPACE`, `GIT_PREFIX` for every test. PR #108 puts this on the two affected classes; lifting to conftest immunises every future `subprocess git` call site.
4. **Ralph-loop "ship-readiness" gate**: before any iter-N commit merges, require `ruff format --check`, `ruff check`, `pytest --collect-only`, and a `pytest -x -q` with `GIT_DIR` set. Stops iter-179-style leftovers at the source.

## Cost

- Direct: ~3 hours of session time, tens of thousands of tokens, four worktree create/destroy cycles, three force-pushes.
- Indirect: blocked PRs #101-#105 from merging, blocked the `dashboard-coordination` audit PR's deferred docstring fix, exhausted operator patience.

## Action queue

- [ ] Apply branch-protection rule on `main` for `Pipeline CI / checks`. (Operator action — GitHub UI.)
- [ ] Lift the autouse `_isolate_git_env` fixture from per-class to `tests/conftest.py`. (Trivial follow-up after #108 lands.)
- [ ] Tighten pre-commit to validate `.venv` matches `uv.lock` or use `uv run`. (Hook edit, separate PR.)
- [ ] Add a "Ralph ship-readiness" doc to `docs/governance/` if Ralph loops continue producing iter-N leftovers.
