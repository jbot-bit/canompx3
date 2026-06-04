# Drift / Pre-commit / Commit-Push Speed Audit - 2026-06-04

## Status

Active implementation plan, v2. This is a workflow-control plan, not a trading or research result.

## North Star

Make ordinary local commits feel **supersonic** while preserving institutional rigor by moving proof to the cheapest event that can still prove the claim.

Target operator experience:

| Path | Target | What runs |
| --- | ---: | --- |
| Docs-only / plan-only commit | p50 <1s, p95 <3s | hook/branch/line-ending/claim-hygiene checks only |
| Small Python commit | p50 <3s, p95 <8s | staged Ruff, staged compile, mapped smoke tests, scoped cheap drift |
| Pipeline/trading/runtime commit | p50 <8s, p95 <20s | all small-Python gates plus relevant DB/trading drift slice |
| Push / branch publish | p50 <90s, p95 <4min | full blocking integration gate, expanded tests, full-tree lint |
| Merge/readiness/deploy/capital claim | no UX cap | full drift, CI shards, audits, DB freshness/provenance proof |

The speed thesis is **not** "do less rigor." It is: **do less duplicated work at commit time, prove every skip, and move heavy global truth checks to pre-push/CI/readiness where they belong.**

## Current Measured State

Measured in `/workspace/canompx3` on 2026-06-04.

| Probe | Result | Implication |
| --- | --- | --- |
| `python scripts/tools/session_preflight.py` | Blocker: `core.hooksPath=<unset>`; expected `.githooks` | Local hook rigor is not reliably active in this checkout. Speed work must first make the gate visible/installed, otherwise "faster" can become "absent". |
| `python scripts/tools/profile_check_drift.py` | 191.79s across 198 checks; 35 checks >200ms; 18 checks >1s | Full drift is too expensive for high-frequency edit/commit loops. |
| `python -u pipeline/check_drift.py --fast --quiet --skip-crg-advisory` | 18.51s and failed 10 DB-backed checks in this WSL checkout | Even fast drift is not currently sub-5s here, and local DB state can block workflow unrelated to the staged code. |
| `.githooks/pre-commit` | Exists; no `.githooks/pre-push` exists | The repo has one heavy local gate, but no native push-time tier for expensive checks. |

Slowest `profile_check_drift.py` checks in this run:

| Seconds | Check | Supersonic disposition |
| ---: | --- | --- |
| 72.689 | `FAST_LANE PROMOTE queue: no orphan PROMOTEs, no ERROR entries, cache up to date` | Never every-commit. Scope to fast-lane/promote-queue inputs; otherwise pre-push/CI. |
| 22.353 | `DSR reference-universe lock declared...` | Scope to prereg/DSR/result docs; otherwise pre-push/CI. |
| 22.180 | `AM3.3 audit-log/prereg theory_grant parity...` | Scope to prereg/audit-log/theory-grant/ranker inputs; otherwise pre-push/CI. |
| 17.576 | `Doc hygiene contracts (stamps, design-only, generated markers)` | Split into staged-doc cheap checks plus full-doc pre-push/CI sweep. |
| 5.964 | `Fast-lane status roll-up reconstruction parity...` | Scope to fast-lane state graph/status/graveyard inputs; otherwise pre-push/CI. |
| 5.137 | `Stage-file staleness vs landed commits` | Advisory by default; commit-time only when stage files are staged. |
| 4.950 | `Tests forbid canonical production runtime-path literals...` | Convert to path-scoped AST scan over staged tests; full-tree in pre-push/CI. |
| 4.871 | `Research scans reading canonical layers + filter/E2 must call a canonical guard...` | Scope to staged `research/` files; full-tree in pre-push/CI. |

## Current Setup Audit

### What is already good

- The hook serializes concurrent commits before the expensive section, preventing two terminals from racing Git ref writes.
- The hook has stage timing, so speed regressions are visible after every run.
- Drift already has a `--fast` mode, advisory skipping, CRG advisory skipping, shared read-only DB connection reuse, and content-hash pass caching for declared non-DB dependencies.
- Pre-commit tests are staged-file-aware: staged tests run directly, pipeline changes route to `tests/test_pipeline/`, trading app changes route to a curated fast subset, and docs/scripts-only commits skip pytest.
- Drift has a cache honesty meta-check at the tail, which is important because stale cached PASS results are worse than slow checks.

### Friction / risk found

1. **Hooks can be absent.** `session_preflight.py` reports `core.hooksPath` unset in this checkout, so Git will not run `.githooks/pre-commit` until `git config core.hooksPath .githooks` is applied.
2. **Commit-time drift is still effectively full blocking drift.** The hook invokes `pipeline/check_drift.py --skip-crg-advisory --skip-advisory`, not a path-scoped or truly fast gate; it drops advisory cost but still runs every blocking slow check.
3. **There is no pre-push tier.** Expensive but high-value checks have nowhere native to run except pre-commit/manual/CI.
4. **The drift registry is label-tuple driven.** The `CHECKS` tuple is compact, but check identity, path scope, expected cost, and hook stage are implicit. That makes path-aware routing risky because it would have to infer semantics from labels or ad-hoc lists.
5. **DB-backed checks can block local commits due to checkout-local data state.** In this WSL run, fast drift failed 10 DB-backed checks. Some failures may be legitimate, but a commit that only changes docs or hook plumbing should not necessarily be hostage to stale or mismatched local DB truth unless it touches DB/trading/runtime surfaces.
6. **Ruff runs over whole source trees at commit time.** Ruff is fast and cached, but the hook still asks it to scan `pipeline/ trading_app/ scripts/` and format-check `pipeline/ trading_app/ scripts/ tests/` for every commit instead of defaulting to staged Python files.
7. **The hook does not have a hot/no-op fast path.** A docs-only commit still pays Python startup and broad gate orchestration before it can prove there is nothing relevant to run.

## External Best-practice Research

### Official sources

- Git's `pre-commit` hook is allowed to abort a commit, but it is explicitly bypassable with `--no-verify`; therefore local hooks should be fast feedback, not the only final assurance layer. Source: <https://git-scm.com/docs/githooks#_pre_commit>.
- Git's `pre-push` hook receives the refs being pushed on stdin and can abort the push before transfer; this is the right native place for heavier checks that are too slow for every commit. Source: <https://git-scm.com/docs/githooks#_pre_push>.
- The `pre-commit` framework normally runs hooks against currently staged files; it recommends `--all-files` for manual/CI use, and its docs explicitly warn that always running a hook on all files is slow and deviates from normal expectations. Source: <https://pre-commit.com/>.
- The `pre-commit` framework supports separate hook stages including `pre-commit`, `pre-push`, and `manual`, so tiering by event is first-class rather than a workaround. Source: <https://pre-commit.com/>.
- Ruff is built for fast local hooks and has built-in caching, but it also accepts explicit file lists; the natural local pattern is staged-file lint/format plus full lint in CI or explicit verification. Source: <https://docs.astral.sh/ruff/> and <https://docs.astral.sh/ruff/configuration/>.
- Pytest has built-in cross-run cache support, including `--last-failed` and `--failed-first`; local loops can prioritize known failures without making that the final gate. Source: <https://docs.pytest.org/en/stable/how-to/cache.html>.
- uv ships official pre-commit integration for lockfile consistency when `pyproject.toml` or requirements inputs change; dependency sync checks should be path-triggered, not unconditional. Source: <https://docs.astral.sh/uv/guides/integration/pre-commit/>.

### Unofficial / community practice

- `lint-staged` popularized the staged-file pattern: run linters/formatters only on staged files in the commit hook, while retaining full checks elsewhere. Source: <https://www.npmjs.com/package/lint-staged>.
- Experienced-developer discussions repeatedly converge on: pre-commit should be quick staged-file lint/format/smoke checks; full suites belong in CI or pre-push; because hooks can be skipped, CI/server checks remain the final authority. Sources: <https://www.reddit.com/r/ExperiencedDevs/comments/144fcqo/what_are_your_precommit_hooks/> and <https://survivejs.com/maintenance/infrastructure/automation/>.

## Supersonic Architecture

### 1. Hot-path hook classifier

Add a tiny classifier before Python-heavy work:

```text
staged files -> change class -> hook plan
```

Change classes:

| Class | Examples | Commit-time action |
| --- | --- | --- |
| `docs_plan_only` | `docs/plans/**`, non-result markdown | no drift; markdown/claim hygiene only |
| `audit_result_docs` | `docs/audit/results/**/*.md`, prereg docs | claim hygiene + relevant prereg/audit drift slice |
| `tooling_only` | `.githooks/**`, `scripts/tools/**`, `tests/test_tools/**` | staged Ruff/compile + mapped tool tests |
| `pipeline_code` | `pipeline/**/*.py`, migrations | staged Ruff/compile + mapped pipeline tests + DB/pipeline drift slice |
| `trading_runtime` | `trading_app/**`, `scripts/run_live_session.py`, profiles/allocation | fail-closed: mapped tests + live/runtime drift slice |
| `research_scan` | `research/**`, fast-lane/prereg surfaces | mapped research tests + research/prereg drift slice |
| `dependency` | `pyproject.toml`, `uv.lock`, `.python-version` | uv/lock/python-version checks + full lint smoke |
| `unknown_or_global` | governance, root config, drift registry | fail-closed to current broad gate or pre-push-required commit marker |

This classifier must print the plan before running it, e.g.:

```text
HOOK PLAN: class=docs_plan_only, py=0, drift=0, pytest=0, prepush_required=false
```

### 2. Typed drift registry

Replace the implicit tuple as the source of planning truth with metadata while preserving the current tuple API for compatibility.

Required fields:

- stable `id`: machine key that never changes when labels are edited
- existing `label`
- `fn`
- `is_advisory`
- `requires_db`
- `cost_class`: `cheap | normal | slow | heavyweight`
- `stages`: `post_edit_fast | pre_commit | pre_push | ci | manual`
- `path_globs`: canonical input files/globs that make the check relevant
- `always_when`: optional change classes that force inclusion
- `cache_policy`: `none | file_deps | tree_deps | db_sensitive | external_tool`
- `capital_class`: boolean, forcing conservative routing for live/capital surfaces
- `max_commit_ms`: target budget for checks allowed at commit time

Fail-closed rule: a blocking check with no `path_globs` and no explicit `stages` must remain broad until classified.

### 3. Two-level drift execution

Add explicit drift modes:

| Mode | Command shape | Purpose |
| --- | --- | --- |
| `--staged --budget-ms 3000` | commit hot path | runs cheap global + scoped checks; errors if selected checks exceed budget unless capital-class |
| `--changed-from <ref> --budget-ms 90000` | pre-push | runs all checks relevant to branch delta plus global blocking checks |
| `--full-blocking` | CI/local integration | all blocking checks, advisory optional |
| `--full` | audit/readiness | all checks, including advisory and CRG |
| `--explain-selection` | any mode | emits selected/skipped reasons for reviewability |

Every mode should write a small JSON sidecar under `.git/canompx3/verification/last-drift-selection.json` containing command, selected check IDs, skip reasons, timings, cache hits, and DB availability.

### 4. Persistent timing ledger

Promote the timing data from ephemeral terminal output into a local JSONL ledger:

```text
.git/canompx3/timing/precommit.jsonl
.git/canompx3/timing/drift.jsonl
.git/canompx3/timing/prepush.jsonl
```

Uses:

- automatically demote checks whose p95 commit-time cost exceeds their budget unless they are capital-class;
- identify cache misses that should have hit;
- prove the plan is actually getting faster;
- surface "new slow check added without metadata" in CI.

### 5. Staged-file Ruff and compile only

Commit hook should run Ruff and `py_compile` only over staged Python files by default:

```bash
STAGED_PY=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)
ruff check $STAGED_PY
ruff format --check $STAGED_PY
python -m py_compile $STAGED_PY
```

Full-tree Ruff belongs in pre-push and CI. If a global config file changes (`ruff.toml`, `pyproject.toml`), run full-tree Ruff at commit time because the config can affect every file.

### 6. Test impact map

Create `scripts/tools/test_impact_map.py` with a small explicit map first, then optionally a generated import graph later.

Initial examples:

| Changed path | Commit tests |
| --- | --- |
| `.githooks/pre-commit`, `.githooks/pre-push` | `tests/test_tools/test_git_hooks_env.py` plus hook smoke tests |
| `scripts/tools/workflow_doctor.py` | `tests/test_tools/test_workflow_doctor.py` |
| `scripts/tools/worktree_guard.py` | `tests/test_tools/test_worktree_guard.py` |
| `pipeline/check_drift.py` | drift metadata/selection tests + focused check tests, not full pipeline suite |
| `pipeline/dst.py` | DST/session focused tests + selected drift checks |
| `trading_app/live/**` | live preflight/runtime-control tests + live audit phase when pushing |

Commit-time tests should be selected and short. Pre-push expands to all changed-package tests. CI remains the final full matrix.

### 7. DB truth lanes

Split DB-backed checks into three classes:

| DB class | Commit behavior | Push/readiness behavior |
| --- | --- | --- |
| `db_required_for_touched_surface` | run if staged files can affect DB/trading/runtime truth | run |
| `db_snapshot_ok` | use last verified read-only snapshot hash if staged files cannot affect DB truth | refresh on push |
| `db_global_health` | never every-commit unless DB files/config changed | run pre-push/CI/readiness |

No DB write is authorized by this plan. Missing/stale DB evidence should block readiness/deploy claims, not every docs-only commit.

### 8. Parallel pre-push runner

Pre-push should run independent lanes concurrently with bounded output:

- lane A: full-tree Ruff / format check
- lane B: full blocking drift (`--full-blocking --skip-advisory --skip-crg-advisory` initially)
- lane C: changed-package tests
- lane D: hook/config/governance checks

Fail fast on the first hard failure, but keep enough output to identify all lanes that had already failed. This makes push-time rigor faster without deleting checks.

## Proposed Verification Tiers

### Tier 0 - Always-on commit hygiene, target <1-3s

Runs in `.githooks/pre-commit` for every commit:

- commit serialization / branch-flip / stale lock guard
- dev-dep and shell/venv guard only when Python work is needed, not for pure docs-only hot path
- staged CRLF renormalization
- staged Python `ruff check` and `ruff format --check` / auto-format restage
- syntax compile on staged Python files
- claim hygiene for staged audit result docs
- checkpoint guard / behavioral audit only if measured p95 stays sub-second or relevant files changed

Coverage principle: blocks cheap, deterministic hazards that directly affect the staged commit.

### Tier 1 - Path-scoped commit checks, target <3-20s by change class

Runs in `.githooks/pre-commit` only when staged files match registered scopes:

- relevant focused pytest target(s)
- selected drift checks whose declared `path_globs` intersect staged files
- uv lock check only when `pyproject.toml`, `uv.lock`, or dependency files change
- DB-backed drift only when staged files can change DB/trading/runtime truth: `pipeline/`, `trading_app/`, `research/` promotion surfaces, migrations, profile/allocation docs, or runtime config

Coverage principle: expensive checks must declare what inputs make them relevant. Unknown or missing scope fails closed to Tier 2 or the current broad gate, not silently skipped.

### Tier 2 - Pre-push integration gate, target <90s p50 / <4min p95

Add `.githooks/pre-push` for pushes of local branches:

- full blocking drift: `python -u pipeline/check_drift.py --quiet --skip-advisory --skip-crg-advisory`
- ruff full-tree check/format-check
- changed-package test expansion since merge-base, with `pytest --failed-first` so known failures surface early
- optional DB-backed checks when canonical DB exists and is not locked
- run independent lanes in parallel after Stage E lands

Coverage principle: pushing is less frequent than committing, so it can carry the expensive integration burden without slowing every local checkpoint.

### Tier 3 - CI / readiness / deploy / capital gate, unbounded by local UX

Runs before merge, promotion, deploy, live readiness, or capital-class claims:

- full drift including advisory/CRG where appropriate
- full test matrix / shard CI
- system audit phase(s) relevant to touched surfaces
- DB freshness/provenance proofs for research/live claims
- `project_pulse.py --fast` and task-specific verification profile from `context_resolver.py`

Coverage principle: final truth remains CI/readiness evidence, not a local hook that can be skipped.

## Implementation Roadmap

### Stage 0 - Baseline and safety rails, no behavior change

Deliverables:

- make hook activation loud: exact `git config core.hooksPath .githooks` fix in `session_preflight.py` / `workflow_doctor.py`;
- add `scripts/tools/install_git_hooks.py` if we want one-command repair;
- add timing ledger writer for pre-commit/drift/pre-push;
- add a `workflow_doctor.py drift --json` field for p50/p95 and last full-drift evidence age.

Verification:

- tests for unset hook-path messaging;
- `git diff --check`;
- no gate semantics changed.

### Stage 1 - Supersonic docs/tooling hot path

Deliverables:

- hook classifier emits `HOOK PLAN`;
- docs-plan-only commits skip drift and pytest;
- staged Ruff/format/compile only;
- full-tree Ruff still runs when global config changes;
- claim hygiene remains for staged result docs.

Expected win:

- docs-only commit drops from broad pre-commit cost to sub-1-3s.

Safety tests:

- staged docs-plan file selects no drift;
- staged audit-result doc selects claim hygiene;
- staged `pyproject.toml` selects full Ruff/dependency checks;
- unknown root config falls back broad/pre-push-required.

### Stage 2 - Drift metadata compatibility layer

Deliverables:

- add `DriftCheck` metadata;
- generate legacy `CHECKS` tuple from metadata so existing tests continue to pass;
- validate metadata at import: unique IDs, no missing labels, capital-class checks cannot be commit-skipped without explicit scope;
- add `scripts/tools/profile_check_drift.py --json` to write cost classes.

Expected win:

- no speed win yet; creates safe substrate for speed.

Safety tests:

- legacy `CHECKS` order unchanged;
- duplicate ID fails;
- blocking unscoped heavyweight check is not silently skipped.

### Stage 3 - Path-scoped drift selection

Deliverables:

- `pipeline/check_drift.py --staged --explain-selection`;
- `pipeline/check_drift.py --changed-from <ref> --explain-selection`;
- selection JSON sidecar;
- first four heavyweight checks receive scopes and tests.

Expected win:

- ordinary commits avoid 70s/22s/22s/17s heavyweight checks unless relevant files are staged.

Safety tests:

- touching `docs/runtime/promote_queue.yaml` selects fast-lane promote queue parity;
- touching unrelated docs skips it and records skip reason;
- touching prereg/audit-log surfaces selects DSR/theory checks;
- touching `pipeline/check_drift.py` selects drift meta-tests and conservative global checks.

### Stage 4 - Test impact map

Deliverables:

- `scripts/tools/test_impact_map.py --staged`;
- explicit path-to-test rules for hooks/tools/drift/pipeline/trading/live/research;
- pre-commit uses mapped tests, pre-push expands changed-package tests.

Expected win:

- pipeline edits stop paying the whole pipeline test directory when a focused mapped set exists.

Safety tests:

- each mapped production file has at least one test target;
- unknown Python production file falls back to package smoke tests;
- live/capital paths force live safety tests.

### Stage 5 - Pre-push integration hook

Deliverables:

- `.githooks/pre-push` with full blocking drift, full-tree Ruff, changed-package tests;
- parallel lane runner after serial version is stable;
- clear bypass policy: local `--no-verify` can skip Git hooks, but PR/CI/readiness cannot.

Expected win:

- commit path becomes fast; push absorbs integration proof.

Safety tests:

- pre-push parses stdin refs;
- delete/tag pushes do not run branch checks unless configured;
- branch push runs Tier 2 once;
- lane failure blocks push and prints exact manual command.

### Stage 6 - DB snapshot and readiness evidence

Deliverables:

- DB-backed drift checks classified into `db_required_for_touched_surface`, `db_snapshot_ok`, `db_global_health`;
- last verified DB snapshot hash/mtime evidence recorded locally;
- readiness/deploy commands require fresh DB evidence even if commits do not.

Expected win:

- docs/tooling commits are no longer hostage to stale local DB state;
- runtime/research/readiness remains fail-closed.

Safety tests:

- touching profiles/allocation forces DB lane;
- docs-only commit can skip DB lane with recorded reason;
- readiness command fails if DB proof is stale/missing.

### Stage 7 - Guard against speed regressions

Deliverables:

- CI check validates every commit-stage check has `max_commit_ms` and recent measured p95 below budget or explicit waiver;
- `workflow_doctor.py drift` shows top regressions and recommended demotions;
- new drift checks must declare metadata before landing.

Expected win:

- speed stays fast after the initial cleanup.

Safety tests:

- adding a heavyweight check without metadata fails;
- stale timing ledger does not block CI but warns locally;
- capital-class checks cannot be demoted by timing alone.

## First Supersonic Patch Set

Do these first, in order, because they create a big UX win without weakening drift semantics:

1. **Hook activation repair:** improve `session_preflight.py` / `workflow_doctor.py` messaging and/or add `install_git_hooks.py`.
2. **Staged Ruff/compile:** switch commit hook from whole-tree Ruff/format to staged-file Ruff/format unless config changed.
3. **Docs-only hot path:** classify `docs/plans/**` and non-result markdown as no-drift/no-pytest commit path.
4. **Pre-push serial hook:** add full blocking drift + full Ruff + changed tests before moving more checks out of commit.
5. **Drift metadata substrate:** add typed metadata and selection explanation in compatibility mode.
6. **Scope the four whale checks:** promote queue, DSR lock, AM3.3 theory parity, doc hygiene.

Expected result after Patch Set 1:

- docs-only commits: sub-1-3s;
- small Python/tooling commits: mostly sub-3-8s;
- heavy global proof preserved at pre-push/CI/readiness;
- every skipped heavy check has a printed reason and a pre-push backstop.

## Non-goals / Safety Boundaries

- Do not relax live/capital gates.
- Do not skip DB-backed checks for commits that touch runtime profiles, allocation, migrations, pipeline outputs, or strategy promotion surfaces.
- Do not remove the cache honesty meta-check.
- Do not make advisories silently disappear from CI/manual full drift.
- Do not replace final CI/readiness verification with local hooks.
- Do not add background daemons that mutate repo or DB state.
- Do not let timing auto-demote capital-class checks.
