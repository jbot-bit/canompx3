# Workflow Speed Audit — remove autopilot overhead from the hot path

**Date:** 2026-06-06  
**Status:** active  
**Owner:** cross-tool repo workflow  
**Scope:** startup, commit hooks, task routing, planning docs, and repeat verification loops

## Executive decision

The repo has accumulated safety gates from real incidents, but several were
promoted into the everyday create/design/implement/fix path when they belong in
pre-push, CI, explicit readiness, or live-capital workflows.

Default rule going forward:

> **Fast path first. Escalate only when the touched surface can actually break
> the protected thing.**

Safety is still mandatory for live money, DB truth, research claims, and schema
changes. The waste is running those same checks for docs-only notes, small design
iterations, or unrelated one-file fixes.

## Findings

### 1. Pre-commit was doing whole-repo Python work for staged-file commits

Before this audit, pre-commit linted `pipeline/ trading_app/ scripts/` and
format-checked `pipeline/ trading_app/ scripts/ tests/` every time, even when the
commit staged one doc or one small file.

Decision:

- pre-commit now computes staged path sets once;
- Ruff lint/format runs only on staged Python under `pipeline/`, `trading_app/`,
  `scripts/`, and `tests/`;
- full-project Ruff remains an explicit verification/CI responsibility.

Expected impact: docs-only and small-file commits stop paying whole-repo lint and
format startup/scan cost.

### 2. DB trade-window sync was an everyday commit tax

The validated-setups trade-window backfill is useful when code/runtime-data paths
may affect DB truth. It is overkill for docs-only/design commits because it can
start Python, inspect/mutate DB-adjacent state, and fail a commit unrelated to DB
freshness.

Decision:

- pre-commit runs the backfill only when staged paths include code/runtime-data
  surfaces such as `pipeline/`, `trading_app/`, selected live scripts,
  `research/`, migrations, or runtime YAML/JSON;
- docs-only commits skip it;
- drift/pre-push/CI remain the broader stale-state backstops.

### 3. CRG updates belong to code/navigation changes, not every note

`code-review-graph` is a navigation surface, not a truth surface. Updating it on
non-Python commits is noise.

Decision:

- pre-commit skips the CRG update when no Python is staged;
- when Python is staged, the existing advisory/failure-counter behavior remains.

### 4. Startup/context overhead is real, but the existing resolver is the right front door

Measured token hygiene reported:

- 3 startup docs loaded by default;
- 35 Claude rules total, 13 always-on;
- 70 active stage files;
- top always-on rules dominated by long examples/rationale.

Decision:

- do **not** add more startup doctrine for this audit;
- keep using `context_resolver.py --task ...` to avoid broad cold-start wandering;
- future cleanup should split long always-on rules into short rule stubs plus
  opt-in appendices/skills.

### 5. Active plan inventory is noisy

There are many active plans across April–June. Several are likely design history,
not active work. This slows orientation because agents keep rediscovering stale
workstreams.

Decision for next cleanup pass:

- archive/close stale active plans in a dedicated docs-only commit;
- preserve live/readiness/research plans that still gate capital or DB truth;
- do not mass-delete history.

## New speed policy

### Hot path: create/design/implement/fix

Use this for normal repo work:

1. Resolve context narrowly.
2. Edit the smallest relevant files.
3. Run staged/targeted checks.
4. Commit.
5. Escalate only if touched paths demand it.

### Escalation path: live/research/DB/schema/capital

Use full gates when touching:

- live execution, broker, webhook, account routing, or position sizing;
- pipeline schema, canonical DB writes, migrations, or data provenance;
- research claims, promotion criteria, audit results, or statistical thresholds;
- hook/launcher/session isolation logic.

### Do not autopilot these for every task

- full drift/profile drift;
- full trading app test suite;
- full audit phases;
- broad plan archaeology;
- multi-agent review;
- external doc/web lookup unless current truth or spend/risk requires it.

## First patch landed in this audit

Changed `.githooks/pre-commit` so the commit hot path is path-scoped:

- staged path sets are computed once;
- Ruff lint/format only staged Python files;
- trade-window sync only code/runtime-data paths;
- CRG update only Python changes;
- existing drift, tests, checkpoint, behavioral audit, claim hygiene, and syntax
  gates remain in place.

## Remaining candidates for future simplification

1. **Always-on Claude rules:** split long examples/rationale out of always-on
   files into opt-in references.
2. **Active plans:** archive stale April/May plans so `project_pulse` and human
   orientation see the real queue.
3. **Drift registry:** continue the staged drift metadata work so code commits
   run only the checks that match touched surfaces.
4. **Session preflight:** keep launch preflight for mutating sessions, but provide
   a documented read-only/design entry that avoids DB/env setup when no code will
   run.
5. **Hook timing ledger:** keep the existing pre-commit timing output and review
   the slowest stages monthly; anything advisory and non-truth should default to
   skip/fail-open outside CI.

## Verification notes from this audit

- `python3 scripts/infra/codex_local_env.py doctor --platform wsl` could not
  complete setup because PyPI download through the tunnel failed for `propcache`.
- `python scripts/tools/project_pulse.py --fast` and
  `python scripts/tools/system_context.py --format text --max-lines 120` could
  not run under the ambient interpreter because `yaml` is unavailable without the
  WSL venv.
- `python3 scripts/tools/token_hygiene_report.py` completed and supplied the
  context-overhead measurements above.


## 2026-06-07 current-state recheck

Follow-up audit: `docs/audits/2026-06-07-precommit-hotpath-current-state-audit.md`.

Current findings:

- Local Git initially had `core.hooksPath=<unset>`, so the tracked hook speed/safety work was not active in this checkout. The session applied `git config core.hooksPath .githooks`; this is local repo config, not a tracked file change.
- `project_pulse.py --fast --format json` and `system_context.py --context codex-wsl --action orientation` now run under ambient Python in this checkout; `.venv-wsl` is still missing, so the repo-managed interpreter warning remains real.
- Pre-commit now blocks staged Python files that also have unstaged working-tree changes before Ruff/format/syntax steps, preventing the staged-file fast path from checking or re-staging unrelated hunks.
