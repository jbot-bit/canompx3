# Current-state audit — pre-commit hot path and workflow-speed fixes

**Date:** 2026-06-07  
**HEAD audited before follow-up patch:** `9539661 pre-commit: path-scoped hot path (staged-file Ruff/format, conditional backfill/CRG) and workflow-speed docs`  
**Scope:** verify the previous pre-commit/process-speed fixes against the repo state that is actually present now, then identify safe improvements.  
**Evidence boundary:** local checkout, local git history, static code/docs/tests, and local command output only. No remote/cloud artifacts, live DB, MCP output, secrets, dashboards, or unpushed branches were assumed.

## Commands/evidence used

- `git log --oneline -12`
- `git show --stat --oneline HEAD`
- `python scripts/tools/context_resolver.py --task "audit current repo state for previous pre-commit path scoped hot path workflow speed fixes after repo moved; find regressions and improvements" --format markdown`
- `bash -n .githooks/pre-commit`
- `python3 scripts/tools/token_hygiene_report.py`
- `python scripts/tools/project_pulse.py --fast --format json`
- `python scripts/tools/system_context.py --context codex-wsl --action orientation`
- Static reads of `.githooks/pre-commit`, `.githooks/pre-push`, `tests/test_tools/test_git_hooks_env.py`, `docs/audits/2026-06-06-codex-process-debt-local-only.md`, and `docs/plans/active/2026-06/2026-06-06-workflow-speed-audit.md`.

## Findings

### 1. Hook speed work existed but local Git was not using it

- **Process/file/path:** local Git config `core.hooksPath`; `.githooks/pre-commit`; `.githooks/pre-push`.
- **Original purpose:** the hook changes only help if Git actually dispatches `.githooks/*`.
- **Current local evidence:** `system_context.py --context codex-wsl --action orientation` initially warned: `Git pre-commit guardrail is not active. (core.hooksPath=<unset>; expected .githooks)`.
- **Cost/friction:** speed work can look landed while commits bypass both the new fast path and the safety gates.
- **Benefit of keeping:** repo-local hooks are the intended fast feedback tier.
- **Risk if removed/ignored:** high; every local hook guarantee becomes documentation-only.
- **Recommendation:** **AUTOMATE / KEEP**.
- **Safe minimal change applied in this session:** set local config with `git config core.hooksPath .githooks`. Re-running `system_context.py` removed the hook-inactive warning. This is local repo state, not a tracked file change.
- **Future improvement:** make session/preflight and workflow-doctor print the exact fix command prominently and consider an explicit `--fix-hooks-path` helper.

### 2. Path-scoped Ruff/format is aligned with current pre-commit tests, but mixed staged/unstaged Python was unsafe

- **Process/file/path:** `.githooks/pre-commit`; `tests/test_tools/test_git_hooks_env.py`.
- **Original purpose:** speed up commits by running Ruff only on staged Python paths.
- **Current local evidence:** the hook computes staged file sets once and scopes Ruff to staged Python under `pipeline/`, `trading_app/`, `scripts/`, and `tests/`; existing tests already cover the path-scoped drift classifier and pre-push full-drift gate.
- **Cost/friction:** low for clean staged files.
- **Benefit:** avoids whole-tree lint/format work for docs-only or small commits.
- **Risk found:** Ruff checks working-tree paths, not staged blobs; Ruff format also re-stages whole files. If a staged Python file also had unstaged hunks, the hook could verify/format/stage content outside the intended commit.
- **Recommendation:** **SIMPLIFY with safety guard**.
- **Safe minimal change applied:** pre-commit now blocks staged Python files that also have unstaged working-tree changes before Ruff or syntax checks run.

### 3. Conditional trade-window backfill remains sound but should stay code/runtime-data-only

- **Process/file/path:** `.githooks/pre-commit` step `[2b/8] Trade window sync`.
- **Original purpose:** keep validated setup trade-window metadata synchronized when code/runtime data may affect DB truth.
- **Current local evidence:** the hook runs the backfill only when `STAGED_CODE_OR_DATA` matches `pipeline/`, `trading_app/`, selected live scripts, `research/`, migrations, or runtime YAML/JSON.
- **Cost/friction:** avoids DB-adjacent work for docs-only commits.
- **Benefit:** preserves fail-stop behavior for surfaces likely to affect DB-backed truth.
- **Risk if removed:** stale validated setup windows could survive code/data changes.
- **Recommendation:** **KEEP**.
- **Safe minimal change:** none in this pass.

### 4. CRG gating is locally aligned, but advisory graph work should remain out of docs-only commits

- **Process/file/path:** `.githooks/pre-commit` step `[3b/8] CRG update`; `tests/test_pipeline/test_check_drift_crg.py`.
- **Original purpose:** keep the code-review graph useful for navigation and drift advisory checks.
- **Current local evidence:** the hook now skips CRG when no Python is staged and keeps existing advisory/failure-counter semantics when Python is staged.
- **Cost/friction:** removing CRG from docs-only commits avoids non-truth navigation work.
- **Benefit:** Python changes still refresh the graph path.
- **Risk if removed:** graph can stale across code changes.
- **Recommendation:** **KEEP**.
- **Safe minimal change:** none in this pass.

### 5. The process-debt audit doc was stale relative to the final applied commit

- **Process/file/path:** `docs/audits/2026-06-06-codex-process-debt-local-only.md`.
- **Original purpose:** record pass-1 process debt without assuming remote/DB state.
- **Current local evidence:** it listed an intermediate audited commit `97e4900`, while the current repo has the final applied commit `9539661` on top of `99caaa4`.
- **Cost/friction:** future readers may audit the wrong baseline.
- **Benefit:** the report content is still useful.
- **Risk if ignored:** medium documentation drift around exactly which patch was reviewed.
- **Recommendation:** **SIMPLIFY / UPDATE**.
- **Safe minimal change applied:** update the report header to name the current applied commit and cross-link this current-state audit.

### 6. Environment warnings in the speed doc were stale after current-state recheck

- **Process/file/path:** `docs/plans/active/2026-06/2026-06-06-workflow-speed-audit.md`.
- **Original purpose:** preserve local verification notes from the first pass.
- **Current local evidence:** current recheck showed `project_pulse.py --fast --format json` and `system_context.py --context codex-wsl --action orientation` both run under ambient Python, while `.venv-wsl` is still missing.
- **Cost/friction:** stale warnings can make the environment look more broken than it is.
- **Benefit:** original warning explained the first-pass limitation.
- **Risk if ignored:** low/medium; confuses future workflow-speed decisions.
- **Recommendation:** **UPDATE**.
- **Safe minimal change applied:** add a 2026-06-07 current-state note rather than deleting historical evidence.

## Other improvements found

1. **Hook activation should be a first-class repair path.** The repo already detects inactive hooks; the next improvement is a command or preflight flag that applies `git config core.hooksPath .githooks` intentionally.
2. **Pre-commit should eventually operate on staged blobs for lint/format.** The mixed-change guard prevents accidental staging now; a structural improvement would run Ruff against a temporary staged tree or use a real pre-commit framework pattern.
3. **Active-stage clutter remains real.** Current `system_context.py` reports 21 active stage files and token hygiene still reports 70 active stage files by its own measurement. Reconcile those definitions and archive proven-done stages.
4. **Codex repo-local default is still not implemented.** The pass-1 audit recommendation still stands: Codex startup should not assume DB/MCP availability unless the task requires it.

## Result

The previous patch was mostly aligned with the current repo state, but two follow-up fixes were needed:

1. Activate local hooks in this checkout with `git config core.hooksPath .githooks`.
2. Add a mixed staged/unstaged Python guard so staged-file Ruff/format cannot accidentally verify or stage unrelated local hunks.
