# Codex Process-Debt Audit — local-only pass 1

**Date:** 2026-06-06
**Commit audited:** `9539661 pre-commit: path-scoped hot path (staged-file Ruff/format, conditional backfill/CRG) and workflow-speed docs`
**Scope:** workflows, checks, prompts, scripts, gates, docs, scans, git/worktree processes, and automation that may waste time or overcomplicate solo development.
**Constraint:** no code edits in pass 1. This report uses only files in this checkout, local git history, static code/docs/tests, and explicit terminal evidence from this session.
**Principle:** preserve truth, delete theatre.

## Evidence boundary

Local-only evidence used:

- `git log --oneline -10`
- `python scripts/tools/context_resolver.py --task "Codex local-only audit repo process debt workflows checks prompts scripts gates docs scans git worktree automation wasting time solo development pass 1 no code edits" --format markdown`
- `python3 scripts/tools/token_hygiene_report.py`
- Static reads of `AGENTS.md`, `CLAUDE.md`, `CODEX.md`, `HANDOFF.md`, `.claude/settings.json`, `.githooks/pre-commit`, `.githooks/pre-push`, `pipeline/check_drift.py`, `scripts/audits/run_all.py`, `scripts/infra/codex-project.sh`, `scripts/infra/codex-project-search.sh`, `scripts/tools/session_preflight.py`, `docs/workflows/opencode.md`, and related plan/docs inventories.

Unavailable by constraint:

- Remote/cloud-only artifacts.
- Live DBs, `gold.db`, MCP outputs, secrets, dashboards, unpushed branches.
- Any claim requiring those is marked `NEED_REMOTE_EVIDENCE`.

## 2026-06-07 current-state re-audit note

This report was rechecked against current HEAD `9539661` in `docs/audits/2026-06-07-precommit-hotpath-current-state-audit.md`. That follow-up found the pass-1 recommendations still directionally valid, but added two concrete current-state corrections: local `core.hooksPath` had to be set to `.githooks`, and staged-file Ruff/format needed a mixed staged/unstaged Python guard.

## Top 10 process time-wasters

### 1. Broad drift gate remains the dominant iteration blocker

- **Process/file/path:** `.githooks/pre-commit`; `pipeline/check_drift.py`; `.githooks/pre-push`.
- **Original purpose:** catch repo drift before broken or stale truth leaves a developer machine.
- **Current local evidence:** pre-commit still routes any non-doc-safe staged path to `pipeline/check_drift.py --skip-crg-advisory --skip-advisory`; `pipeline/check_drift.py` is a 16,853-line registry/runner with DB-aware checks; pre-push runs the full drift command before every push.
- **Cost/friction:** small code edits can trigger a large all-purpose scanner rather than a path-relevant subset. Pushes can also pay the full drift cost.
- **Benefit:** protects canonical source-of-truth, lookahead, stale-doc, live/readiness, and config invariants.
- **Risk if removed:** high. Removing drift wholesale can allow stale trading/research truth, missing source updates, or live safety regressions.
- **Recommendation:** **SIMPLIFY**.
- **Safe minimal change:** in pass 2, add a repo-local, fail-closed staged drift selection mode that prints selected/skipped reasons. Unknown paths select the broad gate. DB-required checks must not silently pass from local absence; emit `NEED_REMOTE_EVIDENCE` when DB evidence is unavailable.

### 2. Pre-push full drift is safety theatre when DB evidence is unavailable locally

- **Process/file/path:** `.githooks/pre-push`; `pipeline/check_drift.py` DB skip behavior.
- **Original purpose:** act as the safety net for commit-time drift skips before work leaves the machine.
- **Current local evidence:** pre-push runs `pipeline/check_drift.py` without skip flags; the drift runner marks DB-busy/unavailable DB checks as skipped and still summarizes clean if no blocking non-DB violations remain.
- **Cost/friction:** can block or slow every push, but still cannot prove DB-backed truth in a checkout without the DB.
- **Benefit:** useful for static blocking checks and for local environments that really have the canonical DB.
- **Risk if removed:** medium/high for push discipline, especially on live/research changes.
- **Recommendation:** **SIMPLIFY**.
- **Safe minimal change:** split pre-push into `local-static-blocking` and `requires-db-evidence` sections. For DB-required checks, fail closed for protected surfaces by writing `NEED_REMOTE_EVIDENCE: pipeline/check_drift.py full run with canonical gold.db, including DB-required check summary` instead of treating unavailable DB as clean.

### 3. Codex launchers assume DB/MCP/local env availability by default

- **Process/file/path:** `scripts/infra/codex-project.sh`; `scripts/infra/codex-project-search.sh`; `CODEX.md`.
- **Original purpose:** give Codex repo-state, research-catalog, strategy-lab, and gold-db MCP context at session start.
- **Current local evidence:** both normal and search launchers attach `repo-state`, `research-catalog`, `strategy-lab`, and `gold-db` MCPs; both set up `.venv-wsl` with `uv sync` if absent; `CODEX.md` states normal/search launchers attach read-only `gold-db` by default.
- **Cost/friction:** startup can fail or slow down on missing network/cache/DB even for purely local code/design tasks. This violates the requested Codex default: repo-local, fail-closed, no DB assumptions.
- **Benefit:** strong context when trading-data truth is actually needed and available.
- **Risk if removed:** medium. Some research/live tasks need DB-backed truth.
- **Recommendation:** **SIMPLIFY**.
- **Safe minimal change:** make default Codex startup repo-local (`repo-state` only, no `gold-db`/strategy-lab unless requested). Add explicit `codex db`/`CANOMPX3_CODEX_WITH_DB=1` path that fails closed when DB is required but absent.

### 4. Claude hook stack is large enough to tax every action

- **Process/file/path:** `.claude/settings.json`; `.claude/hooks/*`.
- **Original purpose:** protect against wrong-branch edits, stale sessions, data-first violations, prompt drift, stage misuse, memory loss, and risky commands.
- **Current local evidence:** settings define 11 `PreToolUse` hook entries, 8 `PostToolUse` entries, 8 `UserPromptSubmit` entries, plus notification/session/memory hooks. Several fire on broad matchers such as `Bash`, `Edit|Write`, `Read|Bash`, or every prompt.
- **Cost/friction:** every action becomes a mini pipeline. Multiple 3-5 second timeouts stack, even if most hooks usually return quickly.
- **Benefit:** real protection against cross-tool collisions and live/research mistakes.
- **Risk if removed:** high if live/protected-surface guards are deleted indiscriminately.
- **Recommendation:** **MERGE**.
- **Safe minimal change:** keep hard guards for branch/worktree/live/protected paths, but merge prompt routers into a single classifier hook and demote purely advisory nudges to report-only summaries.

### 5. Active plans and stage files create orientation drag

- **Process/file/path:** `docs/plans/active/`; `docs/runtime/stages/`; `scripts/tools/stage_reaper_audit.py`.
- **Original purpose:** durable design decisions and stage state for cross-tool continuity.
- **Current local evidence:** local inventory shows 22 active plan files and 78 runtime stage files; `stage_reaper_audit.py` exists and can classify done-safe files for archival.
- **Cost/friction:** agents and humans repeatedly rediscover stale plans/stages, increasing context load and making the true queue harder to see.
- **Benefit:** avoids losing design history and open safety debts.
- **Risk if removed:** medium. Archiving the wrong stage can hide an unresolved live/research debt.
- **Recommendation:** **AUTOMATE**.
- **Safe minimal change:** run `stage_reaper_audit.py` in dry-run by default and archive only `DONE_SAFE` files in a docs-only pass. Add an active-plan index with `active | parked | archive-candidate` statuses instead of leaving all plans equally active.

### 6. Audits-of-audits are too heavy for everyday fixes

- **Process/file/path:** `.claude/skills/audit/SKILL.md`; `scripts/audits/run_all.py`; `docs/prompts/SYSTEM_AUDIT.md`; guardian prompts.
- **Original purpose:** comprehensive integrity review for data, research, live trading, build chain, docs, and CI.
- **Current local evidence:** `run_all.py` orchestrates 11 phases with up to 600 seconds per phase and stops on critical; `SYSTEM_AUDIT.md` defines extensive manual and mechanical sections; multiple guardian prompts exist for entry models, pipeline data, post-result sanity, prereg, and edge audits.
- **Cost/friction:** easy to invoke a full institutional audit when a targeted static check would do.
- **Benefit:** essential for promotion/readiness/live-capital decisions.
- **Risk if removed:** high for research integrity and live safety.
- **Recommendation:** **KEEP, but SIMPLIFY routing**.
- **Safe minimal change:** add a decision table: normal code fix = targeted tests; research claim = hypothesis protocol; live/capital = phase 7/readiness; broad system audit only by explicit request or release gate.

### 7. Worktree/session claims optimize for multi-agent collision, not solo speed

- **Process/file/path:** `AGENTS.md`; `CODEX.md`; `scripts/tools/session_preflight.py`; `scripts/infra/codex-project.sh`; `scripts/tools/worktree_manager.py`.
- **Original purpose:** prevent Claude/Codex/OpenCode from editing the same branch or stale state simultaneously.
- **Current local evidence:** project instructions prefer isolated worktrees for parallel sessions; Codex launch path runs session preflight claim checks; `session_preflight.py` manages read-only/mutating claims and imports the system context layer.
- **Cost/friction:** solo development can be forced through worktree/claim ceremony built for concurrent AI fleets.
- **Benefit:** strong protection during real parallel editing and live/protected work.
- **Risk if removed:** high only when multiple tools are active or protected surfaces are edited.
- **Recommendation:** **SIMPLIFY**.
- **Safe minimal change:** add a local solo mode that keeps dirty-branch and protected-surface checks fail-closed but makes stale peer-lease warnings advisory unless another live mutating process is actually evidenced in the checkout.

### 8. Optional third-POV/OpenCode workflow adds governance layers beyond current solo need

- **Process/file/path:** `docs/workflows/opencode.md`; `.githooks/pre-commit` OpenCode review gate.
- **Original purpose:** allow OpenCode as a read-only reviewer or disposable-worktree implementer without becoming a canonical authority.
- **Current local evidence:** OpenCode docs define required opening checks, managed-worktree build mode, forbidden surfaces, and a pre-commit review gate when `OPENCODE_AGENT_ACTIVE=1`.
- **Cost/friction:** another tool lane, another managed-worktree process, another review gate.
- **Benefit:** useful third POV for high-risk reviews.
- **Risk if removed:** low/medium for solo work; medium for adversarial review coverage.
- **Recommendation:** **KEEP read-only, SIMPLIFY mutating path**.
- **Safe minimal change:** keep OpenCode read-only. Require explicit operator approval for mutating OpenCode and do not advertise it as normal implementation flow.

### 9. Codex adapter layer has grown into a second orientation surface

- **Process/file/path:** `CODEX.md`; `.codex/*.md`; `.codex/skills/*`; `.agents/skills/*` wrappers.
- **Original purpose:** thin adapter over the canonical Claude layer.
- **Current local evidence:** `CODEX.md` says not to build a second project workflow, but it also lists many supporting docs: startup, workflows, project brief, current state, next steps, authority, workspace map, commands, agents, rules, integrations, memory, Codex standards, automations, and improvement plan.
- **Cost/friction:** Codex can spend time resolving Codex-specific docs instead of repo files and canonical rules.
- **Benefit:** helps Codex navigate a Claude-first repo.
- **Risk if removed:** low if a small index remains; medium if Codex loses critical launcher/env notes.
- **Recommendation:** **MERGE**.
- **Safe minimal change:** collapse `.codex/` docs into a single compact index plus task-specific references; archive stale narrative docs; keep skills as wrappers only.

### 10. Cloud/vendor automation scripts are discoverable enough to distract from repo-local work

- **Process/file/path:** `scripts/tools/sync_pinecone.py`; `scripts/tools/eval_openrouter_profiles.py`; `scripts/tools/m25_audit.py`; related prompt/docs references.
- **Original purpose:** optional knowledge sync, model/profile evaluation, and external-model audit/review support.
- **Current local evidence:** static text shows Pinecone sync reads `PINECONE_API_KEY`, OpenRouter evaluator is explicitly an offline envelope unless executed, and M25 audit requires `MINIMAX_API_KEY`.
- **Cost/friction:** secrets/vendor setup and cloud mental overhead can leak into ordinary local development.
- **Benefit:** useful when explicitly doing knowledge sync or external-model review.
- **Risk if removed:** low for core repo development; medium for optional AI-review workflows.
- **Recommendation:** **KEEP as opt-in, hide from default routing**.
- **Safe minimal change:** mark these scripts as optional/cloud-only in command routing and exclude them from default process suggestions unless the user explicitly asks for Pinecone/OpenRouter/MiniMax.

## Quick wins (<1h)

1. Add a `repo-local Codex` launcher/profile plan: default MCPs = `repo-state` only; DB/strategy lab opt-in.
2. Document a local-only drift evidence contract: DB-required checks must produce `NEED_REMOTE_EVIDENCE` instead of being inferred clean when DB is absent.
3. Run `stage_reaper_audit.py` dry-run and list archive candidates; do not apply until reviewed.
4. Add a short “normal fix workflow” doc: context resolver → edit → targeted tests → commit; no full audit unless touched surface requires it.
5. Mark cloud/vendor scripts as optional in the command index.

## Medium wins (<1 day)

1. Implement a staged drift selection dry-run mode with `--explain-selection`, no behavior change at first.
2. Merge the eight prompt-submit hooks into one classifier that emits one advisory packet.
3. Split Codex startup into `repo-local`, `db-required`, and `live-readiness` profiles.
4. Create an active-plan index with statuses and archive obvious stale plans after review.
5. Add a pre-push summary that separates local-static pass/fail from DB-required evidence status.

## Structural wins (<1 week)

1. Convert `pipeline/check_drift.py` from a monolithic all-purpose runner to a typed check registry with stages, path scopes, cache policy, DB requirement, and capital class.
2. Build a local verification ledger under `.git/canompx3/verification/` with timings and selected/skipped check reasons.
3. Reduce `.codex/` to a compact adapter index and move long explanations to archival/reference docs.
4. Turn stage/plan reaping into a regular report-only maintenance command with a reviewed apply step.
5. Define protected-surface classes once and reuse them across pre-commit, pre-push, Codex startup, session preflight, and OpenCode restrictions.

## Items requiring remote evidence

- `NEED_REMOTE_EVIDENCE: pipeline/check_drift.py full run against canonical gold.db, including DB-required pass/skip/fail summary.`
- `NEED_REMOTE_EVIDENCE: confirmation of whether default Codex sessions in the operator environment have reliable local gold.db symlink/access, or whether gold-db MCP startup commonly fails/slows sessions.`
- `NEED_REMOTE_EVIDENCE: pre-commit/pre-push timing ledger from the operator machine for docs-only, small Python, pipeline/trading, and push workflows.`
- `NEED_REMOTE_EVIDENCE: list of currently open unpushed branches/worktrees from the operator environment if worktree/session friction is caused by real concurrent work rather than stale local claims.`
- `NEED_REMOTE_EVIDENCE: cloud/vendor usage evidence before deleting Pinecone/OpenRouter/MiniMax scripts: last successful run date, current owner, and whether any active workflow depends on them.`

## Proposed PASS 2 patch plan only after approval

1. **Rework Codex default startup to repo-local.** Add/adjust a launcher profile that starts Codex with repo-state only; require explicit opt-in for DB/strategy-lab/gold-db MCPs.
2. **Add drift selection dry-run.** Implement `pipeline/check_drift.py --staged --explain-selection` as report-only first. Unknown/protected paths select broad checks.
3. **Make DB absence explicit.** For DB-required checks in local-only modes, output `NEED_REMOTE_EVIDENCE` with exact command/artifact needed; do not infer clean.
4. **Consolidate prompt hooks.** Merge prompt routers into one local classifier while keeping hard branch/worktree/live guards.
5. **Archive process clutter.** Use `stage_reaper_audit.py` dry-run output plus active-plan review to move stale stage/plan files to archive in docs-only commits.

## Do not remove in PASS 2

- Research integrity gates.
- Lookahead protection.
- Holdout discipline.
- Cost/slippage realism checks.
- Canonical source-of-truth checks.
- Live safety gates.
