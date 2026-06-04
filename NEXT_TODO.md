# Next TODO — saved 2026-06-04 (pre-restart)

Worktree: `canompx3/.claude/worktrees/joshd-radar-review`
Branch: see `git branch --show-current`

---

## JUST DONE (this session) — committing now

**Stage 2: drift check forbidding literal `.git/<marker>` paths in hooks** (worktree footgun, documented n=2 2026-06-04).

5 files, all gates green (drift Check 194 PASS, 176 checks, 10/10 tests, ruff clean):
- `pipeline/check_drift.py` — new blocking check `check_no_literal_git_marker_paths_in_hooks` (Check 194) + CHECKS entry. 197→198 checks.
- `tests/test_pipeline/test_check_drift.py` — `TestNoLiteralGitMarkerPathsInHooks`, 10 tests incl. 4 false-positive guards.
- `scripts/tools/reap_stale_claude_processes.py` — REAL fix: `_session_lock_path()` via `git rev-parse --git-path .claude.pid`.
- `.githooks/post-commit` — REAL fix (plan missed it): rebase/merge/cherry-pick skip-guard used literal `.git/<marker>`, dead in every worktree. Now `git rev-parse --git-path`.
- `.claude/hooks/session-start.py` — hygiene: lock read uses `_git_dir()` to match write idiom.

Key decision: switched fix idiom to git's built-in **`git rev-parse --git-path <marker>`** (operator: "isn't there a simple built-in" — yes, this is it). Resolves correctly in main checkout AND worktree, one call. Drop hand-rolled resolvers where it applies.

Commit running foreground (pre-commit ~4 min). **On restart: verify the commit landed** — `git log -1 --oneline`; if HEAD didn't move or output shows `fatal: cannot lock ref`, the ref-lock race bit (live peers on main) → re-commit. Per memory: read the OUTPUT FILE, not the wrapper "exit 0".

---

## NEXT (deferred Stages from the same plan)

### Stage 3 — landing-pattern rule for hook authors (small, do next)
Write `.claude/rules/` doc documenting the canonical git-dir-resolution idiom:
**use `git rev-parse --git-path <marker>`** (NOT literal `.git/<marker>`, NOT even the `$(git rev-parse --git-dir)/marker` two-step).
- The Check-194 error message already prescribes the fix; Stage 3 makes it a referenceable rule.
- Cite the two memory files + the new check. Cite this as the n=2 mechanical-check escalation.
- Once the rule exists, update `check_no_literal_git_marker_paths_in_hooks`'s `fix_hint` to point at the rule file.

### Stage 4 — fsmonitor investigation (separate concern, deferred)
Unrelated to the marker-path bug class. Investigate whether fsmonitor interacts badly with the worktree setup. Low priority.

---

## Bigger-picture note (operator raised, parked)
Operator asked whether a **plugin/framework** should handle hook hygiene instead of bespoke `.githooks/`.
- Verdict: `git rev-parse --git-path` (built-in) solves the path-resolution class — no dependency needed; now adopted.
- The heavier option `pre-commit.com` framework would manage install/env/path for the ~8 bespoke hooks (lease guards, drift gate, branch-flip, HANDOFF stamper) but is a big migration fighting the current `core.hooksPath=.githooks` setup. **Parked unless operator wants to pursue.**

---

## Resume command
```
cd C:/Users/joshd/canompx3/.claude/worktrees/joshd-radar-review
git log -1 --oneline   # confirm Stage-2 commit landed
# then start Stage 3 (rule doc) — small, ~30 min
```
