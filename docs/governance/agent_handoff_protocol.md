# Agent Handoff Protocol

**Authority:** binding on every agent (Claude, Codex, future) handing off or inheriting work. Lives under `docs/governance/` because it governs how agents interact, not what they build.

**Origin:** 2026-04-20 `codex/live-book-reaudit` rehabilitation. Codex delivered real, honest code but the handoff failed on four dimensions (empty commit body, stale base, undisclosed scope, unverified test claims). Inheriting agents trusted the HANDOFF text and almost merged an unusable branch. This doc codifies the lessons.

---

## Scope

Applies when:
- an agent commits work and stops without completing the task
- another agent (or same agent in a new session) resumes
- a cross-terminal handoff occurs via `HANDOFF.md`, `docs/plans/`, or any stage file

Does NOT apply to:
- a single uninterrupted agent session on a single task
- human-driven commits with human-written messages

---

## The contract — what the handing-off agent MUST do

### 1. Commits carry their own justification

- Every commit that touches more than 5 files OR 100 changed lines MUST have a commit-message body ≥ 100 characters.
- The commit-msg hook (`.githooks/commit-msg`) enforces this. Do not bypass with `COMMIT_MSG_SKIP=1` except in genuine emergency (and document why).
- The body must cover: WHAT changed (file-level), WHY (linked to an issue / hypothesis / stage), KNOWN GAPS (what isn't done), and VERIFICATION performed (with raw output citations, not claims).

### 2. Branch base is current

- Before a PR or push that targets `main`, the branch MUST be rebased onto the current `origin/main` — never branch from a stale local `main`.
- The pre-push hook (`scripts/tools/verify_branch_scope.py`) enforces staleness thresholds:
  - 3 days old → warning
  - 7 days old OR origin/main ≥ 25 commits ahead of merge-base → block
- Rebase strategy when conflicts arise with main's recent work: **always read main's version first**. Reverting main's work silently is the failure mode we are preventing.

### 3. Scope is bounded by a scope document

- Any branch with 60+ files changed OR 3000+ lines changed MUST have a matching doc in `docs/runtime/stages/`, `docs/plans/`, or `docs/audit/hypotheses/`.
- The doc must exist BEFORE the commits, not backfilled after. Pre-registration is the only acceptable proof of intent.
- The pre-push hook enforces this.

### 4. Test claims must be reproducible

- When HANDOFF or a commit body says "pytest X → 170 passed", an inheriting agent MUST be able to reproduce that exact number in the current working tree.
- The command shown in the claim MUST be verbatim — no shell-variable expansion, no absolute paths that only exist on the author's machine, no deleted test files.
- If the command uses an absolute python path (e.g., `/mnt/c/.../bin/python`), rewrite it in relative form when committing to HANDOFF so the next agent can run it.

### 5. Line endings are LF

- Every text file goes to the index with LF line endings. Enforced by `.gitattributes`.
- The pre-commit CRLF guard in `.githooks/pre-commit` catches any bypass.
- This is non-negotiable. Cross-platform teams have too many agents to debate EOL.

### 6. HANDOFF entries are append-only

- New HANDOFF session entries are added at the top with a dated header.
- Old entries are preserved unless they are provably wrong — in which case the correction is recorded in a new entry that explicitly supersedes the old one with a citation.
- Never silently rewrite HANDOFF history. The audit trail is load-bearing.

---

## The contract — what the inheriting agent MUST do

### A. Audit before trust

- Do not take a prior agent's HANDOFF text as ground truth. Treat it as a claim, not evidence.
- Run the Seven Verification Questions on inherited work:

  1. Does the branch rebase cleanly on current `origin/main`, or does it revert any work?
  2. Do all claimed test numbers reproduce (`pytest` exactly, not "all green")?
  3. Does `pipeline/check_drift.py` pass? If new violations exist, are they caused by this branch?
  4. Is every commit message body ≥ 100 chars when the diff is substantial?
  5. Is every new config/feature value traceable to a source (code, hypothesis YAML, canonical table)?
  6. For any mentioned data claim (row counts, p-values, survival rates) — can it be re-derived from the canonical DB right now?
  7. Does the scope match the branch name and scope document, or has it crept?

### B. Call out failures explicitly

- If the audit catches process failures (empty commit body, stale base, inflated scope, unverifiable claims), document them BEFORE proposing fixes. The user deserves a plain-language map of what went wrong.
- The audit output must classify each finding as VERIFIED / DOWNGRADED / BAD — no hedging.

### C. Fix the branch, not the symptom

- Acceptable fixes: rebase, reword commit, split commit, drop stale reverts, add tests, add docs.
- Unacceptable fixes: "we'll handle it in the PR description", "squash everything and call it clean", force-push to reuse the original branch name without the user's explicit approval.
- The fix must be reproducible by writing down the exact sequence of git commands.

### D. Run the real work, not just the cleanup

- If the inherited work's goal was to produce an audit result / deployment decision / research finding, produce it. Branch rehabilitation is necessary but not sufficient.
- The handoff failed if the inheriting agent only cleaned up process debt and did not advance the actual research question.

---

## Bypass / emergency

- `COMMIT_MSG_SKIP=1 git commit ...` — bypasses commit-msg adequacy
- `VERIFY_BRANCH_SCOPE_SKIP=1 git push ...` — bypasses pre-push staleness/scope
- `M25_SKIP=1 git commit ...` — bypasses full pre-commit (ruff/drift/tests)

**Every bypass leaves a shell history trail.** If you see one in a transcript without explanation, treat it as a handoff-protocol violation and flag it.

---

## Attached checks

| Surface | Enforces | Location |
|---------|----------|----------|
| `.gitattributes` | LF line endings (text files), CRLF (.bat/.cmd/.ps1) | Repo root |
| `.githooks/pre-commit` § 0b | No CR bytes in staged text index blobs | Pre-commit |
| `.githooks/commit-msg` | Body ≥ 100 chars if > 5 files or > 100 lines | commit-msg |
| `.githooks/pre-push` | Calls `verify_branch_scope.py` | Pre-push |
| `scripts/tools/verify_branch_scope.py` | merge-base age, origin/main drift, scope doc existence | Pre-push |

**Activation requirement (one-time per clone):** `git config core.hooksPath .githooks`. The 2026-04-20 incident rehabilitation discovered this setting was never applied, so pre-commit checks had been silently dormant. Verify on every new clone / worktree.

---

## Non-goals

- This doc does NOT replace `CLAUDE.md`, `TRADING_RULES.md`, `RESEARCH_RULES.md`, or any other canonical doc. It describes HOW agents hand off, not WHAT they build.
- This doc does NOT address single-agent session discipline — see `.claude/rules/institutional-rigor.md` for that.
- This doc does NOT govern human-to-human handoffs. Humans have unwritten trust signals agents do not.
