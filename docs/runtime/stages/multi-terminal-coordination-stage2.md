---
task: "Multi-terminal coordination Stage 2 — Hollow-Commit Prevention Gate. A git commit-msg hook that BLOCKS a commit before it lands when the message asserts a high-confidence literal token the STAGED diff additions contain zero of. Closes the 2026-06-09 n=1 hollow-commit gap (dd63be8b: correct message, stale staged snapshot). Guard B (double-commit block) was KILLED — already exists in worktree_guard.py (commit subcommand block, proven by execution); re-encoding it would violate institutional-rigor §4."
mode: DONE
scope_lock:
  - scripts/tools/commit_message_content_gate.py
  - .githooks/commit-msg
  - tests/test_hooks/test_commit_msg_content_gate.py
  - docs/superpowers/specs/2026-06-09-multi-terminal-coordination-design.md
  - docs/runtime/stages/multi-terminal-coordination-stage2.md
  - .claude/settings.local.json
updated: 2026-06-09T12:00:00+10:00
---

## DONE (2026-06-09) — proof
- 33/33 tests pass (`pytest tests/test_hooks/test_commit_msg_content_gate.py -v`):
  pure-fn + CLI + live `.githooks/commit-msg` under a real `core.hooksPath` repo.
- Calibration (as-built) = 0 false-blocks / 120 commits; `dd63be8b` BLOCKED
  (`${CLAUDE_PROJECT_DIR}` in subject, 0 in staged additions).
- Live injection in THIS repo: hollow → exit 1 BLOCK; honest → 0; `[hollow-ack]`
  → 0.
- `ruff check` clean; `python pipeline/check_drift.py` → 184 passed, 0 violations.
- Adversarial self-review ("step back, reassess") caught + fixed a real
  false-block (PROBE-4: `#`-comment placeholder) via git-cleanup-replicating
  `strip_git_comments`; regression-locked live.
- Dead-code sweep: all 6 module fns on the live call path (find_unmatched_tokens
  ← evaluate; strip_git_comments ← extract_literal_tokens).
- Not a capital/truth-layer path (.githooks/scripts/tools/tests/docs) → formal
  evidence-auditor gate not triggered; rigorous self-review performed instead.
---

## Stage purpose
Prevent the *hollow commit* failure mode proven by the 2026-06-09 n=1 incident:
a commit whose MESSAGE is correct but whose STAGED SNAPSHOT is stale (real fixes
in the working tree, old content staged). This is a single-session commit-time
integrity gap — the worktree mutex does NOT catch it.

PREVENTION, not detection: a `commit-msg` git hook BLOCKS the commit before the
object is finalized when the message asserts a high-confidence literal token
(`${...}` placeholder, backtick-fenced token, repo file path, quoted CLI flag)
that appears in ZERO ADDED lines of `git diff --cached`. Conservative rule —
minimize false-blocks (calibrated: ~1 of last 50 commit subjects carries any
literal-looking token).

## Why Guard B was killed (audit-by-execution, this session)
`worktree_guard.py` is a PreToolUse hook whose `_GIT_MUTATING_SUBCOMMANDS`
includes `"commit"` and which BLOCKS (exit 2) when a live peer holds the worktree
lease. Proven by execution: a `git commit` payload + faked live-peer beat in this
tree returned EXIT 2 "BLOCKED: a live peer Claude session holds THIS worktree's
lease." Stage 1's `${CLAUDE_PROJECT_DIR}` fix (324588bf, origin/main) activated it
for worktrees. Building a second commit-block guard re-encodes an existing hook →
institutional-rigor §4 violation. NOT built. The old `fleet-commit-guard.py`
design is abandoned.

Likewise the old "Guard C advisory" (PostToolUse) was REPLACED by this commit-msg
gate: an advisory only flags AFTER the hollow commit lands (cure, downstream
patch for an upstream gap — the band-aid the operator's "prevention > cure"
directive forbids). `commit-msg` is the institutionally-correct prevention seam.

## Grounding (primary-source verified this session)
- PreToolUse exit-2 / `permissionDecision:"deny"` BOTH block; PostToolUse CANNOT
  block — code.claude.com/docs/en/hooks. So advisory could never prevent.
- `core.hooksPath=.githooks` verified; no `commit-msg` hook exists yet → a new
  `.githooks/commit-msg` fires automatically.
- `commit-msg` receives the final message file as `$1`; `git diff --cached` is
  readable at commit-msg time. Plumbing proven end-to-end in a throwaway repo.
- dd63be8b's message subject literally contains `${CLAUDE_PROJECT_DIR}`; the
  hollow staged settings.json had 0 substitutions → the conservative rule fires.

## Documented known-limits (honest scope — NOT defects)
- L1: catches the LITERAL-TOKEN hollow-commit sub-case only. A message asserting
  only quantitative prose ("32 occurrences") with no quotable token is NOT caught
  (conservative rule; prose-count parsing has no clean low-false-block signal).
- L2: match is ADDED lines only (`^+`, excluding `^+++` headers). A token in a
  removed/context line does not count as present — by design (that is exactly the
  hollow case). Verified against dd63be8b.
- L3: Guard B's blind spots (idle peer >90s; Codex peers writing `.codex/`) live
  in the EXISTING worktree_guard.py, out of this stage's scope.

## Blast Radius
- scripts/tools/commit_message_content_gate.py — NEW pure-Python module; ZERO
  callers/importers (grep-proven). Importable by the test + invoked by the hook.
- .githooks/commit-msg — NEW bash hook; fired by git via core.hooksPath only. No
  Python imports it. Reuses pre-commit's PYTHON/IS_WINDOWS_SHELL/_IN_MULTISTEP
  blocks verbatim (no new resolver).
- tests/test_hooks/test_commit_msg_content_gate.py — NEW; importlib in-process.
- docs/superpowers/specs/... + this stage file — decision-class artifacts.
- .claude/settings.local.json — local-only (gitignored); BRANCH_CONTEXT_OVERRIDE
  env so this session can create the scripts/tools/ file. Not committed.
- Reads at runtime: the commit-msg file ($1) + `git diff --cached` (read-only).
  No gold.db. No trading_app. No capital path. No schema.
- Fail direction: BLOCK is the only active path — fail-OPEN on every error (any
  exception/parse failure/missing arg → exit 0 allow). False-block = annoyance +
  `[hollow-ack]` override. Never wedges legit work (capital-guard fail-direction).
- No drift check enumerates `.githooks/` presence or `scripts/tools/` membership
  for these files (grep-verified) → no registration required.

## Acceptance
- pytest tests/test_hooks/test_commit_msg_content_gate.py -v  → all green
- Live injection: hollow message → blocked; honest → allowed; no-token → allowed;
  `[hollow-ack]` → allowed
- python pipeline/check_drift.py  → no new violations
- Calibration replay over last ~50–100 commits → 0 false-blocks
- Recovery: malformed message / missing arg / not-a-repo / multistep → exit 0

## Lineage
Continues the multi-terminal-coordination plan. Builds on Stage 1 (hook-path
${CLAUDE_PROJECT_DIR} fix, 324588bf on origin/main). Supersedes the abandoned
Guard-B/Guard-C design recorded in earlier revisions of this file.
