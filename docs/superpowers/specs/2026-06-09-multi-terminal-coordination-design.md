# Multi-Terminal Coordination — Design

**Date:** 2026-06-09
**Author:** Claude (brainstorming + institutional audit)
**Status:** Stage 1 DONE (`324588bf`, origin/main). Stage 2 REVISED → hollow-commit prevention gate (`.githooks/commit-msg`) — built + verified (see "Stage 2 — REVISED SCOPE" below); Guard B killed (already exists in `worktree_guard.py`); fleet_state surfacing + Stage 3 still deferred.

---

## Problem (measured, not assumed)

Operator runs many parallel terminals/worktrees and repeatedly hits messy rebases
from overlapping work. Audit of the **actual** failure modes (2026-06-09):

1. **Coordination hooks fire the MAIN checkout's copy.** All 35 hook `command`s in
   `.claude/settings.json` hardcode `C:/Users/joshd/canompx3/.claude/hooks/X.py`
   (the main checkout). A worktree session therefore fires **main's** copy of each
   guard, not the worktree's. While hooks are byte-identical to main the bug is
   latent; the moment a hook is edited in a worktree the copies diverge and the
   guard silently misbehaves. This is the documented "doesn't always fire"
   (`.claude/rules/auto-memory-capture.md` § "Verifying it is wired").
   **PROVEN:** `session-start.py` fired from the worktree path exits 0 and runs the
   worktree's concurrent-session detector; the main copy fired with the same
   payload exits 1 — different behavior by launch path.

2. **`fleet_state.py` already sees all peers** (correctly ID'd the live peer terminal
   + this session in a live run) but nothing surfaces it at session-start or
   edit-time — it is a manual-only tool.

3. **Three task-state surfaces** (61 `docs/runtime/stages/*.md`, `action-queue.yaml`,
   memory batons) — not contradictory, but no single "what is safe to work on now"
   view unifies them.

4. **`HANDOFF.md` is the #1 mechanical overlap** — dirty in 3 of 4 live trees at
   once. A single shared scratch file every session edits.

The official mechanism (Claude Code hooks docs + worktrees docs, fetched
2026-06-09) is **isolation via worktrees**; there is no native peer-registry. The
closest native coordination (Agent Teams: shared task-list + file-locking) is an
orchestrator-spawns-subagents topology — wrong for N independent human terminals.
So the leverage is *making the coordination layer the repo ALREADY built reliable*,
not adding a new system.

---

## Approach — smallest diff first, build-new only if still needed

Three stages, smallest-first, each independently shippable. **Stage 1 is the only
one approved now.**

### Stage 1 — Fix the hook-path-resolution bug (THIS change)

Switch coordination/awareness hook `command`s in `.claude/settings.json` from the
hardcoded main path to the official `${CLAUDE_PROJECT_DIR}` placeholder, which
resolves at runtime to the **current worktree's** root (Claude Code hooks doc).
This makes every existing coordination guard fire the current worktree's copy.

**Carve-outs (from the audit — NOT a blind find-replace):**
- **EXCLUDE** the 2 `memory-capture-*` hooks (`memory-capture-sessionstart.py`,
  `memory-capture-advisory.py`). They are *deliberately* main-pointed per
  `auto-memory-capture.md` (designed to activate only post-merge). Switching them
  would break documented intent. Operator chose "exclude" (2026-06-09).
- **DO NOT touch** `pre-edit-guard.py:29 CANONICAL_REPO` — that hardcoded path is
  the CRG shared-graph root, intentional per `auto-skill-routing.md`
  (`feedback_crg_worktree_repo_root_resolution`). It is internal logic, unrelated
  to the launch path.

**Scope:** `.claude/settings.json` only. ~33 `command` string edits, zero Python
logic change.

**Fail direction (per `capital_guard_fail_direction` memory):** every switched hook
is fail-open → a path-resolution miss = silent pass, never a false block of legit
work. Safe direction. No capital path touched.

### Stage 2 — Surface fleet_state at session-start + edit-time (DEFERRED)

Operator wants BOTH intervention points (chosen 2026-06-09):
- **SessionStart:** one briefing — live peers, files they hold (dirty across live
  trees), and a safe-lane suggestion sourced from non-overlapping stage-file
  `scope_lock`s, falling back to "just show the conflict map" when no clean lane
  is found (never guess silently).
- **PreToolUse/Edit:** warn on overlap; **hard-block ONLY on capital files**
  (`trading_app/live/`, `START_BOT.bat`, session/risk paths) — tiered by risk
  (operator choice). Reuses the `branch-flip-guard.py` / `shared-state-commit-guard.py`
  pattern; **imports `fleet_state` — adds no new liveness oracle**
  (`fleet_state.py` docstring forbids a third oracle).

Deferred deliberately: Stage 1 may make existing guards reliable enough that
Stage 2 can be smaller. Observe first.

### Stage 3 — Unify the 3 task-state surfaces (DEFERRED, optional)

Only if Stage 2 proves a single "safe work" view is needed.

---

## Testing / Verification (Stage 1)

1. **Prove `${CLAUDE_PROJECT_DIR}` resolves to the worktree** — DONE (live proof
   above: worktree copy exit 0, behaves differently from main copy).
2. **Each switched hook still executes** post-edit — pipe synthetic payloads to
   `session-start`, `branch-flip-guard`, `shared-state-commit-guard`.
3. **settings.json remains valid JSON** + `/hooks` registration intact.
4. **Diff confirms the 2 memory-capture hooks are unchanged.**
5. `python pipeline/check_drift.py` unchanged (no new violations).

## Risk

- **Capital:** none — coordination/awareness hooks, not live-trading logic.
- **Reversibility:** settings.json is file-watched; revert = one `git checkout`.
  Tier A reversible.
- **Residual risk:** firing an unmerged, mid-edit worktree hook copy that is broken
  — bounded by fail-open design + the memory-capture carve-out.

## Blast radius (Stage 1)

- `.claude/settings.json` — every coordination hook `command` string. Reads: none.
  Writes: none at runtime. No Python module imports change. No test file asserts
  the hardcoded path (verified). The change only affects WHICH copy of each hook
  the harness launches.

---

## Stage 2 — REVISED SCOPE (2026-06-09, post-Stage-1 audit-by-execution)

**Stage 1 landed (`324588bf`, origin/main).** Re-auditing the 2026-06-09 n=1
incident by EXECUTION reframed Stage 2 entirely. The incident had two failure
modes; one is already solved and one was mis-scoped.

### Guard B (double-commit block) — KILLED, already exists. Do NOT build.

The "two Claudes, one worktree, double-commit" mode is **already prevented** by
`.claude/hooks/worktree_guard.py`: a PreToolUse hook whose
`_GIT_MUTATING_SUBCOMMANDS` includes `"commit"` (`worktree_guard.py:98`) and which
BLOCKS (exit 2) when a live peer holds the worktree lease. **Proven by execution
this session:** a `git commit` payload + a faked live-peer beat in this tree
returned `EXIT 2 "BLOCKED: a live peer Claude session holds THIS worktree's
lease."` It fired the MAIN checkout's copy during the incident (the hardcoded-path
bug); **Stage 1's `${CLAUDE_PROJECT_DIR}` fix activated it for worktrees.**
Building a second commit-block guard would re-encode an existing hook —
institutional-rigor §4 violation. **Not built.** The earlier `fleet-commit-guard.py`
Guard-B/Guard-C design is abandoned.

### Hollow commit (message ≠ staged content) — THE REAL Stage 2.

Commit `dd63be8b` carried a correct MESSAGE but a STALE staged SNAPSHOT (real
fixes in the working tree, old content staged; nothing verified staged↔message).
A single-session commit-time integrity gap — the worktree mutex does not catch it.

Operator directive: *"prevention > cure, no band-aid, official fixes."* This
overrode an earlier PostToolUse advisory (which only flags AFTER the hollow commit
lands — cure, a downstream patch for an upstream gap → the exact band-aid the
directive forbids). PostToolUse cannot block (CC hooks doc); only a git
`commit-msg` hook PREVENTS before the object finalizes.

**Built:**
- `scripts/tools/commit_message_content_gate.py` — pure, testable rule core.
- `.githooks/commit-msg` — fires via `core.hooksPath=.githooks`; reuses
  pre-commit's PYTHON / `IS_WINDOWS_SHELL` / `_IN_MULTISTEP` blocks; fail-OPEN.
- `tests/test_hooks/test_commit_msg_content_gate.py` — 33 tests (pure + CLI + live
  hook in a throwaway repo).

**Rule (calibrated, not assumed):** the plan proposed 4 token classes
(`${...}` placeholder, backtick, repo path, quoted flag). Phase-0 calibration —
replaying the rule over real commit history — MEASURED:
- 4-class: 8 false-blocks / 80 commits (10%); the backtick/path/flag classes fire
  on legitimate explanatory prose (moved paths, removed/behavioral symbols).
- `${...}`-placeholder-ONLY: **0 false-blocks / 120 commits**, AND still catches
  `dd63be8b` (`${CLAUDE_PROJECT_DIR}` in subject, 0 in staged additions → BLOCK).
So the shipped rule is `${...}`-placeholder-only. Match target = ADDED lines only
(`^+` excluding `^+++` headers) — a token in a context/removed line is exactly the
hollow case. (memory:
`feedback_hollow_commit_gate_placeholder_only_calibration_2026_06_09.md`)

**Adversarial-reassessment fix (official, not ad-hoc):** at commit-msg time git
has NOT yet applied cleanup — the raw `COMMIT_EDITMSG` still carries `#`-comment
lines and (under `commit -v`) the diff dump below the scissors line. A placeholder
there is not an authored claim → the gate now strips them first
(`strip_git_comments`, replicating git `cleanup=strip` + scissors), regression-
tested live. (memory:
`feedback_commit_msg_hook_must_strip_git_comments_2026_06_09.md`)

**Honest residual scope (stated, not hidden):**
- Catches the `${...}`-placeholder hollow-commit sub-case only. A message
  asserting only quantitative prose ("32 occurrences") or a backtick/path/flag
  token is NOT caught — those classes have no clean low-false-block signal
  (calibration above). The `dd63be8b` incident IS caught via its placeholder.
- Guard B's documented blind spots (idle peer >90s; Codex peers writing `.codex/`
  not `.claude-heartbeats/`) live in the existing `worktree_guard.py`, out of
  scope here.
- This is prevention for the proven `dd63be8b` mechanism — not a universal
  commit-integrity system.

### Verification (Stage 2)
- `pytest tests/test_hooks/test_commit_msg_content_gate.py -v` → 33 passed.
- Live injection (real repo): hollow → BLOCK; honest → allow; no-token → allow;
  `[hollow-ack]` → allow.
- Calibration replay (as-built) → 0 false-blocks / 120 commits; `dd63be8b` blocked.
- `python pipeline/check_drift.py` → 184 passed, 0 violations.
- `ruff check` → clean.

### fleet_state surfacing (the ORIGINAL Stage 2) — still DEFERRED
The session-start / edit-time fleet briefing described above remains a separate,
deferred change. It is unrelated to the hollow-commit gate and not built here.
