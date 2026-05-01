# Discovery-Loop Hardening — Tiered Forcing-Function Design

**Status:** Tier 1 SHIPPED (this PR). Tiers 2–4 PLANNED, staged below.
**Owner:** Josh + Claude
**Created:** 2026-05-01
**Trigger:** 2026-05-01 session — pasted Codex status ("I'm reading the remaining adapter files before I patch them") raised the question: what protects us when an agent — Claude, Codex, or another — slips into infinite-discovery mode?

---

## Problem

Agents (and humans paired with them) drift into a failure mode:

1. Vague hardening / future-proofing language with no falsifiable target.
2. Pasted third-party agent narration treated as a task.
3. "Reading more files before patching" loops that consume context without producing a diff.

These never trip existing guards (drift, stage-gate, branch-flip) because the failure mode is *inaction wrapped in motion* — the agent is reading, not editing, so PreToolUse(Edit) hooks never fire. Context exhausts; the original symptom drifts out of the window; band-aid fixes follow.

## Strategy

Asymmetric cost. Make the loose path expensive (must produce one of three artifacts) and the disciplined path cheap (one command). Layer enforcement at the prompt-entry boundary, the edit boundary, and the read-budget boundary.

The three artifacts that satisfy any tier:
- **REPRO:** failing command + actual vs expected output
- **`context_resolver.py` output** narrowing the blast radius
- **TRIVIAL:** declaration with file list and diff <100 lines

---

## Tier 1 — Prompt-Entry Hardening (SHIPPED)

`.claude/hooks/discovery-loop-guard.py` (UserPromptSubmit, fail-open, 15-min cooldown).

**Detects:** narration phrases (`reading the remaining`, `let me check more`, `isolating weak spots`, `before I patch`, `i'm reading`) and open-ended verbs (`harden`, `future-proof`, `tighten up`, `shore up`, `audit everything`).

**Escape hatch:** any of `REPRO:`, `TRIVIAL:`, `EXPLORE:`, `context_resolver.py` in the prompt silences the guard.

**Output:** compact `additionalContext` block listing the three artifact options plus the explicit instruction to treat pasted agent-status text as narration, not a task.

**Acceptance criteria — met:**
- Fires on the 2026-05-01 trigger paste verbatim.
- Silent on normal prompts ("merge the PR when CI is green").
- Silent on cooldown repeat.
- Silent on escape-hatch prompts.
- Fail-open on malformed JSON.
- Drift check passes.

---

## Tier 2 — Edit-Boundary Hardening (PLANNED)

**Goal:** before any `Edit`/`Write` to `pipeline/` or `trading_app/`, require a session marker proving discovery converged.

**Hook:** extend `pre-edit-guard.py` (or a sibling `pre-edit-discovery-marker.py`).

**Logic:**
- Read the last N user messages and tool results from the session transcript (location: `~/.claude/projects/<slug>/transcripts/`).
- Pass conditions (any one):
  - `REPRO:` line in user prompt within last 10 turns.
  - `context_resolver.py` invocation in last 20 turns with non-empty output.
  - `TRIVIAL:` declaration in last 5 turns AND staged change diff <100 net lines.
- Fail-closed message: lists the three options, names the file the agent was about to edit, and includes the exit code escape (`echo "TRIVIAL: <reason>"` to a recognized marker file).

**Files to touch:**
- `.claude/hooks/pre-edit-guard.py` (extend) OR new `.claude/hooks/pre-edit-discovery-marker.py`
- `.claude/settings.json` (one PreToolUse(Edit|Write) entry — already wired to `pre-edit-guard.py`, so prefer extension)
- `.claude/rules/institutional-rigor.md` (document the marker requirement)

**Blast radius:**
- All `Edit`/`Write` to production paths gated. False positives possible on legitimate small fixes — escape hatch via `TRIVIAL:` declaration or a marker file `.claude/scratch/discovery-marker.json`.
- Trivial paths (docs, tests, scripts/tools) excluded from the gate.

**Acceptance criteria:**
- Fires on a session that hasn't produced any of the three artifacts.
- Silent when `REPRO:` was stated.
- Silent when `context_resolver.py` ran successfully.
- Silent for trivial paths.
- Fail-open on transcript read errors.
- Test: synthetic transcript fixtures for each branch.

**Risk:** transcript path / format may change across Claude Code versions. Mitigation: feature-detect, fall back to the marker-file-only path.

---

## Tier 3 — Context-Budget Hardening (PLANNED)

**Goal:** observable read-budget; warn at soft cap, force a checkpoint at hard cap.

**Hook:** new `.claude/hooks/read-budget-guard.py` (PostToolUse(Read), increments a session counter; UserPromptSubmit reads it).

**Logic:**
- Increment counter on every `Read` tool use.
- Decrement / reset on `Edit`/`Write` to a non-test, non-doc file.
- Soft cap (warn): >15 reads with zero edits. Inject "Discovery has read N files without an edit — is this converging?"
- Hard cap (force checkpoint): >25 reads. Inject "Hard cap reached. Either commit a `TRIVIAL:` declaration with the diff plan, or run `context_resolver.py`, or declare exploration complete and abort."
- Resets on session start.

**Acceptance criteria:**
- Counter increments on `Read`, decrements on `Edit` to `pipeline/`/`trading_app/`.
- Soft warning fires at 16 reads / 0 edits.
- Hard message fires at 26.
- No double-counting across PostCompact / SessionStart.

**Risk:** counter drift across compaction. Mitigation: PostCompact hook resets counter to 0 (loss of state is safe — it just delays the next warning).

---

## Tier 4 — Culture / Rule Hardening (PLANNED, no code)

**`workflow-preferences.md`** — add section:

> ### Pasted Agent Narration ≠ Task
>
> When the user pastes another agent's status (`I'm reading X`, `let me check Y`, `isolating weak spots`), the response is to ask for the failing repro, not to expand the read set. Narration describes another agent's process; it is not an instruction to mirror that process.

**`institutional-rigor.md`** — add section:

> ### Discovery-Loop Tells
>
> "Reading remaining files before patching" is a discovery-loop tell. Stop. Write the failing repro. Then read only files in its blast radius. The 2-Pass Implementation Method has a stopping rule for Discovery — honor it.

**`MEMORY.md` / `feedback_*.md`** — capture the 2026-05-01 trigger event with regex set and lesson, so future agents inherit the pattern recognition.

---

## Stage-Gate Plan

| Stage | Scope | Acceptance | Risk |
|-------|-------|------------|------|
| **S1 (this PR)** | Tier 1 hook + design doc | Hook fires/silences correctly on 6 test cases; drift clean | Low — additive UserPromptSubmit, fail-open |
| **S2** | Tier 2 edit-boundary marker | All branches covered by transcript fixtures; trivial paths excluded; fail-open on transcript errors | Medium — transcript format coupling |
| **S3** | Tier 3 read-budget counter | Counter behavior verified across compact + session-start; warnings fire at thresholds | Low — additive PostToolUse |
| **S4** | Tier 4 rule docs + memory entry | Rule files updated; memory entry written | None — docs only |

Each subsequent stage opens its own branch off `origin/main`, lands its own PR, and is independently revertable.

---

## Why This Design

- **Smallest possible Tier 1.** One file, one settings.json edit, one design doc. Ships today, immediately catches the 2026-05-01 pattern.
- **Persistent for the rest.** Tiers 2-4 are written down with acceptance criteria, blast radius, and stage-gate slots. Nothing falls through.
- **Composable.** Each tier is independently useful. Skipping Tier 3 doesn't break Tiers 1 or 2.
- **Aligned with existing canon.** `workflow-preferences.md`, `institutional-rigor.md`, `2-Pass Implementation Method`, `Design Proposal Gate` — Tier 4 just makes the existing rules cite this failure mode by name.
- **Reversible.** Every layer is fail-open. A bad regex or stale transcript path degrades to current behavior, not blocked work.

## Out of Scope (future, not committed)

- Detection of *third-party* agent failure modes beyond Claude/Codex (Gemini, Grok, etc.). Same regex set probably covers them — verify when seen.
- ML-based intent classifier. Regex is good enough; complexity not justified.
- Automatic conversion of narration paste into a `repro_request.md` skeleton. Could be a future ergonomic; orthogonal to the safety question.
