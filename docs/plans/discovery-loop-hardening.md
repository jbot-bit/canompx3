# Discovery-Loop Hardening — Tiered Forcing-Function Design

**Status:** Tier 1 SHIPPED (PR #198). Tier 2 SHIPPED (PR #199). Tier 3 SHIPPED (this PR). Tier 4 PLANNED, staged below.
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

## Tier 2 — Edit-Boundary Hardening (SHIPPED)

`.claude/hooks/pre-edit-discovery-marker.py` (PreToolUse(Edit|Write), fail-open on any error).

**Detects:** edits to `pipeline/` or `trading_app/` when the session transcript contains no discovery-convergence artifact. Walks the last 200 transcript records (≈one full turn block) extracting user-text and assistant `tool_use` Bash commands.

**Pass conditions (any one):**
- `REPRO:` substring in any user message
- `context_resolver.py` substring in any tool_use Bash command
- `TRIVIAL:` substring AND `git diff --cached --shortstat` reports <100 net lines

**Trivial-path exclusions:** `docs/`, `tests/`, `scripts/tools/`, `.claude/`, `.github/`, `.codex/`, `memory/`, plus any `.md`/`.yaml`/`.json`/`.toml`/`.txt` file.

**Manual escape hatch:** `.claude/scratch/discovery-marker.json` with `{"valid_until": "<ISO timestamp>"}`. Skips the gate while valid.

**Fail-open paths (all return exit 0, no block):**
- transcript file missing
- transcript has <5 records (session just started)
- session_id missing from PreToolUse stdin
- JSON parse error on transcript line
- any unhandled exception

**Sibling-hook design (vs extending `pre-edit-guard.py`):** kept separate so the existing CRG advisory and the new blocking gate fail-open independently and have isolated unit tests.

**Acceptance criteria — met:**
- 11/11 synthetic-transcript pytest cases pass (REPRO/context_resolver/TRIVIAL pass paths, no-marker block, large-diff block, trivial path skip, missing transcript fail-open, missing session_id fail-open, active marker pass, expired marker block, short transcript fail-open).
- Drift check passes.
- Hook lands on `harden-discovery-loop-tier2` branch off `origin/main`.

---

## Tier 3 — Context-Budget Hardening (SHIPPED)

`.claude/hooks/read-budget-guard.py` — multi-mode hook (`increment` / `reset-on-edit` / `inject` / `reset-session`) wired across PostToolUse(Read), PostToolUse(Edit|Write), UserPromptSubmit, PostCompact, and SessionStart.

**Counter behavior:**
- `reads` increments on every `Read` tool use (PostToolUse).
- Resets to 0 on any `Edit`/`Write` to `pipeline/` or `trading_app/`.
- Doc/test/script edits do NOT reset (they count as in-discovery).
- Cleared to 0 on SessionStart and PostCompact (state file persists; reset is intentional).

**Warning thresholds:**
- **Soft cap** at 16 reads (no production edit yet) → injects soft warning, 5-minute cooldown to avoid spam.
- **Hard cap** at 26 reads → injects hard checkpoint message every prompt (no cooldown — it must keep firing until the agent acts).

**Soft message:** "Discovery may be drifting — is the read set converging on a falsifiable change? If yes, write the diff. If no, narrow with `context_resolver.py`."

**Hard message:** Stop and pick one of (a) `TRIVIAL:` declaration with diff plan, (b) `context_resolver.py` run, (c) declare exploration complete and abort.

**State file:** `.claude/hooks/state/read-budget.json` (`{"reads", "edits_to_prod", "last_warned_at"}`).

**Fail-open:** every command catches all exceptions, exits 0. Corrupted state file → fresh state.

**Acceptance criteria — met:**
- 14/14 pytest cases pass: increment-from-zero, increment-accumulates, reset-on-prod-edit, reset-on-trading-app-edit, no-reset-on-doc, no-reset-on-test, reset-session-clears, silent-below-soft, soft-fires-at-16, hard-fires-at-26, soft-cooldown-silences, hard-ignores-cooldown, after-reset-silent, unknown-command-silent.
- Drift check passes.
- Hook lands on `harden-discovery-loop-tier3` branch off `origin/main`.

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
| **S1 (PR #198)** | Tier 1 hook + design doc | Hook fires/silences correctly on 6 test cases; drift clean | Low — additive UserPromptSubmit, fail-open ✅ |
| **S2 (PR #199)** | Tier 2 edit-boundary marker | 11 transcript fixtures pass; trivial paths excluded; fail-open on transcript errors | Medium — transcript format coupling ✅ |
| **S3 (this PR)** | Tier 3 read-budget counter | 14 pytest cases pass; reset on prod edit + session-start + post-compact; soft @16 / hard @26 | Low — additive PostToolUse ✅ |
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
