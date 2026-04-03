---
name: next
description: Auto-determine and start the next concrete coding task from stage state, handoff, and action queue
allowed-tools: Bash, Read, Grep, Glob, Edit, Write, Agent
---
Determine the next concrete task and start executing it: $ARGUMENTS

Use when: "next", "what's next", "continue", "keep going", "auto", "pick up", session start with no explicit task
Triggers: "next", "auto-implement", "what should I do", "keep going"

## STEP 0: GATHER STATE (parallel)

Run ALL of these in parallel — do not read sequentially:

1. Read `docs/runtime/STAGE_STATE.md` (if exists)
2. Read HANDOFF.md — scan for the FIRST `## Update` block. Within it, look for any of: `### Next Session`, `### Next Sensible Step`, `### Next`, or bullet lines starting with `- Next:`. These vary across tools (Claude, Codex).
3. `git log --oneline -5`
4. `git status --short`
5. `python -c "import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace'); from scripts.tools.claude_superpower_brief import build_brief; from pathlib import Path; print(build_brief(root=Path('.').resolve(), mode='interactive'))"`

## STEP 1: DECISION TREE

Work through this tree top-to-bottom. Take the FIRST match.

### Case A: Active IMPLEMENTATION stage — check if DONE first

STAGE_STATE.md has `mode: IMPLEMENTATION` and a `task:` field.

1. **Acceptance check (do this BEFORE resuming):** Read the `acceptance:` field. For each criterion, run the command or check the condition. If ALL acceptance criteria are already met:
   → Announce: "**Stage complete:** [task]. All acceptance criteria pass."
   → Show evidence (command outputs).
   → Delete STAGE_STATE.md.
   → Fall through to Case E to find the NEXT task.

2. **Stale check:** `git log --oneline --since="[updated]" -- [scope_lock files]`
   - Commits since update? → Flag: "Stage may be stale. Re-reading scope files before continuing."
   - No commits? → FRESH.

3. **Scope files exist?** Verify each `scope_lock` path exists (glob check).

4. **Action:** Announce: "**Resuming:** [task]. Remaining: [unmet criteria]. Scope: [files]."
   → Run preflight on scope files (read them, check they compile)
   → Start implementing the unmet criteria. Follow scope_lock — do not expand.

### Case B: Active DESIGN stage exists

STAGE_STATE.md has `mode: DESIGN`.

**Action:** Announce: "**Active design:** [task]. Continuing design iteration."
→ Re-read the design state and continue where it left off.

### Case C: Active TRIVIAL stage exists

STAGE_STATE.md has `mode: TRIVIAL`.

**Action:** Just do it. Read the scope file, make the fix, verify, close the stage.

### Case D: STAGE_STATE.md exists but is STALE or ABANDONED

Stale detection — two signals, BOTH required to declare abandoned:
- `updated` is >8 hours old
- AND there are commits touching OTHER files since the update (= user moved on to different work)

If only old but no other activity → user just paused. Treat as Case A/B/C (resume).
If old AND other activity → truly abandoned.

**Action:** Announce: "Stale stage detected: [task] (last updated [timestamp], [N] unrelated commits since). Reclassifying."
→ Delete STAGE_STATE.md
→ Fall through to Case E.

### Case E: No active stage — derive next task

No STAGE_STATE.md, or it was just cleaned up.

**Priority order for finding the next task:**

1. **Brief broken/decaying signals** — if the pulse shows BROKEN items, those take absolute priority. Fix infrastructure before features.

2. **HANDOFF.md next steps** — from the LATEST `## Update` section ONLY. Look for `### Next Sensible Step`, `### Next Session`, `### Next`, or inline `- Next:` bullets. Pick the FIRST concrete, actionable item. Concrete = has a clear deliverable (file to create, bug to fix, feature to build). Vague = "consider", "think about", "investigate" → skip unless nothing else.

3. **Memory action queue** — read `MEMORY.md`, find the `## ACTION QUEUE` section. Pick P1 if it's concrete and not already done (verify with git log / code check).

4. **Brief recommendation** — the `Next:` line from pulse. If it says "Prep: [session] in Xh" → that's a pre-session check, not a coding task. Note as sidebar: "FYI: [session] in [X]h" but don't make it the task.

**Once you have a task:**

- Announce: "**Next task:** [description]. Source: [handoff/memory/pulse]."
- Classify: Is it TRIVIAL or needs full staging?
  - TRIVIAL (<=2 files, mechanical) → write minimal STAGE_STATE, implement immediately
  - Non-trivial → write STAGE_STATE with scope_lock + acceptance, run preflight, start implementing
- **Do NOT ask "should I proceed?" — the user invoked /next, which means GO.**

### Case F: Nothing actionable found

**Action:** Run `/orient` to give the user a full status. End with: "No concrete next task found. What would you like to work on?"

## STEP 2: EXECUTE

Once the task is identified and staged:

1. Read ALL scope files before writing any code (2-pass method)
2. Write code
3. Verify: `python pipeline/check_drift.py` + targeted tests
4. Report result concisely (3 lines max)

## STEP 3: SIDEBAR — session awareness

After identifying and starting the task, check the pulse `Upcoming:` line. If a trading session is <2 hours away, add a one-line note:
"**Sidebar:** [SESSION] in [X]h — run `/trade-book` if you need to prep."

Do not let this override the implementation task. It's information, not a redirect.

## HARD RULES

- /next means GO. Do not ask permission. Do not present menus. Pick ONE task and start it.
- If multiple tasks tie in priority, pick the one with smallest blast radius.
- NEVER start work without writing STAGE_STATE.md first (even TRIVIAL).
- If the derived task touches NEVER_TRIVIAL files, use full staging — no shortcuts.
- If HANDOFF.md contradicts current code state, trust code. Flag the contradiction.
- Do not pick tasks that require user input (research decisions, architecture choices). Those need /design, not /next.
- Maximum 1 task. If there's a queue, do the first one. The user will say /next again for the second.
- Acceptance check in Case A is MANDATORY. Do not resume work that's already done — close it and move on.
- Memory action queue items may be stale. Before picking one, verify it hasn't already been completed (quick git log or code check).
