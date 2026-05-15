---
description: Subagent invocation discipline — token budget rules for the main agent. Auto-loads when the main agent is about to use the Agent tool.
globs:
  - "**/*"
alwaysApply: false
---

# Subagent Budget — Token Discipline for the Main Agent

**Load-policy:** referenced from CLAUDE.md § Subagent Budget. Main agent should consult before invoking the Agent tool.

**Authority:** prevents the dominant token sink — fanned-out subagents that each reload full project context cold and return verbose summaries.

---

## Why subagents are expensive

Every Agent tool spawn:
1. Loads `CLAUDE.md` + global instructions cold (~10K tokens).
2. Loads auto-load rules matched by the agent's file globs (~5-15K tokens).
3. Loads the agent's own definition file (1-20KB).
4. Runs its own tool calls, each with their own results.
5. Returns a free-form summary that lands in the main agent's context.

**Net effect:** a "quick" subagent spawn costs 30-60K tokens of *additional* main-context growth and ~50K-200K tokens of internal work. Two parallel spawns = ~100K added to main context easily.

---

## Hard rules

1. **Max one subagent per turn.** Exception: two genuinely independent tasks where serial execution would force the main agent to re-read large file sets. Never three.

2. **Threshold for spawning.** Only spawn when the inline alternative would cost ≥5 file reads, ≥3 grep rounds, or genuinely needs a *separate* context (independent reviewer to counter bias). Below that: do it inline.

3. **Narrow scope in the prompt.** Every spawn prompt must include:
   - The specific question (one sentence).
   - The expected return format (e.g., "bullet list, ≤200 words").
   - The hard word cap (the agent's own `## Return Budget` is the floor, you can ask for less).
   - Any pre-known paths / line numbers the agent should NOT re-discover.

4. **Never relay another subagent's verbose output into a new spawn.** Summarize the prior output to ≤100 words first, then pass that summary.

5. **Enforce the Return Budget.** If a subagent returns a wall of text exceeding its agent-file budget, do not paraphrase it for the user — note "subagent over-returned, summarizing to N words" and compress aggressively.

6. **Past 100K tokens — stop spawning.** Cache pressure makes every subsequent turn more expensive. Finish the current task inline, then `/clear` with a handoff note.

---

## Decision flow

Before calling the Agent tool, answer:

- **Is this task <5 file reads / <3 greps?** → Inline. No subagent.
- **Is the main agent already past 100K tokens?** → Inline. No subagent.
- **Am I about to spawn 2+ in parallel because they "feel independent"?** → Pick the one with higher information value; do the other inline.
- **Do I have a specific question + expected format + word cap?** → Spawn. Otherwise, refine the prompt first.

---

## Anti-patterns observed

- Spawning `blast-radius` + `test-coverage-scout` + `evidence-auditor` for a 30-line refactor. Each is ~30K tokens of fresh context. The refactor itself is 5 inline greps.
- Asking `db-analyst` "tell me about the current trade book" without specifying columns or count — agent returns 40+ rows of detail.
- Calling `planner` for changes that fit one stage.
- Spawning `verify-complete` after every micro-edit instead of batching to natural verification points (end of stage).

---

## Related

- `.claude/rules/workflow-preferences.md` § Subagents And Teams (older, lighter version of this rule)
- `.claude/agents/*.md` § Return Budget (every agent definition now carries a 250-500 word cap)
- `CLAUDE.md` § Context-Window Hygiene (cross-cutting context-size rule)
