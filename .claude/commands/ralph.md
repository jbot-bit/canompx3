Run one Ralph Loop iteration as a dedicated subagent (token-optimized).

Use when: "ralph", "run ralph", "ralph loop", "ralph audit", "autonomous audit"

Scope: $ARGUMENTS (e.g. "live_config.py", "outcome_builder", "all deferred"). If empty, the agent reads Next Targets from docs/ralph-loop/ralph-loop-audit.md.

---

## Dispatch

Launch the `ralph-loop` agent with this prompt:

```
Run one Ralph Loop iteration.
Scope: [SCOPE or "use Next Targets from audit file"]
Today's date: YYYY-MM-DD
```

The agent runs autonomously: audit → find → fix → verify → commit → report.
It returns a structured report when done.

**Do NOT duplicate the agent's work.** Just dispatch and report results.

## After Agent Returns

1. Print the agent's final report verbatim (the `=== RALPH LOOP ITER ... ===` block)
2. If the agent reports REJECT or escalation needed → surface to user
3. If ACCEPT → done, no further action needed
