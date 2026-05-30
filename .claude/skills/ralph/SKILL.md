---
name: ralph
description: >
  Run one Ralph Loop audit iteration. Finds Seven Sins violations, canonical
  integrity issues, and silent failures. Clusters related findings before fixing
  for maximum impact per iteration. Fixes highest-priority cluster. Returns structured report.
when_to_use: ["ralph", "run ralph", "ralph loop", "ralph audit", "autonomous audit"]
disable-model-invocation: true
---
Run one Ralph Loop iteration as a dedicated subagent (token-optimized).

Scope: $ARGUMENTS (e.g. "live_config.py", "outcome_builder", "all deferred", "session_orchestrator").
If empty, the agent reads Next Targets from docs/ralph-loop/ralph-loop-audit.md and auto-selects
the highest-priority target using P0 (open CRIT/HIGH) → P1 (unscanned critical/high) → P2 (stale)
→ P3/P4.

---

## Dispatch

Launch the `ralph-loop` agent with this prompt:

```
Run one Ralph Loop iteration.
Scope: [SCOPE or "auto — use P0→P4 priority queue from audit file"]
Today's date: 2026-05-30
Priority override: ALWAYS fix CRITICAL/HIGH findings before LOW. If a CRIT/HIGH exists
anywhere in the open deferred list or HANDOFF.md, that is this iteration's scope regardless
of what the auto-targeting queue says.
```

The agent runs autonomously: audit → cluster → select → fix → verify → commit → report.
It returns a structured report when done.

**Do NOT duplicate the agent's work.** Just dispatch and report results.

## After Agent Returns

1. Print the agent's final report verbatim (the `=== RALPH LOOP ITER ... ===` block)
2. If the agent reports REJECT or NEEDS_REVIEW → surface to user with the finding details
3. If DIMINISHING_RETURNS → tell user: "Ralph reports no high-impact targets remain. Run again after new code changes accumulate, or provide a specific scope."
4. If ACCEPT → done, no further action needed
