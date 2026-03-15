---
name: 4tp
description: Plan, design, and proceed to implementation
---

Plan, design, and proceed to implementation for: $ARGUMENTS

Use when: user says "4tp", "plan and build", "design and implement", "full pipeline", "just do it and plan"

## 4TP = 4T + Proceed

This is the full-pipeline version of /4t. Run all 4 turns (orient, design, detail, validate), then AUTOMATICALLY proceed to planning and implementation. Zero unnecessary stops.

### Philosophy

Zero stops means zero unnecessary pauses for human approval — NOT zero verification. Efficiency = correct autonomy. Speed without understanding = reckless, not efficient. Wrong autonomy costs more than no autonomy — measure twice, cut once.

### Phase 1: Run /4t

Execute the full 4T flow for $ARGUMENTS. All 4 turns, no shortcuts:
1. ORIENT -- read all affected files, map blast radius, articulate purpose
2. DESIGN -- data model, interfaces, data flow, recommend approach
3. DETAIL -- ordered implementation steps, test strategy, migration needs
4. VALIDATE -- risks, failure modes, rollback plan, guardian prompts

### Phase 2: Auto-Proceed (NO PAUSE)

After Turn 4 completes:

1. **Write design doc** to `docs/plans/YYYY-MM-DD-<topic>-design.md`
2. **Commit it**: `git add docs/plans/<file> && git commit -m "docs: 4TP design -- <topic>"`
3. **Invoke writing-plans skill** to create the implementation plan from the design

Do NOT pause for approval between Phase 1 and Phase 2. The whole point of 4TP is zero stops.

### Safety Override

If the design reveals the task is:
- **Trivial** (< 3 steps): skip writing-plans, just do it
- **Dangerous** (schema change, entry model change, pipeline data flow change): STOP and ask -- override "no stops" for safety. Read the relevant guardian prompt first (ENTRY_MODEL_GUARDIAN or PIPELINE_DATA_GUARDIAN).

### Blast Radius Escalation

If during any Turn you discover blast radius > 5 files, or the change touches schema/entry models/pipeline data flow:
- STOP even though 4TP says "no stops" — safety overrides speed
- Read the relevant guardian prompt (ENTRY_MODEL_GUARDIAN or PIPELINE_DATA_GUARDIAN)
- Map ALL callers and callees before proceeding

### Rules

- Same rules as /4t apply (ORIENT is mandatory, read before proposing, YAGNI)
- The entire value of 4TP is NO UNNECESSARY STOPS between design and planning — but verification is never a "stop"
- If a spec exists in `docs/specs/`, follow it -- do not redesign
