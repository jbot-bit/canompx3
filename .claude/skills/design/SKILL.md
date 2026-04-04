---
name: design
description: >
  Design-before-build. Modes: plan (4-turn structured, wait for approval),
  explore (ask questions first), auto (plan then proceed). Default: plan.
  Use when: "design", "plan", "4t", "brainstorm", "how should we build",
  "think through", "approach this", "feature planning".
effort: high
---

# Design

Design a feature or change: $ARGUMENTS

**Modes:** `plan` (default) | `explore` | `auto`

**HARD GATE:** Do NOT write code until user approves (except `auto` mode).
**Plan mode language:** In plan/explore mode, describe ALL changes in pure natural language. NEVER use ANY of these tokens in your output: `def `, `class `, `import `, `return `, `->`. This includes compound words — write "immutable data container" not "frozen dataclass", "filter type" not "filter class". This rule exists because plan output is validated for absence of code.

---

## Mode: explore

Start with clarifying questions before the 4-turn flow.
- One question per message, prefer multiple choice
- Focus: purpose, constraints, success criteria, blast radius
- Trading: which instruments? sessions? entry models?
- Pipeline: which tables? rebuild impact?
- Research: hypothesis? kill criterion?

After questions answered → run the 4-turn flow below.

---

## The 4-Turn Flow (all modes)

### Turn 1: ORIENT

1. Parse $ARGUMENTS for the topic
2. **Check `docs/STRATEGY_BLUEPRINT.md`** — route to correct section, NO-GO registry (SS5), assumptions (SS10)
3. Read ALL affected files — trace imports, map blast radius
4. Check `docs/specs/` for existing spec (if one exists, follow it)
5. Check authority docs — **TRADING_RULES.md** for any trading logic, filters, sessions, or entry models; RESEARCH_RULES.md for statistical methodology; CLAUDE.md for architecture
6. Check canonical sources that might be touched (ACTIVE_ORB_INSTRUMENTS, SESSION_CATALOG, config, COST_SPECS, GOLD_DB_PATH)
7. Articulate PURPOSE: why this matters, what breaks without it

### Turn 2: DESIGN (multi-take deliberation)

**Minimum 3 takes for non-trivial designs:**
1. What went wrong before in this domain? (Check `hard_lessons.md`, blueprint SS10-SS11)
2. Design bottom-up from failure prevention
3. Challenge: too complex? too simple? right ordering?
4. Pressure-test against past failures

**Then propose:**
- Data model, interfaces, data flow, layer placement
- One-way dependency check (pipeline -> trading_app only)
- 2-3 approaches with trade-offs, state recommendation

### Turn 3: DETAIL

Ordered implementation steps specific enough to execute blindly:
1. Every file to create/modify/delete with exact paths
2. What changes in each file, in what order
3. Test strategy + migration/rebuild needs
4. Drift check impact

### Turn 4: VALIDATE

1. Failure modes and risks
2. What tests prove correctness (behavior, not just "it runs")
3. Rollback plan
4. Guardian prompts needed? (ENTRY_MODEL_GUARDIAN / PIPELINE_DATA_GUARDIAN)

---

## After All 4 Turns

Save design to `docs/plans/YYYY-MM-DD-<topic>-design.md`.

### Mode: plan (default)

Present design and WAIT for approval ("go", "approved", "do it", "looks good").
- On approval: write Stage 1 to `docs/runtime/STAGE_STATE.md` (IMPLEMENTATION mode, scope_lock from Turn 3, acceptance from Turn 4)
- On "iterate"/"change": revise, stay in design mode

### Mode: auto

After Turn 4, auto-proceed without pause:
1. Write design doc + STAGE_STATE, commit both
2. Begin implementation

**Safety override:** If design reveals schema change, entry model change, or blast radius > 5 files → STOP and ask (safety overrides speed).

---

## Rules

- ONE topic at a time. Never batch.
- If spec exists in `docs/specs/`, follow it — do not redesign.
- NEVER skip ORIENT. Reading code is not optional.
- NEVER propose changes to files you haven't read.
- Apply YAGNI ruthlessly.
- Check prior research in memory before designing from scratch.
