---
name: brainstorm
description: >
  Design-before-build for features and changes. Use before any non-trivial work:
  new features, architecture changes, research programs, pipeline modifications.
  Explores intent, proposes approaches, produces a design doc. No code until approved.
---

# Brainstorm

Turn an idea into a validated design before writing any code.

**HARD GATE: Do NOT write code, edit files, or run implementation commands until the user says "go", "build it", "implement", or "do it".**

This aligns with `workflow-preferences.md` — design mode words vs implement words.

## Process

### 1. Understand Context
- Read relevant files (pipeline code, trading rules, existing specs)
- Check `docs/specs/` for existing specs on this topic
- Check `docs/plans/` for prior design work
- Check memory: "what did we find about X" (Pinecone if deep history needed)

### 2. Ask Clarifying Questions
- One question per message — don't overwhelm
- Prefer multiple choice when possible
- Focus on: purpose, constraints, success criteria, blast radius
- For trading features: which instruments? which sessions? which entry models?
- For pipeline changes: which tables affected? what's the rebuild impact?
- For research: what's the hypothesis? what's the kill criterion?

### 3. Multi-Take Deliberation (MANDATORY for non-trivial designs)

Do NOT propose a structure on the first pass. Instead, do multiple takes that challenge your own assumptions:

1. **Catalog actual failures** in this domain — what went wrong before? (Check `hard_lessons.md`, `STRATEGY_BLUEPRINT.md §10-§11`)
2. **Identify failure patterns** — incomplete search? missing gates? stale info? process skip?
3. **Design bottom-up from prevention** — what structure would have PREVENTED each failure?
4. **Challenge your own structure** — is this too complex? too simple? right ordering? missing a variable?
5. **Pressure-test against specific past failures** — would this design have caught ML RR1.0-only? E0 biases? threshold artifact?
6. **Check `docs/STRATEGY_BLUEPRINT.md`** — does the proposal follow the test sequence (§3)? Check NO-GO registry (§5). Check "What We Might Be Wrong About" (§10).
7. **Only then propose** — with the failure analysis visible, not hidden.

Minimum 3 takes for any design touching trading logic or research methodology. Show the reasoning, not just the conclusion. The user wants to SEE the deliberation, not just the output.

### 4. Propose Approaches
- 2-3 options with trade-offs
- Lead with your recommendation and why
- For each option, state:
  - **Files touched** (create/modify/delete)
  - **Blast radius** (what else could break)
  - **Rebuild impact** (does gold.db need rebuilding?)
  - **Test impact** (new tests needed? existing tests break?)

### 5. Present Design
- Scale detail to complexity — a few sentences for simple, full sections for complex
- Check after each section: "does this look right so far?"
- Cover: what, why, how, what could go wrong

### 6. Write Design Doc
- Save to `docs/plans/YYYY-MM-DD-<topic>-design.md`
- Commit the design doc

### 7. Transition
- If user says "implement" / "go" / "build it" → create implementation plan
- If user says "iterate" / "change X" → revise design
- If user says nothing → stay in design mode, don't assume

## Project-Specific Checks

Before finalizing any design, verify:

- [ ] Does a spec exist in `docs/specs/`? If yes, follow it exactly.
- [ ] Does the design touch trading logic? If yes, defer to `TRADING_RULES.md`.
- [ ] Does the design touch research methodology? If yes, defer to `RESEARCH_RULES.md`.
- [ ] Does the design require a gold.db schema change? If yes, flag rebuild cost.
- [ ] Does the design add a new session/instrument/entry model? If yes, flag the full rebuild chain: outcomes → discovery → validation → edge families.
- [ ] Is there prior research on this topic? Check memory before designing from scratch.
- [ ] Have you checked `docs/STRATEGY_BLUEPRINT.md`? Route to correct section. Check NO-GO registry. Check "What We Might Be Wrong About."
- [ ] Did you do multi-take deliberation (§3 above)? Minimum 3 takes for trading/research designs.

## Anti-Patterns

| Don't | Do |
|-------|-----|
| Jump to code because "it's simple" | Even simple changes get a 3-sentence design |
| Design in a vacuum | Check what already exists first |
| Present one option | Always 2-3 with trade-offs |
| Ask 5 questions at once | One per message |
| Assume "plan" means "implement" | Design mode until explicit green light |

## Next → After Design Approved

- User says "go" / "build it" → `/4tp [topic]` to plan + implement
- User says "review first" → `/bloomey-review` on the design
- User says "research first" → `/research [hypothesis]` to validate assumptions
- User says "iterate" → revise design, stay in brainstorm mode
