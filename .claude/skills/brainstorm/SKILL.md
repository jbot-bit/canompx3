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

### 3. Propose Approaches
- 2-3 options with trade-offs
- Lead with your recommendation and why
- For each option, state:
  - **Files touched** (create/modify/delete)
  - **Blast radius** (what else could break)
  - **Rebuild impact** (does gold.db need rebuilding?)
  - **Test impact** (new tests needed? existing tests break?)

### 4. Present Design
- Scale detail to complexity — a few sentences for simple, full sections for complex
- Check after each section: "does this look right so far?"
- Cover: what, why, how, what could go wrong

### 5. Write Design Doc
- Save to `docs/plans/YYYY-MM-DD-<topic>-design.md`
- Commit the design doc

### 6. Transition
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

## Anti-Patterns

| Don't | Do |
|-------|-----|
| Jump to code because "it's simple" | Even simple changes get a 3-sentence design |
| Design in a vacuum | Check what already exists first |
| Present one option | Always 2-3 with trade-offs |
| Ask 5 questions at once | One per message |
| Assume "plan" means "implement" | Design mode until explicit green light |
