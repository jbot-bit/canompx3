---
name: code-review
description: >
  Post-work code review against plan and project standards. Use after completing
  a task, feature, or fix. Dispatches a reviewer that checks against the plan,
  Seven Sins, canonical integrity, and trading logic correctness.
---

# Code Review

Dispatch a code review after completing work. Catches issues before they compound.

## When to Use

**Mandatory:**
- After completing a plan task or feature
- Before merging to main
- After fixing a complex bug

**Optional:**
- When stuck (fresh perspective)
- Before refactoring (baseline check)

## Process

**1. Identify the scope:**
```bash
BASE_SHA=$(git merge-base HEAD origin/main)  # or HEAD~N for recent work
HEAD_SHA=$(git rev-parse HEAD)
```

**2. Dispatch the review agent:**

Use Agent tool with `pr-review-toolkit:code-reviewer` subagent type. Provide:

- **What was implemented** — one sentence
- **Plan or requirements** — reference the design doc or task
- **BASE_SHA / HEAD_SHA** — commit range
- **Project-specific review criteria** (below)

Alternatively, for a deeper quant-focused review with Seven Sins grading, use `/bloomey-review`.

**3. Project-Specific Review Criteria**

The reviewer MUST check these in addition to standard code quality:

### Seven Sins (from `integrity-guardian.md`)
- [ ] No hardcoded instrument lists, session names, cost values, or DB paths — import from canonical sources
- [ ] No bare `except Exception` returning success in health/audit paths
- [ ] No hardcoded check counts — compute dynamically
- [ ] Subprocess return codes checked — zero is the only success

### Canonical Integrity
- [ ] Active instruments from `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`
- [ ] Sessions from `pipeline.dst.SESSION_CATALOG`
- [ ] Entry models / filters from `trading_app.config`
- [ ] Cost specs from `pipeline.cost_model.COST_SPECS`
- [ ] DB path from `pipeline.paths.GOLD_DB_PATH`

### Trading Logic Guards
- [ ] No changes to entry model behavior without explicit approval
- [ ] No changes to filter logic without explicit approval
- [ ] `orb_outcomes` never queried without `daily_features` filter join
- [ ] Trading day boundary (09:00 Brisbane) respected

### Pipeline Patterns
- [ ] Idempotent write pattern (DELETE then INSERT, not upsert)
- [ ] Fail-closed on validation failure
- [ ] One-way dependency: `pipeline/` never imports from `trading_app/`

### Test & Drift
- [ ] Companion test updated if production file changed (check `TEST_MAP`)
- [ ] `python pipeline/check_drift.py` passes
- [ ] No research stats inlined in code comments or docstrings

**4. Act on feedback:**
- **Critical** — fix immediately, do not proceed
- **Important** — fix before committing
- **Minor** — note for later
- **Wrong** — push back with evidence (test output, code reference)

## Issue Severity for This Project

| Severity | Examples |
|----------|----------|
| Critical | Hardcoded instrument list, bare except in health path, trading logic change without approval, one-way dependency violation |
| Important | Missing companion test, drift check not run, research stat inlined in docstring |
| Minor | Naming convention, comment clarity, import ordering |
