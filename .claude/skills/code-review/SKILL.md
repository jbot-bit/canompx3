---
name: code-review
description: >
  Post-work code review against plan, project standards, and execution verification.
  Dispatches a reviewer that checks Seven Sins, canonical integrity, trading logic,
  caller discipline, integration paths, and runs execution tests on critical claims.
  Grounded in 15 hard lessons + Google eng-practices.
---

# Code Review

Dispatch a thorough code review after completing work. Catches issues before they compound.

## When to Use

**Mandatory:** After completing a feature, plan task, or complex fix. Before merging.
**Optional:** When stuck (fresh perspective), before refactoring (baseline check).

## Process

### Step 1: Identify the scope

```bash
# Last commit:
git log --oneline -1
git diff HEAD~1

# Session work:
git log --oneline HEAD~N..HEAD  # N = number of commits this session

# Branch work:
BASE_SHA=$(git merge-base HEAD origin/main)
```

### Step 2: Load stage context

If `docs/runtime/STAGE_STATE.md` exists, read it for: task description, scope_lock (flag changes outside it), acceptance criteria, blast radius. Also check `docs/plans/` for the design doc.

### Step 3: Dispatch the review agent

Use Agent tool with `pr-review-toolkit:code-reviewer` subagent type. Include ALL of the following in the prompt:

1. **What was implemented** — from STAGE_STATE or one sentence
2. **Commit range** — BASE_SHA..HEAD_SHA
3. **Scope lock** — reviewer should flag edits outside scope
4. **Semi-formal reasoning instruction** (below)
5. **ALL review sections A-G** (below) — copy them into the agent prompt

**Semi-formal reasoning (MANDATORY for every finding):**

> For each finding, complete this chain before reporting:
> 1. **PREMISE:** One-sentence claim
> 2. **TRACE:** file:line → call/import → file:line (follow the chain, don't guess)
> 3. **EVIDENCE:** Quote code lines or execution output
> 4. **CONCLUSION:** SUPPORT (report) | REFUTE (discard silently) | INSUFFICIENT (say UNSUPPORTED)
>
> Do NOT report findings where TRACE or EVIDENCE is empty. Show full chain for Critical. Line citation suffices for Important/Minor.

---

## Review Sections (ALL MANDATORY)

### Section A: Seven Sins (from `integrity-guardian.md`)
- [ ] No hardcoded instrument lists, session names, cost values, or DB paths — import from canonical sources
- [ ] No bare `except Exception` returning success in health/audit paths
- [ ] No hardcoded check counts — compute dynamically
- [ ] Subprocess return codes checked — zero is the only success
- [ ] No `except Exception: pass` outside atexit handlers
- [ ] No research stats inlined in code comments or docstrings

### Section B: Canonical Integrity
- [ ] Active instruments from `pipeline.asset_configs.ACTIVE_ORB_INSTRUMENTS`
- [ ] Sessions from `pipeline.dst.SESSION_CATALOG`
- [ ] Entry models / filters from `trading_app.config`
- [ ] Cost specs from `pipeline.cost_model.COST_SPECS`
- [ ] DB path from `pipeline.paths.GOLD_DB_PATH`
- [ ] One-way dependency: `pipeline/` never imports from `trading_app/`

### Section C: Trading Logic Guards
- [ ] No changes to entry model behavior without explicit approval
- [ ] No changes to filter logic without explicit approval
- [ ] `orb_outcomes` never queried without `daily_features` filter join (triple-join: trading_day + symbol + orb_minutes)
- [ ] Trading day boundary (09:00 Brisbane) respected
- [ ] Session times never computed manually — use `SESSION_CATALOG` resolvers

### Section D: Caller Discipline (Hard Lesson #2, #8)
- [ ] For ANY changed function signature: `grep -r "function_name"` for ALL callers — verify every caller updated
- [ ] For ANY changed type (set→dict, str→enum): check every `sorted()`, `for x in`, f-string that touched the old type
- [ ] Blast radius scanned in BOTH directions: what does this call (callees) AND what calls this (callers)
- [ ] "Backward compatible default" is NOT an excuse to skip caller updates — masks silent bugs

### Section E: Integration Path Verification (NEW — from Apr 4 session failures)
- [ ] For CLI entry points (scripts/): trace the full call chain and verify it doesn't crash on real data
- [ ] For multi-component changes: verify the components actually connect (not just individually correct)
- [ ] Run `python -c "from X import Y; Y(args)"` on at least ONE critical new code path
- [ ] Check: "what SHOULD exist here but doesn't?" (missing gates, missing validation, missing error handling)
- [ ] For profiles/config: verify the config can actually be USED by the execution engine end-to-end

### Section F: Execution Verification (Rule #7: Never Trust Metadata)
- [ ] For fail-open/fail-closed claims: inject the failure condition, verify the behavior
- [ ] For new gates/guards: verify they actually block (not just claim to) — send a test event through
- [ ] Labels, field values, docstrings are NOT evidence — execution output is evidence
- [ ] For `except Exception` blocks: verify the except path does what the comment claims

### Section G: Test & Drift
- [ ] Companion test updated if production file changed (check `TEST_MAP` in `.claude/hooks/post-edit-pipeline.py`)
- [ ] Tests fail when code is broken (not just pass when code works)
- [ ] `python pipeline/check_drift.py` passes
- [ ] Idempotent write pattern (DELETE then INSERT, not upsert)

### Section H: Blueprint Cross-Check (if changes touch strategy/research/trading logic)
- [ ] `docs/STRATEGY_BLUEPRINT.md §5` — are we reimplementing a NO-GO?
- [ ] `docs/STRATEGY_BLUEPRINT.md §10` — do we depend on a flagged assumption?
- [ ] ML is DEAD (0/12 BH survivors) — flag any ML revival without explicit approval

### Section I: Improvements (NEW)
After all findings, suggest 1-3 concrete improvements grounded in what the review revealed:
- Missing guard that SHOULD exist (gap analysis)
- Test that would catch this class of bug in the future
- Architectural cleanup that would prevent the issue category

---

## Issue Severity

| Severity | Examples | Action |
|----------|----------|--------|
| Critical | Hardcoded instrument list, bare except in health path, trading logic change without approval, one-way dependency violation, execution path crashes | Fix immediately |
| Important | Missing companion test, caller not updated, integration path untested, research stat inlined | Fix before committing |
| Minor | Naming, comment clarity, import ordering | Note for later |

## Act on Feedback

- **Critical** — fix immediately, do not proceed
- **Important** — fix before committing
- **Minor** — note for later
- **Wrong** — push back with execution evidence (test output, code trace)

## Next → After Review

- All clean? → `/verify done` then commit
- Critical findings? → fix, re-review
- Improvements identified? → add to action queue or HANDOFF.md
