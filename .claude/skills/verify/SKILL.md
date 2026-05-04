---
name: verify
description: >
  Run verification gates. Modes: quick (5 gates pre-commit), full (impact map + gates),
  done (stage acceptance + lint/types + gates). Default: quick.
  Use when: "verify", "health check", "pre-commit", "is everything ok", "are we good",
  "integrity check", "done?", "all green?", "run the gates".
---

# Verify

Run verification gates: $ARGUMENTS

**Modes:** `quick` (default) | `full` | `done`

## Mode: quick (pre-commit gates)

Run ALL 5 gates. ANY failure = STOP.

```bash
python pipeline/check_drift.py
python scripts/tools/audit_integrity.py
python scripts/tools/audit_behavioral.py
python -m pytest tests/ -x -q
ruff check pipeline/ trading_app/ scripts/ --quiet
```

Emit evidence block:
```
=== VERIFY (quick) ===
Drift:      PASS/FAIL (N checks passed)
Integrity:  PASS/FAIL
Behavioral: PASS/FAIL
Tests:      PASS/FAIL (N passed)
Lint:       PASS/FAIL
======================
```

## Mode: full (post-task with impact map)

**Step 1:** Restate what changed. Identify governing doc (CLAUDE.md / TRADING_RULES.md / RESEARCH_RULES.md).

**Step 2:** Build impact table for every modified production file:

| File | Test File | Test Updated? | Doc Refs | Drift? |
|------|-----------|---------------|----------|--------|

Use `TEST_MAP` from `.claude/hooks/post-edit-pipeline.py` for companion tests.

**Step 3:** Check fail-closed patterns in modified files (no bare `except Exception` returning success, no hardcoded counts, subprocess return codes checked).

**Step 4:** Run all 5 gates (same as quick mode).

**Step 5:** Emit evidence block with file list and impact table.

## Mode: done (stage completion gate)

**Core rule:** No completion claims without fresh evidence.

**Step 1:** If `docs/runtime/STAGE_STATE.md` (or `stages/*.md`) has acceptance criteria — run those commands FIRST.

**Step 2:** Run lint + type check:
```bash
ruff check pipeline/ trading_app/ scripts/ --quiet
ruff format --check pipeline/ trading_app/ scripts/
```

**Step 3:** Run all 5 gates (drift, integrity, behavioral, tests, lint).

**Step 4:** Run targeted companion tests for changed files (check `TEST_MAP`).

**Step 5:** Severity-specific checks:

| What changed | Extra gate |
|---|---|
| Schema | `python pipeline/init_db.py` test |
| Strategy/research | Blueprint SS3 test sequence check |
| Config/canonical | Manual review of imports |

**Step 6:** Emit evidence block. Update STAGE_STATE to mark DONE if all pass.

## Rules

- ALL gates must pass. No exceptions. No "it's just minor."
- Never hardcode check counts — self-reported at runtime.
- Never claim "done" without fresh evidence in THIS response.
- If ANY gate fails: stop, investigate, fix, re-run ALL gates.
- For institutional code review with grading → `/code-review`
