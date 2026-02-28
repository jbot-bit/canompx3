Post-completion integrity gate. Run after finishing a task to verify quality.

Execute these 6 steps in order. Do NOT skip steps.

## Step 1: Instruction Intake
- Restate what changed in 1-2 sentences
- Identify the governing authority doc (CLAUDE.md / TRADING_RULES.md / RESEARCH_RULES.md)
- List all modified files

## Step 2: Canonical Alignment
For each modified file, check for new hardcoded lists or magic numbers.
Run the behavioral audit:
```bash
python scripts/tools/audit_behavioral.py
```
If violations found, fix them before proceeding.

## Step 3: Impact Map
Build a table for every modified production file:

| File Changed | Test File | Test Updated? | Doc References | Doc Updated? | Drift Check? |
|---|---|---|---|---|---|

Use `TEST_MAP` from `.claude/hooks/post-edit-pipeline.py` to find test files.

## Step 4: Fail-Closed Verification
Review all try/except blocks in modified files:
- No `except Exception` returning success in health/audit paths
- No hardcoded check counts
- All subprocess return codes checked

## Step 5: Pre-Final Gate
Run all quality gates — ALL must pass:
```bash
python pipeline/check_drift.py
python scripts/tools/audit_integrity.py
python scripts/tools/audit_behavioral.py
python -m pytest tests/ -x -q
```

## Step 6: Evidence Block
Emit this structured block:

```
=== INTEGRITY GUARDIAN REPORT ===
Files modified: [list]
Authority doc: [which]
Drift check: PASS/FAIL
Integrity audit: PASS/FAIL
Behavioral audit: PASS/FAIL
Tests: PASS/FAIL (N passed)
Open issues: [any remaining items]
================================
```
