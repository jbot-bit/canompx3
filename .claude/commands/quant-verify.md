Run pre-commit quality gates and verify everything passes: $ARGUMENTS

Use when: about to claim work is done, before commits, before PRs, "verify", "are we good", "run the gates", "check everything", "pre-commit", "all green?"

## Quick Verification Gates

Lightweight pre-commit check. For full impact mapping, use /integrity-guardian instead.
For institutional code review with grading, use /bloomey-review.

### Run ALL 4 gates. ANY failure = STOP.

**Gate 1: Drift Detection**
```bash
python pipeline/check_drift.py
```
Check count is self-reported at runtime. NEVER hardcode "all N checks passed."

**Gate 2: Data Integrity**
```bash
python scripts/tools/audit_integrity.py
```

**Gate 3: Behavioral Audit**
```bash
python scripts/tools/audit_behavioral.py
```

**Gate 4: Test Suite**
```bash
python -m pytest tests/ -x -q
```

### Evidence Block

After all 4 gates, emit:

```
=== VERIFY GATES ===
Drift:      PASS/FAIL
Integrity:  PASS/FAIL
Behavioral: PASS/FAIL
Tests:      PASS/FAIL (N passed)
===================
```

### Rules

- ALL 4 must pass. No exceptions. No "it's just a minor fail."
- If ANY gate fails: stop, investigate, fix, re-run ALL gates.
- Never claim "done" without this evidence block.
- Never hardcode check counts -- they are self-reported at runtime.
- This is the LIGHTWEIGHT check. For full impact mapping + evidence, use /integrity-guardian.
