---
name: verify-done
description: >
  Evidence-before-claims gate. Use before claiming work is complete, fixed, or
  passing. Runs project-specific verification suite and confirms output before
  any success claim. No shortcuts.
---

# Verify Done

**Core rule: No completion claims without fresh verification evidence.**

If you haven't run the command in this response, you cannot claim it passes.

## Stage-Gate Integration

If `docs/runtime/STAGE_STATE.md` exists and has acceptance criteria:
- Read the `## Acceptance` section — these are the SPECIFIC verification targets for this stage
- Run those commands FIRST (they're the task-specific "done" definition)
- Then proceed to the standard gates below
- After all pass: update STAGE_STATE.md to mark stage DONE

If no STAGE_STATE.md → use standard gates only (backward compatible).

## The Gate

Before claiming ANY status (done, fixed, passing, complete, working):

1. **IDENTIFY** — What commands prove this claim?
2. **RUN** — Execute them fresh, in full
3. **READ** — Full output, exit codes, failure counts
4. **VERIFY** — Does output confirm the claim?
5. **REPORT** — State claim WITH evidence, or state actual status

Skip any step = the claim is unverified.

## Project Verification Suite

Run these in order. Stop at first failure.

### Gate 1: Lint + Format
```bash
ruff check pipeline/ trading_app/ scripts/ --quiet
ruff format --check pipeline/ trading_app/ scripts/
```
**Pass criterion:** Exit 0, zero errors.

### Gate 2: Type Check
```bash
pyright --outputjson 2>/dev/null | python -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d.get(\"generalDiagnostics\",[]))} errors')"
```
**Pass criterion:** Zero errors (warnings OK).

### Gate 3: Drift Detection
```bash
python pipeline/check_drift.py
```
**Pass criterion:** All checks pass (count self-reported at runtime — never hardcode).

### Gate 4: Tests
```bash
python -m pytest tests/ -x -q
```
**Pass criterion:** All collected tests pass, exit 0. Note IB-conditional skips are expected (8-9 skipped).

### Gate 5: Targeted Tests (if applicable)
If you changed a specific file, check `TEST_MAP` in `.claude/hooks/post-edit-pipeline.py` for its companion test and run that specifically.

## Red Flags — STOP

| You're thinking... | Reality |
|---------------------|---------|
| "Should work now" | RUN the verification |
| "I'm confident" | Confidence is not evidence |
| "Just a docs change" | Drift checks still apply |
| "Tests passed earlier" | Run them AGAIN — fresh |
| "The agent said success" | Verify independently |
| "Partial check is enough" | Partial proves nothing |

## Severity

| What you changed | Minimum gates |
|------------------|---------------|
| Docs only (no code) | Gate 3 (drift) |
| Pipeline code | Gates 1-4 |
| Trading app code | Gates 1-4 |
| Config / canonical sources | Gates 1-4 + manual review |
| Schema change | Gates 1-4 + `python pipeline/init_db.py` test |
| Strategy/research code | Gates 1-4 + Blueprint §3 test sequence check |
| ML code | Gates 1-4 + lookahead blacklist check + bootstrap if claiming skill |
| Portfolio/paper trading | Gates 1-4 + replay validation matches expectations |

## Trading-Specific Done Criteria

If the work touches strategy, research, or trading logic, also verify:

- [ ] **Blueprint compliance:** Does the implementation follow `STRATEGY_BLUEPRINT.md` test sequence?
- [ ] **NO-GO check:** Did we avoid reimplementing dead paths?
- [ ] **Canonical sources:** All instruments, sessions, costs from imports — nothing hardcoded?
- [ ] **filter_type valid:** Any filter_type used exists in `ALL_FILTERS`?
- [ ] **No stale citations:** Any numbers cited are from fresh queries, not memory?
- [ ] **Lookahead clean:** For ML code — all features in `LOOKAHEAD_BLACKLIST` excluded?

## Output Format

After running verification, report:

```
Verification: [PASS / FAIL]
- Lint: [result]
- Types: [result]
- Drift: [N checks passed]
- Tests: [X passed, Y skipped, Z failed]
- Evidence: [paste key output lines]
```

Never say "done" without this block.
