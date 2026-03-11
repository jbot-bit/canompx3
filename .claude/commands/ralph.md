Run one Ralph Loop iteration (Audit → Understand → Implement → Verify) inline. No subagents. Sonnet-optimized: tight scope, targeted tests only.

Use when: "ralph", "run ralph", "ralph loop", "ralph audit", "autonomous audit"

Scope: $ARGUMENTS (e.g. "live_config.py", "outcome_builder", "all deferred"). If empty, use Next Targets from docs/ralph-loop/ralph-loop-audit.md.

---

## Philosophy

Autonomous ≠ fast. Autonomous = correct without supervision.
Wrong autonomy costs more than no autonomy. Every step earns the right to proceed.

---

## Step 0: State

Read `docs/ralph-loop/ralph-loop-history.md` — count `## Iteration` headers, add 1 = ITER.
Read `docs/ralph-loop/ralph-loop-audit.md` — get deferred findings and Next Targets.
Read `docs/ralph-loop/deferred-findings.md` — check Open Findings table for outstanding debt.

Scope = $ARGUMENTS if provided, else Next Targets[0] from audit file (ONE target, not all).
If scope is `"all deferred"` — work through the deferred findings ledger by priority.

Announce: `=== RALPH LOOP — Iteration ITER | Scope: SCOPE ===`

---

## Step 1: AUDIT PHASE

Read `.claude/agents/ralph-auditor.md` for methodology.

Run infrastructure gates (report exact last line of each):
```
python pipeline/check_drift.py
python scripts/tools/audit_behavioral.py
ruff check pipeline/ trading_app/ scripts/
```

**SKIP `pytest tests/` — it OOMs. Run targeted tests only:**
```
python -m pytest tests/test_trading_app/test_<scope_module>.py -x -q
```
(derive the test file from the scope — e.g. scope=live_config → tests/test_trading_app/test_live_config.py)

Read the 1-2 target files from scope. Scan for Seven Sins:
- Silent failure, fail-open, phantom state, look-ahead bias, cost illusion, orphan risk

Check canonical integrity on scope files only:
- Hardcoded lists, magic numbers, one-way deps

Overwrite `docs/ralph-loop/ralph-loop-audit.md` with structured findings (format from agent prompt). Keep deferred findings list updated.

If drift or behavioral audit FAIL → stop, do not implement.

---

## Step 2: SELECT TOP FINDING

Rank: CRITICAL > HIGH > MEDIUM > LOW.

Select the highest-priority finding that:
- Has a clear provable fix
- NOT schema/entry model change
- NOT already listed as DEFERRED or flagged for human

If no eligible finding → write "no eligible finding" to plan, skip to Step 5.

---

## Step 2.5: UNDERSTAND BEFORE TOUCHING (MANDATORY)

**This step is non-negotiable. You do NOT earn the right to edit code until you complete it.**

### A. Blast Radius — Map It, Don't Guess

For the selected finding, identify ALL affected code:
1. **Grep all callers** of the function/method you plan to change
2. **Grep all callees** — what does the changed code depend on?
3. **Check companion tests** — which test file covers this code?
4. **Check drift checks** — does any drift check reference this code?
5. **Check canonical sources** — does this touch SESSION_CATALOG, ENTRY_MODELS, COST_SPECS, ORB_LABELS, or ACTIVE_ORB_INSTRUMENTS?

If blast radius > 5 files → STOP. Escalate to 4T orient phase before proceeding.
If blast radius touches schema/entry models/pipeline data flow → STOP. Read the relevant guardian prompt (ENTRY_MODEL_GUARDIAN or PIPELINE_DATA_GUARDIAN).

### B. Prove Understanding — State What Must NOT Change

Before writing any code, write to `docs/ralph-loop/ralph-loop-plan.md`:
```
## Iteration: ITER
## Phase: implement
## Target: file:line
## Finding: 1-sentence
## Decision: implement
## Rationale: why safe, why now
## Blast Radius:
  - Callers: [list files that call this function]
  - Callees: [list files this function calls]
  - Tests: [companion test file]
  - Drift checks: [any referencing this code]
## Invariants (MUST NOT change):
  - [list 2-3 specific behaviors that must be preserved]
## Diff estimate: N lines
```

The **Invariants** section is the key. If you can't articulate what must NOT break, you don't understand the change well enough to make it.

### C. Escalation — When to Invoke 4T

If during this step you discover:
- Blast radius > 5 files → run /4t orient on the finding first
- Schema change needed → read PIPELINE_DATA_GUARDIAN, then /4t
- Entry model change → read ENTRY_MODEL_GUARDIAN, then /4t
- Multiple files need coordinated changes → /4t design phase
- You're unsure about the invariants → /4t orient until you're sure

4T is not overhead. 4T is how you avoid rework. The time spent understanding saves 3x the time spent fixing regressions.

---

## Step 3: IMPLEMENT

Read `.claude/agents/ralph-implementer.md`.

**Pass 1 (Verify Understanding):** Re-read the target file, blast radius files, and companion test. Confirm the invariants from Step 2.5 are correct. If anything surprises you → STOP, update the plan.

**Pass 2 (Implementation):** Apply the minimal fix. Then:
```
python -m pytest tests/test_trading_app/test_<module>.py -x -q
python pipeline/check_drift.py
```

If either fails → revert change, mark REJECTED in plan, skip to Step 5.

Update plan with: lines changed, test result, drift result, Ready=YES/NO.

---

## Step 4: VERIFY

Read `.claude/agents/ralph-verifier.md`.

Run these gates only (not full suite):
```
Gate 1: python pipeline/check_drift.py
Gate 2: python scripts/tools/audit_behavioral.py
Gate 3: python -m pytest tests/test_trading_app/test_<module>.py -x -q
Gate 4: ruff check <changed files>
Gate 5: Blast radius — grep callers, verify all handle new behavior
Gate 6: Targeted regression — python -m pytest <specific test class> -x -v
```

Verdict:
- All 6 PASS → ACCEPT, commit: `git add <changed files> && git commit -m "fix: Ralph Loop iter ITER — <finding>"`
- Only Gate 4 fails → ACCEPT with NOTE (lint non-blocking)
- Gate 1 or Gate 3 fail → HARD REJECT

Write verdict to plan file.

---

## Step 5: HISTORY + REPORT

Append to `docs/ralph-loop/ralph-loop-history.md`:
```
## Iteration ITER — YYYY-MM-DD
- Phase: fix | audit-only | rejected
- Target: file:line or "full audit"
- Finding: 1-sentence
- Action: what was done
- Blast radius: N files checked
- Verification: PASS/REJECT — gate summary
- Commit: hash or NONE
```

**Update deferred findings ledger** (`docs/ralph-loop/deferred-findings.md`):
- Any finding deferred this iteration → add row to Open Findings with ID, severity, target
- Any finding resolved this iteration → move from Open to Resolved with commit hash
- NEVER silently drop a finding. If it was deferred, it goes in the ledger.

Report:
```
=== RALPH LOOP ITER COMPLETE ===
Scope: [target]
Audit: N findings (X CRIT, X HIGH, X MED, X LOW)
Action: [fix | audit-only]
Target: [file:line]
Blast radius: [N files, key callers]
Verdict: [ACCEPT | REJECT | SKIPPED]
Commit: [hash or NONE]
Deferred debt: [N open items in ledger]
Next: [top deferred finding]
================================
```

---

## Critical Rules

- **NO `pytest tests/` ever** — causes OOM. Targeted tests only.
- **No subagents** — do all 3 phases directly in this session.
- **One target at a time** — 1-2 files per iteration, not 5.
- **Fail-closed** — unknown state = block, not pass.
- **Evidence over assertion** — show command output, not claims.
- **Understand before editing** — Step 2.5 is mandatory. No exceptions. No shortcuts.
- **Escalate, don't force** — if the change is bigger than expected, invoke /4t. That's not failure, that's discipline.
