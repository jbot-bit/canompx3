Run one Ralph Loop iteration (Audit → Understand → Implement → Verify) inline. No subagents. Sonnet-optimized: tight scope, targeted tests only.

Use when: "ralph", "run ralph", "ralph loop", "ralph audit", "autonomous audit"

Scope: $ARGUMENTS (e.g. "live_config.py", "outcome_builder", "all deferred"). If empty, use Next Targets from docs/ralph-loop/ralph-loop-audit.md.

---

## Philosophy

Autonomous ≠ fast. Autonomous = correct without supervision.
Wrong autonomy costs more than no autonomy. Every step earns the right to proceed.

---

## Step 0: State

Read `docs/ralph-loop/ralph-loop-audit.md` — get ITER from `## Last iteration: N` line (+1 = ITER), get deferred findings and Next Targets.
Read `docs/ralph-loop/deferred-findings.md` — check Open Findings table for outstanding debt.

Scope = $ARGUMENTS if provided, else Next Targets[0] from audit file (ONE target, not all).
If scope is `"all deferred"` — work through the deferred findings ledger by priority.

Announce: `=== RALPH LOOP — Iteration ITER | Scope: SCOPE ===`

---

## Step 1: AUDIT PHASE

Run all 3 infrastructure gates in **parallel** — make 3 simultaneous Bash tool calls in a single response (report exact last line of each):
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

**Batching rule:** If ALL eligible findings are LOW + same file + same fix type (annotation / logging / validation) → batch up to 5 in one pass.

---

## Step 2.5: UNDERSTAND BEFORE TOUCHING (MANDATORY)

**This step is non-negotiable. You do NOT earn the right to edit code until you complete it.**

### A. Blast Radius — Dispatch Agent

Dispatch the `blast-radius` agent (model: haiku) with:
- Target: `file:function_or_line`
- Change: 1-sentence description of what will be modified

It maps callers, importers, companion tests, drift checks, canonical sources, and DB impact.
Write the compact impact report to `docs/ralph-loop/ralph-loop-plan.md`.

If blast radius returns SIGNIFICANT or CRITICAL → STOP. Run /4t orient.
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

**Pass 1 (Verify Understanding):** Review the blast-radius report from Step 2.5. Confirm invariants are correct. If anything surprises you → STOP, update the plan.

**Pass 2 (Implementation):** Apply the minimal fix. Then:
```
python -m pytest tests/test_trading_app/test_<module>.py -x -q
python pipeline/check_drift.py
```

If either fails → revert change, mark REJECTED in plan, skip to Step 5.

Update plan with: lines changed, test result, drift result, Ready=YES/NO.

---

## Step 4: VERIFY

Dispatch `verify-complete` agent (model: haiku) with:
- Changed files from this iteration
- Scope module name (for targeted test selection)
- Note: "SKIP Gate 4 full suite — targeted tests only (OOM risk)"

It runs all gates and returns ACCEPT / ACCEPT WITH NOTE / HARD REJECT with a compact gate summary.

Verdict (from agent report):
- ACCEPT → commit: `git add <changed files> && git commit -m "fix: Ralph Loop iter ITER — <finding>"`
- ACCEPT WITH NOTE → commit with note appended
- HARD REJECT (Gate 1 or Gate 3 fail) → revert change, mark REJECTED in plan, skip to Step 5

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
- **Subagents are Haiku** — blast-radius and verify-complete dispatch with `model: haiku`. Mechanical execution, no reasoning needed. Don't pay Sonnet prices for grep and gate-running.
- **One target at a time** — 1-2 files per iteration, not 5.
- **Fail-closed** — unknown state = block, not pass.
- **Evidence over assertion** — show command output, not claims.
- **Understand before editing** — Step 2.5 is mandatory. No exceptions. No shortcuts.
- **Escalate, don't force** — if the change is bigger than expected, invoke /4t. That's not failure, that's discipline.
- **Update Last iteration** — ralph-loop-audit.md (overwritten each iteration) must include `## Last iteration: N`. This is the single source of ITER for Step 0.
