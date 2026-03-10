Run one Ralph Loop iteration (Audit → Implement → Verify) inline. No subagents. Sonnet-optimized: tight scope, fast gates, targeted tests only.

Use when: "ralph", "run ralph", "ralph loop", "ralph audit", "autonomous audit"

Scope: $ARGUMENTS (e.g. "live_config.py", "outcome_builder", "all deferred"). If empty, use Next Targets from docs/ralph-loop/ralph-loop-audit.md.

---

## Step 0: State

Read `docs/ralph-loop/ralph-loop-history.md` — count `## Iteration` headers, add 1 = ITER.
Read `docs/ralph-loop/ralph-loop-audit.md` — get deferred findings and Next Targets.

Scope = $ARGUMENTS if provided, else Next Targets[0] from audit file (ONE target, not all).

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
- Blast radius ≤ 3 files
- NOT schema/entry model change
- NOT already listed as DEFERRED or flagged for human

If no eligible finding → write "no eligible finding" to plan, skip to Step 5.

Write to `docs/ralph-loop/ralph-loop-plan.md`:
```
## Iteration: ITER
## Phase: implement
## Target: file:line
## Finding: 1-sentence
## Decision: implement
## Rationale: why safe, why now
## Blast Radius: callers of changed function (list files)
## Diff estimate: N lines
```

---

## Step 3: IMPLEMENT

Read `.claude/agents/ralph-implementer.md`.

**Pass 1 (Discovery):** Read the target file and all blast radius files. Read companion test. Understand what must NOT change.

**Pass 2 (Implementation):** Apply the minimal fix. Then:
```
python -m pytest tests/test_trading_app/test_<module>.py -x -q
python pipeline/check_drift.py
```

If either fails → revert change, mark REJECTED in plan, skip to Step 5.

Update plan with: lines changed, test result, drift result, Ready=YES/NO.

---

## Step 4: VERIFY

Read `.clone/agents/ralph-verifier.md`.

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
- Verification: PASS/REJECT — gate summary
- Commit: hash or NONE
```

Report:
```
=== RALPH LOOP ITER COMPLETE ===
Scope: [target]
Audit: N findings (X CRIT, X HIGH, X MED, X LOW)
Action: [fix | audit-only]
Target: [file:line]
Verdict: [ACCEPT | REJECT | SKIPPED]
Commit: [hash or NONE]
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
