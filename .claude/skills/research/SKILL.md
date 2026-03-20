---
name: research
description: >
  Structured research with Blueprint test sequence, multi-take deliberation,
  and adversarial verification. For any trading hypothesis, feature evaluation,
  or edge investigation.
allowed-tools: Read, Grep, Glob, Bash
---

Research this topic and produce actionable findings: $ARGUMENTS

Use when: "research", "investigate", "test this hypothesis", "is X real", "does Y work", "analyze", "deep dive on"

## Philosophy

**Research is adversarial by default.** Your job is to DISPROVE the hypothesis, not confirm it. If it survives your best attempts to kill it, it might be real. RESEARCH_RULES.md is the governing document.

**System is not proven right — just not-yet-wrong.** Every finding is provisional until forward-tested.

## Step 0: Pre-Research Checks (MANDATORY)

Before doing ANY analysis:

1. **Blueprint NO-GO check (§5):** Is this topic already dead?
   - Read `docs/STRATEGY_BLUEPRINT.md` §5 NO-GO Registry
   - If the path is dead, tell the user immediately with the evidence pointer
   - If the user insists, note the NO-GO and proceed with extra adversarial scrutiny

2. **Blueprint "What We Might Be Wrong About" (§10):** Does this research depend on an assumption that could be wrong?

3. **Previous research check:** Search memory files and `TRADING_RULES.md` for prior findings on this topic. Don't re-run dead-end research.

4. **Variable space scoping:** What variables will you search? List them explicitly.
   - **CRITICAL RULE:** Before declaring ANY dimension dead, test ≥3 values (Blueprint §3 Gate 2)

## Step 1: Frame the Hypothesis

State explicitly:
- **H0 (null):** There is no effect. (This is what you're trying to fail to reject.)
- **H1 (alternative):** [specific, testable claim]
- **Mechanism:** WHY should H1 be true? Structural market reason?
- **Kill criterion:** What result would make you abandon this?
- **Variables to test:** [list with ranges]

If no mechanism exists → flag as "extra scrutiny required" and proceed with adversarial mindset.

## Step 2: Multi-Take Deliberation

Before running any queries, do at least 3 takes:

1. **What could go wrong?** What biases could fake this result? (lookahead, survivorship, threshold artifact, fill assumptions)
2. **What went wrong before?** Check `hard_lessons.md` and Blueprint §11 Failure Patterns for relevant precedents.
3. **Is this the right test?** Am I testing what I think I'm testing? Could the same result arise from a confound?

## Step 3: Run the Analysis

Follow Blueprint §3 test sequence:

**Gate 1 — Mechanism:** Already stated in Step 1. If missing, flag it.

**Gate 2 — Baseline viability:**
```bash
python -c "
import duckdb
from pipeline.paths import GOLD_DB_PATH
con = duckdb.connect(str(GOLD_DB_PATH), read_only=True)
# [WRITE QUERY SPECIFIC TO YOUR HYPOTHESIS]
# Always test ≥3 values per dimension
con.close()
"
```

**Gate 3 — Statistical significance:**
- Run the ACTUAL test (t-test, permutation, BH FDR). NEVER eyeball.
- Report: N, time span, exact p-value, number of variations tested.

**Gate 4 — OOS if applicable:**
- Time split: check if effect persists across train/val/test periods
- Year-by-year: does it hold every year or just 1-2?

**Gate 5 — Adversarial:**
- Sensitivity: change parameter ±20%. Does it survive?
- If ML: bootstrap permutation (200 perms minimum)
- If novel: consider fresh agent audit

## Step 4: Report Findings

Use the RESEARCH_RULES.md honest summary format:

```
HYPOTHESIS: [H1 statement]
MECHANISM: [structural reason, or "none identified"]

SURVIVED SCRUTINY:
- [Finding 1]: N=X, p=Y, time span Z
- [Finding 2]: ...

DID NOT SURVIVE:
- [Finding A]: killed by [reason]
- [Finding B]: ...

VARIABLE COVERAGE:
- RR tested: [list] — missing: [list]
- Apertures: [list] — missing: [list]
- Sessions: [list] — missing: [list]
- Entry models: [list] — missing: [list]

CAVEATS:
- [What could still be wrong]
- [What assumptions are untested]

NEXT STEPS:
- [What would validate or invalidate]
- [Which Blueprint gate is next]

LABEL: [validated finding / promising hypothesis / statistical observation / NO-GO]
```

## Step 5: Update Records

- If NO-GO: suggest updating Blueprint §5 NO-GO Registry
- If finding: save to appropriate memory file with `@research-source` annotation
- If promising: suggest next Blueprint gate

## Rules

- NEVER say "significant" without exact p-value
- NEVER present counts or eyeball comparisons as analysis — RUN THE TEST
- NEVER trust metadata — read the code, run the query, check the output
- NEVER skip the NO-GO check — it exists because we wasted months on dead paths
- Every claim needs evidence: line numbers, row counts, test output
- Honesty over outcome. If the hypothesis is dead, say so directly.
- The user prefers STATISTICS FIRST — p-value from actual test, not rule-of-thumb
