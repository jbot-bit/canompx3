---
name: research
description: >
  Structured research with Blueprint test sequence, multi-take deliberation,
  and adversarial verification. For any trading hypothesis, feature evaluation,
  or edge investigation.
allowed-tools: Read, Grep, Glob, Bash
effort: high
---

Research this topic and produce actionable findings: $ARGUMENTS

Use when: "research", "investigate", "test this hypothesis", "is X real", "does Y work", "deep dive"

**Research is adversarial by default.** Disprove the hypothesis. If it survives, it might be real.

## Step 0: Pre-Research (MANDATORY)

1. **Blueprint NO-GO (SS5):** Already dead? Tell user, STOP.
2. **Assumptions (SS10):** Does this depend on a flagged assumption?
3. **Previous research:** Check memory + TRADING_RULES.md. Don't re-run dead ends.
4. **Variable scoping:** List variables to test. Before declaring ANY dimension dead, test >=3 values (Gate 2).

## Step 1: Frame the Hypothesis

- **H0:** No effect (trying to fail to reject)
- **H1:** Specific, testable claim
- **Mechanism:** WHY should H1 be true? Structural market reason?
- **Kill criterion:** What result kills this?

## Step 2: Multi-Take Deliberation (min 3 takes)

1. What biases could fake this? (lookahead, survivorship, threshold artifact)
2. What went wrong before? (check hard_lessons.md, Blueprint SS11)
3. Is this the right test? Could a confound produce the same result?

## Step 3: Run Analysis (Blueprint Gates)

**Gate 1 — Mechanism:** Stated in Step 1.
**Gate 2 — Baseline:** Query with >=3 values per dimension.
**Gate 3 — Statistics:** Run ACTUAL test. Report N, time span, exact p-value, K.
**Gate 4 — OOS:** Time split + year-by-year.
**Gate 5 — Adversarial:** Sensitivity +-20%. Bootstrap if ML.

## Step 4: Report

```
HYPOTHESIS: [H1]
MECHANISM: [reason or "none"]

SURVIVED: [findings with N, p, time span]
DID NOT SURVIVE: [findings with kill reason]
VARIABLE COVERAGE: [tested vs missing per dimension]
CAVEATS: [what could still be wrong]
NEXT STEPS: [which Gate is next]
LABEL: [validated / promising / observation / NO-GO]
```

## Step 5: Update Records

- NO-GO → update Blueprint SS5
- Finding → save to memory with @research-source
- Promising → suggest next Gate

## Rules

- NEVER "significant" without p-value. NEVER "edge" without BH FDR.
- NEVER skip NO-GO check. NEVER trust metadata — run the query.
- Honesty over outcome. Dead hypothesis = say so directly.
- STATISTICS FIRST — p-value from actual test, not rule-of-thumb.
