---
name: quant-audit
description: Institutional hypothesis validator — falsify claims before accepting them. Fail-closed at every step.
---
Institutional hypothesis validator — falsify research claims before accepting them: $ARGUMENTS

Use when: "audit this claim", "is this real", "stress test", "validate finding", "quant audit", "falsify", "test this hypothesis", "prop desk review"

## Process

Follow `.claude/rules/quant-audit-protocol.md` EXACTLY. The protocol has 6 steps:

1. **PRE-FLIGHT** — DB freshness, row counts, schema confirmation. HALT if stale.
2. **CLAIM DECOMPOSITION** — Number each claim, classify type, state decision impact.
3. **FAILURE MODE ANALYSIS** — Score 4 risks (multiple testing, selection bias, overfitting, pipeline) per claim.
4. **TEST PLAN** — Concrete SQL skeletons, IS/OOS windows, sensitivity sweeps, pass/fail thresholds. Define BEFORE running.
5. **EXECUTION ORDER** — Highest decision-impact first.
6. **DECISION RULES** — KEEP/KILL/DOWNGRADE/SUSPEND criteria defined BEFORE seeing results.

**Output the plan first. Do NOT run tests until user says "go" or "run it."**

## Hard Rules

- Canonical layers only (`bars_1m`, `daily_features`, `orb_outcomes`)
- NO conclusions before test results
- NO "looks promising" / "likely valid" / "interesting" language
- Fail-closed: unknown = reject, stale = halt, ambiguous = clarify
- All p-values adjusted for honest K (per `RESEARCH_RULES.md` BH K selection rule)
- Lookahead variables banned: `double_break` is LOOKAHEAD (code line 393)
- Trade-time-knowable only: `risk_dollars`, `break_delay_min`, `rel_vol`, `atr_20`, `orb_size`

## Required Tests for `validated_finding` Upgrade

ALL must pass:
1. In-sample baseline
2. Walk-forward (expanding window, WFE > 0.50)
3. Sensitivity ±20% on each parameter
4. All-session family comparison
5. Bootstrap permutation (1000+ perms)
6. Per-year stability (positive ≥7/10 years)
7. Cross-instrument directional consistency
