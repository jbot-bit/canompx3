---
name: audit
description: >
  System audit and hypothesis validation. Modes: full (11 phases), quick
  (phases 0,1,3,6), phase N (single phase), hypothesis (T0-T8 test battery),
  prompts (guardian routing). Default: full.
  Use when: "audit", "health check", "system check", "quant audit", "validate
  finding", "falsify", "stress test", "guardian prompt", "spec check".
disable-model-invocation: true
---

# Audit

System audit and hypothesis validation: $ARGUMENTS

**Modes:** `full` (default) | `quick` | `phase N` | `hypothesis <claim>` | `prompts`

## Mode: full (11 phases)

```bash
python scripts/audits/run_all.py
```
Runs phases 0-10, stopping on CRITICAL. See `docs/prompts/SYSTEM_AUDIT.md`.

## Mode: quick (fast check)

```bash
python scripts/audits/run_all.py --quick
```
Covers: triage (0), automated checks (1), docs (3), build chain (6).

## Mode: phase N (single phase)

```bash
python scripts/audits/run_all.py --phase $ARGUMENTS
```
Phases: 0=triage, 1=automated, 2=infrastructure, 3=docs, 4=config sync, 5=DB integrity, 6=build chain, 7=live trading, 8=tests, 9=research, 10=git/CI.

## Mode: hypothesis (T0-T8 quant audit)

Follow `.claude/rules/quant-audit-protocol.md` EXACTLY. 6 steps, 9 tests.

**Output the plan first. Do NOT run tests until user says "go".**

Steps: PRE-FLIGHT (DB freshness) -> CLAIM DECOMPOSITION -> FAILURE MODE ANALYSIS -> TEST BATTERY (T0-T8 in order) -> DECISION RULES (defined BEFORE results) -> OUTPUT CONTRACT.

Key distinctions:
- **ARITHMETIC_ONLY** = cost screen (payoff improves, WR flat). NOT a signal.
- **SIGNAL** = WR changes across bins. May be genuine edge.
- **TAUTOLOGY** = mathematically equivalent to existing filter (|r|>0.70 = KILL).

## Mode: prompts (guardian routing)

Read the relevant guardian or audit prompt for the task:
- **Entry model changes** -> `docs/prompts/ENTRY_MODEL_GUARDIAN.md`
- **Pipeline data changes** -> `docs/prompts/PIPELINE_DATA_GUARDIAN.md`
- **Feature specs** -> check `docs/specs/` first (follow exactly if exists)
- **Post-result claim / closure / readiness checks** -> `docs/prompts/POST_RESULT_SANITY_PASS.md`

"Significant change" = adding/removing entry models, schema changes, pipeline data flow, strategy lifecycle, drift check definitions.
