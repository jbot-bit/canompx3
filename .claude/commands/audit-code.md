---
description: Adversarial code + quant audit reviewer. Read-only — no Edit/Write tools. Inline, ≤400-word severity-tiered report.
argument-hint: <path-or-PR-or-symbol>
allowed-tools: Read, Grep, Glob, Bash(rg:*), Bash(git show:*), Bash(git log:*), Bash(git diff:*), Bash(python:*)
---

Operate as an institutional code + quant audit reviewer on: $ARGUMENTS

Actually read relevant project resources: CLAUDE.md, RESEARCH_RULES.md, TRADING_RULES.md, HANDOFF.md, relevant docs/specs, local literature extracts. Quote file:line for every load-bearing claim. NOT READ if you didn't open it. UNSUPPORTED if not grounded.

AUDIT ORDER
1. SOURCE-OF-TRUTH MAP — which layer owns this logic? Canonical vs derived? Silent downstream redefinition?
2. CODE PATH — trace inputs → transformations → joins → filters → assumptions → outputs → downstream consumers.
3. LOGIC + BIAS — look-ahead / leakage / stale config / wrong join keys / bad denominators / sample shrinkage / execution optimism / holdout contamination / FDR or K misuse / timezone or session drift / silent fallbacks / swallowed errors / hardcoded assumptions.
4. RESOURCE GAP — docs that should have been consulted but weren't; behavior contradicting canon; missing provenance annotations; rule-less assumptions.
5. TEST + GUARDRAIL — real failure-mode coverage? Drift check present? Negative tests? Fail-closed behavior?
6. BLAST RADIUS — data integrity / discovery / validation / portfolio / live execution / monitoring / docs.

RETURN BUDGET: ≤400 words. Format:
- Verdict: PASS / PASS_WITH_RISKS / FAIL / BLOCKED
- Load-bearing findings (severity-tiered Critical/High/Medium/Low): each — finding, file:line, evidence source, why it matters, fix.
- Gaps / silences: missing / undocumented / unsupported.
- Tests to add: exact tests / drift checks.
- Next action: fix now / audit upstream / park / safe to proceed.

RULES: adversarial, no skimmed sources, no assumptions, no broad refactors, no code changes (read-only — Edit/Write not allowed; if you need to fix, surface in "Next action", do not patch).
