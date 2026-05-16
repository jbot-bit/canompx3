---
description: Adversarial verify of a single research finding. Canonical-grounded, fail-closed, inline.
argument-hint: <result-md-path-or-strategy-id>
allowed-tools: Read, Grep, Glob, Bash(python:*), Bash(rg:*), Bash(git show:*), mcp__gold-db__query_trading_db, mcp__gold-db__get_strategy_fitness
---

Operate as an institutional trading researcher verifying: $ARGUMENTS

Inline. Read result MD + paired pre-reg + runner script. Auto-loaded rules already injected — do NOT re-read. No subagent fan-out.

Truth source: canonical layers only (bars_1m, daily_features, orb_outcomes). Derived layers (validated_setups, docs, memory) orient but are NOT proof. UNSUPPORTED if not grounded. NOT READ if a resource wasn't actually opened. Cite file:line for every load-bearing claim.

1. CALCULATION + TEST VALIDITY — exact test, right question or proxy? Verify: sample/eligibility, cost/friction/R math, entry-model realism, no look-ahead/leakage, correct baseline, honest K/FDR. Classify VALID / CONDITIONAL / UNVERIFIED / WRONG.
2. FRAMING / TUNNEL-VISION — 3 alternative interpretations (role: standalone vs filter vs allocator vs confluence; layer: signal vs execution vs portfolio; mechanism). State fairly-tested vs ignored vs prematurely-killed.
3. EDGE LOCATION — local vs global; conditional vs standalone; interaction/portfolio; signal vs implementation loss. Name where edge lives or "nowhere".
4. BLOCKERS — what limits performance? Stale / duplicated / mis-scoped? Implementation drag?
5. LITERATURE / MECHANISM — aligns with local lit (`docs/institutional/literature/<file>.md`)? Plausible flow/vol/structure mechanism? WEAK / UNSUPPORTED if no source.
6. BRUTAL FILTER — looks good but isn't? Fragile / overfit / prop desk would reject?

OUTPUT (exactly these five lines, ≤300 words total):
- Verdict: VALID / CONDITIONAL / DEAD / UNVERIFIED
- Edge lives at: <where, or "nowhere">
- Biggest issue: <wrong framing / wrong test / blocker / implementation / etc.>
- Missed opportunity: <if any, else "none">
- Next best step: <highest-EV shortest-path action>

RULES: no assumptions without proof, no tunnel vision, no post-hoc rescue, no fluff. Uncertain → say uncertain. Dead → say dead.
