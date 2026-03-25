# Edge Maximization Discovery — Self-Orienting Prompt

Drop this into a fresh session. It discovers before it assumes.

---

## PROMPT START

I have a futures ORB breakout research pipeline with validated strategies across multiple instruments and sessions. A recent edge improvement investigation produced ranked findings (see below). I want to extract maximum value from these findings honestly — no confirmation bias, no cherry-picking, no premature implementation.

**Your job is adversarial researcher first, implementer last.**

### PHASE 0 — ORIENT (mandatory, no shortcuts)

Before touching ANY finding:

1. **Read governing docs in this order:**
   - `CLAUDE.md` (architecture, guardrails, behavioral rules)
   - `TRADING_RULES.md` (sessions, filters, entry models, cost models)
   - `RESEARCH_RULES.md` (statistical standards, FDR, discovery discipline)
   - `docs/STRATEGY_BLUEPRINT.md` (test sequence, NO-GO registry, variable space)
   - `.claude/rules/quant-audit-protocol.md` (the full audit battery)

2. **Query current state — don't trust docs:**
   - What's actually in `gold.db` right now? (table counts, date ranges, freshness)
   - What strategies are actually validated? (query, don't read memory files)
   - What's the current Apex roster? (query `live_config` or equivalent)
   - What sessions have been fully discovered vs. partially explored?

3. **Build a session profile for every session you'll touch.** Don't assume sessions are interchangeable. For each session in the Apex roster + any session under investigation, query and report:
   - Session time in Brisbane TZ and the exchange it maps to (from `SESSION_CATALOG`)
   - Market structure context: What's happening at that time? (e.g., NYSE_OPEN = US cash open + highest liquidity of the day. SINGAPORE_OPEN = overnight/Asian session, thin liquidity, different participant mix. CME_PRECLOSE = last 30 min before CME settlement, mechanical flows.)
   - Unfiltered baseline ExpR and N (per instrument — don't pool MNQ/MGC/MES)
   - Which filters are currently applied (from validated strategies or Apex config)
   - Which filters have been TESTED but failed for this session (check discovery grid results)
   - Known behavioral quirks: Does break delay help or hurt? Does volatility filtering work? Does it have overnight gap exposure?

   **Why:** The Crabel finding proved filters reverse direction across sessions. A break delay filter that helps NYSE_OPEN HURTS SINGAPORE_OPEN. If you don't build session profiles first, you'll apply findings from one session's market structure to another where they don't hold. This has already caused errors in this project.

4. **Identify what I DON'T know yet:**
   - Has anything changed since the research report was generated?
   - Are there pipeline rebuilds, new data, or config changes that invalidate findings?
   - What assumptions in the report have I not verified against live DB state?
   - Which sessions have I NOT profiled that might matter? (Check for sessions with validated strategies that aren't in Apex — they may be the real opportunity.)

5. **Report your orientation before proceeding.** Tell me:
   - What's fresh, what's stale
   - What the report claims vs. what the DB shows
   - Any contradictions or gaps
   - Your honest assessment of which findings have the highest REAL expected value (not just the report's ranking)

### PHASE 1 — CHALLENGE THE RANKINGS

The research report ranked 6 findings. Your job is to stress-test that ranking:

- **For each PURSUE/MONITOR item:** What's the actual expected portfolio impact in R-terms? Not vibes — math. If CME_PRECLOSE adds a lane at +0.12 unfiltered ExpR, what does that mean after filters, FDR, costs, and correlation with existing lanes? Could be +50R/year or +5R/year — find out which.

- **For each DROP item:** Am I dropping it for the right reason? "Not significant" after Bonferroni with K=20 is different from "genuinely no effect." Is the test underpowered? Would a different test design (paired, within-session) find something?

- **For each NEEDS_DATA item:** What's the expected information value of collecting that data? Is it worth 4-6 hours if the prior is weak? Or is the prior actually strong and the data collection is high-EV?

- **What's NOT on this list that should be?** The report investigated 6 questions. Are there obvious ones it missed? Check the Blueprint variable space — what dimensions haven't been explored?

- **SESSION-LEVEL DECOMPOSITION (mandatory for every finding).** Do not evaluate ANY finding at the portfolio level without first breaking it down per-session. Specifically:
  - For new session candidates (CME_PRECLOSE, US_DATA_1000): Profile the session's market structure. Why would edge exist here? What's the mechanism? Who's trading at that time and why would they create predictable order flow? If you can't articulate a plausible mechanism, flag it.
  - For filter hypotheses (break delay, gap, momentum): Run the analysis SEPARATELY for each session. Present a session × finding matrix showing which sessions the filter helps, hurts, or is neutral for. Do NOT average across sessions — that hides reversals.
  - For existing lanes being modified: What's the session-specific reason the current config was chosen? Would the proposed change break the logic that made the original config work?
  - **Template for per-session analysis:**
    ```
    SESSION: [name]
    TIME: [Brisbane TZ] → [exchange context]
    MECHANISM: [why would edge exist here — liquidity, participant mix, mechanical flows]
    FINDING APPLICABILITY: [does this specific finding's logic apply to this session's structure?]
    RESULT: [numbers — ExpR, WR, N, p-value — for THIS session only]
    VERDICT: [APPLIES / DOES_NOT_APPLY / REVERSES / UNDERPOWERED]
    ```

### PHASE 2 — HONEST PRIORITIZATION

After challenging, produce a revised priority list with:

| Rank | Item | Expected R impact (range) | Confidence | Hours to test | Hours to implement | Dependencies | Kill criteria |
|------|------|--------------------------|------------|---------------|-------------------|--------------|---------------|

Rules:
- "Expected R impact" must be a RANGE, not a point estimate
- "Confidence" = how sure are you the impact estimate is right (not how sure the edge exists)
- Include kill criteria BEFORE starting work (what would make you stop)
- If two items are close, say so — don't force-rank noise
- If the honest answer is "the current portfolio is already near-optimal and marginal improvements are small" — SAY THAT. Don't manufacture urgency.

### PHASE 3 — EXECUTION (only after Phase 2 approval)

For whatever I greenlight from Phase 2:

1. **Design the test BEFORE running it — PER SESSION.** For each session the finding applies to, define separately:
   - Which session, and WHY this finding should theoretically apply here (mechanism)
   - Null hypothesis (what you're trying to reject FOR THIS SESSION)
   - Sample size requirement (power analysis — N per session, not pooled)
   - Success criteria (what number makes this worth implementing FOR THIS SESSION)
   - Kill criteria (what number kills it FOR THIS SESSION)
   - Pre-register the analysis plan — no post-hoc adjustments
   - If a finding applies to multiple sessions, you need MULTIPLE test designs. A break delay filter for NYSE_OPEN is a DIFFERENT test than a break delay filter for COMEX_SETTLE, even if the variable name is the same.

2. **Run the full audit battery PER SESSION** from `quant-audit-protocol.md` — T0 through T8, in order, no skipping. T5 (family comparison) becomes: "does this finding hold across instruments FOR THIS SPECIFIC SESSION" — not "does it hold across sessions" (that's already handled by running separate audits per session).

3. **Report results in the structured output format** from the audit protocol. No prose conclusions. Numbers first, interpretation second.

4. **Implementation proposal ONLY if tests pass.** Follow the Design Proposal Gate from CLAUDE.md — what, files, blast radius, approach. Wait for my go.

### BEHAVIORAL CONSTRAINTS

- **No "promising" or "interesting" until T5 clears.** These words are banned pre-validation.
- **No sunk cost.** If you spend 2 hours on CME_PRECLOSE and it's dying at T3, kill it. Don't try to save it.
- **No scope creep.** One finding at a time. Finish → verify → next. Don't start RQ2 while RQ6 is half-done.
- **Honest uncertainty.** If you can't distinguish signal from noise at the available sample size, say "underpowered, need N=X to resolve" — don't squint at p=0.08 and call it "marginal."
- **Challenge my priors.** If I say "CME_PRECLOSE is obviously the best next step" and your analysis says otherwise, push back. I want truth, not agreement.
- **Session-specificity is structural, not cosmetic.** Each session represents a different market microstructure — different participants, different liquidity, different motivations for order flow. A filter that works at NYSE_OPEN (high liquidity, directional conviction from cash open) may REVERSE at SINGAPORE_OPEN (thin overnight liquidity, false breaks into air). This is not an edge case — it's the central finding of this project. Rules:
  - NEVER pool results across sessions to inflate N or smooth over reversals.
  - NEVER apply a finding from one session to another without testing it there first.
  - ALWAYS present the session × finding matrix before any aggregate conclusion.
  - If a finding works for 2 sessions and reverses for 2, that's NOT "works for 50% of sessions" — it's TWO DIFFERENT FINDINGS that need separate implementation paths.
  - When proposing a new filter, state upfront which sessions it should theoretically help (with mechanism) and which it shouldn't. Then test. If reality doesn't match theory, theory is wrong.

### REFERENCE FINDINGS (from prior investigation)

```
#1  CME_PRECLOSE discovery (RQ6)     — PURSUE   — strongest unfiltered non-Apex session
#2  Break delay filter (RQ2)         — MONITOR  — Crabel decay confirmed for 2/4 sessions, REVERSES for 2/4
#3  News calendar integration (RQ4)  — NEEDS_DATA — NFP appears POSITIVE (contradicts hypothesis)
#4  Momentum confirmation (RQ5)      — NEEDS_DATA — deferred pending RQ2
#5  ORB offset optimization (RQ1)    — DROP     — no significant differences after correction
#6  Gap filter (RQ3)                 — DROP     — real but immaterial (1.4% of days)

Key insight #1: Break delay decay is session-specific. Immediate breaks best for NYSE_OPEN/COMEX_SETTLE,
delayed breaks best for SINGAPORE_OPEN/NYSE_CLOSE. Universal filters don't work here.

Key insight #2: US_DATA_1000 was NOT investigated in the original report but has 4 validated strategies,
stable 2026 forward performance (+0.065 ExpR), and is not in Apex. This is a gap — investigate it.

Key insight #3: CME_PRECLOSE failed BH FDR at K=55 (p=0.007 vs threshold 0.003). It's on single-test
pre-registration probation, not validated. Zero MNQ strategies have survived the full pipeline.

Phase 0 finding: Sessions are not interchangeable. Each has different market structure, participant mix,
and filter response. The prompt enforces per-session decomposition at every phase.

ML blocked: 3 open FAILs (sample size, negative baselines, selection bias). Don't revisit until resolved.
```

### START

Begin with Phase 0. Orient fully. Then tell me what you found before we do anything else.

## PROMPT END
