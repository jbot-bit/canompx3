# Adversarial Review Findings — Two Independent Zero-Context Reviews

**Date:** 2026-03-18
**Reviewers:** Two fresh Claude agents with NO project memory — one as adversarial quant reviewer, one as bias/methodology researcher
**Input:** RESEARCH_RULES.md + methodology summary only
**Status:** FINDINGS VERIFIED — 3 of 8 confirmed against code

---

## Convergent Findings (Both Reviews Agree)

### 1. FDR is cosmetic, not a gate — CONFIRMED IN CODE
`strategy_validator.py:1144` — strategies that fail BH FDR get tagged `fdr_significant=False` but are NOT rejected. They proceed to `validated_setups`. The claim of "FDR-corrected" is misleading.

### 2. Headline count is inflated — CONFIRMED IN CODE
- Validated strategies (active): **2,372**
- Edge families (non-purged): **253**
- Ratio: 9.4 correlated near-duplicates per family
- The honest number is 253, not 2,372.

### 3. DSR and FST are informational only — CONFIRMED IN CODE
`strategy_validator.py:499` — "logged, not rejected"
`strategy_validator.py:515` — "logged, not rejected"
Both gates exist in code but are explicitly bypassed.

### 4. No synthetic null pipeline test (GAP 5)
**Both reviews rank this as the #1 missing piece.** Generate 5+ years of synthetic bars with ZERO trend/signal. Run the full pipeline. If ANY strategies "validate," the pipeline manufactures false edges. Every institutional quant shop does this. We haven't.

### 5. Gold walk-forward validates only high-vol regime
Gold ATR: 11.5 (2018) → 105.3 (2026) = 9.2x variation. All WF windows fall in the high-vol era. Zero evidence of edge at ATR < 30 (which was normal for most of gold's history).

### 6. AI echo chamber is structural
41 memory files + CLAUDE.md + TRADING_RULES.md = Claude enforces YOUR beliefs, not independent reality. The "mechanism test" lists YOUR pre-approved mechanisms. Claude validates findings against YOUR framework. This is a mirror, not a reviewer.

### 7. Cost model is structurally optimistic
Fixed slippage assumption. Real slippage correlates with ORB size (bigger breakouts = more competition = more slippage). The 1.5x stress test should be 2.0-2.5x for micro futures at off-peak sessions.

### 8. "Mechanism" is a tautology
"Friction eats small ORBs" describes what a size filter does, not why breakout edge exists. A real mechanism would explain why price continues after a breakout.

---

## Priority Actions (Both Reviews Converge)

| Priority | Action | Cost | Time | What It Proves |
|----------|--------|------|------|----------------|
| **1** | **Synthetic null pipeline test** | Free | 4-6h | Whether the pipeline manufactures false edges |
| **2** | **Make FDR a hard gate** | Free | 1h | How many strategies survive honest correction |
| **3** | **Fresh-AI adversarial review** (different model, zero context) | Free | 1h | Cross-model consistency of findings |
| **4** | **External human quant review** | $200-500 | 1 week | Genuinely independent assessment |
| **5** | **20-trade manual spot-check** | Free | 2h | Whether fills/costs are computed correctly |
| **6** | **Paper trading with tracking** | Free | 3 months | Reality vs backtest |

---

## The Meta-Rule

**No AI that has access to this project's memory, rules, or history should be the final arbiter of whether this project works.**

The validator must be independent of the thing being validated. This means:
- Different model (GPT-4, Gemini, Grok) for cross-model review
- Clean session (no CLAUDE.md, no memory) for same-model review
- Human reviewer for genuine independence
- Synthetic null data for pipeline-level ground truth
- Live paper trading for reality-level ground truth

---

## What The Project Does Right (Acknowledged by Both Reviewers)

- NO-GO list is longer than success list — intellectual honesty
- E0 purged when bias found (33/33 artifact)
- Calendar effects killed despite prior belief
- BH FDR, walk-forward, sensitivity analysis all implemented
- Cost model exists and is applied
- RESEARCH_RULES.md is genuinely rigorous on paper
- The user asking "am I fooling myself?" is itself evidence against the worst case
