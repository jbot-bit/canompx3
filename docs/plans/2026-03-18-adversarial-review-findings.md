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

## Priority Actions — Status (updated 2026-03-18)

| Priority | Action | Status | Result |
|----------|--------|--------|--------|
| **1** | **Synthetic null pipeline test** | **DONE** | **FAIL: 60-63 false positives from noise survive pipeline.** 20-33 survive even BH FDR. E2 near-breakeven on random walks (-0.004R) enables lucky streaks. Confirmed across 2 seeds (42, 99). |
| **2** | **Make FDR a hard gate** | **DONE** | Global K (120,376 tests), BH at α=0.05. Cuts ~1,500 strategies. Now rejects, not just tags. |
| **3** | **DSR/FST fake gates removed** | **DONE** | Were "logged, not rejected" — dead code pretending to be safety nets. FST passed 13/116,900 (broken hurdle). Removed. |
| **4** | **MGC regime documented honestly** | **DONE** | All WF windows 2025-2026 only. "RESOLVED" claim in config.py was false — corrected. |
| **5** | **Cost model limitations documented** | **DONE** | Flat slippage is structurally optimistic for ORB entries. No fix without live fills. |
| **6** | **Mechanism test reclassified** | **DONE** | Now called "artifact screen" not "mechanism proof." Honest about what it does/doesn't do. |
| **7** | **Null envelope (10 seeds)** | **DONE** | 10 seeds, 611 survivors. E1 floor=0.25, E2 floor=0.32. `scripts/tools/null_envelope.py`. |
| **8** | **E2-specific ExpR floor** | **DONE** | Phase 2b hard gate in validator. E1=0.25, E2=0.32. Drift check enforces. |
| **9** | **DSR analytical cross-check** | **DONE** | `trading_app/dsr.py`. Informational (not hard gate — N_eff unknown). DSR kills everything at N_eff>=10. Agrees with null test: most is noise. |
| **10** | **M2K dropped** | **DONE** | 0/18 families survive any noise threshold. Added to DEAD_ORB_INSTRUMENTS. |
| **11** | **ONC N_eff estimation** | TODO | Required to make DSR a hard gate. Lopez de Prado clustering algorithm. |
| **12** | **Block bootstrap null** | TODO | Politis-Romano on actual returns. Realistic null (fat tails, vol clustering). |
| **13** | **Cross-model AI review** | TODO | GPT-4/Gemini zero-context adversarial methodology review. |
| **14** | **Paper trading with tracking** | TODO | 3 months live. Kill criteria: slippage > 2x modeled, P&L < 50% backtest. |

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
