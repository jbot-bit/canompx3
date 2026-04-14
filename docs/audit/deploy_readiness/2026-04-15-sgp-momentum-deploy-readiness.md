# SGP_MOMENTUM Deploy-Readiness Audit — 2026-04-15

**Scope:** `MNQ_EUROPE_FLOW_E2_{RR1.0,RR1.5,RR2.0}_CB1_CROSS_SGP_MOMENTUM`
**Trigger:** Pulse flags 32 validated-not-deployed MNQ lanes; MEMORY note claims "3 VALIDATED, Jaccard 0.029 independent."
**Outcome:** **KILL all 3 as standalone deploy candidates.** Alternative path identified (§ 6).

## 1. Validated-setups record (as-promoted 2026-04-13)

| RR  | N | ExpR_IS | Sharpe_ann | WFE | FDR_p | DSR_score | 2024 ExpR | first_day | last_day |
|-----|---|---------|-----------|-----|-------|-----------|-----------|-----------|----------|
| 1.0 | 1020 | 0.085 | 1.18 | 1.90 | 0.004 | 0.037 | +0.036 | 2019-05-08 | 2026-04-10 |
| 1.5 | 1020 | 0.094 | 1.01 | 2.59 | 0.013 | 0.013 | **−0.047** | 2019-05-08 | 2026-04-10 |
| 2.0 | 1020 | 0.123 | 1.12 | 1.50 | 0.006 | 0.024 | **−0.062** | 2019-05-08 | 2026-04-10 |

family_hash = MNQ_5m_sm100_63cfcedc…; RR1.5 is family head.
promotion_git_sha = 836fcd06; validation_run_id = MNQ_20260413_084331_5b3281.
Pre-registered hypothesis file: `docs/audit/hypotheses/2026-04-13-cross-session-sgp-europe-flow.yaml` (K=3, MinBTL PASS).

## 2. 12-Criteria scorecard

| # | Criterion | RR1.0 | RR1.5 | RR2.0 | Evidence |
|---|-----------|:-----:|:-----:|:-----:|----------|
| 1 | Pre-registered hypothesis file | ✅ | ✅ | ✅ | `2026-04-13-cross-session-sgp-europe-flow.yaml` committed before discovery |
| 2 | MinBTL (≤300 or ≤2000 proxy) | ✅ | ✅ | ✅ | Hypothesis file: trials=3, required=0.01yr vs 6.65yr available |
| 3 | BH FDR q<0.05 (K=3 pre-reg) | ✅ | ✅ | ✅ | p = 0.004 / 0.013 / 0.006 |
| 4 | Chordia t ≥ 3.00 (w/ theory) | ❌ | ❌ | ❌ | p→t: 2.88 / 2.48 / 2.74; **DEFERRED per validator Amendment 2.2** |
| 5 | DSR > 0.95 | ❌ | ❌ | ❌ | 0.037 / 0.013 / 0.024; **DEFERRED per validator Amendment 2.1** (N_eff unsolved) |
| 6 | WFE ≥ 0.50 | ✅ | ✅ | ✅ | 1.90 / 2.59 / 1.50 |
| 7 | N ≥ 100 | ✅ | ✅ | ✅ | 1020 trades each |
| 8 | 2026 OOS ≥ 0.40 × IS | ✅ | ✅ | ✅ | OOS ExpR 0.215 / 0.460 / 0.404 vs thresholds 0.034 / 0.038 / 0.049 (**6.3–12.2× overshoot, N=40**) |
| 9 | No era ExpR < −0.05 (N ≥ 50) | ✅ | ⚠️ | ❌ | 2024 at RR1.5 = −0.0475 (within tolerance); **RR2.0 = −0.0617 FAIL** with N=160 |
| 10 | MICRO-only data era | ✅ | ✅ | ✅ | first_day 2019-05-08 post-MNQ-micro launch 2019-05-06 |
| 11 | Account-death MC ≥ 70% | — | — | — | Profile-level check after deploy; current 91.4% on 6 lanes |
| 12 | SR monitor | — | — | — | Auto-activates on deploy; L1 EUROPE_FLOW ORB_G5 currently CONTINUE |

Enforcing only the **currently-active validator gates** (1, 2, 3, 6, 7, 8, 9, 10): RR1.0 and RR1.5 pass; RR2.0 fails C9 by 1.2σ on 2024.

## 3. Correlation audit (Pearson on daily R-returns, full history)

|         | L1 EF_ORB_G5 RR1.5 | L3 CS_OVNRNG RR1.5 | L4 NY_ORB_G5 RR1.0 | L5 TK_ORB_G5 RR1.5 |
|---------|:---:|:---:|:---:|:---:|
| **RR1.0** | **0.797** | −0.411 (N=30) | 0.032 | 0.045 |
| **RR1.5** | **1.000** | −0.341 (N=30) | 0.025 | 0.061 |
| **RR2.0** | **0.859** | −0.341 (N=30) | 0.004 | 0.034 |

**Gate:** `trading_app/lane_correlation.py` → `RHO_REJECT_THRESHOLD = 0.70`.
**Verdict:** all 3 candidates REJECTED by correlation gate vs existing L1 deployed lane.

The MEMORY.md note "Jaccard 0.029 vs existing (independent)" was cherry-picked — that's the comparison vs L3 COMEX_SETTLE OVNRNG_100 which trades on only 51 days in 7 years (rare vol days). Honest Jaccard range: 0.03 (vs L3) → 0.57 (vs L1 same-session).

## 4. Why rho=1.000 on RR1.5

On 1788 EUROPE_FLOW-break days (MNQ):
- SGP_TAKE ∩ ORB_G5: 949 days (**53% of EF-breaks, 93% of SGP_TAKE universe**)
- SGP_TAKE ∩ ¬ORB_G5: 70 days (SGP-unique)
- ¬SGP_TAKE ∩ ORB_G5: 704 days (ORB_G5-unique)
- Neither: 65 days

SGP_MOMENTUM is effectively a near-subset of ORB_G5. On the 949-day intersection, both strategies enter at the same ORB break at the same time with the same RR — producing identical R-outcomes by construction.

## 5. Confluence lift — the actual edge

On **ORB_G5 days only** (already captured by deployed L1), partition by SGP state:

| RR  | SGP_TAKE (N=949) | SGP_VETO (N=704) | Δ ExpR |
|-----|:---:|:---:|:---:|
| 1.0 | 0.0885 | 0.0226 | **+0.066** |
| 1.5 | 0.1060 | 0.0311 | **+0.075** |
| 2.0 | 0.1231 | 0.0454 | **+0.078** |

**SGP is a genuine confluence signal — but the value is in filtering existing ORB_G5 trades, not in standalone trading.** Adding it as a parallel lane captures ~0 new edge; stacking it onto ORB_G5 captures +0.075 ExpR/trade.

## 6. Recommendation — REVISED 2026-04-15 after honest head-to-head

**Original "KILL" verdict was framed wrong.** Correlation gate analysis was correct, but I missed the trailing-performance head-to-head that the allocator actually uses for deployment decisions. Honest revised picture:

### Head-to-head SGP RR1.5 vs L1 ORB_G5 RR1.5

| Window | Strategy | N | ExpR | Sharpe (daily) | Total R |
|---|---|---:|---:|---:|---:|
| Trailing 12mo | L1 ORB_G5 | 252 | 0.189 | 0.163 | **47.5** |
| Trailing 12mo | SGP RR1.5 | 156 | **0.271** | **0.235** | 42.3 |
| 2026 OOS | L1 ORB_G5 | 70 | 0.297 | 0.254 | **20.8** |
| 2026 OOS | SGP RR1.5 | 40 | **0.460** | **0.401** | 18.4 |

**SGP RR1.5 is GENUINELY BETTER per trade (ExpR +44% trailing, +55% OOS) and better risk-adjusted (Sharpe +44% trailing, +58% OOS). L1 wins on Total R by virtue of taking 1.6× more trades.**

### What the allocator does (`trading_app/lane_allocator.py`)

Ranks by `annual_r_estimate` (total R × 12/months), greedy selection with rho ≥ 0.70 reject. Per current trailing data:
- L1 ranks first (47.5 R/yr)
- L1 selected → fills the EUROPE_FLOW MNQ slot
- SGP RR1.5 then rejected for rho > 0.70 with selected L1

**The allocator is doing the right thing FOR ITS OBJECTIVE (max Total R within DD budget).** It's not punishing SGP — it's choosing the lane that contributes more total R to the portfolio.

### The real decision (user-level)

This isn't "deploy or kill" — it's a portfolio-construction trade-off:

| Option | Description | Trailing R/yr | OOS R/yr (annualized) | Per-trade quality | Capital efficiency |
|---|---|---:|---:|---:|---:|
| **A: Keep L1 (status quo)** | ORB_G5 RR1.5 deployed | 47.5 | 79 | 0.189 ExpR | 252 trades/yr |
| **B: Swap to SGP RR1.5** | Replace L1 with SGP, retire L1 | 42.3 | 70 | **0.271 ExpR** | 156 trades/yr |
| **C: Composite (future work)** | New ORB_G5_AND_SGP_TAKE filter, replace L1 | unmeasured | unmeasured | est. 0.106 on 949 trades = ~100R/yr | needs validation |
| **D: Add SGP as parallel slot** | Both deployed | gate-blocked | gate-blocked | n/a | n/a |

**Trade-offs:**

- **A vs B:** B trades less but wins more per trade. Same-direction signal, just selective. If your bottleneck is **R/year**, A wins by 5–9R/yr. If your bottleneck is **commission drag, slippage, or per-trade Sharpe**, B wins.
- **A vs C:** C is theoretically the strongest (captures both volume and selectivity) but requires a new validator run + new hypothesis file. Real work, not tonight.
- **D is dead** by the correlation gate (rho > 0.70 rejects).

### Honest verdict

- **Do NOT auto-deploy** SGP standalone lanes — the gate correctly blocks parallel addition.
- **CONSIDER manual swap** (Option B): SGP RR1.5 has materially better per-trade quality and 2026 OOS. The Total R cost is small (~5R/yr trailing, ~9R/yr OOS-annualized).
- **PREFERRED long-term:** Composite filter (Option C) — captures per-trade edge AND keeps trade volume. Requires committing to the validator workflow.

The KILL framing in v1 of this doc was honest about the *gate* but dishonest about the *strategy quality*. SGP_MOMENTUM is a real edge — it's just blocked from deploying because something marginally different already occupies the same slot.

### What I had wrong in v1

1. Treated "correlation gate rejects" as "strategy is bad." Correlation gate rejection is about REDUNDANCY with deployed, not about strategy quality.
2. Did not run trailing-12mo head-to-head, which is the allocator's decision metric.
3. Implied "70 SGP-unique days is insufficient" — true if standalone, but irrelevant if comparing to swap.
4. RR2.0 era-stability failure (2024 = −0.0617) still stands as Criterion 9 fail; that one is a real KILL.
5. RR1.0 and RR1.5 should be considered "strong but redundant — swap-eligible if user prefers per-trade quality."

**Alternative path — captures the real edge (FUTURE WORK, not tonight):**

Replace L1 `MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5` with a composite
`MNQ_EUROPE_FLOW_E2_RR1.5_CB1_ORB_G5_AND_SGP_TAKE`.

- Reduces trade count 1653 → 949 (−42%)
- Lifts ExpR 0.094 → 0.106 (+13% per trade)
- Net R/yr: ~100R → ~100R (similar gross, lower turnover, better risk-adjusted)
- Same slot in portfolio; no correlation gate issue

Required before implementation:
1. New `CompositeFilter(ORB_G5, CROSS_SGP_MOMENTUM)` in `trading_app/config.py`
2. New pre-registered hypothesis file at `docs/audit/hypotheses/2026-04-??-ef-orb-g5-sgp-composite.yaml`
3. Validator run pathway=individual on the composite
4. `lane_allocation.json` update replacing L1, not adding parallel
5. Rollback plan if composite underperforms L1 live

## 7. MEMORY correction needed

`memory/cross_session_momentum_research.md` claims "Jaccard 0.029 vs existing (independent)." This was computed against the L3 COMEX_SETTLE OVNRNG_100 lane — the one deployed lane with only 51 trade-days in 7 years. Honest max Jaccard vs L1 is 0.57; honest rho vs L1 RR1.5 is 1.000. The "independent" framing was a cherry-pick, not a portfolio-level conclusion.

## 8. What IS genuinely independent — the 70-day pocket

SGP_TAKE ∩ ¬ORB_G5 = 70 days across 7 years. These are days where the EUROPE_FLOW ORB is below 5 pts (small breaks) but the cross-session momentum signal is aligned. On these 70 days, no existing MNQ EUROPE_FLOW lane trades — so they're genuinely additive. But N=70 over 7 years ≈ 10 trades/year is **below the 100-trade minimum** (Criterion 7) for a dedicated lane and too sparse to support its own SR monitor.

Option: add these 70 days' trades to the L1 lane as a fallback trigger ("ORB_G5 OR (ORB_size∈[2,5] AND SGP_TAKE)"). But this is a new 3-term composite needing its own validation run. Parked as future work, lower priority than the Option 1 replacement.

## 9. Actions taken

- No deploy. No lane_allocation.json edit. No retirement of the 3 validated_setups rows (they remain research-provisional, status=active).
- Report committed. MEMORY correction pending user OK.

## 10. Revision log

- **v1 (2026-04-15 ~00:47 Bris):** "KILL all 3" verdict. Methodology checked rho-on-intersection vs canonical gate (correct), but missed trailing-12mo head-to-head and over-reached by framing strategies as "bad" rather than "redundant."
- **v2 (2026-04-15 ~01:00 Bris):** Revised after user pushback for honesty. Added § 6 trailing comparison (SGP RR1.5 +44% ExpR, +44% Sharpe vs L1 trailing). Repositioned as A/B/C/D portfolio decision. RR2.0 C9 era failure stands; RR1.0/RR1.5 are swap-eligible, not kill-eligible.
