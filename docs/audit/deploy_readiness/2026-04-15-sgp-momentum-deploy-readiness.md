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

## 6. Recommendation

**KILL the 3 SGP_MOMENTUM lanes as standalone deployment candidates.**

Evidence:
1. Correlation gate (0.70 threshold) rejects all 3 vs existing L1.
2. RR1.5 rho = 1.000 — literally the same strategy on overlap days.
3. RR2.0 has C9 era stability failure (2024 = −0.0617).
4. Adding them alongside L1 = concentration, not diversification.
5. 70 SGP-unique days (10 per year) is insufficient to carry a standalone lane.

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
