# MES EUROPE_FLOW ORB_G5 AND CROSS_SGP_MOMENTUM composite — C1-C12 audit

**Date:** 2026-04-19
**Pre-registration:** `docs/audit/hypotheses/2026-04-19-mes-europe-flow-g5-sgp-composite-v1.yaml`
**Origin:** Phase 2.4 third-pass adversarial reframe (commit 01ec8ecd) — proposed the MES composite as a "golden egg." This audit tests that claim against the locked 12 criteria. No deploy decision.
**Script:** `research/phase_2_4_mes_composite_c1_c12_audit.py`
**Output:** `research/output/phase_2_4_mes_composite_audit.csv`

## Executive verdict: **KILL — C3 FAIL, C4 FAIL**

The "+0.20 R/trade lift vs unfiltered" headline from Phase 2.4 pass-3 was an **arithmetic illusion.** Under C1-C12:

- Mode A IS composite: N=112, ExpR=+0.0459, **sd=1.1295, t=+0.43, p=0.668**
- Chordia t-statistic (threshold 3.00, grounded in Chan Ch 7 + Fitschen Ch 3 theory): **FAILS by a factor of 7**
- BH-FDR at K=1 requires p<0.05: **FAILS (p=0.668)**
- The per-trade "edge" is within one standard error of zero. On N=112, the standard error is 1.1295/√112 = 0.1067, making the 95% CI roughly [−0.16, +0.25] — straddles zero.

The visible "lift" came from comparing a filtered subset to an unfiltered MES EUROPE_FLOW baseline (ExpR −0.1573) that includes 738 days the composite does NOT trade — those aren't trades we'd take anyway, so the arithmetic delta is misleading.

## Criterion-by-criterion results

| # | Criterion | Threshold | Observed | Pass? |
|---|-----------|-----------|----------|:-----:|
| C1 | Pre-reg committed before audit | exists | yes | ✅ |
| C2 | MinBTL (K=1 trivially) | — | K=1 | ✅ |
| C3 | BH-FDR q<0.05 | p<0.05 (K=1) | **p=0.668** | ❌ |
| C4 | Chordia t-stat (w/theory) | ≥ 3.00 | **t=+0.43** | ❌ |
| C5 | Deflated Sharpe (informational) | > 0.95 | not computed (N_eff unknown) | ⚠ |
| C6 | Walk-forward efficiency | ≥ 0.50 | WFE=0.717 (4 folds) | ✅ |
| C7 | Deployable sample | N ≥ 100 | N=112 | ✅ |
| C8 | 2026 OOS consistency | sign match + positive | N=5, ExpR=+0.830, power_tier=DIRECTIONAL | ✅ (DIRECTIONAL, not CONFIRMATORY — per backtesting-methodology.md RULE 3.2, 5≤N<30 is directional-only evidence) |
| C9 | Era stability | no year ExpR<−0.05 w/ N≥50 | no qualifying era failures | ✅ |
| C10 | MICRO-only era | first trade ≥ 2019-05-06 | 2019-08-06 | ✅ |
| T0 | Tautology (vs g5) | \|rho\| ≤ 0.90 | 0.687 | ✅ |
| T0 | Tautology (vs sgp) | \|rho\| ≤ 0.90 | 0.303 | ✅ |
| T5 | Direction asymmetry | short ≤ long | short=−0.076 (N=100) < long=+0.046 | ✅ |
| T8 | Cross-instrument sign | match MNQ sign | MES=+0.046, MNQ=+0.093 | ✅ |

**Pass: 11. Fail: 2 (C3, C4). Verdict: KILL.**

C6/C7/C8/C9/T0/T5/T8 all passing DOES NOT override C3+C4 — the Chordia gate is the load-bearing significance check per `docs/institutional/literature/chordia_et_al_2018_two_million_strategies.md` and `docs/institutional/pre_registered_criteria.md § Criterion 4`. A claim that fails C3+C4 is not a claim.

## Year-by-year IS composite

| Year | N | ExpR |
|-----:|--:|-----:|
| 2019 | 1 | +1.2634 |
| 2020 | 26 | +0.0470 |
| 2021 | 10 | +0.5474 |
| 2022 | 41 | −0.0706 |
| 2023 | 3 | +0.4730 |
| 2024 | 9 | −0.2358 |
| 2025 | 22 | +0.0355 |

Extreme year-to-year variance (−0.24 to +1.26). Low N per year. Positive average is driven by a few high-R trades in 2019/2021/2023, offset by negative years. Consistent with "noise that happens to be mildly positive in aggregate" rather than a persistent signal.

No year has N ≥ 50, so C9 auto-passes by exclusion — but this is a LOW-POWER pass, not a strong-positive pass.

## Honest re-framing of the "golden egg" finding

In the Phase 2.4 third-pass reframe, I computed:

```
Unfiltered MES EUROPE_FLOW RR1.5 long:  ExpR = −0.1573, N=850
Composite ORB_G5 ∧ CROSS_SGP_MOMENTUM:  ExpR = +0.0459, N=112
Lift:                                    +0.2032 R/trade
```

I framed this as a "session-rescue" finding. That framing was BIASED:

1. **No significance test on the composite itself.** I computed only the ExpR, not the t-statistic on a null of zero. t=+0.43 means this ExpR is indistinguishable from zero noise.

2. **Comparison-to-unfiltered is the wrong gate.** The "lift vs unfiltered" metric compares two disjoint populations (composite fires on 112 days; unfiltered is 850 days; they're NOT the same universe). The relevant question is "does the composite have non-zero ExpR?" — not "does it have higher ExpR than trading every break-day?"

3. **Arithmetic vs signal.** ORB_G5 removes 643 losing trades from the unfiltered baseline (G5 OFF ExpR=−0.19, N=643). That's what drives the G5-alone lift of +0.11. The incremental SGP gate moves from G5-alone ExpR=−0.05 (N=207) to composite ExpR=+0.046 (N=112) — an increment that doesn't reach significance on N=112 (p=0.67).

4. **Institutional-rigor rule violated.** Per `.claude/rules/quant-audit-protocol.md § Anti-patterns`: *"'This looks promising' — prohibited until Step 5 clears."* The golden-egg framing used exactly this language. Corrected.

The honest statement is: on MES EUROPE_FLOW, ORB_G5 alone lifts from a negative baseline to near-zero (N=207, ExpR=−0.046 — still fails C3/C4), and adding CROSS_SGP_MOMENTUM on top doesn't significantly improve it further on N=112.

## Mechanism check (post-hoc)

The theory citations (Chan Ch 7 stop-cascade + Fitschen Ch 3 intraday trend) predict momentum on equity-index breakouts. The data rejects this on MES EUROPE_FLOW at RR1.5 long. Possible reasons:

1. **MES microstructure differs from Chan's FSTX.** Chan's p.156 result (APR 13%, Sharpe 1.4) was opening-gap strategy on Dow Jones STOXX 50 trading on Eurex, 2004-2012. MES is S&P-500 micro trading on CME, with different liquidity profile and different participant mix. The analogy doesn't mechanically transfer.
2. **EUROPE_FLOW is the WRONG session for this theory.** Chan's mechanism is "overnight gap → stop cascade at European open." MNQ EUROPE_FLOW starts 18:00 Brisbane (~08:00 London). That's ~4-5 hours after the actual London cash-market open for equity indices in their domestic session. The stop-cascade event may have already happened before our ORB window begins.
3. **SGP is not the right prior-session proxy for European equity indices.** Singapore's trading hours overlap mostly with Asian-specific drivers, not the European/US flow that moves S&P. TOKYO_OPEN or LONDON_METALS would be closer stakeholders.

The KILL result is consistent with the mechanism being wrong for this specific cell, not with "ORB-breakout momentum is false." Fitschen Ch 3 stands as literature support; what fails here is the *specific filter composition* on *this specific session+instrument*.

## What this audit does NOT claim

- Does NOT kill ORB-breakout intraday momentum in general
- Does NOT kill CROSS_SGP_MOMENTUM on all instruments (MNQ RR1.5 long still shows +0.021 lift, still fails C3/C4 standalone — audit per Phase 2.4 pass 2)
- Does NOT kill the composite filter concept — other (session, instrument, RR) combos not tested here
- Does NOT claim MES EUROPE_FLOW is dead for all strategies — only that this specific composite fails

## What this audit DOES claim

- The Phase 2.4 pass-3 "golden egg" language was biased/premature
- Under strict C1-C12, the MES EUROPE_FLOW composite at RR1.5 long is a KILL
- This is the correct institutional-rigor outcome: fail-closed on C3+C4, regardless of how attractive the unfiltered-lift number appeared
- The lesson belongs in the historical failure log:

> **2026-04-19: Lift-vs-unfiltered framing without significance test on the subset.** Phase 2.4 third-pass reframe flagged MES EUROPE_FLOW composite ORB_G5 ∧ CROSS_SGP_MOMENTUM as a "golden egg" based on +0.20 R/trade lift vs unfiltered baseline (ExpR=-0.16 N=850 → +0.046 N=112). Full C1-C12 audit: composite fails C3 (p=0.668) and C4 (t=+0.43). Lesson: a high "lift vs noise-baseline" delta is not evidence when the post-filter subset's ExpR is noise-indistinguishable from zero. Always compute Chordia t on the subset itself before reporting lift. File: this audit.

## Follow-up (optional, user's call)

If interest persists in the cross-session-momentum-on-MES space:

1. **Test LONDON_METALS → EUROPE_FLOW** or **TOKYO_OPEN → EUROPE_FLOW** as prior-session proxies (new pre-reg; Pathway A K=2 family).
2. **Test MES at different RR targets** (RR1.0, RR2.0, RR2.5) — not tested in this audit.
3. **Accept the result and move on.** The MES EUROPE_FLOW long lane appears genuinely weak for composite filters; resources better spent elsewhere (Task #4 Phase 0 branch hygiene, Task #8 MNQ regime-tailwind lock, Amendment v3.2 lock — all listed in Phase 2 task board).

## Audit trail

- Pre-reg committed before script ran (C1 PASS verified)
- Canonical delegations: `filter_signal`, `HOLDOUT_SACRED_FROM`, `GOLD_DB_PATH`
- Triple-join on `(trading_day, symbol, orb_minutes)`
- No `validated_setups` consulted (Mode-B contamination avoided)
- MinBTL trivially satisfied at K=1
