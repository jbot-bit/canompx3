# Second-pass multi-angle audit — 2026-04-19 post-push review

**Generated:** 2026-04-19
**Trigger:** user push-back "LOOK FROM OTHER ANGLES ENSURE WE GOT IT ALL. NO BIAS INST. RESOURCES" after the correction cycle was pushed at `0e0fe27d`.
**Purpose:** final integrity sweep using institutional-literature-grounded angles the pre-push review may have missed.

---

## Finding 1 (MATERIAL) — EUROPE_FLOW long is a SESSION-WIDE decay, not lane-specific

Until now the retirement-queue narrative treated the 8-of-13 EUROPE_FLOW concentration as "lanes that decayed independently, happen to share a session." A fresh session-wide unfiltered query on MNQ O5 E2 (all RRs pooled) shows a stronger pattern:

| Session | Direction | Early 2022-23 ExpR | Late 2024-25 ExpR | Shift |
|---|---|---:|---:|---|
| **EUROPE_FLOW** | **long** | **+0.115** | **+0.006** | **−0.109 DECAY** |
| EUROPE_FLOW | short | +0.100 | +0.159 | +0.059 improved |
| NYSE_OPEN | long | +0.098 | +0.132 | +0.034 improved |
| NYSE_OPEN | short | +0.056 | +0.202 | +0.146 **strongly improved** |
| COMEX_SETTLE | long | +0.163 | +0.169 | stable |
| COMEX_SETTLE | short | +0.009 | +0.148 | +0.139 strongly improved |
| TOKYO_OPEN | long | +0.030 | +0.075 | improved |
| TOKYO_OPEN | short | +0.070 | −0.012 | −0.082 dropped |
| SINGAPORE_OPEN | long | −0.013 | +0.090 | improved |
| SINGAPORE_OPEN | short | +0.010 | −0.120 | −0.130 dropped |

**Interpretation:** EUROPE_FLOW long edge has lost ~95% of its unfiltered ExpR — a session-wide phenomenon, not a property of any specific filter. NYSE_OPEN and COMEX_SETTLE shorts are strongly improving (a REGIME SHIFT observation). TOKYO_OPEN and SINGAPORE_OPEN shorts are deteriorating.

**Direct implication for retirement queue:**
- The 4 Tier 1 retirement candidates are ALL `EUROPE_FLOW` or `US_DATA_1000` lanes, predominantly long direction. Consistent with the session-wide-decay interpretation.
- The retirement verdict for EUROPE_FLOW lanes is STRONGER than the per-lane framing suggested: the entire session-direction is fading, so lane-by-lane retrial with different filters is unlikely to produce a replacement.
- Any new MNQ discovery pre-reg should DE-PRIORITIZE EUROPE_FLOW long and prioritize NYSE_OPEN short, COMEX_SETTLE short, SINGAPORE_OPEN long.

**Action:** retirement-queue doc does not need to change (the 4 Tier 1 lanes remain the right vote), but the RATIONALE strengthens from "lane-specific decay" to "session-wide deterioration + ride the regime-shift winners." Captured as a mechanism finding below.

---

## Finding 2 (CORRECTION) — MES has 2 active validated lanes, not 1

Correction-cycle narrative (handover doc, synthesis doc, MES K=40 commit message) repeatedly said "1 validated MES lane." Fresh DB query shows TWO:

| Strategy ID | Filter | ExpR | Sharpe | N |
|---|---|---:|---:|---:|
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_COST_LT08` | COST_LT08 | 0.196 | 1.25 | 194 |
| `MES_CME_PRECLOSE_E2_RR1.0_CB1_ORB_G8` | ORB_G8 | 0.173 | 1.34 | 287 |

**Reconciliation:** per `.claude/rules/quant-audit-protocol.md` KNOWN FAILURE PATTERNS, `cost_risk_pct` has |corr|=1.0 with `1/orb_size_pts` — COST_LT08 and ORB_G8 are the SAME underlying size-family filter in different parameterizations. So the substantive narrative ("MES edge is size-based on CME_PRECLOSE long") is correct; the count was wrong.

**Action:** correct future references to say "1 size-family MES edge surfacing on both ORB_G8 and COST_LT08 filter parameterizations" rather than "1 validated MES lane."

---

## Finding 3 (FALSIFIES SYNTHESIS ANGLE 3) — MES COMEX_SETTLE ORB_G8 long RR1.5 is NOT a near-miss worth shadowing

Synthesis § Angle 3 flagged `H35 MES COMEX_SETTLE ORB_G8 long RR1.5` (K=40 cell: N=106, ExpR+0.196, t=+1.71) as a candidate for optional shadow pre-reg ("same session-family as the validated MES lane"). Per-year ExpR on the UNFILTERED parent lane (MES COMEX_SETTLE long RR1.5):

| 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −0.370 | −0.170 | −0.147 | −0.077 | −0.024 | +0.023 | −0.013 | −0.073 |

Even the UNFILTERED baseline has negative ExpR in 6 of 8 years. The K=40 filtered-on ExpR of +0.196 at N=106 averaged across these years is a mix dominated by favorable-regime days, NOT a stable within-year signal. Combined with Finding 1 (COMEX_SETTLE long is session-wide STABLE not improving), there's no regime tailwind to ride.

**Action:** synthesis § Angle 3 should be DOWNGRADED from "optional shadow candidate" to "MONITOR via regime-check only; do NOT pre-register." Will update synthesis doc.

---

## Finding 4 (HOUSEKEEPING) — MEMORY.md is ~1 week stale

MEMORY.md top-level portfolio state note says "MNQ=8 MES=1 validated" — current truth is MNQ=36 MES=2. Stale entry is from 2026-04-11. The correction cycle of 2026-04-19 should append a new entry pointing at the authoritative docs.

**Action:** append compact entry to MEMORY.md (index only, detail in topic file if needed).

---

## Finding 5 (LITERATURE CROSS-CHECK) — what we did NOT cite that we should

Local institutional-literature files in `docs/institutional/literature/` include extracts not cited in the correction cycle:

- **Aronson Ch 6 (data-mining bias)** — NOT cited. The correction cycle's data-snooping catch (synthesis Angle 1 REFUTED by formal test) is a textbook Aronson case. Adding the citation strengthens the audit trail.
- **Chan Ch 1 (backtesting + look-ahead)** — cited in the overnight session but NOT in the correction cycle. Relevant to the IS-only quantile sensitivity (Correction 6).
- **Chan Ch 7 (intraday momentum)** — cited in the overnight session. Relevant to Finding 1 above (NYSE_OPEN short strong improvement is consistent with Chan Ch 7 intraday momentum-cascade mechanism).
- **Carver Ch 9-10 (volatility targeting + Kelly + forecast combination)** — cited in institutional rigor rule but not in correction cycle. Relevant framing for future multi-signal composite pre-regs; flagged for Phase-D Carver-pilot design doc.

**Action:** this audit doc cites Aronson Ch 6 and Chan Ch 1/7 for the findings above.

---

## Finding 6 (NOT A GAP, CONFIRMATION) — MES unfiltered negativity interpretation holds

Cross-check of MES structural baseline against COMEX_SETTLE RR1.5 unfiltered per-year (Finding 3 table) confirms the MES low-ATR scan's substantive finding: MES unfiltered E2 is structurally unprofitable across time AND ATR regimes. The validated size-family edge on CME_PRECLOSE long RR1.0 remains the exception.

---

## Finding 7 (RISK) — regime-shift implications for the active book

Combining Finding 1 (session-direction regime shifts) with the 36 active MNQ lanes:

**Lanes operating against the current regime (long bias on DETERIORATING sessions):**
- 10 EUROPE_FLOW long lanes (all RRs, all filters) — riding a session-dead mechanism
- Unknown count of TOKYO_OPEN short / SINGAPORE_OPEN short (need query)

**Lanes operating WITH the current regime (directions of IMPROVING sessions):**
- NYSE_OPEN short lanes
- COMEX_SETTLE short lanes

**Action:** new pre-reg candidate — "MNQ regime-tailwind discovery pre-reg" testing filter overlays on NYSE_OPEN short, COMEX_SETTLE short, and SINGAPORE_OPEN long as the directions aligned with recent regime shift. Would be a Pathway A family scan, K=~15-25, Mode A strict.

---

## Summary — what this audit adds

| Finding | Type | Impact |
|---|---|---|
| 1 | MATERIAL | Strengthens Tier 1 retirement rationale; surfaces NYSE_OPEN short / COMEX_SETTLE short tailwind |
| 2 | CORRECTION | MES count fix (2 not 1); narrative unchanged |
| 3 | FALSIFIES synthesis | Angle 3 shadow candidate DOWNGRADED to monitor-only |
| 4 | HOUSEKEEPING | MEMORY.md append |
| 5 | AUDIT TRAIL | Literature cross-check (Aronson, Chan) — now cited |
| 6 | CONFIRMATION | MES unfiltered-negative story holds |
| 7 | RISK | New pre-reg candidate (regime-tailwind scan) |

## Action items for next session

1. Synthesis § Angle 3 downgrade edit (trivial).
2. MEMORY.md append (trivial).
3. Consider DRAFT pre-reg for MNQ regime-tailwind discovery (Finding 7).
4. Formalize EUROPE_FLOW session-wide retirement rationale in the retirement-queue doc as an explanatory appendix (not a vote change).

## Reproduction

Session-wide query (Finding 1): inline Python in this session's audit log — querying `orb_outcomes JOIN daily_features` by session, pooling across all rr_targets, grouping by period × direction. No filter applied (unfiltered = session-wide baseline).

Read-only. No writes.

## Literature citations added

- Aronson, D. (2006) *Evidence-Based Technical Analysis*, Ch 6 "Data Mining Bias." Grounds the correction-cycle's Angle 1 REFUTATION mechanism.
- Chan, E. (2013) *Algorithmic Trading*, Ch 1 "Backtesting and look-ahead," Ch 7 "Intraday momentum." Chan Ch 7 grounds the NYSE_OPEN regime-shift finding.
- Bailey, D. et al (2013) "Pseudo-mathematics and financial charlatanism." MinBTL bound applied across correction-cycle pre-regs.
- Fitschen, K. (2013) *Path of Least Resistance*, Ch 3 "Commodity intraday trend-follow." Grounds every Pathway A family in the cycle.
